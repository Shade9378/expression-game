from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from constants import CLASSES
from games.feedback import movement_features
from inference.backends import InferenceBackend, load_backend
from inference.landmarks import FaceLandmarker, align_face, bounding_box
from inference.temporal import EMASmoother
from ui.colors import EXPRESSION_RGB, UNCLASSIFIED_RGB


class LatestQueue:
    def __init__(self) -> None:
        self.queue: queue.Queue = queue.Queue(maxsize=1)
        self.dropped = 0

    def put(self, item: object) -> None:
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self.queue.put_nowait(item)

    def get(self, timeout: float = 0.1) -> object:
        return self.queue.get(timeout=timeout)


def intersection_over_union(first: tuple, second: tuple) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return overlap / max(1, area_a + area_b - overlap)


class FaceTracker:
    def __init__(self) -> None:
        self.tracks: dict[int, tuple[tuple, float]] = {}
        self.next_id = 1

    def update(self, boxes: list[tuple], timestamp: float) -> list[int]:
        self.tracks = {
            key: value for key, value in self.tracks.items() if timestamp - value[1] < 0.75
        }
        available = set(self.tracks)
        ids = []
        for box in boxes:
            best = max(
                available,
                key=lambda key: intersection_over_union(box, self.tracks[key][0]),
                default=None,
            )
            if best is not None and intersection_over_union(box, self.tracks[best][0]) >= 0.3:
                track_id = best
                available.remove(best)
            else:
                track_id = self.next_id
                self.next_id += 1
            self.tracks[track_id] = (box, timestamp)
            ids.append(track_id)
        return ids


@dataclass
class ProcessedFrame:
    rgb: np.ndarray
    captured_at: float
    ready_at: float
    faces: list[dict]
    inference_ms: float
    status: str
    stages: dict[str, float] = field(default_factory=dict)


class FrameProcessor:
    def __init__(
        self, backend: InferenceBackend, asset: Path, overlays: bool = False, alpha: float = 0.3
    ) -> None:
        self.backend = backend
        expected_asset = getattr(backend, "metadata", {}).get("landmarker_sha256")
        if expected_asset and hashlib.sha256(asset.read_bytes()).hexdigest() != expected_asset:
            raise ValueError("Landmarker asset differs from the trained feature schema")
        self.landmarker = FaceLandmarker(asset, video=True)
        self.tracker = FaceTracker()
        self.overlays = overlays
        self.draw_detections = True
        self.smoother = EMASmoother(alpha)
        self.game_feedback = False

    def process(self, bgr: np.ndarray, captured_at: float) -> ProcessedFrame:
        started = time.perf_counter()
        stages = {"input_queue_ms": (started - captured_at) * 1000}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        converted = time.perf_counter()
        stages["color_conversion_ms"] = (converted - started) * 1000
        detected = self.landmarker.detect(rgb, int(captured_at * 1000))
        detected_at = time.perf_counter()
        stages["face_landmarks_ms"] = (detected_at - converted) * 1000
        boxes = [bounding_box(face, rgb.shape) for face in detected]
        ids = self.tracker.update(boxes, captured_at)
        self.smoother.retain(set(self.tracker.tracks))
        stages["tracking_ms"] = (time.perf_counter() - detected_at) * 1000
        for key in ("alignment_ms", "preprocessing_ms", "forward_ms", "probabilities_ms", "ema_ms"):
            stages[key] = 0.0
        faces = []
        inference_ms = 0.0
        stages["movement_features_ms"] = 0.0
        for face, box, track_id in zip(detected, boxes, ids, strict=True):
            alignment_started = time.perf_counter()
            crop, status = align_face(rgb, face)
            stages["alignment_ms"] += (time.perf_counter() - alignment_started) * 1000
            probabilities = None
            if crop is not None:
                start = time.perf_counter()
                if hasattr(self.backend, "predict_face"):
                    probabilities = self.backend.predict_face(crop, face, rgb.shape)
                else:
                    probabilities = self.backend.predict_rgb(crop)
                inference_ms += (time.perf_counter() - start) * 1000
                for key, value in getattr(self.backend, "last_timings", {}).items():
                    stages[key] += value
                if not np.isfinite(probabilities).all():
                    raise ValueError("Nonfinite expression probabilities")
            smoothing_started = time.perf_counter()
            smoothed = (
                self.smoother.update(track_id, probabilities) if probabilities is not None else None
            )
            if probabilities is None:
                self.smoother.reset(track_id)
            stages["ema_ms"] += (time.perf_counter() - smoothing_started) * 1000
            faces.append(
                {
                    "track_id": track_id,
                    "box": box,
                    "raw": probabilities,
                    "probabilities": smoothed,
                    "status": status,
                }
            )
            if self.game_feedback and crop is not None:
                faces[-1]["landmarks"] = face.points.copy()
                movement_started = time.perf_counter()
                try:
                    faces[-1]["movements"] = movement_features(
                        face.points,
                        dict(zip(face.blendshape_names, face.blendshapes, strict=True)),
                        rgb.shape,
                    )
                except (ValueError, IndexError):
                    faces[-1]["movements"] = None
                stages["movement_features_ms"] += (time.perf_counter() - movement_started) * 1000
        annotation_started = time.perf_counter()
        annotated = rgb.copy()
        for face, result in zip(detected, faces, strict=True):
            if not self.draw_detections:
                continue
            box = result["box"]
            probabilities = result["probabilities"]
            label = result["status"]
            color = UNCLASSIFIED_RGB
            if probabilities is not None:
                winner = int(np.argmax(probabilities))
                if probabilities[winner] >= 0.4:
                    color = EXPRESSION_RGB[winner]
                label = (
                    f"{CLASSES[winner]} {probabilities[winner]:.0%}"
                    if probabilities[winner] >= 0.4
                    else "Uncertain expression"
                )
            cv2.rectangle(annotated, box[:2], box[2:], color, 2)
            cv2.putText(
                annotated,
                f"Face {result['track_id']}: {label}",
                (box[0], max(20, box[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
            if self.overlays:
                for x, y, _ in face.points[::3]:
                    cv2.circle(
                        annotated,
                        (int(x * rgb.shape[1]), int(y * rgb.shape[0])),
                        1,
                        (255, 180, 60),
                        -1,
                    )
        stages["annotation_ms"] = (time.perf_counter() - annotation_started) * 1000
        stages["processing_ms"] = (time.perf_counter() - started) * 1000
        return ProcessedFrame(
            annotated,
            captured_at,
            time.perf_counter(),
            faces,
            inference_ms,
            "No face detected" if not faces else "Local processing",
            stages,
        )

    def close(self) -> None:
        self.landmarker.close()


class WebcamSession:
    def __init__(
        self, model: Path, asset: Path, camera: int = 0, overlays: bool = False, alpha: float = 0.3
    ) -> None:
        self.model, self.asset, self.camera, self.overlays = model, asset, camera, overlays
        self.alpha = alpha
        self.draw_detections = True
        self.game_feedback = False
        self.frames, self.results = LatestQueue(), LatestQueue()
        self.stop_event = threading.Event()
        self.error: str | None = None
        self.capture_thread = threading.Thread(target=self._capture, daemon=True)
        self.inference_thread = threading.Thread(target=self._infer, daemon=True)

    def start(self) -> None:
        self.capture_thread.start()
        self.inference_thread.start()

    def _capture(self) -> None:
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            if not cap.isOpened():
                raise RuntimeError(f"Camera {self.camera} unavailable or in use")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            failures = 0
            while not self.stop_event.is_set():
                read_started = time.perf_counter()
                ok, frame = cap.read()
                captured_at = time.perf_counter()
                if not ok:
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError("Camera stopped delivering frames")
                    self.stop_event.wait(0.03)
                    continue
                failures = 0
                self.frames.put((frame, captured_at, (captured_at - read_started) * 1000))
        except Exception as error:
            self.error = self.error or str(error)
            self.stop_event.set()
        finally:
            if cap is not None:
                cap.release()

    def _infer(self) -> None:
        processor = None
        try:
            processor = FrameProcessor(
                load_backend(self.model), self.asset, self.overlays, self.alpha
            )
            while not self.stop_event.is_set():
                try:
                    frame, timestamp, capture_ms = self.frames.get()
                except queue.Empty:
                    continue
                processor.overlays = self.overlays
                processor.draw_detections = self.draw_detections
                processor.smoother.alpha = self.alpha
                processor.game_feedback = self.game_feedback
                result = processor.process(frame, timestamp)
                result.stages["capture_read_ms"] = capture_ms
                self.results.put(result)
        except Exception as error:
            self.error = self.error or str(error)
            self.stop_event.set()
        finally:
            if processor is not None:
                processor.close()

    def stop(self) -> None:
        self.stop_event.set()
        for thread in (self.capture_thread, self.inference_thread):
            if thread.ident is not None:
                thread.join(timeout=3)


def benchmark_webcam(
    model: Path,
    asset: Path,
    output: Path,
    seconds: float = 15,
    camera: int = 0,
    timeline: Path | None = None,
    alpha: float = 0.3,
) -> dict:
    session = WebcamSession(model, asset, camera, alpha=alpha)
    session.start()
    rows = []
    started = time.perf_counter()
    first = None
    trace_rows = []
    try:
        while time.perf_counter() - started < seconds + 30:
            if session.error:
                break
            try:
                frame = session.results.get()
            except queue.Empty:
                continue
            if first is None:
                first = frame.ready_at
            rows.append(
                {
                    "elapsed": frame.ready_at - first,
                    "faces": len(frame.faces),
                    "classified_faces": sum(f["probabilities"] is not None for f in frame.faces),
                    "inference_ms": frame.inference_ms,
                    "face_statuses": [f["status"] for f in frame.faces],
                    "capture_to_annotation_ms": (frame.ready_at - frame.captured_at) * 1000,
                }
            )
            if timeline is not None:
                for face in frame.faces:
                    if face["raw"] is not None:
                        trace_rows.append(
                            {
                                "timestamp": frame.captured_at,
                                "track_id": face["track_id"],
                                "raw": face["raw"].tolist(),
                                "smoothed": face["probabilities"].tolist(),
                            }
                        )
            if time.perf_counter() - first >= seconds:
                break
    finally:
        session.stop()
    elapsed = rows[-1]["elapsed"] if rows else 0
    if not rows and session.error is None:
        session.error = "No processed frames within the startup deadline"
    result = {
        "model": str(model),
        "camera": camera,
        "frames": len(rows),
        "elapsed_seconds": elapsed,
        "processed_fps": (len(rows) - 1) / elapsed if elapsed else None,
        "error": session.error,
        "input_frames_dropped": session.frames.dropped,
        "result_frames_dropped": session.results.dropped,
        "faces_detected": sum(row["faces"] for row in rows),
        "faces_classified": sum(row["classified_faces"] for row in rows),
        "frames_with_classification": sum(bool(row["classified_faces"]) for row in rows),
        "classification_frame_throughput": (
            sum(bool(row["classified_faces"]) for row in rows) / elapsed if elapsed else None
        ),
        "mean_capture_to_annotation_ms": (
            float(np.mean([r["capture_to_annotation_ms"] for r in rows])) if rows else None
        ),
        "mean_inference_ms_when_classified": (
            float(np.mean([r["inference_ms"] for r in rows if r["classified_faces"]]))
            if any(r["classified_faces"] for r in rows)
            else None
        ),
        "scope": "Capture, detection, landmarks, alignment, normalization, inference, annotation",
        "gui_paint_included": False,
        "footage_saved": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "webcam.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "measurements.json").write_text(json.dumps(rows, indent=2) + "\n")
    if timeline is not None:
        timeline.parent.mkdir(parents=True, exist_ok=True)
        timeline.write_text(
            json.dumps(
                {
                    "source": "live_webcam",
                    "labels_available": False,
                    "camera": camera,
                    "alpha": alpha,
                    "rows": trace_rows,
                },
                indent=2,
            )
        )
    print(json.dumps(result, indent=2))
    return result
