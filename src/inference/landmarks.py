from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Face:
    points: np.ndarray
    blendshapes: np.ndarray
    blendshape_names: tuple[str, ...]


class FaceLandmarker:
    def __init__(self, asset: Path, video: bool = False, max_faces: int = 4) -> None:
        import mediapipe as mp

        self.mp = mp
        self.video = video
        self.last_timestamp = -1
        if not asset.is_file():
            raise FileNotFoundError(f"Local MediaPipe model missing: {asset}")
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=asset.read_bytes()),
            running_mode=(
                mp.tasks.vision.RunningMode.VIDEO if video else mp.tasks.vision.RunningMode.IMAGE
            ),
            num_faces=max_faces,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray, timestamp_ms: int = 0) -> list[Face]:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.video:
            self.last_timestamp = max(timestamp_ms, self.last_timestamp + 1)
            result = self.detector.detect_for_video(image, self.last_timestamp)
        else:
            result = self.detector.detect(image)
        faces = []
        for index, points in enumerate(result.face_landmarks):
            categories = result.face_blendshapes[index] if result.face_blendshapes else []
            categories = sorted(categories, key=lambda c: c.category_name)
            faces.append(
                Face(
                    np.asarray([[p.x, p.y, p.z] for p in points], dtype=np.float32),
                    np.asarray([c.score for c in categories], dtype=np.float32),
                    tuple(c.category_name for c in categories),
                )
            )
        return faces

    def close(self) -> None:
        self.detector.close()


def bounding_box(face: Face, shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    xy = face.points[:, :2] * [width, height]
    low, high = xy.min(axis=0), xy.max(axis=0)
    padding = (high - low) * 0.12
    x1, y1 = np.maximum(low - padding, 0).astype(int)
    x2, y2 = np.minimum(high + padding, [width, height]).astype(int)
    return int(x1), int(y1), int(x2), int(y2)


def align_face(rgb: np.ndarray, face: Face) -> tuple[np.ndarray | None, str]:
    height, width = rgb.shape[:2]
    xy = face.points[:, :2] * [width, height]
    left = xy[[33, 133]].mean(axis=0)
    right = xy[[362, 263]].mean(axis=0)
    distance = float(np.linalg.norm(right - left))
    if distance < 8:
        return None, "Face too small"
    if abs(float((xy[1, 0] - (left[0] + right[0]) / 2) / distance)) > 0.65:
        return None, "Turn toward camera"
    angle = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))
    matrix = cv2.getRotationMatrix2D(tuple((left + right) / 2), angle, 1)
    aligned = cv2.warpAffine(rgb, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    transformed = np.column_stack([xy, np.ones(len(xy))]) @ matrix.T
    low, high = transformed.min(axis=0), transformed.max(axis=0)
    padding = (high - low) * 0.12
    x1, y1 = np.maximum(low - padding, 0).astype(int)
    x2, y2 = np.minimum(high + padding, [width, height]).astype(int)
    crop = aligned[y1:y2, x1:x2]
    if not crop.size:
        return None, "Invalid face crop"
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    if gray.mean() < 30 or gray.std() < 10:
        return None, "Insufficient lighting/contrast"
    return crop, "OK"
