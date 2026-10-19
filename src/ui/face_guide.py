from __future__ import annotations

import time
from dataclasses import replace
from functools import lru_cache

import cv2
import numpy as np

from games.face_guide import FaceGuide, build_face_guide
from games.feedback import NeutralBaseline
from inference.webcam import ProcessedFrame


def render_guide_frame(
    frame: ProcessedFrame,
    expression: str,
    baseline: NeutralBaseline | None,
    sensitivity: float = 1.0,
    max_frame_age: float = 0.45,
) -> ProcessedFrame:
    if (
        baseline is None
        or len(frame.faces) != 1
        or not 0 <= time.perf_counter() - frame.captured_at <= max_frame_age
    ):
        return frame
    face = frame.faces[0]
    points = face.get("landmarks")
    features = face.get("movements")
    if face["status"] != "OK" or points is None or features is None:
        return frame
    guide = build_face_guide(points, frame.rgb.shape, expression, features, baseline, sensitivity)
    return replace(frame, rgb=draw_face_guide(frame.rgb, guide)) if guide is not None else frame


@lru_cache(maxsize=1)
def guide_connections() -> tuple[tuple[int, int], ...]:
    from mediapipe.python.solutions import face_mesh_connections as connections

    return tuple(sorted(connections.FACEMESH_CONTOURS | connections.FACEMESH_NOSE))


def draw_face_guide(rgb: np.ndarray, guide: FaceGuide) -> np.ndarray:
    output = rgb.copy()
    current = np.rint(guide.current).astype(int)
    suggested = np.rint(guide.suggested).astype(int)
    for first, second in guide_connections():
        cv2.line(
            output,
            tuple(suggested[first]),
            tuple(suggested[second]),
            (70, 225, 235),
            2,
            cv2.LINE_AA,
        )
    for index in guide.arrows:
        if np.linalg.norm(guide.suggested[index] - guide.current[index]) < 2:
            continue
        cv2.arrowedLine(
            output,
            tuple(current[index]),
            tuple(suggested[index]),
            (255, 200, 65),
            2,
            cv2.LINE_AA,
            tipLength=0.45,
        )
    return output
