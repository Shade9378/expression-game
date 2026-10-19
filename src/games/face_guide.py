from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants import CLASSES
from games.feedback import GEOMETRY_KEYS, NeutralBaseline

UPPER_LIP = (
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    409,
    291,
    78,
    191,
    80,
    81,
    82,
    13,
    312,
    311,
    310,
    415,
    308,
)
LOWER_LIP = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
)
RIGHT_BROW = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
LEFT_BROW = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
UPPER_EYES = (246, 161, 160, 159, 158, 157, 173, 466, 388, 387, 386, 385, 384, 398)
LOWER_EYES = (7, 163, 144, 145, 153, 154, 155, 249, 390, 373, 374, 380, 381, 382)


@dataclass(frozen=True)
class FaceGuide:
    current: np.ndarray
    suggested: np.ndarray
    arrows: tuple[int, ...]


def build_face_guide(
    points: np.ndarray,
    shape: tuple[int, ...],
    expression: str,
    features: dict[str, float],
    baseline: NeutralBaseline,
    sensitivity: float = 1.0,
) -> FaceGuide | None:
    if (
        points.shape != (478, 3)
        or not np.isfinite(points).all()
        or not 0.5 <= sensitivity <= 2
        or set(features) != set(baseline.median)
        or not set(GEOMETRY_KEYS) <= set(features)
        or not all(np.isfinite(value) for value in features.values())
        or len(shape) < 2
        or min(shape[:2]) <= 0
    ):
        return None
    current = points[:, :2].astype(float) * [shape[1], shape[0]]
    left = current[[33, 133]].mean(axis=0)
    right = current[[362, 263]].mean(axis=0)
    distance = float(np.linalg.norm(right - left))
    if distance < 8:
        return None
    if expression not in CLASSES:
        return None
    horizontal = (right - left) / distance
    vertical = np.array([-horizontal[1], horizontal[0]])
    origin = (left + right) / 2
    local = (
        np.column_stack(((current - origin) @ horizontal, (current - origin) @ vertical)) / distance
    )
    target = local.copy()
    strength = 1 / sensitivity
    neutral = baseline.median
    mouth_open = float(np.clip(neutral["mouth_open"], 0.01, 0.25))
    mouth_width = float(np.clip(neutral["mouth_width"], 0.35, 1.2))
    corner_raise = float(np.clip(neutral["corner_raise"], -0.08, 0.08))
    eye_open = float(np.clip(neutral["eye_open"], 0.035, 0.2))
    brow_raise = float(np.clip(neutral["brow_raise"], 0.08, 0.4))
    brow_lift = 0.0
    inner_lift = 0.0
    nose_lift = 0.0
    upper_lip_lift = 0.0
    if expression == "happy":
        corner_raise += 0.14 * strength
        mouth_width *= 1.12
        mouth_open = max(mouth_open, 0.08 * strength)
        eye_open *= 0.8
    elif expression == "surprise":
        mouth_open = max(mouth_open, 0.32 * strength)
        mouth_width *= 0.78
        eye_open += 0.06 * strength
        brow_lift = 0.14 * strength
    elif expression == "fear":
        mouth_open = max(mouth_open, 0.13 * strength)
        mouth_width *= 1.08
        eye_open += 0.08 * strength
        inner_lift = 0.14 * strength
    elif expression == "sad":
        corner_raise -= 0.13 * strength
        inner_lift = 0.14 * strength
    elif expression == "angry":
        mouth_open = 0.008
        eye_open *= 0.8
        inner_lift = -0.12 * strength
    elif expression == "disgust":
        nose_lift = 0.055 * strength
        upper_lip_lift = 0.075 * strength
        eye_open *= 0.8
    mouth_open = float(np.clip(mouth_open, 0.008, 0.45))
    mouth_width = float(np.clip(mouth_width, 0.3, 1.3))
    corner_raise = float(np.clip(corner_raise, -0.2, 0.25))
    eye_open = float(np.clip(eye_open, 0.025, 0.28))
    mouth_center = local[[13, 14]].mean(axis=0)
    for indices, sign in ((UPPER_LIP, -1), (LOWER_LIP, 1)):
        for segment, inner in ((indices[:11], False), (indices[11:], True)):
            u = np.linspace(-1, 1, len(segment))
            arc = np.sqrt(np.maximum(0, 1 - u * u))
            half_width = mouth_width / 2 * (0.85 if inner else 1)
            thickness = 0.0 if inner else 0.025
            target[list(segment), 0] = mouth_center[0] + u * half_width
            target[list(segment), 1] = (
                mouth_center[1] - corner_raise * u * u + sign * (mouth_open / 2 + thickness) * arc
            )
            if sign < 0:
                target[list(segment), 1] -= upper_lip_lift * arc
    for corners, upper, lower, brows, inward in (
        ((33, 133), UPPER_EYES[:7], LOWER_EYES[:7], RIGHT_BROW, 1),
        ((362, 263), UPPER_EYES[7:], LOWER_EYES[7:], LEFT_BROW, -1),
    ):
        center = local[list(corners)].mean(axis=0)
        half_width = max(0.04, abs(local[corners[1], 0] - local[corners[0], 0]) / 2)
        for indices, sign in ((upper, -1), (lower, 1)):
            u = np.clip((local[list(indices), 0] - center[0]) / half_width, -1, 1)
            target[list(indices), 1] = center[1] + sign * eye_open / 2 * np.sqrt(1 - u * u)
        u = np.clip((local[list(brows), 0] - center[0]) / half_width, -1.3, 1.3)
        inner_weight = np.clip((1 + inward * u) / 2, 0, 1)
        target[list(brows), 1] = (
            center[1]
            - brow_raise
            - brow_lift
            - inner_lift * inner_weight
            - 0.025 * np.maximum(0, 1 - u * u)
        )
        target[list(brows[5:]), 1] -= 0.025
    nose_indices = (1, 2, 4, 19, 94, 97, 98, 327, 326)
    target[list(nose_indices), 1] -= nose_lift
    suggested = origin + distance * (target[:, :1] * horizontal + target[:, 1:] * vertical)
    suggested = np.clip(suggested, [0, 0], [shape[1] - 1, shape[0] - 1])
    anchors = (61, 291, 13, 14, 105, 334, 107, 336, 159, 145, 386, 374, 98, 327)
    arrows = tuple(
        index for index in anchors if np.linalg.norm(suggested[index] - current[index]) >= 2
    )
    return FaceGuide(current, suggested, arrows)
