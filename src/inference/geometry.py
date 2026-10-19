from __future__ import annotations

import numpy as np

LANDMARK_INDICES = (
    33,
    133,
    362,
    263,
    1,
    4,
    61,
    291,
    13,
    14,
    70,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    300,
    78,
    308,
    81,
    178,
    311,
    402,
    152,
    234,
    454,
    10,
    168,
    6,
)
GEOMETRY_DIM = len(LANDMARK_INDICES) * 3


def geometric_features(points: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3 or len(points) <= max(LANDMARK_INDICES):
        raise ValueError("Missing required facial landmarks")
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite facial landmarks")
    height, width = image_shape[:2]
    pixels = points.astype(np.float64) * [width, height, width]
    left = pixels[[33, 133]].mean(axis=0)
    right = pixels[[362, 263]].mean(axis=0)
    center = (left + right) / 2
    scale = np.linalg.norm(right[:2] - left[:2])
    if scale < 1e-6:
        raise ValueError("Degenerate interocular distance")
    centered = (pixels[list(LANDMARK_INDICES)] - center) / scale
    angle = np.arctan2(right[1] - left[1], right[0] - left[0])
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    centered[:, :2] = centered[:, :2] @ rotation.T
    return centered.astype(np.float32).reshape(-1)


def feature_columns(mode: str, embedding_dim: int, blendshape_dim: int = 52) -> np.ndarray:
    image = np.arange(embedding_dim)
    landmarks = np.arange(embedding_dim, embedding_dim + GEOMETRY_DIM)
    blends = np.arange(embedding_dim + GEOMETRY_DIM, embedding_dim + GEOMETRY_DIM + blendshape_dim)
    modes = {
        "image_only": image,
        "landmark_only": landmarks,
        "geometry_blendshapes": np.concatenate([landmarks, blends]),
        "fused_landmarks": np.concatenate([image, landmarks]),
        "fused": np.concatenate([image, landmarks, blends]),
    }
    if mode not in modes:
        raise ValueError(f"Unknown feature mode: {mode}")
    return modes[mode]
