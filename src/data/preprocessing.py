from __future__ import annotations

import cv2
import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocess_rgb(image: np.ndarray, size: int = 112) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HWC uint8 RGB image")
    if size < 32:
        raise ValueError("Input size must be at least 32")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = np.repeat(resized[..., None], 3, axis=2).astype(np.float32) / 255.0
    normalized = (rgb - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))
