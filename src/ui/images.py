from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel


def show_rgb(label: QLabel, rgb: np.ndarray) -> None:
    array = np.ascontiguousarray(rgb)
    image = QImage(
        array.data, array.shape[1], array.shape[0], array.strides[0], QImage.Format.Format_RGB888
    )
    pixmap = QPixmap.fromImage(image.copy())
    label.setPixmap(
        pixmap.scaled(
            label.contentsRect().size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    )
