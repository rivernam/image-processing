"""Compact persistent preview for the trained ROI template."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QResizeEvent
from PySide6.QtWidgets import QFrame, QLabel


class TemplatePreview(QLabel):
    clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._source_pixmap: QPixmap | None = None
        self.setFixedSize(240, 140)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.clear()

    @property
    def image(self) -> np.ndarray | None:
        return None if self._image is None else self._image.copy()

    def clear(self) -> None:
        self._image = None
        self._source_pixmap = None
        self.setPixmap(QPixmap())
        self.setText("No ROI template")
        self.unsetCursor()

    def set_image(self, image: np.ndarray) -> None:
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError("image must be a non-empty HxWx3 uint8 BGR array")
        self._image = image.copy()
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        height, width = rgb.shape[:2]
        qimage = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
        ).copy()
        self._source_pixmap = QPixmap.fromImage(qimage)
        self.setText("")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        width = min(self._source_pixmap.width(), self.contentsRect().width())
        height = min(self._source_pixmap.height(), self.contentsRect().height())
        pixmap = self._source_pixmap.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        if self._image is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
