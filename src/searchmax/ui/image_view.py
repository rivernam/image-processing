"""Zoomable image view with ROI selection and detection overlays."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)

from searchmax.models import MatchResult, Rect


class ImageView(QGraphicsView):
    """Display a BGR image and image-coordinate ROI/result overlays."""

    roi_changed = Signal(Rect)

    ITEM_KIND_ROLE = 0
    ITEM_INDEX_ROLE = 1
    MIN_ZOOM = 0.1
    MAX_ZOOM = 20.0
    CENTER_RADIUS = 3.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#202124"))

        self._image_item: QGraphicsPixmapItem | None = None
        self._image_size = (0, 0)
        self._roi_enabled = False
        self._roi_item: QGraphicsRectItem | None = None
        self._roi_drag_start: QPointF | None = None
        self._match_items: list[
            tuple[QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem]
        ] = []
        self._truth_items: tuple[QGraphicsRectItem, QGraphicsEllipseItem] | None = None
        self._zoom_factor = 1.0
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    @property
    def image_item(self) -> QGraphicsPixmapItem | None:
        """The current pixmap item, exposed for read-only UI integration."""
        return self._image_item

    @property
    def zoom_factor(self) -> float:
        """Current zoom relative to the most recent fit operation."""
        return self._zoom_factor

    def set_image(self, image: np.ndarray) -> None:
        """Replace the current image, interpreting three channels as BGR."""
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be an HxWx3 uint8 BGR array")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("image must not be empty")

        rgb = np.ascontiguousarray(image[:, :, ::-1])
        height, width = rgb.shape[:2]
        qimage = QImage(
            rgb.data,
            width,
            height,
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        ).copy()

        self.scene().clear()
        self._roi_item = None
        self._truth_items = None
        self._match_items.clear()
        self._roi_drag_start = None

        self._image_item = self.scene().addPixmap(QPixmap.fromImage(qimage))
        self._image_item.setData(self.ITEM_KIND_ROLE, "image")
        self._image_item.setZValue(0)
        self._image_size = (width, height)
        self.scene().setSceneRect(QRectF(0, 0, width, height))
        self.fit_image()

    def fit_image(self) -> None:
        """Fit the full image in the viewport while preserving its aspect ratio."""
        if self._image_item is None:
            return
        self.resetTransform()
        self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = 1.0

    def image_to_scene(self, point: QPointF) -> QPointF:
        """Map an image pixel coordinate to the graphics scene."""
        if self._image_item is None:
            return QPointF(point)
        return self._image_item.mapToScene(point)

    def scene_to_image(self, point: QPointF) -> QPointF:
        """Map a graphics scene coordinate to the image coordinate system."""
        if self._image_item is None:
            return QPointF(point)
        return self._image_item.mapFromScene(point)

    def set_roi_enabled(self, enabled: bool) -> None:
        self._roi_enabled = bool(enabled)
        if not self._roi_enabled and self._roi_drag_start is not None:
            self._roi_drag_start = None
            if self._roi_item is not None:
                self.scene().removeItem(self._roi_item)
                self._roi_item = None
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if self._roi_enabled
            else QGraphicsView.DragMode.ScrollHandDrag
        )

    def selected_roi(self) -> Rect | None:
        """Return the normalized ROI, clamped to integral image bounds."""
        if self._roi_item is None or self._image_item is None:
            return None

        width, height = self._image_size
        roi = self._roi_item.rect().normalized().intersected(QRectF(0, 0, width, height))
        if roi.isEmpty():
            return None

        left = max(0, min(width, round(roi.left())))
        top = max(0, min(height, round(roi.top())))
        right = max(0, min(width, round(roi.right())))
        bottom = max(0, min(height, round(roi.bottom())))
        if right <= left or bottom <= top:
            return None
        return Rect(left, top, right - left, bottom - top)

    def set_matches(self, matches: Sequence[MatchResult]) -> None:
        """Replace all match boxes and their rank/score labels."""
        validated: list[tuple[MatchResult, str]] = []
        for index, match in enumerate(matches):
            if not isinstance(match, MatchResult):
                raise TypeError("matches must contain MatchResult values")
            self._validate_rect(match.box)
            if isinstance(match.score, bool) or not isinstance(match.score, Real):
                raise TypeError("match score must be a real number")
            if not math.isfinite(float(match.score)):
                raise ValueError("match score must be finite")
            validated.append((match, f"#{index + 1} {match.score:.3f}"))

        self._remove_match_items()
        for index, (match, label_text) in enumerate(validated):
            box, center = self._add_box(
                match.box, QColor("#ff5252"), "match", index, 2.0
            )
            label = self.scene().addText(label_text)
            label.setDefaultTextColor(QColor("#ffeb3b"))
            label.setPos(match.box.x, match.box.y)
            label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            label.setData(self.ITEM_KIND_ROLE, "match-label")
            label.setData(self.ITEM_INDEX_ROLE, index)
            label.setZValue(3)
            self._match_items.append((box, label, center))

    def highlight_match(self, index: int | None) -> None:
        """Highlight one zero-based match index, or clear the highlight."""
        for item_index, (box, _, _) in enumerate(self._match_items):
            pen = box.pen()
            pen.setWidthF(4.0 if item_index == index else 2.0)
            box.setPen(pen)

    def set_truth(self, truth: Rect | None) -> None:
        """Replace the ground-truth overlay, or clear it with ``None``."""
        if truth is not None:
            self._validate_rect(truth)

        if self._truth_items is not None:
            for item in self._truth_items:
                self.scene().removeItem(item)
            self._truth_items = None
        if truth is not None:
            self._truth_items = self._add_box(
                truth, QColor("#00e676"), "truth", None, 2.0
            )

    def set_truth_box(self, truth: Rect | None) -> None:
        """Compatibility spelling for :meth:`set_truth`."""
        self.set_truth(truth)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._image_item is None or event.angleDelta().y() == 0:
            super().wheelEvent(event)
            return
        step = 1.25 if event.angleDelta().y() > 0 else 0.8
        target = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom_factor * step))
        ratio = target / self._zoom_factor
        if not math.isclose(ratio, 1.0):
            self.scale(ratio, ratio)
            self._zoom_factor = target
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._roi_enabled and event.button() == Qt.MouseButton.LeftButton:
            self._roi_drag_start = self._event_image_position(event)
            if self._roi_item is None:
                self._roi_item = QGraphicsRectItem()
                self._roi_item.setData(self.ITEM_KIND_ROLE, "roi")
                self._roi_item.setPen(self._overlay_pen(QColor("#40c4ff"), 2.0))
                self._roi_item.setBrush(QColor(64, 196, 255, 35))
                self._roi_item.setZValue(4)
                self.scene().addItem(self._roi_item)
            self._roi_item.setRect(QRectF(self._roi_drag_start, self._roi_drag_start))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._roi_drag_start is not None and self._roi_item is not None:
            current = self._event_image_position(event)
            self._roi_item.setRect(QRectF(self._roi_drag_start, current).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._roi_drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            if self._roi_item is not None:
                current = self._event_image_position(event)
                self._roi_item.setRect(QRectF(self._roi_drag_start, current).normalized())
            self._roi_drag_start = None
            roi = self.selected_roi()
            if roi is not None:
                self.roi_changed.emit(roi)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _event_image_position(self, event: QMouseEvent) -> QPointF:
        scene_position = self.mapToScene(event.position().toPoint())
        return self.scene_to_image(scene_position)

    def _add_box(
        self,
        rect: Rect,
        color: QColor,
        kind: str,
        index: int | None,
        width: float,
    ) -> tuple[QGraphicsRectItem, QGraphicsEllipseItem]:
        box = self.scene().addRect(
            QRectF(rect.x, rect.y, rect.width, rect.height),
            self._overlay_pen(color, width),
        )
        box.setData(self.ITEM_KIND_ROLE, kind)
        if index is not None:
            box.setData(self.ITEM_INDEX_ROLE, index)
        box.setZValue(2)

        center_x = rect.x + rect.width / 2.0
        center_y = rect.y + rect.height / 2.0
        radius = self.CENTER_RADIUS
        center = self.scene().addEllipse(
            QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2),
            self._overlay_pen(color, 2.0),
        )
        center.setData(self.ITEM_KIND_ROLE, f"{kind}-center")
        if index is not None:
            center.setData(self.ITEM_INDEX_ROLE, index)
        center.setZValue(3)
        return box, center

    @staticmethod
    def _validate_rect(rect: Rect) -> None:
        if not isinstance(rect, Rect):
            raise TypeError("overlay box must be a Rect")
        values = (rect.x, rect.y, rect.width, rect.height)
        if any(type(value) is not int for value in values):
            raise TypeError("Rect coordinates and dimensions must be integers")
        if rect.width <= 0 or rect.height <= 0:
            raise ValueError("Rect width and height must be positive")

    @staticmethod
    def _overlay_pen(color: QColor, width: float) -> QPen:
        pen = QPen(color)
        pen.setWidthF(width)
        pen.setCosmetic(True)
        return pen

    def _remove_match_items(self) -> None:
        for box, label, center in self._match_items:
            self.scene().removeItem(box)
            self.scene().removeItem(label)
            self.scene().removeItem(center)
        self._match_items.clear()
