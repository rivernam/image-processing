import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsTextItem

from searchmax.models import MatchResult, Rect
from searchmax.ui.image_view import ImageView


def _make_view(qtbot, width=320, height=240):
    view = ImageView()
    qtbot.addWidget(view)
    view.resize(640, 480)
    view.show()
    view.set_image(np.zeros((height, width, 3), np.uint8))
    view.fit_image()
    return view


def _wheel(view, delta):
    position = QPointF(view.viewport().rect().center())
    event = QWheelEvent(
        position,
        view.mapToGlobal(position.toPoint()),
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    view.wheelEvent(event)


def test_image_view_round_trips_scene_coordinates(qtbot):
    view = _make_view(qtbot)
    point = QPointF(120, 80)

    restored = view.scene_to_image(view.image_to_scene(point))

    assert restored.x() == pytest.approx(120, abs=0.5)
    assert restored.y() == pytest.approx(80, abs=0.5)


def test_set_image_converts_bgr_to_rgb_and_replaces_the_old_image(qtbot):
    view = _make_view(qtbot, width=1, height=1)
    view.set_image(np.array([[[10, 20, 30]]], dtype=np.uint8))

    pixel = view.image_item.pixmap().toImage().pixelColor(0, 0)

    assert pixel.getRgb()[:3] == (30, 20, 10)
    assert sum(item is view.image_item for item in view.scene().items()) == 1


def test_roi_drag_clamps_to_image_bounds_and_emits_change(qtbot):
    view = _make_view(qtbot, width=100, height=80)
    view.set_roi_enabled(True)
    start = view.mapFromScene(view.image_to_scene(QPointF(10, 15)))
    end = view.mapFromScene(view.image_to_scene(QPointF(130, 110)))

    with qtbot.waitSignal(view.roi_changed, timeout=1000) as signal:
        qtbot.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        qtbot.mouseMove(view.viewport(), pos=end)
        qtbot.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)

    assert view.selected_roi() == Rect(10, 15, 90, 65)
    assert signal.args == [Rect(10, 15, 90, 65)]


def test_set_matches_adds_ranked_labels_and_replaces_old_overlays(qtbot):
    view = _make_view(qtbot)
    matches = [
        MatchResult(0.95, Rect(5, 6, 20, 30), 1.0),
        MatchResult(0.8754, Rect(40, 50, 25, 35), 0.9),
    ]

    view.set_matches(matches)
    view.highlight_match(1)

    labels = sorted(
        item.toPlainText()
        for item in view.scene().items()
        if isinstance(item, QGraphicsTextItem)
    )
    boxes = [
        item
        for item in view.scene().items()
        if isinstance(item, QGraphicsRectItem) and item.data(ImageView.ITEM_KIND_ROLE) == "match"
    ]
    assert labels == ["#1 0.950", "#2 0.875"]
    assert sorted(box.pen().widthF() for box in boxes) == [2.0, 4.0]

    view.set_matches(matches[:1])

    labels = [
        item.toPlainText()
        for item in view.scene().items()
        if isinstance(item, QGraphicsTextItem)
    ]
    assert labels == ["#1 0.950"]


def test_truth_overlay_can_be_replaced_and_cleared(qtbot):
    view = _make_view(qtbot)

    view.set_truth(Rect(1, 2, 30, 40))
    view.set_truth(Rect(5, 6, 10, 20))

    truth_boxes = [
        item
        for item in view.scene().items()
        if isinstance(item, QGraphicsRectItem) and item.data(ImageView.ITEM_KIND_ROLE) == "truth"
    ]
    assert len(truth_boxes) == 1
    assert truth_boxes[0].rect().getRect() == (5.0, 6.0, 10.0, 20.0)

    view.set_truth(None)
    assert not any(item.data(ImageView.ITEM_KIND_ROLE) == "truth" for item in view.scene().items())


def test_wheel_zoom_is_clamped(qtbot):
    view = _make_view(qtbot)

    for _ in range(100):
        _wheel(view, 120)
    assert view.zoom_factor == pytest.approx(20.0)

    for _ in range(200):
        _wheel(view, -120)
    assert view.zoom_factor == pytest.approx(0.1)


def test_wheel_direction_is_preserved_when_fit_scale_exceeds_zoom_limit(qtbot):
    view = _make_view(qtbot, width=1, height=1)
    fitted_scale = view.transform().m11()

    _wheel(view, 120)

    assert view.transform().m11() > fitted_scale
