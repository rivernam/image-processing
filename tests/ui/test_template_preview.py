import numpy as np
from PySide6.QtCore import Qt

from searchmax.ui.template_preview import TemplatePreview


def test_template_preview_starts_empty(qtbot):
    preview = TemplatePreview()
    qtbot.addWidget(preview)

    assert preview.image is None
    assert preview.text() == "No ROI template"


def test_template_preview_stores_image_without_upscaling(qtbot):
    preview = TemplatePreview()
    qtbot.addWidget(preview)
    image = np.zeros((20, 30, 3), np.uint8)
    image[:, :15] = (20, 120, 240)

    preview.set_image(image)

    assert np.array_equal(preview.image, image)
    assert preview.pixmap().width() <= 30
    assert preview.pixmap().height() <= 20


def test_template_preview_emits_click_only_when_populated(qtbot):
    preview = TemplatePreview()
    qtbot.addWidget(preview)
    clicks = []
    preview.clicked.connect(lambda: clicks.append(True))

    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)
    preview.set_image(np.zeros((20, 30, 3), np.uint8))
    qtbot.mouseClick(preview, Qt.MouseButton.LeftButton)

    assert clicks == [True]
