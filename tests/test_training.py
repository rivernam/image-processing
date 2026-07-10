from pathlib import Path

import numpy as np
import pytest

from searchmax.image_io import read_image, write_image
from searchmax.models import Rect
from searchmax.training import train_from_roi


def test_train_from_roi_copies_expected_pixels():
    image = np.zeros((80, 100, 3), np.uint8)
    image[20:50, 10:50] = np.arange(40, dtype=np.uint8)[None, :, None]
    model = train_from_roi(image, Path("한글화면.png"), Rect(10, 20, 40, 30))
    assert model.color.shape == (30, 40, 3)
    assert model.gray.shape == (30, 40)


def test_train_rejects_low_variance_template():
    with pytest.raises(ValueError, match="contrast"):
        train_from_roi(
            np.full((50, 50, 3), 127, np.uint8),
            Path("x.png"),
            Rect(0, 0, 50, 50),
        )


def test_unicode_path_image_round_trip(tmp_path):
    path = tmp_path / "한글 이미지.png"
    image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)

    write_image(path, image)

    assert np.array_equal(read_image(path), image)
