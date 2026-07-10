from pathlib import Path

import cv2
import numpy as np

from .image_io import read_image
from .models import Rect, TrainModel


MIN_SIDE, MIN_STDDEV = 8, 2.0


def train_from_roi(image: np.ndarray, source: Path, roi: Rect) -> TrainModel:
    h, w = image.shape[:2]
    if roi.width < MIN_SIDE or roi.height < MIN_SIDE:
        raise ValueError("ROI is too small")
    if (
        roi.x < 0
        or roi.y < 0
        or roi.x + roi.width > w
        or roi.y + roi.height > h
    ):
        raise ValueError("ROI outside image")
    color = image[
        roi.y : roi.y + roi.height,
        roi.x : roi.x + roi.width,
    ].copy()
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < MIN_STDDEV:
        raise ValueError("template contrast is too low")
    return TrainModel(color, gray, source, roi)


def train_from_file(path: Path) -> TrainModel:
    image = read_image(path)
    return train_from_roi(image, path, Rect(0, 0, image.shape[1], image.shape[0]))
