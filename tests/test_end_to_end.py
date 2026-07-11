from pathlib import Path

import cv2
import numpy as np

from searchmax.evaluator import evaluate_sample
from searchmax.generator import GenerationSettings, generate_samples
from searchmax.image_io import read_image
from searchmax.matcher import match
from searchmax.models import Rect, SearchSettings
from searchmax.training import train_from_roi


def test_generated_clean_samples_are_detected(tmp_path):
    pattern = np.zeros((28, 44, 3), np.uint8)
    cv2.rectangle(pattern, (2, 2), (41, 25), (40, 180, 240), 2)
    cv2.putText(pattern, "OK", (7, 20), 0, 0.5, (255, 255, 255), 1)
    model = train_from_roi(pattern, Path("ok.png"), Rect(0, 0, 44, 28))
    samples = generate_samples(
        model,
        [],
        tmp_path,
        GenerationSettings(
            count=5,
            seed=7,
            brightness_range=(0, 0),
            contrast_range=(1, 1),
            blur_choices=(0,),
            noise_sigma_range=(0, 0),
        ),
    )

    records = []
    for sample in samples:
        image = read_image(sample.image_path)
        matches = match(model, image, SearchSettings(threshold=0.75))
        records.append(evaluate_sample(sample, matches, (44, 28), 0.5))

    assert all(record.success for record in records)
