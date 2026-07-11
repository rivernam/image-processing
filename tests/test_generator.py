from pathlib import Path

import cv2
import numpy as np
import pytest

from searchmax.generator import GenerationSettings, generate_samples
from searchmax.image_io import read_image
from searchmax.models import Rect
from searchmax.training import train_from_roi


def test_generation_is_reproducible(tmp_path):
    template = np.zeros((20, 30, 3), np.uint8)
    template[:, 5:25] = (20, 140, 250)
    model = train_from_roi(template, Path("t.png"), Rect(0, 0, 30, 20))
    settings = GenerationSettings(count=2, seed=42, output_size=(320, 200))

    first = generate_samples(model, [], tmp_path / "a", settings)
    second = generate_samples(model, [], tmp_path / "b", settings)

    assert [(sample.truth_box, sample.transform) for sample in first] == [
        (sample.truth_box, sample.transform) for sample in second
    ]
    assert all(sample.image_path.exists() for sample in first)
    assert all(
        np.array_equal(read_image(a.image_path), read_image(b.image_path))
        for a, b in zip(first, second, strict=True)
    )


def test_settings_reject_invalid_output_size():
    with pytest.raises(ValueError, match="output_size"):
        GenerationSettings(output_size=(0, 200))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"count": 0}, "count"),
        ({"count": True}, "count"),
        ({"seed": 1.5}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": -1}, "seed"),
        ({"min_scale": 0}, "scale"),
        ({"min_scale": 1.1, "max_scale": 1.0}, "scale"),
        ({"max_scale": float("inf")}, "scale"),
        ({"brightness_range": (2, -2)}, "brightness_range"),
        ({"brightness_range": (0, float("nan"))}, "brightness_range"),
        ({"contrast_range": (0, 1)}, "contrast_range"),
        ({"contrast_range": (1.1, 0.9)}, "contrast_range"),
        ({"blur_choices": ()}, "blur_choices"),
        ({"blur_choices": (0, 2)}, "blur_choices"),
        ({"noise_sigma_range": (-1, 2)}, "noise_sigma_range"),
        ({"noise_sigma_range": (2, 1)}, "noise_sigma_range"),
    ],
)
def test_settings_reject_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GenerationSettings(**kwargs)


def test_generated_boxes_and_scales_stay_in_bounds(tmp_path):
    template = np.zeros((20, 30, 3), np.uint8)
    template[:, 5:25] = (20, 140, 250)
    model = train_from_roi(template, Path("t.png"), Rect(0, 0, 30, 20))
    settings = GenerationSettings(count=10, seed=3, output_size=(35, 22))

    samples = generate_samples(model, [], tmp_path, settings)

    assert len(samples) == settings.count
    for index, sample in enumerate(samples, start=1):
        box = sample.truth_box
        assert sample.image_path.name == f"sample_{index:04d}.png"
        assert read_image(sample.image_path).shape == (22, 35, 3)
        assert 0 <= box.x <= box.x + box.width <= 35
        assert 0 <= box.y <= box.y + box.height <= 22
        assert settings.min_scale <= sample.transform.scale <= settings.max_scale
        assert box.width == max(1, round(30 * sample.transform.scale))
        assert box.height == max(1, round(20 * sample.transform.scale))


def test_fitting_scale_never_falls_below_minimum_at_rounding_boundary(tmp_path):
    template = np.zeros((8, 8, 3), np.uint8)
    template[:, 2:6] = (20, 140, 250)
    model = train_from_roi(template, Path("t.png"), Rect(0, 0, 8, 8))
    settings = GenerationSettings(
        count=1,
        seed=4,
        output_size=(10, 10),
        min_scale=1.3125,
        max_scale=2,
        brightness_range=(0, 0),
        contrast_range=(1, 1),
        blur_choices=(0,),
        noise_sigma_range=(0, 0),
    )

    [sample] = generate_samples(model, [], tmp_path, settings)

    assert sample.transform.scale >= settings.min_scale
    assert sample.truth_box.width <= 10
    assert sample.truth_box.height <= 10


def test_supplied_background_and_recorded_full_image_transform_are_exact(tmp_path):
    template = np.zeros((10, 12, 3), np.uint8)
    template[:, 2:10] = (30, 150, 240)
    model = train_from_roi(template, Path("template.png"), Rect(0, 0, 12, 10))
    background = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    settings = GenerationSettings(
        count=1,
        seed=8,
        output_size=(32, 24),
        min_scale=1.25,
        max_scale=1.25,
        brightness_range=(8, 8),
        contrast_range=(1.1, 1.1),
        blur_choices=(3,),
        noise_sigma_range=(0, 0),
    )

    [sample] = generate_samples(
        model, [background], tmp_path / "한글 결과", settings
    )

    assert sample.transform.scale == 1.25
    assert sample.transform.brightness == 8
    assert sample.transform.contrast == 1.1
    assert sample.transform.blur_kernel == 3
    assert sample.transform.noise_sigma == 0
    assert sample.seed == 8
    box = sample.truth_box
    expected = cv2.resize(background, settings.output_size)
    expected[box.y : box.y + box.height, box.x : box.x + box.width] = cv2.resize(
        template, (box.width, box.height)
    )
    expected = cv2.convertScaleAbs(expected, alpha=1.1, beta=8)
    expected = cv2.GaussianBlur(expected, (3, 3), 0)
    assert np.array_equal(read_image(sample.image_path), expected)


def test_seeded_noise_is_applied_outside_the_truth_box(tmp_path):
    template = np.zeros((10, 12, 3), np.uint8)
    template[:, 2:10] = (30, 150, 240)
    model = train_from_roi(template, Path("template.png"), Rect(0, 0, 12, 10))
    background = np.full((40, 50, 3), 100, np.uint8)
    common = dict(
        count=1,
        seed=11,
        output_size=(50, 40),
        min_scale=1,
        max_scale=1,
        brightness_range=(0, 0),
        contrast_range=(1, 1),
        blur_choices=(0,),
    )

    [clean_sample] = generate_samples(
        model,
        [background],
        tmp_path / "clean",
        GenerationSettings(**common, noise_sigma_range=(0, 0)),
    )
    [noisy_sample] = generate_samples(
        model,
        [background],
        tmp_path / "noisy",
        GenerationSettings(**common, noise_sigma_range=(5, 5)),
    )

    assert clean_sample.truth_box == noisy_sample.truth_box
    assert noisy_sample.transform.noise_sigma == 5
    clean = read_image(clean_sample.image_path)
    noisy = read_image(noisy_sample.image_path)
    outside = np.ones(clean.shape[:2], dtype=bool)
    box = clean_sample.truth_box
    outside[box.y : box.y + box.height, box.x : box.x + box.width] = False
    assert np.any(clean[outside] != noisy[outside])
