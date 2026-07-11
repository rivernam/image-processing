from dataclasses import dataclass
import math
from numbers import Integral, Real
from pathlib import Path

import cv2
import numpy as np

from .image_io import write_image
from .models import GeneratedSample, Rect, TrainModel, TransformRecord


def _validate_range(
    name: str, values: tuple[float, float], *, positive: bool = False
) -> None:
    if (
        not isinstance(values, tuple)
        or len(values) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in values
        )
    ):
        raise ValueError(f"{name} must contain two finite numbers")
    lower, upper = values
    if lower > upper or (positive and lower <= 0):
        qualifier = "positive and ordered" if positive else "ordered"
        raise ValueError(f"{name} must be {qualifier}")


@dataclass(frozen=True)
class GenerationSettings:
    count: int = 20
    seed: int = 1234
    output_size: tuple[int, int] = (1280, 720)
    min_scale: float = 0.8
    max_scale: float = 1.5
    brightness_range: tuple[float, float] = (-12, 12)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    blur_choices: tuple[int, ...] = (0, 3)
    noise_sigma_range: tuple[float, float] = (0, 3)

    def __post_init__(self) -> None:
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, Integral)
            or self.count <= 0
        ):
            raise ValueError("count must be a positive integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if (
            not isinstance(self.output_size, tuple)
            or len(self.output_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0
                for value in self.output_size
            )
        ):
            raise ValueError("output_size must contain two positive integers")
        if (
            isinstance(self.min_scale, bool)
            or isinstance(self.max_scale, bool)
            or not isinstance(self.min_scale, Real)
            or not isinstance(self.max_scale, Real)
            or not math.isfinite(float(self.min_scale))
            or not math.isfinite(float(self.max_scale))
            or self.min_scale <= 0
            or self.max_scale < self.min_scale
        ):
            raise ValueError("invalid scale range")
        _validate_range("brightness_range", self.brightness_range)
        _validate_range("contrast_range", self.contrast_range, positive=True)
        if (
            not isinstance(self.blur_choices, tuple)
            or not self.blur_choices
            or any(
                isinstance(kernel, bool)
                or not isinstance(kernel, Integral)
                or kernel < 0
                or (kernel != 0 and kernel % 2 == 0)
                for kernel in self.blur_choices
            )
        ):
            raise ValueError("blur_choices must contain 0 or positive odd integers")
        _validate_range("noise_sigma_range", self.noise_sigma_range)
        if self.noise_sigma_range[0] < 0:
            raise ValueError("noise_sigma_range must be non-negative")


def _fallback_background(
    rng: np.random.Generator, width: int, height: int
) -> np.ndarray:
    image = np.full((height, width, 3), 224, dtype=np.uint8)
    for _ in range(6):
        x1 = int(rng.integers(0, width))
        y1 = int(rng.integers(0, height))
        x2 = int(rng.integers(x1, width + 1))
        y2 = int(rng.integers(y1, height + 1))
        shade = int(rng.integers(180, 241))
        cv2.rectangle(image, (x1, y1), (x2, y2), (shade,) * 3, -1)
    return image


def _scaled_size(
    template_width: int, template_height: int, scale: float
) -> tuple[int, int]:
    return (
        max(1, round(template_width * scale)),
        max(1, round(template_height * scale)),
    )


def _validate_backgrounds(backgrounds: list[np.ndarray]) -> None:
    for index, background in enumerate(backgrounds):
        if not isinstance(background, np.ndarray):
            raise ValueError(f"background {index} must be a NumPy array")
        if background.dtype != np.uint8:
            raise ValueError(f"background {index} must have uint8 dtype")
        if (
            background.ndim != 3
            or background.shape[2] != 3
            or background.shape[0] == 0
            or background.shape[1] == 0
        ):
            raise ValueError(f"background {index} must be a non-empty BGR image")


def generate_samples(
    model: TrainModel,
    backgrounds: list[np.ndarray],
    output_dir: Path,
    settings: GenerationSettings,
) -> list[GeneratedSample]:
    rng = np.random.default_rng(settings.seed)
    output_width, output_height = settings.output_size
    template_height, template_width = model.color.shape[:2]
    min_width, min_height = _scaled_size(
        template_width, template_height, settings.min_scale
    )
    if min_width > output_width or min_height > output_height:
        raise ValueError("template does not fit output_size at min_scale")
    max_width, max_height = _scaled_size(
        template_width, template_height, settings.max_scale
    )
    if max_width > output_width or max_height > output_height:
        raise ValueError("template does not fit output_size at max_scale")
    _validate_backgrounds(backgrounds)
    samples = []

    for index in range(settings.count):
        if backgrounds:
            background = backgrounds[int(rng.integers(len(backgrounds)))]
            image = cv2.resize(background, settings.output_size)
        else:
            image = _fallback_background(rng, output_width, output_height)

        scale = float(rng.uniform(settings.min_scale, settings.max_scale))
        scaled_width, scaled_height = _scaled_size(
            template_width, template_height, scale
        )
        resized = cv2.resize(model.color, (scaled_width, scaled_height))
        x = int(rng.integers(0, output_width - scaled_width + 1))
        y = int(rng.integers(0, output_height - scaled_height + 1))
        image[y : y + scaled_height, x : x + scaled_width] = resized

        brightness = float(rng.uniform(*settings.brightness_range))
        contrast = float(rng.uniform(*settings.contrast_range))
        blur_kernel = int(rng.choice(settings.blur_choices))
        noise_sigma = float(rng.uniform(*settings.noise_sigma_range))
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        if blur_kernel:
            image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
        if noise_sigma:
            noise = rng.normal(0, noise_sigma, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        path = output_dir / f"sample_{index + 1:04d}.png"
        write_image(path, image)
        samples.append(
            GeneratedSample(
                path,
                Rect(x, y, scaled_width, scaled_height),
                TransformRecord(
                    scale, brightness, contrast, blur_kernel, noise_sigma
                ),
                settings.seed,
            )
        )

    return samples
