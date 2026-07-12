import csv
import json
import math
import os
import tempfile
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from .generator import GenerationSettings
from .models import (
    EvaluationRecord,
    GeneratedSample,
    Rect,
    SearchSettings,
    TransformRecord,
)


SCHEMA_VERSION = 1
CSV_COLUMNS = (
    "image",
    "success",
    "score",
    "iou",
    "center_error",
    "scale_error_percent",
    "elapsed_ms",
)


def _portable_path(path: Path, base_dir: Path) -> str:
    normalized_path = path.resolve()
    normalized_base = base_dir.resolve()
    try:
        return normalized_path.relative_to(normalized_base).as_posix()
    except ValueError:
        return str(normalized_path)


def _resolved_path(value: object, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    normalized_base = base_dir.resolve()
    resolved = (normalized_base / path).resolve()
    try:
        resolved.relative_to(normalized_base)
    except ValueError as error:
        raise ValueError(f"{field} must not escape the JSON directory") from error
    return resolved


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} must contain exactly {sorted(expected)}")


def _integer(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return result


def _number(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if minimum is not None and (
        result < minimum or (strict_minimum and result == minimum)
    ):
        comparison = "greater than" if strict_minimum else "at least"
        raise ValueError(f"{field} must be {comparison} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return result


def _array(value: object, field: str, *, length: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    if length is not None and len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} values")
    return value


def _rect_to_dict(rect: Rect) -> dict[str, int]:
    return {
        "x": rect.x,
        "y": rect.y,
        "width": rect.width,
        "height": rect.height,
    }


def _rect_from_dict(value: object, field: str) -> Rect:
    data = _object(value, field)
    _keys(data, {"x", "y", "width", "height"}, field)
    return Rect(
        x=_integer(data["x"], f"{field}.x"),
        y=_integer(data["y"], f"{field}.y"),
        width=_integer(data["width"], f"{field}.width", minimum=1),
        height=_integer(data["height"], f"{field}.height", minimum=1),
    )


def _search_settings_to_dict(settings: SearchSettings) -> dict[str, Any]:
    return {
        "min_scale": settings.min_scale,
        "max_scale": settings.max_scale,
        "scale_step": settings.scale_step,
        "threshold": settings.threshold,
        "max_results": settings.max_results,
        "nms_iou_threshold": settings.nms_iou_threshold,
        "color_mode": settings.color_mode,
    }


def _generation_settings_to_dict(
    settings: GenerationSettings,
) -> dict[str, Any]:
    return {
        "count": settings.count,
        "seed": settings.seed,
        "output_size": list(settings.output_size),
        "min_scale": settings.min_scale,
        "max_scale": settings.max_scale,
        "brightness_range": list(settings.brightness_range),
        "contrast_range": list(settings.contrast_range),
        "blur_choices": list(settings.blur_choices),
        "noise_sigma_range": list(settings.noise_sigma_range),
        "hue_shift_range": list(settings.hue_shift_range),
        "saturation_scale_range": list(settings.saturation_scale_range),
    }


def _search_settings_from_dict(value: object) -> SearchSettings:
    field = "search_settings"
    data = _object(value, field)
    _keys(
        data,
        {
            "min_scale",
            "max_scale",
            "scale_step",
            "threshold",
            "max_results",
            "nms_iou_threshold",
            "color_mode",
        },
        field,
    )
    min_scale = _number(
        data["min_scale"],
        f"{field}.min_scale",
        minimum=0,
        strict_minimum=True,
    )
    max_scale = _number(data["max_scale"], f"{field}.max_scale", minimum=min_scale)
    color_mode = data["color_mode"]
    if not isinstance(color_mode, str) or color_mode not in {"color", "gray"}:
        raise ValueError(f"{field}.color_mode must be 'color' or 'gray'")
    return SearchSettings(
        min_scale=min_scale,
        max_scale=max_scale,
        scale_step=_number(
            data["scale_step"],
            f"{field}.scale_step",
            minimum=0,
            strict_minimum=True,
        ),
        threshold=_number(
            data["threshold"], f"{field}.threshold", minimum=0, maximum=1
        ),
        max_results=_integer(
            data["max_results"],
            f"{field}.max_results",
            minimum=1,
            maximum=100,
        ),
        nms_iou_threshold=_number(
            data["nms_iou_threshold"],
            f"{field}.nms_iou_threshold",
            minimum=0,
            maximum=1,
        ),
        color_mode=color_mode,
    )


def _number_range(
    value: object, field: str, *, positive: bool = False
) -> tuple[float, float]:
    values = _array(value, field, length=2)
    lower = _number(
        values[0],
        f"{field}[0]",
        minimum=0 if positive else None,
        strict_minimum=positive,
    )
    upper = _number(values[1], f"{field}[1]")
    if upper < lower:
        raise ValueError(f"{field} must be ordered")
    return lower, upper


def _generation_settings_from_dict(value: object) -> GenerationSettings:
    field = "generation_settings"
    data = _object(value, field)
    required = {
            "count",
            "seed",
            "output_size",
            "min_scale",
            "max_scale",
            "brightness_range",
            "contrast_range",
            "blur_choices",
            "noise_sigma_range",
        }
    optional = {
        name for name in ("hue_shift_range", "saturation_scale_range")
        if name in data
    }
    _keys(data, required | optional, field)
    output_values = _array(data["output_size"], f"{field}.output_size", length=2)
    output_size = tuple(
        _integer(item, f"{field}.output_size[{index}]", minimum=1)
        for index, item in enumerate(output_values)
    )
    min_scale = _number(
        data["min_scale"],
        f"{field}.min_scale",
        minimum=0,
        strict_minimum=True,
    )
    max_scale = _number(data["max_scale"], f"{field}.max_scale", minimum=min_scale)
    blur_values = _array(data["blur_choices"], f"{field}.blur_choices")
    if not blur_values:
        raise ValueError(f"{field}.blur_choices must not be empty")
    blur_choices = tuple(
        _integer(item, f"{field}.blur_choices[{index}]", minimum=0)
        for index, item in enumerate(blur_values)
    )
    if any(kernel != 0 and kernel % 2 == 0 for kernel in blur_choices):
        raise ValueError(f"{field}.blur_choices must contain 0 or odd integers")
    noise_range = _number_range(data["noise_sigma_range"], f"{field}.noise_sigma_range")
    if noise_range[0] < 0:
        raise ValueError(f"{field}.noise_sigma_range must be non-negative")
    return GenerationSettings(
        count=_integer(data["count"], f"{field}.count", minimum=1),
        seed=_integer(data["seed"], f"{field}.seed", minimum=0),
        output_size=output_size,
        min_scale=min_scale,
        max_scale=max_scale,
        brightness_range=_number_range(
            data["brightness_range"], f"{field}.brightness_range"
        ),
        contrast_range=_number_range(
            data["contrast_range"],
            f"{field}.contrast_range",
            positive=True,
        ),
        blur_choices=blur_choices,
        noise_sigma_range=noise_range,
        hue_shift_range=(
            _number_range(data["hue_shift_range"], f"{field}.hue_shift_range")
            if "hue_shift_range" in data
            else (0.0, 0.0)
        ),
        saturation_scale_range=(
            _number_range(
                data["saturation_scale_range"],
                f"{field}.saturation_scale_range",
            )
            if "saturation_scale_range" in data
            else (1.0, 1.0)
        ),
    )


def _transform_from_dict(value: object, field: str) -> TransformRecord:
    data = _object(value, field)
    required = {"scale", "brightness", "contrast", "blur_kernel", "noise_sigma"}
    optional = {
        name for name in ("hue_shift_degrees", "saturation_scale") if name in data
    }
    _keys(data, required | optional, field)
    blur_kernel = _integer(data["blur_kernel"], f"{field}.blur_kernel", minimum=0)
    if blur_kernel != 0 and blur_kernel % 2 == 0:
        raise ValueError(f"{field}.blur_kernel must be 0 or an odd integer")
    return TransformRecord(
        scale=_number(data["scale"], f"{field}.scale", minimum=0, strict_minimum=True),
        brightness=_number(data["brightness"], f"{field}.brightness"),
        contrast=_number(
            data["contrast"],
            f"{field}.contrast",
            minimum=0,
            strict_minimum=True,
        ),
        blur_kernel=blur_kernel,
        noise_sigma=_number(data["noise_sigma"], f"{field}.noise_sigma", minimum=0),
        hue_shift_degrees=_number(
            data.get("hue_shift_degrees", 0.0),
            f"{field}.hue_shift_degrees",
            minimum=-180,
            maximum=180,
        ),
        saturation_scale=_number(
            data.get("saturation_scale", 1.0),
            f"{field}.saturation_scale",
            minimum=0,
            maximum=2,
        ),
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _read_schema(path: Path, kind: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {kind} schema: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"invalid {kind} schema: root must be an object")
    version = data.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(
            f"invalid {kind} schema: expected schema_version {SCHEMA_VERSION}"
        )
    return data


def save_project(
    path: Path,
    train_source: Path,
    train_roi: Rect | None,
    search_settings: SearchSettings,
    generation_settings: GenerationSettings,
    test_image_paths: tuple[Path, ...] = (),
    background_paths: tuple[Path, ...] = (),
) -> None:
    base_dir = path.resolve().parent
    _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "train": {
                "source": _portable_path(train_source, base_dir),
                "roi": _rect_to_dict(train_roi) if train_roi is not None else None,
            },
            "search_settings": _search_settings_to_dict(search_settings),
            "generation_settings": _generation_settings_to_dict(
                generation_settings
            ),
            "test_image_paths": [_portable_path(item, base_dir) for item in test_image_paths],
            "background_paths": [_portable_path(item, base_dir) for item in background_paths],
        },
    )


def load_project(
    path: Path,
    *, include_recent_paths: bool = False,
):
    data = _read_schema(path, "project")
    try:
        required = {
                "schema_version",
                "train",
                "search_settings",
                "generation_settings",
            }
        optional = {"test_image_paths", "background_paths"}
        if not required <= set(data) or set(data) - required - optional:
            raise ValueError("project contains invalid fields")
        train = _object(data["train"], "train")
        _keys(train, {"source", "roi"}, "train")
        roi_data = train["roi"]
        roi = None if roi_data is None else _rect_from_dict(roi_data, "train.roi")
        result = (
            _resolved_path(train["source"], path.resolve().parent, "train.source"),
            roi,
            _search_settings_from_dict(data["search_settings"]),
            _generation_settings_from_dict(data["generation_settings"]),
        )
        recent = []
        for name in ("test_image_paths", "background_paths"):
            values = _array(data.get(name, []), name)
            recent.append(tuple(_resolved_path(value, path.resolve().parent, f"{name}[{i}]") for i, value in enumerate(values)))
        return result + tuple(recent) if include_recent_paths else result
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid project schema: {error}") from error


def save_samples(path: Path, samples: list[GeneratedSample]) -> None:
    _write_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "samples": [
                {
                    "image_path": _portable_path(sample.image_path, path.parent),
                    "truth_box": _rect_to_dict(sample.truth_box),
                    "transform": {
                        "scale": sample.transform.scale,
                        "brightness": sample.transform.brightness,
                        "contrast": sample.transform.contrast,
                        "blur_kernel": sample.transform.blur_kernel,
                        "noise_sigma": sample.transform.noise_sigma,
                        "hue_shift_degrees": sample.transform.hue_shift_degrees,
                        "saturation_scale": sample.transform.saturation_scale,
                    },
                    "seed": sample.seed,
                }
                for sample in samples
            ],
        },
    )


def load_samples(path: Path) -> list[GeneratedSample]:
    data = _read_schema(path, "sample")
    try:
        _keys(data, {"schema_version", "samples"}, "sample")
        values = data["samples"]
        if not isinstance(values, list):
            raise ValueError("samples must be an array")
        samples = []
        for index, value in enumerate(values):
            field = f"samples[{index}]"
            sample = _object(value, field)
            _keys(sample, {"image_path", "truth_box", "transform", "seed"}, field)
            samples.append(
                GeneratedSample(
                    image_path=_resolved_path(
                        sample["image_path"],
                        path.resolve().parent,
                        f"{field}.image_path",
                    ),
                    truth_box=_rect_from_dict(
                        sample["truth_box"], f"{field}.truth_box"
                    ),
                    transform=_transform_from_dict(
                        sample["transform"], f"{field}.transform"
                    ),
                    seed=_integer(sample["seed"], f"{field}.seed", minimum=0),
                )
            )
        return samples
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid sample schema: {error}") from error


def export_results_csv(path: Path, records: list[EvaluationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        file = os.fdopen(fd, "w", encoding="utf-8-sig", newline="")
        with file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "image": record.image_path.as_posix(),
                        "success": record.success,
                        "score": record.score,
                        "iou": record.iou,
                        "center_error": record.center_error,
                        "scale_error_percent": record.scale_error_percent,
                        "elapsed_ms": record.elapsed_ms,
                    }
                )
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
