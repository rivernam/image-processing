import csv
import json
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
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return str(path)


def _resolved_path(value: object, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("path must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _rect_to_dict(rect: Rect) -> dict[str, int]:
    return {
        "x": rect.x,
        "y": rect.y,
        "width": rect.width,
        "height": rect.height,
    }


def _rect_from_dict(value: object) -> Rect:
    if not isinstance(value, dict):
        raise ValueError("rectangle must be an object")
    return Rect(
        x=value["x"],
        y=value["y"],
        width=value["width"],
        height=value["height"],
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
    }


def _generation_settings_from_dict(value: object) -> GenerationSettings:
    if not isinstance(value, dict):
        raise ValueError("generation settings must be an object")
    return GenerationSettings(
        count=value["count"],
        seed=value["seed"],
        output_size=tuple(value["output_size"]),
        min_scale=value["min_scale"],
        max_scale=value["max_scale"],
        brightness_range=tuple(value["brightness_range"]),
        contrast_range=tuple(value["contrast_range"]),
        blur_choices=tuple(value["blur_choices"]),
        noise_sigma_range=tuple(value["noise_sigma_range"]),
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _read_schema(path: Path, kind: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {kind} schema: {error}") from error
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
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
) -> None:
    base_dir = path.parent
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
        },
    )


def load_project(
    path: Path,
) -> tuple[Path, Rect | None, SearchSettings, GenerationSettings]:
    data = _read_schema(path, "project")
    try:
        train = data["train"]
        if not isinstance(train, dict):
            raise ValueError("train must be an object")
        roi_data = train["roi"]
        roi = None if roi_data is None else _rect_from_dict(roi_data)
        search_data = data["search_settings"]
        if not isinstance(search_data, dict):
            raise ValueError("search settings must be an object")
        return (
            _resolved_path(train["source"], path.parent),
            roi,
            SearchSettings(**search_data),
            _generation_settings_from_dict(data["generation_settings"]),
        )
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
        values = data["samples"]
        if not isinstance(values, list):
            raise ValueError("samples must be an array")
        samples = []
        for value in values:
            if not isinstance(value, dict):
                raise ValueError("sample must be an object")
            transform = value["transform"]
            if not isinstance(transform, dict):
                raise ValueError("transform must be an object")
            samples.append(
                GeneratedSample(
                    image_path=_resolved_path(value["image_path"], path.parent),
                    truth_box=_rect_from_dict(value["truth_box"]),
                    transform=TransformRecord(**transform),
                    seed=value["seed"],
                )
            )
        return samples
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid sample schema: {error}") from error


def export_results_csv(path: Path, records: list[EvaluationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image": str(record.image_path),
                    "success": record.success,
                    "score": record.score,
                    "iou": record.iou,
                    "center_error": record.center_error,
                    "scale_error_percent": record.scale_error_percent,
                    "elapsed_ms": record.elapsed_ms,
                }
            )
