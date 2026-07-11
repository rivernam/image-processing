import csv
import copy
import json
from pathlib import Path

import pytest

from searchmax.generator import GenerationSettings
from searchmax.models import (
    EvaluationRecord,
    GeneratedSample,
    Rect,
    SearchSettings,
    TransformRecord,
)
from searchmax.persistence import (
    export_results_csv,
    load_project,
    load_samples,
    save_project,
    save_samples,
)


def test_project_round_trip_optionally_preserves_test_and_background_paths(tmp_path):
    project = tmp_path / "project" / "settings.json"
    source = project.parent / "train.png"
    tests = (project.parent / "tests" / "one.png", tmp_path / "external.png")
    backgrounds = (project.parent / "background.png",)
    save_project(project, source, None, SearchSettings(), GenerationSettings(), tests, backgrounds)
    loaded = load_project(project, include_recent_paths=True)
    assert loaded[4:] == (tests, backgrounds)


def test_project_round_trip_preserves_settings_and_korean_relative_path(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project" / "설정.json"
    source = project_path.parent / "이미지" / "훈련.png"
    roi = Rect(10, 20, 30, 40)
    search = SearchSettings(
        min_scale=0.7,
        max_scale=1.8,
        scale_step=0.05,
        threshold=0.91,
        max_results=3,
        nms_iou_threshold=0.25,
        color_mode="gray",
    )
    generation = GenerationSettings(
        count=4,
        seed=77,
        output_size=(640, 360),
        min_scale=0.7,
        max_scale=1.8,
        brightness_range=(-5, 8),
        contrast_range=(0.8, 1.2),
        blur_choices=(0, 3, 5),
        noise_sigma_range=(0, 2),
    )

    save_project(project_path, source, roi, search, generation)

    assert load_project(project_path) == (source, roi, search, generation)
    raw = project_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["schema_version"] == 1
    assert data["train"]["source"] == "이미지/훈련.png"
    assert "훈련" in raw
    assert raw.startswith("{\n  ")


def test_project_round_trip_supports_no_roi_and_absolute_external_path(
    tmp_path: Path,
) -> None:
    project_path = tmp_path / "project.json"
    external = tmp_path.parent / "external.png"
    search = SearchSettings()
    generation = GenerationSettings()

    save_project(project_path, external, None, search, generation)

    assert load_project(project_path) == (external, None, search, generation)
    data = json.loads(project_path.read_text(encoding="utf-8"))
    assert Path(data["train"]["source"]).is_absolute()


def test_relative_external_source_is_resolved_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    project_path = Path("project/settings.json")

    save_project(
        project_path,
        Path("external.png"),
        None,
        SearchSettings(),
        GenerationSettings(),
    )

    data = json.loads(project_path.read_text(encoding="utf-8"))
    assert data["train"]["source"] == str((tmp_path / "external.png").resolve())
    assert load_project(project_path)[0] == (tmp_path / "external.png").resolve()


def test_absolute_child_is_relative_with_relative_project_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    project_path = Path("project/settings.json")
    source = (tmp_path / "project/images/train.png").resolve()

    save_project(
        project_path,
        source,
        None,
        SearchSettings(),
        GenerationSettings(),
    )

    data = json.loads(project_path.read_text(encoding="utf-8"))
    assert data["train"]["source"] == "images/train.png"
    assert load_project(project_path)[0] == source


def test_normalized_parent_escape_is_stored_as_absolute(tmp_path: Path) -> None:
    project_path = tmp_path / "project/settings.json"
    source = tmp_path / "project/../outside.png"

    save_project(
        project_path,
        source,
        None,
        SearchSettings(),
        GenerationSettings(),
    )

    data = json.loads(project_path.read_text(encoding="utf-8"))
    assert data["train"]["source"] == str((tmp_path / "outside.png").resolve())


def _valid_project_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "train": {
            "source": "images/train.png",
            "roi": {"x": 1, "y": 2, "width": 30, "height": 40},
        },
        "search_settings": {
            "min_scale": 0.8,
            "max_scale": 1.5,
            "scale_step": 0.02,
            "threshold": 0.8,
            "max_results": 1,
            "nms_iou_threshold": 0.3,
            "color_mode": "color",
        },
        "generation_settings": {
            "count": 20,
            "seed": 1234,
            "output_size": [1280, 720],
            "min_scale": 0.8,
            "max_scale": 1.5,
            "brightness_range": [-12, 12],
            "contrast_range": [0.9, 1.1],
            "blur_choices": [0, 3],
            "noise_sigma_range": [0, 3],
        },
    }


def _nested_payload(
    base: dict[str, object], keys: tuple[str, ...], value: object
) -> dict[str, object]:
    payload = copy.deepcopy(base)
    target: dict[str, object] = payload
    for key in keys[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[keys[-1]] = value
    return payload


@pytest.mark.parametrize("version", [True, 1.0])
def test_project_schema_version_must_be_exact_integer(
    tmp_path: Path, version: object
) -> None:
    payload = _nested_payload(_valid_project_payload(), ("schema_version",), version)
    path = tmp_path / "project.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="project schema.*schema_version"):
        load_project(path)


@pytest.mark.parametrize(
    "keys,value,field",
    [
        (("train", "roi", "width"), 0, "train.roi.width"),
        (("train", "roi", "x"), True, "train.roi.x"),
        (("search_settings", "min_scale"), True, "search_settings.min_scale"),
        (("search_settings", "threshold"), float("nan"), "search_settings.threshold"),
        (("search_settings", "max_results"), 1.5, "search_settings.max_results"),
        (("search_settings", "color_mode"), 3, "search_settings.color_mode"),
        (("generation_settings", "count"), True, "generation_settings.count"),
        (("generation_settings", "seed"), -1, "generation_settings.seed"),
        (
            ("generation_settings", "output_size"),
            "1280,720",
            "generation_settings.output_size",
        ),
        (
            ("generation_settings", "output_size"),
            [True, 720],
            "generation_settings.output_size",
        ),
        (
            ("generation_settings", "brightness_range"),
            [0, float("inf")],
            "generation_settings.brightness_range",
        ),
        (
            ("generation_settings", "blur_choices"),
            [0, 2],
            "generation_settings.blur_choices",
        ),
        (
            ("generation_settings", "noise_sigma_range"),
            [-1, 3],
            "generation_settings.noise_sigma_range",
        ),
    ],
)
def test_load_project_rejects_invalid_primitive_with_field_name(
    tmp_path: Path, keys: tuple[str, ...], value: object, field: str
) -> None:
    path = tmp_path / "project.json"
    path.write_text(
        json.dumps(_nested_payload(_valid_project_payload(), keys, value)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"project schema.*{field}"):
        load_project(path)


def test_load_project_rejects_relative_path_escape(tmp_path: Path) -> None:
    payload = _nested_payload(
        _valid_project_payload(), ("train", "source"), "../outside.png"
    )
    path = tmp_path / "project.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="project schema.*train.source"):
        load_project(path)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": 2},
        {"schema_version": 1, "train": {}},
    ],
)
def test_load_project_rejects_malformed_schema(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="project schema"):
        load_project(path)


def test_samples_round_trip_preserves_korean_paths(tmp_path: Path) -> None:
    path = tmp_path / "자료" / "샘플.json"
    samples = [
        GeneratedSample(
            path.parent / "이미지" / "첫째.png",
            Rect(1, 2, 30, 40),
            TransformRecord(1.25, -3.0, 0.9, 3, 1.5),
            42,
        )
    ]

    save_samples(path, samples)

    assert load_samples(path) == samples
    raw = path.read_text(encoding="utf-8")
    assert "첫째" in raw
    assert "이미지/첫째.png" in raw


def _valid_sample_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "samples": [
            {
                "image_path": "images/sample.png",
                "truth_box": {"x": 1, "y": 2, "width": 30, "height": 40},
                "transform": {
                    "scale": 1.25,
                    "brightness": -3.0,
                    "contrast": 0.9,
                    "blur_kernel": 3,
                    "noise_sigma": 1.5,
                },
                "seed": 42,
            }
        ],
    }


@pytest.mark.parametrize(
    "keys,value,field",
    [
        (("samples", "0", "truth_box", "width"), 0, "truth_box.width"),
        (("samples", "0", "truth_box", "height"), True, "truth_box.height"),
        (("samples", "0", "seed"), True, "seed"),
        (("samples", "0", "seed"), -1, "seed"),
        (("samples", "0", "transform", "scale"), 0, "transform.scale"),
        (("samples", "0", "transform", "scale"), float("nan"), "transform.scale"),
        (("samples", "0", "transform", "brightness"), "bright", "transform.brightness"),
        (("samples", "0", "transform", "contrast"), 0, "transform.contrast"),
        (("samples", "0", "transform", "blur_kernel"), True, "transform.blur_kernel"),
        (("samples", "0", "transform", "blur_kernel"), 2, "transform.blur_kernel"),
        (("samples", "0", "transform", "noise_sigma"), -1, "transform.noise_sigma"),
    ],
)
def test_load_samples_rejects_invalid_primitive_with_field_name(
    tmp_path: Path, keys: tuple[str, ...], value: object, field: str
) -> None:
    payload = copy.deepcopy(_valid_sample_payload())
    sample = payload["samples"][0]  # type: ignore[index]
    target: dict[str, object] = sample  # type: ignore[assignment]
    for key in keys[2:-1]:
        target = target[key]  # type: ignore[assignment]
    target[keys[-1]] = value
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"sample schema.*{field}"):
        load_samples(path)


def test_load_samples_rejects_relative_path_escape(tmp_path: Path) -> None:
    payload = copy.deepcopy(_valid_sample_payload())
    payload["samples"][0]["image_path"] = "../outside.png"  # type: ignore[index]
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sample schema.*image_path"):
        load_samples(path)


def test_load_samples_rejects_malformed_schema(tmp_path: Path) -> None:
    path = tmp_path / "samples.json"
    path.write_text('{"schema_version": 9, "samples": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="sample schema"):
        load_samples(path)


def test_export_results_csv_uses_exact_columns_and_values(tmp_path: Path) -> None:
    path = tmp_path / "결과.csv"
    records = [
        EvaluationRecord(
            Path("이미지/첫째.png"), True, 0.92, 0.88, 1.5, -1.0, 12.0
        ),
        EvaluationRecord(Path("둘째.png"), False, None, 0.0, None, None, 3.5),
    ]

    export_results_csv(path, records)

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    assert rows == [
        [
            "image",
            "success",
            "score",
            "iou",
            "center_error",
            "scale_error_percent",
            "elapsed_ms",
        ],
        ["이미지/첫째.png", "True", "0.92", "0.88", "1.5", "-1.0", "12.0"],
        ["둘째.png", "False", "", "0.0", "", "", "3.5"],
    ]
