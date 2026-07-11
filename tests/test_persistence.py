import csv
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
