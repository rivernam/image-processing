from pathlib import Path

import pytest

from searchmax.evaluator import evaluate_sample, summarize
from searchmax.models import (
    EvaluationRecord,
    GeneratedSample,
    MatchResult,
    Rect,
    TransformRecord,
)


def _sample() -> GeneratedSample:
    return GeneratedSample(
        Path("x.png"),
        Rect(50, 40, 100, 60),
        TransformRecord(1, 0, 1, 0, 0),
        1,
    )


def test_evaluation_uses_best_iou_and_summarizes() -> None:
    matches = [
        MatchResult(0.99, Rect(0, 0, 100, 60), 1.0, 5.0),
        MatchResult(0.92, Rect(52, 41, 98, 59), 0.99, 12.0),
    ]

    record = evaluate_sample(_sample(), matches, (100, 60), 0.5)
    summary = summarize([record])

    assert record.success
    assert record.score == 0.92
    assert record.iou > 0.9
    assert record.center_error == pytest.approx(1.118033988749895)
    assert record.scale_error_percent == pytest.approx(-1.0)
    assert record.elapsed_ms == 12.0
    assert summary.success_rate == 1.0


def test_evaluation_requires_iou_threshold() -> None:
    match = MatchResult(0.95, Rect(100, 40, 100, 60), 1.0, 3.0)

    record = evaluate_sample(_sample(), [match], (100, 60), 0.5)

    assert not record.success
    assert record.iou == pytest.approx(1 / 3)


def test_evaluation_without_matches_uses_empty_values() -> None:
    record = evaluate_sample(_sample(), [], (100, 60))

    assert record == EvaluationRecord(
        Path("x.png"), False, None, 0.0, None, None, 0.0
    )


@pytest.mark.parametrize(
    "trained_size",
    [(0, 60), (100, -1), (True, 60), (100, float("inf"))],
)
def test_evaluation_rejects_invalid_trained_size(
    trained_size: tuple[object, object],
) -> None:
    with pytest.raises(ValueError, match="trained_size"):
        evaluate_sample(_sample(), [], trained_size)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, float("nan")])
def test_evaluation_rejects_invalid_iou_threshold(threshold: object) -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_sample(_sample(), [], (100, 60), threshold)  # type: ignore[arg-type]


@pytest.mark.parametrize("width,height", [(-1, 60), (100, 0)])
def test_evaluation_rejects_each_invalid_truth_dimension(
    width: int, height: int
) -> None:
    sample = GeneratedSample(
        Path("x.png"),
        Rect(50, 40, width, height),
        TransformRecord(1, 0, 1, 0, 0),
        1,
    )

    with pytest.raises(ValueError, match="truth box"):
        evaluate_sample(sample, [], (100, 60))


@pytest.mark.parametrize("scale", [0.0, -1.0, True, float("inf")])
def test_evaluation_rejects_invalid_detected_scale(scale: object) -> None:
    match = MatchResult(
        0.9, Rect(50, 40, 100, 60), scale, 1.0  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="detected scale"):
        evaluate_sample(_sample(), [match], (100, 60))


def test_summarize_uses_arithmetic_means_and_handles_empty_input() -> None:
    records = [
        EvaluationRecord(Path("a.png"), True, 0.9, 0.8, 2.0, -10.0, 4.0),
        EvaluationRecord(Path("b.png"), False, None, 0.0, None, None, 8.0),
    ]

    summary = summarize(records)
    empty = summarize([])

    assert summary.total == 2
    assert summary.successes == 1
    assert summary.success_rate == 0.5
    assert summary.mean_iou == 0.4
    assert summary.mean_center_error == 2.0
    assert summary.mean_scale_error_percent == -10.0
    assert summary.mean_elapsed_ms == 6.0
    assert empty.total == 0
    assert empty.successes == 0
    assert empty.success_rate == 0.0
    assert empty.mean_iou == 0.0
    assert empty.mean_center_error == 0.0
    assert empty.mean_scale_error_percent == 0.0
    assert empty.mean_elapsed_ms == 0.0
