import math
from numbers import Integral, Real
from statistics import fmean

from .matcher import iou
from .models import (
    EvaluationRecord,
    EvaluationSummary,
    GeneratedSample,
    MatchResult,
)


def evaluate_sample(
    sample: GeneratedSample,
    matches: list[MatchResult],
    trained_size: tuple[int, int],
    iou_threshold: float = 0.5,
) -> EvaluationRecord:
    if (
        not isinstance(trained_size, tuple)
        or len(trained_size) != 2
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, Real)
            or not math.isfinite(float(dimension))
            or dimension <= 0
            for dimension in trained_size
        )
    ):
        raise ValueError("trained_size must contain two positive finite numbers")
    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, Real)
        or not math.isfinite(float(iou_threshold))
        or not 0 <= iou_threshold <= 1
    ):
        raise ValueError("iou_threshold must be a finite number in [0, 1]")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, Integral)
        or dimension <= 0
        for dimension in (sample.truth_box.width, sample.truth_box.height)
    ):
        raise ValueError("truth box width and height must be positive integers")
    if not matches:
        return EvaluationRecord(
            image_path=sample.image_path,
            success=False,
            score=None,
            iou=0.0,
            center_error=None,
            scale_error_percent=None,
            elapsed_ms=float(getattr(matches, "elapsed_ms", 0.0)),
        )

    best = max(matches, key=lambda match: iou(sample.truth_box, match.box))
    if (
        isinstance(best.scale, bool)
        or not isinstance(best.scale, Real)
        or not math.isfinite(float(best.scale))
        or best.scale <= 0
    ):
        raise ValueError("detected scale must be a positive finite number")
    best_iou = iou(sample.truth_box, best.box)
    truth_center = (
        sample.truth_box.x + sample.truth_box.width / 2,
        sample.truth_box.y + sample.truth_box.height / 2,
    )
    detected_center = (
        best.box.x + best.box.width / 2,
        best.box.y + best.box.height / 2,
    )
    center_error = math.dist(truth_center, detected_center)

    trained_width, trained_height = trained_size
    truth_scale = fmean(
        (
            sample.truth_box.width / trained_width,
            sample.truth_box.height / trained_height,
        )
    )
    scale_error_percent = (best.scale / truth_scale - 1) * 100

    return EvaluationRecord(
        image_path=sample.image_path,
        success=best_iou >= iou_threshold,
        score=best.score,
        iou=best_iou,
        center_error=center_error,
        scale_error_percent=scale_error_percent,
        elapsed_ms=best.elapsed_ms,
    )


def _mean_or_zero(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def summarize(records: list[EvaluationRecord]) -> EvaluationSummary:
    total = len(records)
    successes = sum(record.success for record in records)
    center_errors = [
        record.center_error
        for record in records
        if record.center_error is not None
    ]
    scale_errors = [
        record.scale_error_percent
        for record in records
        if record.scale_error_percent is not None
    ]
    return EvaluationSummary(
        total=total,
        successes=successes,
        success_rate=successes / total if total else 0.0,
        mean_iou=_mean_or_zero([record.iou for record in records]),
        mean_center_error=_mean_or_zero(center_errors),
        mean_scale_error_percent=_mean_or_zero(scale_errors),
        mean_elapsed_ms=_mean_or_zero(
            [record.elapsed_ms for record in records]
        ),
    )
