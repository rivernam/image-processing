import math
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
    if not matches:
        return EvaluationRecord(
            image_path=sample.image_path,
            success=False,
            score=None,
            iou=0.0,
            center_error=None,
            scale_error_percent=None,
            elapsed_ms=0.0,
        )

    best = max(matches, key=lambda match: iou(sample.truth_box, match.box))
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
    if trained_width <= 0 or trained_height <= 0:
        raise ValueError("trained_size must contain two positive dimensions")
    truth_scale = fmean(
        (
            sample.truth_box.width / trained_width,
            sample.truth_box.height / trained_height,
        )
    )
    if truth_scale <= 0:
        raise ValueError("truth box must have positive dimensions")
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
