from dataclasses import replace
from time import perf_counter

import cv2
import numpy as np

from .models import MatchResult, Rect, SearchSettings, TrainModel


def iou(a: Rect, b: Rect) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - intersection
    return intersection / union if union else 0.0


def non_max_suppression(
    candidates: list[MatchResult],
    threshold: float,
    limit: int,
) -> list[MatchResult]:
    if limit <= 0:
        return []

    kept: list[MatchResult] = []
    for candidate in sorted(candidates, key=lambda result: result.score, reverse=True):
        if all(iou(candidate.box, item.box) <= threshold for item in kept):
            kept.append(candidate)
        if len(kept) == limit:
            break
    return kept


def _local_peaks(
    score_map: np.ndarray,
    threshold: float,
) -> list[tuple[int, int, float]]:
    dilated = cv2.dilate(score_map, np.ones((3, 3), np.uint8))
    ys, xs = np.where((score_map >= threshold) & (score_map == dilated))
    return [
        (int(x), int(y), float(score_map[y, x]))
        for y, x in zip(ys, xs)
    ]


def match(
    model: TrainModel,
    search_image: np.ndarray,
    settings: SearchSettings,
) -> list[MatchResult]:
    started = perf_counter()
    candidates: list[MatchResult] = []
    search = (
        search_image
        if settings.color_mode == "color"
        else cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    )
    template0 = model.color if settings.color_mode == "color" else model.gray
    scales = np.arange(
        settings.min_scale,
        settings.max_scale + settings.scale_step / 2,
        settings.scale_step,
    )

    for scale in scales:
        scaled_width = round(template0.shape[1] * float(scale))
        scaled_height = round(template0.shape[0] * float(scale))
        if scaled_width < 1 or scaled_height < 1:
            continue
        if scaled_width > search.shape[1] or scaled_height > search.shape[0]:
            continue

        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        template = cv2.resize(
            template0,
            (scaled_width, scaled_height),
            interpolation=interpolation,
        )
        score_map = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        for x, y, score in _local_peaks(score_map, settings.threshold):
            candidates.append(
                MatchResult(
                    max(0.0, min(1.0, score)),
                    Rect(x, y, scaled_width, scaled_height),
                    float(scale),
                )
            )

    if not candidates:
        return []

    elapsed_ms = (perf_counter() - started) * 1000
    return [
        replace(result, elapsed_ms=elapsed_ms)
        for result in non_max_suppression(
            candidates,
            settings.nms_iou_threshold,
            settings.max_results,
        )
    ]
