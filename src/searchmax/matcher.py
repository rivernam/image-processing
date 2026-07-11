from dataclasses import replace
from time import perf_counter

import cv2
import numpy as np

from .models import MatchResult, MatchResults, Rect, SearchSettings, TrainModel


MAX_CANDIDATES_PER_SCALE = 1_000


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
    peak_mask = ((score_map >= threshold) & (score_map == dilated)).astype(np.uint8)
    if not np.any(peak_mask):
        return []

    _, labels = cv2.connectedComponents(peak_mask)
    ys, xs = np.where(peak_mask)
    point_labels = labels[ys, xs]
    _, first_indices = np.unique(point_labels, return_index=True)
    ys = ys[first_indices]
    xs = xs[first_indices]
    scores = score_map[ys, xs]
    order = np.lexsort((xs, ys, -scores))[:MAX_CANDIDATES_PER_SCALE]
    return [
        (int(x), int(y), float(score_map[y, x]))
        for y, x in zip(ys[order], xs[order])
    ]


def _scale_values(settings: SearchSettings) -> list[float]:
    span = settings.max_scale - settings.min_scale
    tolerance = max(
        abs(settings.min_scale),
        abs(settings.max_scale),
        abs(settings.scale_step),
        1.0,
    ) * 1e-12
    step_count = int(np.floor(span / settings.scale_step + 1e-12))
    scales: list[float] = []
    for index in range(step_count + 1):
        scale = settings.min_scale + index * settings.scale_step
        if scale > settings.max_scale:
            if scale - settings.max_scale <= tolerance:
                scale = settings.max_scale
            else:
                break
        scales.append(scale)
    return scales


def _collect_candidates(
    model: TrainModel,
    search_image: np.ndarray,
    settings: SearchSettings,
) -> list[MatchResult]:
    candidates: list[MatchResult] = []
    search = (
        search_image
        if settings.color_mode == "color"
        else cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    )
    template0 = model.color if settings.color_mode == "color" else model.gray
    for scale in _scale_values(settings):
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

    return sorted(candidates, key=lambda r: (-r.score, r.box.y, r.box.x, r.scale))


def diagnostic_candidates(model, search_image, settings, limit: int = 100) -> list[MatchResult]:
    """Return deterministic, bounded candidates before NMS/max_results."""
    if not 0 <= limit <= 100:
        raise ValueError("diagnostic limit must be in [0, 100]")
    return _collect_candidates(model, search_image, settings)[:limit]


def match_with_diagnostics(model, search_image, settings, limit: int = 100):
    started = perf_counter()
    candidates = _collect_candidates(model, search_image, settings)
    selected = non_max_suppression(candidates, settings.nms_iou_threshold, settings.max_results)
    elapsed_ms = (perf_counter() - started) * 1000
    final = MatchResults((replace(item, elapsed_ms=elapsed_ms) for item in selected), elapsed_ms=elapsed_ms)
    diagnostics = [replace(item, elapsed_ms=elapsed_ms) for item in candidates[:limit]]
    return final, diagnostics


def match(model: TrainModel, search_image: np.ndarray, settings: SearchSettings) -> MatchResults:
    return match_with_diagnostics(model, search_image, settings, 0)[0]
