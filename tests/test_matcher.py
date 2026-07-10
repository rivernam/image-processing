from pathlib import Path

import cv2
import numpy as np
import pytest

from searchmax.matcher import _local_peaks, iou, match, non_max_suppression
from searchmax.models import MatchResult, Rect, SearchSettings
from searchmax.training import train_from_roi


def pattern() -> np.ndarray:
    template = np.zeros((24, 32, 3), np.uint8)
    cv2.rectangle(template, (2, 2), (29, 21), (20, 180, 240), 2)
    cv2.line(template, (4, 18), (27, 5), (255, 50, 40), 2)
    return template


def train_pattern():
    template = pattern()
    return template, train_from_roi(
        template,
        Path("p.png"),
        Rect(0, 0, 32, 24),
    )


def test_match_returns_requested_number_in_score_order():
    template, model = train_pattern()
    image = np.full((180, 260, 3), 35, np.uint8)
    image[20:44, 30:62] = template
    image[100:124, 180:212] = template
    settings = SearchSettings(
        min_scale=1,
        max_scale=1,
        scale_step=.02,
        threshold=.95,
        max_results=2,
    )

    results = match(model, image, settings)

    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert {(result.box.x, result.box.y) for result in results} == {
        (30, 20),
        (180, 100),
    }


def test_match_finds_scaled_pattern():
    template, model = train_pattern()
    scaled = cv2.resize(template, None, fx=1.25, fy=1.25)
    image = np.full((160, 220, 3), 35, np.uint8)
    image[50:50 + scaled.shape[0], 70:70 + scaled.shape[1]] = scaled

    result = match(model, image, SearchSettings(threshold=.9))[0]

    assert abs(result.scale - 1.25) <= .02
    assert result.box.x == 70
    assert result.box.y == 50


def test_local_peaks_keep_only_neighborhood_maxima_above_threshold():
    score_map = np.array([
        [.1, .1, .1, .1, .1],
        [.1, .95, .90, .1, .91],
        [.1, .1, .1, .1, .1],
    ], dtype=np.float32)

    peaks = _local_peaks(score_map, .8)

    assert [(x, y) for x, y, _ in peaks] == [(1, 1), (4, 1)]
    assert [score for _, _, score in peaks] == pytest.approx([.95, .91])


def test_iou_uses_intersection_over_union():
    assert iou(Rect(0, 0, 10, 10), Rect(5, 5, 10, 10)) == pytest.approx(25 / 175)


def test_non_max_suppression_removes_overlaps_orders_scores_and_limits_results():
    candidates = [
        MatchResult(.97, Rect(30, 30, 10, 10), 1),
        MatchResult(.98, Rect(1, 1, 10, 10), 1),
        MatchResult(.99, Rect(0, 0, 10, 10), 1),
        MatchResult(.96, Rect(60, 60, 10, 10), 1),
    ]

    results = non_max_suppression(candidates, threshold=.3, limit=2)

    assert [(result.score, result.box.x) for result in results] == [
        (.99, 0),
        (.97, 30),
    ]


def test_match_supports_gray_mode():
    template, model = train_pattern()
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_as_bgr = cv2.cvtColor(gray_template, cv2.COLOR_GRAY2BGR)
    image = np.full((100, 140, 3), 35, np.uint8)
    image[40:64, 50:82] = gray_as_bgr
    settings = SearchSettings(
        min_scale=1,
        max_scale=1,
        threshold=.99,
        color_mode="gray",
    )

    result = match(model, image, settings)[0]

    assert result.box == Rect(50, 40, 32, 24)
    assert result.score == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [.001, 10])
def test_match_skips_unusable_template_scales(scale):
    _, model = train_pattern()
    image = np.full((60, 80, 3), 35, np.uint8)
    settings = SearchSettings(
        min_scale=scale,
        max_scale=scale,
        scale_step=.02,
        threshold=.8,
    )

    assert match(model, image, settings) == []


def test_match_honors_max_results_one():
    template, model = train_pattern()
    image = np.full((100, 180, 3), 35, np.uint8)
    image[10:34, 20:52] = template
    image[60:84, 120:152] = template
    settings = SearchSettings(
        min_scale=1,
        max_scale=1,
        threshold=.95,
        max_results=1,
    )

    assert len(match(model, image, settings)) == 1
