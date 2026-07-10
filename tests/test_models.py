import pytest

from searchmax.models import SearchSettings


@pytest.mark.parametrize("field,value", [
    ("min_scale", 0.0), ("max_scale", 0.79), ("scale_step", 0.0),
    ("threshold", 1.1), ("max_results", 0), ("max_results", 101),
    ("nms_iou_threshold", -0.1),
])
def test_invalid_search_settings(field, value):
    values = {"min_scale": .8, "max_scale": 1.5, "scale_step": .02,
              "threshold": .8, "max_results": 1, "nms_iou_threshold": .3,
              "color_mode": "color"}
    values[field] = value
    with pytest.raises(ValueError):
        SearchSettings(**values)
