from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Rect:
    x: int; y: int; width: int; height: int

    @property
    def area(self) -> int: return max(0, self.width) * max(0, self.height)


@dataclass(frozen=True)
class SearchSettings:
    min_scale: float = .8; max_scale: float = 1.5; scale_step: float = .02
    threshold: float = .8; max_results: int = 1
    nms_iou_threshold: float = .3; color_mode: str = "color"

    def __post_init__(self):
        if self.min_scale <= 0 or self.max_scale < self.min_scale: raise ValueError("invalid scale range")
        if self.scale_step <= 0: raise ValueError("scale_step must be positive")
        if not 0 <= self.threshold <= 1: raise ValueError("threshold must be in [0, 1]")
        if not 1 <= self.max_results <= 100: raise ValueError("max_results must be in [1, 100]")
        if not 0 <= self.nms_iou_threshold <= 1: raise ValueError("nms_iou_threshold must be in [0, 1]")
        if self.color_mode not in {"color", "gray"}: raise ValueError("invalid color_mode")


@dataclass
class TrainModel:
    color: np.ndarray; gray: np.ndarray; source: Path; roi: Rect | None


@dataclass(frozen=True)
class MatchResult:
    score: float; box: Rect; scale: float; elapsed_ms: float = 0.0


@dataclass(frozen=True)
class TransformRecord:
    scale: float; brightness: float; contrast: float; blur_kernel: int; noise_sigma: float


@dataclass(frozen=True)
class GeneratedSample:
    image_path: Path; truth_box: Rect; transform: TransformRecord; seed: int


@dataclass(frozen=True)
class EvaluationRecord:
    image_path: Path; success: bool; score: float | None; iou: float
    center_error: float | None; scale_error_percent: float | None; elapsed_ms: float


@dataclass(frozen=True)
class EvaluationSummary:
    total: int; successes: int; success_rate: float; mean_iou: float
    mean_center_error: float; mean_scale_error_percent: float; mean_elapsed_ms: float
