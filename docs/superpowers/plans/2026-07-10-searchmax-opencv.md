# OpenCV SearchMax 유사 도구 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Windows 11에서 UI 이미지 Train, 80~150% 다중 스케일 NCC 검색, 제한 개수의 점수별 검출 결과, 합성 테스트 생성과 자동 평가를 제공하는 PySide6 앱을 만든다.

**Architecture:** 순수 Python 도메인 모델과 OpenCV 서비스를 `src/searchmax/`에 두고 PySide6 UI는 이 인터페이스만 호출한다. Matcher는 스케일별 지역 최댓값을 합친 뒤 IoU NMS와 최대 검출 개수를 적용하며, Generator와 Evaluator는 정답 메타데이터를 통해 독립적으로 검증한다.

**Tech Stack:** Python 3.11+, PySide6, OpenCV (`opencv-python`), NumPy, pytest, pytest-qt

## Global Constraints

- 대상 운영체제는 Windows 11이다.
- Python 3.11 이상을 사용한다.
- 기본 검색 범위는 균일 스케일 80~150%, 간격 2%이며 회전·skew·원근 검색은 하지 않는다.
- 기본 매칭은 컬러 `cv2.TM_CCOEFF_NORMED`, 선택 모드는 그레이스케일이다.
- 최대 검출 개수는 기본 1, 허용 범위 1~100이다.
- 기본 합격 임계값은 0.80, 기본 NMS IoU 임계값은 0.30이다.
- 한글 경로는 `numpy.fromfile`/`cv2.imdecode`와 `cv2.imencode`/`tofile` 조합으로 지원한다.
- UI 스레드에서 일괄 검색을 직접 실행하지 않는다.
- Cognex VPP 호환과 Cognex 점수의 수치적 동일성은 범위 밖이다.

## File Map

- `pyproject.toml`: 패키지, 런타임 및 테스트 의존성, pytest 설정
- `README.md`: Windows 설치, 실행, 사용자 흐름, 제한 사항
- `src/searchmax/__init__.py`: 공개 패키지 버전
- `src/searchmax/models.py`: 설정, ROI, Train 모델, 검출 및 평가 데이터 클래스
- `src/searchmax/image_io.py`: 한글 경로 안전 이미지 입출력
- `src/searchmax/training.py`: ROI/파일 Train과 템플릿 유효성 검사
- `src/searchmax/matcher.py`: 다중 스케일 NCC, 지역 최댓값, NMS
- `src/searchmax/generator.py`: 재현 가능한 합성 테스트 세트 생성
- `src/searchmax/evaluator.py`: IoU, 일대일 대응, 요약 통계
- `src/searchmax/persistence.py`: 프로젝트 JSON, 메타데이터 JSON, CSV 저장
- `src/searchmax/ui/image_view.py`: 확대·패닝·ROI·오버레이 이미지 뷰
- `src/searchmax/ui/workers.py`: 검색 및 생성 백그라운드 작업
- `src/searchmax/ui/main_window.py`: 설정 패널, 결과 표, 전체 워크플로
- `src/searchmax/app.py`: QApplication 진입점
- `tests/`: 위 모듈과 같은 이름의 단위·통합 테스트

---

### Task 1: 프로젝트 기반과 도메인 모델

**Files:**
- Create: `pyproject.toml`
- Create: `src/searchmax/__init__.py`
- Create: `src/searchmax/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: 없음
- Produces: `Rect`, `SearchSettings`, `TrainModel`, `MatchResult`, `TransformRecord`, `GeneratedSample`, `EvaluationRecord`, `EvaluationSummary`

- [ ] **Step 1: 설정 유효성 실패 테스트 작성**

```python
# tests/test_models.py
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
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'searchmax'`

- [ ] **Step 3: 패키지 설정과 모델 구현**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "opencv-searchmax"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "opencv-python>=4.9", "PySide6>=6.7"]

[project.optional-dependencies]
test = ["pytest>=8.0", "pytest-qt>=4.4"]

[project.scripts]
opencv-searchmax = "searchmax.app:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

```python
# src/searchmax/models.py
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
```

- [ ] **Step 4: 모델 테스트 통과 확인**

Run: `python -m pip install -e '.[test]' && python -m pytest tests/test_models.py -v`
Expected: all parameterized cases PASS

- [ ] **Step 5: 커밋**

```bash
git add pyproject.toml src/searchmax tests/test_models.py
git commit -m "feat: define searchmax domain models"
```

### Task 2: 이미지 입출력과 Train

**Files:**
- Create: `src/searchmax/image_io.py`
- Create: `src/searchmax/training.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Consumes: `Rect`, `TrainModel`
- Produces: `read_image(path) -> np.ndarray`, `write_image(path, image) -> None`, `train_from_roi(image, source, roi) -> TrainModel`, `train_from_file(path) -> TrainModel`

- [ ] **Step 1: ROI와 저분산 템플릿 테스트 작성**

```python
# tests/test_training.py
from pathlib import Path
import numpy as np
import pytest
from searchmax.models import Rect
from searchmax.training import train_from_roi

def test_train_from_roi_copies_expected_pixels():
    image = np.zeros((80, 100, 3), np.uint8)
    image[20:50, 10:50] = np.arange(40, dtype=np.uint8)[None, :, None]
    model = train_from_roi(image, Path("한글화면.png"), Rect(10, 20, 40, 30))
    assert model.color.shape == (30, 40, 3)
    assert model.gray.shape == (30, 40)

def test_train_rejects_low_variance_template():
    with pytest.raises(ValueError, match="contrast"):
        train_from_roi(np.full((50, 50, 3), 127, np.uint8), Path("x.png"), Rect(0, 0, 50, 50))
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_training.py -v`
Expected: FAIL because `searchmax.training` does not exist

- [ ] **Step 3: 한글 경로 입출력과 Train 구현**

```python
# src/searchmax/image_io.py
from pathlib import Path
import cv2
import numpy as np

def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f"cannot decode image: {path}")
    return image

def write_image(path: Path, image: np.ndarray) -> None:
    ext = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok: raise ValueError(f"cannot encode image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)
```

```python
# src/searchmax/training.py
from pathlib import Path
import cv2
import numpy as np
from .image_io import read_image
from .models import Rect, TrainModel

MIN_SIDE, MIN_STDDEV = 8, 2.0

def train_from_roi(image: np.ndarray, source: Path, roi: Rect) -> TrainModel:
    h, w = image.shape[:2]
    if roi.width < MIN_SIDE or roi.height < MIN_SIDE: raise ValueError("ROI is too small")
    if roi.x < 0 or roi.y < 0 or roi.x + roi.width > w or roi.y + roi.height > h: raise ValueError("ROI outside image")
    color = image[roi.y:roi.y + roi.height, roi.x:roi.x + roi.width].copy()
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < MIN_STDDEV: raise ValueError("template contrast is too low")
    return TrainModel(color, gray, source, roi)

def train_from_file(path: Path) -> TrainModel:
    image = read_image(path)
    return train_from_roi(image, path, Rect(0, 0, image.shape[1], image.shape[0]))
```

- [ ] **Step 4: 테스트 통과와 한글 경로 왕복 확인**

Add a `tmp_path / "한글 이미지.png"` round-trip assertion using `write_image` and `read_image`, then run: `python -m pytest tests/test_training.py -v`
Expected: all tests PASS

- [ ] **Step 5: 커밋**

```bash
git add src/searchmax/image_io.py src/searchmax/training.py tests/test_training.py
git commit -m "feat: add template training and unicode image IO"
```

### Task 3: 다중 스케일 Matcher와 최대 결과 개수

**Files:**
- Create: `src/searchmax/matcher.py`
- Test: `tests/test_matcher.py`

**Interfaces:**
- Consumes: `TrainModel`, `SearchSettings`, `Rect`, `MatchResult`
- Produces: `iou(a, b) -> float`, `non_max_suppression(candidates, threshold, limit) -> list[MatchResult]`, `match(model, search_image, settings) -> list[MatchResult]`

- [ ] **Step 1: 단일·스케일·다중 검출 테스트 작성**

```python
# tests/test_matcher.py
from pathlib import Path
import cv2, numpy as np
from searchmax.matcher import match
from searchmax.models import Rect, SearchSettings
from searchmax.training import train_from_roi

def pattern():
    p = np.zeros((24, 32, 3), np.uint8)
    cv2.rectangle(p, (2, 2), (29, 21), (20, 180, 240), 2)
    cv2.line(p, (4, 18), (27, 5), (255, 50, 40), 2)
    return p

def test_match_returns_requested_number_in_score_order():
    p = pattern(); image = np.full((180, 260, 3), 35, np.uint8)
    image[20:44, 30:62] = p; image[100:124, 180:212] = p
    model = train_from_roi(p, Path("p.png"), Rect(0, 0, 32, 24))
    settings = SearchSettings(min_scale=1, max_scale=1, scale_step=.02, threshold=.95, max_results=2)
    results = match(model, image, settings)
    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert {(r.box.x, r.box.y) for r in results} == {(30, 20), (180, 100)}

def test_match_finds_scaled_pattern():
    p = pattern(); scaled = cv2.resize(p, None, fx=1.25, fy=1.25)
    image = np.full((160, 220, 3), 35, np.uint8)
    image[50:50+scaled.shape[0], 70:70+scaled.shape[1]] = scaled
    model = train_from_roi(p, Path("p.png"), Rect(0, 0, 32, 24))
    result = match(model, image, SearchSettings(threshold=.9))[0]
    assert abs(result.scale - 1.25) <= .02
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_matcher.py -v`
Expected: FAIL because `searchmax.matcher` does not exist

- [ ] **Step 3: Matcher 최소 구현**

```python
# src/searchmax/matcher.py
from dataclasses import replace
from time import perf_counter
import cv2
import numpy as np
from .models import MatchResult, Rect, SearchSettings, TrainModel

def iou(a: Rect, b: Rect) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x+a.width, b.x+b.width), min(a.y+a.height, b.y+b.height)
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = a.area + b.area - intersection
    return intersection / union if union else 0.0

def non_max_suppression(candidates: list[MatchResult], threshold: float, limit: int) -> list[MatchResult]:
    kept: list[MatchResult] = []
    for candidate in sorted(candidates, key=lambda r: r.score, reverse=True):
        if all(iou(candidate.box, item.box) <= threshold for item in kept): kept.append(candidate)
        if len(kept) == limit: break
    return kept

def _local_peaks(score_map: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    dilated = cv2.dilate(score_map, np.ones((3, 3), np.uint8))
    ys, xs = np.where((score_map >= threshold) & (score_map == dilated))
    return [(int(x), int(y), float(score_map[y, x])) for y, x in zip(ys, xs)]

def match(model: TrainModel, search_image: np.ndarray, settings: SearchSettings) -> list[MatchResult]:
    started = perf_counter(); candidates = []
    search = search_image if settings.color_mode == "color" else cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    template0 = model.color if settings.color_mode == "color" else model.gray
    scales = np.arange(settings.min_scale, settings.max_scale + settings.scale_step / 2, settings.scale_step)
    for scale in scales:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        template = cv2.resize(template0, None, fx=float(scale), fy=float(scale), interpolation=interpolation)
        th, tw = template.shape[:2]
        if th > search.shape[0] or tw > search.shape[1]: continue
        score_map = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        for x, y, score in _local_peaks(score_map, settings.threshold):
            candidates.append(MatchResult(max(0.0, min(1.0, score)), Rect(x, y, tw, th), float(scale)))
    if not candidates: return []
    elapsed = (perf_counter() - started) * 1000
    return [replace(r, elapsed_ms=elapsed) for r in non_max_suppression(candidates, settings.nms_iou_threshold, settings.max_results)]
```

- [ ] **Step 4: 경계·NMS 테스트 추가 후 전체 Matcher 테스트 실행**

Add assertions for `iou(Rect(0,0,10,10), Rect(5,5,10,10)) == pytest.approx(25/175)`, no valid scale returning `[]`, gray mode, and `max_results=1`. Run: `python -m pytest tests/test_matcher.py -v`
Expected: all tests PASS

- [ ] **Step 5: 커밋**

```bash
git add src/searchmax/matcher.py tests/test_matcher.py
git commit -m "feat: implement multiscale normalized correlation search"
```

### Task 4: 테스트 생성과 정답 메타데이터

**Files:**
- Create: `src/searchmax/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: `TrainModel`, `Rect`, `TransformRecord`, `GeneratedSample`, `write_image`
- Produces: `GenerationSettings`, `generate_samples(model, backgrounds, output_dir, settings) -> list[GeneratedSample]`

- [ ] **Step 1: 같은 시드 재현성 테스트 작성**

```python
# tests/test_generator.py
from pathlib import Path
import numpy as np
from searchmax.generator import GenerationSettings, generate_samples
from searchmax.models import Rect
from searchmax.training import train_from_roi

def test_generation_is_reproducible(tmp_path):
    template = np.zeros((20, 30, 3), np.uint8); template[:, 5:25] = (20, 140, 250)
    model = train_from_roi(template, Path("t.png"), Rect(0, 0, 30, 20))
    settings = GenerationSettings(count=2, seed=42, output_size=(320, 200))
    a = generate_samples(model, [], tmp_path / "a", settings)
    b = generate_samples(model, [], tmp_path / "b", settings)
    assert [(x.truth_box, x.transform) for x in a] == [(x.truth_box, x.transform) for x in b]
    assert all(x.image_path.exists() for x in a)
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_generator.py -v`
Expected: FAIL because `searchmax.generator` does not exist

- [ ] **Step 3: 생성기 구현**

Implement `GenerationSettings(count=20, seed=1234, output_size=(1280,720), min_scale=.8, max_scale=1.5, brightness_range=(-12,12), contrast_range=(.9,1.1), blur_choices=(0,3), noise_sigma_range=(0,3))` with validation. Use one `np.random.default_rng(seed)`, choose/resize a background or create a neutral background with simple rectangles, resize the template, choose an in-bounds `(x,y)`, paste it, then apply `cv2.convertScaleAbs`, optional Gaussian blur, and seeded Gaussian noise to the full image. Save `sample_0001.png` and return exact `GeneratedSample` records.

- [ ] **Step 4: 재현성과 경계 테스트 실행**

Add assertions that every box lies inside `output_size`, scale stays in `[.8, 1.5]`, and invalid output size raises `ValueError`. Run: `python -m pytest tests/test_generator.py -v`
Expected: all tests PASS

- [ ] **Step 5: 커밋**

```bash
git add src/searchmax/generator.py tests/test_generator.py
git commit -m "feat: generate reproducible transformed UI samples"
```

### Task 5: 평가와 영속화

**Files:**
- Create: `src/searchmax/evaluator.py`
- Create: `src/searchmax/persistence.py`
- Test: `tests/test_evaluator.py`
- Test: `tests/test_persistence.py`

**Interfaces:**
- Consumes: 모든 도메인 모델, `matcher.iou`
- Produces: `evaluate_sample(sample, matches, trained_size, iou_threshold=.5) -> EvaluationRecord`, `summarize(records) -> EvaluationSummary`, `save_project`, `load_project`, `save_samples`, `load_samples`, `export_results_csv`

- [ ] **Step 1: 성공 판정과 요약 테스트 작성**

```python
# tests/test_evaluator.py
from pathlib import Path
from searchmax.evaluator import evaluate_sample, summarize
from searchmax.models import GeneratedSample, MatchResult, Rect, TransformRecord

def test_evaluation_uses_best_iou_and_summarizes():
    sample = GeneratedSample(Path("x.png"), Rect(50, 40, 100, 60), TransformRecord(1,0,1,0,0), 1)
    matches = [MatchResult(.92, Rect(52, 41, 98, 59), .99, 12.0)]
    record = evaluate_sample(sample, matches, (100, 60), .5)
    summary = summarize([record])
    assert record.success
    assert record.iou > .9
    assert summary.success_rate == 1.0
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/test_evaluator.py tests/test_persistence.py -v`
Expected: FAIL because evaluator and persistence modules do not exist

- [ ] **Step 3: 평가 구현**

Implement best-IoU selection, Euclidean center error, `(detected_scale / truth_scale - 1) * 100`, no-match values, and arithmetic means with empty-list values set to `0.0`. A result succeeds only when its IoU is at least the passed `iou_threshold` (default `.5`); the Matcher already enforces score threshold.

- [ ] **Step 4: JSON/CSV 저장 구현**

Implement explicit dict conversion rather than pickling NumPy arrays. Store project schema version `1`, Train source/ROI, `SearchSettings`, generation settings, and relative paths when the target lies below the project directory. JSON uses UTF-8 with `ensure_ascii=False, indent=2`; CSV uses `utf-8-sig` and columns `image,success,score,iou,center_error,scale_error_percent,elapsed_ms`.

- [ ] **Step 5: 왕복 테스트와 전체 테스트 실행**

Test JSON save/load equality, Korean paths, malformed schema rejection, and CSV header/row values. Run: `python -m pytest tests/test_evaluator.py tests/test_persistence.py -v`
Expected: all tests PASS

- [ ] **Step 6: 커밋**

```bash
git add src/searchmax/evaluator.py src/searchmax/persistence.py tests/test_evaluator.py tests/test_persistence.py
git commit -m "feat: evaluate detections and persist projects"
```

### Task 6: ROI와 오버레이 이미지 뷰

**Files:**
- Create: `src/searchmax/ui/__init__.py`
- Create: `src/searchmax/ui/image_view.py`
- Test: `tests/ui/test_image_view.py`

**Interfaces:**
- Consumes: `Rect`, `MatchResult`
- Produces: `ImageView.set_image(np.ndarray)`, `set_roi_enabled(bool)`, `selected_roi() -> Rect | None`, `set_matches(list[MatchResult])`, signal `roi_changed(Rect)`

- [ ] **Step 1: 좌표 변환과 ROI 테스트 작성**

```python
# tests/ui/test_image_view.py
import numpy as np
from PySide6.QtCore import QPointF
from searchmax.ui.image_view import ImageView

def test_image_view_round_trips_scene_coordinates(qtbot):
    view = ImageView(); qtbot.addWidget(view)
    view.resize(640, 480); view.set_image(np.zeros((240, 320, 3), np.uint8)); view.fit_image()
    point = QPointF(120, 80)
    restored = view.scene_to_image(view.image_to_scene(point))
    assert restored.x() == pytest.approx(120, abs=.5)
    assert restored.y() == pytest.approx(80, abs=.5)
```

- [ ] **Step 2: 실패 확인**

Run: `python -m pytest tests/ui/test_image_view.py -v`
Expected: FAIL because `searchmax.ui.image_view` does not exist

- [ ] **Step 3: `QGraphicsView` 구현**

Use `QGraphicsPixmapItem` for the BGR-to-RGB converted image, a normalized `QGraphicsRectItem` for ROI, and separate graphics items for truth and match boxes. Left drag creates ROI when enabled; otherwise left drag pans. Mouse wheel zoom is clamped to `[0.1, 20.0]`. `selected_roi()` clamps integer coordinates to the image bounds. Match labels render `#rank score` and the selected result uses a thicker pen.

- [ ] **Step 4: ROI 클램프와 오버레이 테스트 실행**

Add tests for image-edge ROI clamping and two match items with labels. Run with Windows offscreen fallback: `set QT_QPA_PLATFORM=offscreen && python -m pytest tests/ui/test_image_view.py -v` (PowerShell: `$env:QT_QPA_PLATFORM='offscreen'`).
Expected: all tests PASS

- [ ] **Step 5: 커밋**

```bash
git add src/searchmax/ui tests/ui/test_image_view.py
git commit -m "feat: add ROI and match image viewer"
```

### Task 7: 메인 창과 백그라운드 워크플로

**Files:**
- Create: `src/searchmax/ui/workers.py`
- Create: `src/searchmax/ui/main_window.py`
- Create: `src/searchmax/app.py`
- Test: `tests/ui/test_main_window.py`

**Interfaces:**
- Consumes: Task 1~6의 모든 서비스 인터페이스
- Produces: `MainWindow`, CLI entry point `searchmax.app:main`

- [ ] **Step 1: 기본 설정과 결과 표 테스트 작성**

```python
# tests/ui/test_main_window.py
from searchmax.models import MatchResult, Rect
from searchmax.ui.main_window import MainWindow

def test_main_window_defaults_and_result_rows(qtbot):
    window = MainWindow(); qtbot.addWidget(window)
    assert window.min_scale.value() == 80
    assert window.max_scale.value() == 150
    assert window.max_results.value() == 1
    window.show_results([MatchResult(.91, Rect(10, 20, 30, 40), 1.2, 8.5)])
    assert window.results_table.rowCount() == 1
    assert window.results_table.item(0, 1).text() == "0.910"
```

- [ ] **Step 2: 실패 확인**

Run: `set QT_QPA_PLATFORM=offscreen && python -m pytest tests/ui/test_main_window.py -v`
Expected: FAIL because `searchmax.ui.main_window` does not exist

- [ ] **Step 3: 검색 작업자 구현**

Create `SearchWorker(QObject)` with `progress(current,total)`, `item_finished(path,results)`, `failed(path,message)`, `finished()`, and `cancelled()` signals. Its slot loops through immutable job inputs, checks an interruption flag between images, invokes `match`, emits per-file failure without stopping the batch, and never touches widgets.

- [ ] **Step 4: 메인 창 구성과 Train 흐름 구현**

Build a `QMainWindow` using a top `QSplitter` for Train, Test/Generator, and Search Settings groups; put `ImageView` in the center and a tabbed final/diagnostic result table plus summary below. Wire file dialogs to `read_image`, ROI Train to `train_from_roi`, direct Train to `train_from_file`, and validate that a Train model and test image exist before enabling Run.

- [ ] **Step 5: 검색·생성·평가 흐름 연결**

Move each worker to `QThread`, disable conflicting actions while running, show progress and Cancel, and always call `thread.quit()` plus `deleteLater()` on completion. Clicking a result row calls `ImageView.highlight_match(row)`. Final table columns are rank, score, X, Y, width, height, scale, elapsed ms; diagnostic candidates are shown only when the option is enabled. Wire project save/load and CSV export to persistence functions.

- [ ] **Step 6: 앱 진입점 구현**

```python
# src/searchmax/app.py
import sys
from PySide6.QtWidgets import QApplication
from .ui.main_window import MainWindow

def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("OpenCV SearchMax")
    window = MainWindow(); window.resize(1500, 900); window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: UI 테스트 실행**

Add tests that invalid Run shows a message, a result-row click highlights its box, `max_results` accepts 1 and 100 but not 0/101, and a mocked worker failure leaves the next input processable. Run: `set QT_QPA_PLATFORM=offscreen && python -m pytest tests/ui -v`
Expected: all UI tests PASS

- [ ] **Step 8: 커밋**

```bash
git add src/searchmax/ui src/searchmax/app.py tests/ui
git commit -m "feat: build PySide6 searchmax workflow"
```

### Task 8: 통합 검증과 Windows 문서

**Files:**
- Create: `tests/test_end_to_end.py`
- Create: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: 완성된 전체 앱
- Produces: 재현 가능한 설치·실행·검증 절차

- [ ] **Step 1: 생성→검색→평가 통합 테스트 작성**

```python
# tests/test_end_to_end.py
from pathlib import Path
import cv2, numpy as np
from searchmax.evaluator import evaluate_sample
from searchmax.generator import GenerationSettings, generate_samples
from searchmax.matcher import match
from searchmax.models import Rect, SearchSettings
from searchmax.training import train_from_roi

def test_generated_clean_samples_are_detected(tmp_path):
    p = np.zeros((28, 44, 3), np.uint8)
    cv2.rectangle(p, (2,2), (41,25), (40,180,240), 2); cv2.putText(p, "OK", (7,20), 0, .5, (255,255,255), 1)
    model = train_from_roi(p, Path("ok.png"), Rect(0,0,44,28))
    samples = generate_samples(model, [], tmp_path, GenerationSettings(count=5, seed=7, brightness_range=(0,0), contrast_range=(1,1), blur_choices=(0,), noise_sigma_range=(0,0)))
    records = []
    for sample in samples:
        image = read_image(sample.image_path)
        records.append(evaluate_sample(sample, match(model, image, SearchSettings(threshold=.75)), (44,28), .5))
    assert all(record.success for record in records)
```

- [ ] **Step 2: 통합 테스트 실패/통과 확인**

Run: `python -m pytest tests/test_end_to_end.py -v`
Expected before final import correction: FAIL with missing `read_image`; add `from searchmax.image_io import read_image`, rerun, Expected: PASS

- [ ] **Step 3: README 작성**

Document Windows commands `py -3.11 -m venv .venv`, `.venv\Scripts\activate`, `python -m pip install -e .`, and `opencv-searchmax`. Include the Train→Test Generate/Load→Search→result table workflow, field defaults, final-vs-raw candidate explanation, project/CSV formats, and explicit limitations (no rotation, perspective, VPP compatibility, or Cognex score equivalence).

- [ ] **Step 4: 전체 자동 검증**

Run: `python -m pytest -v`
Expected: all tests PASS with zero failures

Run: `python -m compileall -q src`
Expected: exit code 0 with no output

- [ ] **Step 5: Windows 수동 스모크 테스트**

Run: `opencv-searchmax`
Expected: app opens; load an image whose path contains Korean text, drag an ROI, Train, generate 20 samples with seed 1234, set maximum results to 3, run all, select each result row, export CSV, save and reopen the project. Confirm no clipped controls at Windows display scales 100%, 125%, and 150%.

- [ ] **Step 6: 최종 커밋**

```bash
git add README.md tests/test_end_to_end.py .gitignore
git commit -m "docs: add Windows workflow and end-to-end coverage"
```
