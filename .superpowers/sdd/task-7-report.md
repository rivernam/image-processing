# Task 7 Report: 메인 창과 백그라운드 워크플로

## Status

완료. `MainWindow`, 검색/생성 `QObject` 작업자, 애플리케이션 진입점과 focused UI 테스트를 구현했다.

## Implementation

- 상단 `QSplitter`에 Train, Test/Generator, Search Settings를 두고 중앙 `ImageView`, 하단 final/diagnostic tab, 평가 summary/progress/cancel 영역을 구성했다.
- 검색 기본값은 scale 80~150%, step 2%, threshold 0.80, max results 1~100(기본 1), NMS 0.30이며 color/gray mode를 제공한다.
- ROI/file Train, 복수 test/background loading, sample generation, batch search, generated sample evaluation을 서비스 인터페이스에 연결했다.
- final 표는 rank, score, X/Y/width/height, scale, elapsed ms를 표시하고 선택 행을 `ImageView.highlight_match`로 연결했다.
- diagnostic option 사용 시 worker가 최대 100개의 post-NMS 후보를 별도 signal로 전달하고 final 표는 설정된 `max_results`로 제한한다.
- `SearchWorker`는 immutable path tuple을 순회하며 파일 사이 cancellation을 확인하고, decode/match 실패를 파일별로 emit한 뒤 다음 파일을 처리한다.
- `GenerationWorker`는 sample별 독립 service call과 임시 경로를 사용해 파일 사이 cancellation 및 per-file failure continuation을 제공한다.
- 모든 작업은 `QThread`로 이동하고 completion/cancellation에서 `quit`/`deleteLater`를 호출한다. 종료 전 window close는 취소 요청 후 thread 종료까지 defer한다.
- 실행 중 충돌 가능한 UI/file action을 비활성화하고, 완료 후 thread reference를 정리한 다음 action 상태를 복원한다.
- 오류는 기존 결과/model/test state를 먼저 지우지 않으며, batch 완료 status에도 failure count와 마지막 concrete error를 유지한다.
- project save/load와 evaluation CSV export를 persistence 서비스에 연결했고, search와 별개인 generation 설정도 load/save 사이 보존한다.
- `searchmax.app:main`은 application name, 1500×900 기본 창과 event loop를 제공한다.

## TDD Evidence

- 최초 RED: `ModuleNotFoundError: No module named 'searchmax.ui.main_window'`.
- worker continuation/cancellation, invalid Run, row highlight, max-results 범위의 최초 GREEN: 6 passed.
- self-review RED/GREEN:
  - thread 종료 후 Run/Generate action이 복구되지 않음 (`run_button.isEnabled() == False`).
  - generation이 service를 한 번만 호출해 개별 실패 뒤 다음 sample을 처리하지 못함 (`len(calls) == 1`).
  - concrete worker error가 `Search complete`로 덮임.
  - busy 상태에서 background/project/CSV action이 활성 상태로 남음.
  - active thread 중 close event가 accept되어 `QThread` 조기 파괴 가능.
  - load한 generation 범위가 search 범위로 덮임.
  - diagnostic 요청 인자/signal이 없어 후보를 별도 제공하지 못함.
- 각 회귀는 실패를 확인한 뒤 최소 구현으로 GREEN을 확인했다.

## Tests

Fresh verification:

- `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ui -v`: 25 passed.
- `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -v`: 133 passed.
- `.venv/bin/python -m compileall -q src/searchmax tests/ui/test_main_window.py`: exit 0.
- `git diff --check`: exit 0.

## Files

- `src/searchmax/ui/workers.py`
- `src/searchmax/ui/main_window.py`
- `src/searchmax/app.py`
- `tests/ui/test_main_window.py`

## Concerns

- Matcher의 public API는 raw pre-NMS score-map candidate를 노출하지 않는다. 따라서 diagnostic 표는 public `match` API가 제공할 수 있는 최대 100개의 post-NMS 후보를 표시한다.
- 기존 untracked `src/opencv_searchmax.egg-info/`는 사용자 작업으로 간주해 수정하거나 commit하지 않았다.

## Review Fix Wave

리뷰에서 지적된 thread affinity, GUI 재-decode, worker input snapshot, scale 정밀도,
generation 임시 파일 정리를 회귀 테스트로 재현한 뒤 수정했다.

### Fix RED (exact output)

Command: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ui/test_main_window.py -v`

```text
collected 18 items

tests/ui/test_main_window.py ....FEFE......FEFFFFFE                      [100%]

FAILED tests/ui/test_main_window.py::test_search_worker_failure_does_not_stop_later_input
FAILED tests/ui/test_main_window.py::test_search_worker_cancellation_is_checked_between_files
FAILED tests/ui/test_main_window.py::test_search_worker_emits_extra_diagnostics_only_when_requested
FAILED tests/ui/test_main_window.py::test_search_result_callback_renders_worker_pixels_without_rereading
FAILED tests/ui/test_main_window.py::test_workers_snapshot_models_backgrounds_and_settings
FAILED tests/ui/test_main_window.py::test_search_scale_controls_roundtrip_fractional_percentages
FAILED tests/ui/test_main_window.py::test_generation_worker_removes_temporary_trees_on_all_outcomes
FAILED tests/ui/test_main_window.py::test_terminal_worker_signal_runs_main_window_slot_on_gui_thread
ERROR tests/ui/test_main_window.py::test_search_worker_failure_does_not_stop_later_input
ERROR tests/ui/test_main_window.py::test_search_worker_cancellation_is_checked_between_files
ERROR tests/ui/test_main_window.py::test_search_worker_emits_extra_diagnostics_only_when_requested
ERROR tests/ui/test_main_window.py::test_terminal_worker_signal_runs_main_window_slot_on_gui_thread
==================== 8 failed, 10 passed, 4 errors in 0.28s ====================
```

The three signal-arity teardown errors were expected RED fallout from specifying the new
decoded-image payload. The terminal receiver failure recorded the worker `QThread`, proving
the anonymous/non-QObject dispatch problem before the explicit slot implementation.

### Fix GREEN (exact outputs)

Command: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ui/test_main_window.py -v`

```text
collected 18 items

tests/ui/test_main_window.py ..................                          [100%]

============================== 18 passed in 0.25s ==============================
```

Command: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ui -v`

```text
collected 30 items

tests/ui/test_image_view.py ............                                 [ 40%]
tests/ui/test_main_window.py ..................                          [100%]

============================== 30 passed in 0.25s ==============================
```

Command: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -v`

```text
collected 138 items

tests/test_evaluator.py ..................                               [ 13%]
tests/test_generator.py ............................                     [ 33%]
tests/test_matcher.py .............                                      [ 42%]
tests/test_models.py .......                                             [ 47%]
tests/test_persistence.py .......................................        [ 76%]
tests/test_training.py ...                                               [ 78%]
tests/ui/test_image_view.py ............                                 [ 86%]
tests/ui/test_main_window.py ..................                          [100%]

============================= 138 passed in 0.39s ==============================
```
