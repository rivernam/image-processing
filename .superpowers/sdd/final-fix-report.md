# Final Important Fix Wave Report

## Scope and outcome

Implemented every whole-branch Important finding: generated metadata is persisted and reloadable from the UI; projects retain recent test/background paths; diagnostics expose deterministic bounded pre-NMS candidates; generated truth is invalidated on retrain/external test load; empty searches retain actual elapsed time; and the summary includes center/scale errors. JSON and CSV writes now use same-directory temporary files plus atomic replacement, and the unused `models.field` import was removed.

## TDD evidence

RED command:

`QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_matcher.py tests/test_persistence.py tests/ui/test_main_window.py -q`

Initial result: collection failed with `ImportError: cannot import name 'diagnostic_candidates'`, proving the missing matcher diagnostic API. After the first implementation, the focused run reported two compatibility failures: elapsed clock ordering was `clock, clock, nms` instead of `clock, nms, clock`, and the old worker diagnostic test still expected post-NMS expansion through `match(max_results=100)`. The implementation was corrected to measure through final selection and the worker test was updated to assert the new pre-NMS interface while retaining configured `max_results`.

GREEN focused command:

`QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_matcher.py tests/test_evaluator.py tests/test_persistence.py tests/ui/test_main_window.py -q`

Result: 95 passed.

Coverage added:

- generated `samples.json` save, restart metadata load, path/truth restoration, and evaluation
- project recent test/background path roundtrip with portable paths
- truth invalidation after retraining and external test loading
- deterministic bounded pre-NMS diagnostics differing from final NMS/max-results
- measured empty-search elapsed time propagated into failed evaluation
- complete summary label with mean center and scale errors

## Fresh verification

- `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ui -v`: 34 passed.
- `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -v`: 147 passed in 6.92s.
- `.venv/bin/python -m compileall -q src/searchmax tests`: exit 0.
- `git diff --check`: exit 0.

## Files changed

- `README.md`
- `src/searchmax/{models,matcher,evaluator,persistence}.py`
- `src/searchmax/ui/{workers,main_window}.py`
- `tests/test_{matcher,evaluator,persistence}.py`
- `tests/ui/test_main_window.py`

## Concerns

- Metadata schema remains version 1 and therefore carries sample truth/path data rather than an embedded template fingerprint. Safety is enforced in application state: every successful retrain and every external test load clears truth/evaluation mappings, while metadata loading rebuilds mappings only for image files that decode successfully.
- Project loading validates and decodes all restored recent images before committing any UI state; a single missing/invalid project path rejects the load with its concrete path/error. Metadata loading is deliberately per-file: valid samples are restored and concrete failures are reported, while an all-invalid load preserves existing state.

## Final re-review follow-up

Two remaining Important findings were fixed with TDD.

RED command:

`QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ui/test_main_window.py -q`

Result: 3 failed, 22 passed. The failures showed: worker slicing converted empty `MatchResults(elapsed_ms=42.5)` into a plain list and evaluation recorded `0.0`; generation start retained the old evaluation record; metadata replacement retained the old evaluation record.

GREEN focused command:

`QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_matcher.py tests/test_evaluator.py tests/ui/test_main_window.py -q`

Result: 59 passed.

Changes:

- `SearchWorker` now emits the list-compatible matcher outcome unchanged, preserving empty-result elapsed time while `match()` retains configured NMS/max-results. Pre-NMS diagnostics remain a separate signal.
- Generation start and successful metadata replacement clear evaluation records and reset summary/export state. Metadata replacement performs this reset without clearing the newly loaded samples/truth mapping.
- Public diagnostic matcher functions now have concrete argument and return annotations.

Fresh final verification:

- `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ui -v`: 37 passed.
- `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -v`: 150 passed in 6.86s.
- `.venv/bin/python -m compileall -q src/searchmax tests`: exit 0.
- `git diff --check`: exit 0.

No new schema concern was introduced; per review direction, template fingerprints were not added.
