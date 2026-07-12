# ROI Saturation Variation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate ROI objects from grayscale through original saturation while preserving backgrounds and seeded reproducibility.

**Architecture:** Extend the existing ROI HSV transform with a saturation multiplier, then thread the setting and applied value through metadata persistence and UI controls. Legacy serialized files use neutral scale `1.0`.

**Tech Stack:** Python 3.11+, NumPy, OpenCV, PySide6, pytest

## Global Constraints

- Default saturation scale range is `(0.0, 1.0)`.
- Only ROI pixels are transformed.
- Persisted legacy data defaults to neutral saturation scale `1.0`.
- Seeded generation remains deterministic.

---

### Task 1: Generator saturation transform

**Files:** `src/searchmax/generator.py`, `src/searchmax/models.py`, `tests/test_generator.py`

- [ ] Add failing tests for invalid ranges, grayscale at `0.0`, preservation at `1.0`, background stability, and recorded scale.
- [ ] Run focused tests and confirm failures reference missing saturation fields.
- [ ] Add `GenerationSettings.saturation_scale_range`, `TransformRecord.saturation_scale`, validation, seeded sampling, and HSV Saturation scaling.
- [ ] Run `python -m pytest tests/test_generator.py -q` and commit `feat: vary generated ROI saturation`.

### Task 2: Persistence compatibility

**Files:** `src/searchmax/persistence.py`, `tests/test_persistence.py`

- [ ] Add failing round-trip and legacy default tests.
- [ ] Serialize both saturation fields and load them optionally with neutral legacy defaults.
- [ ] Run persistence tests, accepting only the documented Windows CSV separator failure, and commit `feat: persist ROI saturation variation`.

### Task 3: Saturation UI

**Files:** `src/searchmax/ui/main_window.py`, `tests/ui/test_main_window.py`

- [ ] Add failing tests for defaults, settings construction, and restoration.
- [ ] Add `Saturation min` and `Saturation max` controls and thread them through generation settings.
- [ ] Run UI tests and the full suite, accepting only the documented Windows CSV separator failure.
- [ ] Commit `ui: configure generated ROI saturation`.
