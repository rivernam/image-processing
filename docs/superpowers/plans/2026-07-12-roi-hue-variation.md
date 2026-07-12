# ROI Hue Variation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add seeded medium-strength hue variation to only the ROI object in generated samples.

**Architecture:** Extend generation settings and transform metadata, apply Hue rotation to the resized template before compositing, then thread the new fields through persistence and the existing generation UI. Legacy serialized data receives zero-shift compatibility defaults.

**Tech Stack:** Python 3.11+, NumPy, OpenCV, PySide6, pytest

## Global Constraints

- Default hue range is exactly `(-60.0, 60.0)` degrees.
- Only ROI pixels receive Hue rotation; background pixels do not.
- Values must remain within `[-180.0, 180.0]`.
- Old project and sample metadata without Hue fields must load with zero shift.
- Seeded generation remains reproducible.

---

### Task 1: Generator and transform metadata

**Files:**
- Modify: `src/searchmax/generator.py`
- Modify: `src/searchmax/models.py`
- Modify: `tests/test_generator.py`

**Interfaces:**
- Produces: `GenerationSettings.hue_shift_range: tuple[float, float]`
- Produces: `TransformRecord.hue_shift_degrees: float`

- [ ] Add failing tests for range validation, zero-shift preservation, ROI-only color change, and seeded transform reproducibility.
- [ ] Run `python -m pytest tests/test_generator.py -v` and confirm failures reference missing Hue fields.
- [ ] Add the two fields, validation, HSV Hue rotation helper, seeded sampling, and pre-composite application.
- [ ] Run `python -m pytest tests/test_generator.py -v` and confirm all tests pass.
- [ ] Commit with `feat: vary generated ROI hue`.

### Task 2: Persistence and legacy compatibility

**Files:**
- Modify: `src/searchmax/persistence.py`
- Modify: `tests/test_persistence.py`

**Interfaces:**
- Consumes: `GenerationSettings.hue_shift_range`
- Consumes: `TransformRecord.hue_shift_degrees`

- [ ] Add failing round-trip tests for both fields and legacy-load tests expecting `(0.0, 0.0)` and `0.0`.
- [ ] Run focused persistence tests and confirm failures reference missing serialized Hue fields.
- [ ] Serialize new fields; accept them as optional on load and inject zero defaults when absent.
- [ ] Run `python -m pytest tests/test_persistence.py -v`; expect only the documented Windows CSV separator failure.
- [ ] Commit with `feat: persist generated hue variation`.

### Task 3: Generation UI

**Files:**
- Modify: `src/searchmax/ui/main_window.py`
- Modify: `tests/ui/test_main_window.py`

**Interfaces:**
- Produces: `hue_min: QDoubleSpinBox`, `hue_max: QDoubleSpinBox`
- Consumes and restores: `GenerationSettings.hue_shift_range`

- [ ] Add failing tests for `-60.0`/`60.0` defaults, settings construction, and restoration.
- [ ] Run focused UI tests and confirm failures reference missing Hue controls.
- [ ] Add the two controls to the synthetic panel and thread values through `generation_settings` and `_apply_generation_settings`.
- [ ] Run `python -m pytest tests/ui/test_main_window.py -q` and confirm all tests pass.
- [ ] Run `python -m pytest -q`; expect no new failures beyond the documented Windows CSV separator baseline.
- [ ] Commit with `ui: configure generated ROI hue`.
