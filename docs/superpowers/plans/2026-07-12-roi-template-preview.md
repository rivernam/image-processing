# ROI Template Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent clickable preview of the active ROI template to the Train panel.

**Architecture:** Create a focused QLabel-based preview widget that owns BGR-to-pixmap conversion and click signaling. MainWindow derives preview content from `TrainModel.color` and uses the existing central ImageView for expanded inspection.

**Tech Stack:** Python 3.11+, NumPy, PySide6, pytest-qt

## Global Constraints

- Preview source is exactly `TrainModel.color`.
- Preview preserves aspect ratio and does not upscale.
- Test image display never clears the preview.
- No persistence schema changes.

---

### Task 1: Template preview widget

**Files:** Create `src/searchmax/ui/template_preview.py`; create `tests/ui/test_template_preview.py`

- [ ] Add failing tests for empty state, image storage, aspect-preserving bounded pixmap, and populated click signal.
- [ ] Implement `TemplatePreview` with `set_image`, `clear`, `image`, and `clicked`.
- [ ] Run `python -m pytest tests/ui/test_template_preview.py -q`.

### Task 2: Main-window integration

**Files:** Modify `src/searchmax/ui/main_window.py`; modify `tests/ui/test_main_window.py`

- [ ] Add failing tests that `_set_model` populates the preview, test-image loading preserves it, and clicking previews in the central ImageView.
- [ ] Add the labeled preview to Train, update it in `_set_model`, and connect its click to a central-display slot.
- [ ] Run UI tests and the full suite, accepting only the documented Windows CSV separator failure.
- [ ] Commit `ui: show trained ROI template preview`.
