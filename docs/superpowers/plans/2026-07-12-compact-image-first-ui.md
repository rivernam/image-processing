# Compact Image-First UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the image workspace dominant by moving controls to a compact sidebar and making results and diagnostic settings collapsible.

**Architecture:** Recompose `MainWindow._build_ui` around a left/right splitter without changing existing action widgets or signal flow. Add presentation-only toggles for results and advanced search controls.

**Tech Stack:** Python 3.11+, PySide6, pytest, pytest-qt

## Global Constraints

- Preserve all training, generation, search, persistence, and worker behavior.
- Preserve existing widget attributes and signal connections.
- Results and advanced diagnostic controls start collapsed.
- Resizing primarily benefits the image workspace.

---

### Task 1: Image-first main-window layout

**Files:**
- Modify: `src/searchmax/ui/main_window.py`
- Modify: `tests/ui/test_main_window.py`

**Interfaces:**
- Produces: `main_splitter`, `control_panel`, `workspace_panel`, `results_toggle`, `advanced_search_toggle`, `advanced_search_panel`
- Produces: `_toggle_results(bool)` and `_toggle_advanced_search(bool)` presentation slots

- [ ] Add failing tests asserting splitter order/stretch, hidden results, toggle behavior without row clearing, and hidden advanced diagnostics with revised copy.
- [ ] Run the focused tests and confirm they fail because the new layout attributes do not exist.
- [ ] Rebuild `_build_ui` with a 320px left control pane and stretching right image workspace.
- [ ] Add the result toggle and advanced-search toggle, preserving existing result tabs and diagnostic checkbox behavior.
- [ ] Run `python -m pytest tests/ui/test_main_window.py -q` and confirm all tests pass.
- [ ] Run `python -m pytest -q` and confirm no new failures beyond the documented Windows CSV path-separator baseline.
- [ ] Commit with `ui: prioritize image workspace`.
