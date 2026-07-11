from pathlib import Path

import numpy as np
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMessageBox

from searchmax.generator import GenerationSettings
from searchmax.models import (
    GeneratedSample,
    MatchResult,
    Rect,
    SearchSettings,
    TrainModel,
    TransformRecord,
)
from searchmax.ui.main_window import MainWindow
from searchmax.ui.workers import GenerationWorker, SearchWorker


def _model() -> TrainModel:
    color = np.zeros((10, 12, 3), dtype=np.uint8)
    return TrainModel(color, np.zeros((10, 12), dtype=np.uint8), Path("train.png"), None)


def test_main_window_defaults_and_result_rows(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window.min_scale.value() == 80
    assert window.max_scale.value() == 150
    assert window.scale_step.value() == 2
    assert window.threshold.value() == 0.80
    assert window.max_results.value() == 1
    assert window.nms_threshold.value() == 0.30
    window.show_results([MatchResult(.91, Rect(10, 20, 30, 40), 1.2, 8.5)])

    assert window.results_table.rowCount() == 1
    assert window.results_table.item(0, 1).text() == "0.910"
    assert window.results_table.item(0, 2).text() == "10"
    assert window.results_table.item(0, 7).text() == "8.50"


def test_invalid_run_shows_concrete_message_without_clearing_results(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show_results([MatchResult(.91, Rect(10, 20, 30, 40), 1.2)])
    messages = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda parent, title, message: messages.append((title, message)),
    )

    window.run_search()

    assert messages == [("Cannot Run Search", "Train a template before running search.")]
    assert window.results_table.rowCount() == 1


def test_result_row_selection_highlights_corresponding_box(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    matches = [
        MatchResult(.91, Rect(10, 20, 30, 40), 1.2),
        MatchResult(.85, Rect(50, 60, 20, 25), .9),
    ]
    window.show_results(matches)
    highlighted = []
    monkeypatch.setattr(window.image_view, "highlight_match", highlighted.append)

    window.results_table.selectRow(1)

    assert highlighted[-1] == 1


def test_max_results_enforces_documented_range(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert (window.max_results.minimum(), window.max_results.maximum()) == (1, 100)
    window.max_results.setValue(0)
    assert window.max_results.value() == 1
    window.max_results.setValue(101)
    assert window.max_results.value() == 100


def test_search_worker_failure_does_not_stop_later_input(monkeypatch, qtbot):
    first, second = Path("broken.png"), Path("valid.png")
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    processed = []

    def fake_read(path):
        if path == first:
            raise ValueError("cannot decode")
        return image

    def fake_match(model, value, settings):
        processed.append(value)
        return [MatchResult(.9, Rect(1, 2, 12, 10), 1.0)]

    monkeypatch.setattr("searchmax.ui.workers.read_image", fake_read)
    monkeypatch.setattr("searchmax.ui.workers.match", fake_match)
    worker = SearchWorker((first, second), _model(), SearchSettings())
    failures, successes, progress = [], [], []
    worker.failed.connect(lambda path, message: failures.append((path, message)))
    worker.item_finished.connect(lambda path, matches: successes.append((path, matches)))
    worker.progress.connect(lambda current, total: progress.append((current, total)))

    worker.run()

    assert failures == [(first, "cannot decode")]
    assert successes[0][0] == second
    assert len(processed) == 1
    assert progress == [(1, 2), (2, 2)]


def test_search_worker_cancellation_is_checked_between_files(monkeypatch, qtbot):
    paths = (Path("one.png"), Path("two.png"))
    worker = SearchWorker(paths, _model(), SearchSettings())
    completed, cancelled = [], []
    monkeypatch.setattr(
        "searchmax.ui.workers.read_image",
        lambda path: np.zeros((20, 20, 3), dtype=np.uint8),
    )

    def fake_match(model, image, settings):
        worker.cancel()
        return []

    monkeypatch.setattr("searchmax.ui.workers.match", fake_match)
    worker.item_finished.connect(lambda path, matches: completed.append(path))
    worker.cancelled.connect(lambda: cancelled.append(True))

    worker.run()

    assert completed == [paths[0]]
    assert cancelled == [True]


def test_search_thread_exit_restores_actions(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._model = _model()
    window._test_paths = (Path("test.png"),)
    window._search_thread = object()
    window._set_busy(True, 1)

    window._finish_search(False)
    window._clear_search_thread()

    assert window.run_button.isEnabled()
    assert window.generate_button.isEnabled()


def test_generation_worker_continues_after_individual_failure(monkeypatch, tmp_path):
    calls = []

    def fake_generate(model, backgrounds, output_dir, settings):
        calls.append((output_dir, settings))
        if len(calls) == 1:
            raise ValueError("disk error")
        path = output_dir / "sample_0001.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return [
            GeneratedSample(
                path,
                Rect(1, 2, 12, 10),
                TransformRecord(1.0, 0.0, 1.0, 0, 0.0),
                settings.seed,
            )
        ]

    monkeypatch.setattr("searchmax.ui.workers.generate_samples", fake_generate)
    worker = GenerationWorker(
        _model(), (), tmp_path, GenerationSettings(count=2, seed=50)
    )
    failures, samples, progress = [], [], []
    worker.failed.connect(lambda path, message: failures.append((path, message)))
    worker.item_finished.connect(samples.append)
    worker.progress.connect(lambda current, total: progress.append((current, total)))

    worker.run()

    assert len(calls) == 2
    assert [call[1].count for call in calls] == [1, 1]
    assert [call[1].seed for call in calls] == [50, 51]
    assert failures[0][1] == "disk error"
    assert samples[0].image_path == tmp_path / "sample_0002.png"
    assert progress == [(1, 2), (2, 2)]


def test_batch_completion_keeps_concrete_failure_status(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._worker_failed(Path("bad.png"), "cannot decode")
    window._finish_search(False)

    assert "1 failure" in window.statusBar().currentMessage()
    assert "bad.png: cannot decode" in window.statusBar().currentMessage()


def test_busy_state_disables_conflicting_file_and_background_actions(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._set_busy(True, 3)

    assert not window.load_background_button.isEnabled()
    assert not window.save_project_action.isEnabled()
    assert not window.load_project_action.isEnabled()
    assert not window.export_csv_action.isEnabled()
    assert window.cancel_button.isEnabled()


def test_close_is_deferred_until_active_thread_finishes(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window._search_thread = object()
    cancellations = []
    monkeypatch.setattr(window, "cancel_work", lambda: cancellations.append(True))
    event = QCloseEvent()

    window.closeEvent(event)

    assert not event.isAccepted()
    assert cancellations == [True]
    assert window._close_requested


def test_generation_settings_preserve_loaded_non_search_ranges(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    loaded = GenerationSettings(
        count=4,
        seed=77,
        output_size=(640, 360),
        min_scale=.65,
        max_scale=1.85,
        brightness_range=(-5, 8),
        contrast_range=(.8, 1.2),
        blur_choices=(0, 3, 5),
        noise_sigma_range=(0, 2),
    )

    window._apply_generation_settings(loaded)

    assert window.generation_settings() == loaded


def test_search_worker_emits_extra_diagnostics_only_when_requested(monkeypatch):
    path = Path("test.png")
    candidates = [
        MatchResult(.95, Rect(1, 2, 12, 10), 1.0),
        MatchResult(.85, Rect(20, 22, 12, 10), 1.0),
    ]
    seen_settings = []
    monkeypatch.setattr(
        "searchmax.ui.workers.read_image",
        lambda value: np.zeros((30, 40, 3), dtype=np.uint8),
    )

    def fake_match(model, image, settings):
        seen_settings.append(settings)
        return candidates

    monkeypatch.setattr("searchmax.ui.workers.match", fake_match)
    worker = SearchWorker(
        (path,), _model(), SearchSettings(max_results=1), diagnostics=True
    )
    final, diagnostic = [], []
    worker.item_finished.connect(lambda value, results: final.extend(results))
    worker.diagnostic_finished.connect(
        lambda value, results: diagnostic.extend(results)
    )

    worker.run()

    assert seen_settings[0].max_results == 100
    assert final == candidates[:1]
    assert diagnostic == candidates
