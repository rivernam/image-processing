from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMessageBox

from searchmax.generator import GenerationSettings
from searchmax.models import (
    EvaluationRecord,
    GeneratedSample,
    MatchResult,
    MatchResults,
    Rect,
    SearchSettings,
    TrainModel,
    TransformRecord,
)
from searchmax.ui.main_window import MainWindow
from searchmax.ui.workers import GenerationWorker, SearchWorker
from searchmax.persistence import load_samples, save_samples
from searchmax.image_io import write_image


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
    worker.item_finished.connect(
        lambda path, pixels, matches: successes.append((path, pixels, matches))
    )
    worker.progress.connect(lambda current, total: progress.append((current, total)))

    worker.run()

    assert failures == [(first, "cannot decode")]
    assert successes[0][0] == second
    assert successes[0][1] is image
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
    worker.item_finished.connect(lambda path, pixels, matches: completed.append(path))
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

    def fake_match(model, image, settings, limit):
        seen_settings.append(settings)
        return candidates[:1], candidates

    monkeypatch.setattr("searchmax.ui.workers.match_with_diagnostics", fake_match)
    worker = SearchWorker(
        (path,), _model(), SearchSettings(max_results=1), diagnostics=True
    )
    final, diagnostic = [], []
    worker.item_finished.connect(lambda value, pixels, results: final.extend(results))
    worker.diagnostic_finished.connect(
        lambda value, results: diagnostic.extend(results)
    )

    worker.run()

    assert seen_settings[0].max_results == 1
    assert final == candidates[:1]
    assert diagnostic == candidates


def test_search_result_callback_renders_worker_pixels_without_rereading(
    qtbot, monkeypatch
):
    window = MainWindow()
    qtbot.addWidget(window)
    image = np.full((8, 9, 3), 17, dtype=np.uint8)
    shown = []
    monkeypatch.setattr(window.image_view, "set_image", shown.append)
    monkeypatch.setattr(
        "searchmax.ui.main_window.read_image",
        lambda path: (_ for _ in ()).throw(AssertionError("GUI re-read")),
    )

    window._search_item_finished(Path("test.png"), image, [])

    assert shown == [image]


def test_workers_snapshot_models_backgrounds_and_settings(monkeypatch, tmp_path):
    color = np.full((4, 5, 3), 3, dtype=np.uint8)
    gray = np.full((4, 5), 4, dtype=np.uint8)
    background = np.full((6, 7, 3), 5, dtype=np.uint8)
    model = TrainModel(color, gray, Path("train.png"), Rect(1, 2, 3, 4))
    search_settings = SearchSettings(threshold=.81)
    generation_settings = GenerationSettings(count=1, seed=9)
    search = SearchWorker((Path("test.png"),), model, search_settings)
    generation = GenerationWorker(
        model, (background,), tmp_path, generation_settings
    )

    color[:] = 30
    gray[:] = 40
    background[:] = 50
    seen = {}
    monkeypatch.setattr(
        "searchmax.ui.workers.read_image",
        lambda path: np.zeros((10, 10, 3), dtype=np.uint8),
    )

    def fake_match(snapshot, image, settings):
        seen["search"] = (snapshot, settings)
        return []

    def fake_generate(snapshot, backgrounds, output_dir, settings):
        seen["generation"] = (snapshot, backgrounds, settings)
        path = output_dir / "sample_0001.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return [
            GeneratedSample(
                path,
                Rect(0, 0, 1, 1),
                TransformRecord(1, 0, 1, 0, 0),
                9,
            )
        ]

    monkeypatch.setattr("searchmax.ui.workers.match", fake_match)
    monkeypatch.setattr("searchmax.ui.workers.generate_samples", fake_generate)
    search.run()
    generation.run()

    search_model, captured_search_settings = seen["search"]
    generation_model, backgrounds, captured_generation_settings = seen["generation"]
    assert np.all(search_model.color == 3) and np.all(search_model.gray == 4)
    assert np.all(generation_model.color == 3) and np.all(backgrounds[0] == 5)
    assert not search_model.color.flags.writeable
    assert not search_model.gray.flags.writeable
    assert not generation_model.color.flags.writeable
    assert not backgrounds[0].flags.writeable
    assert search_model is not model and generation_model is not model
    assert search_model.roi == model.roi and search_model.roi is not model.roi
    assert captured_search_settings is not search_settings
    assert captured_generation_settings is not generation_settings
    assert captured_search_settings == search_settings
    assert captured_generation_settings == generation_settings


def test_search_scale_controls_roundtrip_fractional_percentages(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    loaded = SearchSettings(min_scale=.805, max_scale=1.505, scale_step=.025)

    window._apply_search_settings(loaded)

    assert window.search_settings() == loaded


def test_generation_worker_removes_temporary_trees_on_all_outcomes(
    monkeypatch, tmp_path
):
    calls = 0

    def fake_generate(model, backgrounds, output_dir, settings):
        nonlocal calls
        calls += 1
        nested = output_dir / "nested"
        nested.mkdir(parents=True)
        (nested / "debris.txt").write_text("debris")
        if calls == 1:
            raise ValueError("failure")
        image_path = output_dir / "sample_0001.png"
        image_path.touch()
        if calls == 3:
            worker.cancel()
            raise ValueError("cancelled generation")
        return [
            GeneratedSample(
                image_path,
                Rect(0, 0, 1, 1),
                TransformRecord(1, 0, 1, 0, 0),
                calls,
            )
        ]

    monkeypatch.setattr("searchmax.ui.workers.generate_samples", fake_generate)
    worker = GenerationWorker(_model(), (), tmp_path, GenerationSettings(count=3))
    worker.run()

    assert not list(tmp_path.glob(".searchmax-*"))


def test_terminal_worker_signal_runs_main_window_slot_on_gui_thread(qtbot):
    class Emitter(QObject):
        terminal = Signal()

        @Slot()
        def emit_terminal(self):
            self.terminal.emit()

    window = MainWindow()
    qtbot.addWidget(window)
    receiver_threads = []
    window._finish_search = lambda cancelled: receiver_threads.append(
        QThread.currentThread()
    )
    emitter = Emitter()
    thread = QThread()
    emitter.moveToThread(thread)
    emitter.terminal.connect(window._search_finished)
    thread.started.connect(emitter.emit_terminal)
    emitter.terminal.connect(thread.quit)
    thread.start()
    qtbot.waitUntil(lambda: bool(receiver_threads))
    thread.wait()

    assert receiver_threads == [window.thread()]


def test_new_model_and_external_tests_invalidate_generated_truth(qtbot, monkeypatch):
    window = MainWindow(); qtbot.addWidget(window)
    sample = GeneratedSample(Path("old.png"), Rect(1, 2, 3, 4), TransformRecord(1, 0, 1, 0, 0), 1)
    window._generated_samples = [sample]; window._samples_by_path = {sample.image_path: sample}
    window._set_model(_model())
    assert not window._generated_samples and not window._samples_by_path
    window._generated_samples = [sample]; window._samples_by_path = {sample.image_path: sample}
    monkeypatch.setattr("searchmax.ui.main_window.QFileDialog.getOpenFileNames", lambda *a, **k: (["external.png"], ""))
    monkeypatch.setattr("searchmax.ui.main_window.read_image", lambda p: np.zeros((5, 5, 3), np.uint8))
    window.load_test_images()
    assert not window._generated_samples and not window._samples_by_path


def test_complete_summary_label_includes_center_and_scale_errors(qtbot):
    window = MainWindow(); qtbot.addWidget(window)
    window._evaluation_records = [EvaluationRecord(Path("x"), True, .9, .8, 2.5, 4.0, 10.0)]
    window._finish_search(False)
    assert "mean center error 2.50 px" in window.summary_label.text()
    assert "mean scale error 4.00%" in window.summary_label.text()


def test_saved_metadata_loads_and_evaluates_after_restart(qtbot, monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    write_image(image_path, np.zeros((30, 40, 3), np.uint8))
    sample = GeneratedSample(image_path, Rect(1, 2, 12, 10), TransformRecord(1, 0, 1, 0, 0), 1)
    metadata = tmp_path / "samples.json"; save_samples(metadata, [sample])
    window = MainWindow(); qtbot.addWidget(window); window._model = _model()
    monkeypatch.setattr("searchmax.ui.main_window.QFileDialog.getOpenFileName", lambda *a, **k: (str(metadata), ""))
    window.load_metadata_dialog()
    window._search_item_finished(image_path, np.zeros((30, 40, 3), np.uint8), [MatchResult(.9, sample.truth_box, 1, 5)])
    assert window._test_paths == (image_path,)
    assert len(window._evaluation_records) == 1 and window._evaluation_records[0].success


def test_successful_generation_persists_version_one_metadata(qtbot, tmp_path):
    window = MainWindow(); qtbot.addWidget(window)
    sample = GeneratedSample(tmp_path / "sample.png", Rect(1, 2, 3, 4), TransformRecord(1, 0, 1, 0, 0), 2)
    window._generated_samples = [sample]
    window._generation_output = tmp_path
    window._finish_generation(False)
    assert load_samples(tmp_path / "samples.json") == [sample]


def test_empty_worker_result_preserves_elapsed_through_evaluation_and_summary(
    qtbot, monkeypatch
):
    path = Path("generated.png")
    sample = GeneratedSample(path, Rect(1, 2, 12, 10), TransformRecord(1, 0, 1, 0, 0), 1)
    window = MainWindow(); qtbot.addWidget(window); window._model = _model()
    window._samples_by_path = {path: sample}
    monkeypatch.setattr("searchmax.ui.workers.read_image", lambda p: np.zeros((30, 40, 3), np.uint8))
    monkeypatch.setattr("searchmax.ui.workers.match", lambda *a: MatchResults(elapsed_ms=42.5))
    worker = SearchWorker((path,), window._model, SearchSettings())
    worker.item_finished.connect(window._search_item_finished)
    worker.run()
    window._finish_search(False)
    assert window._evaluation_records[0].elapsed_ms == 42.5
    assert "42.50 ms" in window.summary_label.text()


def test_generation_start_clears_prior_evaluation_summary_and_export(qtbot, monkeypatch, tmp_path):
    window = MainWindow(); qtbot.addWidget(window); window._model = _model()
    window._evaluation_records = [EvaluationRecord(Path("old"), True, .9, 1, 0, 0, 1)]
    window.summary_label.setText("old summary"); window._update_actions()
    monkeypatch.setattr("searchmax.ui.main_window.QFileDialog.getExistingDirectory", lambda *a: str(tmp_path))
    monkeypatch.setattr("searchmax.ui.main_window.QThread.start", lambda self: None)
    window.generate_samples()
    assert window._evaluation_records == []
    assert window.summary_label.text() == "No evaluation results"
    assert not window.export_csv_action.isEnabled()


def test_metadata_replacement_clears_evaluation_without_discarding_new_truth(qtbot, monkeypatch, tmp_path):
    image = tmp_path / "new.png"; write_image(image, np.zeros((20, 20, 3), np.uint8))
    sample = GeneratedSample(image, Rect(1, 2, 3, 4), TransformRecord(1, 0, 1, 0, 0), 1)
    metadata = tmp_path / "samples.json"; save_samples(metadata, [sample])
    window = MainWindow(); qtbot.addWidget(window); window._evaluation_records = [EvaluationRecord(Path("old"), True, .9, 1, 0, 0, 1)]
    window.summary_label.setText("old summary")
    monkeypatch.setattr("searchmax.ui.main_window.QFileDialog.getOpenFileName", lambda *a, **k: (str(metadata), ""))
    window.load_metadata_dialog()
    assert window._evaluation_records == []
    assert window.summary_label.text() == "No evaluation results"
    assert window._samples_by_path == {image: sample}
    assert not window.export_csv_action.isEnabled()
