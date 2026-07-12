"""Main SearchMax desktop workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Qt, Slot
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from searchmax.evaluator import evaluate_sample, summarize
from searchmax.generator import GenerationSettings
from searchmax.image_io import read_image
from searchmax.models import (
    EvaluationRecord,
    GeneratedSample,
    MatchResult,
    SearchSettings,
    TrainModel,
)
from searchmax.persistence import export_results_csv, load_project, load_samples, save_project, save_samples
from searchmax.training import train_from_file, train_from_roi
from searchmax.ui.image_view import ImageView
from searchmax.ui.workers import GenerationWorker, SearchWorker


class MainWindow(QMainWindow):
    RESULT_HEADERS = ("Rank", "Score", "X", "Y", "Width", "Height", "Scale", "Elapsed ms")
    DIAGNOSTIC_HEADERS = ("Image",) + RESULT_HEADERS

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("OpenCV SearchMax")
        self._model: TrainModel | None = None
        self._train_image: np.ndarray | None = None
        self._train_source: Path | None = None
        self._test_paths: tuple[Path, ...] = ()
        self._backgrounds: tuple[np.ndarray, ...] = ()
        self._background_paths: tuple[Path, ...] = ()
        self._generated_samples: list[GeneratedSample] = []
        self._samples_by_path: dict[Path, GeneratedSample] = {}
        self._evaluation_records: list[EvaluationRecord] = []
        self._current_results: list[MatchResult] = []
        self._worker_failures: list[str] = []
        self._generation_settings = GenerationSettings()
        self._search_thread: QThread | None = None
        self._search_worker: SearchWorker | None = None
        self._generation_thread: QThread | None = None
        self._generation_worker: GenerationWorker | None = None
        self._close_requested = False
        self._generation_output: Path | None = None

        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._update_actions()
        self.statusBar().showMessage("Ready")

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        controls = QSplitter(Qt.Orientation.Horizontal)
        controls.addWidget(self._build_train_group())
        controls.addWidget(self._build_test_group())
        controls.addWidget(self._build_search_group())
        layout.addWidget(controls)

        self.image_view = ImageView()
        layout.addWidget(self.image_view, 1)

        self.results_table = self._table(self.RESULT_HEADERS)
        self.diagnostics_table = self._table(self.DIAGNOSTIC_HEADERS)
        self.result_tabs = QTabWidget()
        self.result_tabs.addTab(self.results_table, "Final Results")
        self.result_tabs.addTab(self.diagnostics_table, "Diagnostics")
        layout.addWidget(self.result_tabs)

        footer = QHBoxLayout()
        self.summary_label = QLabel("No evaluation results")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        footer.addWidget(self.summary_label, 1)
        footer.addWidget(self.progress_bar)
        footer.addWidget(self.cancel_button)
        layout.addLayout(footer)
        self.setCentralWidget(root)

    def _build_train_group(self) -> QGroupBox:
        group = QGroupBox("Train")
        layout = QVBoxLayout(group)
        self.load_train_image_button = QPushButton("Load Image for ROI")
        self.train_roi_button = QPushButton("Train from ROI")
        self.train_file_button = QPushButton("Train from File")
        self.roi_checkbox = QCheckBox("Select ROI")
        self.train_status = QLabel("No template")
        layout.addWidget(self.load_train_image_button)
        layout.addWidget(self.roi_checkbox)
        layout.addWidget(self.train_roi_button)
        layout.addWidget(self.train_file_button)
        layout.addWidget(self.train_status)
        return group

    def _build_test_group(self) -> QGroupBox:
        group = QGroupBox("Test / Generator")
        layout = QVBoxLayout(group)

        self.existing_images_group = QGroupBox("Existing Images")
        existing_layout = QVBoxLayout(self.existing_images_group)
        self.load_test_button = QPushButton("Open Images to Search")
        self.run_button = QPushButton("Run Search")
        existing_layout.addWidget(self.load_test_button)
        existing_layout.addWidget(self.run_button)

        self.synthetic_images_group = QGroupBox("Synthetic Test Images")
        synthetic_layout = QFormLayout(self.synthetic_images_group)
        self.load_background_button = QPushButton("Add Backgrounds for Generation")
        self.generate_button = QPushButton("Generate Test Images")
        self.generation_count = QSpinBox()
        self.generation_count.setRange(1, 10_000)
        self.generation_count.setValue(20)
        self.generation_seed = QSpinBox()
        self.generation_seed.setRange(0, 2_147_483_647)
        self.generation_seed.setValue(1234)
        synthetic_layout.addRow(self.load_background_button)
        synthetic_layout.addRow("Count", self.generation_count)
        synthetic_layout.addRow("Seed", self.generation_seed)
        synthetic_layout.addRow(self.generate_button)

        layout.addWidget(self.existing_images_group)
        layout.addWidget(self.synthetic_images_group)
        return group

    def _build_search_group(self) -> QGroupBox:
        group = QGroupBox("Search Settings")
        layout = QFormLayout(group)
        self.min_scale = self._percent_spin(1, 1000, 80)
        self.max_scale = self._percent_spin(1, 1000, 150)
        self.scale_step = self._percent_spin(1, 100, 2)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setDecimals(2)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(0.80)
        self.max_results = QSpinBox()
        self.max_results.setRange(1, 100)
        self.max_results.setValue(1)
        self.nms_threshold = QDoubleSpinBox()
        self.nms_threshold.setRange(0.0, 1.0)
        self.nms_threshold.setDecimals(2)
        self.nms_threshold.setSingleStep(0.01)
        self.nms_threshold.setValue(0.30)
        self.color_mode = QComboBox()
        self.color_mode.addItem("Color", "color")
        self.color_mode.addItem("Grayscale", "gray")
        self.show_diagnostics = QCheckBox("Show diagnostic candidates")
        layout.addRow("Min scale (%)", self.min_scale)
        layout.addRow("Max scale (%)", self.max_scale)
        layout.addRow("Step (%)", self.scale_step)
        layout.addRow("Threshold", self.threshold)
        layout.addRow("Maximum results", self.max_results)
        layout.addRow("NMS IoU", self.nms_threshold)
        layout.addRow("Mode", self.color_mode)
        layout.addRow(self.show_diagnostics)
        return group

    @staticmethod
    def _percent_spin(minimum: int, maximum: int, value: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        return spin

    @staticmethod
    def _table(headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        return table

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        self.save_project_action = QAction("Save Project", self)
        self.load_project_action = QAction("Load Project", self)
        self.export_csv_action = QAction("Export Evaluation CSV", self)
        self.load_metadata_action = QAction("Load Sample Metadata", self)
        file_menu.addActions(
            (self.save_project_action, self.load_project_action, self.load_metadata_action, self.export_csv_action)
        )

    def _connect_signals(self) -> None:
        self.load_train_image_button.clicked.connect(self.load_train_image)
        self.roi_checkbox.toggled.connect(self.image_view.set_roi_enabled)
        self.train_roi_button.clicked.connect(self.train_selected_roi)
        self.train_file_button.clicked.connect(self.train_direct_file)
        self.load_test_button.clicked.connect(self.load_test_images)
        self.load_background_button.clicked.connect(self.load_backgrounds)
        self.generate_button.clicked.connect(self.generate_samples)
        self.run_button.clicked.connect(self.run_search)
        self.cancel_button.clicked.connect(self.cancel_work)
        self.results_table.itemSelectionChanged.connect(self._highlight_selected_result)
        self.show_diagnostics.toggled.connect(self._toggle_diagnostics)
        self.save_project_action.triggered.connect(self.save_project_dialog)
        self.load_project_action.triggered.connect(self.load_project_dialog)
        self.load_metadata_action.triggered.connect(self.load_metadata_dialog)
        self.export_csv_action.triggered.connect(self.export_csv_dialog)

    def search_settings(self) -> SearchSettings:
        return SearchSettings(
            min_scale=self.min_scale.value() / 100,
            max_scale=self.max_scale.value() / 100,
            scale_step=self.scale_step.value() / 100,
            threshold=self.threshold.value(),
            max_results=self.max_results.value(),
            nms_iou_threshold=self.nms_threshold.value(),
            color_mode=str(self.color_mode.currentData()),
        )

    def generation_settings(self) -> GenerationSettings:
        previous = self._generation_settings
        return GenerationSettings(
            count=self.generation_count.value(),
            seed=self.generation_seed.value(),
            output_size=previous.output_size,
            min_scale=previous.min_scale,
            max_scale=previous.max_scale,
            brightness_range=previous.brightness_range,
            contrast_range=previous.contrast_range,
            blur_choices=previous.blur_choices,
            noise_sigma_range=previous.noise_sigma_range,
        )

    def show_results(self, results: list[MatchResult]) -> None:
        self._current_results = list(results)
        self.results_table.setRowCount(len(results))
        for row, result in enumerate(results):
            values = self._result_values(row, result)
            for column, value in enumerate(values):
                self.results_table.setItem(row, column, QTableWidgetItem(value))
        self.image_view.set_matches(results)

    @staticmethod
    def _result_values(row: int, result: MatchResult) -> tuple[str, ...]:
        box = result.box
        return (
            str(row + 1),
            f"{result.score:.3f}",
            str(box.x),
            str(box.y),
            str(box.width),
            str(box.height),
            f"{result.scale:.3f}",
            f"{result.elapsed_ms:.2f}",
        )

    def _highlight_selected_result(self) -> None:
        rows = self.results_table.selectionModel().selectedRows()
        self.image_view.highlight_match(rows[0].row() if rows else None)

    def _toggle_diagnostics(self, enabled: bool) -> None:
        if not enabled:
            self.diagnostics_table.setRowCount(0)

    def load_train_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Load Train Image")
        if not filename:
            return
        try:
            image = read_image(Path(filename))
        except Exception as error:
            self._show_error("Load Train Image Failed", error)
            return
        self._train_source = Path(filename)
        self._train_image = image
        self.image_view.set_image(image)
        self.roi_checkbox.setChecked(True)
        self.statusBar().showMessage(f"Loaded train image: {filename}")

    def train_selected_roi(self) -> None:
        if self._train_image is None or self._train_source is None:
            self._warn("Cannot Train", "Load a train image before selecting an ROI.")
            return
        roi = self.image_view.selected_roi()
        if roi is None:
            self._warn("Cannot Train", "Select a non-empty ROI before training.")
            return
        try:
            model = train_from_roi(self._train_image, self._train_source, roi)
        except Exception as error:
            self._show_error("Train Failed", error)
            return
        self._set_model(model)

    def train_direct_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Train from File")
        if not filename:
            return
        try:
            model = train_from_file(Path(filename))
        except Exception as error:
            self._show_error("Train Failed", error)
            return
        self._train_image = model.color.copy()
        self.image_view.set_image(self._train_image)
        self._set_model(model)

    def _set_model(self, model: TrainModel) -> None:
        self._clear_generated_truth()
        self._model = model
        self._train_source = model.source
        self.train_status.setText(f"Trained: {model.source.name}")
        self.statusBar().showMessage(f"Template trained from {model.source}")
        self._update_actions()

    def load_test_images(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(self, "Load Test Images")
        if not filenames:
            return
        paths = tuple(Path(filename) for filename in filenames)
        try:
            image = read_image(paths[0])
        except Exception as error:
            self._show_error("Load Test Image Failed", error)
            return
        self._test_paths = paths
        self._clear_generated_truth()
        self.image_view.set_image(image)
        self.statusBar().showMessage(f"Loaded {len(paths)} test image(s)")
        self._update_actions()

    def load_backgrounds(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(self, "Load Background Images")
        if not filenames:
            return
        images: list[np.ndarray] = []
        valid_paths: list[Path] = []
        failures: list[str] = []
        for filename in filenames:
            try:
                path = Path(filename)
                images.append(read_image(path))
                valid_paths.append(path)
            except Exception as error:
                failures.append(f"{filename}: {error}")
        self._backgrounds = tuple(images)
        self._background_paths = tuple(valid_paths)
        if failures:
            self._warn("Some Backgrounds Failed", "\n".join(failures))
        self.statusBar().showMessage(f"Loaded {len(images)} background(s)")

    def run_search(self) -> None:
        if self._model is None:
            self._warn("Cannot Run Search", "Train a template before running search.")
            return
        if not self._test_paths:
            self._warn("Cannot Run Search", "Load at least one test image before running search.")
            return
        try:
            settings = self.search_settings()
        except ValueError as error:
            self._show_error("Invalid Search Settings", error)
            return
        self._evaluation_records.clear()
        self._worker_failures.clear()
        self.diagnostics_table.setRowCount(0)
        worker = SearchWorker(
            self._test_paths,
            self._model,
            settings,
            diagnostics=self.show_diagnostics.isChecked(),
        )
        thread = QThread(self)
        self._search_worker, self._search_thread = worker, thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._set_progress)
        worker.item_finished.connect(self._search_item_finished)
        worker.diagnostic_finished.connect(self._append_diagnostics)
        worker.failed.connect(self._worker_failed)
        worker.finished.connect(self._search_finished)
        worker.cancelled.connect(self._search_cancelled)
        worker.finished.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_search_thread)
        self._set_busy(True, len(self._test_paths))
        self.statusBar().showMessage("Searching…")
        thread.start()

    @Slot(object, object, object)
    def _search_item_finished(
        self, path: Path, image: np.ndarray, results: list[MatchResult]
    ) -> None:
        self.image_view.set_image(image)
        self.show_results(results)
        sample = self._samples_by_path.get(Path(path))
        if sample is not None and self._model is not None:
            trained_size = (self._model.color.shape[1], self._model.color.shape[0])
            self._evaluation_records.append(evaluate_sample(sample, results, trained_size))

    def _append_diagnostics(self, path: Path, results: list[MatchResult]) -> None:
        for index, result in enumerate(results):
            row = self.diagnostics_table.rowCount()
            self.diagnostics_table.insertRow(row)
            values = (str(path),) + self._result_values(index, result)
            for column, value in enumerate(values):
                self.diagnostics_table.setItem(row, column, QTableWidgetItem(value))

    def _finish_search(self, was_cancelled: bool) -> None:
        self._set_busy(False)
        if self._evaluation_records:
            summary = summarize(self._evaluation_records)
            self.summary_label.setText(
                f"{summary.successes}/{summary.total} succeeded "
                f"({summary.success_rate:.1%}); mean IoU {summary.mean_iou:.3f}; "
                f"mean center error {summary.mean_center_error:.2f} px; "
                f"mean scale error {summary.mean_scale_error_percent:.2f}%; "
                f"mean elapsed {summary.mean_elapsed_ms:.2f} ms"
            )
        self.statusBar().showMessage(self._completion_status("Search", was_cancelled))

    @Slot()
    def _search_finished(self) -> None:
        self._finish_search(False)

    @Slot()
    def _search_cancelled(self) -> None:
        self._finish_search(True)

    def _clear_search_thread(self) -> None:
        self._search_worker = None
        self._search_thread = None
        self._update_actions()
        self._close_if_idle()

    def generate_samples(self) -> None:
        if self._model is None:
            self._warn("Cannot Generate", "Train a template before generating samples.")
            return
        output = QFileDialog.getExistingDirectory(self, "Select Sample Output Directory")
        if not output:
            return
        try:
            settings = self.generation_settings()
        except ValueError as error:
            self._show_error("Invalid Generation Settings", error)
            return
        self._generation_settings = settings
        self._clear_evaluation_state()
        self._worker_failures.clear()
        worker = GenerationWorker(self._model, self._backgrounds, Path(output), settings)
        self._generation_output = Path(output)
        thread = QThread(self)
        self._generation_worker, self._generation_thread = worker, thread
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._set_progress)
        worker.item_finished.connect(self._generation_item_finished)
        worker.failed.connect(self._worker_failed)
        worker.finished.connect(self._generation_finished)
        worker.cancelled.connect(self._generation_cancelled)
        worker.finished.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_generation_thread)
        self._generated_samples.clear()
        self._samples_by_path.clear()
        self._set_busy(True, settings.count)
        self.statusBar().showMessage("Generating…")
        thread.start()

    def _generation_item_finished(self, sample: GeneratedSample) -> None:
        self._generated_samples.append(sample)
        self._samples_by_path[sample.image_path] = sample

    def _finish_generation(self, was_cancelled: bool) -> None:
        self._set_busy(False)
        if self._generated_samples:
            self._test_paths = tuple(sample.image_path for sample in self._generated_samples)
            if not was_cancelled and self._generation_output is not None:
                try:
                    save_samples(self._generation_output / "samples.json", self._generated_samples)
                except Exception as error:
                    self._worker_failed(self._generation_output / "samples.json", str(error))
            self._update_actions()
        if self._worker_failures:
            self.statusBar().showMessage(self._completion_status("Generation", was_cancelled))
        else:
            self.statusBar().showMessage(
                "Generation cancelled"
                if was_cancelled
                else f"Generated {len(self._generated_samples)} sample(s)"
            )

    @Slot()
    def _generation_finished(self) -> None:
        self._finish_generation(False)

    @Slot()
    def _generation_cancelled(self) -> None:
        self._finish_generation(True)

    def _clear_generation_thread(self) -> None:
        self._generation_worker = None
        self._generation_thread = None
        self._update_actions()
        self._close_if_idle()

    def cancel_work(self) -> None:
        if self._search_worker is not None:
            self._search_worker.cancel()
        if self._generation_worker is not None:
            self._generation_worker.cancel()
        self.statusBar().showMessage("Cancellation requested…")

    def _worker_failed(self, path: Path, message: str) -> None:
        detail = f"{path}: {message}"
        self._worker_failures.append(detail)
        self.statusBar().showMessage(f"Failed {detail}")

    def _completion_status(self, operation: str, was_cancelled: bool) -> str:
        outcome = "cancelled" if was_cancelled else "complete"
        if not self._worker_failures:
            return f"{operation} {outcome}"
        count = len(self._worker_failures)
        return (
            f"{operation} {outcome} with {count} failure(s); "
            f"last: {self._worker_failures[-1]}"
        )

    def _set_busy(self, busy: bool, total: int = 1) -> None:
        self.cancel_button.setEnabled(busy)
        self.progress_bar.setRange(0, max(1, total))
        if busy:
            self.progress_bar.setValue(0)
        for action in (
            self.load_train_image_button,
            self.train_roi_button,
            self.train_file_button,
            self.load_test_button,
            self.load_background_button,
            self.generate_button,
            self.run_button,
            self.save_project_action,
            self.load_project_action,
            self.load_metadata_action,
            self.export_csv_action,
        ):
            action.setEnabled(not busy)
        if not busy:
            self._update_actions()

    def _set_progress(self, current: int, total: int) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)

    def _update_actions(self) -> None:
        idle = self._search_thread is None and self._generation_thread is None
        self.run_button.setEnabled(
            idle and self._model is not None and bool(self._test_paths)
        )
        self.generate_button.setEnabled(idle and self._model is not None)
        self.train_roi_button.setEnabled(idle and self._train_image is not None)
        self.export_csv_action.setEnabled(idle and bool(self._evaluation_records))

    def save_project_dialog(self) -> None:
        if self._model is None:
            self._warn("Cannot Save Project", "Train a template before saving a project.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Project", filter="JSON (*.json)"
        )
        if not filename:
            return
        try:
            generation = self.generation_settings()
            save_project(
                Path(filename),
                self._model.source,
                self._model.roi,
                self.search_settings(),
                generation,
                self._test_paths,
                self._background_paths,
            )
        except Exception as error:
            self._show_error("Save Project Failed", error)
            return
        self._generation_settings = generation
        self.statusBar().showMessage(f"Saved project: {filename}")

    def load_project_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Project", filter="JSON (*.json)"
        )
        if not filename:
            return
        try:
            source, roi, search, generation, test_paths, background_paths = load_project(Path(filename), include_recent_paths=True)
            image = read_image(source)
            model = (
                train_from_roi(image, source, roi)
                if roi is not None
                else train_from_file(source)
            )
            test_images = [read_image(item) for item in test_paths]
            background_images = [read_image(item) for item in background_paths]
        except Exception as error:
            self._show_error("Load Project Failed", error)
            return
        self._train_image = image
        self.image_view.set_image(image)
        self._apply_search_settings(search)
        self._apply_generation_settings(generation)
        self._set_model(model)
        self._test_paths = test_paths
        self._background_paths = background_paths
        self._backgrounds = tuple(background_images)
        if test_images:
            self.image_view.set_image(test_images[0])
        self.statusBar().showMessage(f"Loaded project: {filename}")

    def _clear_generated_truth(self) -> None:
        self._generated_samples.clear()
        self._samples_by_path.clear()
        self._clear_evaluation_state()

    def _clear_evaluation_state(self) -> None:
        self._evaluation_records.clear()
        self.summary_label.setText("No evaluation results")
        self._update_actions()

    def load_metadata_dialog(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Load Sample Metadata", filter="JSON (*.json)")
        if not filename:
            return
        try:
            samples = load_samples(Path(filename))
        except Exception as error:
            self._show_error("Load Sample Metadata Failed", error)
            return
        valid, failures = [], []
        for sample in samples:
            try:
                read_image(sample.image_path)
            except Exception as error:
                failures.append(f"{sample.image_path}: {error}")
            else:
                valid.append(sample)
        if not valid:
            self._warn("Load Sample Metadata Failed", "\n".join(failures) or "Metadata contains no samples.")
            return
        self._clear_evaluation_state()
        self._generated_samples = valid
        self._samples_by_path = {sample.image_path: sample for sample in valid}
        self._test_paths = tuple(sample.image_path for sample in valid)
        self._update_actions()
        if failures:
            self._warn("Some Sample Images Failed", "\n".join(failures))
        else:
            self.statusBar().showMessage(f"Loaded {len(valid)} generated sample(s)")

    def _apply_search_settings(self, settings: SearchSettings) -> None:
        self.min_scale.setValue(settings.min_scale * 100)
        self.max_scale.setValue(settings.max_scale * 100)
        self.scale_step.setValue(settings.scale_step * 100)
        self.threshold.setValue(settings.threshold)
        self.max_results.setValue(settings.max_results)
        self.nms_threshold.setValue(settings.nms_iou_threshold)
        index = self.color_mode.findData(settings.color_mode)
        if index >= 0:
            self.color_mode.setCurrentIndex(index)

    def _apply_generation_settings(self, settings: GenerationSettings) -> None:
        self._generation_settings = settings
        self.generation_count.setValue(settings.count)
        self.generation_seed.setValue(settings.seed)

    def export_csv_dialog(self) -> None:
        if not self._evaluation_records:
            self._warn(
                "Cannot Export CSV",
                "Run generated-sample evaluation before exporting CSV.",
            )
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Evaluation CSV", filter="CSV (*.csv)"
        )
        if not filename:
            return
        try:
            export_results_csv(Path(filename), self._evaluation_records)
        except Exception as error:
            self._show_error("Export CSV Failed", error)
            return
        self.statusBar().showMessage(f"Exported CSV: {filename}")

    def _warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)
        self.statusBar().showMessage(message)

    def _show_error(self, title: str, error: Exception) -> None:
        self._warn(title, str(error) or type(error).__name__)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._search_thread is not None or self._generation_thread is not None:
            self._close_requested = True
            self.cancel_work()
            event.ignore()
            return
        super().closeEvent(event)

    def _close_if_idle(self) -> None:
        if (
            self._close_requested
            and self._search_thread is None
            and self._generation_thread is None
        ):
            self._close_requested = False
            self.close()
