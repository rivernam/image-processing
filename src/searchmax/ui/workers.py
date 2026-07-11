"""Widget-free background workers for SearchMax batch operations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
from threading import Event

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from searchmax.generator import GenerationSettings, generate_samples
from searchmax.image_io import read_image
from searchmax.matcher import match
from searchmax.models import Rect, SearchSettings, TrainModel


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    snapshot = array.copy()
    snapshot.setflags(write=False)
    return snapshot


def _snapshot_model(model: TrainModel) -> TrainModel:
    roi = model.roi
    return TrainModel(
        _readonly_copy(model.color),
        _readonly_copy(model.gray),
        Path(str(model.source)),
        None if roi is None else Rect(roi.x, roi.y, roi.width, roi.height),
    )


class SearchWorker(QObject):
    """Decode and match a fixed batch, reporting failures per input file."""

    progress = Signal(int, int)
    item_finished = Signal(object, object, object)
    diagnostic_finished = Signal(object, object)
    failed = Signal(object, str)
    finished = Signal()
    cancelled = Signal()

    def __init__(
        self,
        paths: tuple[Path, ...],
        model: TrainModel,
        settings: SearchSettings,
        diagnostics: bool = False,
    ) -> None:
        super().__init__()
        self._paths = tuple(Path(path) for path in paths)
        self._model = _snapshot_model(model)
        self._settings = replace(settings)
        self._diagnostics = diagnostics
        self._stop = Event()

    @Slot()
    def run(self) -> None:
        total = len(self._paths)
        for index, path in enumerate(self._paths, start=1):
            if self._stop.is_set():
                self.cancelled.emit()
                return
            try:
                match_settings = (
                    replace(self._settings, max_results=100)
                    if self._diagnostics
                    else self._settings
                )
                image = read_image(path)
                results = match(self._model, image, match_settings)
            except Exception as error:  # each bad file must not abort the batch
                self.failed.emit(path, str(error) or type(error).__name__)
            else:
                self.item_finished.emit(
                    path, image, results[: self._settings.max_results]
                )
                if self._diagnostics:
                    self.diagnostic_finished.emit(path, results)
            self.progress.emit(index, total)
        if self._stop.is_set():
            self.cancelled.emit()
        else:
            self.finished.emit()

    @Slot()
    def cancel(self) -> None:
        self._stop.set()


class GenerationWorker(QObject):
    """Run sample generation away from the GUI thread."""

    progress = Signal(int, int)
    item_finished = Signal(object)
    failed = Signal(object, str)
    finished = Signal()
    cancelled = Signal()

    def __init__(
        self,
        model: TrainModel,
        backgrounds: tuple[np.ndarray, ...],
        output_dir: Path,
        settings: GenerationSettings,
    ) -> None:
        super().__init__()
        self._model = _snapshot_model(model)
        self._backgrounds = tuple(
            _readonly_copy(background) for background in backgrounds
        )
        self._output_dir = Path(output_dir)
        self._settings = replace(settings)
        self._stop = Event()

    @Slot()
    def run(self) -> None:
        total = self._settings.count
        for index in range(1, total + 1):
            if self._stop.is_set():
                self.cancelled.emit()
                return
            destination = self._output_dir / f"sample_{index:04d}.png"
            temporary = self._output_dir / f".searchmax-{index:04d}"
            settings = replace(
                self._settings,
                count=1,
                seed=self._settings.seed + index - 1,
            )
            try:
                [sample] = generate_samples(
                    self._model,
                    list(self._backgrounds),
                    temporary,
                    settings,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                sample.image_path.replace(destination)
                self.item_finished.emit(replace(sample, image_path=destination))
            except Exception as error:
                self.failed.emit(destination, str(error) or type(error).__name__)
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
            self.progress.emit(index, total)
        if self._stop.is_set():
            self.cancelled.emit()
        else:
            self.finished.emit()

    @Slot()
    def cancel(self) -> None:
        self._stop.set()
