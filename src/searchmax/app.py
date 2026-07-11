"""Application entry point for OpenCV SearchMax."""

import sys

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("OpenCV SearchMax")
    window = MainWindow()
    window.resize(1500, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
