import sys
import threading
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

from gui.main_window import MainWindow

_SRC = str(Path(__file__).parent.parent)


def _prewarm():
    """Import heavy pipeline modules in the background so the first calibration
    load feels instant.  Runs once in a daemon thread at startup."""
    sys.path.insert(0, _SRC)
    import AnalysisPipeline   # pulls in polars, tifffile, numpy, pandas
    import IOFunctions        # tifffile, imageio and related I/O stack
    import Constants          # dataclasses used by every pipeline call


def run():
    app = QApplication(sys.argv)
    app.setApplicationName("pyS3M")
    app.setOrganizationName("LeeGroup")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)

    threading.Thread(target=_prewarm, daemon=True, name="prewarm").start()

    w = MainWindow()
    w.show()
    sys.exit(app.exec())
