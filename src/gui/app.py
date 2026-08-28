import multiprocessing
import sys
import threading
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

from pyS3M.gui.main_window import MainWindow

_SRC = str(Path(__file__).parent.parent)


def _prewarm():
    """Import heavy pipeline modules in the background so the first calibration
    load feels instant.  Runs once in a daemon thread at startup."""
    sys.path.insert(0, _SRC)
    import AnalysisPipeline   # pulls in polars, tifffile, numpy, pandas
    import IOFunctions        # tifffile, imageio and related I/O stack
    import Constants          # dataclasses used by every pipeline call


def run():
    # Must be the first call: on Windows, the fitting pipeline's ProcessPoolExecutor
    # use (SpotDetectionFunctions.detect_puncta_in_stack_parallel etc.) spawns child
    # interpreters that re-import this entry point. freeze_support() is a no-op on
    # Linux/macOS but is required on Windows for any app launched from a console-script
    # entry point (as opposed to a plain `if __name__ == "__main__":`-guarded script)
    # that uses multiprocessing, or child processes can fail to bootstrap correctly.
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    app.setApplicationName("pyS3M")
    app.setOrganizationName("LeeGroup")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)

    threading.Thread(target=_prewarm, daemon=True, name="prewarm").start()

    w = MainWindow()
    w.show()
    sys.exit(app.exec())
