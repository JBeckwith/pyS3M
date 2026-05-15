import traceback
from PyQt6.QtCore import QThread, pyqtSignal


class AnalysisWorker(QThread):
    """Background thread for pipeline operations.

    Re-create one instance per pipeline step — do not reuse QThread instances.
    """

    progress = pyqtSignal(float, str)
    log = pyqtSignal(str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            out = self._fn(*self._args, **self._kwargs)
            self.result.emit(out)
        except Exception:
            self.error.emit(traceback.format_exc())
