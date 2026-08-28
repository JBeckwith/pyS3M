import logging
from PyQt6.QtCore import pyqtSignal, QObject
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit


class _Signaller(QObject):
    """Plain QObject to hold `message` -- logging.Handler isn't a QObject
    itself, so it can't emit Qt signals directly; QtLogHandler owns one of
    these and re-exposes its signal as its own `message` attribute."""

    message = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    """logging.Handler that posts records to a Qt signal (thread-safe)."""

    def __init__(self):
        super().__init__()
        self._s = _Signaller()
        self.message = self._s.message

    def emit(self, record: logging.LogRecord):
        self._s.message.emit(self.format(record))


class LogWidget(QWidget):
    """A read-only, capped-length (2000 blocks) scrolling log view.
    `append` adds a line and auto-scrolls to the bottom — pair with
    `QtLogHandler` to stream `logging` records into it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(2000)
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        lay.addWidget(self._text)

    def append(self, msg: str):
        self._text.appendPlainText(msg.rstrip())
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())
