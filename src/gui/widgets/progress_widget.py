from PyQt6.QtWidgets import QWidget, QHBoxLayout, QProgressBar, QLabel


class ProgressWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._label = QLabel("")
        self._label.setMinimumWidth(200)
        lay.addWidget(self._bar)
        lay.addWidget(self._label)

    def update(self, fraction: float, msg: str = ""):
        self._bar.setValue(int(fraction * 100))
        self._label.setText(msg)

    def reset(self):
        self._bar.setValue(0)
        self._label.setText("")
