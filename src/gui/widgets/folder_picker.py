from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton, QFileDialog
from PyQt6.QtCore import pyqtSignal


class FolderPicker(QWidget):
    path_changed = pyqtSignal(str)

    def __init__(self, placeholder="Select folder…", parent=None):
        super().__init__(parent)
        self._path = ""
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText(placeholder)
        self._edit.setReadOnly(True)
        btn = QPushButton("Browse…")
        btn.setFixedWidth(72)
        btn.clicked.connect(self._browse)
        lay.addWidget(self._edit)
        lay.addWidget(btn)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder", self._path or "")
        if d:
            self._path = d
            self._edit.setText(d)
            self.path_changed.emit(d)

    @property
    def path(self) -> str:
        return self._path

    def set_path(self, p: str):
        self._path = str(p)
        self._edit.setText(str(p))
