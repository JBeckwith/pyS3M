from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton, QFileDialog
from PyQt6.QtCore import pyqtSignal


class ImagePicker(QWidget):
    """Same shape as FolderPicker, for picking a single image file."""

    path_changed = pyqtSignal(str)

    def __init__(self, placeholder="Select pattern image…", default_dir="", parent=None):
        super().__init__(parent)
        self._path = ""
        self._default_dir = default_dir
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
        path, _ = QFileDialog.getOpenFileName(
            self, "Select pattern image", self._path or self._default_dir or "",
            "Images (*.png *.tif *.tiff *.jpg *.jpeg *.bmp)",
        )
        if path:
            self._path = path
            self._edit.setText(path)
            self.path_changed.emit(path)

    @property
    def path(self) -> str:
        return self._path

    def set_path(self, p: str):
        self._path = str(p)
        self._edit.setText(str(p))
