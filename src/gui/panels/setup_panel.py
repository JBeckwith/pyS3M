from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QPushButton, QLabel,
)
from PyQt6.QtCore import pyqtSignal

from gui.widgets.folder_picker import FolderPicker

_CAMERA_PIXEL_SIZES = {"ximea": 0.069, "zwo": 0.0715}


class SetupPanel(QWidget):
    calibration_requested = pyqtSignal(str, float, str)  # camera, pixel_size, cal_dir

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Camera & Calibration")
        form = QFormLayout(grp)

        self._camera = QComboBox()
        self._camera.addItems(["ximea", "zwo"])
        self._camera.currentTextChanged.connect(self._on_camera_changed)
        form.addRow("Camera:", self._camera)

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(0.001, 10.0)
        self._pixel_size.setDecimals(4)
        self._pixel_size.setSingleStep(0.001)
        self._pixel_size.setSuffix(" µm")
        form.addRow("Pixel size:", self._pixel_size)

        self._cal_dir = FolderPicker("Select calibration folder…")
        self._cal_dir.path_changed.connect(self._update_load_btn)
        form.addRow("Calibration dir:", self._cal_dir)

        self._load_btn = QPushButton("Load Calibration")
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._on_load_clicked)
        form.addRow(self._load_btn)

        self._status = QLabel("—")
        form.addRow("Status:", self._status)

        outer.addWidget(grp)
        self._on_camera_changed(self._camera.currentText())

    def _on_camera_changed(self, name: str):
        self._pixel_size.setValue(_CAMERA_PIXEL_SIZES.get(name, 0.069))

    def _update_load_btn(self):
        self._load_btn.setEnabled(bool(self._cal_dir.path))

    def _on_load_clicked(self):
        self.calibration_requested.emit(
            self._camera.currentText(),
            self._pixel_size.value(),
            self._cal_dir.path,
        )

    # ── public interface ──────────────────────────────────────────────

    @property
    def cal_dir(self) -> str:
        return self._cal_dir.path

    def set_cal_dir(self, path: str):
        self._cal_dir.set_path(path)
        self._update_load_btn()

    def set_busy(self, busy: bool):
        self._load_btn.setEnabled(not busy)
        self._load_btn.setText("Loading…" if busy else "Load Calibration")

    def show_calibration_status(self, msg: str):
        self._status.setText(msg)

    def on_state_changed(self, state: str):
        pass  # setup panel is always active
