from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QPushButton, QLabel,
)
from PyQt6.QtCore import pyqtSignal

from pyS3M.gui.widgets.folder_picker import FolderPicker

# Project root is four levels up from this file (src/gui/panels/setup_panel.py)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

_CAMERA_PIXEL_SIZES_NM = {"ximea": 69.0, "zwo": 71.5}

# Ordered candidate calibration dirs per camera (first valid one wins)
_CAL_CANDIDATES: dict[str, list[Path]] = {
    "ximea": [
        _PROJECT_ROOT / "Camera_Calibrations" / "Ximea_Camera",
        _PROJECT_ROOT / "Camera_Calibrations" / "CS505CU_Camera",
    ],
    "zwo": [
        _PROJECT_ROOT / "Camera_Calibrations" / "ZWO_Camera",
    ],
}

_REQUIRED_FILES = ("gain.tif", "offset.tif", "variance.tif", "readnoise.tif", "rqe.tif")


def _find_default_cal_dir(camera: str) -> Path | None:
    """Return the first candidate directory that contains all required calibration files."""
    for candidate in _CAL_CANDIDATES.get(camera, []):
        if all((candidate / f).exists() for f in _REQUIRED_FILES):
            return candidate
    return None


class SetupPanel(QWidget):
    calibration_requested = pyqtSignal(str, float, str)  # camera, pixel_size_um, cal_dir

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
        self._pixel_size.setRange(1.0, 1000.0)
        self._pixel_size.setDecimals(1)
        self._pixel_size.setSingleStep(0.5)
        self._pixel_size.setSuffix(" nm")
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

        # Trigger auto-fill for the default camera on construction
        self._on_camera_changed(self._camera.currentText())

    def _on_camera_changed(self, name: str):
        self._pixel_size.setValue(_CAMERA_PIXEL_SIZES_NM.get(name, 69.0))
        default = _find_default_cal_dir(name)
        if default is not None:
            self._cal_dir.set_path(str(default))
            self._status.setText("—")
        else:
            self._cal_dir.set_path("")
            self._status.setText("Default not found — please Browse")
        self._update_load_btn()

    def _update_load_btn(self):
        self._load_btn.setEnabled(bool(self._cal_dir.path))

    def _on_load_clicked(self):
        self.calibration_requested.emit(
            self._camera.currentText(),
            self._pixel_size.value() / 1000.0,  # nm → µm for pipeline
            self._cal_dir.path,
        )

    # ── public interface ──────────────────────────────────────────────

    @property
    def cal_dir(self) -> str:
        return self._cal_dir.path

    def set_cal_dir(self, path: str):
        """Called by MainWindow to restore a saved path (overrides auto-fill)."""
        self._cal_dir.set_path(path)
        self._update_load_btn()

    def set_busy(self, busy: bool):
        self._load_btn.setEnabled(not busy)
        self._load_btn.setText("Loading…" if busy else "Load Calibration")

    def show_calibration_status(self, msg: str):
        self._status.setText(msg)

    def on_state_changed(self, state: str):
        pass  # setup panel is always active
