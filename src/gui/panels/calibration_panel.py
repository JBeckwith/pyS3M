from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QPushButton, QLabel,
)
from PyQt6.QtCore import Qt, pyqtSignal

from pyS3M.gui.widgets.folder_picker import FolderPicker

_INSTRUCTIONS_RGB = (
    "Raw-data folder layout required (RGB calibration):\n\n"
    "• One subdirectory with \"dark\" in its name, holding dark frames. Filenames "
    "must contain both \"dark\" and \".tif\".\n\n"
    "• One subdirectory per Bayer colour, named exactly \"R\", \"G\", \"B\" "
    "(uppercase), each holding flat/bright-field frames at several illumination "
    "intensities. Filenames must contain \"Intensity_<value>\" and \".tif\", and the "
    "same set of intensity values must exist in every colour subfolder."
)

_INSTRUCTIONS_NIR = (
    "Raw-data folder layout required (NIR calibration):\n\n"
    "• One subdirectory with \"dark\" in its name, holding dark frames. Filenames "
    "must contain both \"dark\" and \".tif\".\n\n"
    "• Exactly one other subdirectory (any name), holding flat/bright-field frames "
    "taken with a >750 nm (near-infrared) light source, at several illumination "
    "intensities. Filenames must contain \"Intensity_<value>\" and \".tif\".\n\n"
    "The Ximea/ZWO Bayer filters' R/G/B transmission spectra converge above ~750 nm, "
    "so every pixel responds identically regardless of its Bayer colour — one "
    "flat-field folder is applied uniformly to the whole sensor, no per-colour split "
    "needed."
)


class CalibrationCalcPanel(QWidget):
    """Controls for computing a CMOS calibration from raw dark/flat-field frames
    (CalibrationFunctions.Calibration_Functions.calibrate_multicolour_camera, via
    AnalysisPipeline.calibrate), as opposed to SetupPanel which only ever loads an
    already-computed calibration. Always available (no state gating) — this is the
    entry point that produces what SetupPanel/the rest of the pipeline consumes."""

    # camera, raw_dir, mode ("rgb"/"nir")
    calibration_compute_requested = pyqtSignal(str, str, str)

    _MODES = ["rgb", "nir"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._instructions = QLabel(_INSTRUCTIONS_RGB)
        self._instructions.setWordWrap(True)
        self._instructions.setStyleSheet("color: gray;")
        self._instructions.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self._instructions)

        grp = QGroupBox("Compute Calibration")
        form = QFormLayout(grp)

        self._calibration_type = QComboBox()
        self._calibration_type.addItems(["RGB (per-colour flats)", "NIR (750 nm+, single flat-field folder)"])
        self._calibration_type.currentIndexChanged.connect(self._on_calibration_type_changed)
        form.addRow("Calibration type:", self._calibration_type)

        self._camera = QComboBox()
        self._camera.addItems(["ximea", "zwo"])
        form.addRow("Camera:", self._camera)

        self._raw_dir = FolderPicker("Select raw calibration data folder…")
        self._raw_dir.path_changed.connect(self._update_run_btn)
        form.addRow("Raw data dir:", self._raw_dir)

        self._run_btn = QPushButton("Run Calibration")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run_clicked)
        form.addRow(self._run_btn)

        self._status = QLabel("—")
        form.addRow("Status:", self._status)

        outer.addWidget(grp)
        outer.addStretch()

    def _on_calibration_type_changed(self, index: int):
        self._instructions.setText(_INSTRUCTIONS_NIR if self._MODES[index] == "nir" else _INSTRUCTIONS_RGB)

    def _update_run_btn(self):
        self._run_btn.setEnabled(bool(self._raw_dir.path))

    def _on_run_clicked(self):
        self.calibration_compute_requested.emit(
            self._camera.currentText(),
            self._raw_dir.path,
            self._MODES[self._calibration_type.currentIndex()],
        )

    # ── public interface ──────────────────────────────────────────────

    def set_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy and bool(self._raw_dir.path))
        self._run_btn.setText("Running…" if busy else "Run Calibration")

    def show_calibration_status(self, msg: str):
        self._status.setText(msg)

    def on_state_changed(self, state: str):
        pass  # always active, like SetupPanel/SimulationPanel
