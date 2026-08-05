from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QPushButton, QLabel,
)
from PyQt6.QtCore import Qt, pyqtSignal

from pyS3M.gui.widgets.folder_picker import FolderPicker

_INSTRUCTIONS = (
    "Raw-data folder layout required:\n\n"
    "• One subdirectory with \"dark\" in its name, holding dark frames. Filenames "
    "must contain both \"dark\" and \".tif\".\n\n"
    "• One subdirectory per Bayer colour, named exactly \"R\", \"G\", \"B\" "
    "(uppercase), each holding flat/bright-field frames at several illumination "
    "intensities. Filenames must contain \"Intensity_<value>\" and \".tif\", and the "
    "same set of intensity values must exist in every colour subfolder."
)


class CalibrationCalcPanel(QWidget):
    """Controls for computing a CMOS calibration from raw dark/flat-field frames
    (CalibrationFunctions.Calibration_Functions.calibrate_multicolour_camera, via
    AnalysisPipeline.calibrate), as opposed to SetupPanel which only ever loads an
    already-computed calibration. Always available (no state gating) — this is the
    entry point that produces what SetupPanel/the rest of the pipeline consumes."""

    # camera, raw_dir
    calibration_compute_requested = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        instructions = QLabel(_INSTRUCTIONS)
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: gray;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        outer.addWidget(instructions)

        grp = QGroupBox("Compute Calibration")
        form = QFormLayout(grp)

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

    def _update_run_btn(self):
        self._run_btn.setEnabled(bool(self._raw_dir.path))

    def _on_run_clicked(self):
        self.calibration_compute_requested.emit(
            self._camera.currentText(),
            self._raw_dir.path,
        )

    # ── public interface ──────────────────────────────────────────────

    def set_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy and bool(self._raw_dir.path))
        self._run_btn.setText("Running…" if busy else "Run Calibration")

    def show_calibration_status(self, msg: str):
        self._status.setText(msg)

    def on_state_changed(self, state: str):
        pass  # always active, like SetupPanel/SimulationPanel
