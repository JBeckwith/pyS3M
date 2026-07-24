from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QLabel,
)
from PyQt6.QtCore import pyqtSignal


class DriftPanel(QWidget):
    """Controls for drift correction. AIM is the only method currently
    exposed in the GUI (DriftCorrectionFunctions also supports "fiducial"
    and "auto", but "auto" always dispatches to AIM internally today, and
    "fiducial" needs a pre-existing 'group' column that the pipeline doesn't
    yet produce)."""

    # segmentation (frames), intersect_d (nm), roi_r (nm)
    undrift_requested = pyqtSignal(int, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled_by_state = False
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        grp = QGroupBox("Drift Correction")
        form = QFormLayout(grp)

        form.addRow("Method:", QLabel("AIM"))

        self._segmentation = QSpinBox()
        self._segmentation.setRange(1, 100_000)
        self._segmentation.setValue(100)
        self._segmentation.setToolTip("Frames per drift-estimation segment.")
        form.addRow("Segmentation frames:", self._segmentation)

        self._intersect_d = QDoubleSpinBox()
        self._intersect_d.setRange(0.1, 10_000.0)
        self._intersect_d.setDecimals(1)
        self._intersect_d.setValue(20.0)
        self._intersect_d.setSuffix(" nm")
        self._intersect_d.setToolTip("AIM intersection distance.")
        form.addRow("Intersect distance:", self._intersect_d)

        self._roi_r = QDoubleSpinBox()
        self._roi_r.setRange(0.1, 10_000.0)
        self._roi_r.setDecimals(1)
        self._roi_r.setValue(60.0)
        self._roi_r.setSuffix(" nm")
        self._roi_r.setToolTip("AIM search ROI radius.")
        form.addRow("ROI radius:", self._roi_r)

        self._run_btn = QPushButton("Run Undrift")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run_clicked)

        outer.addWidget(grp)
        outer.addWidget(self._run_btn)

    def _on_run_clicked(self):
        self.undrift_requested.emit(
            self._segmentation.value(),
            self._intersect_d.value(),
            self._roi_r.value(),
        )

    # ── public interface ──────────────────────────────────────────────

    def set_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy and self._enabled_by_state)
        self._run_btn.setText("Running…" if busy else "Run Undrift")

    def on_state_changed(self, state: str):
        self._enabled_by_state = state in ("fitted", "undrifted")
        if not self._run_btn.text().startswith("Running"):
            self._run_btn.setEnabled(self._enabled_by_state)
