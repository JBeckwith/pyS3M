from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QPushButton,
)
from PyQt6.QtCore import pyqtSignal


class FRCPanel(QWidget):
    """Controls for Fourier Ring Correlation (FIRE) resolution estimation
    (FRCFunctions.fire), run on whatever localisations are currently
    available (undrifted in-memory result preferred, else the fitted
    dataset on disk). Computes a single spatial-resolution estimate over
    all localisations — splitting by dye/channel is a Channel Unmixing
    concern (once that panel produces a categorical `channel` column, FRC
    can be re-run per selected channel from there), not something this
    panel does on its own."""

    # zoom (SR px per camera px), n_blocks, reps
    frc_requested = pyqtSignal(float, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled_by_state = False
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        grp = QGroupBox("FRC Resolution")
        form = QFormLayout(grp)

        self._zoom = QDoubleSpinBox()
        self._zoom.setRange(1.0, 20.0)
        self._zoom.setDecimals(1)
        self._zoom.setValue(5.0)
        self._zoom.setToolTip("SR magnification: super-resolution pixels per camera pixel.")
        form.addRow("Zoom:", self._zoom)

        self._n_blocks = QSpinBox()
        self._n_blocks.setRange(2, 500)
        self._n_blocks.setValue(50)
        self._n_blocks.setToolTip("Number of temporal blocks for the random half-split.")
        form.addRow("N blocks:", self._n_blocks)

        self._reps = QSpinBox()
        self._reps.setRange(1, 100)
        self._reps.setValue(10)
        self._reps.setToolTip("Independent half-split repeats to average over.")
        form.addRow("Repeats:", self._reps)

        self._run_btn = QPushButton("Run FRC")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run_clicked)

        outer.addWidget(grp)
        outer.addWidget(self._run_btn)

    def _on_run_clicked(self):
        self.frc_requested.emit(
            self._zoom.value(),
            self._n_blocks.value(),
            self._reps.value(),
        )

    # ── public interface ──────────────────────────────────────────────

    def set_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy and self._enabled_by_state)
        self._run_btn.setText("Running…" if busy else "Run FRC")

    def on_state_changed(self, state: str):
        self._enabled_by_state = state in ("fitted", "undrifted", "clustered")
        if not self._run_btn.text().startswith("Running"):
            self._run_btn.setEnabled(self._enabled_by_state)
