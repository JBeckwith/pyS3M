from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QCheckBox, QLabel,
)
from PyQt6.QtCore import pyqtSignal


class ChannelUnmixingPanel(QWidget):
    """Controls for spectral channel unmixing (channel_unmixing.unmix_channels).

    Enabled as soon as there's any analysed data (fitted, undrifted, or
    clustered) — MainWindow picks the best-available localisation table to run
    on: the post-clustering per-molecule table (sm_db) if clustering has
    happened (its averaged A_R/A_G values give much tighter spectral clusters
    than any single-frame row), otherwise the undrifted or raw per-frame fitted
    locs, letting a user preview unmixing before committing to the full
    pipeline. Only a run against sm_db is written back into it and unlocks the
    "Per-Channel FRC" group below (which needs the real clustered/molecule-level
    table); earlier-stage runs are preview-only, rather than FRCPanel growing
    its own colour-based split (see claude/gui.md §3.2a's noted future
    integration)."""

    # n_channels, channels_to_use, confidence_threshold, outlier_rejection
    unmixing_requested = pyqtSignal(int, list, float, str)
    # channels (list[int]), zoom, n_blocks, reps
    frc_per_channel_requested = pyqtSignal(list, float, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled_by_state = False
        self._channel_checks: list[tuple[QCheckBox, int]] = []
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        grp = QGroupBox("Channel Unmixing")
        form = QFormLayout(grp)

        self._n_channels = QSpinBox()
        self._n_channels.setRange(2, 5)
        self._n_channels.setValue(2)
        self._n_channels.setToolTip("Number of spectrally distinct dye populations.")
        form.addRow("N channels:", self._n_channels)

        self._features = QComboBox()
        self._features.addItem("A_R, A_G (2D)", ["A_R", "A_G"])
        self._features.addItem("A_R, A_G, A_B (3D)", ["A_R", "A_G", "A_B"])
        self._features.setToolTip("Spectral features used for classification.")
        form.addRow("Spectral features:", self._features)

        self._confidence = QDoubleSpinBox()
        self._confidence.setRange(0.50, 0.999)
        self._confidence.setDecimals(3)
        self._confidence.setSingleStep(0.01)
        self._confidence.setValue(0.95)
        self._confidence.setToolTip(
            "Minimum GMM posterior probability for a confident channel assignment."
        )
        form.addRow("Confidence threshold:", self._confidence)

        self._outlier_rejection = QComboBox()
        self._outlier_rejection.addItems(["mahalanobis", "none"])
        self._outlier_rejection.setToolTip("Outlier handling during GMM fitting.")
        form.addRow("Outlier rejection:", self._outlier_rejection)

        self._run_btn = QPushButton("Run Channel Unmixing")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run_clicked)

        outer.addWidget(grp)
        outer.addWidget(self._run_btn)

        # --- Per-Channel FRC (enabled once unmixing has produced a `channel`
        # column; see set_available_channels) ---
        frc_grp = QGroupBox("Per-Channel FRC")
        frc_outer = QVBoxLayout(frc_grp)

        self._channel_check_box = QGroupBox("Channels")
        self._channel_check_layout = QVBoxLayout(self._channel_check_box)
        self._no_channels_label = QLabel("Run Channel Unmixing first.")
        self._no_channels_label.setStyleSheet("color: gray;")
        self._channel_check_layout.addWidget(self._no_channels_label)
        frc_outer.addWidget(self._channel_check_box)

        frc_form = QFormLayout()
        self._frc_zoom = QDoubleSpinBox()
        self._frc_zoom.setRange(1.0, 20.0)
        self._frc_zoom.setDecimals(1)
        self._frc_zoom.setValue(5.0)
        self._frc_zoom.setToolTip("SR magnification: super-resolution pixels per camera pixel.")
        frc_form.addRow("Zoom:", self._frc_zoom)

        self._frc_n_blocks = QSpinBox()
        self._frc_n_blocks.setRange(2, 500)
        self._frc_n_blocks.setValue(50)
        self._frc_n_blocks.setToolTip("Number of temporal blocks for the random half-split.")
        frc_form.addRow("N blocks:", self._frc_n_blocks)

        self._frc_reps = QSpinBox()
        self._frc_reps.setRange(1, 100)
        self._frc_reps.setValue(10)
        self._frc_reps.setToolTip("Independent half-split repeats to average over.")
        frc_form.addRow("Repeats:", self._frc_reps)
        frc_outer.addLayout(frc_form)

        self._frc_run_btn = QPushButton("Run FRC per Channel")
        self._frc_run_btn.setEnabled(False)
        self._frc_run_btn.clicked.connect(self._on_run_frc_per_channel_clicked)
        frc_outer.addWidget(self._frc_run_btn)

        outer.addWidget(frc_grp)

    def _on_run_clicked(self):
        self.unmixing_requested.emit(
            self._n_channels.value(),
            self._features.currentData(),
            self._confidence.value(),
            self._outlier_rejection.currentText(),
        )

    def _on_run_frc_per_channel_clicked(self):
        channels = [ch for cb, ch in self._channel_checks if cb.isChecked()]
        if not channels:
            return
        self.frc_per_channel_requested.emit(
            channels,
            self._frc_zoom.value(),
            self._frc_n_blocks.value(),
            self._frc_reps.value(),
        )

    # ── public interface ──────────────────────────────────────────────

    def set_available_channels(self, channels: list[int]):
        """Populate the per-channel checklist (e.g. after unmixing produces a
        `channel` column on sm_db). All channels start checked."""
        while self._channel_check_layout.count():
            item = self._channel_check_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._channel_checks = []

        if not channels:
            self._no_channels_label = QLabel("Run Channel Unmixing first.")
            self._no_channels_label.setStyleSheet("color: gray;")
            self._channel_check_layout.addWidget(self._no_channels_label)
        else:
            for ch in channels:
                cb = QCheckBox(f"Channel {ch}")
                cb.setChecked(True)
                self._channel_checks.append((cb, ch))
                self._channel_check_layout.addWidget(cb)

        if not self._frc_run_btn.text().startswith("Running"):
            self._frc_run_btn.setEnabled(bool(channels) and self._enabled_by_state)

    def set_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy and self._enabled_by_state)
        self._run_btn.setText("Running…" if busy else "Run Channel Unmixing")
        self._frc_run_btn.setEnabled(
            not busy and self._enabled_by_state and bool(self._channel_checks)
        )
        self._frc_run_btn.setText("Running…" if busy else "Run FRC per Channel")

    def on_state_changed(self, state: str):
        self._enabled_by_state = state in ("fitted", "undrifted", "clustered")
        if not self._run_btn.text().startswith("Running"):
            self._run_btn.setEnabled(self._enabled_by_state)
            self._frc_run_btn.setEnabled(self._enabled_by_state and bool(self._channel_checks))
