from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QCheckBox,
)
from PyQt6.QtCore import pyqtSignal

# Curated dyes commonly used in pyS3M notebooks
_CURATED_DYES = [
    "Cy3B",
    "Cy3",
    "Cy5",
    "Alexa Fluor 647",
    "ATTO 488",
    "ATTO 532",
    "ATTO 550",
    "ATTO 647N",
    "Nile Red",
    "Janelia Fluor JF549-HaloTag conjugate",
    "Janelia Fluor JF646-HaloTag conjugate",
    "SiR",
    "Abberior STAR RED",
]

# Curated filters: (display_label, filter_id)
_CURATED_FILTERS = [
    ("Notch 405/488/561/635", "semrock-nf03-405-488-561-635e"),
    ("BP 520/44",             "semrock-ff01-520-44"),
    ("BP 540/80",             "semrock-ff01-540-80"),
    ("BP 582/64",             "semrock-ff01-582-64"),
    ("BP 650/200",            "semrock-ff01-650-200"),
    ("LP 488",               "semrock-blp01-488r"),
    ("LP 561",               "semrock-blp02-561r"),
    ("LP 568",               "semrock-blp01-568r"),
    ("LP 635",               "semrock-blp01-635r"),
]

# Fixed photon levels to sweep (rows in the PSF grid)
_PHOTON_LEVELS = [200, 500, 1_000, 5_000]


class SimulationPanel(QWidget):
    """Controls for the Simulation tab — exemplar PSF grid."""

    # dye_name, filter_ids, bg_photons_per_px, na, pixel_size_nm,
    # read_noise_e, peak_qe, n_replicates
    simulation_requested = pyqtSignal(str, list, float, float, float, float, float, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # --- Dye ---
        dye_grp = QGroupBox("Dye")
        dye_form = QFormLayout(dye_grp)
        self._dye = QComboBox()
        self._dye.addItems(_CURATED_DYES)
        dye_form.addRow("Dye:", self._dye)
        outer.addWidget(dye_grp)

        # --- Filters ---
        filt_grp = QGroupBox("Filters")
        filt_layout = QVBoxLayout(filt_grp)
        self._filter_checks: list[tuple[QCheckBox, str]] = []
        for label, fid in _CURATED_FILTERS:
            cb = QCheckBox(label)
            self._filter_checks.append((cb, fid))
            filt_layout.addWidget(cb)
        outer.addWidget(filt_grp)

        # --- Parameters ---
        param_grp = QGroupBox("Parameters")
        param_form = QFormLayout(param_grp)

        self._bg = QDoubleSpinBox()
        self._bg.setRange(0.0, 1000.0)
        self._bg.setValue(5.0)
        self._bg.setDecimals(1)
        self._bg.setSuffix(" ph/px")
        param_form.addRow("Background:", self._bg)

        self._na = QDoubleSpinBox()
        self._na.setRange(0.5, 1.7)
        self._na.setValue(1.49)
        self._na.setDecimals(2)
        param_form.addRow("NA:", self._na)

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(10.0, 500.0)
        self._pixel_size.setValue(80.0)
        self._pixel_size.setDecimals(1)
        self._pixel_size.setSuffix(" nm")
        param_form.addRow("Pixel size:", self._pixel_size)

        self._read_noise = QDoubleSpinBox()
        self._read_noise.setRange(0.0, 20.0)
        self._read_noise.setValue(1.0)
        self._read_noise.setDecimals(2)
        self._read_noise.setSuffix(" e⁻ RMS")
        param_form.addRow("Read noise:", self._read_noise)

        self._peak_qe = QDoubleSpinBox()
        self._peak_qe.setRange(0.01, 1.0)
        self._peak_qe.setValue(0.7)
        self._peak_qe.setDecimals(2)
        param_form.addRow("Peak QE:", self._peak_qe)

        self._n_rep = QSpinBox()
        self._n_rep.setRange(1, 10)
        self._n_rep.setValue(5)
        param_form.addRow("Replicates:", self._n_rep)

        outer.addWidget(param_grp)

        # --- Run ---
        self._run_btn = QPushButton("Run Simulation")
        self._run_btn.clicked.connect(self._on_run)
        outer.addWidget(self._run_btn)

        outer.addStretch()

    def _on_run(self):
        self.simulation_requested.emit(
            self._dye.currentText(),
            [fid for cb, fid in self._filter_checks if cb.isChecked()],
            self._bg.value(),
            self._na.value(),
            self._pixel_size.value(),
            self._read_noise.value(),
            self._peak_qe.value(),
            self._n_rep.value(),
        )

    def set_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy)
        self._run_btn.setText("Running…" if busy else "Run Simulation")

    def on_state_changed(self, _state: str):
        pass
