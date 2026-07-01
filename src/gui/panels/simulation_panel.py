from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton,
    QLabel, QScrollArea, QCheckBox, QGridLayout,
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

    simulation_requested = pyqtSignal(str, list, float, float, int)
    # args: dye_name, filter_ids, bg_photons_per_px, NA, n_replicates

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
        dye = self._dye.currentText()
        filters = [fid for cb, fid in self._filter_checks if cb.isChecked()]
        bg = self._bg.value()
        na = self._na.value()
        n_rep = self._n_rep.value()
        self.simulation_requested.emit(dye, filters, bg, na, n_rep)

    def set_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy)
        self._run_btn.setText("Running…" if busy else "Run Simulation")

    def on_state_changed(self, _state: str):
        pass
