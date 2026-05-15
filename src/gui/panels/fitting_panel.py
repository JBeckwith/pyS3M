from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QLineEdit, QCheckBox,
)
from PyQt6.QtCore import pyqtSignal

from gui.widgets.folder_picker import FolderPicker


class FittingPanel(QWidget):
    fit_requested     = pyqtSignal(str, str, object)  # data_dir, mode, FittingConfig
    preview_requested = pyqtSignal(str, object)        # data_dir, FittingConfig

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled_by_state = False
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Fitting")
        form = QFormLayout(grp)

        self._data_dir = FolderPicker("Select data folder…")
        self._data_dir.path_changed.connect(self._update_btns)
        form.addRow("Data folder:", self._data_dir)

        self._mode = QComboBox()
        self._mode.addItems(["smlm", "imaging"])
        form.addRow("Mode:", self._mode)

        self._pfa = QLineEdit("1e-3")
        self._pfa.setPlaceholderText("e.g. 1e-3")
        form.addRow("PFA:", self._pfa)

        self._roi_size = QSpinBox()
        self._roi_size.setRange(4, 64)
        self._roi_size.setValue(16)
        self._roi_size.setSuffix(" px")
        form.addRow("ROI size:", self._roi_size)

        self._wavelength = QDoubleSpinBox()
        self._wavelength.setRange(0.4, 1.0)
        self._wavelength.setDecimals(3)
        self._wavelength.setSingleStep(0.01)
        self._wavelength.setValue(0.638)
        self._wavelength.setSuffix(" µm")
        form.addRow("Peak λ:", self._wavelength)

        self._na = QDoubleSpinBox()
        self._na.setRange(0.1, 2.0)
        self._na.setDecimals(2)
        self._na.setSingleStep(0.01)
        self._na.setValue(1.49)
        form.addRow("NA:", self._na)

        self._sigma = QDoubleSpinBox()
        self._sigma.setRange(0.1, 10.0)
        self._sigma.setDecimals(2)
        self._sigma.setSingleStep(0.1)
        self._sigma.setValue(1.5)
        form.addRow("Sigma:", self._sigma)

        self._frac_true = QDoubleSpinBox()
        self._frac_true.setRange(0.01, 1.0)
        self._frac_true.setDecimals(2)
        self._frac_true.setSingleStep(0.05)
        self._frac_true.setValue(0.2)
        form.addRow("Fraction true:", self._frac_true)

        self._var_demosaic = QCheckBox("Variance-aware demosaic")
        self._var_demosaic.setChecked(True)
        form.addRow(self._var_demosaic)

        # Preview + Run buttons side by side
        btn_row = QWidget()
        btn_lay = QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(6)
        self._preview_btn = QPushButton("Preview Fit")
        self._preview_btn.setEnabled(False)
        self._preview_btn.setToolTip("Run spot detection + fitting on one frame")
        self._preview_btn.clicked.connect(self._on_preview_clicked)
        self._run_btn = QPushButton("Run Fitting")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run_clicked)
        btn_lay.addWidget(self._preview_btn)
        btn_lay.addWidget(self._run_btn)
        form.addRow(btn_row)

        outer.addWidget(grp)

    # ── helpers ──────────────────────────────────────────────────────

    def _make_fitting_config(self):
        from AnalysisPipeline import FittingConfig
        try:
            pfa = float(self._pfa.text())
        except ValueError:
            pfa = 1e-3
        return FittingConfig(
            pfa=pfa,
            ROI_size=self._roi_size.value(),
            peak_wavelength=self._wavelength.value(),
            NA=self._na.value(),
            sigma=self._sigma.value(),
            fraction_true=self._frac_true.value(),
            use_variance_aware_demosaic=self._var_demosaic.isChecked(),
        )

    def _update_btns(self):
        ok = self._enabled_by_state and bool(self._data_dir.path)
        if not self._preview_btn.text().startswith("Preview…"):
            self._preview_btn.setEnabled(ok)
        if not self._run_btn.text().startswith("Running"):
            self._run_btn.setEnabled(ok)

    def _on_preview_clicked(self):
        self.preview_requested.emit(self._data_dir.path, self._make_fitting_config())

    def _on_run_clicked(self):
        self.fit_requested.emit(
            self._data_dir.path, self._mode.currentText(), self._make_fitting_config()
        )

    # ── public interface ──────────────────────────────────────────────

    @property
    def data_dir(self) -> str:
        return self._data_dir.path

    def set_data_dir(self, path: str):
        self._data_dir.set_path(path)
        self._update_btns()

    def set_preview_busy(self, busy: bool):
        self._preview_btn.setEnabled(not busy)
        self._preview_btn.setText("Preview…" if busy else "Preview Fit")
        self._run_btn.setEnabled(not busy and self._enabled_by_state and bool(self._data_dir.path))

    def set_fit_busy(self, busy: bool):
        self._run_btn.setEnabled(not busy)
        self._run_btn.setText("Running…" if busy else "Run Fitting")
        self._preview_btn.setEnabled(not busy and self._enabled_by_state and bool(self._data_dir.path))

    def set_busy(self, busy: bool):
        """Reset both buttons (used by error handler)."""
        self._preview_btn.setEnabled(not busy)
        self._preview_btn.setText("Preview Fit")
        self._run_btn.setEnabled(not busy)
        self._run_btn.setText("Run Fitting")
        if not busy:
            self._update_btns()

    def on_state_changed(self, state: str):
        self._enabled_by_state = state in ("calibrated", "fitted", "clustered")
        self._update_btns()
