from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QLabel,
)
from PyQt6.QtCore import pyqtSignal


class PostProcPanel(QWidget):
    cluster_requested = pyqtSignal(object, object)  # FilteringCriteria, ClusteringConfig

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled_by_state = False
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Filtering group
        fgrp = QGroupBox("Filtering")
        fform = QFormLayout(fgrp)

        self._min_photons = QSpinBox()
        self._min_photons.setRange(1, 1_000_000)
        self._min_photons.setValue(500)
        fform.addRow("Min photons:", self._min_photons)

        self._max_colour_err = QDoubleSpinBox()
        self._max_colour_err.setRange(0.0, 1.0)
        self._max_colour_err.setDecimals(3)
        self._max_colour_err.setSingleStep(0.01)
        self._max_colour_err.setValue(0.15)
        fform.addRow("Max colour error:", self._max_colour_err)

        self._max_loc_err = QDoubleSpinBox()
        self._max_loc_err.setRange(0.0, 10.0)
        self._max_loc_err.setDecimals(2)
        self._max_loc_err.setSingleStep(0.1)
        self._max_loc_err.setValue(1.0)
        self._max_loc_err.setSuffix(" px")
        fform.addRow("Max loc error:", self._max_loc_err)

        # Clustering group
        cgrp = QGroupBox("Clustering")
        cform = QFormLayout(cgrp)

        self._method = QComboBox()
        self._method.addItems(["HDBSCAN", "DBSCAN"])
        cform.addRow("Method:", self._method)

        self._min_cluster = QSpinBox()
        self._min_cluster.setRange(1, 1000)
        self._min_cluster.setValue(10)
        cform.addRow("Min cluster size:", self._min_cluster)

        self._start_frame = QSpinBox()
        self._start_frame.setRange(0, 10_000_000)
        self._start_frame.setValue(0)
        cform.addRow("Start frame:", self._start_frame)

        self._cluster_btn = QPushButton("Filter & Cluster")
        self._cluster_btn.setEnabled(False)
        self._cluster_btn.clicked.connect(self._on_cluster_clicked)

        self._result_label = QLabel("—")

        outer.addWidget(fgrp)
        outer.addWidget(cgrp)
        outer.addWidget(self._cluster_btn)
        outer.addWidget(self._result_label)

    def _on_cluster_clicked(self):
        from Constants import FilteringCriteria
        from clustering import ClusteringConfig
        filt = FilteringCriteria(
            min_photons=self._min_photons.value(),
            max_colour_error=self._max_colour_err.value(),
            max_localisation_error=self._max_loc_err.value(),
        )
        cc = ClusteringConfig(
            clustering_method=self._method.currentText(),
            min_cluster_size=self._min_cluster.value(),
            start_frame=self._start_frame.value(),
        )
        self.cluster_requested.emit(filt, cc)

    # ── public interface ──────────────────────────────────────────────

    def set_busy(self, busy: bool):
        self._cluster_btn.setEnabled(not busy)
        self._cluster_btn.setText("Running…" if busy else "Filter & Cluster")

    def show_result(self, n_sm: int, n_sf: int):
        self._result_label.setText(f"{n_sm} molecules from {n_sf} localisations")

    def on_state_changed(self, state: str):
        self._enabled_by_state = state in ("fitted", "clustered")
        if not self._cluster_btn.text().startswith("Running"):
            self._cluster_btn.setEnabled(self._enabled_by_state)
