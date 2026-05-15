import enum
import logging
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout,
    QScrollArea, QLabel, QMessageBox, QApplication,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings

from gui.panels.setup_panel import SetupPanel
from gui.panels.fitting_panel import FittingPanel
from gui.panels.postproc_panel import PostProcPanel
from gui.panels.results_panel import ResultsPanel
from gui.widgets.log_widget import LogWidget, QtLogHandler
from gui.widgets.progress_widget import ProgressWidget
from gui.worker import AnalysisWorker

logger = logging.getLogger(__name__)


class AppState(enum.Enum):
    IDLE = "idle"
    CALIBRATED = "calibrated"
    FITTED = "fitted"
    CLUSTERED = "clustered"


class MainWindow(QMainWindow):
    state_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyBayerSMLM")
        self.resize(1400, 900)

        self._state = AppState.IDLE
        self.pipeline = None
        self._worker: AnalysisWorker | None = None      # main pipeline worker
        self._aux_worker: AnalysisWorker | None = None  # viz-only worker (non-blocking)
        self._fitted_data_dir: str | None = None
        self._sm_db = None
        self._sf_db = None

        self._build_ui()
        self._connect_signals()
        self._install_log_handler()
        self._restore_settings()
        self._update_state(AppState.IDLE)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.results_panel = ResultsPanel(self)
        self.setCentralWidget(self.results_panel)

        self.setup_panel = SetupPanel(self)
        self.fitting_panel = FittingPanel(self)
        self.postproc_panel = PostProcPanel(self)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(8)
        vbox.addWidget(self.setup_panel)
        vbox.addWidget(self.fitting_panel)
        vbox.addWidget(self.postproc_panel)
        vbox.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        ctrl_dock = QDockWidget("Controls", self)
        ctrl_dock.setWidget(scroll)
        ctrl_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        ctrl_dock.setMinimumWidth(330)
        ctrl_dock.setMaximumWidth(440)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, ctrl_dock)

        bottom = QWidget()
        blay = QVBoxLayout(bottom)
        blay.setContentsMargins(4, 4, 4, 4)
        blay.setSpacing(4)
        self.progress_widget = ProgressWidget(bottom)
        self.log_widget = LogWidget(bottom)
        blay.addWidget(self.progress_widget)
        blay.addWidget(self.log_widget)

        log_dock = QDockWidget("Log", self)
        log_dock.setWidget(bottom)
        log_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        log_dock.setMaximumHeight(220)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)

        self._status_label = QLabel("Idle")
        self.statusBar().addPermanentWidget(self._status_label)

    def _connect_signals(self):
        self.state_changed.connect(self.setup_panel.on_state_changed)
        self.state_changed.connect(self.fitting_panel.on_state_changed)
        self.state_changed.connect(self.postproc_panel.on_state_changed)

        self.setup_panel.calibration_requested.connect(self._on_load_calibration)
        self.fitting_panel.preview_requested.connect(self._on_preview_fitting)
        self.fitting_panel.fit_requested.connect(self._on_run_fitting)
        self.postproc_panel.cluster_requested.connect(self._on_run_clustering)

    def _install_log_handler(self):
        self._log_handler = QtLogHandler()
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s  %(name)s  %(message)s", datefmt="%H:%M:%S")
        )
        self._log_handler.message.connect(self.log_widget.append)
        logging.getLogger().addHandler(self._log_handler)
        logging.getLogger().setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _update_state(self, new_state: AppState):
        self._state = new_state
        self._status_label.setText(new_state.value.replace("_", " ").title())
        self.state_changed.emit(new_state.value)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _restore_settings(self):
        s = QSettings()
        if cal := s.value("cal_dir", ""):
            self.setup_panel.set_cal_dir(cal)
        if dat := s.value("data_dir", ""):
            self.fitting_panel.set_data_dir(dat)

    def _save_settings(self):
        s = QSettings()
        s.setValue("cal_dir", self.setup_panel.cal_dir)
        s.setValue("data_dir", self.fitting_panel.data_dir)

    # ------------------------------------------------------------------
    # Worker lifecycle helpers
    # ------------------------------------------------------------------

    def _worker_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _reset_busy(self):
        self.setup_panel.set_busy(False)
        self.fitting_panel.set_busy(False)
        self.postproc_panel.set_busy(False)
        self.progress_widget.reset()

    def _start_worker(self, fn) -> AnalysisWorker:
        """Create, store, and wire a main-pipeline worker.

        The worker clears ``self._worker`` via ``finished`` (after the thread
        has fully exited) rather than inside result/error slots, which would
        drop the Python reference while the C++ QThread is still cleaning up.
        """
        worker = AnalysisWorker(fn)
        self._worker = worker
        worker.progress.connect(self.progress_widget.update)
        worker.log.connect(self.log_widget.append)
        worker.error.connect(self._on_worker_error)
        # Clear the reference only after the OS thread is fully done.
        worker.finished.connect(lambda w=worker: self._on_main_worker_finished(w))
        return worker

    def _on_main_worker_finished(self, w: AnalysisWorker):
        if self._worker is w:
            self._worker = None

    def _on_worker_error(self, msg: str):
        self._reset_busy()
        self.log_widget.append(f"ERROR:\n{msg}")
        self._status_label.setText("Error")
        QMessageBox.critical(self, "Pipeline error", msg[:600])

    # ------------------------------------------------------------------
    # Figure helpers (use Figure directly — no pyplot, thread-safe)
    # ------------------------------------------------------------------

    def _make_scatter_figure(self, locs, title: str = "Localisations") -> Figure:
        pixel_size_nm = self.pipeline.pixel_size * 1000
        x = locs["xc"].values * pixel_size_nm
        y = locs["yc"].values * pixel_size_nm

        fig = Figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)

        spectral_cols = ("A_R", "A_G", "A_B")
        if all(c in locs.columns for c in spectral_cols):
            total = locs["A_R"].values + locs["A_G"].values + locs["A_B"].values
            total = np.where(total == 0, 1.0, total)
            c = locs["A_R"].values / total
            sc = ax.scatter(x, y, c=c, s=1, cmap="RdYlBu_r",
                            vmin=0, vmax=1, rasterized=True, alpha=0.5, linewidths=0)
            fig.colorbar(sc, ax=ax, label="A_R fraction", shrink=0.8)
        else:
            ax.scatter(x, y, s=1, rasterized=True, alpha=0.5)

        ax.set_xlabel("x (nm)")
        ax.set_ylabel("y (nm)")
        ax.set_title(f"{title}  ({len(locs):,})")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _on_load_calibration(self, camera: str, pixel_size: float, cal_dir: str):
        if self._worker_running():
            return

        def _do():
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from AnalysisPipeline import AnalysisPipeline
            from Constants import AnalysisConfig
            cfg = AnalysisConfig(
                display=False,
                progress_callback=lambda f, m: worker.progress.emit(f, m),
                logging_callback=lambda m: worker.log.emit(m),
            )
            pipe = AnalysisPipeline(camera=camera, pixel_size=pixel_size, config=cfg)
            pipe.load_calibration(Path(cal_dir))
            return pipe

        worker = self._start_worker(_do)
        worker.result.connect(self._on_calibration_done)
        self.setup_panel.set_busy(True)
        worker.start()

    def _on_calibration_done(self, pipe):
        self.pipeline = pipe
        self._save_settings()
        self.setup_panel.set_busy(False)
        shape = pipe.gain_map.shape
        self.setup_panel.show_calibration_status(f"✓ {shape[0]}×{shape[1]}")
        self.progress_widget.update(1.0, "Calibration loaded")
        self._update_state(AppState.CALIBRATED)

    # ------------------------------------------------------------------
    # Preview fitting — runs on the main thread because example_spots_singleframe
    # calls plt.subplots() via PlottingBase, which creates a Qt canvas.
    # Qt widgets must be created on the main thread.
    # ------------------------------------------------------------------

    def _on_preview_fitting(self, data_dir: str, fitting_config):
        if self._worker_running() or self.pipeline is None:
            return

        self.fitting_panel.set_preview_busy(True)
        self.progress_widget.update(0.0, "Running preview…")
        self._status_label.setText("Preview running…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()   # flush UI updates before blocking

        try:
            sf = self.pipeline.make_smoothing_function(fitting_config.sigma)
            result = self.pipeline.sr.example_spots_singleframe(
                image_folder=Path(data_dir),
                smoothing_function=sf,
                gain_map=self.pipeline.gain_map,
                offset_map=self.pipeline.offset_map,
                rqe=self.pipeline.rqe,
                read_noise=self.pipeline.read_noise,
                variance=self.pipeline.variance,
                pfa=fitting_config.pfa,
                ROI_size=fitting_config.ROI_size,
                peak_wavelength=fitting_config.peak_wavelength,
                NA=fitting_config.NA,
                pixel_size=self.pipeline.pixel_size,
                sigma=fitting_config.sigma,
                fraction_true=fitting_config.fraction_true,
                use_variance_aware_demosaic=fitting_config.use_variance_aware_demosaic,
            )
            if result is not None:
                fig, _ = result
                self.results_panel.set_preview_figure(fig)
            self.progress_widget.update(1.0, "Preview ready")
        except Exception:
            import traceback as _tb
            self.log_widget.append(f"Preview error:\n{_tb.format_exc()}")
            self.progress_widget.update(0.0, "Preview failed")
        finally:
            QApplication.restoreOverrideCursor()
            self.fitting_panel.set_preview_busy(False)
            self._update_state(self._state)  # restore status bar label

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _on_run_fitting(self, data_dir: str, mode: str, fitting_config):
        if self._worker_running() or self.pipeline is None:
            return

        def _do():
            self.pipeline.config.progress_callback = lambda f, m: worker.progress.emit(f, m)
            self.pipeline.config.logging_callback = lambda m: worker.log.emit(m)
            self.pipeline.fit(Path(data_dir), mode=mode, fitting_config=fitting_config)

        worker = self._start_worker(_do)
        worker.result.connect(lambda _: self._on_fitting_done(data_dir))
        self.fitting_panel.set_fit_busy(True)
        worker.start()

    def _on_fitting_done(self, data_dir: str):
        self._fitted_data_dir = data_dir
        self.fitting_panel.set_fit_busy(False)
        self.progress_widget.update(1.0, "Fitting complete")
        self._update_state(AppState.FITTED)
        self._refresh_locs_figure(data_dir)

    # ------------------------------------------------------------------
    # Locs scatter (aux worker — non-blocking visualisation)
    # ------------------------------------------------------------------

    def _refresh_locs_figure(self, data_dir: str):
        def _do():
            locs = self.pipeline.load_localisations(data_dir)
            if locs.empty:
                return None
            return self._make_scatter_figure(locs, title="Localisations")

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(self._on_locs_figure_ready)
        aux.error.connect(
            lambda msg: self.log_widget.append(f"Warning: locs scatter failed: {msg[:120]}")
        )
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    def _on_aux_worker_finished(self, w: AnalysisWorker):
        if self._aux_worker is w:
            self._aux_worker = None

    def _on_locs_figure_ready(self, fig):
        if fig is not None:
            self.results_panel.set_localisations_figure(fig)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _on_run_clustering(self, criteria, clustering_config):
        if self._worker_running() or self.pipeline is None or self._fitted_data_dir is None:
            return

        def _do():
            locs = self.pipeline.load_localisations(self._fitted_data_dir)
            return self.pipeline.filter_and_cluster(
                locs, criteria=criteria, clustering_config=clustering_config
            )

        worker = self._start_worker(_do)
        worker.result.connect(self._on_clustering_done)
        self.postproc_panel.set_busy(True)
        worker.start()

    def _on_clustering_done(self, result):
        sm_db, sf_db = result
        self._sm_db = sm_db
        self._sf_db = sf_db
        self.postproc_panel.set_busy(False)
        self.postproc_panel.show_result(len(sm_db), len(sf_db))
        self.progress_widget.update(1.0, "Clustering complete")
        self._update_state(AppState.CLUSTERED)
        if not sm_db.empty:
            fig = self._make_scatter_figure(sm_db, title="Single molecules")
            self.results_panel.set_localisations_figure(fig)

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        for w in (self._worker, self._aux_worker):
            if w is not None and w.isRunning():
                w.quit()
                w.wait(2000)
        logging.getLogger().removeHandler(self._log_handler)
        super().closeEvent(event)
