import enum
import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout,
    QScrollArea, QLabel, QMessageBox,
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
        self._worker: AnalysisWorker | None = None
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

        # ── Left dock: control panels ──────────────────────────────────
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

        # ── Bottom dock: progress + log ────────────────────────────────
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
        label = new_state.value.replace("_", " ").title()
        self._status_label.setText(label)
        self.state_changed.emit(new_state.value)
        logger.debug("State → %s", new_state.value)

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
    # Worker helpers
    # ------------------------------------------------------------------

    def _worker_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _reset_busy(self):
        self.setup_panel.set_busy(False)
        self.fitting_panel.set_busy(False)
        self.postproc_panel.set_busy(False)
        self.progress_widget.reset()

    def _on_worker_error(self, msg: str):
        self._worker = None
        self._reset_busy()
        self.log_widget.append(f"ERROR:\n{msg}")
        self._status_label.setText("Error")
        QMessageBox.critical(self, "Pipeline error", msg[:600])

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

        worker = AnalysisWorker(_do)
        self._worker = worker
        worker.progress.connect(self.progress_widget.update)
        worker.log.connect(self.log_widget.append)
        worker.result.connect(self._on_calibration_done)
        worker.error.connect(self._on_worker_error)
        self.setup_panel.set_busy(True)
        worker.start()

    def _on_calibration_done(self, pipe):
        self._worker = None
        self.pipeline = pipe
        self._save_settings()
        self.setup_panel.set_busy(False)
        shape = pipe.gain_map.shape
        self.setup_panel.show_calibration_status(f"✓ {shape[0]}×{shape[1]}")
        self.progress_widget.update(1.0, "Calibration loaded")
        self._update_state(AppState.CALIBRATED)

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

        worker = AnalysisWorker(_do)
        self._worker = worker
        worker.progress.connect(self.progress_widget.update)
        worker.log.connect(self.log_widget.append)
        worker.result.connect(lambda _: self._on_fitting_done(data_dir))
        worker.error.connect(self._on_worker_error)
        self.fitting_panel.set_busy(True)
        worker.start()

    def _on_fitting_done(self, data_dir: str):
        self._worker = None
        self._fitted_data_dir = data_dir
        self.fitting_panel.set_busy(False)
        self.progress_widget.update(1.0, "Fitting complete")
        self._update_state(AppState.FITTED)

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

        worker = AnalysisWorker(_do)
        self._worker = worker
        worker.progress.connect(self.progress_widget.update)
        worker.log.connect(self.log_widget.append)
        worker.result.connect(self._on_clustering_done)
        worker.error.connect(self._on_worker_error)
        self.postproc_panel.set_busy(True)
        worker.start()

    def _on_clustering_done(self, result):
        self._worker = None
        sm_db, sf_db = result
        self._sm_db = sm_db
        self._sf_db = sf_db
        self.postproc_panel.set_busy(False)
        self.postproc_panel.show_result(len(sm_db), len(sf_db))
        self.progress_widget.update(1.0, "Clustering complete")
        self._update_state(AppState.CLUSTERED)

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._worker_running():
            self._worker.quit()
            self._worker.wait(2000)
        logging.getLogger().removeHandler(self._log_handler)
        super().closeEvent(event)
