import enum
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QWidget, QVBoxLayout,
    QScrollArea, QLabel, QMessageBox, QApplication, QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings

from gui.panels.setup_panel import SetupPanel
from gui.panels.fitting_panel import FittingPanel
from gui.panels.postproc_panel import PostProcPanel
from gui.panels.results_panel import ResultsPanel
from gui.panels.simulation_panel import SimulationPanel, _PHOTON_LEVELS
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
        self.setWindowTitle("pyS3M")
        self.resize(1400, 900)

        self._state = AppState.IDLE
        self.pipeline = None
        self._worker: AnalysisWorker | None = None      # main pipeline worker
        self._aux_worker: AnalysisWorker | None = None  # viz-only worker (non-blocking)
        self._fitted_data_dir: str | None = None
        self._fov_data: list = []                       # [(locs_df, tif_path)] per FOV
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
        self.simulation_panel = SimulationPanel(self)

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

        ctrl_tabs = QTabWidget()
        ctrl_tabs.addTab(scroll, "Pipeline")

        _ph_posthoc = QLabel("Post-hoc analysis — coming soon.")
        _ph_posthoc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        _ph_posthoc.setStyleSheet("color: gray; font-size: 11pt;")
        ctrl_tabs.addTab(_ph_posthoc, "Post-Hoc")

        sim_scroll = QScrollArea()
        sim_scroll.setWidget(self.simulation_panel)
        sim_scroll.setWidgetResizable(True)
        sim_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ctrl_tabs.addTab(sim_scroll, "Simulation")

        ctrl_dock = QDockWidget("Controls", self)
        ctrl_dock.setWidget(ctrl_tabs)
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
        self.fitting_panel.stats_refresh_requested.connect(self._on_stats_refresh)
        self.postproc_panel.load_locs_requested.connect(self._on_load_locs)
        self.postproc_panel.cluster_requested.connect(self._on_run_clustering)
        self.postproc_panel.save_requested.connect(self._on_save_clustering)
        self.results_panel.fov_requested.connect(self._on_fov_requested)
        self.simulation_panel.simulation_requested.connect(self._on_run_simulation)

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
        self.simulation_panel.set_busy(False)
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

    def _compute_mip(self, tif_path: str, max_frames: int = 200) -> np.ndarray:
        """Max-intensity projection over the first max_frames frames of a TIFF."""
        import tifffile
        mip = None
        chunk_size = 50
        with tifffile.TiffFile(tif_path) as tif:
            n_frames = min(len(tif.pages), max_frames)
            for start in range(0, n_frames, chunk_size):
                end = min(start + chunk_size, n_frames)
                chunk = np.stack(
                    [tif.pages[i].asarray() for i in range(start, end)], axis=0
                ).astype(np.float32)
                chunk_mip = chunk.max(axis=0)
                mip = chunk_mip if mip is None else np.maximum(mip, chunk_mip)
        return mip

    def _make_locs_render_figure(
        self,
        locs,
        data_dir: str | None = None,
        tif_path: str | None = None,
        title: str = "Localisations",
    ) -> Figure:
        """Side-by-side MIP + gaussian_colour render.

        *tif_path* — specific TIFF file to use for the MIP (preferred).
        *data_dir* — folder; first TIFF found is used when *tif_path* is absent.
        Falls back to the plain scatter figure if the render cannot be produced.
        """
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).parent.parent))

        # --- MIP ---
        mip = None
        if tif_path is not None:
            try:
                mip = self._compute_mip(tif_path)
            except Exception:
                pass
        elif data_dir is not None:
            tif_files = sorted(_Path(data_dir).glob("*.tif")) + sorted(_Path(data_dir).glob("*.tiff"))
            if tif_files:
                try:
                    mip = self._compute_mip(str(tif_files[0]))
                except Exception:
                    pass

        # --- render: true RGB if all three spectral columns present, else scalar colourmap ---
        render_img = None
        spectral_cols = ("A_R", "A_G", "A_B")
        has_rgb = all(c in locs.columns for c in spectral_cols)
        if all(c in locs.columns for c in ("xc", "yc", "xc_err", "yc_err")) and len(locs) > 0:
            try:
                import render as _render
                base_cols = ["xc", "yc", "xc_err", "yc_err"]
                extra = [c for c in (("A_R", "A_G", "A_B") if has_rgb else ("A_R",))
                         if c in locs.columns]
                if "photons" in locs.columns:
                    extra.append("photons")
                subset = locs[base_cols + extra].dropna()
                if len(subset) > 0:
                    locs_rec = subset.to_records(index=False)
                    y_min = max(0.0, float(locs_rec["yc"].min()) - 1)
                    x_min = max(0.0, float(locs_rec["xc"].min()) - 1)
                    y_max = float(locs_rec["yc"].max()) + 1
                    x_max = float(locs_rec["xc"].max()) + 1
                    if has_rgb and all(c in locs_rec.dtype.names for c in spectral_cols):
                        _, _, render_img = _render.render_gaussian_RGB(
                            locs_rec, 1.0,
                            y_min, x_min, y_max, x_max,
                            min_blur_width=1.0,
                            mindensperc=1, maxdensperc=99.9,
                            densitymin=0.1,
                        )
                    else:
                        _, _, render_img = _render.render_gaussian_colour(
                            locs_rec, 1.0,
                            y_min, x_min, y_max, x_max,
                            min_blur_width=1.0,
                            cparam="A_R",
                            c_min=0.3, c_max=0.75,
                            mindensperc=1, maxdensperc=99.9,
                            densitymin=0.1,
                            cmap_string="jet",
                        )
            except Exception:
                pass

        if render_img is None:
            return self._make_scatter_figure(locs, title)

        ncols = 2 if mip is not None else 1
        fig = Figure(figsize=(6 * ncols, 5), dpi=100, layout="constrained")

        if mip is not None:
            ax_mip = fig.add_subplot(1, 2, 1)
            ax_mip.imshow(
                mip, cmap="gray", aspect="auto",
                vmin=np.percentile(mip, 1), vmax=np.percentile(mip, 99.8),
            )
            ax_mip.set_title("Max Intensity Projection")
            ax_mip.set_xticks([])
            ax_mip.set_yticks([])
            ax_render = fig.add_subplot(1, 2, 2)
        else:
            ax_render = fig.add_subplot(1, 1, 1)

        ax_render.imshow(render_img, aspect="auto", origin="upper")
        ax_render.set_title(f"{title}  ({len(locs):,})")
        ax_render.set_xticks([])
        ax_render.set_yticks([])

        return fig

    def _make_scatter_figure(self, locs, title: str = "Localisations") -> Figure:
        pixel_size_nm = self.pipeline.pixel_size * 1000
        x = locs["xc"].values * pixel_size_nm
        y = locs["yc"].values * pixel_size_nm

        fig = Figure(figsize=(6, 6), dpi=100, layout="constrained")
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
        return fig

    def _make_stats_figure(
        self,
        locs,
        photon_range=(100, 1_000_000),
        sm_locs=None,
    ) -> Figure:
        # After H5 write, A_R/G/B are normalised fractions; the true total is in "photons".
        # Use "photons" if present; fall back to raw sum only for un-normalised data.
        # When sm_locs is provided (post-clustering), render a 2×3 grid comparing
        # per-localisation (locs / sf_db) with per-molecule (sm_locs / sm_db) stats.
        col_colors = [("A_R", "tomato"), ("A_G", "mediumseagreen"), ("A_B", "cornflowerblue")]
        min_ph, max_ph = photon_range

        def _extract(df):
            """Return (filtered_df, total_photons_array) applying photon_range mask."""
            has_photons = "photons" in df.columns
            spectral_cols = ("A_R", "A_G", "A_B")
            has_spectral = has_photons or all(c in df.columns for c in spectral_cols)
            if not has_spectral:
                return None, None
            total_all = (
                df["photons"].values if has_photons
                else df["A_R"].values + df["A_G"].values + df["A_B"].values
            )
            mask = np.isfinite(total_all) & (total_all >= min_ph) & (total_all <= max_ph)
            return df[mask], total_all[mask]

        two_rows = sm_locs is not None and not sm_locs.empty

        if two_rows:
            fig = Figure(figsize=(13, 6), dpi=100, layout="constrained")
            rows = [
                ("Localisations", locs),
                ("Molecules",     sm_locs),
            ]
            for row_idx, (row_label, df) in enumerate(rows):
                filtered, total = _extract(df)
                axes = [fig.add_subplot(2, 4, row_idx * 4 + col + 1) for col in range(4)]
                if filtered is None:
                    axes[0].text(0.5, 0.5, "No data", ha="center", va="center",
                                 transform=axes[0].transAxes, color="gray")
                    continue
                for ax, (col, color) in zip(axes[:3], col_colors):
                    ch = filtered[col].values if col in filtered.columns else np.array([])
                    ch = ch[np.isfinite(ch)]
                    ax.hist(ch, bins=60, color=color, alpha=0.8, range=(0, 1))
                    ax.set_xlabel(f"{col} fraction")
                    ax.set_ylabel(f"{row_label}\nCount" if col == "A_R" else "Count")
                axes[3].hist(total, bins=60, color="slategray", alpha=0.8)
                axes[3].set_xlabel("Total photons")
                axes[3].set_ylabel("Count")
                axes[3].set_title(f"n = {len(filtered):,}")
        else:
            filtered, total = _extract(locs)
            fig = Figure(figsize=(10, 3), dpi=100, layout="constrained")
            if filtered is None:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, "No photon data in file", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            else:
                axes = fig.subplots(1, 4)
                for ax, (col, color) in zip(axes[:3], col_colors):
                    ch = filtered[col].values if col in filtered.columns else np.array([])
                    ch = ch[np.isfinite(ch)]
                    ax.hist(ch, bins=60, color=color, alpha=0.8, range=(0, 1))
                    ax.set_xlabel(f"{col} fraction")
                    ax.set_ylabel("Count")
                    ax.set_title(col)
                axes[3].hist(total, bins=60, color="slategray", alpha=0.8)
                axes[3].set_xlabel("Total photons")
                axes[3].set_ylabel("Count")
                axes[3].set_title(f"Total  (n={len(filtered):,})")
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

    def _on_run_fitting(self, data_dir: str, mode: str, fitting_config, extra_kwargs: dict):
        if self._worker_running() or self.pipeline is None:
            return

        photon_range = self.fitting_panel.photon_range

        def _do():
            self.pipeline.config.progress_callback = lambda f, m: worker.progress.emit(f, m)
            self.pipeline.config.logging_callback = lambda m: worker.log.emit(m)
            # fov_data: list of (locs_df, tif_path_or_None), one entry per TIFF/H5
            fov_data = self.pipeline.fit(
                Path(data_dir), mode=mode, fitting_config=fitting_config, **extra_kwargs
            )
            if not fov_data:
                return data_dir, [], None, None
            n = len(fov_data)
            first_df, first_tif = fov_data[0]
            title = f"FOV 1 / {n}" if n > 1 else "Localisations"
            locs_fig = self._make_locs_render_figure(first_df, tif_path=first_tif, title=title)
            all_locs = pd.concat([df for df, _ in fov_data], ignore_index=True)
            stats_fig = self._make_stats_figure(all_locs, photon_range=photon_range)
            return data_dir, fov_data, locs_fig, stats_fig

        worker = self._start_worker(_do)
        worker.result.connect(self._on_fitting_done)
        self.fitting_panel.set_fit_busy(True)
        worker.start()

    def _on_fitting_done(self, result):
        data_dir, fov_data, locs_fig, stats_fig = result
        self._fitted_data_dir = data_dir
        self._fov_data = fov_data
        self.fitting_panel.set_fit_busy(False)
        self.progress_widget.update(1.0, "Fitting complete")
        self._update_state(AppState.FITTED)
        self.results_panel.set_fov_count(len(fov_data))
        if locs_fig is not None:
            self.results_panel.set_localisations_figure(locs_fig)
        if stats_fig is not None:
            self.results_panel.set_stats_figure(stats_fig)

    # ------------------------------------------------------------------
    # FOV navigation
    # ------------------------------------------------------------------

    def _on_fov_requested(self, idx: int):
        if not self._fov_data or idx >= len(self._fov_data):
            return
        if self._aux_worker is not None and self._aux_worker.isRunning():
            return
        locs_df, tif_path = self._fov_data[idx]
        n = len(self._fov_data)
        title = f"FOV {idx + 1} / {n}"

        def _do():
            return self._make_locs_render_figure(locs_df, tif_path=tif_path, title=title)

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(
            lambda fig: self.results_panel.set_localisations_figure(fig) if fig is not None else None
        )
        aux.error.connect(
            lambda msg: self.log_widget.append(f"Warning: FOV render failed:\n{msg}")
        )
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    # ------------------------------------------------------------------
    def _on_aux_worker_finished(self, w: AnalysisWorker):
        if self._aux_worker is w:
            self._aux_worker = None

    def _on_stats_refresh(self, photon_range: tuple):
        if self._fitted_data_dir is None or self.pipeline is None:
            return
        if self._aux_worker is not None and self._aux_worker.isRunning():
            return

        def _do():
            locs = self.pipeline.load_localisations(self._fitted_data_dir)
            if locs.empty:
                return None
            return self._make_stats_figure(locs, photon_range=photon_range)

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(lambda fig: self.results_panel.set_stats_figure(fig) if fig is not None else None)
        aux.error.connect(lambda msg: self.log_widget.append(f"Stats refresh failed:\n{msg}"))
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    # ------------------------------------------------------------------
    # Load existing H5 (skip fitting)
    # ------------------------------------------------------------------

    def _on_load_locs(self, h5_path: str):
        if self._worker_running():
            return

        h5 = Path(h5_path)
        folder = h5.parent
        filename = h5.name
        photon_range = self.fitting_panel.photon_range

        if self.pipeline is None:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from AnalysisPipeline import AnalysisPipeline
            from Constants import AnalysisConfig
            self.pipeline = AnalysisPipeline(
                config=AnalysisConfig(display=False)
            )
            logger.info("Created default pipeline for H5 load (no calibration)")

        def _do():
            locs = self.pipeline.load_localisations(folder, pattern=filename)
            if locs.empty:
                raise ValueError(f"No localisations found in {h5}")
            locs_fig = self._make_locs_render_figure(locs, data_dir=str(folder), title="Localisations (loaded)")
            stats = self._make_stats_figure(locs, photon_range=photon_range)
            return str(folder), locs_fig, stats

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(self._on_h5_loaded)
        aux.error.connect(lambda msg: self.log_widget.append(f"Load H5 failed: {msg[:300]}"))
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    def _on_h5_loaded(self, result):
        folder, scatter_fig, stats_fig = result
        self._fitted_data_dir = folder
        self._fov_data = []
        self.results_panel.set_fov_count(1)
        self.results_panel.set_localisations_figure(scatter_fig)
        self.results_panel.set_stats_figure(stats_fig)
        self.progress_widget.update(1.0, "Localisations loaded")
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
            self._refresh_render_figure(sm_db, title="Single molecules", stats_locs=sf_db, stats_sm=sm_db)

    def _refresh_render_figure(self, locs, title: str = "Localisations", stats_locs=None, stats_sm=None):
        """Launch an aux worker to build and display the render (and optionally stats) figure."""
        data_dir = self._fitted_data_dir
        photon_range = self.fitting_panel.photon_range

        def _do():
            locs_fig = self._make_locs_render_figure(locs, data_dir=data_dir, title=title)
            stats_fig = (
                self._make_stats_figure(stats_locs, photon_range=photon_range, sm_locs=stats_sm)
                if stats_locs is not None and not stats_locs.empty
                else None
            )
            return locs_fig, stats_fig

        def _on_result(result):
            locs_fig, stats_fig = result
            if locs_fig is not None:
                self.results_panel.set_localisations_figure(locs_fig)
            if stats_fig is not None:
                self.results_panel.set_stats_figure(stats_fig)

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(_on_result)
        aux.error.connect(
            lambda msg: self.log_widget.append(f"Warning: render failed:\n{msg}")
        )
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    def _on_save_clustering(self):
        if self._sm_db is None or self._sm_db.empty:
            QMessageBox.warning(self, "No data", "No clustering results to save.")
            return

        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save clustering results", "", "HDF5 files (*.h5)"
        )
        if not path:
            return
        if not path.endswith(".h5"):
            path += ".h5"

        try:
            self._sm_db.to_hdf(path, key="sm_db", mode="w")
            self._sf_db.to_hdf(path, key="sf_db", mode="a")
            self.log_widget.append(
                f"Saved {len(self._sm_db)} molecules / {len(self._sf_db)} localisations → {path}"
            )
            self.progress_widget.update(1.0, "Results saved")
        except Exception:
            import traceback as _tb
            self.log_widget.append(f"Save failed:\n{_tb.format_exc()}")
            QMessageBox.critical(self, "Save failed", "Could not write HDF5 file — see log.")

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _on_run_simulation(
        self,
        dye: str,
        filters: list,
        bg_photons: float,
        na: float,
        pixel_size_nm: float,
        read_noise_e: float,
        peak_qe: float,
        n_rep: int,
    ):
        if self._aux_worker is not None and self._aux_worker.isRunning():
            return

        self.simulation_panel.set_busy(True)
        self.progress_widget.update(0.0, "Building simulation…")

        def _do():
            return self._make_simulation_figure(
                dye, filters, bg_photons, na, pixel_size_nm, read_noise_e, peak_qe, n_rep
            )

        aux = AnalysisWorker(_do)
        self._aux_worker = aux
        aux.result.connect(self._on_simulation_done)
        aux.error.connect(
            lambda msg: (
                self.log_widget.append(f"Simulation failed:\n{msg}"),
                self.simulation_panel.set_busy(False),
                self.progress_widget.update(0.0, "Simulation failed"),
            )
        )
        aux.finished.connect(lambda w=aux: self._on_aux_worker_finished(w))
        aux.start()

    def _on_simulation_done(self, fig):
        self.simulation_panel.set_busy(False)
        self.progress_widget.update(1.0, "Simulation ready")
        if fig is not None:
            self.results_panel.set_simulation_figure(fig)

    def _make_simulation_figure(
        self,
        dye: str,
        filters: list,
        bg_photons: float,
        na: float,
        pixel_size_nm: float,
        read_noise_e: float,
        peak_qe: float,
        n_rep: int,
    ) -> Figure:
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).parent.parent))

        import numpy as np
        import SpectralFunctions
        from PSFFunctions import PSF_Functions

        S_F = SpectralFunctions.Spectral_Funcs()
        R_qe, G_qe, B_qe, wl = S_F.getpixelefficiency()

        # Scale Bayer QE curves so the peak across all channels equals peak_qe.
        # This changes the relative colour fractions (the filter+QE overlap per channel)
        # while keeping the peak QE at the user-specified value.
        raw_peak = max(R_qe.max(), G_qe.max(), B_qe.max())
        scale = peak_qe / raw_peak if raw_peak > 0 else 1.0
        R_sc, G_sc, B_sc = R_qe * scale, G_qe * scale, B_qe * scale

        # pixel_QYs convention: (n_channels, n_wavelengths) in BGR order
        pixel_QYs = np.vstack([B_sc, G_sc, R_sc])

        # normalized=True → per-channel colour fractions (sum to 1).
        # The Bayer-geometry weighting means the QE curves can't be summed
        # directly to get a total efficiency; instead we use peak_qe as the
        # overall detection efficiency and fracs for the colour split.
        avg_wl, fracs = S_F.get_pixel_fractions_dye_and_filters(
            dyes=[dye],
            filters=filters if filters else None,
            wavelength=wl,
            pixel_QYs=pixel_QYs,
            normalized=True,
        )
        # fracs shape (3,): [B_frac, G_frac, R_frac], sum = 1
        b_frac, g_frac, r_frac = fracs
        # Total photoelectrons = n_photons × peak_qe; split by fracs
        b_eff, g_eff, r_eff = b_frac * peak_qe, g_frac * peak_qe, r_frac * peak_qe

        # PSF sigma from dye emission wavelength (avg_wl in nm) and NA
        sigma_nm = PSF_Functions.sigma_PSF(float(avg_wl) * 1e-9, na) * 1e9  # m → nm
        sigma_px = sigma_nm / pixel_size_nm

        # Normalised Gaussian PSF kernel
        patch_size = 13
        c = patch_size // 2
        yy, xx = np.mgrid[:patch_size, :patch_size]
        psf = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma_px ** 2))
        psf /= psf.sum()

        # RGGB Bayer masks over the patch
        r_mask = (yy % 2 == 0) & (xx % 2 == 0)
        g_mask = ((yy % 2 == 0) & (xx % 2 == 1)) | ((yy % 2 == 1) & (xx % 2 == 0))
        b_mask = (yy % 2 == 1) & (xx % 2 == 1)

        rng = np.random.default_rng(42)
        photon_levels = _PHOTON_LEVELS
        n_rows = len(photon_levels)

        fig = Figure(
            figsize=(1.8 * n_rep + 1.0, 1.8 * n_rows + 0.7),
            dpi=100,
            layout="constrained",
        )
        fig.patch.set_facecolor("#1a1a1a")
        axes = fig.subplots(n_rows, n_rep, squeeze=False)

        for row_idx, n_ph in enumerate(photon_levels):
            # Generate all replicates for this photon level first so we can
            # compute a shared percentile scale across the row.
            row_patches = []
            for col_idx in range(n_rep):
                bayer = np.zeros((patch_size, patch_size))
                for mask, eff in ((r_mask, r_eff), (g_mask, g_eff), (b_mask, b_eff)):
                    n_px = int(mask.sum())
                    sig = rng.poisson(n_ph * eff * psf[mask] + bg_photons).astype(float)
                    sig += rng.normal(0.0, read_noise_e, size=n_px)
                    bayer[mask] = sig - bg_photons
                row_patches.append(bayer)

            # Per-row percentile stretch so every photon level is clearly visible
            all_vals = np.concatenate([p.ravel() for p in row_patches])
            vmin = float(np.percentile(all_vals, 0.1))
            vmax = float(np.percentile(all_vals, 99.9))
            if vmax <= vmin:
                vmax = vmin + 1.0

            for col_idx, bayer in enumerate(row_patches):
                ax = axes[row_idx, col_idx]
                ax.set_facecolor("#1a1a1a")

                ax.imshow(
                    bayer,
                    cmap="gray",
                    origin="upper",
                    interpolation="nearest",
                    aspect="equal",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333333")
                    spine.set_linewidth(0.5)

                if col_idx == 0:
                    ax.set_ylabel(
                        f"{n_ph:,} ph",
                        color="white",
                        fontsize=7,
                        rotation=0,
                        labelpad=34,
                        va="center",
                    )

        filter_label = ", ".join(f.split("-", 1)[-1] for f in filters) if filters else "no filter"
        fig.suptitle(
            (
                f"{dye}  |  {filter_label}  |  "
                f"λ={float(avg_wl):.0f} nm  σ={sigma_nm:.0f} nm  "
                f"QE={peak_qe:.2f}  BG={bg_photons:.0f} ph/px  "
                f"RN={read_noise_e:.1f} e⁻"
            ),
            color="white",
            fontsize=8,
        )
        return fig

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
