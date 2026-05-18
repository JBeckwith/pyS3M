# pyS3M GUI — Design & Implementation Plan

**Date:** 2026-05-08
**Status:** Planning
**Target:** PyQt6 desktop application wrapping `AnalysisPipeline`

---

## 1. Goals

Provide a point-and-click interface for the full SMLM analysis pipeline without requiring users to write Python. All scientific logic stays in `src/`; the GUI is a thin shell that:

- Selects folders and parameters via forms
- Runs analysis on a background thread with live progress
- Embeds matplotlib figures in-window
- Streams log output to a scrollable panel

Non-goals for MVP: 3D visualisation, batch multi-folder scheduling, cloud/remote execution.

---

## 2. Framework Decision

**PyQt6** + `matplotlib.backends.backend_qtagg`

Rationale:
- `PlottingBase.py` already produces `(fig, ax)` pairs — embedding via `FigureCanvasQTAgg` is one line
- `AnalysisConfig.progress_callback` and `logging_callback` slot directly into Qt signals
- `QThread` keeps the UI responsive during multi-minute fitting runs
- PySide6 is a drop-in alternative if licensing is a concern (identical API)

Install: `pip install PyQt6 pyqt6-qt6`

---

## 3. Architecture

```
GUI layer (src/gui/)
│
├── MainWindow          — dock manager, menu bar, status bar
├── panels/
│   ├── SetupPanel      — camera, calibration dir, pixel size
│   ├── FittingPanel    — mode selector + FittingConfig form
│   ├── PostProcPanel   — FilteringCriteria + ClusteringConfig form
│   ├── DriftPanel      — method selector + undrift params
│   ├── SpectralPanel   — Nile Red options
│   └── ResultsPanel    — FigureCanvasQTAgg + toolbar
├── widgets/
│   ├── LogWidget       — QPlainTextEdit fed by logging_callback
│   ├── ProgressWidget  — QProgressBar fed by progress_callback
│   └── FolderPicker    — QLineEdit + browse button
└── worker.py           — QThread subclass wrapping AnalysisPipeline calls
```

**Model:** `AnalysisPipeline` (already in `src/AnalysisPipeline.py`). The GUI never calls `src/` code directly — everything goes through the pipeline instance.

**Threading rule:** Any call that could take > 0.5 s runs in `AnalysisWorker` (QThread). Results and exceptions are posted back to the main thread via Qt signals. The UI thread only updates widgets.

---

## 4. State Machine

The GUI enforces a linear pipeline by enabling/disabling buttons:

```
IDLE
  → [Load Calibration] → CALIBRATED
      → [Preview] (optional, stays CALIBRATED)
      → [Run Fitting] → FITTED
          → [Filter & Cluster] → CLUSTERED
              → [Undrift] (optional) → DRIFT_CORRECTED
              → [Spectral Analysis] (optional) → SPECTRAL_DONE
```

State is stored as an enum on `MainWindow`. Each state transition emits a `state_changed` signal that updates button enable/disable across all panels.

---

## 5. Panel Specifications

### 5.1 SetupPanel

Controls which map to `AnalysisPipeline.__init__` and `load_calibration` / `calibrate`:

| Widget | Type | Maps to |
|---|---|---|
| Camera | `QComboBox` | `camera=` ("ximea", "zwo") |
| Pixel size | `QDoubleSpinBox` (µm) | `pixel_size=` (auto-filled from camera; editable) |
| Calibration dir | `FolderPicker` | `pipe.load_calibration(path)` |
| Mode | `QRadioButton` | "Load pre-computed files" / "Compute from frames" |
| Load Calibration | `QPushButton` | triggers worker |

On success: shows ✓ and the shape of the loaded gain map; transitions to CALIBRATED.

### 5.2 FittingPanel

Controls which map to `FittingConfig` and `AnalysisPipeline.fit`:

| Widget | Type | Maps to |
|---|---|---|
| Data folder | `FolderPicker` | `image_folder` |
| Mode | `QComboBox` | `mode=` ("smlm", "fret", "qd", "tracking", "imaging") |
| PFA | `QDoubleSpinBox` (sci notation) | `FittingConfig.pfa` |
| ROI size | `QSpinBox` (px) | `FittingConfig.ROI_size` |
| Peak wavelength | `QDoubleSpinBox` (µm) | `FittingConfig.peak_wavelength` |
| NA | `QDoubleSpinBox` | `FittingConfig.NA` |
| Detection sigma | `QDoubleSpinBox` | `FittingConfig.sigma` |
| Fraction true | `QDoubleSpinBox` | `FittingConfig.fraction_true` |
| Variance-aware demosaic | `QCheckBox` | `FittingConfig.use_variance_aware_demosaic` |
| Preview (single frame) | `QPushButton` | `pipe.sr.example_spots_singleframe(...)` |
| Run Fitting | `QPushButton` | `pipe.fit(...)` |

Mode-specific extras (n_frames_sum, use_elliptical, etc.) appear/disappear in a `QStackedWidget` below the shared controls when the mode selector changes.

### 5.3 PostProcPanel

Controls which map to `FilteringCriteria` and `ClusteringConfig`:

| Widget | Type | Maps to |
|---|---|---|
| Min photons | `QSpinBox` | `FilteringCriteria.min_photons` |
| Max colour error | `QDoubleSpinBox` | `FilteringCriteria.max_colour_error` |
| Max localisation error | `QDoubleSpinBox` | `FilteringCriteria.max_localisation_error` |
| Start frame | `QSpinBox` | `ClusteringConfig.start_frame` |
| Clustering method | `QComboBox` | `ClusteringConfig.clustering_method` |
| Min cluster size | `QSpinBox` | `ClusteringConfig.min_cluster_size` |
| Filter & Cluster | `QPushButton` | `pipe.filter_and_cluster(...)` |

Results label: "N single molecules from M localisations".

### 5.4 DriftPanel

| Widget | Type | Maps to |
|---|---|---|
| Method | `QComboBox` | `method=` ("aim", "fiducial", "auto") |
| Segmentation frames | `QSpinBox` | `segmentation=` kwarg |
| Undrift | `QPushButton` | `pipe.undrift(...)` |

### 5.5 SpectralPanel (Nile Red)

| Widget | Type | Maps to |
|---|---|---|
| Wavelength bounds | two `QDoubleSpinBox` | `wavelength_bounds=(lo, hi)` |
| Run Spectral Fit | `QPushButton` | `NileRed_Functions.fit_wavelengths_pixelated(...)` |

### 5.6 ResultsPanel

A `QTabWidget` with one tab per result type. Each tab holds a `FigureCanvasQTAgg` + `NavigationToolbar2QT`.

| Tab | Figure source |
|---|---|
| Preview | `example_spots_singleframe` fig |
| Localisations | rendered super-resolution image via `render.render()` |
| Single molecules | ternary scatter / histogram via `PlottingBase` |
| Drift | drift correction summary plot |
| Wavelengths | wavelength histogram |

Tabs are hidden until the relevant step has been run (state machine controls `setVisible`).

---

## 6. Worker Thread

```python
# src/gui/worker.py (sketch)
from PyQt6.QtCore import QThread, pyqtSignal

class AnalysisWorker(QThread):
    progress = pyqtSignal(float, str)   # fraction, message
    log      = pyqtSignal(str)          # log line
    result   = pyqtSignal(object)       # step-specific return value
    error    = pyqtSignal(str)          # exception message

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            out = self._fn(*self._args, **self._kwargs)
            self.result.emit(out)
        except Exception as e:
            self.error.emit(str(e))
```

`AnalysisConfig` is constructed with:
```python
cfg = AnalysisConfig(
    display=False,
    progress_callback=lambda f, msg: worker.progress.emit(f, msg),
    logging_callback=lambda msg: worker.log.emit(msg),
)
```

The worker is re-created for each step (don't reuse QThread instances).

---

## 7. LogWidget

`QPlainTextEdit` in read-only mode, appended to via the `logging_callback`. Also install a `logging.Handler` subclass that posts to the same widget so that `logger.info(...)` calls from `src/` appear automatically:

```python
class QtLogHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def emit(self, record):
        self._signal.emit(self.format(record))
```

Attach to the root logger at app startup so all `src/` modules feed into the panel.

---

## 8. Settings Persistence

Use `QSettings` (cross-platform INI storage) to remember:
- Last calibration directory
- Last data directory
- Camera selection
- All `FittingConfig` field values

Restore on startup; save on each successful fitting run.

---

## 9. File Layout

```
src/
  gui/
    __init__.py
    app.py              — QApplication entry point
    main_window.py      — MainWindow, dock layout, state machine
    worker.py           — AnalysisWorker QThread
    panels/
      __init__.py
      setup_panel.py
      fitting_panel.py
      postproc_panel.py
      drift_panel.py
      spectral_panel.py
      results_panel.py
    widgets/
      __init__.py
      folder_picker.py
      log_widget.py
      progress_widget.py
```

Entry point: `python -m pyS3M.gui` or a top-level `run_gui.py` script.

---

## 10. MVP Scope

**In for MVP:**
- SetupPanel (load pre-computed calibration only)
- FittingPanel (smlm and imaging modes; mode-specific extras as stretch)
- PostProcPanel (HDBSCAN only; other methods as stretch)
- ResultsPanel (Preview + Localisations tabs)
- LogWidget + ProgressWidget
- QSettings persistence for directories and FittingConfig

**Out for MVP (post-MVP):**
- Calibrate-from-frames mode (SetupPanel)
- DriftPanel
- SpectralPanel
- FRET / QD / tracking modes in FittingPanel
- DBSCAN / LINKED clustering in PostProcPanel
- Batch multi-folder scheduling

---

## 11. Estimated Effort

| Component | Effort |
|---|---|
| Project scaffold, QApplication, MainWindow skeleton | 1 h |
| SetupPanel + worker + state machine | 2 h |
| FittingPanel (smlm + imaging) + Preview | 3 h |
| PostProcPanel (HDBSCAN) | 2 h |
| ResultsPanel (Preview + Localisations tabs) | 2 h |
| LogWidget + QtLogHandler + ProgressWidget | 1 h |
| QSettings persistence | 0.5 h |
| Polish, error handling, integration test | 2 h |
| **MVP total** | **~13 h** |

---

## 12. Dependencies

Already available in the virtualenv: `numpy`, `matplotlib`, `pandas`, `scipy`.

Additional: `PyQt6` (`pip install PyQt6`). No other new dependencies needed for MVP.
