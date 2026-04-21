# pyBayerSMLM — Refactoring Analysis & Plan

**Date:** 2026-04-10
**Scope:** Full `src/` codebase (~40 500 lines across 32 files)
**Goal:** Compact, production-ready code that can be driven by a GUI without re-architecture

---

## 1. File Inventory

| File | Lines | Primary responsibility |
|------|------:|----------------------|
| SM_extractionfunctions.py | 5 301 | SM clustering (HDBSCAN, DBSCAN, linked), filtering, photon stats |
| PlottingBase.py | 4 136 | Master plotting classes (publication, analysis, ternary, datashader) |
| Multicolour_Simulation_Functions.py | 3 743 | Simulation, bootstrap, fitting delegation, file I/O |
| DriftCorrectionFunctions.py | 3 494 | Drift correction (AIM, fiducial, auto); ABC + strategy classes |
| SR_Functions.py | 2 553 | Super-resolution pipeline orchestrator |
| postprocess.py | 1 976 | NENA, pair correlation, index blocks, linking utilities |
| NileRedFunctions.py | 1 915 | Nile Red spectral wavelength extraction |
| DiffusionSimulation.py | 1 883 | 2D Langevin diffusion, MSD, binding kinetics |
| ImageAnalysisFunctions.py | 1 777 | Gaussian fitting (9 strategies, polymorphic) |
| SpectralFunctions.py | 1 526 | Spectral models, filter data, pixel QYs |
| IOFunctions.py | 1 337 | HDF5/TIFF I/O, photon normalisation |
| FiducialDetection.py | 1 263 | Fiducial bead detection pipeline + DriftPlotter |
| SpotDetectionFunctions.py | 1 081 | Spot detection (matched filter, CA-CFAR) |
| render.py | 848 | Histogram rendering, Gaussian convolution, smoothing |
| PSFFunctions.py | 774 | Wavelength-dependent PSF models |
| AIMAlgorithm.py | 770 | AIM drift algorithm |
| gaussoptfuncs.py | 652 | Gaussian fitting model variants (8 models) |
| lib.py | 574 | Record-array utilities, polygon/rectangle selection |
| CoordinateProcessing.py | 527 | Segmentation, coordinate validation |
| sCMOSFunctions.py | 520 | sCMOS calibration, demosaicing, variance weighting |
| CalibrationFunctions.py | 472 | Camera calibration |
| ImportManager.py | 439 | Graceful optional-dependency loading |
| FRCFunctions.py | 433 | Fourier Ring Correlation |
| localise.py | 388 | Spot identification, initial fitting |
| ProgressUtils.py | 399 | Progress-bar context managers |
| StepDetector.py | 384 | Change-point detection |
| LoggingFramework.py | 306 | Structured logging with progress tracking |
| LinkingFunctions.py | 302 | Post-hoc linking of blinking detections |
| MaskFunctions.py | 270 | Bayer mask generation |
| HelperFunctions.py | 253 | Coordinate maths, small utilities |
| Constants.py | 191 | Global constants (DriftConstants, FilteringConstants, ResultColumns) |
| CameraDefaults.py | 67 | Camera config registry (ximea, zwo) |

---

## 2. Duplication

---

### ~~2b. Scattered HDF5 I/O~~ — DONE

All raw `pd.read_hdf` / `df.to_hdf` calls replaced with `self.io.read_h5_database` / `self.io.write_h5_database` across all callers. See Completed table.

---

## 3. Structural Problems

### 3a. `SM_extractionfunctions.py` — 5 300-line god file

Mixes five concerns that should be separate:
- Schema compatibility / DataFrame loading
- Quality filtering (χ², localisation error, colour error, sigma bounds)
- Clustering (three algorithms: HDBSCAN, DBSCAN, linked)
- Photon accumulation statistics
- Plotting and figure export (19 direct `matplotlib` calls, 7 `plt.show()` calls)

Method signatures carry 10–15 positional parameters representing filtering criteria;
these values are constant within a session and are re-passed on every call.

**Proposed split:**
```
SM_extractionfunctions.py  (keep, ~1 500 lines — orchestration only)
filtering.py               (~400 lines — FilteringCriteria dataclass + apply_filters())
clustering/
    __init__.py
    hdbscan_clusterer.py   (~500 lines)
    dbscan_clusterer.py    (~400 lines)
    linked_clusterer.py    (~400 lines)
```

---

### 3b. `DriftCorrectionFunctions.py` — 3 494-line large file

Good ABC pattern already exists (three correctors: AIM, Fiducial, Auto), but all strategies
plus their rendering and validation logic live in one file, making individual strategies hard
to test or extend. RCC has been removed (see Completed table).

**Proposed split:**
```
drift_correction/
    __init__.py            (public imports only)
    _base.py               (~300 lines — DriftMethod enum, DriftParameters, DriftResult, ABC)
    aim.py                 (~500 lines — AIMDriftCorrector, coordinate processing)
    fiducial.py            (~700 lines — FiducialDriftCorrector + DriftPlotter, detection logic)
    auto.py                (~200 lines — AutoDriftCorrector, strategy selection)
```

Each file stays under 700 lines, clear single responsibility.

---

### ~~3c. Magic numbers scattered across the codebase~~ — DONE

`DriftConstants` and `FilteringConstants` added to `Constants.py`; pixel sizes derived
from `CameraDefaults` so they stay in sync. `DriftParameters` gains `pixel_size_nm`
(defaults to Ximea); `intersect_d`/`roi_r` computed from nm constants in `__post_init__`.
All `pixel_size=69` / `min_sigma=75/69` / `max_colour_error=0.15` defaults across 8 files
replaced with constant references. See Completed table.

---

## 4. Parameter-Group Dataclasses

Several functions already use dataclasses well (`DriftParameters`, `SimulationConfig`,
`CameraParameters`).  The following groups are still passed as long positional lists and
should be converted:

| Function group | New dataclass | Home file |
|---|---|---|
| `extract_single_molecules_*` filtering args (10 params) | `FilteringCriteria` | `filtering.py` |
| `extract_single_molecules_*` clustering args | `ClusteringConfig` | `clustering/__init__.py` |
| `render.*` rendering args (8 params) | `RenderingConfig` | `render.py` |
| Analysis output path + flags | `AnalysisConfig` | `Constants.py` |

---

## 5. Plotting Architecture

`PlottingBase.py` is the canonical plotting entry point and is well-designed.  The problem
is that not all files use it:

| File | Plotting approach | Problem |
|---|---|---|
| `SM_extractionfunctions.py` | Direct `import matplotlib.pyplot as plt` | 7 `plt.show()` calls — GUI-blocking |
| `DriftCorrectionFunctions.py` | Mixed: imports `DriftPlotter` but also direct `plt` | 3 `plt.show()` calls |
| `postprocess.py` | Fallback matplotlib with optional plotter | 1 `plt.show()` call |
| `Multicolour_Simulation_Functions.py` | Direct `plt` | 4 `plt.show()` calls |
| `FiducialDetection.py` | ✓ DriftPlotter(AnalysisPlotter) subclass | Correct |
| `SR_Functions.py` | ✓ PublicationPlotter instance | Correct |

Total: **15 `plt.show()` calls across 4 files** that will block a GUI event loop.

**Fix:** Every function that produces a figure should:
1. Accept an optional `plotter` argument (defaults to creating its own `AnalysisPlotter`).
2. Return `(fig, ax)` always.
3. Call `plotter.save_or_show(fig, display=config.display)` rather than `plt.show()`.

---

## 6. GUI Readiness

The codebase is **not yet GUI-ready**, but the gap is smaller than it looks.

### What's already good
- `LoggingFramework.py` — structured logging with callback hooks; a GUI progress bar can
  subscribe directly.
- `ProgressUtils.py` — context-manager progress bars; could be wrapped by a Qt widget.
- `ImportManager.py` — graceful optional-dependency loading; no hard crashes on missing libs.
- `PlottingBase.save_or_show()` — already accepts `display=False` and a save path.
- `CameraDefaults.py` — camera registry already GUI-friendly (dropdown data source).

### What blocks a GUI

| Blocker | Location | Fix |
|---|---|---|
| `plt.show()` embedded in processing | 4 files, 15 calls | Config-driven display (see §5) |
| Progress via `print()` | ~100 call sites | Route through `LoggingFramework` |
| No clean result objects | Most analysis functions return recarrays | Add thin result dataclasses |
| File paths as bare strings | Throughout | `pathlib.Path` + `AnalysisConfig` |
| No headless mode | Everywhere | `matplotlib.use('Agg')` guard in `AnalysisConfig` |

### Proposed `AnalysisConfig` dataclass

```python
@dataclass
class AnalysisConfig:
    """Passed into any analysis function to control I/O and display behaviour."""
    output_dir: Optional[Path]      = None
    display: bool                   = True   # False for GUI/server
    save_figures: bool              = False
    figure_format: str              = 'svg'
    dpi: int                        = 300
    progress_callback: Optional[Callable[[float, str], None]] = None
    logging_callback: Optional[Callable[[str], None]]         = None
```

Passing `AnalysisConfig(display=False, save_figures=True, output_dir=Path('results/'))` to
any analysis function should produce zero interactive windows and write all outputs to disk.
The GUI can pass its own callbacks for progress bars and log panels.

### Recommended GUI framework: PyQt6 / PySide6 + matplotlib Qt backend

The existing class structure maps cleanly onto a Qt model:

| pyBayerSMLM class | Qt role |
|---|---|
| `AnalysisConfig` | Settings/preferences store |
| `SR_Functions` | Worker thread (QRunnable) |
| `LoggingFramework` → callback | Qt signal → log panel |
| `ProgressUtils` → callback | Qt signal → QProgressBar |
| `PlottingBase` figures | Embedded `FigureCanvasQTAgg` |
| `CameraDefaults` | Dropdown model for camera selection |
| `FittingStrategy` enum | Dropdown / radio buttons |

---

## 7. Prioritised Refactoring Plan

### Tier 1 — Unblocks GUI (do first, no algorithm changes)

| # | Action | Files touched | Effort |
|---|---|---|---|
| ~~1.1~~ | ~~Replace all `plt.show()` with config-driven display~~ | ~~done~~ | ~~done~~ |
| ~~1.2~~ | ~~Add `DriftConstants` + `FilteringConstants` to `Constants.py`~~ | ~~done~~ | ~~done~~ |
| ~~1.3~~ | ~~Add `AnalysisConfig` dataclass; thread into `SR_Functions` as proof-of-concept~~ | ~~done~~ | ~~done~~ |

### Tier 2 — Reduces file size and complexity

| # | Action | Files touched | Effort |
|---|---|---|---|
| ~~2.0~~ | ~~Remove `RCCDriftCorrector` + all RCC branches from `DriftCorrectionFunctions.py`~~ | ~~done~~ | ~~done~~ |
| ~~2.1~~ | ~~Split `DriftCorrectionFunctions.py` into `drift_correction/` submodule~~ | ~~done~~ | ~~done~~ |
| ~~2.2~~ | ~~Extract `FilteringCriteria` dataclass; reduce `extract_single_molecules_*` signatures~~ | ~~done~~ | ~~done~~ |
| 2.3 | Split clustering logic out of `SM_extractionfunctions.py` | SM_extract + new files | 2 days |
| ~~2.4~~ | ~~Consolidate HDF5 I/O via `IOFunctions`~~ | ~~done~~ | ~~done~~ |

### Tier 3 — Code quality and long-term maintainability

| # | Action | Files touched | Effort |
|---|---|---|---|
| 3.1 | `RenderingConfig` dataclass in `render.py` | render.py + callers | 0.5 day |
| 3.2 | Replace `print()` progress with `LoggingFramework` throughout | ~8 files | 1 day |
| 3.3 | Comprehensive type hints on public methods | All | 3 days |
| 3.4 | Add `output_handler` callback pattern to all major analysis classes | 6 classes | 2 days |

### Tier 4 — Polish

| # | Action | Effort |
|---|---|---|
| 4.1 | Create a high-level `AnalysisPipeline` orchestrator (GUI entry point) | 1 day |
| 4.2 | Package `DiffusionSimulation.py` into `simulation/` submodule | 1 day |
| 4.3 | `pathlib.Path` throughout (replace bare strings) | 1 day |

**Estimated total:** ~17 person-days to fully production-ready + GUI-ready state.
Tier 1 alone (2 days) unblocks GUI development.

---

## 8. What NOT to refactor

- `gaussoptfuncs.py` — Numba-JIT models; clean, self-contained, do not restructure.
- `PlottingBase.py` — already excellent; only additive changes needed.
- `SpotDetectionFunctions.py` — ArrayPool/KernelCache design is clean and correct.
- `ImageAnalysisFunctions.py` — polymorphic strategy pattern is well implemented.
- `DiffusionSimulation.py` — legitimately complex physics; 1 900 lines is appropriate.
- `CameraDefaults.py` — tiny and correct; just add constants as needed.

---

## Completed

| Date | Action | Result |
|---|---|---|
| 2026-04-10 | `postprocess.py` dead code removal (~18 functions) | −753 lines (−27%); backed up to `claude/backup/old_postprocess.py` |
| 2026-04-10 | Consolidate `_link_group_*` into `LinkingFunctions.py` | 5 duplicate JIT functions removed from `postprocess.py`; canonical in `LinkingFunctions.py`; forwarding import added |
| 2026-04-10 | `CalibrationFunctions.calculate_variance` removal | Superseded by `calculate_offset_and_variance()`; backed up to `claude/backup/old_calibration.py` |
| 2026-04-10 | `TernaryPlotMixin` refactor in `PlottingBase.py` | Black background, clean single-panel layout, `MultipleLocator` grid; `create_ternary_plot` halved to ~80 lines |
| 2026-04-10 | Eliminate `DriftPlotting.py` | `DriftPlotter` moved into `FiducialDetection.py`; `_ensure_plotter()` removed from both files; `DriftCorrectionFunctions.py` updated to import from `FiducialDetection`; file deleted |
| 2026-04-10 | Delete `PhotonStreamFunctions.py` | Zero callers in src/ or notebooks; backed up to `claude/backup/old_PhotonStreamFunctions.py` |
| 2026-04-10 | Consolidate HDF5 I/O via `IOFunctions` | `_write_h5_database` made public (`write_h5_database`); `read_h5_database` added; frame-column cast made conditional; all raw `pd.read_hdf`/`to_hdf` calls removed from `SM_extractionfunctions`, `NileRedFunctions`, `Multicolour_Simulation_Functions`, `SR_Functions` |
| 2026-04-10 | Remove all RCC code | `RCCDriftCorrector` class deleted (~140 lines); `DriftMethod.RCC` enum value removed; `undrift_rcc()` removed; `AutoDriftCorrector` simplified (always AIM); `imageprocess` import removed; `DriftResult` default fixed to AIM; `test_undrift_with_rcc` removed from both unit test files; `imageprocess.py` deleted (backed up) |
| 2026-04-10 | Eliminate magic numbers | `DriftConstants` + `FilteringConstants` added to `Constants.py` (pixel sizes derived from `CameraDefaults`); `DriftParameters` gains `pixel_size_nm` with AIM distances computed in `__post_init__`; all `pixel_size=69` / `min_sigma=75/69` / `max_colour_error=0.15` / `min_photons=500` defaults replaced across 8 files (`DriftCorrectionFunctions`, `AIMAlgorithm`, `SM_extractionfunctions`, `FiducialDetection`, `NileRedFunctions`, `DiffusionSimulation`, `PlottingBase`, `Multicolour_Simulation_Functions`) |
| 2026-04-20 | Add `AnalysisConfig` dataclass (Tier 1.3) | `AnalysisConfig` added to `Constants.py` with `output_dir`, `display`, `save_figures`, `figure_format`, `dpi`, `progress_callback`, `logging_callback`; threaded into `SR_Functions.__init__` (`config` param, `self.config`, wired into `example_spots_singleframe` via `plotter.save_or_show`) |
| 2026-04-20 | Replace all `plt.show()` with display-gated calls (Tier 1.1) | 9 calls across 4 files replaced with `if display: plt.show()` / `elif display: plt.show()`; `display: bool = True` added to `_plot_initial_guess_2d`, `_plot_unmixing_results`, `plot_refinement_diagnostics`, `_plot_spatial_distribution` (SM_extractionfunctions), `_plot_single_gaussian_validation` (DriftCorrectionFunctions), `_plot_drift_analysis` (postprocess); all callers backwards-compatible via default `display=True` |
| 2026-04-20 | Extract `FilteringCriteria` dataclass (Tier 2.2) | `FilteringCriteria` dataclass added to `Constants.py` (8 fields with defaults from `FilteringConstants`); imported into `SM_extractionfunctions`; `filter_quality_localisations` accepts `criteria: FilteringCriteria = None`; `extract_single_molecules_HDBSCAN`, `_DBSCAN`, `_linked`, `_spectral_lap`, `_batch` all accept and thread `criteria=criteria`; `max_localisation_error=1.0` literals replaced with `FilteringConstants.MAX_LOCALISATION_ERROR_PX` |
| 2026-04-21 | Split `DriftCorrectionFunctions.py` into `drift_correction/` subpackage (Tier 2.1) | 3,524-line monolith split into `_base.py` (218), `aim.py` (881), `fiducial.py` (368), `auto.py` (138), `_facade.py` (1,672), `__init__.py` (40); `DriftCorrectionFunctions.py` reduced to 45-line backward-compat shim — all callers unchanged; fixed stale import in `FiducialDetection.py:757` (`CoordinateProcessor` now imported from `CoordinateProcessing` not `DriftCorrectionFunctions`) |
