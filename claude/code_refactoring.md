# pyBayerSMLM — Refactoring Analysis & Plan

**Date:** 2026-04-22 (last updated 2026-04-30)
**Scope:** Full `src/` codebase (~37 100 lines across 32 files + `drift_correction/` + `clustering/` subpackages)
**Goal:** Compact, production-ready code that can be driven by a GUI without re-architecture

---

## 1. File Inventory

| File | Lines | Primary responsibility |
|------|------:|----------------------|
| SM_extractionfunctions.py | 262 | Core `extract_SMs` class: `__init__`, `filter_quality_localisations`, `average_parameters`, `collect_traces`; inherits all methods from six mixins |
| mixture_analysis.py | 1 863 | `MixtureAnalysisMixin`: GMM mixture analysis, `_fit_gmm_mle`, `extract_reference_means`, `fit_covariances_*`, `calculate_analytical_misidentification`, `analyze_photon_dependent_*` |
| channel_unmixing.py | 2 048 | `ChannelUnmixingMixin`: `unmix_channels`, `unmix_channels_with_spatial_refinement`, `_cluster_seeds_spatially`, `_refine_spectral_model_from_puncta`, all helpers |
| PlottingBase.py | 4 137 | Master plotting classes (publication, analysis, ternary, datashader) |
| Multicolour_Simulation_Functions.py | 3 761 | Simulation, bootstrap, fitting delegation, file I/O |
| drift_correction/_facade.py | 1 672 | Drift correction pipeline orchestration (post-split) |
| SR_Functions.py | 2 569 | Super-resolution pipeline orchestrator |
| postprocess.py | 1 977 | NENA, pair correlation, index blocks, linking utilities |
| NileRedFunctions.py | 1 936 | Nile Red spectral wavelength extraction |
| DiffusionSimulation.py | 1 884 | 2D Langevin diffusion, MSD, binding kinetics |
| ImageAnalysisFunctions.py | 1 777 | Gaussian fitting (9 strategies, polymorphic) |
| SpectralFunctions.py | 1 538 | Spectral models, filter data, pixel QYs |
| IOFunctions.py | 1 337 | HDF5/TIFF I/O, photon normalisation |
| FiducialDetection.py | 1 264 | Fiducial bead detection pipeline + DriftPlotter |
| SpotDetectionFunctions.py | 1 081 | Spot detection (matched filter, CA-CFAR) |
| drift_correction/aim.py | 881 | AIM drift algorithm implementation |
| render.py | 848 | Histogram rendering, Gaussian convolution, smoothing |
| AIMAlgorithm.py | 785 | AIM drift algorithm (legacy location; shim only) |
| PSFFunctions.py | 774 | Wavelength-dependent PSF models |
| gaussoptfuncs.py | 652 | Gaussian fitting model variants (8 models) |
| CoordinateProcessing.py | 527 | Segmentation, coordinate validation |
| sCMOSFunctions.py | 520 | sCMOS calibration, demosaicing, variance weighting |
| CalibrationFunctions.py | 472 | Camera calibration |
| ImportManager.py | 439 | Graceful optional-dependency loading |
| FRCFunctions.py | 433 | Fourier Ring Correlation |
| ProgressUtils.py | 403 | Progress-bar context managers |
| localise.py | 388 | Spot identification, initial fitting |
| StepDetector.py | 384 | Change-point detection |
| drift_correction/fiducial.py | 368 | FiducialDriftCorrector + DriftPlotter |
| LoggingFramework.py | 306 | Structured logging with progress tracking |
| lib.py | 305 | Record-array utilities, polygon/rectangle selection |
| LinkingFunctions.py | 302 | Post-hoc linking of blinking detections |
| MaskFunctions.py | 270 | Bayer mask generation |
| Constants.py | 265 | Global constants + AnalysisConfig + FilteringCriteria |
| HelperFunctions.py | 253 | Coordinate maths, small utilities |
| drift_correction/_base.py | 218 | DriftMethod enum, DriftParameters, DriftResult, ABC |
| DriftCorrectionFunctions.py | 45 | Backward-compat shim → drift_correction/ |
| drift_correction/auto.py | 138 | AutoDriftCorrector, strategy selection |
| drift_correction/__init__.py | 40 | Public imports only |
| clustering/linked_clusterer.py | 304 | LinkedMixin: linked + spectral LAP linking |
| clustering/batch.py | 286 | BatchMixin: batch, photon accumulation, multi-FOV |
| clustering/hdbscan_clusterer.py | 95 | HDBSCANMixin: HDBSCAN extraction |
| clustering/dbscan_clusterer.py | 90 | DBSCANMixin: DBSCAN extraction |
| clustering/__init__.py | 22 | Public imports only |
| CameraDefaults.py | 75 | Camera config registry |

---

## 2. Duplication

All known duplication resolved — see Completed table.

---

## 3. Structural Problems

### 3a. `SM_extractionfunctions.py` — DONE (Tier 2.3 + Tier 2.4 complete)

The 5 354-line god-file has been fully split across six mixin subclasses:

```
clustering/
    __init__.py             (re-exports)
    hdbscan_clusterer.py    HDBSCANMixin
    dbscan_clusterer.py     DBSCANMixin
    linked_clusterer.py     LinkedMixin
    batch.py                BatchMixin

mixture_analysis.py         MixtureAnalysisMixin  (1 863 lines)
channel_unmixing.py         ChannelUnmixingMixin  (2 048 lines)
```

`extract_SMs` (262 lines) inherits `(HDBSCANMixin, DBSCANMixin, LinkedMixin, BatchMixin,
MixtureAnalysisMixin, ChannelUnmixingMixin)`.  Core utilities
(`filter_quality_localisations`, `average_parameters`, `collect_traces`) remain on
the host class.  All callers unchanged; zero regressions confirmed by test suite.

---

### 3b. `drift_correction/_facade.py` — 1 672 lines

The Tier 2.1 split moved the algorithm files cleanly, but `_facade.py` absorbed
the full pipeline orchestration and is still large. Not an immediate problem — it has
a single clear responsibility — but worth revisiting once 3a is done.

---

### 3c. Magic numbers — DONE (see Completed)

---

## 4. Parameter-Group Dataclasses

`FilteringCriteria` and `AnalysisConfig` are done.  Remaining:

| Function group | New dataclass | Home file | Status |
|---|---|---|---|
| `render.*` rendering args (8 params) | `RenderingConfig` | `render.py` | Pending |
| `extract_single_molecules_*` clustering args | `ClusteringConfig` | `clustering/__init__.py` | Blocked on 3a |

---

## 5. Plotting Architecture

`plt.show()` blocking calls resolved (Tier 1.1 complete — 9 calls across 4 files replaced
with `if display: plt.show()` guards; `display: bool = True` added to all affected methods).

Remaining concern: some plotting methods still create their own `matplotlib` figures directly
rather than accepting an optional `plotter` argument and returning `(fig, ax)`.  This matters
for embedding in a GUI canvas.  Tracking under Tier 3.4.

---

## 6. GUI Readiness

### What's already good
- `LoggingFramework.py` — structured logging with callback hooks.
- `ProgressUtils.py` — context-manager progress bars; wrappable by Qt widget.
- `ImportManager.py` — graceful optional-dependency loading.
- `PlottingBase.save_or_show()` — accepts `display=False` and a save path.
- `CameraDefaults.py` — camera registry already GUI-friendly.
- `AnalysisConfig` — exists in `Constants.py`; threaded into `SR_Functions`.
- `plt.show()` blocking calls — removed from all analysis functions.

### Remaining blockers

| Blocker | Location | Fix | Tier |
|---|---|---|---|
| Progress via `print()` | ~100 call sites | Route through `LoggingFramework` | 3.2 |
| No clean result objects | Most analysis functions return recarrays | Thin result dataclasses | 3.4 |
| File paths as bare strings | Throughout | `pathlib.Path` + `AnalysisConfig` | 4.3 |
| `AnalysisConfig` not fully threaded | Only in `SR_Functions` | Wire into remaining major classes | 3.4 |

---

## 7. Prioritised Refactoring Plan

### Tier 2 — Reduces file size and complexity

| # | Action | Files touched | Effort |
|---|---|---|---|
| 2.3 | ✅ Split clustering logic out of `SM_extractionfunctions.py` | SM_extract + new clustering/ | done |
| 2.4 | ✅ Split GMM/unmixing code out of `SM_extractionfunctions.py` | SM_extract + new mixture_analysis.py + channel_unmixing.py | done |

### Tier 3 — Code quality and GUI readiness

| # | Action | Files touched | Effort |
|---|---|---|---|
| 3.1 | `RenderingConfig` dataclass in `render.py` | render.py + callers | 0.5 day |
| 3.2 | Replace `print()` progress with `LoggingFramework` throughout | ~8 files | 1 day |
| 3.4 | Thread `AnalysisConfig` into remaining major classes; add `output_handler` callbacks | 6 classes | 2 days |
| 3.3 | Comprehensive type hints on public methods | All | 3 days |

### Tier 4 — Polish

| # | Action | Effort |
|---|---|---|
| 4.3 | `pathlib.Path` throughout (replace bare strings) | 1 day |
| 4.1 | Create a high-level `AnalysisPipeline` orchestrator (GUI entry point) | 1 day |
| 4.2 | Package `DiffusionSimulation.py` into `simulation/` submodule | 1 day |

**Estimated remaining effort:** ~11 person-days to fully GUI-ready state (2.4 = 2 days, Tier 3 = 6.5 days, Tier 4 = 3 days).

---

## 8. What NOT to refactor

- `gaussoptfuncs.py` — Numba-JIT models; clean, self-contained.
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
| 2026-04-10 | Consolidate HDF5 I/O via `IOFunctions` (Tier 2.4) | `_write_h5_database` made public; `read_h5_database` added; all raw `pd.read_hdf`/`to_hdf` calls removed from `SM_extractionfunctions`, `NileRedFunctions`, `Multicolour_Simulation_Functions`, `SR_Functions` |
| 2026-04-10 | Remove all RCC code (Tier 2.0) | `RCCDriftCorrector` class deleted (~140 lines); `DriftMethod.RCC` removed; `undrift_rcc()` removed; `AutoDriftCorrector` simplified; `imageprocess.py` deleted |
| 2026-04-10 | Eliminate magic numbers (Tier 1.2) | `DriftConstants` + `FilteringConstants` added to `Constants.py`; all `pixel_size=69` / `min_sigma=75/69` / `max_colour_error=0.15` / `min_photons=500` defaults replaced across 8 files |
| 2026-04-20 | Add `AnalysisConfig` dataclass (Tier 1.3) | Added to `Constants.py`; threaded into `SR_Functions.__init__` |
| 2026-04-20 | Replace all `plt.show()` with display-gated calls (Tier 1.1) | 9 calls across 4 files; `display: bool = True` added to all affected methods |
| 2026-04-20 | Extract `FilteringCriteria` dataclass (Tier 2.2) | Added to `Constants.py`; `filter_quality_localisations` + all `extract_single_molecules_*` methods accept `criteria=criteria` |
| 2026-04-21 | Split `DriftCorrectionFunctions.py` into `drift_correction/` subpackage (Tier 2.1) | 3,524-line monolith → `_base.py` (218), `aim.py` (881), `fiducial.py` (368), `auto.py` (138), `_facade.py` (1,672), `__init__.py` (40); shim at 45 lines; all callers unchanged |
| 2026-04-30 | Split clustering logic out of `SM_extractionfunctions.py` (Tier 2.3) | 5,354-line god-file → clustering mixin subpackage (`HDBSCANMixin`, `DBSCANMixin`, `LinkedMixin`, `BatchMixin`); 1,220 lines moved; SM_extractionfunctions.py now 4,132 lines; all callers unchanged |
| 2026-05-01 | Split GMM + channel-unmixing code out of `SM_extractionfunctions.py` (Tier 2.4) | 4,132-line file → `mixture_analysis.py` (1,863 lines, `MixtureAnalysisMixin`) + `channel_unmixing.py` (2,048 lines, `ChannelUnmixingMixin`); `SM_extractionfunctions.py` now 262 lines; six-mixin inheritance; zero regressions |
