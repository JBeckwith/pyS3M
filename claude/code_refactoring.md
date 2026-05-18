# pyS3M — Refactoring Analysis & Plan

**Date:** 2026-04-22 (last updated 2026-05-17 — all Tier 5 deduplication items complete)
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
| Multicolour_Simulation_Functions.py | 3 761 | Simulation, bootstrap, fitting delegation, file I/O → **moving to `simulation/multicolour.py`** |
| drift_correction/_facade.py | 1 672 | Drift correction pipeline orchestration (post-split) |
| SR_Functions.py | 2 569 | Super-resolution pipeline orchestrator |
| postprocess.py | 1 977 | NENA, pair correlation, index blocks, linking utilities |
| NileRedFunctions.py | 1 936 | Nile Red spectral wavelength extraction |
| DiffusionSimulation.py | 1 884 | 2D Langevin diffusion, MSD, binding kinetics → **moving to `simulation/diffusion.py`** |
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

Programmatic analysis (2026-05-15) found six duplications. Ordered by severity:

---

### D1 (Critical) — `fit_SM_data` vs `fit_imaging_data` in `SR_Functions.py`  ← **Tier 5.1** ✅ DONE

`fit_SM_data` (≈ 245 lines) and `fit_imaging_data` (≈ 260 lines) were ~95% identical.
Both shared: file search, ROI/metadata load, mask + calibration-map crop, chunked
demosaic → detect → ROI-extract loop, quality-metric accumulation, parallel Gaussian
fitting, and `_postprocess_fit_results`.

The **only** behavioural differences were:

| | `fit_SM_data` | `fit_imaging_data` |
|---|---|---|
| `frame_offset` to `_process_detected_puncta_batch` | `chunk_start` (resets per file) | `total_frames + chunk_start` (global) |
| Output H5 name | `<tif_stem>.h5` per file | `Localisations.h5` for whole folder |
| H5 `append` flag | always `False` | `False` for first file, `True` thereafter |

**Fix applied (2026-05-17):** Collapsed into `_fit_files(accumulate_frame_numbers, combined_output)`.
Both public wrappers are now thin 9-line delegations.  Also fixed a latent bug in the old
`fit_imaging_data`: the `quality_metrics` dict was never deleted in the cleanup path.
Call-site catalogue: `claude/SR_Functions_Notebook_Corrections_20260517.md` (24 executable
notebook sites — wrappers kept so these remain non-urgent).

---

### D2 (High) — DBSCAN vs HDBSCAN clustering pipelines ✅ DONE

`clustering/dbscan_clusterer.py` and `clustering/hdbscan_clusterer.py` shared ~85 lines of
identical pre/post-processing (load → filter → empty checks → XY extraction →
precision calc → `average_parameters`). Only the ~12-line algorithm block differed:

```python
# DBSCAN:  DBSCAN(min_samples=min_cluster_size, eps=loc_precision * epsilon_multiplier)
# HDBSCAN: HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon=loc_precision)
```

`extract_single_molecules_linked` in `clustering/linked_clusterer.py` shared the same
load/filter/average_parameters boilerplate (~55 lines overlap with each of the above).

**Fix applied (2026-05-17):** Extracted `_prepare_locs`, `_check_min_locs`, and
`_finish_clustering` into `clustering/_base.py` → `ClusteringBaseMixin`.  All three
clusterer files refactored to use the helpers.  `spectral_lap` deliberately excluded
(its loading and finishing diverge from the simple pattern).  `SM_extractionfunctions.py`
MRO updated to inherit `ClusteringBaseMixin` last.

---

### D3 (High) — `AIMDriftCorrector` contains parallel AIM implementation alongside `AIMAlgorithm.py` ✅ DONE

`drift_correction/aim.py` had its own `_intersection_max`, `_intersection_max_z`, and
`_point_intersect_2d` methods duplicating `AIMAlgorithm.py`.  Additionally, a dead
`_aim_algorithm = AIMAlgorithm()` singleton was imported at module level but never used.

**Fix applied (2026-05-16/17):**
- Dead singletons (`_aim_algorithm`, `_coordinate_processor`, `_segmentation_handler`) removed from `drift_correction/aim.py`.
- `_facade.py` `run_aim_2d`/`run_aim_3d` rewritten to call `AIMDriftCorrector.calculate_drift` (the multithreaded reimplementation) instead of the old sequential `AIMAlgorithm`.
- `AIMAlgorithm.py` (785 lines) deleted entirely.
- Test suite updated: `test_drift_correction.py` and `test_drift_correction_simple.py` brought up to new API.  Fixed spline crash (`m > k`) when fewer than 4 segments and `ZeroDivisionError` in `auto.py` when `n_frames < segmentation`.
- Final test result: 18 passed, 3 skipped.

---

### D4 (High) — `select_puncta_from_regions` in two places ✅ DONE

`FiducialDetection.py:182` had the full ~100-line implementation (with plotting).
`drift_correction/_facade.py:941` had its own ~100-line implementation (no plotting,
`memory_optimize` spelling vs `memory_optimise`).

**Fix applied (2026-05-16):** Deleted the 170-line standalone `_facade.py` implementation;
replaced with a 15-line delegation to `self.fiducial_detector.select_puncta_from_regions`.
`memory_optimize` → `memory_optimise` spelling difference handled at call site.

---

### D5 (Low) — `one_column_plot` / `two_column_plot` overridden on `PublicationPlotter` ✅ DONE

Both methods were defined on `BasePlotter` and overridden with nearly identical bodies on
`PublicationPlotter`.  The overrides added a height warning but executed the same logic.

**Fix applied (2026-05-16):** Added the height warning to `BasePlotter.one_column_plot` and
`BasePlotter.two_column_plot`; deleted both overrides from `PublicationPlotter`.

---

### D6 (Trivial) — `_safe_tight_layout` and `DriftCorrectionError` copy-pasted ✅ DONE

- `_safe_tight_layout(fig)` — 8-line function defined identically in `mixture_analysis.py`
  and `channel_unmixing.py`.
  **Fix applied (2026-05-16):** Moved to `PlottingBase.py`; both files now import from there.

- `class DriftCorrectionError(Exception)` — 4-line class defined identically in
  `CoordinateProcessing.py` and `drift_correction/_base.py`.
  **Fix applied (2026-05-16):** Removed from `CoordinateProcessing.py`; imports from
  `drift_correction._base` (circular import resolved by D3).

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

### 3d. `simulation/` subpackage — DONE (Tier 4.2)

Two simulation files will be moved into a `simulation/` subpackage, matching the
pattern already established by `drift_correction/` and `clustering/`:

```
simulation/
    __init__.py        (public re-exports for backward compat)
    diffusion.py       ← DiffusionSimulation.py  (1 884 lines)
    multicolour.py     ← Multicolour_Simulation_Functions.py  (3 761 lines)

DiffusionSimulation.py                  (backward-compat shim, ~10 lines)
Multicolour_Simulation_Functions.py     (backward-compat shim, ~10 lines)
```

Key notes:
- `multicolour.py` will import from `.diffusion` (the sibling module) rather than
  the shim `DiffusionSimulation.py`, eliminating the cross-shim dependency.
- `multicolour.py` imports from `ImageAnalysisFunctions`, `IOFunctions`, etc. — the
  same cross-`src/` pattern already used by `clustering/batch.py` and others.
- All notebook and `src/` callers remain unchanged via the shims.
- Internals of both files are moved as-is; no restructuring of the physics or fitting
  logic is planned.

---

## 4. Parameter-Group Dataclasses

All planned dataclasses complete.

| Function group | New dataclass | Home file | Status |
|---|---|---|---|
| Filtering thresholds | `FilteringCriteria` | `Constants.py` | ✅ Done |
| Pipeline / analysis config | `AnalysisConfig` | `Constants.py` | ✅ Done |
| `render.*` rendering args | `RenderingConfig` | `render.py` | ✅ Done (Tier 3.1) |
| `extract_single_molecules_*` clustering args | `ClusteringConfig` | `clustering/_config.py` | ✅ Done |

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
| ~~File paths as bare strings~~ | ~~Throughout~~ | ~~`pathlib.Path` + `AnalysisConfig`~~ | ✅ 4.3 done |
| ~~`AnalysisConfig` not fully threaded~~ | ~~Only in `SR_Functions`~~ | ~~Wire into remaining major classes~~ | ✅ 3.4 done |

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
| 3.1 | ✅ `RenderingConfig` dataclass in `render.py` | render.py | done |
| 3.2 | ✅ Replace `print()` with `logging` throughout | 18 files | done |
| 3.4 | ✅ Thread `AnalysisConfig` into remaining major classes; add `output_handler` callbacks | 6 classes | done |
| 3.3 | ✅ Comprehensive type hints on public methods | All | done |

### Tier 4 — Polish

| # | Action | Effort |
|---|---|---|
| 4.3 | ✅ `pathlib.Path` throughout (replace bare strings) | done |
| 4.2 | ✅ Package `DiffusionSimulation.py` + `Multicolour_Simulation_Functions.py` into `simulation/` submodule | done |
| 4.1 | ✅ Create a high-level `AnalysisPipeline` orchestrator (GUI entry point) | done |

### Tier 5 — Deduplication ✅ ALL COMPLETE (2026-05-16/17)

| # | Duplication (see §2) | Date |
|---|---|---|
| 5.1 | ✅ D1: `_fit_files(accumulate_frame_numbers, combined_output)` in `SR_Functions.py`; wrappers retained | 2026-05-17 |
| 5.2 | ✅ D2: `ClusteringBaseMixin` (`_prepare_locs`, `_check_min_locs`, `_finish_clustering`) in `clustering/_base.py` | 2026-05-17 |
| 5.3 | ✅ D3: Dead singletons removed from `aim.py`; `AIMAlgorithm.py` deleted (785 lines) | 2026-05-16/17 |
| 5.4 | ✅ D4: `select_puncta_from_regions` standalone deleted from `_facade.py`; 15-line delegation added | 2026-05-16 |
| 5.5 | ✅ D5: `one_column_plot`/`two_column_plot` overrides deleted from `PublicationPlotter` | 2026-05-16 |
| 5.6 | ✅ D6: `_safe_tight_layout` → `PlottingBase.py`; `DriftCorrectionError` → `drift_correction._base` | 2026-05-16 |

---

## 8. What NOT to refactor

- `gaussoptfuncs.py` — Numba-JIT models; clean, self-contained.
- `PlottingBase.py` — already excellent; only additive changes needed.
- `SpotDetectionFunctions.py` — ArrayPool/KernelCache design is clean and correct.
- `ImageAnalysisFunctions.py` — polymorphic strategy pattern is well implemented.
- `simulation/diffusion.py` — legitimately complex physics; 1 900 lines is appropriate. Moved as-is from `DiffusionSimulation.py`; no internal restructuring.
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
| 2026-05-01 | Fix `add_colorbar` argument order in `PlottingBase.py` | `image_plot` and `contour_plot` were passing `(ax, im)` but signature is `(im, ax)`; `make_axes_locatable` received an `AxesImage` → `AttributeError` |
| 2026-05-01/08 | Replace `print()` with `logging` throughout (Tier 3.2) | 625 calls across 20 files → `logger = logging.getLogger(__name__)` per module; info/progress → `logger.info()`, warnings → `logger.warning()`, end=`"\r"` loop progress → `logger.debug()`; `verbose`-gated calls preserve their guard; 3 intentional `print()` retained in `ImportManager.py` (user-facing bootstrap) and `ProgressUtils.py` (fallback) |
| 2026-05-01 | `RenderingConfig` dataclass in `render.py` (Tier 3.1) | `@dataclass` with 10 fields mirroring `render()` kwargs; `render()` gains `config: RenderingConfig = None`; all existing callers unchanged |
| 2026-05-01 | `ClusteringConfig` dataclass (Tier 3) | `@dataclass` in `clustering/_config.py` with 21 fields covering all four extraction methods; wired into `extract_single_molecules_HDBSCAN`, `extract_single_molecules_DBSCAN`, `extract_single_molecules_linked`, `extract_single_molecules_spectral_lap`, `extract_single_molecules_batch`; also fixed bare `logger.info()` call in `batch.py` |
| 2026-05-01 | Thread `AnalysisConfig` into remaining classes (Tier 3.4) | `FiducialDetector`, `DriftPlotter`, `MultiC_Sim_Funcs_Refactored`, `NileRed_Functions`, `_plot_drift_analysis`, `segment_locs_by_rendered_image`, `remove_fiducials`; progress/logging callbacks added at key milestones; fixed pre-existing `save_or_show` bug in `_plot_drift_analysis`; zero regressions |
| 2026-05-06 | Remove `STANDARD_DATA` fitting strategy | `FittingStrategy.STANDARD_DATA` enum value, `StandardDataFittingProcessor` class (~95 lines), and all dispatch/registry entries removed from `ImageAnalysisFunctions.py`; matching local enum + `_fit_standard_data()` method (~85 lines) + dispatch branch removed from `Multicolour_Simulation_Functions.py`; `test_standard_data_fitting.py` deleted; `STANDARD_ITER` confirmed as default strategy throughout |
| 2026-05-06 | `pathlib.Path` throughout src/ (Tier 4.3) | All `os.path.*`, `os.makedirs`, `os.listdir`, `os.remove`, `os.walk` calls replaced with `pathlib.Path` equivalents across 26 files; `import os` removed from all src/ files; zero remaining `os.path` calls |
| 2026-05-06 | `simulation/` subpackage (Tier 4.2) | `DiffusionSimulation.py` (1 885 lines) → `simulation/diffusion.py`; `Multicolour_Simulation_Functions.py` (3 630 lines) → `simulation/multicolour.py`; `simulation/__init__.py` re-exports all 19 public names; both originals replaced with 15-line backward-compat shims; fixed latent missing `import sys` / `from pathlib import Path` in `DiffusionSimulation.py`; all 7 simulation unit tests pass; zero regressions |
| 2026-05-08 | `AnalysisPipeline` orchestrator (Tier 4.1) | `src/AnalysisPipeline.py` (new, 270 lines); `FittingConfig` dataclass groups all shared fit params; lazy-property `sr`/`sm`/`dcf` instances; `load_calibration()` / `calibrate()` / `fit(mode=...)` / `load_localisations()` / `filter_and_cluster()` / `undrift()` public API; 0 mypy errors |
| 2026-05-16 | D4 — Delete duplicate `select_puncta_from_regions` from `_facade.py` (Tier 5.4) | 170-line standalone replaced with 15-line delegation to `self.fiducial_detector`; `memory_optimize` → `memory_optimise` handled at call site |
| 2026-05-16 | D5 — Remove `one_column_plot`/`two_column_plot` overrides from `PublicationPlotter` (Tier 5.5) | Height warning added to `BasePlotter` base methods; both `PublicationPlotter` overrides deleted |
| 2026-05-16 | D6 — Deduplicate `_safe_tight_layout` and `DriftCorrectionError` (Tier 5.6) | `_safe_tight_layout` moved to `PlottingBase.py`; `DriftCorrectionError` removed from `CoordinateProcessing.py`, imported from `drift_correction._base`; circular import resolved by D3 |
| 2026-05-16/17 | D3 — Delete `AIMAlgorithm.py` and dead singletons (Tier 5.3) | Dead `_aim_algorithm`/`_coordinate_processor`/`_segmentation_handler` singletons removed from `aim.py`; `_facade.py` `run_aim_2d`/`run_aim_3d` rewritten to call `AIMDriftCorrector.calculate_drift`; `AIMAlgorithm.py` (785 lines) deleted; test suite updated to new API; fixed spline `m > k` crash and `ZeroDivisionError` in `auto.py`; 18 passed, 3 skipped |
| 2026-05-17 | D2 — Extract `ClusteringBaseMixin` (Tier 5.2) | `clustering/_base.py` new (65 lines): `_prepare_locs`, `_check_min_locs`, `_finish_clustering`; `dbscan_clusterer.py`, `hdbscan_clusterer.py`, `linked_clusterer.py` refactored; `SM_extractionfunctions.py` MRO updated; `spectral_lap` deliberately excluded |
| 2026-05-17 | D1 — Unify `fit_SM_data` / `fit_imaging_data` → `_fit_files()` (Tier 5.1) | `_fit_files(accumulate_frame_numbers, combined_output)` added to `SR_Functions.py` (~130 lines); both wrappers reduced to 9-line delegations; latent `quality_metrics` cleanup bug fixed; 24 notebook call sites catalogued in `claude/SR_Functions_Notebook_Corrections_20260517.md` |
| 2026-05-08 | Comprehensive type hints on public methods (Tier 3.3) | `from __future__ import annotations` + `NDArray[np.dtype]` + built-in generics across all 32 src/ files in 4 batches (A: core modules; B: SR_Functions/IOFunctions/postprocess; C: drift_correction/clustering/mixture/channel_unmixing; D: NileRedFunctions/_facade/multicolour/SpotDetectionFunctions/PlottingBase); 0 mypy errors verified |
