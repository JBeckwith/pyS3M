# Type Hints Plan — pyS3M

**Date:** 2026-05-08
**Scope:** All public methods/functions in `src/` (~37 k lines, 42 files)
**Python target:** 3.10.12
**Estimated effort:** ~3 days

---

## Current State

| Category | Files | Notes |
|---|---|---|
| Complete (≥95%) | FRCFunctions, CameraDefaults, LinkingFunctions, ImageAnalysisFunctions, SpectralFunctions, CoordinateProcessing | Leave as-is |
| Strong (70–94%) | drift_correction/ (all 5), FiducialDetection, NileRedFunctions, simulation/diffusion | Fill gaps only |
| Partial (40–69%) | drift_correction/_facade, simulation/multicolour, channel_unmixing, SpotDetectionFunctions, PlottingBase, render, SR_Functions | Systematic pass |
| Bare (0–20%) | IOFunctions, postprocess, localise, lib, mixture_analysis, clustering/(batch, linked), CalibrationFunctions, HelperFunctions, MaskFunctions, Constants | Full annotation needed |

No file currently uses `numpy.typing`; all use legacy `from typing import ...` style.

---

## Conventions

### Import block (add to each file needing it)

```python
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray
import numpy as np
```

Use `from __future__ import annotations` in every file — it makes all annotations strings
at runtime, eliminating circular-import issues from forward references.

### Common type aliases — define once in `Constants.py`, import elsewhere

```python
# Constants.py  (add near the top, after dataclass definitions)
from numpy.typing import NDArray
import numpy as np

# Localisation record array (the universal data carrier)
Localisations = np.recarray

# Progress/log callbacks (already used throughout, just not aliased)
ProgressCallback = Optional[Callable[[float], None]]
LogCallback = Optional[Callable[[str], None]]

# Image arrays
ImageArray = NDArray[np.float32]    # 2-D or 3-D float image
BayerStack = NDArray[np.float32]    # shape (n_frames, H, W)
```

### Per-type guidance

| Value | Annotation |
|---|---|
| Localisation record array | `np.recarray` |
| Generic numpy array | `NDArray[np.float32]` / `NDArray[np.float64]` / `NDArray[np.int32]` as appropriate |
| 2-D image | `NDArray[np.float32]` with shape comment if non-obvious |
| File path | `Path` (already used everywhere post-4.3) |
| pandas DataFrame | `pd.DataFrame` |
| polars DataFrame | `pl.DataFrame` |
| Fitting strategy enum | `FittingStrategy` |
| Config dataclasses | `AnalysisConfig`, `ClusteringConfig`, `RenderingConfig`, `FilteringCriteria` |
| Result dataclasses | `DriftResult`, `FiducialDetectionResult` |
| Progress callback | `Optional[Callable[[float], None]]` |
| Log callback | `Optional[Callable[[str], None]]` |
| "returns nothing" | `-> None` (add even when obvious — mypy requires it) |

### What NOT to annotate

- Private helpers whose signature is obvious from 2 lines of body (`_parse_threshold`, `_nan_helper`, etc.)
- Numba `@jit`-decorated functions — Numba ignores annotations and they add noise
- `*args / **kwargs` forwarding wrappers (e.g. `_facade.py` delegation shims)

---

## Work Batches

Work smallest-to-largest to build momentum and catch alias/convention issues early.

---

### Batch A — Small utilities (~0.5 days)

**Goal:** Fully annotate five small files with zero dependencies on each other.

#### `Constants.py` (265 lines)
- Has `from __future__ import annotations` already
- Add `ProgressCallback` / `LogCallback` / `Localisations` / `ImageArray` aliases here
- Annotate `AnalysisConfig.__post_init__`, `FilteringCriteria` methods (2 public methods)

#### `HelperFunctions.py` (253 lines)
- 8 public methods, all bare
- Dominant types: `np.recarray`, `NDArray`, `float`, `Tuple[float, float]`
- Add `from __future__ import annotations`, `from numpy.typing import NDArray`

#### `MaskFunctions.py` (270 lines)
- 7 public methods, all bare
- Dominant types: `NDArray[np.bool_]` (masks), `int` (image dimensions)

#### `lib.py` (305 lines)
- 10 module-level functions, all bare
- Mix of `np.recarray`, `NDArray`, `List`, `Tuple`
- No class — straightforward function-by-function pass

#### `CalibrationFunctions.py` (472 lines)
- 7 public methods, all bare
- Dominant types: `NDArray[np.float32]`, `Path`, `Dict[str, float]`

---

### Batch B — Core pipeline files ✅ (2026-05-08)

#### `localise.py`
- `identify_in_frame(frame, minimum_ng, box, roi) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.float32]]`
- `identify_frame(..., resultqueue: Any | None) -> np.recarray`
- `identify_by_frame_number(..., lock: threading.Lock | None) -> np.recarray`
- `identifications_from_futures(futures: list[Any]) -> np.recarray`
- `get_spots(movie: NDArray | Any, identifications: np.recarray, box: int, camera_info: dict[str, float]) -> NDArray[np.float32]`
- `locs_from_fits(identifications, theta, CRLBs, likelihoods, iterations, box) -> np.recarray`
- `check_nena(locs, info, callback: Callable[[float], None] | None) -> float`
- `check_kinetics(locs, info) -> float`

#### `IOFunctions.py`
- All 20 public methods annotated
- `read_h5_database -> pd.DataFrame`, `read_tiff -> NDArray[np.float32]`, `read_hyperstack -> tuple[NDArray[np.float32], str | None]`
- `generate_weights -> NDArray[np.float64]`, `process_roi_to_photoelectrons -> tuple[NDArray, NDArray, NDArray]`
- Camera calibration map params (`gain_map`, `offset_map`, `rqe`, `read_noise`) annotated as `float | NDArray[np.float32]` — scalar defaults valid for uncalibrated use
- `metadata_reader_imageJ` → `tuple[int, int, int, int] | tuple[int, int, int, int, float]` (union: 4 or 5 values depending on `return_exposure` flag)

#### `render.py`
- `render(...) -> tuple[int, NDArray[np.float32]] | tuple[int, NDArray[np.float32], NDArray[np.float32]]`
- `render_hist`, `render_gaussian`, `render_gaussian_iso`, `render_convolve`, `render_smooth` → `tuple[int, NDArray[np.float32]]`
- `render_gaussian_colour` → `tuple[int, NDArray[np.float32], NDArray[np.float32]]`

#### `SR_Functions.py`
- `__init__`: injected deps (`io_functions`, `helper_functions`, etc.) annotated as `Any | None` — **not** `object | None`, because Pylance cannot resolve method access on bare `object`
- `_postprocess_fit_results(fit_results_array: NDArray[np.float64], ..., quality_metrics: dict[str, NDArray] | None) -> pd.DataFrame`
- `_filter_fit_results(fit_results: pd.DataFrame, width: int, height: int) -> pd.DataFrame`
- `example_spots_singleframe(...) -> tuple[Any, Any]` — matplotlib `Figure` / `Axes` returned as `Any` (loaded via `get_module`)
- `fit_SM_data`, `fit_tracking_data`, `fit_imaging_data` → `None`; calibration maps (`gain_map`, etc.) annotated as `NDArray[np.float32]` (not `float | NDArray`) — these high-level methods require real 2D maps; `crop_calibration_maps` uses array slicing so scalars are invalid here

---

### Batch C — Mixin classes ✅ (2026-05-08)

#### `clustering/batch.py`
- `_extract_fov_name(filepath: Path | str) -> str`
- `extract_single_molecules_batch(localisation_files: list[str | Path], ...) -> tuple[pd.DataFrame, pd.DataFrame]`
- `build_photon_accumulation_database(single_frame_database: pd.DataFrame, ...) -> pd.DataFrame`
- `analyse_multi_fov_dataset(...) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]`

#### `clustering/linked_clusterer.py`
- `extract_single_molecules_linked(loc_data: pd.DataFrame, ...) -> tuple[pd.DataFrame, pd.DataFrame]`
- `flag_static_localisations(loc_data: pd.DataFrame, eps: float, min_samples: int) -> NDArray[np.bool_]`
- `spectral_lap_link(loc_data: pd.DataFrame, ...) -> NDArray[np.int32]`
- `extract_single_molecules_spectral_lap(loc_data: pd.DataFrame, ...) -> tuple[pd.DataFrame, pd.DataFrame]`

#### `clustering/hdbscan_clusterer.py` and `clustering/dbscan_clusterer.py`
- `extract_single_molecules_HDBSCAN(loc_data: pd.DataFrame, ...) -> tuple[pd.DataFrame, pd.DataFrame]`
- `extract_single_molecules_DBSCAN(loc_data: pd.DataFrame, ...) -> tuple[pd.DataFrame, pd.DataFrame]`
- Added `FilteringCriteria` import to both (was missing)

#### `mixture_analysis.py`
- `_fit_gmm_mle(X: NDArray[np.float64], initial_means: NDArray[np.float64], n_components: int, ...) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], bool]`
- `_fit_gmm_em(X, initial_means, n_components, ...) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], bool]`
- `extract_reference_means(data_db: pd.DataFrame, ...) -> tuple[NDArray[np.float64], pd.DataFrame, Any]` (Any = GaussianMixture)
- `fit_covariances_fixed_means(X, fixed_means, ...) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]`
- `fit_covariances_fixed_means_mestimator(X, fixed_means, ...) -> tuple[NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]`
- `calculate_analytical_misidentification(fixed_means, covariances, weights, ...) -> dict[str, Any]`
- `analyze_photon_dependent_misidentification_analytical(...) -> pd.DataFrame`

#### `channel_unmixing.py`
- `unmix_channels(loc_data: pd.DataFrame, n_channels: int, ...) -> tuple[pd.DataFrame, dict[str, Any]]`
- `unmix_channels_with_spatial_refinement(...)` — updated `Tuple[pd.DataFrame, Dict]` → `tuple[pd.DataFrame, dict[str, Any]]`
- `find_exemplar_dye_pair(sf_db: pd.DataFrame, mean_0: NDArray, mean_1: NDArray, ...) -> pd.DataFrame | None`
- `get_exemplar_crop(pair_row: pd.Series, data_folder: str | Path, crop_size_px: int) -> tuple[NDArray[np.float32], pd.DataFrame]`
- Replaced `from typing import Tuple, Dict, Optional` with `from typing import Any, Optional` + `from numpy.typing import NDArray`

#### `postprocess.py`
- `get_index_blocks(locs: np.recarray, ...) -> tuple[np.recarray, float, NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32], int, int]`
- `index_blocks_shape(width, height, size) -> tuple[int, int]`
- `get_block_locs_at(x, y, index_blocks: tuple) -> np.recarray`
- `picked_locs(locs: np.recarray, ...) -> np.recarray`
- `nena(locs: np.recarray, info: list[dict[str, Any]], callback: Callable | None) -> tuple[Any, float]`
- `next_frame_neighbor_distance_histogram(locs: np.recarray, ...) -> tuple[NDArray[np.float64], NDArray[np.float64]]`
- `link(locs: np.recarray, info: list[dict[str, Any]], ...) -> np.recarray`
- `link_loc_groups(locs: np.recarray, info, link_group: NDArray[np.int32], ...) -> np.recarray`
- `undrift_from_picked(picked_locs: list[np.recarray], n_frames: int) -> np.recarray`
- `segment_locs_by_rendered_image(locs: pd.DataFrame | np.recarray, ...) -> tuple[pd.DataFrame, pd.DataFrame]`
- `remove_fiducials(aggregate_locs: pd.DataFrame, per_aggregate_stats: pd.DataFrame, n_frames: int, A_R_threshold: float | tuple[float, str] | None, ...) -> tuple[pd.DataFrame, pd.DataFrame, NDArray[np.bool_]]`

---

### Batch D — Partial files, fill gaps ✅ (2026-05-08)

#### `NileRedFunctions.py`
- Added `from __future__ import annotations`, `from numpy.typing import NDArray`
- `from typing import Dict, Tuple, Optional, List, Union` → `from typing import Any, Optional`
- Fixed `pixel_size: float = None` → `float | None = None` in `__init__`, `simulate_wavelength_precision`, `fit_wavelengths_from_h5`, `fit_wavelengths_pixelated`
- `setup_optical_system` → `tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]`
- `nile_red_forward_model` → `dict[str, float]`; `residuals_nile_red` → `NDArray[np.float64]`
- `fit_nile_red_wavelength` → `tuple[float, dict[str, float]]`
- `_normalize_rgb_with_errors` → `tuple[NDArray[np.float64], NDArray[np.float64]]`
- `_weighted_average_with_error` → `tuple[float, float]`; `_parallel_fit_wavelengths` → `list[tuple[float, float]]`
- `fit_wavelengths_pixelated` → `pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]`
- Module-level `_fit_nile_red_wavelength_standalone` → `tuple[float, float]`

#### `drift_correction/_facade.py`
- Added `from __future__ import annotations`, `from numpy.typing import NDArray`
- `from typing import Optional, Callable, Tuple, Union, Dict, Any, List` → `from typing import Optional, Any`
- Fixed `pixel_size: float = None` and `pixelsize: float = None` → `float | None = None`
- `undrift` → `tuple[np.recarray, DriftResult]`; `available_methods` → `list[str]`
- `method_info` → `dict[str, Any]`; `_find_indices_in_original_locs` → `NDArray[np.int64]`
- `detect_high_density_regions_from_image` → `tuple[list[tuple[int, int]], NDArray[np.float64], float, dict[str, Any]]`
- `select_puncta_from_regions` → `tuple[list[np.recarray], dict[str, Any]]`
- `identify_real_fiducials_with_clustering` → `tuple[list[np.recarray], dict[str, Any]]`
- `apply_validated_fiducial_drift_correction` → `tuple[np.recarray, dict[str, NDArray]]`

#### `SpotDetectionFunctions.py`
- Added `from __future__ import annotations`, `from numpy.typing import NDArray`
- `from typing import Union, Tuple` → `from typing import Any`
- Fixed `__init__` injected deps to `Any | None`; `pixel_size: float | None = None`
- `detect_puncta_in_stack_parallel` → `NDArray[np.int32] | tuple[NDArray[np.int32], dict[str, NDArray]]`
- `spots_from_futures(fs: list[Any])` → `NDArray[np.int32]`
- `spots_and_quality_from_futures(fs: list[Any])` → `tuple[NDArray[np.int32], dict[str, NDArray]]`
- `detect_puncta_in_images` → `list[NDArray[np.int32]] | tuple[list[NDArray[np.int32]], dict[str, NDArray]]`
- `detect_puncta_in_image` → `NDArray[np.int32] | tuple[NDArray[np.int32], dict[str, NDArray]]`
- `real_puncta_indices` → `NDArray[np.bool_] | tuple[NDArray[np.bool_], dict[str, NDArray]]`

#### `simulation/multicolour.py`
- Added `from __future__ import annotations`, `from numpy.typing import NDArray`
- `from typing import Dict, List, Tuple, Optional, Union, Any` → `from typing import Any, Optional`
- `CameraParameters` fields: `Dict` → `dict`, `List` → `list`; `validate_and_create` param updated
- `SimulationConfig.background_colour: list[float] | None = None`
- Fixed `pixel_size: float | None = None`, injected deps → `Any | None` in `MultiC_Sim_Funcs_Refactored.__init__`
- `_validate_inputs`: `Dict[str, Any]` → `dict[str, Any]`, `Optional[np.ndarray]` → `np.ndarray | None`
- `_setup_simulation_parameters` → `tuple[np.ndarray, np.ndarray, dict[str, Any]]`
- `_prepare_fitting_data` → `tuple[np.ndarray, np.ndarray, np.ndarray]`

#### `PlottingBase.py`
- Added `from __future__ import annotations`
- `colour_image_plot` → `matplotlib.axes.Axes` (was bare)
- `multichannel_overlay_plot` → `matplotlib.axes.Axes` (was bare)

---

## Verification

After each batch, run:

```bash
~/.virtualenvs/pyS3M/bin/python -m mypy src/ \
    --ignore-missing-imports \
    --no-strict-optional \
    --follow-imports=silent \
    2>&1 | grep "error:" | wc -l
```

Target: zero `error:` lines by end of Batch D.
(Use `--no-strict-optional` initially; tighten to `--strict` in a follow-up pass if desired.)

---

## Out of Scope

- Numba `@jit` / `@njit` / `@prange` decorated functions — Numba ignores annotations
- Test files in `unit_tests/`
- Notebook code
- `__init__.py` re-export files (type stubs would be the right tool there)
- Adding a `py.typed` marker (only needed if distributing as a typed package on PyPI)

---

## Progress Tracking

- [x] **Batch A** — Constants, HelperFunctions, MaskFunctions, lib, CalibrationFunctions (2026-05-08)
- [x] **Batch B** — localise, IOFunctions, render, SR_Functions (2026-05-08)
- [x] **Batch C** — clustering mixins, mixture_analysis, channel_unmixing, postprocess (2026-05-08)
- [x] **Batch D** — NileRedFunctions, _facade, multicolour, SpotDetectionFunctions, PlottingBase (2026-05-08)
