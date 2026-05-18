# AnalysisConfig Threading Plan

**Date:** 2026-05-01
**Scope:** Thread `AnalysisConfig` (from `Constants.py`) into the remaining major classes (Tier 3.4)

---

## 1. What AnalysisConfig Provides

```python
@dataclass
class AnalysisConfig:
    output_dir:        Optional[Path] = None
    display:           bool = True
    save_figures:      bool = False
    figure_format:     str  = "svg"
    dpi:               int  = 300
    progress_callback: Optional[Callable[[float, str], None]] = None
    logging_callback:  Optional[Callable[[str], None]] = None
```

**Already threaded into:** `SR_Functions` (stored as `self.config`, consumed at line 876 for
save/display).  Pattern: `config` in `__init__`; stored as `self.config`; defaults to
`AnalysisConfig()` when `None` (interactive mode, no saving).

---

## 2. Scope — What Gets Threaded, What Doesn't

### Include

| Class / module | File | Reason |
|---|---|---|
| `FiducialDetector` | `FiducialDetection.py` | Has `output_figure_path` already but no `display` control; long-running loops without progress callbacks |
| `DriftPlotter` | `FiducialDetection.py` | 7 figure-producing methods with hardcoded `plt.show()` and `dpi=300` |
| `NileRed_Functions` | `NileRedFunctions.py` | Long-running parallel loops with rich logging but no `progress_callback` or `logging_callback` |
| `MultiC_Sim_Funcs_Refactored` | `Multicolour_Simulation_Functions.py` | 2 plot methods already accept `show`/`save_path` but hardcode `dpi=600`; bootstrap loops need progress hooks |
| `postprocess` (functions) | `postprocess.py` | `segment_locs_by_rendered_image` and `remove_fiducials` are major pipeline steps with extensive logging; `_plot_drift_analysis` has hardcoded `dpi=300` |

### Exclude

| Class | Reason |
|---|---|
| `Image_Analysis_Functions` | Pure numerical computation — no I/O, no display, no logging output |
| `Spectral_Funcs` | Pure spectral calculation and DB query — no I/O or display |
| `DriftPlotter` (standalone) | Handled as part of `FiducialDetector` threading (config propagated down) |

---

## 3. Standard Pattern (from SR_Functions precedent)

### `__init__` change

```python
from Constants import AnalysisConfig   # add to imports

def __init__(self, ..., config: AnalysisConfig = None):
    ...
    self.config = config if config is not None else AnalysisConfig()
```

### Consuming config fields

```python
# Display / save
self.plotter.save_or_show(
    fig,
    save_path=self.config.output_dir / f"{stem}.{self.config.figure_format}"
              if self.config.save_figures and self.config.output_dir else None,
    show=self.config.display,
    dpi=self.config.dpi,
)

# Progress callback (optional callable)
if self.config.progress_callback:
    self.config.progress_callback(i / n_total, f"Frame {i}/{n_total}")

# Logging callback (optional callable)
if self.config.logging_callback:
    self.config.logging_callback(msg)
```

All `config=` parameters default to `None` → `AnalysisConfig()`.  All existing call sites
remain unchanged; `AnalysisConfig()` defaults reproduce current interactive behaviour.

---

## 4. Per-Class Implementation

---

### 4.1 `FiducialDetector` (FiducialDetection.py)

**Effort: moderate (~60 min)**

#### `__init__` (line 61)

```python
# Before
def __init__(self, drift_correction_instance=None):

# After
def __init__(self, drift_correction_instance=None, config: AnalysisConfig = None):
    ...
    self.config = config if config is not None else AnalysisConfig()
```

#### Methods to update

| Method | Current behaviour | Change |
|---|---|---|
| `detect_high_density_regions_from_image` (line ~70) | `output_figure_path: str = None`; hardcoded `dpi=300`, no `display` | Replace `output_figure_path` with `self.config.*`; pass `self.config` to `DriftPlotter(config=self.config)` |
| `select_puncta_from_regions` (line ~182) | `output_figure_path: str = None`; no `display` | Same |
| Region-loop (line ~125) | `for region_id in range(1, n_regions + 1)` — no progress hook | Add `if self.config.progress_callback: self.config.progress_callback(region_id / n_regions, ...)` |
| `logger.info(...)` calls (lines ~331, ~337, ~432) | Log only | Also call `self.config.logging_callback(msg)` when set |

**When instantiating DriftPlotter** (wherever `DriftPlotter()` is called inside this class):
replace with `DriftPlotter(config=self.config)`.

---

### 4.2 `DriftPlotter` (FiducialDetection.py, line 612)

**Effort: high (~2 h)**

`DriftPlotter` is stateless (inherits from `AnalysisPlotter`, `__init__` only calls `super()`).
Add `config` to `__init__` so it can be propagated from `FiducialDetector` or used standalone.

#### `__init__` (line 619)

```python
# Before
def __init__(self):
    super().__init__()

# After
def __init__(self, config: AnalysisConfig = None):
    super().__init__()
    self.config = config if config is not None else AnalysisConfig()
```

#### Methods to update (7 methods, all with same pattern)

For every figure-producing method, replace the final `save_or_show` / `plt.show()` / `plt.savefig` block
with the standard pattern.  Key specifics:

| Method | Line(s) | Problem | Fix |
|---|---|---|---|
| `plot_fiducial_detection_steps` | 737 | `show=True` hardcoded | `show=self.config.display` |
| `plot_fiducial_detection_steps` | 737 | `dpi=300` hardcoded | `dpi=self.config.dpi` |
| `plot_fiducial_detection_results` | ~820 | `show=True` hardcoded | `show=self.config.display` |
| `plot_puncta_selection_results` | ~953 | `dpi=300` hardcoded | `dpi=self.config.dpi` |
| `plot_clustering_results` | ~1101–1104 | `plt.show()` + `plt.savefig(..., dpi=300)` bare calls | Replace with `self.save_or_show(fig, save_path=..., show=self.config.display, dpi=self.config.dpi)` |
| `plot_clustering_summary_only` | ~1193–1196 | same as above | same fix |
| `create_separate_plots` | ~1251–1254 | Two `self.save_or_show` calls with hardcoded show behavior | Gate on `self.config.display` |

For methods that accept `output_figure_path: Optional[str] = None`: when that is `None` and
`self.config.save_figures` is True, auto-construct the path from
`self.config.output_dir / f"{stem}.{self.config.figure_format}"`.

---

### 4.3 `NileRed_Functions` (NileRedFunctions.py)

**Effort: moderate (~45 min)**

No figures to manage — `AnalysisConfig` needed only for `progress_callback` and `logging_callback`.

#### `__init__` (line 47)

```python
# Before
def __init__(self, camera: str = "ximea", pixel_size: float = None, ...):

# After
def __init__(self, camera: str = "ximea", pixel_size: float = None, ...,
             config: AnalysisConfig = None):
    ...
    self.config = config if config is not None else AnalysisConfig()
```

#### Methods to update

| Method | Lines | Change |
|---|---|---|
| `extract_wavelengths` (parallel loop) | ~521, 543, 550 | After `logger.info(msg)`, add `if self.config.logging_callback: self.config.logging_callback(msg)` |
| `extract_wavelengths` (parallel loop) | ~527–543 | After `completed += 1`, add progress callback: `if self.config.progress_callback: self.config.progress_callback(completed / n_total, f"{label}: {completed}/{n_total}")` |
| `simulate_precision` (wavelength loop) | ~854 | After each `logger.info(...)`, add `logging_callback` call |
| `simulate_precision` | ~854 | `for i in range(n_wavelengths)` — add progress callback `(i+1) / n_wavelengths` |
| HDF5 fitting block | ~1085–1144 | Add `logging_callback` to summary `logger.info` calls |

---

### 4.4 `MultiC_Sim_Funcs_Refactored` (Multicolour_Simulation_Functions.py, line 341)

**Effort: low-moderate (~45 min)**

Already partially GUI-ready (both plot methods accept `show: bool` and `save_path`).
Main gaps: hardcoded `dpi=600`, and bootstrap/simulation loops have no progress hooks.

#### `__init__` (line 351)

```python
# Before
def __init__(self, camera: str = "ximea", pixel_size: float = None, ...):

# After
def __init__(self, camera: str = "ximea", pixel_size: float = None, ...,
             config: AnalysisConfig = None):
    ...
    self.config = config if config is not None else AnalysisConfig()
```

#### Methods to update

| Method | Line | Problem | Fix |
|---|---|---|---|
| `plot_dye_selection_results` | 3377 | `dpi=600` hardcoded; `show=show` (local param) | Replace `dpi=600` with `dpi=self.config.dpi`; replace `show=show` with `show=show if show is not None else self.config.display` (keep backward compat of explicit `show` kwarg) |
| `plot_dye_color_distributions` | 3466 | Same | Same |
| Bootstrap loop (~line 753 etc.) | ~753–968 | `for frame in range(config.n_bootstrap)` — no progress | Add `if self.config.progress_callback: self.config.progress_callback(frame / config.n_bootstrap, f"Bootstrap {frame}/{config.n_bootstrap}")` |
| `optimal_dye_selector_simulated` | ~3160 | `for i in range(n_simulations)` — no progress | Add progress callback |

**Note:** The `show` parameter on the two plot methods should remain for backwards compat — if
explicitly passed it takes precedence; otherwise fall back to `self.config.display`.

---

### 4.5 `postprocess` module (postprocess.py)

**Effort: moderate (~60 min)**

No class; functions receive `config` as an optional parameter.

#### Import addition (top of file)

```python
from Constants import AnalysisConfig  # add
```

#### Functions to update

| Function | Signature change | Changes |
|---|---|---|
| `_plot_drift_analysis` (~line 47) | add `config: AnalysisConfig = None` | Replace hardcoded `dpi=300` and bare `plt.show()` with config-driven calls; existing `display: bool = True` param becomes fallback when config is None |
| `segment_locs_by_rendered_image` (~line 1343) | add `config: AnalysisConfig = None` | Call `config.progress_callback(step/5, "Step N/5: ...")` at each of the 5 steps (lines ~1442, 1460, 1478, 1485, 1630); call `config.logging_callback(msg)` alongside `logger.info(msg)` at summary points |
| `remove_fiducials` (~line 1748) | add `config: AnalysisConfig = None` | Call `config.logging_callback(msg)` alongside the summary `logger.info` calls (~lines 1917–1923) |

For `_plot_drift_analysis`: the existing `display: bool = True` kwarg should remain (backwards compat)
but when `config` is provided, `config.display` takes precedence.

---

## 5. Execution Order

| Step | Target | What changes | Time est. |
|---|---|---|---|
| 1 | `DriftPlotter.__init__` | Add `config` param; store `self.config` | 5 min |
| 2 | `DriftPlotter` 7 plot methods | Replace hardcoded `plt.show()` / `dpi=300` | 90 min |
| 3 | `FiducialDetector.__init__` | Add `config` param; pass to DriftPlotter | 10 min |
| 4 | `FiducialDetector` methods | Add progress + logging callbacks; drop hardcoded dpi | 30 min |
| 5 | `MultiC_Sim_Funcs_Refactored.__init__` | Add `config` param | 5 min |
| 6 | `MultiC_Sim_Funcs_Refactored` plot + loop methods | Fix dpi; add progress callbacks | 25 min |
| 7 | `NileRed_Functions.__init__` | Add `config` param | 5 min |
| 8 | `NileRed_Functions` long loops | Add progress + logging callbacks | 30 min |
| 9 | `postprocess` imports + 3 functions | Add `config` param; fix dpi; add callbacks | 40 min |
| 10 | Import-test all modified files | `importlib.import_module(...)` | 5 min |
| 11 | Update docs + commit | LOG.md, TODO.md, code_refactoring.md | 10 min |

**Total estimated time: ~4.5 hours**

---

## 6. Backwards-Compatibility Contract

- Every `config=` kwarg defaults to `None`.
- `None` → `AnalysisConfig()` → same interactive behaviour as today.
- Existing call sites that pass `output_figure_path=`, `show=`, `display=` as individual kwargs
  continue to work.  Where both an old kwarg and config exist, the **explicit kwarg takes
  precedence** (mirrors the ClusteringConfig/RenderingConfig pattern).
- `progress_callback` and `logging_callback` are only invoked when not `None` — no-op for
  all existing call sites.

---

## 7. Testing

After each class:
1. `importlib.import_module(mod)` — confirms no syntax / import errors
2. Instantiate with `AnalysisConfig()` defaults — confirms existing behaviour unchanged
3. Instantiate with `AnalysisConfig(display=False, save_figures=False)` — confirms no windows open
4. For progress callbacks: pass `lambda f, m: print(f, m)` and confirm it fires during a loop

No new test files needed unless a regression is found.
