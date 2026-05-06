# Notebook Correction Plan

**Date:** 2026-05-06
**Scope:** All notebooks in `notebooks/` — audited for breakage after the Tier 3–4 refactoring
**Audit:** 151 notebooks scanned; 0 use `FittingStrategy.STANDARD_DATA` (removed strategy causes no breakage)

---

## Summary of Issues

| Priority | Category | Notebooks affected | Broken? |
|---|---|---|---|
| 1 | Silent logging — no `logging.basicConfig()` | ~20 | No output visible to user |
| 2 | `FittingStrategy.STANDARD` — old default | 10 | No, but wrong strategy used |
| 3 | Direct simulation imports — fragile path setup | 2–3 | Potentially broken |

---

## Standard Notebook Header

Every notebook that imports from `src/` should start with this cell as **cell 1**:

```python
import sys, logging
sys.path.insert(0, '../../src')   # adjust depth to match notebook location

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s — %(levelname)s — %(message)s',
)
```

- The `sys.path.insert` depth varies by notebook location:
  - `notebooks/<subdir>/notebook.ipynb` → `'../../src'`
  - `notebooks/<subdir>/<subsubdir>/notebook.ipynb` → `'../../../src'`
- `logging.basicConfig` must be called **before** any `src` module is imported, otherwise the root logger configuration is ignored and all `logger.info()` / `logger.debug()` calls in src/ are silenced.

---

## Issue 1 — Silent Logging (Priority: HIGH)

### Background

All `print()` calls in `src/` were replaced with `logging` in Tier 3.2. Python's default logging level is `WARNING`, so `logger.info()` and `logger.debug()` calls are invisible unless the notebook configures logging first. Users will see no progress, no frame counts, no fitting feedback — just a silent hang.

### Affected notebooks

Add `logging.basicConfig(level=logging.INFO, ...)` to the standard header in each:

1. `notebooks/calibration/QDot_Variance_Test.ipynb`
2. `notebooks/demosaicing/Demosaicing_then_Fitting.ipynb`
3. `notebooks/demosaicing/Variance_Weighted_Demosaic.ipynb`
4. `notebooks/demosaicing/Variance_Weighted_Demosaic_Updated.ipynb`
5. `notebooks/figures/Figure1_3Dyes_PatternedvsNon.ipynb`
6. `notebooks/figures/Figure1_ResultantPSFs.ipynb`
7. `notebooks/figures/Figure1_maximum_readnoise.ipynb`
8. `notebooks/figures/SI/Debug_Sigma.ipynb`
9. `notebooks/figures/SI/Demosaicing_vs_Fullfit.ipynb`
10. `notebooks/figures/SI/Figure1_3camerapatterns.ipynb`
11. `notebooks/figures/SI/Figure1_CYYMFilter.ipynb`
12. `notebooks/figures/SI/Figure1_DifferentMaskPattern.ipynb`
13. `notebooks/figures/SI/ZWO_vs_Ximea.ipynb`
14. `notebooks/fret/3way_FRET_ImageSim_Cascade.ipynb`
15. `notebooks/fret/FRETFluor_Simulation.ipynb`
16. `notebooks/fret/FRETFluor_TernaryPlot.ipynb`
17. `notebooks/simulation/Pixelsize_FineGrid.ipynb`
18. `notebooks/simulation/Pixelsize_Test.ipynb`
19. `notebooks/testing_notebooks/test_covariance_snr.ipynb`
20. `notebooks/testing_notebooks/testing_initial_guess_fit.ipynb`

### Fix

In each notebook, locate the first cell (the imports cell). Either:

**Option A** — Replace the existing `sys.path` / `import` cell with the standard header above, then re-import everything below it.

**Option B** — Insert a new cell at position 0 containing only:
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s — %(levelname)s — %(message)s',
)
```

Option B is safer for notebooks where the imports cell is complex.

---

## Issue 2 — FittingStrategy.STANDARD Should Be STANDARD_ITER (Priority: MEDIUM)

### Background

`STANDARD_ITER` (2 IRLS model-weight iterations) is now the validated default. `FittingStrategy.STANDARD` still exists and still works, but produces slightly worse fits. Notebooks that pass `strategy=FittingStrategy.STANDARD` explicitly should be updated.

Note: notebooks that rely on `STANDARD` for a published figure or paper comparison should add a comment explaining why, and leave the value unchanged.

### Affected notebooks

| Notebook | Cell content to change |
|---|---|
| `notebooks/demosaicing/Demosaicing_then_Fitting.ipynb` | `strategy=FittingStrategy.STANDARD` → `STANDARD_ITER` |
| `notebooks/figures/Figure1_ResultantPSFs.ipynb` | `strategy=FittingStrategy.STANDARD` → `STANDARD_ITER` |
| `notebooks/figures/Figure1_maximum_readnoise.ipynb` | `strategy=FittingStrategy.STANDARD` → `STANDARD_ITER` |
| `notebooks/figures/SI/Figure1_3camerapatterns.ipynb` | `strategy=FittingStrategy.STANDARD` → `STANDARD_ITER` |
| `notebooks/figures/SI/Figure1_CYYMFilter.ipynb` | `strategy=FittingStrategy.STANDARD` → `STANDARD_ITER` |
| `notebooks/figures/SI/Figure1_DifferentMaskPattern.ipynb` | `strategy=FittingStrategy.STANDARD` → `STANDARD_ITER` |
| `notebooks/figures/SI/Debug_Sigma.ipynb` | `strategy=FittingStrategy.STANDARD` (also references unimplemented `STANDARD_FIXEDSIGMA` — remove that mention) |
| `notebooks/fret/FRETFluor_Simulation.ipynb` | `strategy=FittingStrategy.STANDARD` → `STANDARD_ITER` |
| `notebooks/simulation/Pixelsize_FineGrid.ipynb` | `strategy=FittingStrategy.STANDARD` → `STANDARD_ITER` |
| `notebooks/simulation/Pixelsize_Test.ipynb` | `strategy=FittingStrategy.STANDARD` → `STANDARD_ITER` |

### Fix

Replace occurrences:
```python
# before
strategy=FittingStrategy.STANDARD
# after
strategy=FittingStrategy.STANDARD_ITER
```

If `FittingStrategy` was imported explicitly:
```python
# before
from src.Multicolour_Simulation_Functions import FittingStrategy
# still works — STANDARD_ITER is in the enum
```

---

## Issue 3 — Fragile Direct Simulation Imports (Priority: MEDIUM)

### Background

Two notebooks import `DiffusionSimulation` and `Multicolour_Simulation_Functions` by bare name (relying on `sys.path` being set in an earlier cell). The shims at `src/DiffusionSimulation.py` and `src/Multicolour_Simulation_Functions.py` re-export everything correctly, so these imports will work as long as `src/` is in `sys.path` first. The fragility is that import cells and sys.path cells are separate — if cells run out of order, the import fails.

### Affected notebooks

**`notebooks/simulation/DiffusionBinding_BasicTest.ipynb`**
- Imports: `from DiffusionSimulation import (DiffusionSimulator2D, compute_msd_from_trajectory, estimate_D_from_msd)`
- Verify `sys.path.insert(0, '../../src')` is in cell 1 before all these imports.
- All three names (`DiffusionSimulator2D`, `compute_msd_from_trajectory`, `estimate_D_from_msd`) are in the shim ✓

**`notebooks/tracking/Stepwise_Assembly_Simulation.ipynb`**
- Cell 1: `from DiffusionSimulation import (DiffusionSimulator2D, CameraAdapter)` ✓
- Cell 6: `from DiffusionSimulation import BindingKinetics` ✓
- Cell 17: `from Multicolour_Simulation_Functions import MultiC_Sim_Funcs` ✓
- All names in the shim lists ✓
- Ensure the sys.path cell is cell 0 and runs before any of these.

### Fix

Consolidate all `sys.path` setup and logging config into a single cell 0, then keep all imports in cell 1. Verify that no import cell appears before the sys.path cell.

Preferred pattern for these notebooks (bare-name import style):
```python
# Cell 0
import sys, logging
sys.path.insert(0, '../../src')
logging.basicConfig(level=logging.INFO, format='%(name)s — %(levelname)s — %(message)s')

# Cell 1
from DiffusionSimulation import DiffusionSimulator2D, CameraAdapter, BindingKinetics
from Multicolour_Simulation_Functions import MultiC_Sim_Funcs
```

---

## What Is NOT Broken

- **`FittingStrategy.STANDARD_DATA`** — not referenced in any notebook (0 occurrences). Removal causes no breakage.
- **Clustering subpackage** — all clustering methods remain on the `extract_SMs` class; no notebook imports internal clustering classes directly.
- **drift_correction/ subpackage** — shim at `DriftCorrectionFunctions.py` re-exports everything; no notebook broken.
- **pathlib migration** — fully internal to `src/`; no notebook impact.
- **AnalysisConfig** — optional parameter added to major classes; all notebooks work without passing it.
- **`from src import Multicolour_Simulation_Functions` pattern** — shim re-exports `MultiC_Sim_Funcs`, `FittingStrategy`, `SimulationConfig`, etc. Works correctly.
- **`from src.Multicolour_Simulation_Functions import FittingStrategy, SimulationConfig`** — works via shim.

---

## Execution Order

1. Fix Issue 1 (logging) across all 20 notebooks — highest user impact, mechanical change.
2. Fix Issue 2 (STANDARD → STANDARD_ITER) across 10 notebooks — only skip for paper figures if needed.
3. Fix Issue 3 (fragile imports) in 2 notebooks — verify by running the notebooks.
4. After each notebook is fixed, run all cells top-to-bottom and confirm no errors.

**Estimated effort:** ~1 day for all three categories.
