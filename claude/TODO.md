# pyS3M TODO

**Last Updated:** 2026-05-18

**Note:** For completed work, see `claude/LOG.md`. For refactoring history, see `claude/code_refactoring.md`.

---

## Active

### Documentation — TOP PRIORITY

**Plan:** `claude/documentation_start.md`  
**Model:** pyRASP Sphinx + Read the Docs setup

- [x] Phase 1 — Foundation: add Sphinx deps to `requirements.txt`, create `src/__init__.py`, create `.readthedocs.yaml`
- [ ] Phase 2 — Scaffold: create `docs/` with `conf.py`, `Makefile`, `index.rst`, `modules.rst`, `getting-started.rst`; verify `make html` locally
- [ ] Phase 3 — API RST files: create per-group `.rst` files (9 groups), run coverage pass, fix autodoc failures
- [ ] Phase 4 — RTD deployment: connect repo, verify automated build
- [ ] Phase 5 — Polish: worked example in getting-started, cross-references, coverage gaps

---

### FRET Post-Hoc Analysis

**Notebook:** `notebooks/fret/DNA_HJ_PostHoc_Changepoints.ipynb`

- [ ] Validate change point results on known FRET switching data

---

### Diffusion-Binding Simulation

**Notebook:** `notebooks/tracking/Stepwise_Assembly_Simulation.ipynb`
**Plan:** `claude/diffusion_binding_sim.md`

- [ ] Fit single-molecule time series from FRET image simulator with 2D Gaussian (SR_Functions pipeline)
- [ ] Step 6b — Full Pipeline Validation: simulate diffusion + binding → image → extract → track → compare recovered k_on, k_off, D_free, D_bound to ground truth

---

### Notebook call sites — D1 wrappers (non-urgent)

24 notebooks still call `fit_SM_data` / `fit_imaging_data` wrappers directly. When wrappers are eventually removed, all sites must be updated to call `_fit_files()` with the appropriate flags. Full catalogue in `claude/SR_Functions_Notebook_Corrections_20260517.md`.

---

## Optional / Low-Priority

### Spot Detection Improvements

Current implementation is production-ready (see `claude/spot_detection_analysis.md`).

- [ ] Add usage examples to `SpotDetectionFunctions.py` docstrings (pfa selection, ADU→photoelectrons, camera types)
- [ ] Validation tests: false-positive rate vs theoretical PFP, ROC curves on synthetic data
- [ ] Poisson Matched Filter (PMF) — 1–5% improvement for DNA-PAINT (>1000 photons/spot)
- [ ] Gamma distribution thresholding for EMCCD very-low-signal regime

### Simulation Performance Optimizations

- [ ] Cache spectral data lookups in `SpectralFunctions.py:get_spectral_data` with `lru_cache` (~15 min, saves 5–10 s/simulation)
- [ ] Parallelize photoelectron generation (`simulation/multicolour.py:1579-1677`) with Numba `prange` (~30 min, saves ~5 min/simulation)
- [ ] Optimise Gaussian smoothing in `sCMOSFunctions.py:gaussian_filter_stack` — investigate separable vs FFT for large sigma (~1 h)
- [ ] Drift correction memory: chunked/memory-mapped coordinate arrays during AIM segmentation

### Testing

- [ ] Increase coverage for `FiducialDetection`, `CoordinateProcessing`, `PlottingBase` edge cases
- [ ] End-to-end integration tests for full fitting → clustering → drift correction workflows
- [ ] Performance benchmarks: time/1000 localisations, memory vs dataset size

### Cleanup

- [ ] Remove `*_backup.py` and `.old` files
- [ ] Standardise import ordering; add missing `__all__` exports to remaining modules
