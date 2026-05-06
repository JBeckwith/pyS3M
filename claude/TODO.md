# pyBayerSMLM TODO

**Last Updated:** 2026-05-06 (Tier 4.2 complete; STANDARD_DATA restored for benchmarking)

**Note:** For completed work, see LOG.md

---

## Active Projects

### Priority 1: FRET Post-Hoc Analysis (ACTIVE)

**Status:** 🔨 IN PROGRESS - Validation
**Notebook:** `notebooks/fret/DNA_HJ_PostHoc_Changepoints.ipynb`

**Completed:**
- ✅ Initial guess filtering (A_R, A_G, A_B all ~0.33)
- ✅ Joint multivariate change point detection on [A_R, A_G] with BIC penalty
- ✅ 3-panel visualisation with segment-mean overlays
- ✅ GIF generation pipeline (memory-mapped TIFF, per-punctum demosaicing)
- ✅ PlottingBase `make_animated_gif` layout warning fix
- ✅ Two-panel GIF: raw demosaiced (top) + super-resolved coloured fit reconstruction (bottom)
- ✅ PIL-based GIF writing (replaces slow matplotlib FuncAnimation)

**Remaining:**
- [ ] Validate change point results on known FRET switching data

---

### Priority 1: Diffusion-Binding Simulation (ACTIVE)

**Status:** 🔨 IN PROGRESS - Stepwise Assembly Simulation notebook
**Goal:** Simulate multicolor molecules with binding/unbinding for testing pyBayerSMLM pipeline

**Architecture:**
```
DiffusionSimulator2D → CameraAdapter → Multicolour_Simulation_Functions → TIFF
                            ↓                           ↓
                  (Poisson brightness,           Localization
                   blinking, spectral)           Extraction
                                                      ↓
                                                Track Assignment
                                                  (NEW: Step 6a)
                                                      ↓
                                              Pipeline Validation
```

**Completed (Steps 1-5):**
- ✅ **Step 1: Core 2D Diffusion** (Commit 82c6f36)
  - Realistic Brownian motion (Michalet & Berglund 2012)
  - Motion blur, dynamic localization error
  - OLSF MSD validation: D recovered within 1.8% error
  - Proper covariance structure preserved

- ✅ **Step 2: Binding Kinetics** (Commit b64ed83)
  - Gillespie algorithm with 2D k_on/k_off matrices
  - Color-specific binding rules (R-G, G-B, etc.)
  - Stochastic binding/unbinding events
  - Bound pairs move together at D_bound
  - Validated: 5 bind, 4 unbind events over 3s
  - Average lifetime 1431 ms vs expected 1000 ms ✓

- ✅ **Step 3: Multicolor System**
  - Spectral profiles in Molecule class (A_R, A_G, A_B)
  - Configurable binding via k_on/k_off matrices
  - Event tracking for analysis

- ✅ **Step 4: Camera Imaging Adapter** (Nov 15, 2025)
  - `CameraAdapter` class bridges simulation to imaging
  - `prepare_localisations_for_imaging()`: trajectory → x0y0 format
  - `generate_tiff_stack()`: full TIFF movie generation (Bayer-filtered)
  - `generate_ground_truth_rgb_video()`: perfect RGB visualization
  - **Poisson brightness sampling**: realistic per-frame photon variation
  - **Blinking support**: per-color probabilities (extensible)
  - **Spectral profiles**: molecule-specific A_R/G/B handling
  - **Ground truth videos**: Colored Gaussians for validation
  - Reuses existing PSF/Bayer/sCMOS pipeline (no duplication!)
  - Test: Bayer (5 molecules, 10 frames, Poisson std = 0.93×) ✓
  - Test: RGB (5 colors, 30 frames, spectral ordering correct) ✓

- ✅ **Step 5: Microscopic Binding Framework** (Dec 8, 2025)
  - Implemented Fange et al. (2010) microscopic reaction-diffusion framework
  - Scale-dependent mesoscopic rates q_a(h), q_d(h) replace volume-based rates
  - **60,000× speedup**: binding now μs-ms timescale (not seconds!)
  - Detailed balance preserved: K = k/γ exact to machine precision
  - Fully integrated into DiffusionSimulator2D
  - Tests: Microscopic rates, propensities, full simulation ✓
  - Percentile-based intensity scaling for RGB video brightness ✓

**Completed (Step 6a): Spectral-Assisted LAP Track Assignment**
- ✅ **Step 6a: Multicolor Track Assignment** (COMPLETED - Feb 2026)
  - **Document:** claude/implement_track_assignment.md (comprehensive analysis)
  - **Implementation:** `src/SM_extractionfunctions.py`
    - `spectral_lap_link()` — core LAP linking engine with spatial + spectral cost
    - `extract_single_molecules_spectral_lap()` — public API method (mirrors existing `extract_single_molecules_linked()`)
  - **Algorithm:** Jaqaman et al. (2008) augmented cost matrix with birth/death, solved via `scipy.optimize.linear_sum_assignment`
  - **Cost function:** `w_spatial * (d_spatial / max_distance)² + w_spectral * (d_spectral / spectral_tol)²`
  - **Literature citations:** Jaqaman (2008), Crocker & Grier (1996), Chenouard (2014), Sergé (2008)
  - **Example notebook:** `notebooks/tracking/Example_Track_Analysis.ipynb`
  - **Tests:** 6/6 pass in `unit_tests/test_spectral_lap_tracker.py`

**Completed (Stepwise Assembly notebook — Feb 19–20, 2026):**
- ✅ 16:9 widescreen FOV (53.3 × 30 µm, same 1600 µm² area, 773 × 435 px)
- ✅ Bayer stack orientation fix (transpose (n,W,H) → (n,H,W) after gen_camera_image_stack)
- ✅ HDF5 trajectory save (`stepwise_assembly_trajectories.h5`)
- ✅ Side-wipe GIF (`make_wipe_gif`): Bayer grayscale → HSV-enhanced RGB ground truth — simplified to use pre-computed `rgb_bright`
- ✅ Fixed `duration_ms` bug in `make_wipe_gif` (was `* gif_stride`, now just `1000/fps`)
- ✅ Fixed `duration_ms` bug in plain GIF cell (cell-16): `int(1000/gif_fps * gif_stride)` → `int(round(1000/gif_fps))`; added global PIL palette for faster saving
- ✅ Fixed `traj[:n_frames]` slicing in Bayer simulation cell (trajectory has n_frames+1 entries)
- ✅ Fixed `gen_spatial_PSF` to support rectangular (non-square) FOVs: added `y` parameter, rewrote body with `np.outer`; added `y = np.arange(h)` in `gen_camera_image_stack`
- ✅ Temporal averaging for Bayer GIF frames: each displayed frame averages `gif_stride` consecutive TIFF pages (100 ms effective exposure), eliminating choppiness from fast diffusion (D=10 µm²/s)
- ✅ dt and t_exposure: 20 ms → 50 ms; duration: 60 s → 120 s (2400 frames at new dt)
- ✅ Fixed `binding_events`/`unbinding_events` NameError in cell-11: extracted from `simulator.binding_kinetics` at top of HDF5-save cell before first use

**Remaining — Stepwise Assembly notebook:**
- [ ] Fit the FRET image simulator single-molecule time series with 2D Gaussian (SR_Functions pipeline) — pending from previous session

**Remaining Steps (Step 6b):**
- [ ] **Step 6b: Full Pipeline Validation**
  - Simulate diffusion + binding → Image → Extract → Track → Validate
  - Compare recovered trajectories to ground truth
  - Validate binding event detection from localizations
  - Measure accuracy: k_on, k_off, D_free, D_bound recovery
  - Trajectory statistics and binding kinetics

**Key Features Implemented:**
- 2D k_on/k_off matrices for flexible color pairing
- Binding radius threshold (~100 nm for contact)
- Coupled diffusion (D switches based on binding state)
- Chunk-based processing preserves covariance
- Event logging (all binding/unbinding with timestamps)
- **Poisson photon sampling per frame per molecule**
- **Blinking simulation with per-color probabilities**
- **Frame-by-frame brightness variation**

**Implementation Details:**
- `Molecule`: color, spectral_profile, D_free, D_bound, is_bound, bound_partner
- `BindingKinetics`: Gillespie algorithm, propensity calculations, event tracking
- `LangevinDiffusion2D`: Realistic motion with camera effects
- `DiffusionSimulator2D`: Multi-molecule simulator with binding support
- `CameraAdapter`: Trajectory → TIFF converter with Poisson/blinking

**Files:**
- src/simulation/diffusion.py (canonical location, 1885+ lines, includes microscopic framework)
- src/DiffusionSimulation.py (backward-compat shim → simulation/diffusion.py)
- src/simulation/multicolour.py (canonical location, 3630+ lines)
- src/Multicolour_Simulation_Functions.py (backward-compat shim → simulation/multicolour.py)
- notebooks/simulation/DiffusionBinding_BasicTest.ipynb (validation)
- notebooks/tracking/Stepwise_Assembly_Simulation.ipynb (full example)
- unit_tests/test_simulator_microscopic_integration.py (microscopic tests)
- unit_tests/claude/test_microscopic_propensities.py (propensity validation)
- unit_tests/claude/test_mesoscopic_rates.py (rate calculation tests)
- claude/diffusion_binding_sim.md (detailed plan)
- claude/MICROSCOPIC_FRAMEWORK_SUMMARY.md (complete usage guide)

**See:** claude/MICROSCOPIC_FRAMEWORK_SUMMARY.md for microscopic framework details

---

## Pending Tasks

### Priority 2: Spot Detection Validation & Documentation

**Status:** 📋 OPTIONAL IMPROVEMENTS
**Priority:** LOW-MEDIUM
**Analysis Complete:** ✅ See `claude/spot_detection_analysis.md`

**Background:**
Comprehensive analysis completed comparing `SpotDetectionFunctions.py` implementation against Hekrdla et al. (2025) paper. **Current implementation is production-ready** and correctly handles camera noise statistics (Poisson, sCMOS). Uses Matched Filter (MF) instead of theoretically optimal Poisson Matched Filter (PMF), which provides 95-99% of optimal performance for typical SMLM conditions.

**Optional Improvements (Low Priority):**

#### Priority 1: Documentation Enhancements
- [ ] Add usage examples to `SpotDetectionFunctions.py` docstrings:
  - How to compute variance for sCMOS cameras
  - How to convert ADU to photoelectrons (formula: `photoelectrons = (ADU - offset) / gain`)
  - How to choose `pfa` parameter (typical range: 1e-6 to 1e-3)
  - Example workflows for different camera types
- [ ] Create user guide section on spot detection:
  - Camera noise models and when to use each
  - Parameter selection guidelines
  - Troubleshooting common issues

#### Priority 2: Validation Tests
- [ ] Add unit tests comparing to paper's methodology:
  - Benchmark false positive rate vs. theoretical PFP
  - ROC curve analysis on synthetic data
  - Validate CA-CFAR background estimation accuracy
- [ ] Create synthetic test datasets:
  - Known signal/background ratios
  - Different noise models (Poisson, sCMOS)
  - Ground truth for validation

#### Priority 3: Optional Algorithm Enhancements
- [ ] **Poisson Matched Filter (PMF)** implementation:
  - Expected improvement: 1-5% better detection for high signal (a/b > 10)
  - Use case: DNA-PAINT with long binding times (>1000 photons)
  - Effort: Medium (requires signal amplitude estimation)
- [ ] **Gamma distribution thresholding** for EMCCD low signal:
  - Reduces false positives for very low signal (a < 10 photons)
  - Use case: Fast kinetics, live cell imaging
  - Effort: Medium (requires gamma distribution implementation)
- [ ] **3D spot detection**:
  - Extend to 3D PSFs (astigmatism, double-helix)
  - Required for 3D SMLM
  - Effort: High (requires 3D PSF models, CA-CFAR extension)

**Files:**
- Analysis: `claude/spot_detection_analysis.md`
- Implementation: `src/SpotDetectionFunctions.py` (1067 lines)
- Integration: `src/SR_Functions.py` (lines 100, 699, 1193, 1787)
- Paper: Hekrdla et al. (2025) Nature Communications 16, 601

**Recommendation:** Focus on documentation (Priority 1) before considering algorithmic enhancements. Current implementation is correct and sufficient for typical SMLM experiments.

---

### Priority 2: Spot Detection - Bayer-Patterned Raw Data Adaptation

**Status:** ❌ REMOVED (Dec 19, 2025)
**Decision:** Bayer-aware detection performs **worse** than variance-aware demosaic
**Analysis:** See `claude/spot_detection_analysis_bayeradaptation.md`

**Background:**
Investigated per-channel detection on raw Bayer data to avoid demosaicing artifacts and preserve Poisson noise independence. Hypothesis: demosaicing introduces spatial correlations that degrade matched filter performance.

**Implementation & Testing (Commits: 88521e4, d951baa, afadcb1, 2a4f07d)**

Implemented complete Bayer-aware spot detection pipeline (~393 lines):
- Per-channel extraction (R/G/B from raw Bayer pattern)
- Independent matched filter detection on each channel
- Coordinate mapping back to full resolution
- Cross-channel duplicate removal

**Test Results:**
- ✅ Core functionality working: Single-spot test passed (0.33 px error)
- ❌ **Performance comparison: Bayer-aware detection WORSE than demosaic**

**Decision Rationale (Dec 19, 2025):**

After testing, the standard **variance-aware demosaic approach outperforms** Bayer-aware detection:
- Variance-weighted demosaicing already accounts for noise statistics
- Per-channel detection loses information by operating on subsampled channels
- Coordinate mapping introduces localization errors
- Standard pipeline is simpler and better tested

**Removed Files (Commit: 06580e0):**
- `src/BayerSpotDetection.py` (393 lines) - Core implementation
- `unit_tests/test_bayer_*.py` (2000+ lines) - All validation tests
- Analysis preserved: `claude/spot_detection_analysis_bayeradaptation.md`

**Lesson Learned:**
Variance-aware demosaicing (already implemented in `sCMOSFunctions.py`) is the correct approach. The "demosaicing artifacts" concern was overstated - proper variance weighting mitigates the correlation issues.

**Recommendation:**
Continue using standard variance-aware demosaic pipeline (`sCMOSFunctions.variance_aware_malvar_demosaic()`) for all Bayer-patterned data. No further work needed on Bayer-specific detection.

---

### Priority 2: Variance-Aware Demosaicing - Validation & Documentation

**Status:** ✅ VALIDATED (Dec 19, 2025)
**Decision:** Variance transformation math confirmed correct: `variance_pe = variance_ADU / gain²`

**Background:**
After removing Bayer-aware detection, validated that the existing variance-aware demosaicing approach correctly handles the transformation from ADU to photoelectron space for sCMOS cameras with spatially-varying gain.

**Mathematical Validation (Dec 19, 2025):**

Confirmed that variance transformation requires dividing by `gain²`, not `gain`:
- **Units analysis**: variance in ADU² → photoelectrons²
  - `variance_ADU` has units: ADU²
  - `gain` has units: ADU/photoelectron
  - `variance_pe = variance_ADU / gain²` → (ADU²) / (ADU/pe)² = pe² ✓

**Physical Correctness:**
- For sCMOS cameras with spatially-varying gain, demosaicing **must** interpolate in photoelectron space
- Interpolating in ADU space mixes values scaled by different gains → incorrect results
- Example: Two pixels with 100 ADU but different gains (2.0 vs 1.0) represent 50 pe vs 100 pe
  - Correct interpolation (pe space): (50 + 100)/2 = 75 pe
  - Wrong interpolation (ADU space): (100 + 100)/2 / 1.5 = 67 pe (10% error)

**Test Comparison:**
Compared two approaches:
1. **v1 (CORRECT)**: Convert to pe → weight by inverse variance → demosaic
2. **v2 (INCORRECT)**: Weight in ADU → convert to pe → demosaic

Results: v2 failed for spatially-varying gain with 25 pe error (10% difference)

**Implementation & Testing (Commits: b91d46f, [uncommitted]):**

1. **Enhanced Documentation** (`src/sCMOSFunctions.py:42-84`)
   - Added detailed docstring explaining physical reasoning
   - Included example showing why gain² is necessary
   - Documented units and transformations

2. **Simulation Ground Truth** (`src/Multicolour_Simulation_Functions.py`)
   - Fixed `return_normal_image` bug (was placeholder `pass`)
   - Added `return_photoelectrons` parameter for validation
   - Returns raw photoelectrons instead of ADU for ground truth comparison

3. **Validation Test Notebook** (`notebooks/test_variance_aware_demosaicing.ipynb`)
   - Comprehensive validation against ground truth
   - Quantitative metrics: RMSE, correlation, SNR
   - Visual analysis: difference maps, scatter plots
   - Peak intensity recovery analysis
   - **Status**: Created but not yet run (awaiting user testing)

**Files Modified:**
- `src/sCMOSFunctions.py` - Enhanced documentation (lines 42-84)
- `src/CalibrationFunctions.py` - Fixed variance units in print (ADU² not ADU)
- `src/Multicolour_Simulation_Functions.py` - Added return_photoelectrons, fixed normal_image
- `notebooks/test_variance_aware_demosaicing.ipynb` - Validation test (ready to run)

**Validation Strategy:**
1. Generate ground truth: flat QE image in photoelectrons (no Bayer pattern)
2. Generate Bayer image: same simulation with Bayer filter in ADU
3. Apply variance-aware demosaicing to Bayer image
4. Compare demosaiced photoelectrons to ground truth
5. Metrics: RMSE, correlation, peak intensity recovery

**Expected Performance:**
- High correlation (>0.95) due to proper variance weighting
- Some degradation from Bayer subsampling (25-50% pixel usage per channel)
- Systematic (not random) errors suggesting calibratable bias

**Recommendation:**
Continue using `variance_aware_malvar_demosaic()` with confidence. The math is correct, and the test framework is in place for future validation.

---

### Priority 2: Codebase Refactoring (ongoing — see `claude/code_refactoring.md`)

**Status:** 📋 Tier 3 in progress — Tiers 2, 3.1/3.2/3.4, 4.2 complete

**Completed (May 1–6, 2026):**
- ✅ **3.1** `RenderingConfig` dataclass in `render.py`
- ✅ **3.2** Replace `print()` with `logging` throughout (18 files, 581 calls)
- ✅ **3 (config)** `ClusteringConfig` dataclass wired into all five extraction methods
- ✅ **3.4** `AnalysisConfig` threaded into `FiducialDetector`, `DriftPlotter`, `MultiC_Sim_Funcs_Refactored`, `NileRed_Functions`, `_plot_drift_analysis`, `segment_locs_by_rendered_image`, `remove_fiducials`; progress/logging callbacks added
- ✅ **4.2** `simulation/` subpackage — `diffusion.py` and `multicolour.py` with backward-compat shims

**Remaining Tier 3:**
- [ ] **3.3** Comprehensive type hints on public methods (all files, ~3 days)

**Remaining Tier 4:**
- [ ] **4.1** High-level `AnalysisPipeline` orchestrator (GUI entry point)

**Pending benchmark (May 2026):**
- [ ] Run `notebooks/figures/SI/Standard_vs_ITER_vs_DATA.ipynb` to compare STANDARD / STANDARD_ITER / STANDARD_DATA across photon levels and pick default fitting strategy

---

## Pending Tasks

### Priority 2: Simulation Performance Optimizations

#### 2.1 High-Priority Optimizations (Est. 50-60 min total savings, ~2 hours effort)

##### 2.1.1 Cache Spectral Data Lookups ⚡ QUICK WIN
**Status:** 📋 PLANNED
**Effort:** 15 minutes
**Expected savings:** 5-10 seconds per simulation

**Location:** `src/SpectralFunctions.py:get_spectral_data`

**Problem:** Database queries repeated for same dyes/filters across 200 photon levels

**Solution:**
```python
@lru_cache(maxsize=128)
def _get_spectral_data_cached(self, names_tuple, wavelength_hash, data_type):
    # Actual database query
    pass
```

##### 2.1.2 Parallelize Photoelectron Generation
**Status:** 📋 PLANNED
**Effort:** 30 minutes
**Expected savings:** 5 minutes per simulation

**Location:** `src/Multicolour_Simulation_Functions.py:1579-1677`

**Approach:** Use Numba `prange` to parallelize across frames
**Expected speedup:** 1.2-1.3× additional speedup
**Savings:** 1.5s per photon level × 200 = 5 minutes

##### 2.1.3 Optimize Gaussian Smoothing
**Status:** 📋 PLANNED
**Effort:** 1 hour
**Expected savings:** ~20s per photon level × 200

**Location:** `src/sCMOSFunctions.py:gaussian_filter_stack`

**Investigation needed:**
- Separable convolution vs FFT-based for large sigma
- Batch processing multiple images together
- Pre-compute convolution kernel

#### 2.2 Testing Checklist (Before Production Deployment)
- [ ] Run `test_bootstrap_parallel.py` - verify statistical equivalence
- [ ] Test with various bootstrap sample sizes (1K, 10K, 100K)
- [ ] Verify with all three dyes (ATTO 488, 565, 647N)
- [ ] Test with all three camera types (Bayer, Sharp, Standard)
- [ ] Run full 200-photon-level simulation to confirm total savings
- [ ] Check memory usage hasn't increased
- [ ] Verify saved results match previous simulations

#### 2.3 Optimize Drift Correction Memory Usage
**Status:** 📋 PLANNED (Low priority)
**Current issue:** Full coordinate arrays held in memory during AIM segmentation

**Solutions:**
- Implement memory-mapped arrays for large datasets
- Add chunked processing for frames
- Use generator patterns where possible

**Expected savings:** 50-80% memory reduction for large datasets

---

### Priority 3: Code Organization

#### 3.1 Create Package Structure
**Current:** All files in `src/` directory — see `code_refactoring.md` §3 for proposed submodule splits
(`drift_correction/`, clustering under `SM_extractionfunctions/`, etc.)

**Benefits:**
- Clearer module organization
- Better import paths
- Easier to navigate codebase

#### 3.2 Standardize Function Naming
**Current inconsistencies:**
- Mix of snake_case and camelCase
- Inconsistent verb prefixes (get_, calculate_, compute_)

**Recommendations:**
- Use snake_case consistently
- Standardize prefixes:
  - `get_`: retrieve existing data
  - `calculate_`: perform computation
  - `create_`: instantiate new object
  - `apply_`: modify in-place
  - `validate_`: check correctness

#### 3.3 Add Type Hints Throughout
**Current state:** Partial type hints

**Benefits:**
- Better IDE support
- Catch errors earlier
- Self-documenting code

**Priority files:**
- DriftCorrectionFunctions.py
- All algorithm modules (AIM, RCC, Fiducial, Coordinate)
- PlottingBase.py

---

### Priority 4: Documentation

#### 4.1 Add Module-Level Documentation
**Need:**
- Comprehensive module docstrings
- Usage examples
- API reference

**Format:**
```python
"""
Module: drift.AIMAlgorithm

Adaptive Intersection Maximization (AIM) drift correction algorithm.

Overview
--------
AIM performs drift correction by...

Key Classes
-----------
- AIMAlgorithm: Main algorithm implementation

Key Functions
-------------
- run_aim_2d(): 2D drift correction
- run_aim_3d(): 3D drift correction

Usage Example
-------------
    from drift.AIMAlgorithm import AIMAlgorithm

    aim = AIMAlgorithm()
    drift_x, drift_y, meta = aim.run_aim_2d(locs, params)

Performance Notes
-----------------
- Best for sparse data (<100 locs/frame)
- Supports multithreading
- Memory usage: O(n_frames * n_locs_per_frame)

References
----------
[1] Paper citation...
"""
```

#### 4.2 Create User Guide
**Topics needed:**
- Installation
- Quick start
- Drift correction methods comparison
- Advanced configuration
- Troubleshooting

#### 4.3 Add API Documentation
**Tool:** Sphinx
**Output:** HTML documentation
**Sections:**
- API reference (auto-generated)
- Tutorials
- Examples gallery

---

### Priority 5: Testing

#### 5.1 Increase Test Coverage
**Current coverage:** ~60% (estimated)
**Target:** 90%+

**Missing tests:**
- FiducialDetection methods
- CoordinateProcessing methods
- PlottingBase edge cases
- ImportManager error handling

#### 5.2 Add Integration Tests
**Needed:**
- End-to-end workflows
- Multi-method comparisons
- Real data tests

#### 5.3 Add Performance Benchmarks
**Metrics to track:**
- Time per 1000 localizations
- Memory usage by dataset size
- Rendering speed by point count

---

## Future Enhancements (Phase 3+)

### Feature Additions

#### 1. GPU Acceleration
- Use CuPy for array operations
- GPU-accelerated rendering
- CUDA-based drift correction

#### 2. Real-time Drift Correction
- Stream processing support
- Incremental drift updates
- Live visualization

#### 3. Machine Learning Integration
- ML-based fiducial detection
- Drift prediction
- Quality assessment

### Architecture Improvements

#### 1. Plugin System
- Modular drift correction algorithms
- Custom rendering backends
- Extensible plotting themes

#### 2. Configuration Management
- YAML/JSON config files
- Profile-based settings
- Environment-specific defaults

#### 3. Parallel Processing
- Distributed computing support
- Multi-GPU support
- Cloud processing integration

---

## Cleanup Tasks

### Remove Backup Files
- [ ] Remove `*_backup.py` files
- [ ] Remove `.old` files once deprecation cycle complete

### Code Quality
- [ ] Standardize import ordering across all modules
- [ ] Add missing `__all__` exports
- [ ] Update copyright headers

---

## Dependencies

### Required Dependencies (Core Functionality)
- numpy
- scipy
- matplotlib
- numba

### Optional Dependencies (Enhanced Features)
- datashader (large dataset rendering)
- pandas (data manipulation)
- colorcet (color schemes)
- seaborn (enhanced plotting)
- plotly (interactive plots)
- bokeh (web-based visualization)
- mpltern (ternary plots)

### Dependency Management Recommendations
1. Create `requirements.txt` with pinned versions
2. Create `requirements-optional.txt` for optional features
3. Add `requirements-dev.txt` for development tools
4. Consider using `poetry` for dependency management

---

## Notes

**For completed work details, session summaries, implementation notes, performance metrics, and architecture decisions, see LOG.md**
