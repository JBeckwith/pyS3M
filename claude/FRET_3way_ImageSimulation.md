# Three-Way FRET Image Simulation — Planning Document

**Date:** 2026-04-18  
**Goal:** Simulate Bayer-camera images of three-dye FRET systems under **alternating
488 nm / 561 nm excitation (ALEX)**, then perform a posteriori distance recovery
to determine how well all three inter-dye distances (D–A1, D–A2, A1–A2) can be
recovered and where degenerate / ambiguous solutions arise.

---

## 1. Physical Model

### 1.1 Dye roles in the ALEX scheme

| Role | Example dyes | Principal excitation |
|------|-------------|---------------------|
| D (donor) | Alexa Fluor 488 | 488 nm |
| A1 (intermediate acceptor) | Cy3, ATTO 550 | 561 nm |
| A2 (red acceptor) | ATTO 647N, Cy5, ATTO 655 | neither (direct excitation negligible) |

The ALEX scheme alternates two laser lines per localisation event:
- **488 nm frame**: D is the primary absorber; FRET D→A1, D→A2, and cascade A1→A2 all active.
- **561 nm frame**: A1 is the primary absorber; only A1→A2 cascade is active (D barely absorbs at 561 nm).

This gives **six per-molecule observables** (N_D, N_A1, N_A2 per laser) for **three unknowns** (E1, E2, E12), making the system *overdetermined* — degeneracy is broken.

### 1.2 FRET efficiencies

**Competing FRET at the donor (488 nm frame):**

```
E1  = (r1/R0_DA1)^-6 / [1 + (r1/R0_DA1)^-6 + (r2/R0_DA2)^-6]
E2  = (r2/R0_DA2)^-6 / [1 + (r1/R0_DA1)^-6 + (r2/R0_DA2)^-6]
```

**Cascade at A1 (both frames):**

```
E12 = (r12/R0_A1A2)^-6 / [1 + (r12/R0_A1A2)^-6]
```

### 1.3 Photon budget — 488 nm excitation frame

```
N_D^488   = I0_D × (1 − E1 − E2)
N_A1^488  = (I0_D/QY_D) × E1 × QY_A1 × (1 − E12)  +  I0_D × δ_A1^488 × QY_A1
N_A2^488  = (I0_D/QY_D) × (E2 + E1·E12) × QY_A2  +  I0_D × δ_A2^488 × QY_A2
```

where the δ terms are **direct excitation corrections** (spectral crosstalk):

```
δ_A1^488 = ε_A1(488) / ε_D(488)      (relative molar extinction at 488 nm)
δ_A2^488 = ε_A2(488) / ε_D(488)
```

These come from the `exc_table` already computed in `3way_FRET_simulation_with488.ipynb`.

### 1.4 Photon budget — 561 nm excitation frame

At 561 nm, A1 is the primary absorber.  D has negligible absorption (e.g. AF488
absorbs < 5% of its peak at 561 nm).  A2 is also negligible at 561 nm.

```
N_D^561   ≈ I0_D × δ_D^561 × (1 − E1 − E2)       (usually << N_A1^561)
N_A1^561  = I0_A1 × (1 − E12)
N_A2^561  = I0_A1 × E12 × (QY_A2 / QY_A1)        +  small δ_A2^561 term
```

where:
```
I0_A1 = unquenched A1 photon budget (561 nm frame)
       = set by 561 nm laser power × ε_A1(561) × QY_A1 × (exposure time)
δ_D^561   = ε_D(561) / ε_A1(561)     (relative donor absorption at 561 nm)
δ_A2^561  = ε_A2(561) / ε_A1(561)
```

**Key:** N_A2^561 / N_A1^561 = E12 × QY_A2/QY_A1 (after direct exc correction).  
This directly gives E12 — and hence r12 — independently of r1 and r2.

### 1.5 Two-step distance recovery (exact under noiseless conditions)

**Step 1 — from 561 nm frames: recover E12**

```python
# After subtracting direct excitation:
N_A1_561_corr = N_A1^561 - I0_A1 * delta_D561 * (...)   # usually tiny
N_A2_561_corr = N_A2^561 - I0_A1 * delta_A2561 * QY_A2

E12_est = (N_A2_561_corr / QY_A2) / (N_A1_561_corr/QY_A1 + N_A2_561_corr/QY_A2)
r12_est = R0_A1A2 * (1/E12_est - 1)**(1/6)
```

**Step 2 — from 488 nm frames + known E12: recover E1, E2**

```python
# Subtract direct excitation of A1 under 488 nm
N_A1_488_FRET = N_A1^488 - I0_D * delta_A1488 * QY_A1
E1_eff = N_A1_488_FRET / (I0_D/QY_D * QY_A1)   # = E1 * (1 - E12)
E1_est = E1_eff / (1 - E12_est)                 # now solvable since E12 known

gamma = 1 - N_D^488 / I0_D                       # = E1 + E2
E2_est = gamma - E1_est

r1_est = R0_DA1 * (E1_est / (gamma * E1_est / (E1_est+E2_est) ... ))  # numerical
r2_est = R0_DA2 * (E2_est / ... )                # numerical inversion of competitive FRET
```

For the competitive FRET inversion, r1 and r2 are found by solving:
```
E1 = x1^6 / (1 + x1^6 + x2^6)
E2 = x2^6 / (1 + x1^6 + x2^6)
```
where x1 = R0_DA1/r1, x2 = R0_DA2/r2.  From E1/E2 = x1^6/x2^6 × (R0_DA1/R0_DA2)^-6
this reduces to a 1D problem once E1+E2 is known:
```
r2 = r1 * (E1/E2)^(1/6) * (R0_DA2/R0_DA1)
gamma = x1^6 / (1 + x1^6 + x2^6)  →  solve for r1 given r2(r1)
```

**Full ML alternative**: Rather than the two-step analytic recovery, fit all six
observables jointly using `scipy.optimize.minimize` over (E1, E2, E12) — this
propagates correlated noise optimally and is preferred for noisy data.

### 1.6 Residual degeneracy with ALEX

With ALEX the algebraic degeneracy from §1.5 of the previous version of this
document is fully broken. The remaining sources of uncertainty are:

| Limit | When dominant | Mitigation |
|-------|--------------|-----------|
| Shot noise on N_A1^561 | Low I0_A1, E12 close to 0 or 1 | Increase 561 laser power or exposure |
| Spectral unmixing ill-conditioning | Dye spectra nearly collinear in [B,G,R] | Check condition number of M per triad |
| I0_D or I0_A1 unknown | I0 not calibrated | Use bleach-step calibration or donor-only control |
| Direct excitation corrections uncertain | Dye spectra imprecise | Use fpbase values; assess sensitivity |
| Triangle inequality | Degenerate (r1,r2,r12) configs | Plot feasible posterior volume |

---

## 2. Simulation Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  Setup (run once per triad)                                   │
│  • R0_DA1, R0_DA2, R0_A1A2 from forster_radius_nm()         │
│  • rgb_D, rgb_A1, rgb_A2 from dye_rgb()                      │
│  • δ excitation fractions from exc_table (fpbase)             │
│  • Condition number of spectral mixing matrix M              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  Distance grid                                                │
│  r1  ∈ [0.4, 2.5] × R0_DA1   (20 points)                   │
│  r2  ∈ [0.4, 2.5] × R0_DA2   (20 points)                   │
│  r12 ∈ [0.3, 3.0] × R0_A1A2  (15 points)                   │
│  Optional: enforce triangle inequality |r1-r2| ≤ r12 ≤ r1+r2│
└──────────────────────────┬───────────────────────────────────┘
                           │
       ┌───────────────────┴────────────────────┐
       │                                        │
       ▼                                        ▼
┌─────────────────────┐               ┌─────────────────────────┐
│  488 nm photon draw  │               │  561 nm photon draw      │
│  N_D ~ Pois(μ_D^488) │               │  N_A1 ~ Pois(μ_A1^561)  │
│  N_A1~ Pois(μ_A1^488)│               │  N_A2 ~ Pois(μ_A2^561)  │
│  N_A2~ Pois(μ_A2^488)│               │  (N_D^561 negligible)   │
└──────────┬──────────┘               └───────────┬─────────────┘
           │                                      │
           ▼                                      ▼
┌─────────────────────┐               ┌─────────────────────────┐
│  488 nm Bayer image  │               │  561 nm Bayer image      │
│  via gen_camera_     │               │  via gen_camera_         │
│  image_stack()       │               │  image_stack()           │
└──────────┬──────────┘               └───────────┬─────────────┘
           │                                      │
           └──────────────┬───────────────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Aperture photometry          │
           │  → (N_B, N_G, N_R) per frame │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Linear spectral unmixing     │
           │  M @ [N_D,N_A1,N_A2] =       │
           │       [N_B,N_G,N_R]          │
           │  (per laser frame)            │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  MLE recovery                 │
           │  (r1, r2, r12) from all 6    │
           │  unmixed photon counts        │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Monte Carlo (N_sim per pt)   │
           │  → bias, RMSE, CI coverage   │
           │  → distance recovery maps    │
           └──────────────────────────────┘
```

---

## 3. Parameter Space

### 3.1 Triad selection

Use `practical_rows` from `3way_FRET_simulation_with488.ipynb`.

**Selection criteria for ALEX compatibility:**
- `laser_practical = True` at **488 nm** (standard criterion, already in table)
- A1 must absorb well at 561 nm: `exc_table[acc1][561] >= 0.40` (≥ 40% of A1 peak)
- A2 must absorb poorly at 561 nm: `exc_table[acc2][561] <= 0.15`
- D must absorb poorly at 561 nm: `exc_table[donor][561] <= 0.10` (ensures 561 preferentially excites A1)

Add a column `alex_practical` (bool) to the results table in the simulation notebook.

### 3.2 Distance grid

```python
rho1_vals  = np.linspace(0.2, 2.0, 20)   # r1  / R0_DA1
rho2_vals  = np.linspace(0.2, 2.0, 20)   # r2  / R0_DA2
rho12_vals = np.linspace(0.2, 2.0, 20)   # r12 / R0_A1A2
```

Total grid: 6000 configurations before triangle filter.  
With triangle filter (rigid geometry): typically ~2000–3000 configurations survive.

### 3.3 Photon budgets

```python
I0_D_vals  = [500, 1000, 2000, 5000]   # unquenched donor photons, 488 nm frame
I0_A1_vals = [500, 1000, 2000, 5000]   # unquenched A1 photons,   561 nm frame
```

In a paired experiment these are set by laser power; simulate them independently
(they need not be equal — 561 nm laser power is a free parameter).

The ratio `I0_A1 / I0_D` determines the relative precision of r12 vs (r1, r2).

### 3.4 N_frames sweep

For multi-frame averaging at a fixed position, multiply I0 by N_frames
(equivalent to summing independent Poisson draws):

```python
N_frames_vals = [1, 5, 20, 100]
```

### 3.5 Monte Carlo trials

`N_sim = 500` per (r1, r2, r12, I0_D, I0_A1) configuration.  
Run ~1000 for the "best triad" reference case.

---

## 4. Image Simulation Details

### 4.1 Camera setup

Reuse the flat-calibration approach from `fret_image_simulator.ipynb`:

```python
calib_dir = "../../Camera_Calibrations/Ximea_Camera"
gain_map     = IO.read_tiff(os.path.join(calib_dir, "gain.tif"))
offset_map   = IO.read_tiff(os.path.join(calib_dir, "offset.tif"))
variance_map = IO.read_tiff(os.path.join(calib_dir, "variance.tif"))
read_noise   = IO.read_tiff(os.path.join(calib_dir, "readnoise.tif"))

crop_sz = 14   # pixels
camera_params = {
    "gain":                np.full((crop_sz, crop_sz), np.median(gain_map)),
    "offset":              np.full((crop_sz, crop_sz), np.median(offset_map)),
    "variance":            np.full((crop_sz, crop_sz), np.median(variance_map)),
    "readnoise":           np.full((crop_sz, crop_sz), np.median(read_noise)),
    "rqe":                 np.full((crop_sz, crop_sz), 1.0),
    "pixel_QYs":           pixel_QYs,
    "pixel_order":         ["B", "G", "R"],
    "pixel_order_indices": [0, 1, 2],
    "masks":               M_F.get_masks(size_x=crop_sz, size_y=crop_sz),
}
```

### 4.2 Per-frame simulation procedure

For **each ALEX frame pair** (one trial):

```python
# --- Step A: Draw Poisson photon counts independently per dye ---
rng = np.random.default_rng()
N_D_draw   = rng.poisson(mu_D_488)
N_A1_draw  = rng.poisson(mu_A1_488)
N_A2_draw  = rng.poisson(mu_A2_488)

N_A1_561_draw = rng.poisson(mu_A1_561)
N_A2_561_draw = rng.poisson(mu_A2_561)

# --- Step B: Build composite [B,G,R] vector ---
rgb_488 = N_D_draw * rgb_D + N_A1_draw * rgb_A1 + N_A2_draw * rgb_A2
pe_488  = rgb_488 / rgb_488.sum()
N_488_total = rgb_488.sum()

rgb_561 = N_A1_561_draw * rgb_A1 + N_A2_561_draw * rgb_A2
pe_561  = rgb_561 / rgb_561.sum()
N_561_total = rgb_561.sum()

# --- Step C: Simulate Bayer image (adds read noise, camera model) ---
bayer_488, _, _ = MSF.gen_camera_image_stack(
    camera_params, wavelength, lambda_avg_488,
    pe_488[np.newaxis, :],           # (1, 3)
    {"mol": np.array([N_488_total])},
    x0y0_crop, smoothing_fn, background_photons=bg, NA=NA, pixel_size=px
)

bayer_561, _, _ = MSF.gen_camera_image_stack(
    camera_params, wavelength, lambda_avg_561,
    pe_561[np.newaxis, :],
    {"mol": np.array([N_561_total])},
    x0y0_crop, smoothing_fn, background_photons=bg, NA=NA, pixel_size=px
)
```

**Important:** Draw Poisson samples *before* calling `gen_camera_image_stack`
so each dye's shot noise is independent (physically correct). The composite
then gets additional read/dark noise from the camera model.

### 4.3 Average emission wavelength per frame

For the PSF sigma calculation, the emission-weighted mean wavelength changes
between laser frames:

```python
# 488 nm frame: weighted by N_D, N_A1, N_A2
lambda_avg_488 = (N_D_draw * lambda_D + N_A1_draw * lambda_A1 + N_A2_draw * lambda_A2) / N_488_total

# 561 nm frame: weighted by N_A1^561, N_A2^561
lambda_avg_561 = (N_A1_561_draw * lambda_A1 + N_A2_561_draw * lambda_A2) / N_561_total
```

where `lambda_D`, `lambda_A1`, `lambda_A2` are peak emission wavelengths (from fpbase).

### 4.4 Fast mode (photon counts only)

For the full Monte Carlo grid, skip the Bayer image simulation and work directly
with Poisson-drawn photon counts. This is ~1000× faster and valid when:
- Background is low (< 5 pe/px per frame)
- Aperture photometry has been validated against the image pipeline on a subset

Run the full image pipeline for validation on **100 configurations × 100 trials**
and confirm that the photon-count-only model gives equivalent RMSE. If it does,
use fast mode for the remaining 5900 configurations.

---

## 5. Analysis Pipeline

### 5.1 Aperture photometry (image pipeline)

Identical to `fret_image_simulator.ipynb` (cell `ks6gtb8jut`):

```python
sigma_px    = 0.21 * lambda_avg_nm / NA / pixel_size_nm
aperture_r  = 3.5 * sigma_px
bg_inner    = aperture_r + 0.5
bg_outer    = min(crop_sz/2 - 0.5, bg_inner + 2.0)

for ch in ("B", "G", "R"):
    ap_ch  = ap_mask & ch_masks[ch]
    bg_ch  = bg_mask & ch_masks[ch]
    bg_lev = pe_frame[bg_ch].mean()
    sig[ch] = pe_frame[ap_ch].sum() - bg_lev * ap_ch.sum()
```

Convert ADU → pe with `(ADU − offset) / gain` before summing.  
Apply per laser frame (488 and 561 separately).

### 5.2 Linear spectral unmixing

```python
M = np.column_stack([rgb_D, rgb_A1, rgb_A2])   # (3, 3)
cond_M = np.linalg.cond(M)                      # log this for every triad

# 488 nm frame
b_488 = np.array([sig_B_488, sig_G_488, sig_R_488])
N_unmix_488, _, _, _ = np.linalg.lstsq(M, b_488, rcond=None)
N_unmix_488 = np.clip(N_unmix_488, 0, None)   # [N_D_est, N_A1_est, N_A2_est]

# 561 nm frame: only A1 and A2 contribute; use 2×2 sub-problem
M_561 = np.column_stack([rgb_A1, rgb_A2])      # (3, 2)
b_561 = np.array([sig_B_561, sig_G_561, sig_R_561])
N_unmix_561, _, _, _ = np.linalg.lstsq(M_561, b_561, rcond=None)
N_unmix_561 = np.clip(N_unmix_561, 0, None)   # [N_A1_est, N_A2_est]
```

### 5.3 Joint MLE over (E1, E2, E12)

Given all six unmixed photon estimates, minimise the negative Poisson log-likelihood:

```python
def neg_log_likelihood(params, N_obs, model_params):
    E1, E2, E12 = np.clip(params, 1e-6, 1 - 1e-6)
    if E1 + E2 >= 1:
        return 1e10
    mu_D_488   = I0_D * (1 - E1 - E2)
    mu_A1_488  = (I0_D/QY_D) * E1 * QY_A1 * (1 - E12) + I0_D * delta_A1_488 * QY_A1
    mu_A2_488  = (I0_D/QY_D) * (E2 + E1*E12) * QY_A2 + I0_D * delta_A2_488 * QY_A2
    mu_A1_561  = I0_A1 * (1 - E12)
    mu_A2_561  = I0_A1 * E12 * (QY_A2/QY_A1)
    mus = [mu_D_488, mu_A1_488, mu_A2_488, mu_A1_561, mu_A2_561]
    N_obs_list = [N_D_488, N_A1_488, N_A2_488, N_A1_561, N_A2_561]
    return -sum(poisson_logpmf(n, mu) for n, mu in zip(N_obs_list, mus))

result = minimize(neg_log_likelihood, x0=[0.3, 0.3, 0.5],
                  method='Nelder-Mead', bounds=[(0,1),(0,1),(0,1)])
E1_est, E2_est, E12_est = result.x
```

Convert FRET efficiencies to distances:

```python
r12_est = R0_A1A2 * (1/E12_est - 1)**(1/6)

# Competitive FRET inversion for r1, r2:
# E1/E2 = (R0_DA1/r1)^6 / (R0_DA2/r2)^6
# E1+E2 = gamma  →  solve 1D equation for r1
ratio = E1_est / E2_est
# r2 = r1 * ratio^(-1/6) * (R0_DA2/R0_DA1)
# gamma = (R0_DA1/r1)^6 / [1 + (R0_DA1/r1)^6 + (R0_DA2/r2(r1))^6]  →  solve numerically
from scipy.optimize import brentq
def f_r1(r1):
    r2 = r1 * ratio**(-1/6) * (R0_DA2/R0_DA1)
    x1, x2 = (R0_DA1/r1)**6, (R0_DA2/r2)**6
    return x1/(1 + x1 + x2) - E1_est
r1_est = brentq(f_r1, 0.01*R0_DA1, 10*R0_DA1)
r2_est = r1_est * ratio**(-1/6) * (R0_DA2/R0_DA1)
```

### 5.4 Grid-search posterior (uncertainty quantification)

Rather than a point estimate only, compute the Poisson log-likelihood over the full
(r1, r2, r12) grid — this gives a posterior map and identifies multi-modal regions:

```python
# Vectorised over grid (uses broadcasting)
E1_grid  = E1_grid_arr   # shape (n1, n2, n12)
E2_grid  = E2_grid_arr
E12_grid = E12_grid_arr

ll_grid = (poisson_ll_vec(N_D_488,  I0_D * (1-E1_grid-E2_grid))
         + poisson_ll_vec(N_A1_488, (I0_D/QY_D)*E1_grid*QY_A1*(1-E12_grid) + ...)
         + poisson_ll_vec(N_A2_488, ...)
         + poisson_ll_vec(N_A1_561, I0_A1 * (1-E12_grid))
         + poisson_ll_vec(N_A2_561, I0_A1 * E12_grid * (QY_A2/QY_A1)))

# Marginals
ll_r1  = ll_grid.max(axis=(1,2))   # profile over r2, r12
ll_r2  = ll_grid.max(axis=(0,2))
ll_r12 = ll_grid.max(axis=(0,1))   # with ALEX this should be peaked
```

### 5.5 Monte Carlo aggregation

```python
results = defaultdict(list)
for trial in range(N_sim):
    # draw → image → photometry → unmix → MLE
    r1_est, r2_est, r12_est = run_one_trial(...)
    results["r1"].append(r1_est)
    results["r2"].append(r2_est)
    results["r12"].append(r12_est)

bias_r1  = np.mean(results["r1"])  - r1_true
rmse_r1  = np.sqrt(np.mean((np.array(results["r1"]) - r1_true)**2))
# ... similarly for r2, r12
```

---

## 6. Outputs and Visualisations

### 6.1 RMSE vs I0 for all three distances

Three panels (r1, r2, r12) each showing RMSE vs I0_D (or I0_A1) on a log–log scale.

- r1 and r2 RMSE governed by 488 nm frame photon budget (I0_D)
- r12 RMSE governed by 561 nm frame photon budget (I0_A1)
- Both should show ∝ 1/√I0 (shot-noise limited) in the high-N regime

### 6.2 RMSE maps in (r1/R0_DA1, r2/R0_DA2) space at fixed r12

Four heatmaps (one per I0_D level) showing how distance-recovery quality varies across
the FRET-sensitive range. Marks "good recovery" (RMSE < 0.1 × r_true) vs "poor recovery".

### 6.3 RMSE vs r12/R0_A1A2 at fixed (r1, r2)

Shows the E12-sensitive window (typically 0.5 < r12/R0_A1A2 < 2.0 where E12 ∈ [0.02, 0.98]).

### 6.4 Posterior maps for representative configurations

For three representative (r1, r2, r12) configurations:
- (short, short, short): all three FRET pairs active, high cascade
- (short, long, mid): D→A1 dominant, moderate cascade
- (long, short, long): D→A2 dominant, no cascade

Plot 2D slices of the log-likelihood surface (marginalised over the third distance)
to visualise posterior shape, multi-modality, and the benefit of ALEX.

### 6.5 Ternary plots (488 and 561 frame separately)

For the 488 nm frame: ternary plot of (A_R, A_G, A_B) over the (r1, r2) grid at fixed r12.  
For the 561 nm frame: ternary plot of (A_R, A_G, A_B) over r12 alone (r1, r2 irrelevant).  
Shows that the 561 nm frame cleanly resolves r12 regardless of the D–A geometry.

### 6.6 Bias under model mismatch

Test: simulate with cascade (E12 > 0) but recover assuming no cascade (E12 = 0).
Plot the resulting bias in r1, r2 as a function of E12 — quantifies the error
introduced by ignoring cascade transfer.

### 6.7 Multi-triad comparison table

For each `alex_practical` triad, report:
- Condition number of M (spectral unmixing quality)
- RMSE(r1), RMSE(r2), RMSE(r12) at I0 = 2000 ph, N_sim = 500
- Minimum I0 to achieve RMSE < 0.5 nm

---

## 7. Notebook Structure

**Notebook:** `notebooks/fret/3way_FRET_ImageSim_Cascade.ipynb`

```
Cell 1:   Imports (numpy, scipy, fpbase, matplotlib, mpltern)
          sys.path; load src modules
Cell 2:   Camera calibration (Ximea) + SpectralFunctions setup
          pixel_QYs, wavelength, masks
Cell 3:   Dye list + Förster radii (reuse forster_radius_nm from existing nb)
          dye_rgb() vectors; exc_table at 488 and 561 nm
          Add 'alex_practical' flag to results table
Cell 4:   Select triads: show practical + alex_practical rows
          Print R0_DA1, R0_DA2, R0_A1A2 for each; condition number of M
Cell 5:   Cascade photon budget functions
          triad_photons_cascade(r1,r2,r12,...) for 488 nm
          triad_photons_A1excitation(r12,...) for 561 nm
          Sanity check: no-cascade limit matches existing notebook
Cell 6:   Direct excitation fractions (δ_A1^488, δ_A2^488, δ_D^561, δ_A2^561)
          Plot: relative excitation of each dye at each laser line
Cell 7:   Single-configuration demo (best triad, r1=r2=r12=R0)
          Simulate 488 and 561 Bayer images; show side by side
          Aperture photometry → unmixed counts → recovered distances
Cell 8:   Two-step analytic recovery vs joint MLE: numerical comparison
          Verify both give same answer on noiseless data
Cell 9:   Validation: fast mode (Poisson only) vs full image pipeline
          100 configurations × 100 trials; compare RMSE
Cell 10:  Distance grid generation (np.meshgrid; optional triangle filter)
Cell 11:  Monte Carlo engine (main loop)
          vectorised Poisson draws; fast mode enabled
Cell 12:  RMSE vs I0 plots (three panels: r1, r2, r12)
Cell 13:  RMSE maps in (r1/R0, r2/R0) space at fixed r12, fixed I0
Cell 14:  RMSE vs r12/R0_A1A2 at fixed (r1, r2)
Cell 15:  Posterior map plots for 3 representative configurations
Cell 16:  Ternary plots (488 frame and 561 frame separately)
Cell 17:  Bias under no-cascade assumption vs E12 (model mismatch)
Cell 18:  Multi-triad comparison table
Cell 19:  Conclusions + summary
```

---

## 8. New Functions Needed

```python
def triad_photons_cascade(r1, r2, r12, R0_DA1, R0_DA2, R0_A1A2,
                           QY_D, QY_A1, QY_A2, I0_D,
                           delta_A1_488=0.0, delta_A2_488=0.0):
    """Mean photon counts for 488 nm excitation with A1→A2 cascade."""

def triad_photons_561(r12, R0_A1A2, QY_A1, QY_A2, I0_A1,
                      delta_A2_561=0.0, delta_D_561=0.0, E1=0.0, E2=0.0, I0_D=0.0):
    """Mean photon counts for 561 nm excitation (A1 as primary absorber)."""

def sample_poisson_frame(mu_D, mu_A1, mu_A2, rng):
    """Independent Poisson draws for each dye; returns (N_D, N_A1, N_A2)."""

def build_composite_pe(N_D, N_A1, N_A2, rgb_D, rgb_A1, rgb_A2):
    """Composite [B,G,R] vector and total photon count from per-dye draws."""

def unmix_rgb_3dye(sig_B, sig_G, sig_R, rgb_D, rgb_A1, rgb_A2):
    """Least-squares unmixing (3 dyes); returns (N_D_est, N_A1_est, N_A2_est)."""

def unmix_rgb_2dye(sig_B, sig_G, sig_R, rgb_A1, rgb_A2):
    """Least-squares unmixing (2 dyes for 561 nm frame)."""

def mle_distances(N_obs_488, N_obs_561, I0_D, I0_A1,
                  R0_DA1, R0_DA2, R0_A1A2,
                  QY_D, QY_A1, QY_A2, deltas):
    """Joint MLE over (E1,E2,E12); returns (r1_est, r2_est, r12_est)."""

def log_likelihood_grid(N_obs_488, N_obs_561, E1_grid, E2_grid, E12_grid,
                         I0_D, I0_A1, QY_D, QY_A1, QY_A2, deltas):
    """Vectorised Poisson log-likelihood over 3D (E1,E2,E12) grid."""

def fret_to_distances(E1, E2, E12, R0_DA1, R0_DA2, R0_A1A2):
    """Convert (E1,E2,E12) → (r1,r2,r12) via numerical inversion."""
```

---

## 9. Implementation Notes

### 9.1 Code reuse

| Component | Source |
|-----------|--------|
| `forster_radius_nm()` | `3way_FRET_simulation_with488.ipynb` — copy verbatim |
| `dye_rgb()` | same notebook |
| `triad_metrics()` | same notebook — extend with `alex_practical` flag |
| `gen_camera_image_stack()` | `src/Multicolour_Simulation_Functions.py` |
| Camera calibration loading | `fret_image_simulator.ipynb` |
| Aperture photometry | `fret_image_simulator.ipynb` cell `ks6gtb8jut` |
| Smoothing function setup | `fret_image_simulator.ipynb` cell `fret-sim-005-camera` |

### 9.2 Performance budget

| Task | Time estimate | N calls |
|------|--------------|---------|
| Full image sim (14×14, 1 molecule) | ~1 ms | 2 per trial × 100 trials × 100 pts (validation) = 20 000 |
| Poisson draw + MLE (fast mode) | ~0.1 ms | 2 per trial × 500 trials × 6000 pts = 6 × 10⁶ |
| Grid search posterior (6000 pts) | ~10 ms | 500 trials × 6000 pts |

Validation (full image): ~20 s. Fast Monte Carlo: ~5–10 min. Grid search posterior: ~1 h.
Parallelize Monte Carlo with `joblib.Parallel(n_jobs=-1)` if needed.

### 9.3 I0 calibration

Two scenarios:
1. **I0 known exactly**: use true I0_D, I0_A1 values in analysis.
2. **I0 uncertain (±20%)**: treat I0 as nuisance parameter, marginalise or profile out.

For scenario 2: multiply MLE grid by uniform prior over I0 ∈ [0.8, 1.2] × I0_nominal
and marginalise. This broadens the posterior but does not introduce bias.

### 9.4 Direct excitation correction accuracy

The δ correction factors come from fpbase spectra, which have ~5–10% uncertainty.
Run a sensitivity analysis: vary each δ by ±10% and measure bias in (r1, r2, r12).
This motivates the importance of accurate spectral calibration.

---

## 10. Expected Findings

1. **r12 is fully recoverable with ALEX** (unlike the single-laser case where it is
   algebraically degenerate). Precision scales ∝ 1/√I0_A1 and is best when
   r12 ≈ R0_A1A2 (E12 near 0.5).

2. **r1 and r2 precision** is limited by I0_D and the condition number of the spectral
   mixing matrix M. At high I0 and well-separated dye spectra, RMSE < 1 nm is achievable.

3. **Ignoring cascade** (setting E12=0 in analysis when true E12 > 0) biases r1 upward
   and r2 downward proportionally to E12. Quantifying this bias motivates including ALEX.

4. **The 561 nm frame is largely decoupled from r1 and r2** (D barely absorbs at 561 nm),
   so r12 recovery is approximately independent of the D–A geometry. This makes it a
   robust observable even when r1 and r2 are uncertain.

5. **Spectral unmixing is the binding constraint** at low I0: even with perfect FRET
   physics, collinear dye spectra make M ill-conditioned and inflate all distance errors.

---

## 11. Files

| File | Description |
|------|-------------|
| `claude/FRET_3way_ImageSimulation.md` | This document |
| `notebooks/fret/3way_FRET_ImageSim_Cascade.ipynb` | Simulation notebook (to be created) |
