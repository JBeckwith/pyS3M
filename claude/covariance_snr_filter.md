# Implementation Plan: Covariance-Based Amplitude SNR Filter

**Created:** 2026-03-10
**Implemented:** 2026-03-10 (commit 6e8e49f)
**Status:** IMPLEMENTED
**Threshold adopted:** 2.0 sigma (validated on simulated data; FPR=0.4%, TPR 97% at 2000 pe)

---

## Problem

The current fitting pipeline rejects spots via three hard, data-scale-dependent gates:

| Gate | Location | Failure mode |
|---|---|---|
| `MEDIAN_GATE_THRESHOLD = 2.0 pe` (Stage 1) | `fit_standard_punctum` | Rejects real spots with skewed/negative background (e.g. summed frames) |
| `MIN_PHOTON_THRESHOLD = 50.0 pe` (Stage 2) | `process_fit_results` | Rejects real spots when signal is legitimately low |
| `MAX_CHI_SQUARED = 3.0` (Stage 2) | `process_fit_results` | Rejects all spots from summed frames (high SNR inflates χ²) |

The chi_sqr gate is already patched with `skip_chisqr=True` for summed-frame contexts, but
this is a workaround. All three gates should be replaced by a single principled criterion.

---

## Solution: Amplitude SNR (Wald t-statistic)

After `leastsq` converges, the covariance matrix `pcov` gives the uncertainty on every
fitted parameter. The parameters are optimised in **sqrt-space** to enforce positivity
(i.e. the optimiser finds `q_B = sqrt(A_B)`, `q_G = sqrt(A_G)`, `q_R = sqrt(A_R)`).

For each channel the Wald t-statistic is:

```
t_c = |q_c| / SE(q_c)  =  |pfit[7+c]| / sqrt(pcov[7+c, 7+c])
```

The combined amplitude SNR across all three channels (summing in sqrt-space is fine as
a detection statistic):

```
z_amplitude = (|q_B| + |q_G| + |q_R|) / sqrt(var(q_B) + var(q_G) + var(q_R))
            = sum(|pfit[7:10]|) / sqrt(sum(diag(pcov)[7:10]))
```

**Accept the fit if `z_amplitude >= SNR_THRESHOLD` (suggested default: 3.0).**

### Why this is better

- **Scale-invariant**: the same threshold works for 1 pe or 100,000 pe, single frames or
  summed frames, any camera.
- **Self-calibrating**: if the model fits badly (high χ²), `pcov` is scaled up by χ²
  (via `process_covariance`), making `z_amplitude` smaller — automatically more
  conservative without a separate chi_sqr gate.
- **Physically meaningful**: `z ≥ 3` means "the amplitude is detected at ≥ 3σ
  significance given the noise in the fit".
- **No Stage 1 pre-filter needed**: `MEDIAN_GATE_THRESHOLD` can be removed entirely
  (it was a cheap proxy for this exact quantity, but fails for skewed backgrounds).

### Behaviour when pcov is unavailable

`process_covariance` returns `np.inf` when there are too few degrees of freedom or
`leastsq` returns `pcov = None`. In that case `z_amplitude` cannot be computed; the
fit should be **rejected** (treat as SNR = 0).

---

## Implementation Steps

### Step 1: Update `FittingResultProcessor.process_fit_results`
**File:** `src/ImageAnalysisFunctions.py`

Replace the Stage 2 block (lines ~329–335):

```python
# BEFORE
if total_pe < FittingConstants.MIN_PHOTON_THRESHOLD or chisqr_fail:
    return NaN, NaN

# AFTER
amplitude_snr = _compute_amplitude_snr(pfit, pcov)
if amplitude_snr < FittingConstants.AMPLITUDE_SNR_THRESHOLD:
    return NaN, NaN
```

Add a helper (static or module-level):

```python
def _compute_amplitude_snr(pfit, pcov):
    """Wald SNR for the three amplitude parameters (positions 7-9, sqrt-space)."""
    if pcov is np.inf or not isinstance(pcov, np.ndarray):
        return 0.0
    variances = np.diag(pcov)[7:10]
    if np.any(variances <= 0):
        return 0.0
    return float(np.sum(np.abs(pfit[7:10])) / np.sqrt(np.sum(variances)))
```

Add to `FittingConstants`:

```python
AMPLITUDE_SNR_THRESHOLD = 3.0   # sigma; replaces MIN_PHOTON_THRESHOLD + MAX_CHI_SQUARED
```

### Step 2: Remove `MEDIAN_GATE_THRESHOLD` Stage 1 pre-filter
**File:** `src/ImageAnalysisFunctions.py`, `StandardFittingProcessor.fit_single_punctum`

Remove the `compute_A_median` check entirely:

```python
# Remove these lines:
A_median = gaussoptfuncs.compute_A_median(smoothed_punctum)
if A_median < FittingConstants.MEDIAN_GATE_THRESHOLD:
    ...
```

A fast pre-filter is still valuable for performance (avoids running leastsq on clearly
empty ROIs). Replace with a simpler, more robust check:

```python
# Keep only a positivity check on the peak pixel:
if np.max(smoothed_punctum) <= 0:
    return NaN, NaN
```

This rejects ROIs where the smoothed data is entirely non-positive (truly empty), but
nothing else.

### Step 3: Clean up `skip_chisqr` workaround
**File:** `src/ImageAnalysisFunctions.py`, `src/SR_Functions.py`

Once Step 1 is in place, `skip_chisqr` is no longer needed. Remove:
- `skip_chisqr` parameter from `process_fit_results`, `_perform_wls_fit`,
  `fit_single_punctum`, `fit_puncta_method`, `_fit_puncta_method_standalone`,
  `fit_puncta_parallel_method`
- `skip_chisqr=(actual_frames_summed > 1)` calls in `example_spots_singleframe`
  and `fit_FRET_data`
- `FittingConstants.MAX_CHI_SQUARED` (or keep it as a stored metric only, not a gate)

### Step 4: Keep χ² as a stored quality metric (not a gate)
The `chi_sqr` column in results is genuinely useful for *post-hoc* quality assessment
(e.g. flagging outliers in downstream analysis). Keep it in the output DataFrame; just
don't use it as a hard rejection criterion.

### Step 5: Validation
Run `example_spots_singleframe` on:
- Single-frame data (n_frames_sum=1): confirm behaviour unchanged vs current pipeline
- Summed data (n_frames_sum=25 or 50): confirm spots now survive without `skip_chisqr`
- Very dim data (near detection threshold): confirm SNR gate rejects noise correctly

Compare `z_amplitude` distribution for detected vs non-detected spots to validate the
`AMPLITUDE_SNR_THRESHOLD = 3.0` default.

---

## Files to Modify

| File | Changes |
|---|---|
| `src/ImageAnalysisFunctions.py` | Add `_compute_amplitude_snr`, add `AMPLITUDE_SNR_THRESHOLD` to `FittingConstants`, replace Stage 2 gate, remove Stage 1 A_median gate, remove `skip_chisqr` parameter chain |
| `src/SR_Functions.py` | Remove `skip_chisqr` calls in `example_spots_singleframe` and `fit_FRET_data` |

---

## Simulation-Based Validation Test

**Notebook:** `notebooks/testing_notebooks/test_covariance_snr.ipynb`

Mirrors the structure of `testing_initial_guess_fit.ipynb` (the median-filter validation).
Ground truth is known: we feed the fitter ROIs where a spot is present or absent.

### Design

| ROI type | How generated | Expected outcome |
|---|---|---|
| **Signal** | `gen_camera_image_stack` with `n_photons > 0`, fit at known spot centre | z_amplitude ≥ 3 |
| **Noise** | Same, but `n_photons = 0`; fit at same location | z_amplitude ≈ 0–2 |

The fitter is called directly (bypassing all detection and gate logic) so that z_amplitude,
total_pe, and chi_sqr are all recorded for EVERY ROI — including those that would have been
rejected by the current gates.

### Photon levels

```
photon_levels = [0, 50, 100, 200, 500, 2000, 10000]
```

- **0 pe**: pure noise reference
- **50–500 pe**: typical single-molecule SMLM regime — current MIN_PHOTON_THRESHOLD=50 pe
  gate is most aggressive here
- **2000–10000 pe**: bright (QDot-like) or summed-frame equivalent — regime where
  chi_sqr inflation is known to cause false rejections in production data

### Key plots (one call to `fit_and_compute_z` per ROI, no gates applied)

1. **z_amplitude distributions** — noise vs signal at each photon level.
   Expect bimodal with signal >> 3 and noise ≈ 0, REGARDLESS of photon level.

2. **z_amplitude vs chi_sqr** (scatter, colour = photon level).
   At high SNR, chi_sqr drifts up (pixel-discretisation model mismatch) while z stays
   stable — this is the key failure mode of the current chi_sqr gate.

3. **z_amplitude vs total_pe** (scatter).
   z_amplitude should be uncorrelated with total_pe once above threshold.
   Compare: min_photons gate (vertical line at 50 pe) rejects real spots at low pe,
   and fails entirely at high pe where all noise also exceeds 50 pe.

4. **ROC curves** — TPR vs FPR for three classifiers:
   - z_amplitude ≥ threshold (sweep threshold)
   - total_pe ≥ threshold (current MIN_PHOTON_THRESHOLD gate)
   - chi_sqr ≤ threshold (current MAX_CHI_SQUARED gate)

5. **Summary table** — at z = 3.0: FPR, TPR, false-negatives at each photon level.

### z_amplitude computation inside fit function

```python
pfit, pcov_raw, infodict, errmsg, success = leastsq(
    WLS_chi_nobounds, x0=ig, args=(...), full_output=True,
    ftol=1e-2, xtol=1e-2,
)
residuals = infodict["fvec"]
chisqr = np.dot(residuals, residuals) / max(ravelsize - n_params, 1)
# Scale covariance as process_covariance does:
pcov_scaled = pcov_raw * chisqr   # (None → np.inf → z = 0)
variances = np.diag(pcov_scaled)[7:10]
z_amplitude = sum(|pfit[7:10]|) / sqrt(sum(variances))   # Wald t-statistic
```

`pcov_scaled = pcov_raw * chisqr` is the same scaling used by `process_covariance` in
production, so a high chi_sqr *inflates* the covariance → *reduces* z_amplitude, making
the statistic automatically conservative for poor fits (unlike the current hard gate).

---

## Notes

- `pcov` passed to `process_fit_results` is already scaled by χ² via
  `process_covariance` (line ~483), so the `z_amplitude` statistic automatically
  accounts for goodness-of-fit.
- The `MIN_PHOTON_THRESHOLD = 50 pe` constant can be kept in `FittingConstants` for
  reference / backwards compatibility but removed from the gate logic.
- `AMPLITUDE_SNR_THRESHOLD = 3.0` is a starting point. After validation, this may
  want to be 2.5 (more permissive) or 4.0 (more stringent) depending on false positive
  rate in practice.
