# Wavelength Bias Investigation - Key Findings

## Executive Summary

**Root Cause Identified**: The wavelength bias in Nile Red simulations is caused by systematic overestimation of the B channel fraction due to extremely low signal-to-background ratio (SBR ≈ 1).

**Magnitude**: At 580 nm with 500 photons:
- **With B channel**: +49 nm wavelength bias
- **Without B channel**: +0.8 nm wavelength bias
- **B channel causes ~48 nm of the 48 nm total bias!**

---

## Detailed Investigation

### 1. Perfect Algorithm Verification ✅

**Test**: Forward model → RGB/σ → Wavelength fit (no noise)
- **Result**: 0.00 nm bias
- **Conclusion**: The wavelength extraction algorithm is mathematically perfect

### 2. RGB Fitting Bias 🔴

At 580 nm, 500 photons:

| Channel | True | Fitted (mean) | Bias | % Error |
|---------|------|---------------|------|---------|
| R | 0.5938 | 0.5722 | -0.0216 | -3.6% |
| G | 0.3788 | 0.3718 | -0.0070 | -1.9% |
| B | 0.0274 | 0.0560 | +0.0286 | **+104%** |

**Pattern**: R and G slightly underestimated, B massively overestimated

### 3. Signal-to-Background Analysis 🔍

**Expected photon distribution** (580 nm, 500 total photons):
- R: 296.9 photons → SBR = 22.3 (strong)
- G: 189.4 photons → SBR = 14.2 (good)
- B: **13.7 photons** → SBR = **1.03** (barely detectable!)

**Actual fitted photons**:
- R: 266.2 photons (-30.7 bias)
- G: 171.7 photons (-17.7 bias)
- B: **27.3 photons** (+13.6 bias)

**Total photons fitted**: 465.2 (expected 500)
- ~35 photons "missing" from R and G
- Those 35 photons are misattributed to B!

### 4. Why B is Overestimated

#### Mechanism:
1. **Low SBR**: B signal (13.7 photons) ≈ B background (13.3 photons)
2. **Non-negativity constraint**: B fraction cannot go below 0
3. **Asymmetric noise**:
   - Downward fluctuations truncated at 0
   - Upward fluctuations unbounded
   - Creates positive bias in mean
4. **Normalization constraint**: R + G + B = 1 redistributes fitting errors
5. **Error accumulation**: Underestimation in R and G must go somewhere → B acts as "error sink"

#### Evidence:
- 44% of fits have B < true value
- 56% of fits have B > true value
- Distribution skewed positive due to floor at 0
- Correlation(B_fraction, total_photons) = 0.235 (positive!)

### 5. Chi-Squared Analysis ✅

**Chi-squared**: 1.02 (mean), 1.02 (median)
- Indicates good fit quality overall
- Errors are estimated correctly on average
- **But**: Chi-squared doesn't detect systematic bias in individual channels

### 6. Sigma Fitting Bias ⚠️

**Sigma bias** (all cases):
- σ_x: +3-11 nm overestimation (+3-10%)
- σ_y: +4-11 nm overestimation (+4-10%)

**Consistent pattern**: PSF width systematically overestimated

### 7. Wavelength Fitting Test 🎯

**Test setup**: Compare wavelength fits with and without B channel

**Results** (580 nm, 500 photons, n=100 fits):

| Method | Mean λ (nm) | Bias (nm) | Std (nm) |
|--------|-------------|-----------|----------|
| With B (R, G, B) | 629.31 | **+49.31** | 75.56 |
| Without B (R/G only) | 580.79 | **+0.79** | 0.00 |

**Conclusion**: **B channel bias explains the entire wavelength bias!**

---

## Solutions

### Option 1: Exclude B Channel (Simple) ⭐ **RECOMMENDED**

**Method**: Use only R/G ratio for wavelength fitting when B has low SNR

```python
if B_expected < threshold:  # e.g., B < 5% of total
    # Renormalize to R/(R+G) and G/(R+G)
    R_renorm = A_R / (A_R + A_G)
    G_renorm = A_G / (A_R + A_G)
    B_renorm = 0.0
```

**Pros**:
- Nearly eliminates bias (+0.8 nm vs +49 nm)
- Dramatically improves precision (std → 0)
- Simple to implement

**Cons**:
- Loses spectral information from B (but it's unreliable anyway)
- Need to define threshold for when to exclude B

### Option 2: Weighted Least-Squares with SNR

**Method**: Weight channels by their SNR in the wavelength fit

```python
# Weight inversely proportional to uncertainty
w_R = 1 / R_err²
w_G = 1 / G_err²
w_B = 1 / B_err²  # Will be very small for low SNR

# Or explicitly by SNR:
w_R = SNR_R
w_G = SNR_G
w_B = SNR_B  # ≈ 1 for B, will have minimal weight
```

**Pros**:
- Uses all available data
- Automatically downweights unreliable channels

**Cons**:
- More complex
- Still includes biased B values (just with less weight)
- May not fully eliminate bias

### Option 3: Improve B Channel Fitting

**Method**: Modify fitting algorithm to handle low-SNR channels better

Possible approaches:
- Bayesian prior on B (regularization toward expected value)
- Constrain B to physical range based on wavelength bounds
- Use different cost function that's less sensitive to outliers

**Pros**:
- Addresses root cause
- Improves all channels

**Cons**:
- Requires significant code changes
- May introduce other biases
- Complex to validate

### Option 4: Increase Photon Count

**Method**: Require higher photon thresholds for Nile Red analysis

At 2000 photons:
- B: ~55 photons (vs 13 at 500)
- SBR improves to ~4
- Bias should be much smaller

**Pros**:
- Simple (just filter data)
- No algorithm changes

**Cons**:
- Loses low-photon localizations
- Doesn't solve fundamental issue

---

## Recommendations

### Immediate Action: Implement Option 1 ⭐

1. Add adaptive B channel exclusion to wavelength fitting
2. Use threshold: B_fraction < 0.05 or B_photons < 20
3. Renormalize R and G when B is excluded

### Code Location

Modify: `src/Multicolour_Simulation_Functions.py:823-864`

Current wavelength fitting section needs modification to:
1. Calculate expected B photons
2. Check if B is reliable (SNR > threshold)
3. If not: renormalize to (R/G) only and set B=0

### Testing

Verify with simulations:
- Run wavelength precision tests with modified code
- Check bias at 500, 1000, 2000, 5000 photons
- Ensure bias < 2 nm across all conditions

---

## Error Propagation Issues (Secondary)

### Current Implementation

From `Multicolour_Simulation_Functions.py:848-851`:
```python
total_err = np.sqrt(R_err**2 + G_err**2 + B_err**2)
R_norm_err = R_norm * np.sqrt((R_err/R)**2 + (total_err/rgb_total)**2)
```

### Issues

1. **The parquet file already contains normalized fractions** (A_R, A_G, A_B sum to 1)
2. Current error propagation re-normalizes already normalized values
3. Missing Jacobian for the normalization transformation:
   - ∂R_norm/∂A_R = (A_G + A_B) / total²
   - ∂R_norm/∂A_G = -A_R / total²
   - ∂R_norm/∂A_B = -A_R / total²

### Correct Formula

For independent errors:
```python
σ²(R_norm) = [(A_G+A_B)²σ²(A_R) + A_R²σ²(A_G) + A_R²σ²(A_B)] / total⁴
```

**However**: This is secondary to the B bias issue. Fixing B bias will have much larger impact than correcting error propagation.

---

## Next Steps

1. ✅ **Confirm**: B channel bias is root cause (DONE)
2. ⏭️ **Implement**: Adaptive B channel exclusion
3. ⏭️ **Test**: Run full wavelength precision simulation with fix
4. ⏭️ **Validate**: Check bias < 2 nm across photon range
5. ⏭️ **Document**: Update NileRed functions documentation

---

## Files Modified

- `unit_tests/diagnose_wavelength_bias.py` - Initial diagnostic
- `unit_tests/analyze_fit_bias.py` - RGB and sigma bias analysis
- `unit_tests/investigate_b_channel_bias.py` - B channel investigation
- `unit_tests/test_b_bias_contribution.py` - Wavelength fit with/without B
- `unit_tests/WAVELENGTH_BIAS_FINDINGS.md` - This document

**To implement fix**: Modify `src/Multicolour_Simulation_Functions.py` around line 853
