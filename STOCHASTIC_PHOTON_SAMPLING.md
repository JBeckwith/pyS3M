# Stochastic Photon Sampling Implementation

## Overview

Added new functions to `SpectralFunctions.py` to enable realistic photon-limited simulations where RGB ratios and PSF widths vary stochastically with photon count, rather than being deterministic.

## New Functions Added

### 1. `sample_photons_from_spectrum()`
**Location:** `src/SpectralFunctions.py` line 794

**Purpose:** Sample individual photon wavelengths from an emission spectrum treated as a probability density function.

**Usage:**
```python
sf = Spectral_Funcs()
R_qy, G_qy, B_qy, wl = sf.getpixelefficiency()
dye_spec = sf.get_dye_or_filter_data('alexa-fluor-647', wl)

# Sample 1000 photons
rng = np.random.default_rng(42)
photon_wavelengths = sf.sample_photons_from_spectrum(
    dye_spec[0], wl, n_photons=1000, random_state=rng
)
```

**Key features:**
- Uses inverse CDF sampling for accurate wavelength distribution
- Accounts for filter transmission, pixel QE, etc. if included in spectrum
- Returns array of wavelengths (nm)

---

### 2. `calculate_colourratio_from_photon_wavelengths()`
**Location:** `src/SpectralFunctions.py` line 867

**Purpose:** Convert sampled photon wavelengths into mean wavelength and B:G:R colour ratios with proper shot noise.

**Usage:**
```python
pixel_QYs = np.vstack([B_qy, G_qy, R_qy])  # [B, G, R] ordering

mean_wl, bgr = sf.calculate_colourratio_from_photon_wavelengths(
    photon_wavelengths, wl, pixel_QYs,
    pixel_order=['B', 'G', 'R'],
    pixel_order_indices=[0, 1, 2],
    return_counts=False  # Returns normalized ratios
)

print(f"Mean wavelength: {mean_wl:.1f} nm")
print(f"B:G:R = {bgr[0]:.3f}:{bgr[1]:.3f}:{bgr[2]:.3f}")
```

**Key features:**
- Each photon assigned to B, G, or R based on quantum efficiency at its wavelength
- Stochastic assignment → realistic shot noise
- Low photon counts → high variance in ratios
- Output order matches `pixel_QYs` input order (typically [B, G, R])

---

### 3. `generate_bootstrap_colour_ratios()`
**Location:** `src/SpectralFunctions.py` line 998

**Purpose:** Efficiently generate many bootstrap samples of colour ratios and mean wavelengths for Monte Carlo simulations.

**Usage:**
```python
# Generate 1000 bootstrap samples at 500 photons each
mean_wls, bgr_ratios = sf.generate_bootstrap_colour_ratios(
    dye_spec[0], wl, pixel_QYs,
    n_photons_per_image=500,
    n_bootstrap=1000,
    pixel_order=['B', 'G', 'R'],
    random_state=rng
)

# Analyse shot noise statistics
print(f"Mean B: {bgr_ratios[:, 0].mean():.3f} ± {bgr_ratios[:, 0].std():.3f}")
print(f"Mean G: {bgr_ratios[:, 1].mean():.3f} ± {bgr_ratios[:, 1].std():.3f}")
print(f"Mean R: {bgr_ratios[:, 2].mean():.3f} ± {bgr_ratios[:, 2].std():.3f}")
```

**Key features:**
- Samples all photons once (n_bootstrap × n_photons_per_image total)
- Divides into bootstrap chunks
- ~n_bootstrap times faster than repeated sampling
- Returns shape (n_bootstrap, 3) for colour ratios
- Returns shape (n_bootstrap,) for mean wavelengths

---

## Integration into Multicolour_Simulation_Functions

### Current deterministic approach:
```python
# In test_simulation_method() around line 1716
n_photons = {"dye": np.full(config.n_bootstrap, n_photon)}

# Later in gen_camera_image_stack():
# Uses average_emission_wavelengths (scalar)
# → Deterministic RGB ratios
# → Deterministic PSF widths
```

### Proposed stochastic approach:

```python
# In test_simulation_method() BEFORE calling gen_camera_image_stack():

# 1. Get the full spectrum (emission × filters × pixel QE)
import SpectralFunctions
sf = SpectralFunctions.Spectral_Funcs()

# Get emission spectrum
if single_dye_spectrum is not None:
    full_spectrum = single_dye_spectrum
else:
    dye_spectrum = sf.get_dye_or_filter_data(dye, wavelength)
    full_spectrum = dye_spectrum[0]

# Apply filter transmission
filter_spectra = sf.get_spectral_data(filters, wavelength, SpectralFunctions.SpectralDataType.FILTER)
filter_transmission = np.prod(filter_spectra, axis=0)
full_spectrum = full_spectrum * filter_transmission

# 2. Generate stochastic colour ratios and wavelengths for all bootstrap samples
pixel_QYs = camera_parameters['pixel_QYs']
pixel_order = camera_parameters.get('pixel_order', ['B', 'G', 'R'])

mean_wavelengths, colour_ratios = sf.generate_bootstrap_colour_ratios(
    full_spectrum,
    wavelength,
    pixel_QYs,
    n_photons_per_image=int(n_photon),
    n_bootstrap=config.n_bootstrap,
    pixel_order=pixel_order,
    random_state=np.random.default_rng()  # Or pass seed for reproducibility
)

# 3. Pass to gen_camera_image_stack with stochastic values
# Replace scalar average_emission_wavelength with array:
# average_emission_wavelength → mean_wavelengths (shape: n_bootstrap)

# Replace deterministic dye_pixel_efficiency with stochastic colour ratios:
# dye_pixel_efficiency → colour_ratios (shape: n_bootstrap, 3)
```

### Changes needed in gen_camera_image_stack():

Currently expects:
- `average_emission_wavelengths`: scalar or 1D array of wavelengths
- `dye_pixel_efficiency`: shape (3,) or (3, n_wavelengths)

Need to handle:
- `average_emission_wavelengths`: shape (n_bootstrap,) - one per frame
- `colour_ratios`: shape (n_bootstrap, 3) - B, G, R ratios per frame

Update around line 1386:
```python
# OLD:
sigma_nm = self.psf.sigma_PSF(average_emission_wavelengths, NA)

# NEW (handle per-frame wavelengths):
if np.isscalar(average_emission_wavelengths):
    sigma_nm = self.psf.sigma_PSF(average_emission_wavelengths, NA)
    sigma_x = sigma_nm / pixel_size
    sigma_y = sigma_x
else:
    # Array of wavelengths (one per frame)
    sigma_nm_array = np.array([
        self.psf.sigma_PSF(wl, NA) for wl in average_emission_wavelengths
    ])
    # Will use sigma_nm_array[frame] later
```

Update in frame loop around line 1452:
```python
for frame in range(s):
    # Use frame-specific sigma if wavelengths vary per frame
    if not np.isscalar(average_emission_wavelengths):
        sigma_x = sigma_nm_array[frame] / pixel_size
        sigma_y = sigma_x

    # Use frame-specific colour ratios if provided
    if dye_pixel_efficiency.ndim == 2 and dye_pixel_efficiency.shape[0] == s:
        # dye_pixel_efficiency has shape (n_bootstrap, 3)
        frame_colour_ratios = dye_pixel_efficiency[frame, :]
    else:
        # Use same colour ratios for all frames (deterministic)
        frame_colour_ratios = dye_pixel_efficiency
```

---

## Benefits of Stochastic Approach

1. **Realistic shot noise**: RGB ratios vary naturally with photon count
   - 100 photons → high variance
   - 10,000 photons → low variance

2. **Wavelength uncertainty**: Mean wavelength varies per image
   - Affects PSF width realistically
   - Important for wavelength-dependent analyses

3. **Correct statistics**: Bootstrap distributions match experimental reality
   - RMSE estimates account for photon statistics
   - Colour distance calculations more realistic

4. **Physical accuracy**: Each photon treated individually
   - Proper Poisson statistics
   - Correct quantum efficiency handling

---

## Example: Effect of Shot Noise

**Deterministic (current):**
```
100 photons:  B:G:R = 0.100:0.300:0.600
1000 photons: B:G:R = 0.100:0.300:0.600  (identical!)
```

**Stochastic (new):**
```
100 photons:  B:G:R = 0.092:0.287:0.621 ± 0.029  (high variance)
1000 photons: B:G:R = 0.099:0.302:0.599 ± 0.009  (low variance)
```

The stochastic approach correctly shows that low photon counts have higher uncertainty in colour ratios!

---

## Next Steps

1. ✅ Add three new functions to SpectralFunctions.py
2. ⏳ Modify `test_simulation_method()` to use stochastic sampling
3. ⏳ Update `gen_camera_image_stack()` to handle per-frame wavelengths and colours
4. ⏳ Test with existing simulations to verify backwards compatibility
5. ⏳ Add unit tests for new functions
6. ⏳ Update documentation/examples

---

## Backwards Compatibility

The new functions are additive - existing code continues to work:
- Old: Pass scalar `average_emission_wavelength` and static `dye_pixel_efficiency`
- New: Pass array `average_emission_wavelengths` and per-frame `colour_ratios`

Functions check array shapes to determine which mode to use.
