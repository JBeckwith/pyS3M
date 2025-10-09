"""
Nile Red Spectral Model Functions

Forward and inverse models for predicting Nile Red emission properties (RGB intensities
and PSF widths) based on spectral parameters. Uses skew-Gaussian emission model fitted
from experimental data to extract central emission wavelength from localization data.

Leverages SpectralFunctions for wavelength/energy conversions and spectral models,
and PSFFunctions for wavelength-dependent PSF calculations.

Author: Claude Code (Anthropic)
Date: October 7, 2025
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from typing import Dict, Tuple, Optional
import SpectralFunctions
import PSFFunctions
import duckdb
import hashlib
import os


class NileRed_Functions:
    """
    Nile Red spectral analysis and wavelength extraction.

    This class implements forward and inverse models for Nile Red emission:
    - Forward: central_wavelength → (R, G, B, σ_x, σ_y)
    - Inverse: (R, G, B, σ_x, σ_y) → central_wavelength

    Default spectral parameters from notebook 20251007_NileRedOptimiser:
    - sigma = 0.1630104 eV (Gaussian width in energy space)
    - alpha = -1.56453968 (skewness parameter)
    - wavelength_center = 617.6 nm (initial guess)

    Dependencies:
    - SpectralFunctions: For wavelength/energy conversions, skew-Gaussian models,
                        filter/dye spectra, and pixel quantum efficiencies
    - PSFFunctions: For wavelength-dependent PSF width calculations
    """

    def __init__(
        self,
        sigma_energy: float = 0.1630104,
        alpha: float = -1.56453968,
        wavelength_center_init: float = 617.6
    ):
        """Initialize Nile Red model with spectral parameters.

        Args:
            sigma_energy: Gaussian width in energy space (eV), default from fit
            alpha: Skewness parameter, default from fit
            wavelength_center_init: Initial guess for central wavelength (nm)
        """
        self.default_sigma_energy = sigma_energy
        self.default_alpha = alpha
        self.default_wavelength_center = wavelength_center_init

        # Initialize SpectralFunctions and PSFFunctions for shared functionality
        self.spectral_funcs = SpectralFunctions.Spectral_Funcs()
        self.psf_funcs = PSFFunctions.PSF_Functions()

        # LUT cache
        self._lut_cache = {}
        self._lut_interpolator_cache = {}  # Cache pre-built interpolators
        self._db_path = os.path.join(os.path.dirname(__file__), '..', 'Spectra', 'spectral_data.duckdb')

    def _get_config_hash(self, filter_names: list, NA: float) -> str:
        """Generate unique hash for filter configuration.

        Args:
            filter_names: List of filter names
            NA: Numerical aperture

        Returns:
            MD5 hash string identifying this configuration
        """
        config_str = f"{sorted(filter_names)}_{NA}_{self.default_sigma_energy}_{self.default_alpha}"
        return hashlib.md5(config_str.encode()).hexdigest()

    def _create_lut_table_if_not_exists(self, conn):
        """Create Nile Red LUT table in DuckDB if it doesn't exist."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nile_red_lut (
                config_hash VARCHAR PRIMARY KEY,
                filter_names VARCHAR,
                NA FLOAT,
                sigma_energy FLOAT,
                alpha FLOAT,
                wavelength_min FLOAT,
                wavelength_max FLOAT,
                wavelength_step FLOAT,
                n_points INTEGER,
                wavelengths FLOAT[],
                rgb_r FLOAT[],
                rgb_g FLOAT[],
                rgb_b FLOAT[],
                sigma_psf FLOAT[],
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def generate_lut(
        self,
        filter_names: list,
        NA: float = 1.49,
        wavelength_range: Tuple[float, float] = (550.0, 750.0),
        wavelength_step: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate lookup table for Nile Red forward model.

        Pre-computes RGB and σ_PSF values across wavelength range for fast interpolation.

        Args:
            filter_names: List of filter names for optical system
            NA: Numerical aperture
            wavelength_range: (min, max) wavelength range in nm
            wavelength_step: Wavelength resolution in nm (default: 0.5 nm)

        Returns:
            Tuple of (wavelengths, rgb_array, sigma_psf_array)
            - wavelengths: 1D array of wavelengths (nm)
            - rgb_array: (N, 3) array of [R, G, B] values
            - sigma_psf_array: 1D array of PSF widths (nm)
        """
        print(f"Generating Nile Red LUT for {len(filter_names)} filters, NA={NA}")
        print(f"  Wavelength range: {wavelength_range[0]}-{wavelength_range[1]} nm")
        print(f"  Resolution: {wavelength_step} nm")

        # Setup optical system
        wavelength_array, pixel_QYs, filter_spectra = self.setup_optical_system(filter_names)

        # Generate wavelength grid for LUT
        lut_wavelengths = np.arange(wavelength_range[0], wavelength_range[1] + wavelength_step, wavelength_step)
        n_points = len(lut_wavelengths)

        print(f"  Computing {n_points} forward model evaluations...")

        # Pre-allocate arrays
        lut_rgb = np.zeros((n_points, 3))
        lut_sigma_psf = np.zeros(n_points)

        # Compute forward model for each wavelength
        for i, wl in enumerate(lut_wavelengths):
            if i % 50 == 0:
                print(f"    Progress: {i}/{n_points} ({100*i/n_points:.1f}%)", end='\r', flush=True)

            predictions = self.nile_red_forward_model(
                wl, filter_spectra, wavelength_array, pixel_QYs, NA
            )
            lut_rgb[i] = [predictions['R'], predictions['G'], predictions['B']]
            lut_sigma_psf[i] = predictions['sigma_x']

        print(f"    Progress: {n_points}/{n_points} (100.0%) - Done!       ")

        return lut_wavelengths, lut_rgb, lut_sigma_psf

    def save_lut_to_database(
        self,
        filter_names: list,
        NA: float,
        wavelengths: np.ndarray,
        rgb_array: np.ndarray,
        sigma_psf_array: np.ndarray
    ):
        """Save LUT to DuckDB database.

        Args:
            filter_names: List of filter names
            NA: Numerical aperture
            wavelengths: 1D array of wavelengths
            rgb_array: (N, 3) array of RGB values
            sigma_psf_array: 1D array of PSF widths
        """
        config_hash = self._get_config_hash(filter_names, NA)
        conn = None

        try:
            conn = duckdb.connect(self._db_path)
            self._create_lut_table_if_not_exists(conn)

            # Check if entry exists
            existing = conn.execute(
                "SELECT config_hash FROM nile_red_lut WHERE config_hash = ?",
                [config_hash]
            ).fetchone()

            if existing:
                print(f"LUT already exists for this configuration (hash: {config_hash[:8]}...)")
                print("Updating existing entry...")
                conn.execute("DELETE FROM nile_red_lut WHERE config_hash = ?", [config_hash])

            # Insert new LUT
            conn.execute("""
                INSERT INTO nile_red_lut (
                    config_hash, filter_names, NA, sigma_energy, alpha,
                    wavelength_min, wavelength_max, wavelength_step, n_points,
                    wavelengths, rgb_r, rgb_g, rgb_b, sigma_psf
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                config_hash,
                ','.join(filter_names),
                NA,
                self.default_sigma_energy,
                self.default_alpha,
                float(wavelengths[0]),
                float(wavelengths[-1]),
                float(wavelengths[1] - wavelengths[0]),
                len(wavelengths),
                wavelengths.tolist(),
                rgb_array[:, 0].tolist(),
                rgb_array[:, 1].tolist(),
                rgb_array[:, 2].tolist(),
                sigma_psf_array.tolist()
            ])

            conn.commit()

            print(f"LUT saved to database (hash: {config_hash[:8]}...)")
            print(f"  Database: {self._db_path}")
        except Exception as e:
            print(f"Warning: Could not save LUT to database: {e}")
            print("LUT will remain in memory cache only")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except:
                    pass

    def load_lut_from_database(
        self,
        filter_names: list,
        NA: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load LUT from DuckDB database.

        Args:
            filter_names: List of filter names
            NA: Numerical aperture

        Returns:
            Tuple of (wavelengths, rgb_array, sigma_psf_array) or None if not found
        """
        config_hash = self._get_config_hash(filter_names, NA)

        # Check memory cache first
        if config_hash in self._lut_cache:
            return self._lut_cache[config_hash]

        if not os.path.exists(self._db_path):
            return None

        try:
            conn = duckdb.connect(self._db_path, read_only=True)

            try:
                result = conn.execute("""
                    SELECT wavelengths, rgb_r, rgb_g, rgb_b, sigma_psf
                    FROM nile_red_lut
                    WHERE config_hash = ?
                """, [config_hash]).fetchone()

                if result is None:
                    return None

                wavelengths = np.array(result[0])
                rgb_array = np.column_stack([
                    np.array(result[1]),  # R
                    np.array(result[2]),  # G
                    np.array(result[3])   # B
                ])
                sigma_psf_array = np.array(result[4])

                # Cache in memory for future use
                self._lut_cache[config_hash] = (wavelengths, rgb_array, sigma_psf_array)

                return wavelengths, rgb_array, sigma_psf_array

            finally:
                conn.close()
        except Exception:
            # Database doesn't exist, table doesn't exist, or is locked
            return None

    def get_or_create_lut(
        self,
        filter_names: list,
        NA: float = 1.49,
        wavelength_range: Tuple[float, float] = (550.0, 750.0),
        wavelength_step: float = 0.5,
        force_regenerate: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get LUT from database or generate if not exists.

        Args:
            filter_names: List of filter names
            NA: Numerical aperture
            wavelength_range: Wavelength range for LUT generation
            wavelength_step: Wavelength step for LUT generation
            force_regenerate: If True, regenerate even if exists

        Returns:
            Tuple of (wavelengths, rgb_array, sigma_psf_array)
        """
        if not force_regenerate:
            lut_data = self.load_lut_from_database(filter_names, NA)
            if lut_data is not None:
                return lut_data

        # LUT not found, generate new one
        print(f"LUT not found. Generating new LUT for {len(filter_names)} filters, NA={NA}...")
        wavelengths, rgb_array, sigma_psf_array = self.generate_lut(
            filter_names, NA, wavelength_range, wavelength_step
        )

        self.save_lut_to_database(filter_names, NA, wavelengths, rgb_array, sigma_psf_array)

        return wavelengths, rgb_array, sigma_psf_array

    def nile_red_forward_model_lut(
        self,
        wavelength_center: float,
        filter_names: list,
        NA: float = 1.49
    ) -> Dict[str, float]:
        """Fast forward model using LUT interpolation with cached interpolators.

        This is ~100x faster than the full forward model calculation.
        Interpolation functions are cached to avoid recreation overhead.

        Args:
            wavelength_center: Central emission wavelength (nm)
            filter_names: List of filter names
            NA: Numerical aperture

        Returns:
            predictions: dict with keys 'R', 'G', 'B', 'sigma_x', 'sigma_y'
        """
        # Get configuration hash for cache lookup
        config_hash = self._get_config_hash(filter_names, NA)

        # Check if interpolators are already cached
        if config_hash in self._lut_interpolator_cache:
            rgb_interp, sigma_interp = self._lut_interpolator_cache[config_hash]
        else:
            # Get or create LUT data
            wavelengths, rgb_array, sigma_psf_array = self.get_or_create_lut(filter_names, NA)

            # Create interpolators once and cache them
            rgb_interp = interp1d(wavelengths, rgb_array.T, kind='linear', bounds_error=False, fill_value='extrapolate')  # type: ignore
            sigma_interp = interp1d(wavelengths, sigma_psf_array, kind='linear', bounds_error=False, fill_value='extrapolate')  # type: ignore

            # Store in cache
            self._lut_interpolator_cache[config_hash] = (rgb_interp, sigma_interp)

        # Use cached interpolators
        rgb_values = rgb_interp(wavelength_center)
        sigma_psf = float(sigma_interp(wavelength_center))

        return {
            'R': float(rgb_values[0]),
            'G': float(rgb_values[1]),
            'B': float(rgb_values[2]),
            'sigma_x': sigma_psf,
            'sigma_y': sigma_psf
        }

    def compute_sigma_psf_array(
        self,
        wavelength_array: np.ndarray,
        NA: float = 1.49
    ) -> np.ndarray:
        """Compute wavelength-dependent PSF widths.

        Args:
            wavelength_array: Wavelength grid (nm)
            NA: Numerical aperture (default: 1.49)

        Returns:
            sigma_psf_array: PSF width at each wavelength (nm)
        """
        sigma_psf_array = self.psf_funcs.sigma_PSF(wavelength_array, NA)
        return sigma_psf_array

    def setup_optical_system(
        self,
        filter_names: list
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Setup optical system by loading filter spectra and pixel efficiencies.

        Args:
            filter_names: List of filter/dichroic names to load

        Returns:
            wavelength_array: Wavelength grid (nm)
            pixel_QYs: Pixel quantum yields [B, G, R] shape (3, n_wavelengths)
            filter_spectra: Filter transmission curves (n_filters, n_wavelengths)
        """
        # Get pixel quantum efficiencies
        R, G, B, wavelength = self.spectral_funcs.getpixelefficiency()
        pixel_QYs = np.vstack([B, G, R])

        # Get filter transmission spectra
        filter_spectra = self.spectral_funcs.get_dye_or_filter_data(
            names=filter_names,
            wavelength=wavelength,
            dye_or_filter=False
        )

        return wavelength, pixel_QYs, filter_spectra

    def generate_nile_red_spectrum(
        self,
        wavelength_center: float,
        wavelength_array: np.ndarray,
        sigma_energy: Optional[float] = None,
        alpha: Optional[float] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate Nile Red emission spectrum using skew-Gaussian model.

        Args:
            wavelength_center: Central emission wavelength (nm) - FIT PARAMETER
            wavelength_array: Wavelength grid for spectrum (nm)
            sigma_energy: Gaussian width in energy space (eV), default from __init__
            alpha: Skewness parameter, default from __init__
            normalize: If True, normalize spectrum to unit sum

        Returns:
            spectrum: Emission spectrum I(λ) at wavelength_array points
        """
        # Use defaults if not provided
        if sigma_energy is None:
            sigma_energy = self.default_sigma_energy
        if alpha is None:
            alpha = self.default_alpha

        # Convert center wavelength to energy
        energy_center = self.spectral_funcs.wavelength_to_energy(np.array([wavelength_center]))[0]

        # Convert wavelength array to energy
        energy_array = self.spectral_funcs.wavelength_to_energy(wavelength_array)

        # Create skew-Gaussian in energy space
        # Amplitude will be normalized later, so set to 1 for now
        params = np.array([1.0, energy_center, sigma_energy, alpha])
        spectrum_energy = self.spectral_funcs.skew_gaussian_model(params, energy_array)

        # Transform to wavelength space with Jacobian and dipole moment weighting
        # I(λ) = I(E) / (E^(-3) * λ^2)
        weighting_factor = energy_array ** (-3) * wavelength_array ** 2
        spectrum_wavelength = spectrum_energy / weighting_factor

        # Normalize to unit sum if requested
        if normalize:
            total = np.trapz(spectrum_wavelength, wavelength_array)
            if total > 0:
                spectrum_wavelength = spectrum_wavelength / total

        return spectrum_wavelength

    def apply_optical_filters(
        self,
        spectrum: np.ndarray,
        filter_spectra: np.ndarray
    ) -> np.ndarray:
        """Apply optical filter transmission curves to emission spectrum.

        Args:
            spectrum: Emission spectrum I(λ)
            filter_spectra: Filter transmission curves (n_filters, n_wavelengths)

        Returns:
            spectrum_filtered: Spectrum after passing through optical path
        """
        # Multiply spectrum by all filter transmissions
        # filter_spectra should be (n_filters, n_wavelengths)
        total_transmission = np.prod(filter_spectra, axis=0)
        spectrum_filtered = spectrum * total_transmission

        return spectrum_filtered

    def calculate_rgb_from_spectrum(
        self,
        spectrum_filtered: np.ndarray,
        wavelength: np.ndarray,
        pixel_QYs: np.ndarray
    ) -> np.ndarray:
        """Calculate expected R, G, B intensities on Bayer sensor.

        Args:
            spectrum_filtered: Filtered emission spectrum
            wavelength: Wavelength array (nm)
            pixel_QYs: Pixel quantum yields [B, G, R] shape (3, n_wavelengths)

        Returns:
            rgb_predicted: [R, G, B] intensities (normalized to unit sum)
        """
        # Integrate spectrum weighted by pixel quantum efficiencies
        B_predicted = np.trapz(spectrum_filtered * pixel_QYs[0], wavelength)
        G_predicted = np.trapz(spectrum_filtered * pixel_QYs[1], wavelength)
        R_predicted = np.trapz(spectrum_filtered * pixel_QYs[2], wavelength)

        # Return as array [R, G, B]
        rgb_predicted = np.array([R_predicted, G_predicted, B_predicted])

        # Normalize to unit sum
        total = np.sum(rgb_predicted)
        if total > 0:
            rgb_predicted = rgb_predicted / total

        return rgb_predicted

    def calculate_psf_width_from_spectrum(
        self,
        spectrum_filtered: np.ndarray,
        wavelength: np.ndarray,
        NA: float = 1.49
    ) -> float:
        """Calculate expected PSF width from polychromatic spectrum using 1st moment.

        Uses the first spectral moment to calculate PSF width:
        σ_PSF = σ(<λ>) where <λ> = ∫ I(λ) λ dλ / ∫ I(λ) dλ

        This is ~1000x faster than the weighted average method and gives
        <0.15% difference in results (validated by comparison tests).

        Args:
            spectrum_filtered: Filtered emission spectrum
            wavelength: Wavelength array (nm)
            NA: Numerical aperture (default: 1.49)

        Returns:
            sigma_psf_predicted: Expected PSF width (nm)
        """
        # Calculate first spectral moment (mean wavelength)
        denominator = np.trapz(spectrum_filtered, wavelength)

        if denominator > 0:
            lambda_avg = np.trapz(spectrum_filtered * wavelength, wavelength) / denominator
            # Calculate PSF width at the mean wavelength
            sigma_psf_predicted = self.psf_funcs.sigma_PSF(lambda_avg, NA)
        else:
            # Fallback: use center of wavelength range
            lambda_avg = np.mean(wavelength)
            sigma_psf_predicted = self.psf_funcs.sigma_PSF(lambda_avg, NA)

        return sigma_psf_predicted

    def nile_red_forward_model(
        self,
        wavelength_center: float,
        filter_spectra: np.ndarray,
        wavelength_array: np.ndarray,
        pixel_QYs: np.ndarray,
        NA: float = 1.49
    ) -> Dict[str, float]:
        """Complete forward model: wavelength_center → (R, G, B, σ_PSF).

        Args:
            wavelength_center: Central emission wavelength (nm) - FIT PARAMETER
            filter_spectra: Optical filter transmission curves (n_filters, n_wavelengths)
            wavelength_array: Wavelength grid (nm)
            pixel_QYs: Bayer pixel quantum yields [B, G, R] shape (3, n_wavelengths)
            NA: Numerical aperture (default: 1.49)

        Returns:
            predictions: dict with keys 'R', 'G', 'B', 'sigma_x', 'sigma_y'
        """
        # 1. Generate emission spectrum
        spectrum = self.generate_nile_red_spectrum(
            wavelength_center,
            wavelength_array,
            normalize=True
        )

        # 2. Apply optical filters
        spectrum_filtered = self.apply_optical_filters(
            spectrum,
            filter_spectra
        )

        # 3. Predict RGB values
        rgb = self.calculate_rgb_from_spectrum(
            spectrum_filtered,
            wavelength_array,
            pixel_QYs
        )

        # 4. Predict PSF width (assume circular PSF: σ_x = σ_y)
        sigma_psf = self.calculate_psf_width_from_spectrum(
            spectrum_filtered,
            wavelength_array,
            NA
        )

        predictions = {
            'R': rgb[0],
            'G': rgb[1],
            'B': rgb[2],
            'sigma_x': sigma_psf,
            'sigma_y': sigma_psf
        }

        return predictions

    def chi_squared_nile_red(
        self,
        wavelength_center: float,
        observed_data: Dict[str, float],
        errors: Dict[str, float],
        filter_spectra: np.ndarray,
        wavelength_array: np.ndarray,
        pixel_QYs: np.ndarray,
        NA: float = 1.49
    ) -> float:
        """Chi-squared for fitting central wavelength to experimental data.

        This is a convenience function that computes chi² = sum(residuals²).
        Uses residuals_nile_red internally to avoid code duplication.

        Args:
            wavelength_center: Central wavelength to test (nm)
            observed_data: dict with 'R', 'G', 'B', 'sigma_x', 'sigma_y'
            errors: dict with same keys as observed_data
            filter_spectra: Optical filter transmission curves
            wavelength_array: Wavelength grid (nm)
            pixel_QYs: Pixel quantum yields
            NA: Numerical aperture (default: 1.49)

        Returns:
            chi2: Chi-squared value
        """
        # Get residual vector and compute chi-squared
        residuals = self.residuals_nile_red(
            np.array([wavelength_center]),
            observed_data,
            errors,
            filter_spectra,
            wavelength_array,
            pixel_QYs,
            NA
        )
        return float(np.sum(residuals**2))

    def residuals_nile_red(
        self,
        wavelength_center: np.ndarray,
        observed_data: Dict[str, float],
        errors: Dict[str, float],
        filter_spectra: np.ndarray,
        wavelength_array: np.ndarray,
        pixel_QYs: np.ndarray,
        NA: float = 1.49
    ) -> np.ndarray:
        """Residual vector for least-squares fitting of Nile Red wavelength.

        This function returns the vector of weighted residuals rather than chi-squared,
        which is more appropriate for least-squares optimization algorithms.

        Args:
            wavelength_center: Central wavelength parameter (1D array with single element)
            observed_data: dict with 'R', 'G', 'B', 'sigma_x', 'sigma_y'
            errors: dict with same keys as observed_data
            filter_spectra: Optical filter transmission curves
            wavelength_array: Wavelength grid (nm)
            pixel_QYs: Pixel quantum yields
            NA: Numerical aperture (default: 1.49)

        Returns:
            residuals: Array of weighted residuals (observation - prediction) / error
        """
        # Extract scalar wavelength (least_squares passes 1D array)
        wl = wavelength_center[0] if isinstance(wavelength_center, np.ndarray) else wavelength_center

        # Get predictions from forward model
        predictions = self.nile_red_forward_model(
            wl,
            filter_spectra,
            wavelength_array,
            pixel_QYs,
            NA
        )

        # Build residual vector
        residuals = []
        for key in ['R', 'G', 'B', 'sigma_x', 'sigma_y']:
            if key in observed_data and key in errors:
                if errors[key] > 0:
                    residual = (observed_data[key] - predictions[key]) / errors[key]
                    residuals.append(residual)

        return np.array(residuals)

    def fit_nile_red_wavelength(
        self,
        observed_rgb: np.ndarray,
        observed_sigma_x: float,
        observed_sigma_y: float,
        rgb_errors: np.ndarray,
        sigma_x_error: float,
        sigma_y_error: float,
        filter_spectra: np.ndarray,
        wavelength_array: np.ndarray,
        pixel_QYs: np.ndarray,
        NA: float = 1.49,
        wavelength_bounds: Tuple[float, float] = (550.0, 750.0)
    ) -> Tuple[float, Dict[str, float]]:
        """Fit central wavelength of Nile Red emission from experimental data.

        Args:
            observed_rgb: [R, G, B] measured intensities
            observed_sigma_x, observed_sigma_y: Measured PSF widths (nm)
            rgb_errors: Errors on [R, G, B]
            sigma_x_error, sigma_y_error: Errors on PSF widths (nm)
            filter_spectra: Optical filter transmission curves
            wavelength_array: Wavelength grid (nm)
            pixel_QYs: Pixel quantum yields
            NA: Numerical aperture (default: 1.49)
            wavelength_bounds: Search range for wavelength (nm)

        Returns:
            wavelength_center: Fitted central wavelength (nm)
            predictions: Predicted values at best-fit wavelength
        """
        # Normalize RGB to unit sum
        observed_rgb_norm = observed_rgb / np.sum(observed_rgb)
        rgb_errors_norm = rgb_errors / np.sum(observed_rgb)

        observed_data = {
            'R': observed_rgb_norm[0],
            'G': observed_rgb_norm[1],
            'B': observed_rgb_norm[2],
            'sigma_x': observed_sigma_x,
            'sigma_y': observed_sigma_y
        }

        errors = {
            'R': rgb_errors_norm[0],
            'G': rgb_errors_norm[1],
            'B': rgb_errors_norm[2],
            'sigma_x': sigma_x_error,
            'sigma_y': sigma_y_error
        }

        # Initial guess: use default central wavelength or midpoint of bounds
        x0 = np.array([self.default_wavelength_center])
        if x0[0] < wavelength_bounds[0] or x0[0] > wavelength_bounds[1]:
            x0 = np.array([(wavelength_bounds[0] + wavelength_bounds[1]) / 2])

        # Fit using Trust Region Reflective algorithm (handles bounds well)
        result = least_squares(
            fun=self.residuals_nile_red,
            x0=x0,
            bounds=(wavelength_bounds[0], wavelength_bounds[1]),
            method='trf',  # Trust Region Reflective
            args=(observed_data, errors, filter_spectra, wavelength_array, pixel_QYs, NA)
        )

        wavelength_center = result.x[0]

        # Get predictions at best fit
        predictions = self.nile_red_forward_model(
            wavelength_center,
            filter_spectra,
            wavelength_array,
            pixel_QYs,
            NA
        )

        return wavelength_center, predictions

    def simulate_wavelength_precision(
        self,
        save_folder: str,
        wavelength_range: Tuple[float, float] = (580.0, 690.0),
        wavelength_step: float = 5.0,
        photon_counts: np.ndarray = np.array([1000, 2000, 5000, 10000]),
        n_bootstrap: int = 1000,
        filter_names: Optional[list] = None,
        NA: float = 1.49,
        pixel_size: float = 69.0,
        camera_parameters: Optional[dict] = None,
        image_size: int = 16,
        smoothing_function = None,
        background_photons: float = 40.0,
        starting_flag: str = "",
        save_raw_results: bool = True,
        cpu_fraction: float = 0.9,
        verbose: bool = True,
        use_tqdm: bool = False
    ) -> None:
        """Simulate wavelength precision using two-stage workflow.

        Stage 1: Use Multicolour_Simulation_Functions to simulate images and fit them
        Stage 2: Post-process fit results to extract wavelengths using inverse model

        This method generates Nile Red spectra at different wavelengths, simulates
        imaging and fitting using existing infrastructure, then extracts wavelengths
        from the RGB+PSF fit results.

        Args:
            save_folder: Directory to save results (created if doesn't exist)
            wavelength_range: (min, max) wavelength range in nm (default: 580-690)
            wavelength_step: Wavelength step size in nm (default: 5)
            photon_counts: Array of photon counts to simulate (default: [1k, 2k, 5k, 10k])
            n_bootstrap: Number of Monte Carlo realizations per condition (default: 1000)
            filter_names: List of filter/dichroic names (default: Nile Red filters)
            NA: Numerical aperture (default: 1.49)
            pixel_size: Camera pixel size in nm (default: 69)
            camera_parameters: Camera calibration dict (default: ideal camera)
            image_size: Size of simulated images in pixels (default: 20)
            smoothing_function: Smoothing function for PSF (default: gaussian_filter)
            background_photons: Background photon count (default: 20)
            starting_flag: Prefix for saved files (default: "")
            save_raw_results: If True, save raw fit results (default: True)
            cpu_fraction: Fraction of CPUs to use for parallel processing
            verbose: Print progress messages (default: True)
            use_tqdm: Use tqdm for Jupyter-compatible progress bars (default: False)

        Saves per wavelength:
            - Standard simulation outputs from test_fit_method (RMSE_mean, RMSE_std, etc.)
            - wavelength_precision_summary.csv: Extracted wavelength statistics
        """
        import polars as pl
        import os
        import time
        import Multicolour_Simulation_Functions
        import MaskFunctions

        # Create save folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Default filter configuration for Nile Red
        if filter_names is None:
            filter_names = [
                "semrock-ff01-650-200",
                "semrock-di03-r514-t1-25x36",
                "semrock-ff01-515-lp"
            ]

        # Setup optical system
        wavelength_array, pixel_QYs, _ = self.setup_optical_system(filter_names)

        # Setup default camera parameters if not provided
        if camera_parameters is None:
            M_F = MaskFunctions.Mask_Functions()
            masks = M_F.get_masks(size_x=image_size, size_y=image_size)

            # Use realistic camera parameters (median from calibrations)
            camera_parameters = {
                'gain': np.ones((image_size, image_size)) * 0.48,
                'offset': np.ones((image_size, image_size)) * 100.0,
                'variance': np.ones((image_size, image_size)) * 0.938,
                'readnoise': np.ones((image_size, image_size)) * 2.0,
                'rqe': np.ones((image_size, image_size)),
                'pixel_QYs': pixel_QYs,
                'pixel_order': ['B', 'G', 'R'],
                'pixel_order_indices': [0, 1, 2],
                'masks': masks
            }

        # Initialize simulation
        MSF = Multicolour_Simulation_Functions.MultiC_Sim_Funcs()

        # Setup smoothing function
        if smoothing_function is None:
            import sCMOSFunctions
            import types
            sCMOS = sCMOSFunctions.sCMOS_Functions()
            smoothing_function = types.SimpleNamespace()
            smoothing_function.args = {"sigma": 1.5}
            smoothing_function.extent = 1.5
            smoothing_function.smoothing_function = sCMOS.gaussian_filter_stack
            smoothing_function.data_arg = "image"

        # Generate wavelength grid
        wavelengths_true = np.arange(wavelength_range[0], wavelength_range[1] + wavelength_step, wavelength_step)
        n_wavelengths = len(wavelengths_true)

        start_time = time.time()

        # Setup progress display
        if use_tqdm:
            try:
                from tqdm.auto import tqdm
            except ImportError:
                if verbose:
                    print("Warning: tqdm not installed, falling back to print-based progress")
                use_tqdm = False

        if verbose:
            print(f"\n{'='*60}")
            print(f"Nile Red Wavelength Precision Simulation")
            print(f"{'='*60}")
            print(f"Wavelength range: {wavelength_range[0]}-{wavelength_range[1]} nm (step={wavelength_step} nm)")
            print(f"Number of wavelengths: {n_wavelengths}")
            print(f"Photon counts: {photon_counts}")
            print(f"Bootstrap samples: {n_bootstrap}")
            print(f"Save folder: {save_folder}")
            print(f"{'='*60}\n")

        # STAGE 1: Simulate and fit images for each wavelength
        wavelength_iterator = enumerate(wavelengths_true)
        if use_tqdm and verbose:
            wavelength_iterator = tqdm(wavelength_iterator, total=n_wavelengths, desc="Stage 1: Simulating wavelengths")

        for i, wl_true in wavelength_iterator:
            if verbose and not use_tqdm:
                elapsed = (time.time() - start_time) / 60.0
                print(f"\n[{i+1}/{n_wavelengths}] Processing wavelength {wl_true:.1f} nm (elapsed: {elapsed:.1f} min)")

            # Generate Nile Red spectrum for this wavelength
            spectrum = self.generate_nile_red_spectrum(wl_true, wavelength_array, normalize=True)

            # Create dye name for this wavelength
            dye_name = f"simulated_NileRed_{int(wl_true)}nm"

            # Run standard simulation using test_fit_method
            flag = f"{starting_flag}wl{int(wl_true)}_"

            # Create SimulationConfig
            import Multicolour_Simulation_Functions as MSF_module
            config = MSF_module.SimulationConfig(
                n_bootstrap=n_bootstrap,
                background_photons=background_photons,
                NA=NA,
                pixel_size=pixel_size,
                cpu_fraction=cpu_fraction,
                save_raw_results=save_raw_results,
                subtractx0y0=False,
                saverawimages=False
            )

            MSF.test_fit_method(
                dye=dye_name,
                filters=filter_names,
                wavelength=wavelength_array,
                camera_parameters=camera_parameters,
                save_folder=save_folder,
                n_photon_space=photon_counts,
                smoothing_function=smoothing_function,
                starting_flag=flag,
                config=config,
                single_dye_spectrum=spectrum,  # Pass spectrum directly
                nile_red_wavelength=wl_true  # Pass wavelength for inverse fitting
            )

        if verbose:
            total_elapsed = (time.time() - start_time) / 60.0
            print(f"\n{'='*60}")
            print(f"Stage 1 complete (simulation + wavelength fitting): {total_elapsed:.1f} min")
            print(f"{'='*60}\n")
            print("Starting Stage 2: Calculate statistics from fitted wavelengths...")

        # STAGE 2: Calculate statistics from wavelength columns in raw results
        wavelength_precision_results = []

        stats_iterator = enumerate(wavelengths_true)
        if use_tqdm and verbose:
            stats_iterator = tqdm(stats_iterator, total=n_wavelengths, desc="Stage 2: Calculating statistics")

        for i, wl_true in stats_iterator:
            if verbose and not use_tqdm:
                print(f"  [{i+1}/{n_wavelengths}] Processing statistics for {wl_true:.1f} nm", end='\r', flush=True)

            flag = f"{starting_flag}wl{int(wl_true)}_"

            # Find raw results files for this wavelength (support both parquet and csv)
            raw_files = [f for f in os.listdir(save_folder)
                        if f.startswith(flag) and ('rawresults.parquet' in f or 'rawresults.csv' in f)]

            for raw_file in raw_files:
                # Extract photon count from filename
                parts = raw_file.replace('.parquet', '').replace('.csv', '').split('_')
                photon_str = [p for p in parts if 'p' in p and p.replace('p', '').replace('.', '').isdigit()]
                if not photon_str:
                    continue

                n_photons = float(photon_str[0].replace('p', '.'))

                # Load fit results with wavelength columns (auto-detect format)
                file_path = os.path.join(save_folder, raw_file)
                if raw_file.endswith('.parquet'):
                    df = pl.read_parquet(file_path)
                else:
                    df = pl.read_csv(file_path)

                # Check if wavelength columns exist
                if 'wl_fit' not in df.columns:
                    if verbose:
                        print(f"\nWarning: No 'wl_fit' column found in {raw_file}")
                        print("This may be from an older simulation. Re-run simulation to add wavelength fits.")
                    continue

                # Extract fitted wavelengths (excluding NaN values)
                wavelengths_fitted = df['wl_fit'].to_numpy()
                wavelengths_fitted = wavelengths_fitted[~np.isnan(wavelengths_fitted)]

                # Calculate statistics
                if len(wavelengths_fitted) > 0:
                    precision = np.std(wavelengths_fitted)
                    bias = np.mean(wavelengths_fitted) - wl_true
                    recovery_rate = len(wavelengths_fitted) / n_bootstrap

                    wavelength_precision_results.append({
                        'wavelength_true': wl_true,
                        'n_photons': n_photons,
                        'wavelength_precision': precision,
                        'wavelength_bias': bias,
                        'wavelength_mean': np.mean(wavelengths_fitted),
                        'recovery_rate': recovery_rate,
                        'n_successful': len(wavelengths_fitted)
                    })

        # Save wavelength precision summary
        if len(wavelength_precision_results) > 0:
            summary_df = pl.DataFrame(wavelength_precision_results)
            summary_file = os.path.join(save_folder, f"{starting_flag}wavelength_precision_summary.csv")
            summary_df.write_csv(summary_file)

            if verbose:
                print(f"\n\n{'='*60}")
                print(f"Simulation complete!")
                print(f"Total time: {(time.time() - start_time) / 60.0:.1f} min")
                print(f"Results saved to: {save_folder}")
                print(f"Wavelength precision summary: {summary_file}")
                print(f"{'='*60}\n")
