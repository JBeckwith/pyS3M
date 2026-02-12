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
import pandas as pd
from scipy.optimize import least_squares
from typing import Dict, Tuple, Optional, List
import SpectralFunctions
import PSFFunctions


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
        wavelength_center_init: float = 617.6,
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

    def compute_sigma_psf_array(
        self, wavelength_array: np.ndarray, NA: float = 1.49
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
        self, filter_names: list
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
            names=filter_names, wavelength=wavelength, dye_or_filter=False
        )

        return wavelength, pixel_QYs, filter_spectra

    def generate_nile_red_spectrum(
        self,
        wavelength_center: float,
        wavelength_array: np.ndarray,
        sigma_energy: Optional[float] = None,
        alpha: Optional[float] = None,
        normalize: bool = True,
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
        energy_center = self.spectral_funcs.wavelength_to_energy(
            np.array([wavelength_center])
        )[0]

        # Convert wavelength array to energy
        energy_array = self.spectral_funcs.wavelength_to_energy(wavelength_array)

        # Create skew-Gaussian in energy space
        # Amplitude will be normalized later, so set to 1 for now
        params = np.array([1.0, energy_center, sigma_energy, alpha])
        spectrum_energy = self.spectral_funcs.skew_gaussian_model(params, energy_array)

        # Transform to wavelength space with Jacobian and dipole moment weighting
        # I(λ) = I(E) / (E^(-3) * λ^2)
        weighting_factor = energy_array ** (-3) * wavelength_array**2
        spectrum_wavelength = spectrum_energy / weighting_factor

        # Normalize to unit sum if requested
        if normalize:
            total = np.trapz(spectrum_wavelength, wavelength_array)
            if total > 0:
                spectrum_wavelength = spectrum_wavelength / total

        return spectrum_wavelength

    def apply_optical_filters(
        self, spectrum: np.ndarray, filter_spectra: np.ndarray
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
        pixel_QYs: np.ndarray,
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
        self, spectrum_filtered: np.ndarray, wavelength: np.ndarray, NA: float = 1.49
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
            lambda_avg = (
                np.trapz(spectrum_filtered * wavelength, wavelength) / denominator
            )
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
        NA: float = 1.49,
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
            wavelength_center, wavelength_array, normalize=True
        )

        # 2. Apply optical filters
        spectrum_filtered = self.apply_optical_filters(spectrum, filter_spectra)

        # 3. Predict RGB values
        rgb = self.calculate_rgb_from_spectrum(
            spectrum_filtered, wavelength_array, pixel_QYs
        )

        # 4. Predict PSF width (assume circular PSF: σ_x = σ_y)
        sigma_psf = self.calculate_psf_width_from_spectrum(
            spectrum_filtered, wavelength_array, NA
        )

        predictions = {
            "R": rgb[0],
            "G": rgb[1],
            "B": rgb[2],
            "sigma_x": sigma_psf,
            "sigma_y": sigma_psf,
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
        NA: float = 1.49,
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
            NA,
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
        NA: float = 1.49,
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
        wl = (
            wavelength_center[0]
            if isinstance(wavelength_center, np.ndarray)
            else wavelength_center
        )

        # Get predictions from forward model
        predictions = self.nile_red_forward_model(
            wl, filter_spectra, wavelength_array, pixel_QYs, NA
        )

        # Build residual vector
        residuals = []
        for key in ["R", "G", "B", "sigma_x", "sigma_y"]:
            if key in observed_data and key in errors:
                if errors[key] > 0:
                    residual = (observed_data[key] - predictions[key]) / errors[key]
                    residuals.append(residual)

        return np.array(residuals)

    def _error_inflation_factor(self, snr: float) -> float:
        """Calculate error inflation factor based on SNR.

        At low SNR, fit errors systematically underestimate true uncertainty
        due to noise floor, non-negativity constraints, and normalization bias.
        This function provides empirical inflation factors to correct for this.

        Args:
            snr: Signal-to-noise ratio

        Returns:
            inflation_factor: Multiplicative factor for error (>= 1.0)
        """
        if snr < 2.0:
            return 3.0
        elif snr < 5.0:
            return 2.0
        elif snr < 10.0:
            return 1.5
        else:
            return 1.0

    def _calculate_channel_snr(
        self, observed_rgb: np.ndarray, total_photons: float, background_photons: float
    ) -> np.ndarray:
        """Calculate signal-to-noise ratio for each RGB channel.

        SNR = S / sqrt(S + B) where:
        - S = signal photons in channel
        - B = background photons in channel

        Args:
            observed_rgb: [R, G, B] measured intensities (normalized or absolute)
            total_photons: Total signal photon count
            background_photons: Total background photons (distributed evenly across RGB)

        Returns:
            snr_array: [SNR_R, SNR_G, SNR_B]
        """
        # Normalize RGB if needed
        rgb_fractions = observed_rgb / np.sum(observed_rgb)

        # Calculate signal photons per channel
        signal_photons = rgb_fractions * total_photons

        # Background photons per channel (assume uniform distribution)
        background_per_channel = background_photons / 3.0

        # Calculate SNR for each channel
        snr_array = signal_photons / np.sqrt(signal_photons + background_per_channel)

        return snr_array

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
        wavelength_bounds: Tuple[float, float] = (500.0, 750.0),
        total_photons: Optional[float] = None,
        background_photons: float = 40.0,
        apply_snr_inflation: bool = True,
        wavelength_initial_guess: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Fit central wavelength of Nile Red emission from experimental data.

        Uses all three RGB channels for wavelength fitting. Implements SNR-based error
        inflation to correct for underestimated errors at low photon counts:
        - SNR < 2: inflate by 3.0x
        - SNR 2-5: inflate by 2.0x
        - SNR 5-10: inflate by 1.5x
        - SNR > 10: use as-is (1.0x)

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
            total_photons: Total photon count for SNR calculation (optional)
            background_photons: Background photon count (default: 40.0, distributed across RGB)
            apply_snr_inflation: Apply SNR-based error inflation (default: True)
            wavelength_initial_guess: Custom initial guess for wavelength (nm).
                If None, uses self.default_wavelength_center (617.6 nm).

        Returns:
            wavelength_center: Fitted central wavelength (nm)
            predictions: Predicted values at best-fit wavelength
        """
        # Normalize RGB to unit sum
        observed_rgb_norm = observed_rgb / np.sum(observed_rgb)
        rgb_errors_norm = rgb_errors / np.sum(observed_rgb)

        # Apply SNR-based error inflation if requested and total_photons is provided
        if apply_snr_inflation and total_photons is not None:
            # Calculate SNR for each channel
            snr_rgb = self._calculate_channel_snr(
                observed_rgb, total_photons, background_photons
            )

            # Apply inflation factors
            inflation_factors = np.array(
                [
                    self._error_inflation_factor(snr_rgb[0]),  # R
                    self._error_inflation_factor(snr_rgb[1]),  # G
                    self._error_inflation_factor(snr_rgb[2]),  # B
                ]
            )

            # Inflate errors
            rgb_errors_norm = rgb_errors_norm * inflation_factors

        # Use all three RGB channels for wavelength fitting
        observed_data = {
            "R": observed_rgb_norm[0],
            "G": observed_rgb_norm[1],
            "B": observed_rgb_norm[2],
            "sigma_x": observed_sigma_x,
            "sigma_y": observed_sigma_y,
        }

        errors = {
            "R": rgb_errors_norm[0],
            "G": rgb_errors_norm[1],
            "B": rgb_errors_norm[2],
            "sigma_x": sigma_x_error,
            "sigma_y": sigma_y_error,
        }

        # Initial guess: use custom guess, default central wavelength, or midpoint of bounds
        if wavelength_initial_guess is not None:
            x0 = np.array([wavelength_initial_guess])
        else:
            x0 = np.array([self.default_wavelength_center])
        if x0[0] < wavelength_bounds[0] or x0[0] > wavelength_bounds[1]:
            x0 = np.array([(wavelength_bounds[0] + wavelength_bounds[1]) / 2])

        # Fit using Trust Region Reflective algorithm (handles bounds well)
        result = least_squares(
            fun=self.residuals_nile_red,
            x0=x0,
            bounds=(wavelength_bounds[0], wavelength_bounds[1]),
            method="trf",  # Trust Region Reflective
            args=(
                observed_data,
                errors,
                filter_spectra,
                wavelength_array,
                pixel_QYs,
                NA,
            ),
        )

        wavelength_center = result.x[0]

        # Get predictions at best fit
        predictions = self.nile_red_forward_model(
            wavelength_center, filter_spectra, wavelength_array, pixel_QYs, NA
        )

        return wavelength_center, predictions

    def simulate_wavelength_precision(
        self,
        save_folder: str,
        wavelength_range: Tuple[float, float] = (560.0, 620.0),
        wavelength_step: float = 5.0,
        photon_counts: np.ndarray = np.array([1000, 2000, 5000, 10000]),
        n_bootstrap: int = 1000,
        filter_names: Optional[list] = None,
        NA: float = 1.49,
        pixel_size: float = 69.0,
        camera_parameters: Optional[dict] = None,
        image_size: int = 16,
        smoothing_function=None,
        background_photons: float = 40.0,
        starting_flag: str = "",
        save_raw_results: bool = True,
        cpu_fraction: float = 0.9,
        verbose: bool = True,
        use_tqdm: bool = False,
    ) -> None:
        """Simulate wavelength precision using two-stage workflow.

        Stage 1: Use Multicolour_Simulation_Functions to simulate images and fit them
        Stage 2: Post-process fit results to extract wavelengths using inverse model

        This method generates Nile Red spectra at different wavelengths, simulates
        imaging and fitting using existing infrastructure, then extracts wavelengths
        from the RGB+PSF fit results.

        Args:
            save_folder: Directory to save results (created if doesn't exist)
            wavelength_range: (min, max) wavelength range in nm (default: 560-620)
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
                "semrock-ff01-515-lp",
            ]

        # Setup optical system
        wavelength_array, pixel_QYs, _ = self.setup_optical_system(filter_names)

        # Setup default camera parameters if not provided
        if camera_parameters is None:
            M_F = MaskFunctions.Mask_Functions()
            masks = M_F.get_masks(size_x=image_size, size_y=image_size)

            # Use realistic camera parameters (median from calibrations)
            camera_parameters = {
                "gain": np.ones((image_size, image_size)) * 0.48,
                "offset": np.ones((image_size, image_size)) * 100.0,
                "variance": np.ones((image_size, image_size)) * 0.938,
                "readnoise": np.ones((image_size, image_size)) * 2.0,
                "rqe": np.ones((image_size, image_size)),
                "pixel_QYs": pixel_QYs,
                "pixel_order": ["B", "G", "R"],
                "pixel_order_indices": [0, 1, 2],
                "masks": masks,
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
        wavelengths_true = np.arange(
            wavelength_range[0], wavelength_range[1] + wavelength_step, wavelength_step
        )
        n_wavelengths = len(wavelengths_true)

        start_time = time.time()

        # Setup progress display
        if use_tqdm:
            try:
                from tqdm.auto import tqdm  # type: ignore
            except ImportError:
                if verbose:
                    print(
                        "Warning: tqdm not installed, falling back to print-based progress"
                    )
                use_tqdm = False

        if verbose:
            print(f"\n{'='*60}")
            print(f"Nile Red Wavelength Precision Simulation")
            print(f"{'='*60}")
            print(
                f"Wavelength range: {wavelength_range[0]}-{wavelength_range[1]} nm (step={wavelength_step} nm)"
            )
            print(f"Number of wavelengths: {n_wavelengths}")
            print(f"Photon counts: {photon_counts}")
            print(f"Bootstrap samples: {n_bootstrap}")
            print(f"Save folder: {save_folder}")
            print(f"{'='*60}\n")

        # STAGE 1: Simulate and fit images for each wavelength
        wavelength_iterator = enumerate(wavelengths_true)
        if use_tqdm and verbose:
            from tqdm.auto import tqdm  # Import here to avoid unbound error

            wavelength_iterator = tqdm(
                wavelength_iterator,
                total=n_wavelengths,
                desc="Stage 1: Simulating wavelengths",
            )

        for i, wl_true in wavelength_iterator:
            if verbose and not use_tqdm:
                elapsed = (time.time() - start_time) / 60.0
                print(
                    f"\n[{i+1}/{n_wavelengths}] Processing wavelength {wl_true:.1f} nm (elapsed: {elapsed:.1f} min)"
                )

            # Generate Nile Red spectrum for this wavelength
            spectrum = self.generate_nile_red_spectrum(
                wl_true, wavelength_array, normalize=True
            )

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
                saverawimages=False,
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
                nile_red_wavelength=wl_true,  # Pass wavelength for inverse fitting
            )

        if verbose:
            total_elapsed = (time.time() - start_time) / 60.0
            print(f"\n{'='*60}")
            print(
                f"Stage 1 complete (simulation + wavelength fitting): {total_elapsed:.1f} min"
            )
            print(f"{'='*60}\n")
            print("Starting Stage 2: Calculate statistics from fitted wavelengths...")

        # STAGE 2: Calculate statistics from wavelength columns in raw results
        wavelength_precision_results = []

        stats_iterator = enumerate(wavelengths_true)
        if use_tqdm and verbose:
            from tqdm.auto import tqdm  # Import here to avoid unbound error

            stats_iterator = tqdm(
                stats_iterator,
                total=n_wavelengths,
                desc="Stage 2: Calculating statistics",
            )

        for i, wl_true in stats_iterator:
            if verbose and not use_tqdm:
                print(
                    f"  [{i+1}/{n_wavelengths}] Processing statistics for {wl_true:.1f} nm",
                    end="\r",
                    flush=True,
                )

            flag = f"{starting_flag}wl{int(wl_true)}_"

            # Find raw results h5 files for this wavelength
            raw_files = [
                f
                for f in os.listdir(save_folder)
                if f.startswith(flag) and f.endswith("rawresults.h5")
            ]

            for raw_file in raw_files:
                # Load h5 file using pandas then convert to polars
                import pandas as pd

                file_path = os.path.join(save_folder, raw_file)
                df_pandas = pd.read_hdf(file_path, "data")
                df = pl.from_pandas(df_pandas)

                # Check if wavelength columns exist
                if "wl_fit" not in df.columns:
                    if verbose:
                        print(f"\nWarning: No 'wl_fit' column found in {raw_file}")
                        print(
                            "This may be from an older simulation. Re-run simulation to add wavelength fits."
                        )
                    continue

                # Process each photon level separately
                if "photon_level" not in df.columns:
                    if verbose:
                        print(f"\nWarning: No 'photon_level' column found in {raw_file}")
                    continue

                for level in df["photon_level"].unique():
                    df_level = df.filter(pl.col("photon_level") == level)

                    # Get mean photon count for this level
                    n_photons = df_level["photons"].mean()

                    # Extract fitted wavelengths (excluding NaN values)
                    wavelengths_fitted = df_level["wl_fit"].to_numpy()
                    wavelengths_fitted = wavelengths_fitted[~np.isnan(wavelengths_fitted)]

                    # Calculate statistics
                    if len(wavelengths_fitted) > 0:
                        precision = np.std(wavelengths_fitted)
                        bias = np.mean(wavelengths_fitted) - wl_true
                        recovery_rate = len(wavelengths_fitted) / n_bootstrap

                        wavelength_precision_results.append(
                            {
                                "wavelength_true": wl_true,
                                "n_photons": n_photons,
                                "wavelength_precision": precision,
                                "wavelength_bias": bias,
                                "wavelength_mean": np.mean(wavelengths_fitted),
                                "recovery_rate": recovery_rate,
                                "n_successful": len(wavelengths_fitted),
                            }
                        )

        # Save wavelength precision summary
        if len(wavelength_precision_results) > 0:
            summary_df = pl.DataFrame(wavelength_precision_results)
            summary_file = os.path.join(
                save_folder, f"{starting_flag}wavelength_precision_summary.csv"
            )
            summary_df.write_csv(summary_file)

            if verbose:
                print(f"\n\n{'='*60}")
                print(f"Simulation complete!")
                print(f"Total time: {(time.time() - start_time) / 60.0:.1f} min")
                print(f"Results saved to: {save_folder}")
                print(f"Wavelength precision summary: {summary_file}")
                print(f"{'='*60}\n")

    def fit_wavelengths_from_h5(
        self,
        h5_path: str,
        filter_names: List[str],
        camera_parameters: Dict,
        wavelength_bounds: Tuple[float, float] = (500.0, 750.0),
        NA: float = 1.49,
        pixel_size: float = 69.0,
        output_path: Optional[str] = None,
        cpu_fraction: float = 0.9,
        verbose: bool = True,
        aggregate_id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fit Nile Red wavelengths from localizations stored in HDF5 file.

        Convenience function that loads an HDF5 file containing localization data
        (RGB intensities, PSF widths, and errors), fits the Nile Red wavelength
        for each localization using parallel processing, and returns/saves the
        updated DataFrame with wavelength columns added.

        When aggregate_id_column is provided, uses a two-step fitting approach:
        1. Compute weighted-average A_R, A_G, A_B, s_x, s_y per aggregate and
           fit wavelength for each aggregate (higher SNR → more stable fits)
        2. Use each aggregate's fitted wavelength as the initial guess when
           fitting individual localisations within that aggregate

        Args:
            h5_path: Path to HDF5 file containing localization data
            filter_names: List of filter/dichroic names used in optical path
            camera_parameters: Camera parameters dict containing:
                - 'pixel_QYs': Pixel quantum yields vs wavelength (if 'wavelength' not provided)
                - 'wavelength': Wavelength array (nm) - optional if pixel_QYs shape implies it
            wavelength_bounds: Search range for wavelength fitting (nm), default: (500, 750)
            NA: Numerical aperture, default: 1.49
            pixel_size: Camera pixel size in nm, default: 69.0
            output_path: Optional path to save updated HDF5 file (if None, doesn't save)
            cpu_fraction: Fraction of CPUs to use for parallel fitting, default: 0.9
            verbose: Print progress messages, default: True
            aggregate_id_column: Column name containing aggregate/punctum IDs (e.g. 'cluster_id').
                When provided, enables two-step fitting with aggregate-level priors.
                When None (default), all fits use the default initial guess.

        Returns:
            pd.DataFrame: Updated DataFrame with 'wl_fit' and 'wl_fit_err' columns added.
                When aggregate_id_column is provided, also adds 'wl_fit_aggregate' column
                containing the per-aggregate fitted wavelength.

        Required columns in HDF5 file:
            - A_R, A_G, A_B: RGB amplitudes (normalized)
            - s_x, s_y: PSF widths (pixels)
            - A_R_err, A_G_err, A_B_err: RGB amplitude errors
            - s_x_err, s_y_err: PSF width errors (pixels)
            - photons: Total photon count (for SNR calculation)
            - background_photons: Background photon count (for SNR calculation)

        Example:
            >>> import NileRedFunctions
            >>> nrf = NileRedFunctions.NileRed_Functions()
            >>>
            >>> # Define optical configuration
            >>> filters = [
            ...     "semrock-ff01-650-200",
            ...     "semrock-di03-r514-t1-25x36",
            ...     "semrock-ff01-515-lp",
            ... ]
            >>>
            >>> # Setup camera parameters (or load from calibration)
            >>> import SpectralFunctions
            >>> sf = SpectralFunctions.Spectral_Funcs()
            >>> R, G, B, wavelength = sf.getpixelefficiency()
            >>> pixel_QYs = np.vstack([B, G, R])
            >>> camera_params = {
            ...     'pixel_QYs': pixel_QYs,
            ...     'wavelength': wavelength,
            ... }
            >>>
            >>> # Fit wavelengths with aggregate priors
            >>> df_with_wavelengths = nrf.fit_wavelengths_from_h5(
            ...     h5_path='results_aggregatelocs.h5',
            ...     filter_names=filters,
            ...     camera_parameters=camera_params,
            ...     output_path='results_aggregatelocs.h5',
            ...     aggregate_id_column='cluster_id',
            ... )
        """
        import pandas as pd
        import os
        import multiprocessing
        from concurrent import futures

        if verbose:
            print(f"\n{'='*60}")
            print(f"Fitting Nile Red Wavelengths from HDF5")
            print(f"{'='*60}")
            print(f"Input file: {h5_path}")
            print(f"Wavelength bounds: {wavelength_bounds[0]}-{wavelength_bounds[1]} nm")
            print(f"{'='*60}\n")

        # Load HDF5 file
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        if verbose:
            print("Loading HDF5 file...")

        df = pd.read_hdf(h5_path, "data")
        n_locs = len(df)

        if verbose:
            print(f"Loaded {n_locs} localizations")

        # Check required columns
        required_cols = [
            "A_R",
            "A_G",
            "A_B",
            "s_x",
            "s_y",
            "A_R_err",
            "A_G_err",
            "A_B_err",
            "s_x_err",
            "s_y_err",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"HDF5 file missing required columns: {missing_cols}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Setup optical system
        # Get wavelength array from camera_parameters or use default
        if "wavelength" in camera_parameters:
            wavelength_array = camera_parameters["wavelength"]
        else:
            # Extract from pixel_QYs shape - assumes standard wavelength grid
            if verbose:
                print(
                    "Warning: 'wavelength' not in camera_parameters, using default from getpixelefficiency()"
                )
            _, _, _, wavelength_array = self.spectral_funcs.getpixelefficiency()

        pixel_QYs = camera_parameters["pixel_QYs"]

        # Get filter spectra
        filter_spectra = self.spectral_funcs.get_dye_or_filter_data(
            names=filter_names, wavelength=wavelength_array, dye_or_filter=False
        )

        if verbose:
            print(f"Optical system configured with {len(filter_names)} filters")
            print(f"Wavelength array: {len(wavelength_array)} points")

        # Extract data from DataFrame
        R = df["A_R"].to_numpy()
        G = df["A_G"].to_numpy()
        B = df["A_B"].to_numpy()
        sigma_x = df["s_x"].to_numpy() * pixel_size  # Convert to nm
        sigma_y = df["s_y"].to_numpy() * pixel_size

        R_err = df["A_R_err"].to_numpy()
        G_err = df["A_G_err"].to_numpy()
        B_err = df["A_B_err"].to_numpy()
        sigma_x_err = df["s_x_err"].to_numpy() * pixel_size
        sigma_y_err = df["s_y_err"].to_numpy() * pixel_size

        # Extract photons and background for SNR calculation (if available)
        if "photons" in df.columns and "background_photons" in df.columns:
            fitted_photons = df["photons"].to_numpy()
            fitted_background_photons = df["background_photons"].to_numpy()
            use_snr = True
            if verbose:
                print("Using photon counts for SNR-based error inflation")
        else:
            if verbose:
                print(
                    "Warning: 'photons' or 'background_photons' columns not found, skipping SNR error inflation"
                )
            fitted_photons = None
            fitted_background_photons = None
            use_snr = False

        # --- Phase 1: Fit aggregate priors (if aggregate_id_column provided) ---
        aggregate_wl_map = {}  # {aggregate_id: fitted_wavelength}

        if aggregate_id_column is not None:
            if aggregate_id_column not in df.columns:
                raise ValueError(
                    f"aggregate_id_column '{aggregate_id_column}' not found in DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )

            if verbose:
                print(f"\n--- Phase 1: Fitting aggregate-level priors ---")
                print(f"Grouping by '{aggregate_id_column}'...")

            aggregate_ids = df[aggregate_id_column].unique()
            aggregate_ids = aggregate_ids[~np.isnan(aggregate_ids)]
            n_aggregates = len(aggregate_ids)

            if verbose:
                print(f"Found {n_aggregates} aggregates")

            # Compute weighted averages per aggregate
            agg_fit_args = []
            agg_ids_ordered = []

            for agg_id in aggregate_ids:
                subset = df[df[aggregate_id_column] == agg_id]

                # Weighted average: weight by 1/error (matching postprocess.py convention)
                agg_R = np.average(subset["A_R"], weights=1.0 / subset["A_R_err"])
                agg_G = np.average(subset["A_G"], weights=1.0 / subset["A_G_err"])
                agg_B = np.average(subset["A_B"], weights=1.0 / subset["A_B_err"])
                agg_sx = np.average(subset["s_x"], weights=1.0 / subset["s_x_err"]) * pixel_size
                agg_sy = np.average(subset["s_y"], weights=1.0 / subset["s_y_err"]) * pixel_size

                # Propagate errors: sigma_mean = 1 / sqrt(sum(1/sigma_i^2))
                agg_R_err = 1.0 / np.sqrt(np.sum(1.0 / subset["A_R_err"] ** 2))
                agg_G_err = 1.0 / np.sqrt(np.sum(1.0 / subset["A_G_err"] ** 2))
                agg_B_err = 1.0 / np.sqrt(np.sum(1.0 / subset["A_B_err"] ** 2))
                agg_sx_err = 1.0 / np.sqrt(np.sum(1.0 / subset["s_x_err"] ** 2)) * pixel_size
                agg_sy_err = 1.0 / np.sqrt(np.sum(1.0 / subset["s_y_err"] ** 2)) * pixel_size

                # Normalize RGB
                rgb_total = agg_R + agg_G + agg_B
                if rgb_total <= 0:
                    continue

                R_norm = agg_R / rgb_total
                G_norm = agg_G / rgb_total
                B_norm = agg_B / rgb_total

                total_err = np.sqrt(agg_R_err ** 2 + agg_G_err ** 2 + agg_B_err ** 2)
                R_norm_err = (
                    R_norm * np.sqrt((agg_R_err / agg_R) ** 2 + (total_err / rgb_total) ** 2)
                    if agg_R > 0 else 1e-3
                )
                G_norm_err = (
                    G_norm * np.sqrt((agg_G_err / agg_G) ** 2 + (total_err / rgb_total) ** 2)
                    if agg_G > 0 else 1e-3
                )
                B_norm_err = (
                    B_norm * np.sqrt((agg_B_err / agg_B) ** 2 + (total_err / rgb_total) ** 2)
                    if agg_B > 0 else 1e-3
                )

                # Sum photons across aggregate for SNR
                agg_photons = subset["photons"].sum() if use_snr else None
                agg_bg = subset["background_photons"].sum() if use_snr else None

                args = (
                    np.array([R_norm, G_norm, B_norm]),
                    agg_sx,
                    agg_sy,
                    np.array([R_norm_err, G_norm_err, B_norm_err]),
                    agg_sx_err,
                    agg_sy_err,
                    filter_spectra,
                    wavelength_array,
                    pixel_QYs,
                    NA,
                    agg_photons,
                    agg_bg,
                    wavelength_bounds,
                    None,  # wavelength_initial_guess: use default for aggregates
                )
                agg_fit_args.append(args)
                agg_ids_ordered.append(agg_id)

            # Fit aggregates in parallel
            if len(agg_fit_args) > 0:
                n_cpus = multiprocessing.cpu_count()
                n_workers = max(1, int(n_cpus * cpu_fraction))

                if verbose:
                    print(f"Fitting {len(agg_fit_args)} aggregates with {n_workers} workers...")
                    import time
                    agg_start_time = time.time()

                from Multicolour_Simulation_Functions import (
                    _fit_nile_red_wavelength_standalone,
                )

                with futures.ProcessPoolExecutor(n_workers) as executor:
                    future_list = [
                        executor.submit(_fit_nile_red_wavelength_standalone, *args)
                        for args in agg_fit_args
                    ]

                    agg_completed = 0
                    for idx, future in enumerate(future_list):
                        try:
                            wl, _ = future.result(timeout=30)
                            if not np.isnan(wl):
                                aggregate_wl_map[agg_ids_ordered[idx]] = wl
                            agg_completed += 1

                            if verbose and (agg_completed % 50 == 0 or agg_completed == len(agg_fit_args)):
                                elapsed = time.time() - agg_start_time
                                rate = agg_completed / elapsed if elapsed > 0 else 0
                                print(
                                    f"  Aggregate progress: {agg_completed}/{len(agg_fit_args)} "
                                    f"({100*agg_completed/len(agg_fit_args):.1f}%) - {rate:.1f} fits/s",
                                    end="\r",
                                    flush=True,
                                )
                        except Exception as e:
                            if verbose:
                                print(f"\nWarning: Aggregate fit failed for ID {agg_ids_ordered[idx]}: {e}")

                if verbose:
                    elapsed = time.time() - agg_start_time
                    n_agg_success = len(aggregate_wl_map)
                    print(f"\n  Phase 1 complete: {n_agg_success}/{len(agg_fit_args)} aggregates "
                          f"fitted in {elapsed:.1f} s")
                    if n_agg_success > 0:
                        agg_wls = np.array(list(aggregate_wl_map.values()))
                        print(f"  Aggregate wavelength range: {np.min(agg_wls):.1f} - {np.max(agg_wls):.1f} nm")
                        print(f"  Aggregate mean wavelength: {np.mean(agg_wls):.1f} nm")

            if verbose:
                print(f"\n--- Phase 2: Fitting individual localisations with aggregate priors ---")

        # Prepare arguments for parallel processing
        if verbose:
            print(f"\nPreparing {n_locs} fitting tasks...")

        fit_args = []
        valid_indices = []

        # Look up aggregate IDs for initial guesses
        if aggregate_id_column is not None:
            loc_agg_ids = df[aggregate_id_column].to_numpy()
        else:
            loc_agg_ids = None

        for j in range(n_locs):
            # Skip if RGB total is zero or negative
            rgb_total = R[j] + G[j] + B[j]
            if rgb_total <= 0:
                continue

            # Note: RGB values should already be normalized in the HDF5 file
            # but we reconstruct them here to ensure proper normalization
            R_norm = R[j] / rgb_total
            G_norm = G[j] / rgb_total
            B_norm = B[j] / rgb_total

            # Propagate errors (accounting for normalization)
            total_err = np.sqrt(R_err[j] ** 2 + G_err[j] ** 2 + B_err[j] ** 2)
            R_norm_err = (
                R_norm
                * np.sqrt((R_err[j] / R[j]) ** 2 + (total_err / rgb_total) ** 2)
                if R[j] > 0
                else 1e-3
            )
            G_norm_err = (
                G_norm
                * np.sqrt((G_err[j] / G[j]) ** 2 + (total_err / rgb_total) ** 2)
                if G[j] > 0
                else 1e-3
            )
            B_norm_err = (
                B_norm
                * np.sqrt((B_err[j] / B[j]) ** 2 + (total_err / rgb_total) ** 2)
                if B[j] > 0
                else 1e-3
            )

            # Look up aggregate prior initial guess
            wl_guess = None
            if loc_agg_ids is not None and not np.isnan(loc_agg_ids[j]):
                wl_guess = aggregate_wl_map.get(loc_agg_ids[j], None)

            # Build argument tuple for standalone function
            args = (
                np.array([R_norm, G_norm, B_norm]),
                sigma_x[j],
                sigma_y[j],
                np.array([R_norm_err, G_norm_err, B_norm_err]),
                sigma_x_err[j],
                sigma_y_err[j],
                filter_spectra,
                wavelength_array,
                pixel_QYs,
                NA,
                fitted_photons[j] if use_snr else None,
                fitted_background_photons[j] if use_snr else None,
                wavelength_bounds,
                wl_guess,
            )

            fit_args.append(args)
            valid_indices.append(j)

        if verbose:
            print(f"Valid fitting tasks: {len(fit_args)}/{n_locs}")
            if aggregate_id_column is not None:
                n_with_prior = sum(
                    1 for j in valid_indices
                    if not np.isnan(loc_agg_ids[j]) and loc_agg_ids[j] in aggregate_wl_map
                )
                print(f"Localisations with aggregate prior: {n_with_prior}/{len(fit_args)}")

        # Parallel wavelength fitting
        wl_fits = np.full(n_locs, np.nan)
        wl_fit_errs = np.full(n_locs, np.nan)

        if len(fit_args) > 0:
            # Calculate number of workers
            n_cpus = multiprocessing.cpu_count()
            n_workers = max(1, int(n_cpus * cpu_fraction))

            if verbose:
                print(f"\nStarting parallel fitting with {n_workers} workers...")
                import time

                start_time = time.time()

            # Import standalone function
            from Multicolour_Simulation_Functions import (
                _fit_nile_red_wavelength_standalone,
            )

            with futures.ProcessPoolExecutor(n_workers) as executor:
                # Submit all fitting tasks
                future_list = [
                    executor.submit(_fit_nile_red_wavelength_standalone, *args)
                    for args in fit_args
                ]

                # Collect results with progress tracking
                completed = 0
                for idx, future in enumerate(future_list):
                    try:
                        wl, wl_err = future.result(timeout=30)
                        wl_fits[valid_indices[idx]] = wl
                        wl_fit_errs[valid_indices[idx]] = wl_err
                        completed += 1

                        if verbose and (completed % 100 == 0 or completed == len(fit_args)):
                            elapsed = time.time() - start_time
                            rate = completed / elapsed if elapsed > 0 else 0
                            print(
                                f"  Progress: {completed}/{len(fit_args)} ({100*completed/len(fit_args):.1f}%) - {rate:.1f} fits/s",
                                end="\r",
                                flush=True,
                            )
                    except Exception as e:
                        if verbose:
                            print(
                                f"\nWarning: Fit failed for index {valid_indices[idx]}: {e}"
                            )
                        wl_fits[valid_indices[idx]] = np.nan
                        wl_fit_errs[valid_indices[idx]] = np.nan

            if verbose:
                elapsed = time.time() - start_time
                print(f"\n\nFitting complete: {elapsed:.1f} s ({len(fit_args)/elapsed:.1f} fits/s)")

        # Add wavelength columns to DataFrame
        df["wl_fit"] = wl_fits
        df["wl_fit_err"] = wl_fit_errs

        # Add per-aggregate fitted wavelength column (if using aggregate priors)
        if aggregate_id_column is not None and len(aggregate_wl_map) > 0:
            wl_agg = np.full(n_locs, np.nan)
            for j in range(n_locs):
                agg_id = loc_agg_ids[j]
                if not np.isnan(agg_id) and agg_id in aggregate_wl_map:
                    wl_agg[j] = aggregate_wl_map[agg_id]
            df["wl_fit_aggregate"] = wl_agg

        # Calculate success rate
        n_successful = np.sum(~np.isnan(wl_fits))
        success_rate = 100 * n_successful / n_locs

        if verbose:
            print(f"\nResults:")
            print(f"  Successful fits: {n_successful}/{n_locs} ({success_rate:.1f}%)")
            print(
                f"  Wavelength range: {np.nanmin(wl_fits):.1f} - {np.nanmax(wl_fits):.1f} nm"
            )
            print(f"  Mean wavelength: {np.nanmean(wl_fits):.1f} nm")
            print(f"  Std wavelength: {np.nanstd(wl_fits):.1f} nm")

        # Save to output file if requested
        if output_path is not None:
            if verbose:
                print(f"\nSaving results to: {output_path}")

            df.to_hdf(output_path, key="data", mode="w", format="table")

            if verbose:
                print("Save complete!")

        if verbose:
            print(f"\n{'='*60}\n")

        return df
