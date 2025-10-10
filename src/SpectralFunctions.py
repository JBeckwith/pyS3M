#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains functions for spectral analysis relating to the bayerSMLM concept.

This class provides functionality for:
- Spectral data analysis and fitting
- Camera quantum efficiency calculations
- Fluorophore and filter data management
- Database-driven spectral data retrieval

Created on Tue Dec 10 08:59:38 2024
@author: jbeckwith
jsb92, 2024/01/02
"""

from typing import Optional, List, Tuple, Union, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import os
import sys
import duckdb
import polars as pl
from scipy.optimize import OptimizeResult, differential_evolution
from scipy.constants import electron_volt, Planck, c
from scipy.special import erf

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import IOFunctions


class SpectralDataType(Enum):
    """Enumeration for spectral data types."""

    DYE = "dye"
    FILTER = "filter"


class SpectralConstants:
    """Constants for spectral analysis."""

    # File paths
    DEFAULT_CAMERA_QE_FILE = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "Spectra/Camera_QE/CS505CU_QE.csv"
    )
    DEFAULT_OBJECTIVE_FILE = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "Spectra/Objective_Absorption/Nikon_ApoTIRF_100x.csv",
    )

    # Physical constants
    FWHM_TO_SIGMA_FACTOR = 2 * np.sqrt(2 * np.log(2))

    # Gaussian normalization
    GAUSSIAN_NORM_FACTOR = 1 / np.sqrt(2 * np.pi)


class DatabaseQueryHandler:
    """Handles database queries for spectral data."""

    def __init__(self, db_path: str):
        """Initialize database query handler.

        Args:
            db_path: Path to the spectral database.
        """
        self.db_path = db_path

    def get_available_names(self, data_type: SpectralDataType) -> List[str]:
        """Get list of available dye or filter names from database.

        Args:
            data_type: Type of spectral data to query.

        Returns:
            List of available names.
        """
        table_map = {
            SpectralDataType.DYE: "dye_summary",
            SpectralDataType.FILTER: "filter_summary",
        }

        column_map = {
            SpectralDataType.DYE: "dye_name",
            SpectralDataType.FILTER: "filter_name",
        }

        with duckdb.connect(self.db_path, read_only=True) as conn:
            query = f"SELECT {column_map[data_type]} FROM {table_map[data_type]}"
            return list(conn.sql(query).df()[column_map[data_type]])

    def query_spectral_data(
        self, names: List[str], data_type: SpectralDataType
    ) -> pl.DataFrame:
        """Query spectral data from database.

        Args:
            names: List of names to query.
            data_type: Type of spectral data.

        Returns:
            DataFrame containing spectral data.
        """
        table_map = {SpectralDataType.DYE: "dyes", SpectralDataType.FILTER: "filters"}

        name_column_map = {
            SpectralDataType.DYE: "dye_name",
            SpectralDataType.FILTER: "filter_name",
        }

        with duckdb.connect(self.db_path, read_only=True) as conn:
            # Build safe IN clause with proper parameter binding
            if len(names) == 1:
                query = f"""
                    SELECT * FROM {table_map[data_type]} 
                    WHERE {name_column_map[data_type]} = ?
                    ORDER BY wavelength_nm
                """
                return conn.execute(query, names).df()
            else:
                # For multiple names, create placeholders
                placeholders = ",".join(["?" for _ in names])
                query = f"""
                    SELECT * FROM {table_map[data_type]} 
                    WHERE {name_column_map[data_type]} IN ({placeholders})
                    ORDER BY wavelength_nm
                """
                return conn.execute(query, names).df()


class SpectrumProcessor(ABC):
    """Abstract base class for spectrum processing strategies."""

    @abstractmethod
    def process_spectrum(
        self, spectrum_data: pl.DataFrame, wavelength: np.ndarray
    ) -> np.ndarray:
        """Process spectrum data.

        Args:
            spectrum_data: Raw spectrum data from database.
            wavelength: Target wavelength array.

        Returns:
            Processed spectrum data.
        """
        pass


class DyeSpectrumProcessor(SpectrumProcessor):
    """Processor for dye spectrum data."""

    def process_spectrum(
        self, spectrum_data: pl.DataFrame, wavelength: np.ndarray
    ) -> np.ndarray:
        """Process dye emission spectrum.

        Args:
            spectrum_data: Raw dye data with wavelength_nm and emission_intensity columns.
            wavelength: Target wavelength array for interpolation.

        Returns:
            Normalized emission spectrum interpolated to target wavelengths.
        """
        spectrum_wl = spectrum_data["wavelength_nm"].to_numpy()
        spectrum_fl = spectrum_data["emission_intensity"].to_numpy()

        # Remove negative values
        spectrum_fl = np.maximum(spectrum_fl, 0.0)

        # Interpolate to target wavelengths
        dye_rescaled = np.interp(
            x=wavelength,
            xp=spectrum_wl,
            fp=spectrum_fl,
            left=0,
            right=0,
        )

        # Normalize by total intensity
        total_intensity = np.nansum(dye_rescaled)
        return dye_rescaled / total_intensity if total_intensity > 0 else dye_rescaled


class FilterSpectrumProcessor(SpectrumProcessor):
    """Processor for filter spectrum data."""

    def process_spectrum(
        self, spectrum_data: pl.DataFrame, wavelength: np.ndarray
    ) -> np.ndarray:
        """Process filter transmission spectrum.

        Args:
            spectrum_data: Raw filter data with wavelength_nm and transmission_pct columns.
            wavelength: Target wavelength array for interpolation.

        Returns:
            Transmission spectrum interpolated to target wavelengths.
        """
        spectrum_wl = spectrum_data["wavelength_nm"].to_numpy()
        spectrum_tm = spectrum_data["transmission_pct"].to_numpy()

        # Remove negative values
        spectrum_tm = np.maximum(spectrum_tm, 0.0)

        # Interpolate to target wavelengths
        return np.interp(
            x=wavelength,
            xp=spectrum_wl,
            fp=spectrum_tm,
            left=0,
            right=0,
        )


class Spectral_Funcs:
    """A class for spectral analysis and fluorophore calculations.

    This class provides comprehensive functionality for:
    - Camera quantum efficiency analysis
    - Spectral fitting with Gaussian and skewed Gaussian models
    - Database-driven fluorophore and filter data management
    - Pixel efficiency calculations for Bayer filter cameras

    The class uses a strategy pattern for handling different types of spectral data
    (dyes vs filters) and provides optimised database query handling.
    """

    def __init__(self):
        """Initialize the Spectral_Funcs class.

        Sets up database connection and loads available dye and filter names.
        """
        # Set up database path
        spectra_folder = os.path.join(os.path.split(module_dir)[0], "Spectra")
        db_path = os.path.join(spectra_folder, "spectral_data.duckdb")

        # Initialize database handler
        self.db_handler = DatabaseQueryHandler(db_path)

        # Load available names
        self.dye_names = self.db_handler.get_available_names(SpectralDataType.DYE)
        self.filter_names = self.db_handler.get_available_names(SpectralDataType.FILTER)

        # Initialize spectrum processors
        self.processors = {
            SpectralDataType.DYE: DyeSpectrumProcessor(),
            SpectralDataType.FILTER: FilterSpectrumProcessor(),
        }

        # Initialize IO functions
        self.io = IOFunctions.IO_Functions()

    @staticmethod
    def getpixelefficiency(
        filename: str = SpectralConstants.DEFAULT_CAMERA_QE_FILE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get pixel quantum efficiency for a camera from CSV file.

        Args:
            filename: Path to the CSV file containing camera QE data.
                     Expected columns: wavelength, R, G, B.

        Returns:
            Tuple containing:
                - R: Red pixel quantum efficiency array
                - G: Green pixel quantum efficiency array
                - B: Blue pixel quantum efficiency array
                - wavelength: Wavelength array (nm)

        Raises:
            FileNotFoundError: If the QE file cannot be found.
            ValueError: If the CSV file has incorrect format.
        """
        try:
            data = pl.read_csv(filename)
        except Exception as e:
            raise FileNotFoundError(f"Could not read camera QE file {filename}: {e}")

        required_columns = ["wavelength", "R", "G", "B"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in QE file: {missing_columns}")

        # Extract data
        wavelength_coarse = data["wavelength"].to_numpy()
        R_coarse = data["R"].to_numpy()
        G_coarse = data["G"].to_numpy()
        B_coarse = data["B"].to_numpy()

        # Create fine wavelength grid
        wavelength = np.arange(np.min(wavelength_coarse), np.max(wavelength_coarse))

        # Interpolate to fine grid
        R = np.interp(x=wavelength, xp=wavelength_coarse, fp=R_coarse)
        G = np.interp(x=wavelength, xp=wavelength_coarse, fp=G_coarse)
        B = np.interp(x=wavelength, xp=wavelength_coarse, fp=B_coarse)

        return R, G, B, wavelength

    @staticmethod
    def getobjectiveefficiency(
        wavelength: np.ndarray,
        filename: str = SpectralConstants.DEFAULT_OBJECTIVE_FILE,
    ) -> np.ndarray:
        """Get objective transmission efficiency from CSV file.

        Args:
            wavelength: Wavelength array to interpolate transmission values at.
            filename: Path to CSV file containing objective transmission data.
                     Expected columns: wavelength, transmission.

        Returns:
            Transmission efficiency array at specified wavelengths.

        Raises:
            FileNotFoundError: If the transmission file cannot be found.
            ValueError: If the CSV file has incorrect format.
        """
        try:
            data = pl.read_csv(filename)
        except Exception as e:
            raise FileNotFoundError(f"Could not read objective file {filename}: {e}")

        required_columns = ["wavelength", "transmission"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in transmission file: {missing_columns}"
            )

        wavelength_coarse = data["wavelength"].to_numpy()
        transmission_coarse = np.array(data["transmission"].to_numpy(), dtype=float)

        return np.interp(wavelength, wavelength_coarse, transmission_coarse)

    def fwhm_sigma_conversion(self, x: float, sigma_given: bool = True) -> float:
        """Convert between Full Width at Half Maximum (FWHM) and sigma.

        Args:
            x: Parameter value to convert.
            sigma_given: If True, converts sigma to FWHM. If False, converts FWHM to sigma.

        Returns:
            Converted parameter value.
        """
        if sigma_given:
            return SpectralConstants.FWHM_TO_SIGMA_FACTOR * x
        else:
            return x / SpectralConstants.FWHM_TO_SIGMA_FACTOR

    def moment_calculations(
        self, x: np.ndarray, fx: np.ndarray, order: int = 3
    ) -> np.ndarray:
        """Calculate statistical moments of a spectrum.

        Implements moment calculations as described in:
        Bultmann, T. & Ernsting, N. P. J. Phys. Chem. 100, 19417–19424 (1996).

        Args:
            x: Wavelength or energy array.
            fx: Spectrum intensity array.
            order: Number of moments to calculate (1-4).

        Returns:
            Array of calculated moments [m0, m1, m2, m3][:order].
            - m0: Zeroth moment (total intensity)
            - m1: First moment (mean position)
            - m2: Second moment (standard deviation)
            - m3: Third moment (skewness)
        """
        # Zeroth moment (total area)
        m0 = np.trapz(x=x, y=fx)

        if order >= 1:
            # First moment (mean)
            m1 = np.trapz(x=x, y=fx * x) / m0
        else:
            return np.array([m0])

        if order >= 2:
            # Second moment (standard deviation)
            m2 = np.sqrt(np.trapz(y=(x - m1) ** 2 * fx, x=x) / m0)
        else:
            return np.array([m0, m1])

        if order >= 3:
            # Third moment (skewness)
            m3 = np.power(np.trapz(y=(x - m1) ** 3 * fx, x=x) / m0, 1.0 / 3)
        else:
            return np.array([m0, m1, m2])

        moments = np.array([m0, m1, m2, m3])
        return moments[:order]

    def spectral_initial_guess(
        self, spectrum: np.ndarray, wavelength: np.ndarray, model_length: int = 3
    ) -> np.ndarray:
        """Generate initial parameter guess for spectral fitting.

        Args:
            spectrum: Spectral intensity data.
            wavelength: Wavelength array.
            model_length: Number of parameters for initial guess (3 for Gaussian, 4 for skewed).

        Returns:
            Initial parameter guess array with NaN values replaced by zeros.
        """
        # Convert to energy domain for better fitting properties
        energy = self.wavelength_to_energy(wavelength)

        # Apply weighting factor for dipole moment representation
        weighting_factor = energy ** (-3) * wavelength**2
        spectrum_weighted = spectrum * weighting_factor

        # Sort by energy for proper integration
        sort_indices = np.argsort(energy)
        energy_sorted = energy[sort_indices]
        spectrum_sorted = spectrum_weighted[sort_indices]

        # Calculate moments for initial guess
        initial_guess = self.moment_calculations(
            energy_sorted, spectrum_sorted, model_length
        )

        return np.nan_to_num(initial_guess)

    def wavelength_to_energy(self, wavelength: np.ndarray) -> np.ndarray:
        """Convert wavelength to photon energy.

        Args:
            wavelength: Wavelength array in nanometers.

        Returns:
            Energy array in electron volts (eV).
        """
        # Convert nm to m, then calculate energy in eV
        wavelength_m = wavelength * 1e-9
        energy_j = Planck * c / wavelength_m
        energy_ev = energy_j / electron_volt

        return energy_ev

    def gaussian_model(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calculate Gaussian model for spectral fitting.

        Args:
            params: Parameters [amplitude, mean, sigma].
            x: Independent variable array (energy).

        Returns:
            Gaussian function values at x positions.
        """
        amplitude, mu, sigma = params[:3]

        # Avoid division by zero
        if sigma <= 0:
            return np.zeros_like(x)

        # Gaussian function
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        normalization = SpectralConstants.GAUSSIAN_NORM_FACTOR / sigma

        return amplitude * normalization * np.exp(exponent)

    def skew_gaussian_model(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Calculate skewed Gaussian model for spectral fitting.

        Implements equations 16-18 from:
        Beckwith, J. S., Rumble, C. A. & Vauthey, E. Int. Rev. Phys. Chem. 39, 135–216 (2020).

        Args:
            params: Parameters [amplitude, mean, sigma, skewness].
            x: Independent variable array (energy).

        Returns:
            Skewed Gaussian function values at x positions.
        """
        amplitude, mu, sigma, alpha = params[:4]

        # Calculate base Gaussian
        gaussian = self.gaussian_model(params, x)

        # Calculate skewness factor
        if sigma <= 0:
            return np.zeros_like(x)

        skew_arg = alpha * (x - mu) / sigma
        skew_factor = 1 + erf(skew_arg)

        return gaussian * skew_factor

    def chi2_spectrum(
        self,
        params: np.ndarray,
        wavelength: np.ndarray,
        spectrum: np.ndarray,
        model: str = "gaussian",
        weights: Optional[np.ndarray] = None,
        return_fit: bool = False,
    ) -> Union[np.ndarray, np.ndarray]:
        """Calculate chi-squared residuals for spectral fitting.

        Implements dipole moment representation weighting as described in:
        Angulo, G., Grampp, G. & Rosspeintner, A. Spectrochim. Acta. A. Mol. Biomol.
        Spectrosc. 65, 727–731 (2006).

        Args:
            params: Fitting parameters.
            wavelength: Wavelength array.
            spectrum: Experimental spectrum data.
            model: Model type ("gaussian" or "skew-gaussian").
            weights: Optional weighting array for fitting.
            return_fit: If True, return fitted spectrum instead of residuals.

        Returns:
            Chi-squared residuals array or fitted spectrum if return_fit=True.

        Raises:
            ValueError: If model type is not supported.
        """
        # Convert to energy domain
        energy = self.wavelength_to_energy(wavelength)

        # Calculate model spectrum in energy domain
        if model == "gaussian":
            spectrum_model_energy = self.gaussian_model(params, energy)
        elif model == "skew-gaussian":
            spectrum_model_energy = self.skew_gaussian_model(params, energy)
        else:
            raise ValueError(f"Unsupported model type: {model}")

        # Convert model from energy to wavelength domain
        # Jacobian factor: dλ/dE = hc/E^2 (in appropriate units) = λ^2/E
        # Dipole moment weighting: E^(-3)
        # Combined: I(λ) = I(E) / (E^(-3) * λ^2)
        weighting_factor = energy ** (-3) * wavelength**2
        spectrum_model_wavelength = spectrum_model_energy / weighting_factor

        if return_fit:
            return spectrum_model_wavelength

        # Calculate residuals in wavelength domain (comparing like-to-like)
        residuals = spectrum - spectrum_model_wavelength

        # Apply optional weighting
        if weights is not None:
            residuals = np.sqrt(weights * residuals**2)

        return residuals.ravel()

    def spectral_fit_dye(
        self,
        spectrum: np.ndarray,
        wavelength: np.ndarray,
        model: str = "gaussian",
        weights: Optional[np.ndarray] = None,
        display: bool = False,
    ) -> Union[OptimizeResult, Tuple[OptimizeResult, np.ndarray]]:
        """Fit spectral data with Gaussian or skewed Gaussian model.

        Uses differential evolution (genetic algorithm) for robust global optimization.
        No initial guess required - the algorithm automatically finds the optimal parameters.

        Args:
            spectrum: Experimental spectrum data.
            wavelength: Wavelength array.
            model: Model type ("gaussian" or "skew-gaussian").
            weights: Optional weighting array for fitting.
            display: If True, return both fit result and fitted spectrum.

        Returns:
            Optimization result, or tuple of (result, fitted_spectrum) if display=True.

        Raises:
            ValueError: If model type is not supported.
        """
        # Normalize spectrum to max=1 for better numerical stability
        spectrum_max = np.max(spectrum)
        if spectrum_max > 0:
            spectrum_norm = spectrum / spectrum_max
        else:
            spectrum_norm = spectrum

        # Define objective function for differential evolution (sum of squared residuals)
        def objective(params):
            residuals = self.chi2_spectrum(
                params, wavelength, spectrum_norm, model, weights
            )
            return np.nansum(residuals**2)

        # Set parameter bounds based on physical constraints
        if model == "gaussian":
            # Bounds: [amplitude, center_energy, sigma]
            # Amplitude: 0 to 2 (normalized spectrum has max=1)
            # Center: energy range of wavelengths
            # Sigma: narrow to broad peaks
            energy = self.wavelength_to_energy(wavelength)
            bounds = [
                (0, 1e6),  # amplitude (normalized)
                (np.min(energy) * 0.9, np.max(energy) * 1.1),  # center (energy)
                (0.01, 2.0),  # sigma (energy spread)
            ]
        elif model == "skew-gaussian":
            # Bounds: [amplitude, center_energy, sigma, alpha]
            # Alpha: skewness parameter (-10 to 10 is reasonable)
            energy = self.wavelength_to_energy(wavelength)
            bounds = [
                (0, 1e6),  # amplitude (normalized)
                (np.min(energy) * 0.9, np.max(energy) * 1.1),  # center (energy)
                (0.01, 2.0),  # sigma
                (-10, 10),  # alpha (skewness)
            ]
        else:
            raise ValueError(f"Unsupported model type: {model}")

        # Use differential evolution - robust, no initial guess needed
        # Increase sensitivity with tighter tolerance and more iterations
        result = differential_evolution(
            objective,
            bounds,
            strategy="best1bin",
            maxiter=2000,  # More iterations for better convergence
            popsize=20,  # Larger population for better exploration
            tol=1e-9,  # Tighter tolerance
            mutation=(0.5, 1.5),  # Wider mutation range
            recombination=0.7,
            atol=1e-10,  # Absolute tolerance
            polish=True,  # Refine with L-BFGS-B after genetic algorithm
            workers=1,
        )

        # Scale amplitude back to original spectrum scale
        result.x[0] *= spectrum_max

        if not display:
            return result

        # Generate fitted spectrum for display (using original spectrum scale)
        fitted_spectrum = self.chi2_spectrum(
            result.x,
            wavelength,
            spectrum,
            model=model,
            weights=weights,
            return_fit=True,
        )

        return result, fitted_spectrum

    def get_pixel_fractions_rawspectra(
        self, spectra: np.ndarray, wavelength: np.ndarray, pixel_QYs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate pixel efficiencies for raw spectral data.

        Args:
            spectra: Raw spectrum array (n_spectra x n_wavelengths).
            wavelength: Wavelength array.
            pixel_QYs: Pixel quantum yield array (n_wavelengths x n_pixels).

        Returns:
            Tuple containing:
                - Average emission wavelengths for each spectrum
                - Pixel efficiencies for each spectrum and pixel type
        """
        # handle single spectrum case
        if spectra.ndim == 1:
            spectra = spectra[np.newaxis, :]
        # Normalize spectra
        spectra_normalised = spectra.T / np.trapz(x=wavelength, y=spectra, axis=1)
        spectra_normalised = spectra_normalised.T

        # Calculate average emission wavelengths
        weighted_wavelengths = wavelength * spectra_normalised
        average_wavelengths = np.trapz(y=weighted_wavelengths.T, x=wavelength, axis=0)

        # Calculate pixel efficiencies
        pixel_efficiencies = np.dot(spectra, pixel_QYs.T)

        return np.squeeze(average_wavelengths), np.squeeze(pixel_efficiencies)

    def get_pixel_fractions_dye_and_filters(
        self,
        dyes: List[str],
        filters: Optional[List[str]],
        wavelength: np.ndarray,
        pixel_QYs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate pixel efficiencies for dyes with optional filters.

        Args:
            dyes: List of dye names to analyze.
            filters: List of filter names to apply (None for no filtering).
            wavelength: Wavelength array.
            pixel_QYs: Pixel quantum yield array (n_wavelengths x n_pixels).

        Returns:
            Tuple containing:
                - Average emission wavelengths for each dye
                - Pixel efficiencies for each dye and pixel type
        """
        # Get filter transmission (unity if no filters)
        if filters is None:
            filter_transmission = np.ones_like(wavelength)
        else:
            filter_spectra = self.get_spectral_data(
                filters, wavelength, SpectralDataType.FILTER
            )
            filter_transmission = np.prod(filter_spectra, axis=0)

        # Get dye emission spectra
        dye_spectra = self.get_spectral_data(dyes, wavelength, SpectralDataType.DYE)

        # Apply filter transmission
        dye_filtered_spectra = dye_spectra * filter_transmission

        # Normalize by total intensity for each dye
        total_intensities = np.sum(dye_filtered_spectra, axis=1, keepdims=True)
        total_intensities = np.where(
            total_intensities > 0, total_intensities, 1
        )  # Avoid division by zero
        dye_normalized_spectra = dye_filtered_spectra / total_intensities

        # Calculate average emission wavelengths
        weighted_wavelengths = wavelength * dye_normalized_spectra
        average_wavelengths = np.trapz(y=weighted_wavelengths.T, x=wavelength, axis=0)

        # Calculate pixel efficiencies
        pixel_efficiencies = np.dot(dye_normalized_spectra, pixel_QYs.T)

        return np.squeeze(average_wavelengths), np.squeeze(pixel_efficiencies)

    def get_spectral_data(
        self,
        names: Union[str, List[str]],
        wavelength: np.ndarray,
        data_type: SpectralDataType,
    ) -> np.ndarray:
        """Get spectral data for specified dyes or filters.

        Args:
            names: Name(s) of dyes or filters to retrieve.
            wavelength: Wavelength array for interpolation.
            data_type: Type of spectral data (DYE or FILTER).

        Returns:
            Spectral data array (n_items x n_wavelengths).

        Raises:
            ValueError: If requested names are not found in database.
        """
        # Ensure names is a list
        if isinstance(names, str):
            names = [names]

        # Validate names exist in database
        available_names = (
            self.dye_names if data_type == SpectralDataType.DYE else self.filter_names
        )

        invalid_names = [name for name in names if name not in available_names]
        if invalid_names:
            data_type_str = data_type.value
            raise ValueError(
                f"{data_type_str.capitalize()} names not in database: {invalid_names}"
            )

        # Query database for each name
        spectra = np.zeros((len(names), len(wavelength)))
        processor = self.processors[data_type]

        for i, name in enumerate(names):
            try:
                # Query single item to reduce memory usage
                spectrum_data = self.db_handler.query_spectral_data([name], data_type)

                if not spectrum_data.empty:
                    # Process spectrum using appropriate strategy
                    spectra[i, :] = processor.process_spectrum(
                        spectrum_data, wavelength
                    )

            except Exception as e:
                # Log warning but continue processing other items
                print(
                    f"Warning: Failed to process {data_type.value} '{name}': {e}",
                    flush=True,
                )
                continue

        return spectra

    # Backward compatibility methods
    def get_dye_or_filter_data(
        self,
        names: Union[str, List[str]],
        wavelength: np.ndarray,
        dye_or_filter: bool = True,
    ) -> np.ndarray:
        """Get dye or filter data (legacy interface for backward compatibility).

        Args:
            names: Name(s) of dyes or filters to retrieve.
            wavelength: Wavelength array for interpolation.
            dye_or_filter: If True, looks up dyes. If False, looks up filters.

        Returns:
            Spectral data array.
        """
        data_type = SpectralDataType.DYE if dye_or_filter else SpectralDataType.FILTER
        return self.get_spectral_data(names, wavelength, data_type)
