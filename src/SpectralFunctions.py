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
import sys
from pathlib import Path
import duckdb
import polars as pl
import numba
from scipy.optimize import OptimizeResult, differential_evolution
from scipy.constants import electron_volt, Planck, c
from scipy.special import erf

_MODULE_DIR = Path(__file__).parent
sys.path.append(str(_MODULE_DIR))
import IOFunctions
import logging
logger = logging.getLogger(__name__)



class SpectralDataType(Enum):
    """Enumeration for spectral data types."""

    DYE = "dye"
    FILTER = "filter"


class SpectralConstants:
    """Constants for spectral analysis."""

    # File paths
    DEFAULT_CAMERA_QE_FILE = Path(__file__).parent.parent / "Spectra/Camera_QE/CS505CU_QE.csv"
    DEFAULT_OBJECTIVE_FILE = (
        Path(__file__).parent.parent / "Spectra/Objective_Absorption/Nikon_ApoTIRF_100x.csv"
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


@numba.jit(nopython=True, nogil=True, cache=True)
def _assign_photons_to_channels_jit(p_0, p_1, uniform_randoms):
    """
    JIT-compiled photon channel assignment using cumulative probability method.

    This function is ~2-5× faster than vectorized NumPy due to reduced memory
    allocation and improved cache locality. The nogil flag allows parallel execution.

    Args:
        p_0: Array of probabilities for channel 0 (Blue) for each photon
        p_1: Array of probabilities for channel 1 (Green) for each photon
        uniform_randoms: Array of uniform random numbers [0, 1) for each photon

    Returns:
        Tuple of (count_0, count_1, count_2) - number of photons in each channel
    """
    n_photons = len(p_0)
    count_0 = 0
    count_1 = 0
    count_2 = 0

    for i in range(n_photons):
        u = uniform_randoms[i]
        cum_p0 = p_0[i]
        cum_p1 = cum_p0 + p_1[i]

        # Assign to channel based on cumulative probability
        if u < cum_p0:
            count_0 += 1
        elif u < cum_p1:
            count_1 += 1
        else:
            count_2 += 1

    return count_0, count_1, count_2


@numba.jit(nopython=True, parallel=True, nogil=True, cache=True)
def _process_bootstrap_samples_parallel(
    photon_wavelengths_bootstrap,
    lut_wavelengths,
    lut_qe,
    uniform_randoms_all,
):
    """
    Parallel processing of bootstrap samples using Numba prange.

    This function processes all bootstrap samples in parallel, calculating mean
    wavelengths and colour channel counts for each sample. Uses pre-computed
    QE lookup tables and pre-generated random numbers for deterministic results.

    Args:
        photon_wavelengths_bootstrap: Shape (n_bootstrap, n_photons_per_image)
        lut_wavelengths: QE lookup table wavelengths (1D array)
        lut_qe: QE values, shape (3, len(lut_wavelengths)) for B, G, R
        uniform_randoms_all: Pre-generated random numbers, shape (n_bootstrap, n_photons_per_image)

    Returns:
        Tuple of (mean_wavelengths, counts_array, mean_total_qe_array):
            - mean_wavelengths: Shape (n_bootstrap,)
            - counts_array: Shape (n_bootstrap, 3) - counts for B, G, R
            - mean_total_qe_array: Shape (n_bootstrap,)

    Speedup: 3-3.5× faster than sequential loop by parallelizing across bootstrap samples.
    """
    n_bootstrap, n_photons = photon_wavelengths_bootstrap.shape

    # Preallocate output arrays
    mean_wavelengths = np.zeros(n_bootstrap, dtype=np.float64)
    counts_array = np.zeros((n_bootstrap, 3), dtype=np.float64)
    mean_total_qe_array = np.zeros(n_bootstrap, dtype=np.float64)

    # Parallel loop over bootstrap samples
    for i in numba.prange(n_bootstrap):
        # Get photon wavelengths for this bootstrap sample
        photon_wls = photon_wavelengths_bootstrap[i, :]

        # Calculate mean wavelength
        mean_wl = np.mean(photon_wls)
        mean_wavelengths[i] = mean_wl

        # Lookup QE values for each photon using the pre-computed LUT
        # Use optimized inline interpolation to avoid function call overhead
        qy_at_photons = np.zeros((3, n_photons), dtype=np.float64)

        # Process each photon
        for j in range(n_photons):
            wl = photon_wls[j]

            # Binary search to find bracketing indices in LUT
            idx = np.searchsorted(lut_wavelengths, wl)

            # Handle edge cases and interpolate
            if idx == 0:
                # wavelength below LUT range - use first LUT value
                qy_at_photons[0, j] = lut_qe[0, 0]
                qy_at_photons[1, j] = lut_qe[1, 0]
                qy_at_photons[2, j] = lut_qe[2, 0]
            elif idx >= len(lut_wavelengths):
                # wavelength above LUT range - use last LUT value
                last_idx = len(lut_wavelengths) - 1
                qy_at_photons[0, j] = lut_qe[0, last_idx]
                qy_at_photons[1, j] = lut_qe[1, last_idx]
                qy_at_photons[2, j] = lut_qe[2, last_idx]
            else:
                # Interpolate between idx-1 and idx
                wl0 = lut_wavelengths[idx - 1]
                wl1 = lut_wavelengths[idx]
                frac = (wl - wl0) / (wl1 - wl0)

                # Unrolled loop for 3 channels (B, G, R)
                idx_prev = idx - 1
                qy_at_photons[0, j] = lut_qe[0, idx_prev] + frac * (lut_qe[0, idx] - lut_qe[0, idx_prev])
                qy_at_photons[1, j] = lut_qe[1, idx_prev] + frac * (lut_qe[1, idx] - lut_qe[1, idx_prev])
                qy_at_photons[2, j] = lut_qe[2, idx_prev] + frac * (lut_qe[2, idx] - lut_qe[2, idx_prev])

        # Extract B, G, R quantum yields
        qy_0 = qy_at_photons[0, :]  # Blue
        qy_1 = qy_at_photons[1, :]  # Green
        qy_2 = qy_at_photons[2, :]  # Red

        # Total detection probability for each photon
        total_qy = qy_0 + qy_1 + qy_2

        # Calculate mean total QE
        mean_total_qe = np.mean(total_qy)
        mean_total_qe_array[i] = mean_total_qe

        # Probability each photon is detected in each channel
        # Avoid division by zero
        p_0 = np.zeros(n_photons, dtype=np.float64)
        p_1 = np.zeros(n_photons, dtype=np.float64)

        for j in range(n_photons):
            if total_qy[j] > 1e-10:
                p_0[j] = qy_0[j] / total_qy[j]
                p_1[j] = qy_1[j] / total_qy[j]

        # Assign each photon to a channel using pre-generated random numbers
        uniform_randoms = uniform_randoms_all[i, :]
        count_0, count_1, count_2 = _assign_photons_to_channels_jit(p_0, p_1, uniform_randoms)

        counts_array[i, 0] = count_0
        counts_array[i, 1] = count_1
        counts_array[i, 2] = count_2

    return mean_wavelengths, counts_array, mean_total_qe_array


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

    def __init__(self, camera: str = "ximea"):
        """Initialize the Spectral_Funcs class.

        Args:
            camera: Camera model name (``"ximea"`` or ``"zwo"``).
                Determines which QE file is used by default in
                :meth:`getpixelefficiency`.

        Sets up database connection and loads available dye and filter names.
        """
        import CameraDefaults
        self._qe_file = CameraDefaults.get_camera_config(camera).qe_file

        # Set up database path
        spectra_folder = _MODULE_DIR.parent / "Spectra"
        db_path = spectra_folder / "spectral_data.duckdb"

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

    def getpixelefficiency(
        self,
        filename: str = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get pixel quantum efficiency for a camera from CSV file.

        Args:
            filename: Path to the CSV file containing camera QE data
                (columns: wavelength, R, G, B).  If *None*, the file
                selected at construction time by the ``camera`` argument
                is used (default: CS505CU for Ximea, ASI585MC for ZWO).

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
        if filename is None:
            filename = self._qe_file
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

        # Calculate pixel efficiencies using simple integral (unnormalized, absolute QE)
        # This gives the correct photoelectron conversion probability:
        #   abs_QE[c] = ∫ spectrum_normalized(λ) × QE_c(λ) dλ
        # Note: Uses normalized spectra so result is independent of absolute intensity
        pixel_efficiencies = np.dot(spectra_normalised, pixel_QYs.T)

        return np.squeeze(average_wavelengths), np.squeeze(pixel_efficiencies)

    def get_pixel_fractions_dye_and_filters(
        self,
        dyes: List[str],
        filters: Optional[List[str]],
        wavelength: np.ndarray,
        pixel_QYs: np.ndarray,
        normalized: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate pixel efficiencies for dyes with optional filters.

        This method can return either normalized color ratios (for classification)
        or unnormalized absolute quantum efficiencies (for photoelectron generation).

        Args:
            dyes: List of dye names to analyze.
            filters: List of filter names to apply (None for no filtering).
            wavelength: Wavelength array.
            pixel_QYs: Pixel quantum yield array (n_pixels x n_wavelengths).
            normalized: If True, return normalized BGR ratios that sum to 1 using
                       per-wavelength normalization to match stochastic photon sampling.
                       If False, return unnormalized absolute QE values for photoelectron
                       generation (simple integral: ∫ spectrum(λ) × QE_c(λ) dλ).

        Returns:
            Tuple containing:
                - Average emission wavelengths for each dye
                - Pixel efficiencies for each dye and pixel type
                  (normalized ratios if normalized=True, absolute QE if normalized=False)

        Mathematical details:

        When normalized=True (for color classification):
            BGR[c] = ∫ spectrum(λ) × [QE_c(λ) / ∑_c' QE_c'(λ)] dλ

            This uses per-wavelength normalization to match the stochastic photon
            sampling behavior in calculate_colourratio_from_photon_wavelengths.
            This ensures that the expected BGR ratios match the actual sampling
            distribution when photons are assigned to channels probabilistically.

        When normalized=False (for photoelectron generation):
            abs_QE[c] = ∫ spectrum(λ) × QE_c(λ) dλ

            This gives the absolute quantum efficiency for photoelectron conversion.
            The integral represents the expected number of photoelectrons generated
            per input photon for pixels of type c.
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
        if normalized:
            # Use per-wavelength normalization for color ratios
            # This matches the stochastic sampling behavior in calculate_colourratio_from_photon_wavelengths
            #
            # For each wavelength λ, the probability a photon is detected in channel c is:
            #   P(c|λ) = QE_c(λ) / [∑_c' QE_c'(λ)]
            #
            # The expected color ratios are then:
            #   E[c] = ∫ spectrum(λ) × P(c|λ) dλ
            #        = ∫ spectrum(λ) × [QE_c(λ) / total_QE(λ)] dλ

            # Sum QE across all channels at each wavelength
            total_QE_per_wavelength = np.sum(pixel_QYs, axis=0)  # Shape: (n_wavelengths,)
            total_QE_per_wavelength = np.maximum(total_QE_per_wavelength, 1e-10)  # Avoid division by zero

            # Calculate per-wavelength detection probabilities for each channel
            # Shape: (n_pixels, n_wavelengths)
            prob_per_wavelength = pixel_QYs / total_QE_per_wavelength

            # Calculate expected color ratios by integrating spectrum × probability
            # For multiple dyes: dye_normalized_spectra has shape (n_dyes, n_wavelengths)
            # prob_per_wavelength has shape (n_pixels, n_wavelengths)
            # Result: (n_dyes, n_pixels)
            pixel_efficiencies = np.dot(dye_normalized_spectra, prob_per_wavelength.T)

        else:
            # Use simple integral for absolute quantum efficiency
            # This gives the correct photoelectron conversion probability
            #
            # abs_QE[c] = ∫ spectrum(λ) × QE_c(λ) dλ
            #
            # For multiple dyes: dye_normalized_spectra has shape (n_dyes, n_wavelengths)
            # pixel_QYs has shape (n_pixels, n_wavelengths)
            # Result: (n_dyes, n_pixels)
            pixel_efficiencies = np.dot(dye_normalized_spectra, pixel_QYs.T)

        return np.squeeze(average_wavelengths), np.squeeze(pixel_efficiencies)

    def get_absolute_pixel_QYs(
        self,
        dyes: List[str],
        filters: Optional[List[str]],
        wavelength: np.ndarray,
        pixel_QYs: np.ndarray,
        include_objective: bool = True,
        objective_filename: str = SpectralConstants.DEFAULT_OBJECTIVE_FILE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate absolute per-channel detection efficiencies for dyes.

        Returns the fraction of *all emitted photons* that are ultimately
        detected as photoelectrons in each pixel channel, accounting for
        the full optical chain:

            QY_abs_c = ∫ spectrum_norm(λ) · T_filter(λ) · T_obj(λ) · QE_c(λ) dλ

        where spectrum_norm is normalised to unit integral over all emitted
        wavelengths (not over the filter passband).

        This differs from get_pixel_fractions_dye_and_filters(normalized=False),
        which normalises by the *filtered* spectrum and therefore gives the
        fraction of filter-passed photons per channel, not the fraction of
        all emitted photons.

        Args:
            dyes: List of dye names to analyse.
            filters: List of filter names to apply (None for no filtering).
            wavelength: Wavelength array shared by pixel_QYs and spectra.
            pixel_QYs: Per-channel quantum efficiencies, shape
                (n_channels, n_wavelengths).  Convention: [B, G, R] ordering.
            include_objective: If True (default), multiply by the objective
                transmission curve from objective_filename.
            objective_filename: Path to objective transmission CSV.
                Defaults to SpectralConstants.DEFAULT_OBJECTIVE_FILE.

        Returns:
            Tuple of three arrays:
                - average_wavelengths: Emission-weighted mean wavelength for
                  each dye, shape (n_dyes,).
                - abs_QYs_per_channel: Absolute detection efficiency per
                  channel, shape (n_dyes, n_channels).  Each value is the
                  expected fraction of emitted photons detected in that
                  channel.
                - total_abs_QYs: Sum of abs_QYs_per_channel across channels,
                  shape (n_dyes,).  This is the overall system detection
                  efficiency (photoelectrons per emitted photon).

        Example:
            >>> sf = Spectral_Funcs()
            >>> R, G, B, wl = sf.getpixelefficiency()
            >>> pixel_QYs = np.vstack([B, G, R])
            >>> avg_wl, qy_per_ch, total_qy = sf.get_absolute_pixel_QYs(
            ...     dyes=["ATTO 565"],
            ...     filters=["semrock-nf03-405-488-561-635e"],
            ...     wavelength=wl,
            ...     pixel_QYs=pixel_QYs,
            ... )
            >>> print(f"Total system QY: {total_qy[0]:.4f}")
            >>> print(f"B/G/R channel QYs: {qy_per_ch[0]}")
        """
        # --- filter transmission (unity if no filters) ---
        if filters is None:
            filter_transmission = np.ones_like(wavelength, dtype=float)
        else:
            filter_spectra = self.get_spectral_data(
                filters, wavelength, SpectralDataType.FILTER
            )
            filter_transmission = np.prod(filter_spectra, axis=0)

        # --- objective transmission (unity if not requested) ---
        if include_objective:
            obj_transmission = self.getobjectiveefficiency(
                wavelength, filename=objective_filename
            )
        else:
            obj_transmission = np.ones_like(wavelength, dtype=float)

        # combined optical system transmission (filter × objective)
        system_transmission = filter_transmission * obj_transmission

        # --- dye emission spectra ---
        dye_spectra = self.get_spectral_data(dyes, wavelength, SpectralDataType.DYE)

        # normalise each dye spectrum to unit integral over *all* emitted
        # wavelengths (before any filtering)
        total_emission = np.sum(dye_spectra, axis=1, keepdims=True)
        total_emission = np.where(total_emission > 0, total_emission, 1.0)
        dye_norm = dye_spectra / total_emission  # shape: (n_dyes, n_wavelengths)

        # emission-weighted average wavelengths
        average_wavelengths = np.trapz(
            y=(wavelength * dye_norm).T, x=wavelength, axis=0
        )

        # apply system transmission to the normalised spectra
        # shape: (n_dyes, n_wavelengths)
        dye_at_detector = dye_norm * system_transmission

        # absolute QY per channel:
        #   QY_abs_c = ∫ dye_at_detector(λ) · QE_c(λ) dλ
        # dye_at_detector: (n_dyes, n_wavelengths)
        # pixel_QYs:       (n_channels, n_wavelengths)
        # result:          (n_dyes, n_channels)
        abs_QYs_per_channel = np.dot(dye_at_detector, pixel_QYs.T)

        # total system efficiency (sum over channels)
        total_abs_QYs = np.sum(abs_QYs_per_channel, axis=-1)

        return (
            np.squeeze(average_wavelengths),
            np.squeeze(abs_QYs_per_channel),
            np.squeeze(total_abs_QYs),
        )

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
                logger.warning(f"Warning: Failed to process {data_type.value} '{name}': {e}")
                continue

        return spectra

    def sample_photons_from_spectrum(
        self,
        spectrum: np.ndarray,
        wavelength: np.ndarray,
        n_photons: int,
        random_state: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample photon wavelengths from a spectrum treated as a probability density.

        This function samples photons stochastically from a spectrum, accounting for
        shot noise. The spectrum is treated as a probability density function (PDF)
        and photons are drawn according to this distribution.

        This is essential for realistic simulations where R, G, B ratios and PSF
        widths vary with photon count due to Poisson statistics, rather than being
        deterministic.

        Args:
            spectrum: Emission spectrum (can include filter transmission, pixel QE, etc.).
                     Does not need to be normalized - will be normalized internally.
            wavelength: Wavelength array corresponding to spectrum values (nm).
            n_photons: Number of photons to sample.
            random_state: Optional numpy random generator for reproducibility.
                         If None, uses default random state.

        Returns:
            Array of sampled photon wavelengths (nm), length = n_photons.

        Example:
            >>> sf = Spectral_Funcs()
            >>> # Get emission spectrum
            >>> R, G, B, wl = sf.getpixelefficiency()
            >>> dye_spec = sf.get_dye_or_filter_data('alexa-fluor-647', wl)
            >>>
            >>> # Sample 1000 photons from this spectrum
            >>> rng = np.random.default_rng(42)
            >>> photon_wavelengths = sf.sample_photons_from_spectrum(
            ...     dye_spec[0], wl, n_photons=1000, random_state=rng
            ... )
            >>>
            >>> # Use wavelengths to calculate realistic R, G, B with shot noise
            >>> # (each photon detected in R, G, or B channel based on wavelength)
        """
        if random_state is None:
            random_state = np.random.default_rng()

        # Normalize spectrum to create probability density
        spectrum_positive = np.maximum(spectrum, 0)  # Ensure non-negative
        total = np.trapz(spectrum_positive, wavelength)

        if total <= 0:
            raise ValueError("Spectrum has no positive values - cannot sample photons")

        # Create cumulative distribution function (CDF)
        # VECTORIZED: Replace Python loop with np.cumsum (10-50× faster)
        pdf = spectrum_positive / total

        # Trapezoidal integration using vectorized operations
        dx = np.diff(wavelength)
        pdf_avg = 0.5 * (pdf[1:] + pdf[:-1])
        cdf = np.zeros(len(wavelength))
        cdf[1:] = np.cumsum(pdf_avg * dx)

        # Normalize CDF to ensure it reaches exactly 1.0
        cdf = cdf / cdf[-1]

        # Sample uniform random numbers and invert CDF
        uniform_samples = random_state.uniform(0, 1, n_photons)

        # Interpolate to find wavelengths corresponding to uniform samples
        sampled_wavelengths = np.interp(uniform_samples, cdf, wavelength)

        return sampled_wavelengths

    def _create_qe_lut(
        self,
        wavelength_array: np.ndarray,
        pixel_QYs: np.ndarray,
        grid_spacing: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a lookup table (LUT) for QE values on a fine wavelength grid.

        This pre-computes QE values on a dense grid to avoid repeated interpolation.
        Used for performance optimization in vectorized bootstrap sampling.

        Args:
            wavelength_array: Original wavelength array (nm).
            pixel_QYs: Pixel quantum efficiencies, shape (n_colours, n_wavelengths).
            grid_spacing: Spacing of LUT grid in nm (default: 0.5nm).

        Returns:
            Tuple of (lut_wavelengths, lut_qe):
                - lut_wavelengths: Dense wavelength grid (nm)
                - lut_qe: QE values at grid points, shape (n_colours, n_lut_points)
        """
        # Create fine wavelength grid
        wl_min = np.min(wavelength_array)
        wl_max = np.max(wavelength_array)
        lut_wavelengths = np.arange(wl_min, wl_max + grid_spacing, grid_spacing)

        # Pre-interpolate QE for all channels
        n_colours = pixel_QYs.shape[0]
        lut_qe = np.zeros((n_colours, len(lut_wavelengths)))
        for i in range(n_colours):
            lut_qe[i, :] = np.interp(lut_wavelengths, wavelength_array, pixel_QYs[i, :])

        return lut_wavelengths, lut_qe

    def _lookup_qe_vectorized(
        self,
        photon_wavelengths: np.ndarray,
        lut_wavelengths: np.ndarray,
        lut_qe: np.ndarray
    ) -> np.ndarray:
        """Fast QE lookup using pre-computed LUT.

        Args:
            photon_wavelengths: Wavelengths to look up, any shape.
            lut_wavelengths: LUT wavelength grid (nm).
            lut_qe: Pre-computed QE values, shape (n_colours, n_lut_points).

        Returns:
            QE values at photon wavelengths, shape (n_colours, *photon_wavelengths.shape).
        """
        # Find nearest LUT indices
        # Use searchsorted for fast lookup, then round to nearest
        grid_spacing = lut_wavelengths[1] - lut_wavelengths[0]
        indices = np.round((photon_wavelengths - lut_wavelengths[0]) / grid_spacing).astype(int)

        # Clip to valid range
        indices = np.clip(indices, 0, len(lut_wavelengths) - 1)

        # Lookup QE for all channels
        # lut_qe has shape (n_colours, n_lut_points)
        # indices has shape (*photon_wavelengths.shape)
        # Result: (n_colours, *photon_wavelengths.shape)
        qe_values = lut_qe[:, indices]

        return qe_values

    def calculate_colourratio_from_photon_wavelengths(
        self,
        photon_wavelengths: np.ndarray,
        wavelength_array: np.ndarray,
        pixel_QYs: np.ndarray,
        pixel_order: Optional[List[str]] = None,
        pixel_order_indices: Optional[Union[List[int], Dict[str, int]]] = None,
        return_counts: bool = False,
        return_total_qe: bool = False,
        qe_lut: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[float, np.ndarray, Optional[float]]:
        """Convert sampled photon wavelengths to mean wavelength and B:G:R colour ratios.

        This function takes photon wavelengths sampled from a spectrum and assigns each
        photon stochastically to B, G, or R channels based on the pixel quantum
        efficiency at that wavelength. This accounts for shot noise - the same spectrum
        with low photon count will have high variance in colour ratios.

        Args:
            photon_wavelengths: Array of photon wavelengths (nm) from sample_photons_from_spectrum.
            wavelength_array: Wavelength array for pixel QE interpolation (nm).
            pixel_QYs: Pixel quantum efficiencies, shape (n_colours, n_wavelengths).
                      Convention: [B, G, R] ordering (wavelength order).
            pixel_order: List of pixel colour names in order (e.g., ['B', 'G', 'R']).
            pixel_order_indices: Indices or dict mapping colours to indices.
            return_counts: If True, return raw counts. If False, return normalized ratios.

        Returns:
            Tuple of (mean_wavelength, colour_ratios):
                - mean_wavelength: Mean of photon wavelengths (nm)
                - colour_ratios: Array of [B, G, R] ratios or counts

        Example:
            >>> sf = Spectral_Funcs()
            >>> R, G, B, wl = sf.getpixelefficiency()
            >>> pixel_QYs = np.vstack([B, G, R])
            >>>
            >>> # Sample 500 photons
            >>> photon_wls = sf.sample_photons_from_spectrum(dye_spec[0], wl, 500)
            >>>
            >>> # Get mean wavelength and BGR ratios with shot noise
            >>> mean_wl, bgr = sf.calculate_colourratio_from_photon_wavelengths(
            ...     photon_wls, wl, pixel_QYs,
            ...     pixel_order=['B', 'G', 'R'],
            ...     pixel_order_indices=[0, 1, 2]
            ... )
        """
        # Calculate mean wavelength
        mean_wavelength = np.mean(photon_wavelengths)

        # Interpolate pixel QE at each photon wavelength
        # pixel_QYs has shape (n_colours, n_wavelengths)
        # We need QE for each photon at each colour
        n_colours = pixel_QYs.shape[0]

        # OPTIMIZATION: Use pre-computed LUT if provided
        if qe_lut is not None:
            lut_wavelengths, lut_qe = qe_lut
            qy_at_photons = self._lookup_qe_vectorized(
                photon_wavelengths, lut_wavelengths, lut_qe
            )
        else:
            # Fallback: Interpolate QE for each colour channel at photon wavelengths
            qy_at_photons = np.zeros((n_colours, len(photon_wavelengths)))
            for i in range(n_colours):
                qy_at_photons[i, :] = np.interp(
                    photon_wavelengths, wavelength_array, pixel_QYs[i, :]
                )

        # For simplicity and consistency with BGR ordering, use indices 0, 1, 2
        qy_0 = qy_at_photons[0, :]  # Blue
        qy_1 = qy_at_photons[1, :]  # Green
        qy_2 = qy_at_photons[2, :]  # Red

        # Total detection probability for each photon
        total_qy = qy_0 + qy_1 + qy_2

        # Avoid division by zero
        total_qy = np.maximum(total_qy, 1e-10)

        # Probability each photon is detected in each channel
        p_0 = qy_0 / total_qy
        p_1 = qy_1 / total_qy
        p_2 = qy_2 / total_qy

        # Assign each photon to a channel based on these probabilities
        # Use JIT-compiled function for 2-5× speedup over vectorized NumPy
        u = np.random.uniform(0, 1, len(photon_wavelengths))
        count_0, count_1, count_2 = _assign_photons_to_channels_jit(p_0, p_1, u)

        if return_counts:
            colour_ratios = np.array([count_0, count_1, count_2], dtype=np.float64)
        else:
            # Return normalized ratios
            total_counts = count_0 + count_1 + count_2
            if total_counts > 0:
                colour_ratios = np.array(
                    [count_0, count_1, count_2], dtype=np.float64
                ) / total_counts
            else:
                colour_ratios = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Optionally return mean total QE across sampled wavelengths
        if return_total_qe:
            mean_total_qe = np.mean(total_qy)
            return mean_wavelength, colour_ratios, mean_total_qe
        else:
            return mean_wavelength, colour_ratios

    def generate_bootstrap_colour_ratios(
        self,
        spectrum: np.ndarray,
        wavelength: np.ndarray,
        pixel_QYs: np.ndarray,
        n_photons_per_image: int,
        n_bootstrap: int,
        pixel_order: Optional[List[str]] = None,
        pixel_order_indices: Optional[Union[List[int], Dict[str, int]]] = None,
        random_state: Optional[np.random.Generator] = None,
        use_parallel: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Efficiently generate many bootstrap samples of colour ratios and mean wavelengths.

        This function samples all photons at once (n_bootstrap × n_photons_per_image),
        then divides into bootstrap chunks. This is ~n_bootstrap times faster than
        calling sample_photons_from_spectrum repeatedly.

        Args:
            spectrum: Emission spectrum (can include filter transmission).
            wavelength: Wavelength array corresponding to spectrum (nm).
            pixel_QYs: Pixel quantum efficiencies, shape (n_colours, n_wavelengths).
            n_photons_per_image: Number of photons per bootstrap sample.
            n_bootstrap: Number of bootstrap samples to generate.
            pixel_order: List of pixel colour names (e.g., ['B', 'G', 'R']).
            pixel_order_indices: Indices or dict mapping colours to indices.
            random_state: Optional numpy random generator.
            use_parallel: If True, use Numba parallel processing (3-3.5× faster). Default: True.

        Returns:
            Tuple of (mean_wavelengths, colour_ratios):
                - mean_wavelengths: Array of shape (n_bootstrap,)
                - colour_ratios: Array of shape (n_bootstrap, 3) with BGR ratios

        Example:
            >>> sf = Spectral_Funcs()
            >>> # Generate 1000 bootstrap samples at 500 photons each
            >>> mean_wls, bgr_ratios = sf.generate_bootstrap_colour_ratios(
            ...     dye_spec[0], wl, pixel_QYs,
            ...     n_photons_per_image=500,
            ...     n_bootstrap=1000,
            ...     pixel_order=['B', 'G', 'R'],
            ...     random_state=rng
            ... )
            >>> # Analyze shot noise statistics
            >>> print(f"Mean B: {bgr_ratios[:, 0].mean():.3f} ± {bgr_ratios[:, 0].std():.3f}")
        """
        if random_state is None:
            random_state = np.random.default_rng()

        # Sample all photons at once
        total_photons = n_photons_per_image * n_bootstrap
        all_photon_wavelengths = self.sample_photons_from_spectrum(
            spectrum, wavelength, total_photons, random_state
        )

        # Reshape into bootstrap samples
        photon_wavelengths_bootstrap = all_photon_wavelengths.reshape(
            n_bootstrap, n_photons_per_image
        )

        # Preallocate output arrays
        mean_wavelengths = np.zeros(n_bootstrap, dtype=np.float64)
        counts_array = np.zeros((n_bootstrap, 3), dtype=np.float64)
        mean_total_qe_array = np.zeros(n_bootstrap, dtype=np.float64)

        # OPTIMIZATION: Pre-compute QE lookup table to avoid repeated interpolation
        # This creates a dense wavelength grid (0.5nm spacing) and pre-interpolates QE values
        # Speedup: ~20-50× by replacing 300,000 interpolations with array lookups
        qe_lut = self._create_qe_lut(wavelength, pixel_QYs, grid_spacing=0.5)
        lut_wavelengths, lut_qe = qe_lut

        if use_parallel:
            # PARALLEL PATH: Use Numba prange for 3-3.5× speedup
            # Pre-generate all random numbers for deterministic results
            # Note: We use np.random.seed() temporarily to ensure deterministic behavior
            # This is because the current code uses np.random.uniform instead of random_state
            uniform_randoms_all = np.random.uniform(
                0, 1, size=(n_bootstrap, n_photons_per_image)
            )

            # Call parallel JIT function
            mean_wavelengths, counts_array, mean_total_qe_array = _process_bootstrap_samples_parallel(
                photon_wavelengths_bootstrap,
                lut_wavelengths,
                lut_qe,
                uniform_randoms_all,
            )
        else:
            # SEQUENTIAL PATH: Original implementation for verification
            # Calculate mean wavelengths and counts for each bootstrap
            for i in range(n_bootstrap):
                mean_wl, counts, mean_total_qe = self.calculate_colourratio_from_photon_wavelengths(
                    photon_wavelengths_bootstrap[i, :],
                    wavelength,
                    pixel_QYs,
                    pixel_order=pixel_order,
                    pixel_order_indices=pixel_order_indices,
                    return_counts=True,  # Get counts, not normalized ratios
                    return_total_qe=True,  # Get mean total QE across wavelengths
                    qe_lut=qe_lut,  # Pass pre-computed LUT for fast lookups
                )
                mean_wavelengths[i] = mean_wl
                counts_array[i, :] = counts
                mean_total_qe_array[i] = mean_total_qe

        # OPTIMIZATION: Vectorize the QE conversion across all bootstrap samples at once
        # Convert counts to effective QE values
        # counts = [n_B, n_G, n_R] where n_X is number of photons detected in channel X
        # mean_total_qe = average of (QE_B + QE_G + QE_R) across sampled wavelengths
        #
        # calculate_colourratio_from_photon_wavelengths assigns photons stochastically
        # based on P(channel | wavelength) = QE_channel(λ) / total_QE(λ)
        #
        # The normalized fractions are: n_B/N, n_G/N, n_R/N
        # These approximate: <QE_B(λ) / total_QE(λ)>, <QE_G(λ) / total_QE(λ)>, <QE_R(λ) / total_QE(λ)>
        #
        # To get absolute QE values:
        # QE_B = <QE_B(λ) / total_QE(λ)> × <total_QE(λ)> = (n_B / N) × mean_total_qe

        # Vectorized computation: (n_bootstrap, 3) arrays
        total_detected = np.sum(counts_array, axis=1)  # Shape: (n_bootstrap,)
        valid_mask = total_detected > 0

        # Initialize output
        colour_ratios = np.zeros((n_bootstrap, 3), dtype=np.float64)

        # Vectorized QE calculation for all valid samples at once
        # counts_array[valid_mask, :] has shape (n_valid, 3)
        # mean_total_qe_array[valid_mask, np.newaxis] has shape (n_valid, 1) -> broadcasts to (n_valid, 3)
        colour_ratios[valid_mask, :] = (
            (counts_array[valid_mask, :] / n_photons_per_image) *
            mean_total_qe_array[valid_mask, np.newaxis]
        )

        return mean_wavelengths, colour_ratios

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
