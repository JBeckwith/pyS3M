import os
import sys
import numpy as np
from copy import deepcopy
import time
from scipy.spatial.distance import cdist
import gc
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)

import IOFunctions
import PSFFunctions
import sCMOSFunctions
import ImageAnalysisFunctions
import SpectralFunctions
from ImageAnalysisFunctions import FittingStrategy as IAF_FittingStrategy


class FittingStrategy(Enum):
    """
    Enumeration of available fitting strategies for multicolour SMLM analysis.

    Each strategy represents a different approach to fitting Bayer-filtered camera data:
    - STANDARD: Direct fitting with Bayer pattern masks
    - DEMOSAIC: Full demosaic then fit colour channels
    - DEMOSAIC_FAST: Fast demosaic with optimised fitting
    - DEMOSAIC_IG: Initial grayscale fit then colour refinement
    """

    STANDARD = "standard"
    DEMOSAIC = "demosaic"
    DEMOSAIC_FAST = "demosaic_fast"
    DEMOSAIC_IG = "demosaic_ig"


@dataclass
class CameraParameters:
    """
    Validated camera parameters dataclass with required calibration data.

    Attributes:
        gain (np.ndarray): Pixel-wise gain map for sCMOS camera
        offset (np.ndarray): Pixel-wise offset map for sCMOS camera
        variance (np.ndarray): Pixel-wise variance map for sCMOS camera
        readnoise (float): Camera read noise level
        rqe (np.ndarray): Relative quantum efficiency map
        masks (Dict[str, np.ndarray]): Bayer filter masks by colour channel
        pixel_QYs (np.ndarray): Quantum yields vs wavelength for each pixel type
        pixel_order (List[str]): Order of colour channels (e.g. ['B', 'G', 'R'])
        pixel_order_indices (Dict[str, int]): Mapping from colour to channel index
    """

    gain: np.ndarray
    offset: np.ndarray
    variance: np.ndarray
    readnoise: float
    rqe: np.ndarray
    masks: Dict[str, np.ndarray]
    pixel_QYs: np.ndarray
    pixel_order: List[str]
    pixel_order_indices: Dict[str, int]

    @classmethod
    def validate_and_create(
        cls, camera_parameters: Dict[str, Any]
    ) -> "CameraParameters":
        """
        Validate camera parameters dictionary and create dataclass instance.

        Args:
            camera_parameters (Dict[str, Any]): Dictionary containing camera calibration data
                                               with required keys for sCMOS camera operation

        Returns:
            CameraParameters: Validated dataclass instance with camera parameters

        Raises:
            ValueError: If required camera parameter keys are missing
        """
        required_params = [
            "gain",
            "offset",
            "variance",
            "readnoise",
            "rqe",
            "masks",
            "pixel_QYs",
            "pixel_order",
            "pixel_order_indices",
        ]

        missing_params = [
            param for param in required_params if param not in camera_parameters
        ]
        if missing_params:
            raise ValueError(
                f"Camera parameters missing required keys: {missing_params}"
            )

        return cls(**{param: camera_parameters[param] for param in required_params})


@dataclass
class SimulationConfig:
    """
    Configuration dataclass for simulation parameters and options.

    Attributes:
        n_bootstrap (int): Number of bootstrap simulations to run (default: 100000)
        background_photons (float): Background photons per pixel (default: 40.0)
        background_colour (List[float]): RGB background colour weights (default: [1,1,1])
        NA (float): Numerical aperture of objective lens (default: 1.49)
        pixel_size (float): Camera pixel size in nanometers (default: 69)
        cpu_fraction (float): Fraction of CPU cores to use for parallel processing (default: 0.9)
        save_raw_results (bool): Whether to save raw fitting results (default: False)
        subtractx0y0 (bool): Whether to subtract ground truth positions from results (default: False)
        saverawimages (bool): Whether to save raw Bayer images (default: False)
        use_lut (bool): Use LUT for fast Nile Red wavelength fitting (default: True)
        verbose (bool): Whether to print detailed progress messages (default: True)
    """

    n_bootstrap: int = 100000
    background_photons: float = 40.0
    background_colour: List[float] = None
    NA: float = 1.49
    pixel_size: float = 69
    cpu_fraction: float = 0.9
    save_raw_results: bool = False
    subtractx0y0: bool = False
    saverawimages: bool = False
    use_lut: bool = True
    use_stochastic_photons: bool = True
    verbose: bool = True

    def __post_init__(self):
        """
        Post-initialization to set default background colour if not specified.
        """
        if self.background_colour is None:
            self.background_colour = [1, 1, 1]


class SimulationValidationError(Exception):
    """
    Custom exception raised when simulation input validation fails.

    Used to indicate problems with input parameters such as mismatched array dimensions,
    missing required keys in camera parameters, or invalid wavelength specifications.
    """

    pass


class FittingResultProcessor:
    """
    Handles processing and analysis of fitting results from multicolour localization.

    Provides static methods for averaging and normalising fitting results across colour channels,
    particularly for demosaicking-based fitting strategies where results need to be consolidated
    from individual RGB channel fits.
    """

    @staticmethod
    def colour_fit_averager(
        fit_results: pd.DataFrame, n_bootstrap: int
    ) -> pd.DataFrame:
        """
        Average amplitude and background fits from demosaicking across RGB colour channels.

        Args:
            fit_results (pd.DataFrame): Raw fitting results with 'A', 'b', 'chi_sqr' columns
                                       containing data for 3x n_bootstrap fits (one per RGB channel)
            n_bootstrap (int): Number of bootstrap simulations run

        Returns:
            pd.DataFrame: Averaged results with normalised A_B/G/R and bg_B/G/R columns
        """
        b_toextract = fit_results["b"].to_numpy()
        A_toextract = fit_results["A"].to_numpy()
        chi_toextract = fit_results["chi_sqr"].to_numpy()

        # Initialize arrays
        data_arrays = {
            "A_B": np.zeros(n_bootstrap),
            "A_G": np.zeros(n_bootstrap),
            "A_R": np.zeros(n_bootstrap),
            "bg_B": np.zeros(n_bootstrap),
            "bg_G": np.zeros(n_bootstrap),
            "bg_R": np.zeros(n_bootstrap),
            "chi_sqr": np.zeros(n_bootstrap),
        }

        indices = np.arange(0, n_bootstrap * 3, 3)
        for i, index in enumerate(indices[:-1]):
            data_arrays["chi_sqr"][i] = np.nanmean(
                chi_toextract[index : indices[i + 1]]
            )
            A = np.nansum(A_toextract[index : indices[i + 1]])
            b = np.nansum(b_toextract[index : indices[i + 1]])

            # Avoid division by zero
            if b != 0:
                data_arrays["bg_B"][i] = b_toextract[index] / b
                data_arrays["bg_G"][i] = b_toextract[index + 1] / b
                data_arrays["bg_R"][i] = b_toextract[index + 2] / b

            if A != 0:
                data_arrays["A_B"][i] = A_toextract[index] / A
                data_arrays["A_G"][i] = A_toextract[index + 1] / A
                data_arrays["A_R"][i] = A_toextract[index + 2] / A

        data_arrays["frame"] = np.arange(n_bootstrap)
        return pd.DataFrame(data_arrays)

    @staticmethod
    def fit_averager(fit_results: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
        """
        Average fits from demosaicking including positional and shape parameters.

        Args:
            fit_results (pd.DataFrame): Raw fitting results containing position, shape,
                                       amplitude and background data for RGB channels
            n_bootstrap (int): Number of bootstrap simulations run

        Returns:
            pd.DataFrame: Averaged results with mean positions/shapes and normalised colours
        """
        # Extract arrays
        arrays_to_extract = ["xc", "yc", "s_x", "s_y", "b", "A", "chi_sqr"]
        extracted_arrays = {
            param: fit_results[param].to_numpy() for param in arrays_to_extract
        }

        # Validate that we have enough data
        expected_total_length = n_bootstrap * 3
        actual_length = len(extracted_arrays["xc"]) if "xc" in extracted_arrays else 0
        if actual_length < expected_total_length:
            import logging

            logging.warning(
                f"fit_averager: Expected {expected_total_length} data points but got {actual_length}"
            )
            if actual_length == 0:
                logging.warning(
                    "fit_averager: No valid fitting results - all data is NaN or empty"
                )
                # Return DataFrame with NaN values
                return pd.DataFrame(
                    {
                        "xc": np.full(n_bootstrap, np.nan),
                        "yc": np.full(n_bootstrap, np.nan),
                        "s_x": np.full(n_bootstrap, np.nan),
                        "s_y": np.full(n_bootstrap, np.nan),
                        "A_B": np.full(n_bootstrap, np.nan),
                        "A_G": np.full(n_bootstrap, np.nan),
                        "A_R": np.full(n_bootstrap, np.nan),
                        "bg_B": np.full(n_bootstrap, np.nan),
                        "bg_G": np.full(n_bootstrap, np.nan),
                        "bg_R": np.full(n_bootstrap, np.nan),
                        "chi_sqr": np.full(n_bootstrap, np.nan),
                        "frame": np.arange(n_bootstrap),
                    }
                )

        # Initialize result arrays
        result_data = {
            "xc": np.zeros(n_bootstrap),
            "yc": np.zeros(n_bootstrap),
            "s_x": np.zeros(n_bootstrap),
            "s_y": np.zeros(n_bootstrap),
            "A_B": np.zeros(n_bootstrap),
            "A_G": np.zeros(n_bootstrap),
            "A_R": np.zeros(n_bootstrap),
            "bg_B": np.zeros(n_bootstrap),
            "bg_G": np.zeros(n_bootstrap),
            "bg_R": np.zeros(n_bootstrap),
            "chi_sqr": np.zeros(n_bootstrap),
        }

        indices = np.arange(0, n_bootstrap * 3, 3)
        for i, index in enumerate(indices[:-1]):
            # Average positional and shape parameters
            for param in ["xc", "yc", "s_x", "s_y", "chi_sqr"]:
                slice_data = extracted_arrays[param][index : indices[i + 1]]
                if len(slice_data) > 0:
                    result_data[param][i] = np.nanmean(slice_data)
                else:
                    result_data[param][i] = np.nan

            # Handle amplitude and background
            A_slice = extracted_arrays["A"][index : indices[i + 1]]
            b_slice = extracted_arrays["b"][index : indices[i + 1]]

            if len(A_slice) > 0:
                A = np.nansum(A_slice)
                b = np.nansum(b_slice)
            else:
                A = np.nan
                b = np.nan

            if not np.isnan(b) and b != 0 and len(b_slice) >= 3:
                result_data["bg_B"][i] = extracted_arrays["b"][index] / b
                result_data["bg_G"][i] = extracted_arrays["b"][index + 1] / b
                result_data["bg_R"][i] = extracted_arrays["b"][index + 2] / b
            else:
                result_data["bg_B"][i] = np.nan
                result_data["bg_G"][i] = np.nan
                result_data["bg_R"][i] = np.nan

            if not np.isnan(A) and A != 0 and len(A_slice) >= 3:
                result_data["A_B"][i] = extracted_arrays["A"][index] / A
                result_data["A_G"][i] = extracted_arrays["A"][index + 1] / A
                result_data["A_R"][i] = extracted_arrays["A"][index + 2] / A
            else:
                result_data["A_B"][i] = np.nan
                result_data["A_G"][i] = np.nan
                result_data["A_R"][i] = np.nan

        result_data["frame"] = np.arange(n_bootstrap, dtype=float)
        return pd.DataFrame(result_data)


class MultiC_Sim_Funcs_Refactored:
    """
    Refactored multicolour simulation functions with consolidated duplicate code.

    This class provides the core functionality for simulating and analyzing multicolour
    single-molecule localization microscopy (SMLM) data using Bayer-filtered cameras.
    It consolidates multiple fitting strategies into a unified interface while maintaining
    backward compatibility with the original implementation.
    """

    def __init__(
        self,
        mosaic_unit=None,
        io_functions=None,
        psf_functions=None,
        scmos_functions=None,
        image_analysis_functions=None,
        spectral_functions=None,
    ):
        """
        Initialize the simulation functions with dependency injection support.

        Args:
            mosaic_unit: Optional parameter for mosaic configuration (currently unused)
            io_functions: IO functions instance (default: creates new instance)
            psf_functions: PSF functions instance (default: creates new instance)
            scmos_functions: sCMOS functions instance (default: creates new instance)
            image_analysis_functions: Image analysis functions instance (default: creates new instance)
            spectral_functions: Spectral functions instance (default: creates new instance)
        """
        self.mosaic_unit = mosaic_unit
        self.result_processor = FittingResultProcessor()

        # Dependency injection with sensible defaults
        self.io = (
            io_functions if io_functions is not None else IOFunctions.IO_Functions()
        )
        self.psf = (
            psf_functions if psf_functions is not None else PSFFunctions.PSF_Functions()
        )
        self.scmos = (
            scmos_functions
            if scmos_functions is not None
            else sCMOSFunctions.sCMOS_Functions()
        )
        self.image_analysis = (
            image_analysis_functions
            if image_analysis_functions is not None
            else ImageAnalysisFunctions.Image_Analysis_Functions()
        )
        self.spectral = (
            spectral_functions
            if spectral_functions is not None
            else SpectralFunctions.Spectral_Funcs()
        )

    def _validate_inputs(
        self,
        wavelength: np.ndarray,
        camera_parameters: Dict[str, Any],
        dye_pixel_efficiency: Optional[np.ndarray],
        x0y0: Dict[str, np.ndarray],
    ) -> None:
        """
        Validate common input parameters for simulation consistency.

        Args:
            wavelength (np.ndarray): Wavelength array for spectral calculations
            camera_parameters (Dict[str, Any]): Camera calibration parameters dictionary
            dye_pixel_efficiency (Optional[np.ndarray]): Pixel efficiency values for dye molecules (can be None during early validation)
            x0y0 (Dict[str, np.ndarray]): Dictionary of molecule positions by dye type

        Raises:
            SimulationValidationError: If input parameters are inconsistent or invalid
        """
        try:
            if len(wavelength) != camera_parameters["pixel_QYs"].shape[1]:
                raise SimulationValidationError(
                    "pixel_QYs not defined at all wavelengths."
                )

            # Only validate dye_pixel_efficiency if it's provided (not None)
            if dye_pixel_efficiency is not None:
                if len(dye_pixel_efficiency.shape) > 1:
                    # Check if this is stochastic mode (per-frame colour ratios)
                    # In stochastic mode, shape is (n_bootstrap, 3) for single dye
                    # In deterministic mode, shape is (n_dyes, 3) for multi-dye
                    # We can distinguish by checking if shape[0] matches number of bootstrap samples
                    # from any x0y0 entry (all have same number of frames)
                    n_frames_in_x0y0 = list(x0y0.values())[0].shape[0]

                    # If first dimension matches n_frames, this is stochastic single-dye mode
                    if dye_pixel_efficiency.shape[0] == n_frames_in_x0y0:
                        # Stochastic mode: should have exactly 1 dye
                        if len(x0y0.keys()) != 1:
                            raise SimulationValidationError(
                                "x0y0 dictionary does not contain correct number of localisation arrays "
                                "(stochastic mode requires single dye)."
                            )
                    else:
                        # Deterministic multi-dye mode: first dimension is number of dyes
                        if len(x0y0.keys()) != dye_pixel_efficiency.shape[0]:
                            raise SimulationValidationError(
                                "x0y0 dictionary does not contain correct number of localisation arrays."
                            )
                else:
                    # 1D array: deterministic single-dye mode
                    if len(x0y0.keys()) != 1:
                        raise SimulationValidationError(
                            "x0y0 dictionary does not contain correct number of localisation arrays."
                        )
        except (KeyError, AttributeError, IndexError) as e:
            raise SimulationValidationError(f"Input validation failed: {e}")

    def _setup_simulation_parameters(
        self,
        camera_params: CameraParameters,
        config: SimulationConfig,
        dye_pixel_efficiency: np.ndarray,
        average_emission_wavelength: float,
        dye: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Set up common simulation parameters including positions and expected values.

        Args:
            camera_params (CameraParameters): Validated camera calibration parameters
            config (SimulationConfig): Simulation configuration settings
            dye_pixel_efficiency (np.ndarray): Pixel detection efficiency for the dye
            average_emission_wavelength (float): Average emission wavelength for PSF calculation
            dye (str): Dye name for identification

        Returns:
            Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: x0 positions, y0 positions,
                                                         and dictionary of setup parameters
        """
        image_size = config.pixel_size * np.array(camera_params.gain.shape)

        # Generate random positions around center
        x0 = np.full(config.n_bootstrap, image_size[0] / 2) + np.random.uniform(
            low=-config.pixel_size, high=config.pixel_size, size=config.n_bootstrap
        )
        y0 = np.full(config.n_bootstrap, image_size[1] / 2) + np.random.uniform(
            low=-config.pixel_size, high=config.pixel_size, size=config.n_bootstrap
        )

        # Calculate expected parameters for validation
        sigma_PSF = self.psf.sigma_PSF(average_emission_wavelength, config.NA)
        dye_fit_expectation = dye_pixel_efficiency / np.sum(dye_pixel_efficiency)

        expected_parameters = np.array(
            [
                camera_params.gain.shape[0] / 2,  # xc in pixels
                camera_params.gain.shape[1] / 2,  # yc in pixels
                sigma_PSF / config.pixel_size,  # s_x in pixels
                sigma_PSF / config.pixel_size,  # s_y in pixels
            ]
        )
        expected_parameters = np.hstack(
            [
                expected_parameters,
                np.array([config.background_photons / 3] * 3).ravel(),
                dye_fit_expectation.ravel(),
            ]
        )

        setup_data = {
            "x0": x0,
            "y0": y0,
            "expected_parameters": expected_parameters,
            "dye_fit_expectation": dye_fit_expectation,
            "sigma_PSF": sigma_PSF,
        }

        return x0, y0, setup_data

    def _prepare_fitting_data(
        self,
        bayer_image: np.ndarray,
        smoothed_image: np.ndarray,
        camera_params: CameraParameters,
        strategy: FittingStrategy,
        config: SimulationConfig,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for fitting based on the chosen fitting strategy.

        Args:
            bayer_image (np.ndarray): Raw Bayer-filtered camera image
            smoothed_image (np.ndarray): Smoothed version of the Bayer image
            camera_params (CameraParameters): Camera calibration parameters
            strategy (FittingStrategy): Fitting approach to use
            config (SimulationConfig): Simulation configuration

        Returns:
            Tuple[np.ndarray, np.ndarray, Optional[Tuple]]: Photoelectron data, smoothed data,
                                                          and optional grayscale data for some strategies
        """
        # Convert to photoelectron data
        photoelectron_data = np.divide(
            np.divide(
                np.subtract(bayer_image, camera_params.offset), camera_params.gain
            ),
            camera_params.rqe,
        )

        smoothed_data = np.divide(
            np.divide(
                np.subtract(smoothed_image, camera_params.offset), camera_params.gain
            ),
            camera_params.rqe,
        )

        # Handle different demosaic strategies
        if strategy == FittingStrategy.DEMOSAIC_IG:
            _, grayscale_photoelectron_data = self.scmos.bayer_demosaic_stack(
                photoelectron_data, True
            )
            _, grayscale_smoothed_data = self.scmos.bayer_demosaic_stack(
                smoothed_data, True
            )
            return (
                photoelectron_data,
                smoothed_data,
                (grayscale_photoelectron_data, grayscale_smoothed_data),
            )

        elif strategy in [FittingStrategy.DEMOSAIC_FAST, FittingStrategy.DEMOSAIC]:
            if strategy == FittingStrategy.DEMOSAIC_FAST:
                photoelectron_data, grayscale_data = self.scmos.bayer_demosaic_stack(
                    photoelectron_data, True
                )
                smoothed_data, grayscale_smoothed = self.scmos.bayer_demosaic_stack(
                    smoothed_data, True
                )
            else:
                photoelectron_data, _ = self.scmos.bayer_demosaic_stack(
                    photoelectron_data
                )
                smoothed_data, _ = self.scmos.bayer_demosaic_stack(smoothed_data)
                grayscale_data = grayscale_smoothed = None

            # Destack for colour fitting
            photoelectron_data = self._bayer_destacker(photoelectron_data)
            smoothed_data = self._bayer_destacker(smoothed_data)
            return (
                photoelectron_data,
                smoothed_data,
                (grayscale_data, grayscale_smoothed),
            )

        else:  # STANDARD
            return photoelectron_data, smoothed_data, None

    def _bayer_destacker(self, RGB_image: np.ndarray) -> np.ndarray:
        """
        Destack RGB image into separate colour planes for individual channel fitting.

        Args:
            RGB_image (np.ndarray): RGB image stack with shape (frames, height, width, 3)

        Returns:
            np.ndarray: Destacked image with shape (frames*3, height, width) where
                       each frame contains data from a single colour channel
        """
        destacked_image = np.zeros(
            [RGB_image.shape[0] * 3, RGB_image.shape[1], RGB_image.shape[2]]
        )
        index = 0
        for i in range(RGB_image.shape[0]):
            for j in range(3):
                destacked_image[index] = RGB_image[i, :, :, j]
                index += 1
        return destacked_image

    def _compute_error_maps(
        self,
        smoothed_data: np.ndarray,
        grayscale_smoothed: Optional[np.ndarray],
        camera_params: CameraParameters,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute error and weight maps for weighted least-squares fitting.

        Args:
            smoothed_data (np.ndarray): Smoothed photoelectron data for main fitting
            grayscale_smoothed (Optional[np.ndarray]): Grayscale data if using IG strategy
            camera_params (CameraParameters): Camera parameters including readnoise

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Weight maps for main data and optionally grayscale data
        """
        # Main error map
        error_data = deepcopy(smoothed_data)
        error_data[error_data < 0] = 0
        error_data = error_data + 1
        error_map = np.add(error_data, np.square(camera_params.readnoise))
        weights_map = np.power(error_map, -1)

        # Grayscale error map if provided
        weights_grayscale_map = None
        if grayscale_smoothed is not None:
            error_grayscale = deepcopy(grayscale_smoothed)
            error_grayscale[error_grayscale < 0] = 0
            error_grayscale = error_grayscale + 1
            error_grayscale_map = np.add(
                error_grayscale, np.square(camera_params.readnoise)
            )
            weights_grayscale_map = np.power(error_grayscale_map, -1)

        return weights_map, weights_grayscale_map

    def _perform_fitting(
        self,
        strategy: FittingStrategy,
        photoelectron_data: np.ndarray,
        smoothed_data: np.ndarray,
        weights_map: np.ndarray,
        grayscale_data: Optional[Tuple],
        camera_params: CameraParameters,
        config: SimulationConfig,
    ) -> pd.DataFrame:
        """
        Perform fitting based on the specified strategy.

        Args:
            strategy (FittingStrategy): Fitting approach to use
            photoelectron_data (np.ndarray): Converted photoelectron image data
            smoothed_data (np.ndarray): Smoothed photoelectron data
            weights_map (np.ndarray): Weight maps for fitting
            grayscale_data (Optional[Tuple]): Grayscale data if using demosaic strategies
            camera_params (CameraParameters): Camera calibration parameters
            config (SimulationConfig): Simulation configuration

        Returns:
            pd.DataFrame: Fitting results with localization and photometry parameters
        """

        if strategy == FittingStrategy.STANDARD:
            return self._fit_standard(
                photoelectron_data, smoothed_data, weights_map, camera_params, config
            )
        elif strategy == FittingStrategy.DEMOSAIC_IG:
            return self._fit_demosaic_ig(
                photoelectron_data,
                smoothed_data,
                weights_map,
                grayscale_data,
                camera_params,
                config,
            )
        elif strategy == FittingStrategy.DEMOSAIC_FAST:
            return self._fit_demosaic_fast(
                photoelectron_data,
                smoothed_data,
                weights_map,
                grayscale_data,
                camera_params,
                config,
            )
        elif strategy == FittingStrategy.DEMOSAIC:
            return self._fit_demosaic(
                photoelectron_data, smoothed_data, weights_map, config
            )
        else:
            raise ValueError(f"Unknown fitting strategy: {strategy}")

    def _fit_standard(
        self,
        photoelectron_data: np.ndarray,
        smoothed_data: np.ndarray,
        weights_map: np.ndarray,
        camera_params: CameraParameters,
        config: SimulationConfig,
    ) -> pd.DataFrame:
        """
        Standard fitting approach using Bayer pattern masks directly.

        Args:
            photoelectron_data (np.ndarray): Photoelectron image data
            smoothed_data (np.ndarray): Smoothed photoelectron data for initial guesses
            weights_map (np.ndarray): Fitting weight maps
            camera_params (CameraParameters): Camera parameters including Bayer masks
            config (SimulationConfig): Simulation configuration

        Returns:
            pd.DataFrame: Fitting results with position, shape, and colour information
        """
        masks_3d = np.dstack(
            [camera_params.masks[x] for x in camera_params.masks.keys()]
        )

        # Prepare fitting data
        puncta_tofit, smoothed_puncta_tofit, masks_tofit, weights_tofit = [], [], [], []
        relative_coords, planes = [], []

        for frame in range(config.n_bootstrap):
            puncta_tofit.append(photoelectron_data[frame, :, :])
            smoothed_puncta_tofit.append(smoothed_data[frame, :, :])
            masks_tofit.append(masks_3d)
            weights_tofit.append(weights_map[frame, :, :])
            relative_coords.append((0, 0))
            planes.append(frame)

        # Clean up memory
        del photoelectron_data, smoothed_data, weights_map
        gc.collect()

        # Perform fitting
        fit_results, fit_errors = self.image_analysis.fit_puncta_parallel_method(
            puncta_tofit,
            smoothed_puncta_tofit,
            weights_tofit,
            relative_coords,
            planes,
            IAF_FittingStrategy.STANDARD,
            masks=masks_tofit,
        )

        columns = [
            "xc",
            "yc",
            "s_x",
            "s_y",
            "bg_B",
            "bg_G",
            "bg_R",
            "A_B",
            "A_G",
            "A_R",
            "chi_sqr",
            "frame",
        ]
        error_columns = [
            "xc_err",
            "yc_err",
            "s_x_err",
            "s_y_err",
            "bg_B_err",
            "bg_G_err",
            "bg_R_err",
            "A_B_err",
            "A_G_err",
            "A_R_err",
        ]

        # Combine fit results and errors
        fit_results = pd.DataFrame(fit_results, columns=columns).sort_values(
            by=["frame"]
        )
        fit_errors_df = pd.DataFrame(fit_errors, columns=error_columns)
        fit_results = pd.concat(
            [fit_results.reset_index(drop=True), fit_errors_df], axis=1
        )

        # CRITICAL FIX: Sqrt transformation error correction
        # The fitter works with sqrt(A) and sqrt(bg), but returns A and bg after squaring (line 316 ImageAnalysisFunctions)
        # The errors are for sqrt(A), but we need errors for A
        # Error propagation: if p = sqrt(A), then δA = |dA/dp| × δp = 2*sqrt(A) × δp
        #
        # Since fit_results already contains squared values (A, not sqrt(A)), we need:
        # δA_corrected = 2 × sqrt(A) × δ(sqrt(A))

        # Before squaring was applied, the fitted parameter was sqrt(A)
        # So we need to multiply errors by 2 × sqrt(A) where A is the current (squared) value
        for param, param_err in [("A_R", "A_R_err"), ("A_G", "A_G_err"), ("A_B", "A_B_err"),
                                  ("bg_R", "bg_R_err"), ("bg_G", "bg_G_err"), ("bg_B", "bg_B_err")]:
            # Multiply error by 2*sqrt(A) to account for sqrt transformation
            # Only apply where A > 0 to avoid sqrt of negative/zero
            mask = fit_results[param] > 0
            fit_results.loc[mask, param_err] = (
                fit_results.loc[mask, param_err] * 2.0 * np.sqrt(fit_results.loc[mask, param])
            )

        # Now normalize amplitudes AND their errors
        fit_results["photons"] = (
            fit_results["A_R"] + fit_results["A_G"] + fit_results["A_B"]
        )
        fit_results["background_photons"] = (
            fit_results["bg_R"] + fit_results["bg_G"] + fit_results["bg_B"]
        )

        # Normalize amplitude values by total photons
        for cparam in ["A_R", "A_G", "A_B"]:
            fit_results[cparam] = fit_results[cparam] / fit_results["photons"]

        # Normalize amplitude errors by total photons (same denominator)
        for cparam_err in ["A_R_err", "A_G_err", "A_B_err"]:
            fit_results[cparam_err] = fit_results[cparam_err] / fit_results["photons"]

        # Normalize background values by total background photons
        for cparam in ["bg_R", "bg_G", "bg_B"]:
            fit_results[cparam] = fit_results[cparam] / fit_results["background_photons"]

        # Normalize background errors by total background photons (same denominator)
        for cparam_err in ["bg_R_err", "bg_G_err", "bg_B_err"]:
            fit_results[cparam_err] = fit_results[cparam_err] / fit_results["background_photons"]

        return fit_results

    def _add_nile_red_wavelength_fits(
        self,
        fit_results: pd.DataFrame,
        nile_red_wavelength: float,
        camera_params: CameraParameters,
        camera_parameters: Dict[str, Any],
        wavelength: np.ndarray,
        filters: List[str],
        config: SimulationConfig,
    ) -> pd.DataFrame:
        """
        Add Nile Red wavelength fitting columns to fit results DataFrame.

        For each row in fit_results, fits the Nile Red wavelength from RGB and sigma values,
        then adds 'wl_fit' and 'wl_fit_err' columns.

        Args:
            fit_results: DataFrame with fit results including RGB, sigma, and error columns
            nile_red_wavelength: True wavelength used for simulation (for spectrum generation)
            camera_params: Camera parameters dataclass
            camera_parameters: Full camera parameters dictionary with all optical parameters
            wavelength: Wavelength array (nm)
            filters: List of filter names
            config: Simulation configuration (for pixel_size, NA, etc.)

        Returns:
            DataFrame with added 'wl_fit' and 'wl_fit_err' columns
        """
        try:
            # Import NileRedFunctions and SpectralFunctions
            import NileRedFunctions
            import SpectralFunctions

            nrf = NileRedFunctions.NileRed_Functions()
            spectral_funcs = SpectralFunctions.Spectral_Funcs()

            # Get wavelength array and filter spectra
            wavelength_array = wavelength
            pixel_QYs = camera_params.pixel_QYs

            # Get filter spectra using spectral functions
            filter_spectra = spectral_funcs.get_dye_or_filter_data(
                names=filters, wavelength=wavelength_array, dye_or_filter=False
            )

            # NOTE: LUT caching removed - wavelength fitting now uses direct forward model
            # This is slower but more flexible and avoids the need to pre-generate LUTs

            # Extract data from DataFrame
            R = fit_results["A_R"].to_numpy()
            G = fit_results["A_G"].to_numpy()
            B = fit_results["A_B"].to_numpy()
            sigma_x = fit_results["s_x"].to_numpy() * config.pixel_size  # Convert to nm
            sigma_y = fit_results["s_y"].to_numpy() * config.pixel_size

            R_err = fit_results["A_R_err"].to_numpy()
            G_err = fit_results["A_G_err"].to_numpy()
            B_err = fit_results["A_B_err"].to_numpy()
            sigma_x_err = fit_results["s_x_err"].to_numpy() * config.pixel_size
            sigma_y_err = fit_results["s_y_err"].to_numpy() * config.pixel_size

            # Extract fitted photons and background for SNR calculation
            fitted_photons = fit_results["photons"].to_numpy()
            fitted_bg_R = fit_results["bg_R"].to_numpy()
            fitted_bg_G = fit_results["bg_G"].to_numpy()
            fitted_bg_B = fit_results["bg_B"].to_numpy()

            # Calculate total background photons from fitted background values
            # Note: In _fit_standard (line 765-770), only amplitudes A_R/G/B are normalized
            # Background values bg_R/G/B are kept as absolute photon counts (not normalized)
            # So we simply sum them to get total background
            # This mirrors real experimental analysis where we use fitted (not ground truth) values
            fitted_background_photons = fitted_bg_R + fitted_bg_G + fitted_bg_B

            # Normalize RGB (fit_results already has normalized RGB from _fit_standard)
            # But we need to propagate errors properly
            rgb_total = R + G + B

            # Prepare arguments for parallel processing
            fit_args = []
            valid_indices = []
            for j in range(len(R)):
                if rgb_total[j] <= 0:
                    continue

                R_norm = R[j] / rgb_total[j]
                G_norm = G[j] / rgb_total[j]
                B_norm = B[j] / rgb_total[j]

                # Propagate errors
                total_err = np.sqrt(R_err[j] ** 2 + G_err[j] ** 2 + B_err[j] ** 2)
                R_norm_err = (
                    R_norm
                    * np.sqrt((R_err[j] / R[j]) ** 2 + (total_err / rgb_total[j]) ** 2)
                    if R[j] > 0
                    else 1e-3
                )
                G_norm_err = (
                    G_norm
                    * np.sqrt((G_err[j] / G[j]) ** 2 + (total_err / rgb_total[j]) ** 2)
                    if G[j] > 0
                    else 1e-3
                )
                B_norm_err = (
                    B_norm
                    * np.sqrt((B_err[j] / B[j]) ** 2 + (total_err / rgb_total[j]) ** 2)
                    if B[j] > 0
                    else 1e-3
                )

                fit_args.append(
                    (
                        np.array([R_norm, G_norm, B_norm]),
                        sigma_x[j],
                        sigma_y[j],
                        np.array([R_norm_err, G_norm_err, B_norm_err]),
                        sigma_x_err[j],
                        sigma_y_err[j],
                        filter_spectra,
                        wavelength_array,
                        pixel_QYs,
                        config.NA,
                        fitted_photons[j],  # Pass fitted photon count
                        fitted_background_photons[j],  # Pass fitted background photons
                        (500.0, 750.0),  # wavelength_bounds - extended range for Nile Red
                    )
                )
                valid_indices.append(j)

            # Parallel wavelength fitting using ProcessPoolExecutor
            wl_fits = np.full(len(R), np.nan)
            wl_fit_errs = np.full(len(R), np.nan)

            if len(fit_args) > 0:
                # Use multiprocessing for parallel fitting
                import multiprocessing
                from concurrent import futures

                # Calculate number of workers (use cpu_fraction from config)
                n_cpus = multiprocessing.cpu_count()
                n_workers = max(1, int(n_cpus * config.cpu_fraction))

                with futures.ProcessPoolExecutor(n_workers) as executor:
                    # Submit all fitting tasks
                    future_list = [
                        executor.submit(_fit_nile_red_wavelength_standalone, *args)
                        for args in fit_args
                    ]

                    # Collect results as they complete
                    for idx, future in enumerate(future_list):
                        try:
                            wl, wl_err = future.result(
                                timeout=30
                            )  # 30 second timeout per fit
                            wl_fits[valid_indices[idx]] = wl
                            wl_fit_errs[valid_indices[idx]] = wl_err
                        except Exception as e:
                            logger.warning(
                                f"Wavelength fit failed for index {valid_indices[idx]}: {e}"
                            )
                            wl_fits[valid_indices[idx]] = np.nan
                            wl_fit_errs[valid_indices[idx]] = np.nan

            # Add columns to DataFrame
            fit_results["wl_fit"] = wl_fits
            fit_results["wl_fit_err"] = wl_fit_errs

            return fit_results

        except Exception as e:
            logger.warning(f"Could not add Nile Red wavelength fits: {e}")
            # Return unchanged DataFrame if wavelength fitting fails
            return fit_results

    def _fit_demosaic_ig(
        self,
        photoelectron_data: np.ndarray,
        smoothed_data: np.ndarray,
        weights_map: np.ndarray,
        grayscale_data: Tuple,
        camera_params: CameraParameters,
        config: SimulationConfig,
    ) -> pd.DataFrame:
        """
        Demosaic Initial Guess (IG) fitting approach with two-stage fitting.

        First fits grayscale demosaiced data to get positions and shapes, then
        uses these as fixed parameters to fit colour information from the original Bayer data.

        Args:
            photoelectron_data (np.ndarray): Original Bayer photoelectron data
            smoothed_data (np.ndarray): Smoothed Bayer data
            weights_map (np.ndarray): Weight maps for Bayer data fitting
            grayscale_data (Tuple): Grayscale photoelectron and smoothed data
            camera_params (CameraParameters): Camera parameters
            config (SimulationConfig): Simulation configuration

        Returns:
            pd.DataFrame: Combined fitting results from both stages
        """
        grayscale_photoelectron_data, grayscale_smoothed_data = grayscale_data
        weights_grayscale_map = self._compute_error_maps(
            smoothed_data, grayscale_smoothed_data, camera_params
        )[1]

        # First fit grayscale
        puncta_tofit, smoothed_puncta_tofit, weights_tofit = [], [], []
        relative_coords, planes = [], []

        for frame in range(config.n_bootstrap):
            puncta_tofit.append(grayscale_photoelectron_data[frame, :, :])
            smoothed_puncta_tofit.append(grayscale_smoothed_data[frame, :, :])
            weights_tofit.append(weights_grayscale_map[frame, :, :])
            relative_coords.append((0, 0))
            planes.append(frame)

        del grayscale_photoelectron_data, grayscale_smoothed_data, weights_grayscale_map
        gc.collect()

        default_params = ["xc", "yc", "s_x", "s_y", "b", "A", "chi_sqr", "frame"]
        fit_results, _ = self.image_analysis.fit_puncta_parallel_method(
            puncta_tofit,
            smoothed_puncta_tofit,
            weights_tofit,
            relative_coords,
            planes,
            IAF_FittingStrategy.NOCOLOUR,
        )
        fit_results = pd.DataFrame(fit_results, columns=default_params).sort_values(
            by=["frame"]
        )

        # Second fit for colour using position information
        masks_3d = np.dstack(
            [camera_params.masks[x] for x in camera_params.masks.keys()]
        )
        puncta_tofit, smoothed_puncta_tofit, weights_tofit = [], [], []
        locparams, planes, masks_tofit = [], [], []

        for frame in range(config.n_bootstrap):
            puncta_tofit.append(photoelectron_data[frame, :, :])
            smoothed_puncta_tofit.append(smoothed_data[frame, :, :])
            weights_tofit.append(weights_map[frame, :, :])
            locparams.append(
                (
                    fit_results["xc"][frame],
                    fit_results["yc"][frame],
                    fit_results["s_x"][frame],
                    fit_results["s_y"][frame],
                )
            )
            masks_tofit.append(masks_3d)
            planes.append(frame)

        del photoelectron_data, smoothed_data, weights_map
        gc.collect()

        fit_results_colour, _ = self.image_analysis.fit_puncta_parallel_method(
            puncta_tofit,
            smoothed_puncta_tofit,
            weights_tofit,
            relative_coords,
            planes,
            IAF_FittingStrategy.RAWCOLOUR,
            masks=masks_tofit,
        )

        colour_columns = [
            "bg_B",
            "bg_G",
            "bg_R",
            "A_B",
            "A_G",
            "A_R",
            "chi_sqr",
            "frame",
        ]
        fit_results_colour = pd.DataFrame(
            fit_results_colour, columns=colour_columns
        ).sort_values(by=["frame"])

        # Normalise colour results
        fit_results_colour["photons"] = (
            fit_results_colour["A_B"]
            + fit_results_colour["A_G"]
            + fit_results_colour["A_R"]
        )
        fit_results_colour["background_photons"] = (
            fit_results_colour["bg_B"]
            + fit_results_colour["bg_G"]
            + fit_results_colour["bg_R"]
        )

        for param in ["A_B", "A_G", "A_R"]:
            fit_results_colour[param] = (
                fit_results_colour[param] / fit_results_colour["photons"]
            )
        for param in ["bg_B", "bg_G", "bg_R"]:
            fit_results_colour[param] = (
                fit_results_colour[param] / fit_results_colour["background_photons"]
            )

        # CRITICAL FIX: Normalize errors too! (_fit_demosaic_ig doesn't have error columns, so skip)
        # Note: This method doesn't return error columns in fit_results_colour,
        # so no error normalization needed here

        return pd.concat([fit_results, fit_results_colour], axis=1)

    def _fit_demosaic_fast(
        self,
        photoelectron_data: np.ndarray,
        smoothed_data: np.ndarray,
        weights_map: np.ndarray,
        grayscale_data: Tuple,
        camera_params: CameraParameters,
        config: SimulationConfig,
    ) -> pd.DataFrame:
        """
        Fast demosaic fitting approach with optimised colour channel processing.

        Similar to IG method but uses optimised fitting for colour channels to reduce
        computational time while maintaining reasonable accuracy.

        Args:
            photoelectron_data (np.ndarray): Demosaiced RGB photoelectron data (destacked)
            smoothed_data (np.ndarray): Smoothed RGB data (destacked)
            weights_map (np.ndarray): Weight maps for RGB data
            grayscale_data (Tuple): Grayscale photoelectron and smoothed data
            camera_params (CameraParameters): Camera parameters
            config (SimulationConfig): Simulation configuration

        Returns:
            pd.DataFrame: Averaged fitting results across colour channels
        """
        grayscale_photoelectron_data, grayscale_smoothed_data = grayscale_data
        weights_grayscale_map = self._compute_error_maps(
            smoothed_data, grayscale_smoothed_data, camera_params
        )[1]

        # First grayscale fit (similar to IG method)
        puncta_tofit, smoothed_puncta_tofit, weights_tofit = [], [], []
        relative_coords, planes = [], []

        for frame in range(config.n_bootstrap):
            puncta_tofit.append(grayscale_photoelectron_data[frame, :, :])
            smoothed_puncta_tofit.append(grayscale_smoothed_data[frame, :, :])
            weights_tofit.append(weights_grayscale_map[frame, :, :])
            relative_coords.append((0, 0))
            planes.append(frame)

        del grayscale_photoelectron_data, grayscale_smoothed_data, weights_grayscale_map
        gc.collect()

        default_params = ["xc", "yc", "s_x", "s_y", "b", "A", "chi_sqr", "frame"]
        fit_results, _ = self.image_analysis.fit_puncta_parallel_method(
            puncta_tofit,
            smoothed_puncta_tofit,
            weights_tofit,
            relative_coords,
            planes,
            IAF_FittingStrategy.NOCOLOUR,
        )
        fit_results = pd.DataFrame(fit_results, columns=default_params).sort_values(
            by=["frame"]
        )

        # Fast colour fitting
        puncta_tofit, smoothed_puncta_tofit, weights_tofit = [], [], []
        locparams, planes, masks_tofit = [], [], []
        masks_3d = np.dstack(
            [camera_params.masks[x] for x in camera_params.masks.keys()]
        )

        for frame in range(config.n_bootstrap * 3):
            puncta_tofit.append(photoelectron_data[frame, :, :])
            smoothed_puncta_tofit.append(smoothed_data[frame, :, :])
            weights_tofit.append(weights_map[frame, :, :])
            masks_tofit.append(masks_3d)
            idx = frame // 3
            locparams.append(
                (
                    fit_results["xc"][idx],
                    fit_results["yc"][idx],
                    fit_results["s_x"][idx],
                    fit_results["s_y"][idx],
                )
            )
            planes.append(frame)

        del photoelectron_data, smoothed_data, weights_map
        gc.collect()

        fit_results_colour, _ = self.image_analysis.fit_puncta_parallel_method(
            puncta_tofit,
            smoothed_puncta_tofit,
            weights_tofit,
            relative_coords,
            planes,
            IAF_FittingStrategy.JUSTCOLOUR,
            masks=masks_tofit,
        )

        colour_columns = ["A", "b", "chi_sqr", "frame"]
        fit_results_colour = pd.DataFrame(
            fit_results_colour, columns=colour_columns
        ).sort_values(by=["frame"])
        fit_results_colour = self.result_processor.colour_fit_averager(
            fit_results_colour, config.n_bootstrap
        )

        return pd.concat([fit_results, fit_results_colour], axis=1)

    def _fit_demosaic(
        self,
        photoelectron_data: np.ndarray,
        smoothed_data: np.ndarray,
        weights_map: np.ndarray,
        config: SimulationConfig,
    ) -> pd.DataFrame:
        """
        Standard demosaic fitting approach with full RGB channel fitting.

        Fits each RGB channel separately then averages results to get final
        position, shape and colour information.

        Args:
            photoelectron_data (np.ndarray): Demosaiced RGB photoelectron data (destacked)
            smoothed_data (np.ndarray): Smoothed RGB data (destacked)
            weights_map (np.ndarray): Weight maps for RGB data
            config (SimulationConfig): Simulation configuration

        Returns:
            pd.DataFrame: Averaged fitting results from all RGB channels
        """
        puncta_tofit, smoothed_puncta_tofit, weights_tofit = [], [], []
        relative_coords, planes = [], []

        for frame in range(config.n_bootstrap * 3):
            puncta_tofit.append(photoelectron_data[frame, :, :])
            smoothed_puncta_tofit.append(smoothed_data[frame, :, :])
            weights_tofit.append(weights_map[frame, :, :])
            relative_coords.append((0, 0))
            planes.append(frame)

        del photoelectron_data, smoothed_data, weights_map
        gc.collect()

        default_params = ["xc", "yc", "s_x", "s_y", "b", "A", "chi_sqr", "frame"]
        fit_results, _ = self.image_analysis.fit_puncta_parallel_method(
            puncta_tofit,
            smoothed_puncta_tofit,
            weights_tofit,
            relative_coords,
            planes,
            IAF_FittingStrategy.NOCOLOUR,
        )

        fit_results = pd.DataFrame(fit_results, columns=default_params).sort_values(
            by=["frame"]
        )
        return self.result_processor.fit_averager(fit_results, config.n_bootstrap)

    def _compute_fit_statistics(
        self,
        fit_results: pd.DataFrame,
        setup_data: Dict,
        config: SimulationConfig,
        analysis_save_params: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute RMSE and standard deviation statistics for fit results."""
        x0, y0 = setup_data["x0"], setup_data["y0"]
        expected_parameters = setup_data["expected_parameters"]
        dye_fit_expectation = setup_data["dye_fit_expectation"]

        fit_RMSE_mean = np.zeros(len(analysis_save_params) - 1)
        fit_std = np.zeros(len(analysis_save_params) - 1)

        for loc, param in enumerate(analysis_save_params[:-1]):
            if param == "xc":
                fit_RMSE_mean[loc] = config.pixel_size * np.nanmean(
                    np.sqrt(
                        np.square(
                            fit_results[param].to_numpy() - (x0 / config.pixel_size)
                        )
                    )
                )
                fit_std[loc] = config.pixel_size * np.nanstd(
                    np.sqrt(
                        np.square(
                            fit_results[param].to_numpy() - (x0 / config.pixel_size)
                        )
                    )
                )
            elif param == "yc":
                fit_RMSE_mean[loc] = config.pixel_size * np.nanmean(
                    np.sqrt(
                        np.square(
                            fit_results[param].to_numpy() - (y0 / config.pixel_size)
                        )
                    )
                )
                fit_std[loc] = config.pixel_size * np.nanstd(
                    np.sqrt(
                        np.square(
                            fit_results[param].to_numpy() - (y0 / config.pixel_size)
                        )
                    )
                )
            elif param == "chi_sqr":
                colour_loc = np.expand_dims(dye_fit_expectation, 0)
                colour = np.vstack(
                    [
                        fit_results["A_B"].to_numpy(),
                        fit_results["A_G"].to_numpy(),
                        fit_results["A_R"].to_numpy(),
                    ]
                ).T
                distances = cdist(colour, colour_loc)
                fit_RMSE_mean[loc] = np.nanmean(distances)
                fit_std[loc] = np.nanstd(distances)
            elif param in ["s_x", "s_y"]:
                fit_RMSE_mean[loc] = config.pixel_size * np.nanmean(
                    np.sqrt(
                        np.square(
                            fit_results[param].to_numpy() - expected_parameters[loc]
                        )
                    )
                )
                fit_std[loc] = config.pixel_size * np.nanstd(
                    fit_results[param].to_numpy()
                )
            else:
                fit_RMSE_mean[loc] = np.nanmean(
                    np.sqrt(
                        np.square(
                            fit_results[param].to_numpy() - expected_parameters[loc]
                        )
                    )
                )
                fit_std[loc] = np.nanstd(fit_results[param].to_numpy())

        return fit_RMSE_mean, fit_std

    def gen_camera_image_stack(
        self,
        camera_calibration: Dict[str, Any],
        wavelength: np.ndarray,
        average_emission_wavelengths: Union[float, np.ndarray],
        dye_pixel_efficiency: np.ndarray,
        n_photons: Dict[str, Union[int, np.ndarray]],
        x0y0: Dict[str, np.ndarray],
        smoothing_function,
        background_photons: float = 0,
        background_colour: List[float] = None,
        NA: float = 1.49,
        pixel_size: float = 69,
        return_normal_image: bool = False,
        return_photoelectrons: bool = False,
        use_vectorized_photoelectrons: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Generate camera image stack with optional vectorized photoelectron generation.

        Args:
            return_normal_image: If True, generate a normal (non-Bayer) image with flat QE
            return_photoelectrons: If True, return normal_image in photoelectrons instead of ADU
                                  (only applies when return_normal_image=True)
            use_vectorized_photoelectrons: If True, vectorize photoelectron generation across frames
                                          for 2-5× speedup. If False, use original per-frame loop.
                                          Default: True (recommended)

        Returns:
            Tuple of (bayer_image, smoothed_image, normal_image)
            - bayer_image: Bayer-patterned image in ADU
            - smoothed_image: Smoothed bayer image in ADU
            - normal_image: Normal image in ADU (or photoelectrons if return_photoelectrons=True)
        """
        if background_colour is None:
            background_colour = [1, 1, 1]

        self._validate_inputs(
            wavelength, camera_calibration, dye_pixel_efficiency, x0y0
        )

        dye_names = x0y0.keys()
        gain = camera_calibration["gain"]
        offset = camera_calibration["offset"]
        variance = camera_calibration["variance"]
        relative_QE = camera_calibration["rqe"]

        # Calculate sigma in nm, then convert to pixels for PSF generation
        # Handle both scalar wavelengths (deterministic) and per-frame wavelengths (stochastic)
        # Use ndim to distinguish: 0 or scalar = deterministic, 1+ = stochastic (per-frame)
        if np.ndim(average_emission_wavelengths) == 0 or np.isscalar(average_emission_wavelengths):
            # Deterministic: single wavelength for all frames
            sigma_nm = self.psf.sigma_PSF(average_emission_wavelengths, NA)
            sigma_x = sigma_nm / pixel_size
            sigma_y = sigma_x
            sigma_per_frame = None  # Flag that sigma is constant
        else:
            # Stochastic: array of wavelengths, one per frame
            # Pre-compute sigma for each frame
            sigma_nm_array = np.array([
                self.psf.sigma_PSF(wl, NA) for wl in average_emission_wavelengths
            ])
            sigma_per_frame = sigma_nm_array / pixel_size
            # Initialize sigma_x, sigma_y (will be updated per frame)
            sigma_x = sigma_per_frame[0]
            sigma_y = sigma_x

        pixel_colours = camera_calibration["pixel_order"]

        if return_normal_image:
            overall_QY = np.sum(
                dye_pixel_efficiency, axis=len(dye_pixel_efficiency.shape) - 1
            )

        w, h = gain.shape
        try:
            s = n_photons[list(dye_names)[0]].shape[0]
        except (AttributeError, IndexError):
            s = 1

        # Use pixel coordinates (not nm) for PSF generation
        x = np.arange(w, dtype=np.float32)
        masks = camera_calibration["masks"]

        # OPTIMIZATION: Pre-compute mask stack for vectorized operations
        # Stack masks in the order specified by pixel_order: shape (w, h, n_channels)
        mask_stack = np.stack([masks[colour] for colour in pixel_colours], axis=2)

        # Calculate absolute quantum efficiency per pixel (for deterministic mode)
        # NOTE: This stores QE values in a pixel array for backward compatibility
        # The actual photoelectron generation now uses QE_per_channel to avoid
        # the bug where photons could generate photoelectrons on multiple channels
        abs_QE = np.zeros([w, h, len(dye_names)])
        for j, dye in enumerate(dye_names):
            for i, colour in enumerate(pixel_colours):
                try:
                    dpe = (
                        dye_pixel_efficiency[j, i]
                        if len(dye_pixel_efficiency.shape) > 1
                        else dye_pixel_efficiency[i]
                    )
                except (IndexError, TypeError):
                    dpe = dye_pixel_efficiency
                abs_QE[:, :, j] += masks[colour] * dpe

        # Calculate background photons matrix
        background_photons_perdye = background_photons / len(dye_names)
        background_photons_matrix = np.zeros([w, h, len(dye_names)])

        # Normalize background_colour to ensure total background = background_photons
        background_colour_normalized = np.array(background_colour) / np.sum(
            background_colour
        )

        for j, dye in enumerate(dye_names):
            for i, colour in enumerate(pixel_colours):
                try:
                    dpe = (
                        dye_pixel_efficiency[j, i]
                        if len(dye_pixel_efficiency.shape) > 1
                        else dye_pixel_efficiency[i]
                    )
                except (IndexError, TypeError):
                    dpe = dye_pixel_efficiency

                if dpe != 0:
                    background_photons_matrix[:, :, j] += (
                        masks[colour]
                        * (background_colour_normalized[i] / dpe)
                        * background_photons_perdye
                    )

        bayer_image = np.zeros([s, w, h])
        if return_normal_image:
            normal_image = np.zeros([s, w, h])

        # Choose between vectorized and original per-frame implementation
        if use_vectorized_photoelectrons:
            # ======================================================================
            # VECTORIZED PATH: 2-5× faster photoelectron generation
            # ======================================================================
            # Pre-allocate arrays to accumulate photon counts from all frames
            n_photons_hitting_detector_all = np.zeros([s, w, h, len(dye_names)], dtype=np.int64)
            QE_per_channel_all = np.zeros([s, len(dye_names), len(pixel_colours)])

            # PHASE 1: Generate photon counts (per-frame loop, unchanged)
            for frame in range(s):
                # Update sigma if per-frame wavelengths (stochastic mode)
                if sigma_per_frame is not None:
                    sigma_x = sigma_per_frame[frame]
                    sigma_y = sigma_x

                # Update abs_QE and background if per-frame colour ratios (stochastic mode)
                if dye_pixel_efficiency.ndim == 2 and dye_pixel_efficiency.shape[0] == s:
                    # Stochastic mode
                    QE_per_channel_frame = np.zeros([len(dye_names), len(pixel_colours)])
                    background_photons_matrix_frame = np.zeros([w, h, len(dye_names)])

                    for j, dye in enumerate(dye_names):
                        for i, colour in enumerate(pixel_colours):
                            dpe = dye_pixel_efficiency[frame, i]
                            QE_per_channel_frame[j, i] = dpe

                            if dpe != 0:
                                background_photons_matrix_frame[:, :, j] += (
                                    masks[colour]
                                    * (background_colour_normalized[i] / dpe)
                                    * background_photons_perdye
                                )
                else:
                    # Deterministic mode
                    QE_per_channel_frame = np.zeros([len(dye_names), len(pixel_colours)])
                    for j, dye in enumerate(dye_names):
                        for i, colour in enumerate(pixel_colours):
                            mask_indices = np.where(masks[colour])
                            if len(mask_indices[0]) > 0:
                                QE_per_channel_frame[j, i] = abs_QE[mask_indices[0][0], mask_indices[1][0], j]
                    background_photons_matrix_frame = background_photons_matrix

                # Store QE for this frame
                QE_per_channel_all[frame, :, :] = QE_per_channel_frame

                # Generate photons hitting detector for this frame
                n_photons_hitting_detector = np.zeros([w, h, len(dye_names)], dtype=np.int64)

                for j, dye in enumerate(dye_names):
                    try:
                        n_photons_this_frame = (
                            n_photons[dye][frame]
                            if hasattr(n_photons[dye], "__getitem__")
                            else n_photons[dye]
                        )
                    except (IndexError, TypeError):
                        n_photons_this_frame = n_photons[dye]

                    if n_photons_this_frame > 0:
                        try:
                            x0 = (
                                x0y0[dye][frame, 0, :]
                                if x0y0[dye].ndim > 1
                                else x0y0[dye][0, :]
                            )
                            y0 = (
                                x0y0[dye][frame, 1, :]
                                if x0y0[dye].ndim > 1
                                else x0y0[dye][1, :]
                            )
                        except (IndexError, TypeError):
                            x0, y0 = x0y0[dye][0, :], x0y0[dye][1, :]

                        x0_pixels = x0 / pixel_size
                        y0_pixels = y0 / pixel_size

                        if hasattr(x0, "__len__") and len(x0) > 1:
                            n_photons_array = np.full(len(x0), int(n_photons_this_frame))
                        else:
                            n_photons_array = np.array([int(n_photons_this_frame)])

                        photon_spatial_pdf = self.psf.gen_spatial_PSF(
                            x,
                            sigma_x,
                            sigma_y,
                            x0_pixels,
                            y0_pixels,
                            n_photons_array,
                            relative_QE,
                        )

                        n_photons_total = self.psf.gen_photons_hitting_detector(
                            photon_spatial_pdf, background_photons_matrix_frame[:, :, j]
                        )
                        n_photons_hitting_detector[:, :, j] = n_photons_total

                # Store photon counts for this frame
                n_photons_hitting_detector_all[frame, :, :, :] = n_photons_hitting_detector

            # PHASE 2: ⚡ VECTORIZED photoelectron generation across ALL frames
            n_photoelectrons_all = self.psf.gen_photoelectrons_vectorized_frames(
                n_photons_hitting_detector_all,
                QE_per_channel_all,
                mask_stack,
            )

            # PHASE 3: Convert photoelectrons to images (per-frame loop, fast)
            for frame in range(s):
                bayer_image[frame, :, :] = self.psf.photoelectrons_to_image(
                    np.sum(n_photoelectrons_all[frame, :, :, :], axis=-1),
                    gain,
                    offset,
                    variance,
                )

                if return_normal_image:
                    # Generate normal image: apply overall_QY to photons, not channel-specific
                    # Sum photons across all dyes
                    n_photons_frame_total = np.sum(n_photons_hitting_detector_all[frame, :, :, :], axis=-1)
                    # Apply overall QY (sum of channel QYs) instead of per-channel
                    overall_QY_frame = np.sum(QE_per_channel_all[frame, :, :])  # Sum across all dyes and channels
                    n_photoelectrons_normal = self.psf.gen_photoelectrons(
                        n_photons_frame_total.astype(int),
                        overall_QY_frame / len(pixel_colours)  # Average QY across channels
                    )

                    if return_photoelectrons:
                        # Return raw photoelectrons (ground truth for demosaicing validation)
                        normal_image[frame, :, :] = n_photoelectrons_normal
                    else:
                        # Convert to ADU (standard output)
                        normal_image[frame, :, :] = self.psf.photoelectrons_to_image(
                            n_photoelectrons_normal, gain, offset, variance
                        )

        else:
            # ======================================================================
            # ORIGINAL PER-FRAME PATH: Kept for backward compatibility and testing
            # ======================================================================
            # Generate images for each frame
            for frame in range(s):
                # Update sigma if per-frame wavelengths (stochastic mode)
                if sigma_per_frame is not None:
                    sigma_x = sigma_per_frame[frame]
                    sigma_y = sigma_x

                # Update abs_QE and background if per-frame colour ratios (stochastic mode)
                # Check if dye_pixel_efficiency has per-frame dimension: (n_frames, n_colours)
                if dye_pixel_efficiency.ndim == 2 and dye_pixel_efficiency.shape[0] == s:
                    # Stochastic mode: recalculate abs_QE for this frame
                    # Store QE per channel (not per pixel!) - shape: (n_dyes, n_channels)
                    QE_per_channel_frame = np.zeros([len(dye_names), len(pixel_colours)])
                    background_photons_matrix_frame = np.zeros([w, h, len(dye_names)])
    
                    for j, dye in enumerate(dye_names):
                        for i, colour in enumerate(pixel_colours):
                            # Use frame-specific QE values
                            dpe = dye_pixel_efficiency[frame, i]
                            QE_per_channel_frame[j, i] = dpe
    
                            if dpe != 0:
                                background_photons_matrix_frame[:, :, j] += (
                                    masks[colour]
                                    * (background_colour_normalized[i] / dpe)
                                    * background_photons_perdye
                                )
                else:
                    # Deterministic mode: use pre-computed QE values
                    # Extract QE per channel from abs_QE array
                    QE_per_channel_frame = np.zeros([len(dye_names), len(pixel_colours)])
                    for j, dye in enumerate(dye_names):
                        for i, colour in enumerate(pixel_colours):
                            # Extract QE value from any pixel of this type (they're all the same)
                            mask_indices = np.where(masks[colour])
                            if len(mask_indices[0]) > 0:
                                QE_per_channel_frame[j, i] = abs_QE[mask_indices[0][0], mask_indices[1][0], j]
                    background_photons_matrix_frame = background_photons_matrix
    
                n_photons_hitting_detector = np.zeros([w, h, len(dye_names)], dtype=int)
                n_photoelectrons = np.zeros_like(n_photons_hitting_detector)
    
                for j, dye in enumerate(dye_names):
                    try:
                        n_photons_this_frame = (
                            n_photons[dye][frame]
                            if hasattr(n_photons[dye], "__getitem__")
                            else n_photons[dye]
                        )
                    except (IndexError, TypeError):
                        n_photons_this_frame = n_photons[dye]
    
                    if n_photons_this_frame > 0:
                        try:
                            x0 = (
                                x0y0[dye][frame, 0, :]
                                if x0y0[dye].ndim > 1
                                else x0y0[dye][0, :]
                            )
                            y0 = (
                                x0y0[dye][frame, 1, :]
                                if x0y0[dye].ndim > 1
                                else x0y0[dye][1, :]
                            )
                        except (IndexError, TypeError):
                            x0, y0 = x0y0[dye][0, :], x0y0[dye][1, :]
    
                        # Convert positions from nm to pixels
                        x0_pixels = x0 / pixel_size
                        y0_pixels = y0 / pixel_size
    
                        # Ensure n_photons array matches the number of molecules
                        if hasattr(x0, "__len__") and len(x0) > 1:
                            # Multiple molecules - create array with same photon count for each
                            n_photons_array = np.full(len(x0), int(n_photons_this_frame))
                        else:
                            # Single molecule or scalar position
                            n_photons_array = np.array([int(n_photons_this_frame)])
    
                        photon_spatial_pdf = self.psf.gen_spatial_PSF(
                            x,
                            sigma_x,
                            sigma_y,
                            x0_pixels,
                            y0_pixels,
                            n_photons_array,
                            relative_QE,
                        )
    
                        # Generate photons hitting detector (includes background)
                        n_photons_total = self.psf.gen_photons_hitting_detector(
                            photon_spatial_pdf, background_photons_matrix_frame[:, :, j]
                        )
                        n_photons_hitting_detector[:, :, j] = n_photons_total
    
                        # CRITICAL FIX: Split photons by Bayer pattern FIRST, then apply QE
                        # Each photon can only hit ONE pixel type (B, G, or R)
    
                        # Optimization: Check if all QE values are identical (e.g., Standard camera)
                        # If so, we can apply QE directly without splitting by channel
                        QE_values = QE_per_channel_frame[j, :]
                        all_QE_equal = np.allclose(QE_values, QE_values[0], rtol=1e-9)
    
                        if all_QE_equal:
                            # Fast path: All channels have same QE
                            # Apply uniform QE to all photons regardless of pixel type
                            n_photoelectrons[:, :, j] = self.psf.gen_photoelectrons(
                                n_photons_total.astype(int), QE_values[0]
                            )
                        else:
                            # Accurate path: Different QE per channel
                            # OPTIMIZATION: Vectorize photoelectron generation across all channels at once
                            # Broadcast photons across channels using pre-computed mask_stack
                            # n_photons_total: (w, h)
                            # mask_stack: (w, h, 3)
                            # Result: (w, h, 3) - photons per channel
                            n_photons_per_channel = (n_photons_total[:, :, np.newaxis] * mask_stack).astype(int)
    
                            # Generate photoelectrons for all channels simultaneously
                            # NumPy's binomial can broadcast: n and p can have different shapes
                            # n_photons_per_channel: (w, h, 3)
                            # QE_per_channel_frame[j, :]: (3,) -> broadcasts to (w, h, 3)
                            photoelectrons_per_channel = self.psf.gen_photoelectrons(
                                n_photons_per_channel,
                                QE_per_channel_frame[j, :]  # Shape: (3,), broadcasts across (w, h, 3)
                            )
    
                            # Sum photoelectrons from all channels (each photon contributed to only one)
                            n_photoelectrons[:, :, j] = np.sum(photoelectrons_per_channel, axis=-1)
    
                bayer_image[frame, :, :] = self.psf.photoelectrons_to_image(
                    np.sum(n_photoelectrons, axis=-1), gain, offset, variance
                )

                if return_normal_image:
                    # Generate normal image using overall QY instead of channel-specific
                    n_photons_frame_total = np.sum(n_photons_hitting_detector, axis=-1)
                    overall_QY_frame = np.sum(QE_per_channel_frame) / len(pixel_colours)  # Average QY
                    n_photoelectrons_normal = self.psf.gen_photoelectrons(
                        n_photons_frame_total.astype(int),
                        overall_QY_frame
                    )

                    if return_photoelectrons:
                        # Return raw photoelectrons (ground truth for demosaicing validation)
                        normal_image[frame, :, :] = n_photoelectrons_normal
                    else:
                        # Convert to ADU (standard output)
                        normal_image[frame, :, :] = self.psf.photoelectrons_to_image(
                            n_photoelectrons_normal, gain, offset, variance
                        )

        # Check for bit depth overflow and automatically scale to appropriate bit depth
        max_value = np.max(bayer_image)
        min_value = np.min(bayer_image)

        # Determine appropriate bit depth based on actual data range
        if max_value > 65535 or min_value < 0:
            # Values exceed uint16 range, use float32 for full dynamic range
            print(
                f"WARNING: Pixel values exceed uint16 range (min: {min_value:.1f}, max: {max_value:.1f})"
            )
            print(
                "Automatically using float32 bit depth to preserve high photon count data"
            )
            bayer_image = bayer_image.astype(np.float32)
        elif max_value > 255:
            # Values exceed uint8 but fit in uint16
            if max_value <= 65535:
                bayer_image = bayer_image.astype(np.uint16)
        else:
            # Values fit in uint8
            bayer_image = bayer_image.astype(np.uint8)

        # Apply smoothing
        smoothing_args = smoothing_function.args
        smoothing_args[smoothing_function.data_arg] = bayer_image
        smoothed_image = smoothing_function.smoothing_function(**smoothing_args)

        if return_normal_image:
            return (
                np.squeeze(bayer_image),
                np.squeeze(smoothed_image),
                np.squeeze(normal_image),
            )
        else:
            return np.squeeze(bayer_image), np.squeeze(smoothed_image), None

    def test_simulation_method(
        self,
        dye: str,
        filters: List[str],
        wavelength: np.ndarray,
        camera_parameters: Dict[str, Any],
        save_folder: str,
        n_photon_space: np.ndarray,
        smoothing_function,
        strategy: FittingStrategy,
        starting_flag: str = "simulation_",
        config: Optional[SimulationConfig] = None,
        single_dye_spectrum: Optional[np.ndarray] = None,
        nile_red_wavelength: Optional[float] = None,
        overwrite: bool = True,
    ) -> None:
        """
        Unified method for all fitting strategies, replacing the 4 duplicate methods.

        This single method handles all fitting approaches through the strategy parameter:
        - FittingStrategy.STANDARD: Direct fitting with Bayer patterns
        - FittingStrategy.DEMOSAIC: Demosaic then fit
        - FittingStrategy.DEMOSAIC_FAST: Fast demosaic fitting
        - FittingStrategy.DEMOSAIC_IG: Initial grayscale fit then colour refinement

        Args:
            nile_red_wavelength: If provided, will add wavelength fitting columns to raw results
            overwrite: If True, overwrite existing results. If False, skip already completed
                      photon levels and continue from where simulation crashed (default: True)
        """
        if config is None:
            config = SimulationConfig()

        # Import required modules
        import polars as pl
        import SpectralFunctions

        S_F = SpectralFunctions.Spectral_Funcs()

        # Validate camera parameters
        camera_params = CameraParameters.validate_and_create(camera_parameters)
        self._validate_inputs(wavelength, camera_parameters, None, {})

        # Get dye properties
        if "simulated_" in dye:
            # For simulated spectra, apply filters before calculating average wavelength
            # This ensures the PSF width matches what actually reaches the camera
            filter_spectra = S_F.get_dye_or_filter_data(
                names=filters, wavelength=wavelength, dye_or_filter=False
            )
            # Apply filters to the spectrum
            total_filter_transmission = np.prod(filter_spectra, axis=0)
            filtered_spectrum = single_dye_spectrum * total_filter_transmission

            # Now calculate average wavelength from the FILTERED spectrum
            average_emission_wavelength, dye_pixel_efficiency = (
                self.spectral.get_pixel_fractions_rawspectra(
                    filtered_spectrum, wavelength, camera_params.pixel_QYs
                )
            )
        else:
            average_emission_wavelength, dye_pixel_efficiency = (
                self.spectral.get_pixel_fractions_dye_and_filters(
                    [dye], filters, wavelength, camera_params.pixel_QYs,
                    normalized=False  # Use absolute QE for photoelectron generation
                )
            )

        # Setup simulation parameters
        x0, y0, setup_data = self._setup_simulation_parameters(
            camera_params,
            config,
            dye_pixel_efficiency,
            average_emission_wavelength,
            dye,
        )

        # Create position dictionary
        x0y0 = {"dye": np.zeros([config.n_bootstrap, 2, 1])}
        x0y0["dye"][:, :, :] = np.array([[x0, y0]]).T

        # Define analysis parameters based on strategy
        if strategy == FittingStrategy.STANDARD:
            analysis_save_params = [
                "xc",
                "yc",
                "s_x",
                "s_y",
                "bg_B",
                "bg_G",
                "bg_R",
                "A_B",
                "A_G",
                "A_R",
                "chi_sqr",
                "frame",
            ]
        else:
            analysis_save_params = [
                "xc",
                "yc",
                "s_x",
                "s_y",
                "bg_B",
                "bg_G",
                "bg_R",
                "A_B",
                "A_G",
                "A_R",
                "chi_sqr",
                "frame",
            ]

        # Save expected parameters (only if overwriting or doesn't exist)
        parameters_to_save = analysis_save_params[:-2]
        real_params = pl.DataFrame(
            data=np.expand_dims(setup_data["expected_parameters"], 0),
            schema=parameters_to_save,
        )
        dyestr = dye.replace("/", "-")
        input_params_path = os.path.join(
            save_folder,
            f"{starting_flag}LM_method_{dyestr}_fittesting_input_parameters.csv",
        )
        if overwrite or not os.path.exists(input_params_path):
            real_params.write_csv(input_params_path)

        # Save ground truth positions for standard method
        # CRITICAL: Always save ground truth to ensure it matches the x0, y0 positions used in simulation
        # The x0, y0 are randomly generated each time this function runs, so the file must be updated
        if strategy == FittingStrategy.STANDARD:
            X0Y0 = {"x0": x0, "y0": y0}
            groundtruth_path = os.path.join(
                save_folder,
                f"{starting_flag}LM_method_{dyestr}_fittesting_input_groundtruthpositions.csv",
            )
            # Always write ground truth file to match current x0, y0 positions
            pl.DataFrame(X0Y0).write_csv(groundtruth_path)

        # Initialize results arrays
        fit_RMSE_mean = np.zeros([len(analysis_save_params) - 1, len(n_photon_space)])
        fit_std = np.zeros([len(analysis_save_params) - 1, len(n_photon_space)])

        # Check for existing results and determine which photon levels to process
        completed_photon_levels = set()
        start_index = 0

        if config.save_raw_results:
            # Define HDF5 database path for raw results
            raw_results_h5_path = os.path.join(
                save_folder,
                f"{starting_flag}LM_method_{dyestr}_rawresults.h5",
            )

            # Handle overwrite mode
            if overwrite and os.path.exists(raw_results_h5_path):
                print(f"Overwrite=True: Deleting existing results file: {os.path.basename(raw_results_h5_path)}")
                os.remove(raw_results_h5_path)

            # Check if we should skip existing results
            if not overwrite and os.path.exists(raw_results_h5_path):
                import pandas as pd
                try:
                    # Read existing HDF5 file to find completed photon levels
                    existing_data = pd.read_hdf(raw_results_h5_path, key="data")
                    if "photon_level" in existing_data.columns:
                        completed_photon_levels = set(existing_data["photon_level"].unique())
                        print(f"Found existing results with {len(completed_photon_levels)} completed photon levels")
                        print(f"Completed levels: {sorted(completed_photon_levels)}")

                        # If all photon levels are complete, notify and return
                        if len(completed_photon_levels) >= len(n_photon_space):
                            print(f"All {len(n_photon_space)} photon levels already completed. Use overwrite=True to rerun.")
                            return
                except Exception as e:
                    print(f"Warning: Could not read existing HDF5 file: {e}")
                    print("Starting fresh simulation...")
                    completed_photon_levels = set()

            # Save/update photon levels CSV
            photon_levels_df = pl.DataFrame({
                "photon_level_index": np.arange(len(n_photon_space)),
                "n_photons": n_photon_space,
            })

            # Only write CSV if overwriting or if it doesn't exist
            photon_levels_csv_path = os.path.join(
                save_folder,
                f"{starting_flag}LM_method_{dyestr}_photon_levels.csv",
            )
            if overwrite or not os.path.exists(photon_levels_csv_path):
                photon_levels_df.write_csv(photon_levels_csv_path)

        start = time.time()

        # OPTIMIZATION: Pre-compute full spectrum outside loop (constant across photon levels)
        # This avoids repeated database queries and array operations (200× per simulation)
        if config.use_stochastic_photons:
            if single_dye_spectrum is not None:
                # Use provided spectrum (already filtered)
                full_spectrum_template = filtered_spectrum
            else:
                # Get spectrum from database ONCE
                dye_spectrum = S_F.get_dye_or_filter_data(
                    names=[dye], wavelength=wavelength, dye_or_filter=True
                )
                filter_spectra = S_F.get_dye_or_filter_data(
                    names=filters, wavelength=wavelength, dye_or_filter=False
                )
                total_filter_transmission = np.prod(filter_spectra, axis=0)
                full_spectrum_template = dye_spectrum[0] * total_filter_transmission

        # Process each photon count
        for i, n_photon in enumerate(n_photon_space):
            # Skip if this photon level was already completed
            if i in completed_photon_levels:
                print(f"Skipping photon level {i} ({n_photon} photons) - already completed", flush=True)
                continue

            n_photons = {"dye": np.full(config.n_bootstrap, n_photon)}

            # Stochastic photon sampling for realistic shot noise
            if config.use_stochastic_photons:
                # Use pre-computed spectrum
                full_spectrum = full_spectrum_template

                # Generate stochastic colour ratios and wavelengths for bootstrap samples
                mean_wavelengths_bootstrap, colour_ratios_bootstrap = (
                    S_F.generate_bootstrap_colour_ratios(
                        full_spectrum,
                        wavelength,
                        camera_params.pixel_QYs,
                        n_photons_per_image=int(n_photon),
                        n_bootstrap=config.n_bootstrap,
                        pixel_order=camera_params.pixel_order,
                        pixel_order_indices=camera_params.pixel_order_indices,
                        random_state=np.random.default_rng(),
                    )
                )

                # Use stochastic values instead of deterministic
                average_emission_wavelength_for_this_photon = mean_wavelengths_bootstrap
                dye_pixel_efficiency_for_this_photon = colour_ratios_bootstrap
            else:
                # Use deterministic values (backwards compatibility)
                average_emission_wavelength_for_this_photon = average_emission_wavelength
                dye_pixel_efficiency_for_this_photon = dye_pixel_efficiency

            # Generate images
            bayer_image, smoothed_image, _ = self.gen_camera_image_stack(
                camera_parameters,
                wavelength,
                average_emission_wavelength_for_this_photon,
                dye_pixel_efficiency_for_this_photon,
                n_photons,
                x0y0,
                smoothing_function=smoothing_function,
                background_photons=config.background_photons,
                background_colour=config.background_colour,
                NA=config.NA,
                pixel_size=config.pixel_size,
                return_normal_image=False,
            )

            # Save raw images if requested
            if config.saverawimages:
                filename = f"{starting_flag}LM_method_{dyestr}_{str(np.around(n_photon, 2)).replace('.', 'p').zfill(10)}_rawbayerimage.tiff"
                self.io.write_tiff(bayer_image, os.path.join(save_folder, filename))

            # Prepare fitting data
            photoelectron_data, smoothed_data, grayscale_data = (
                self._prepare_fitting_data(
                    bayer_image, smoothed_image, camera_params, strategy, config
                )
            )

            # Compute error maps
            weights_map, weights_grayscale_map = self._compute_error_maps(
                smoothed_data,
                grayscale_data[1] if grayscale_data else None,
                camera_params,
            )

            # Perform fitting
            fit_results = self._perform_fitting(
                strategy,
                photoelectron_data,
                smoothed_data,
                weights_map,
                grayscale_data,
                camera_params,
                config,
            )

            # Save raw results if requested
            if config.save_raw_results:
                if config.subtractx0y0:
                    fit_results["xc"] = fit_results["xc"] - (x0 / config.pixel_size)
                    fit_results["yc"] = fit_results["yc"] - (y0 / config.pixel_size)

                # Add Nile Red wavelength fitting if wavelength is provided
                if nile_red_wavelength is not None:
                    fit_results = self._add_nile_red_wavelength_fits(
                        fit_results,
                        nile_red_wavelength,
                        camera_params,
                        camera_parameters,
                        wavelength,
                        filters,
                        config,
                    )

                # Add photon_level column to track which photon count this data belongs to
                fit_results["photon_level"] = i

                # Save to HDF5 database using IOFunctions
                # Note: _fit_standard already creates photons and background_photons columns
                # and normalizes A_R/G/B and bg_R/G/B, so we pass normalise_photons=False
                # to avoid double normalization
                # Append if: (1) not the first iteration OR (2) continuing from previous run
                should_append = (i > 0) or (len(completed_photon_levels) > 0)
                self.io._write_h5_database(
                    fit_results,
                    raw_results_h5_path,
                    append=should_append,
                    normalise_photons=False,  # Already normalized in _fit_standard
                    verbose=config.verbose,
                )

            # Compute statistics for this photon count
            fit_RMSE_mean[:, i], fit_std[:, i] = self._compute_fit_statistics(
                fit_results, setup_data, config, analysis_save_params
            )

            # Progress update - use carriage return to update in place
            elapsed = (time.time() - start) / 60.0
            print(
                f"Analysed photon flux {i + 1}/{len(n_photon_space)}    Time elapsed: {elapsed:.3f} min",
                end="\r",
                flush=True,
            )

        # Clear the progress line and show completion
        total_elapsed = (time.time() - start) / 60.0
        print(
            f"\nCompleted analysis of {len(n_photon_space)} photon flux values    Total time: {total_elapsed:.3f} min",
            flush=True,
        )

        # Save final results
        save_params = analysis_save_params[:-2] + ["colour_distance"]
        self.io.save_simulation_results(
            save_folder,
            starting_flag,
            save_params,
            n_photon_space,
            fit_RMSE_mean,
            fit_std,
            config.pixel_size,
            config.NA,
            config.background_photons,
            "LM_fitting",
            "Gaussian_Smoother",
            smoothing_function.extent,
            dye,
        )

        logger.info(f"Simulation completed for strategy {strategy.value}")


# Compatibility methods - these delegate to the unified method with appropriate strategies
class MultiC_Sim_Funcs_Compatibility(MultiC_Sim_Funcs_Refactored):
    """Compatibility layer providing the original method names."""

    def test_fit_method(self, *args, **kwargs):
        """Compatibility wrapper for original test_fit_method."""
        return self.test_simulation_method(
            *args, strategy=FittingStrategy.STANDARD, **kwargs
        )

    def test_demosaic_fit_method(self, *args, **kwargs):
        """
        Compatibility wrapper for original test_demosaic_fit_method.

        Args:
            *args: Positional arguments passed to test_simulation_method
            **kwargs: Keyword arguments passed to test_simulation_method

        Returns:
            None: Delegates to test_simulation_method with DEMOSAIC strategy
        """
        return self.test_simulation_method(
            *args, strategy=FittingStrategy.DEMOSAIC, **kwargs
        )

    def test_demosaic_fast_fit_method(self, *args, **kwargs):
        """
        Compatibility wrapper for original test_demosaic_fast_fit_method.

        Args:
            *args: Positional arguments passed to test_simulation_method
            **kwargs: Keyword arguments passed to test_simulation_method

        Returns:
            None: Delegates to test_simulation_method with DEMOSAIC_FAST strategy
        """
        return self.test_simulation_method(
            *args, strategy=FittingStrategy.DEMOSAIC_FAST, **kwargs
        )

    def test_demosaic_IG_fit_method(self, *args, **kwargs):
        """
        Compatibility wrapper for original test_demosaic_IG_fit_method.

        Args:
            *args: Positional arguments passed to test_simulation_method
            **kwargs: Keyword arguments passed to test_simulation_method

        Returns:
            None: Delegates to test_simulation_method with DEMOSAIC_IG strategy
        """
        return self.test_simulation_method(
            *args, strategy=FittingStrategy.DEMOSAIC_IG, **kwargs
        )


# Main class for external use - provides both new and legacy interfaces
# Module-level standalone function for parallel Nile Red wavelength fitting (must be pickleable)
def _fit_nile_red_wavelength_standalone(
    rgb: np.ndarray,
    sigma_x: float,
    sigma_y: float,
    rgb_err: np.ndarray,
    sigma_x_err: float,
    sigma_y_err: float,
    filter_spectra: np.ndarray,
    wavelength_array: np.ndarray,
    pixel_QYs: np.ndarray,
    NA: float,
    total_photons: Optional[float] = None,
    background_photons: Optional[float] = None,
    wavelength_bounds: Tuple[float, float] = (500.0, 750.0),
) -> Tuple[float, float]:
    """
    Standalone function for fitting Nile Red wavelength from a single localization.

    Must be a module-level function (not a method) to be pickleable for multiprocessing.

    Args:
        rgb: Normalized [R, G, B] intensities
        sigma_x, sigma_y: PSF widths in nm
        rgb_err, sigma_x_err, sigma_y_err: Errors on measurements
        filter_spectra, wavelength_array, pixel_QYs: Optical system parameters
        NA: Numerical aperture
        total_photons: Fitted total photon count (for SNR-based error inflation)
        background_photons: Fitted background photon count (for SNR-based error inflation)
        wavelength_bounds: Search range for wavelength (nm)

    Returns:
        Tuple of (fitted_wavelength, wavelength_error)
        Returns (NaN, NaN) if fit fails
    """
    try:
        import NileRedFunctions

        nrf = NileRedFunctions.NileRed_Functions()

        wl, _ = nrf.fit_nile_red_wavelength(
            observed_rgb=rgb,
            observed_sigma_x=sigma_x,
            observed_sigma_y=sigma_y,
            rgb_errors=rgb_err,
            sigma_x_error=sigma_x_err,
            sigma_y_error=sigma_y_err,
            filter_spectra=filter_spectra,
            wavelength_array=wavelength_array,
            pixel_QYs=pixel_QYs,
            NA=NA,
            wavelength_bounds=wavelength_bounds,
            total_photons=total_photons,
            background_photons=background_photons,
            apply_snr_inflation=True if total_photons is not None else False,
        )
        # TODO: Implement proper error estimation on wavelength
        return (wl, np.nan)
    except Exception as e:
        return (np.nan, np.nan)


class MultiC_Sim_Funcs(MultiC_Sim_Funcs_Compatibility):
    """
    Main class for multicolour single-molecule localization microscopy simulation and analysis.

    This class consolidates the 4 massive duplicate methods from the original implementation:
    - test_fit_method (1939 lines)
    - test_demosaic_fit_method (1562 lines)
    - test_demosaic_fast_fit_method (1190 lines)
    - test_demosaic_IG_fit_method (788 lines)

    Into a single parameterized method with ~40% code reduction while maintaining
    full backward compatibility through the compatibility layer.

    The class provides both new and legacy interfaces:

    New Interface (Recommended):
        sim = MultiC_Sim_Funcs()
        sim.test_simulation_method(dye, filters, wavelength, camera_parameters,
                                 save_folder, n_photon_space, smoothing_function,
                                 strategy=FittingStrategy.DEMOSAIC_IG)

    Legacy Interface (Backward Compatible):
        sim = MultiC_Sim_Funcs()
        sim.test_demosaic_IG_fit_method(dye, filters, wavelength, ...)

    Key Features:
        - Unified simulation method supporting all 4 fitting strategies
        - Comprehensive error handling and input validation
        - Memory-optimised processing with garbage collection
        - Parallel fitting using multiprocessing
        - Detailed statistical analysis of fitting performance
        - Configurable simulation parameters via SimulationConfig
        - Support for both real and simulated fluorophore spectra
    """

    def _filter_dyes_by_photons(
        self,
        potential_dyes: List[str],
        single_molecule_dyes: np.ndarray,
        filters: List[str],
        camera_parameters: Dict[str, Any],
        wavelength: np.ndarray,
        min_photons_per_100ms: int = 500,
        integration_time_ms: float = 100,
        excitation_power_scaling: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Filter candidate dyes by expected photon yield.

        Args:
            potential_dyes: List of dye names to consider
            single_molecule_dyes: Array with columns [name, photons_per_100ms]
                Column 1 contains photons per 100ms under standard conditions
            filters: Filter names for transmission calculation
            camera_parameters: Camera parameters with pixel_QYs
            wavelength: Wavelength array for spectral calculations
            min_photons_per_100ms: Minimum expected photons at detector (default: 500)
            integration_time_ms: Integration time in ms (default: 100)
            excitation_power_scaling: Scaling factor for photon counts (default: 1.0)

        Returns:
            dict: {
                'viable_dyes': list of dye names passing threshold,
                'expected_photons': dict mapping dye_name -> source photons (before camera QE),
                'expected_photons_at_detector': dict mapping dye_name -> detector photons (after camera QE),
                'effective_qe': dict mapping dye_name -> effective camera QE,
                'photons_per_100ms': dict mapping dye_name -> photons_per_100ms
            }
        """
        # Calculate filter transmission
        filter_spectrum = np.prod(
            self.spectral.get_dye_or_filter_data(
                names=filters, wavelength=wavelength, dye_or_filter=False
            ),
            axis=0,
        )

        # Get dye spectra
        dye_spectra = self.spectral.get_dye_or_filter_data(
            potential_dyes, wavelength=wavelength
        )

        # Normalize dye spectra at detector
        dye_at_detector_spectra = np.array(
            np.multiply(dye_spectra, filter_spectrum).T
            / np.sum(np.multiply(dye_spectra, filter_spectrum), axis=1)
        ).T

        # Bayer pattern: B, G, G, R (4 pixels)
        pixel_QYs_bayer = np.vstack(
            [
                camera_parameters["pixel_QYs"][0, :],  # B
                camera_parameters["pixel_QYs"][1, :],  # G
                camera_parameters["pixel_QYs"][1, :],  # G
                camera_parameters["pixel_QYs"][2, :],  # R
            ]
        )

        viable_dyes = []
        expected_photons = {}
        effective_qe = {}
        photons_per_100ms_dict = {}

        for dye_idx, dye_name in enumerate(potential_dyes):
            # Find dye in single_molecule_dyes
            idx = np.where(single_molecule_dyes[:, 0] == dye_name)[0]

            if len(idx) == 0:
                logger.warning(
                    f"{dye_name} not found in single_molecule_dyes array, skipping"
                )
                continue

            # Get photons per 100ms (column 1 of single_molecule_dyes)
            photons_per_100ms = float(single_molecule_dyes[idx[0], 1])
            photons_per_100ms_dict[dye_name] = photons_per_100ms

            # Calculate effective QE (average over Bayer pattern)
            dye_at_detector = dye_at_detector_spectra[dye_idx, :]
            pixel_efficiency_bayer = np.dot(dye_at_detector, pixel_QYs_bayer.T)
            eff_qe = np.mean(pixel_efficiency_bayer)
            effective_qe[dye_name] = eff_qe

            # Expected photons at detector = photons_per_100ms × effective_QE × power_scaling × (integration_time / 100ms)
            exp_photons_at_detector = (
                photons_per_100ms * eff_qe * excitation_power_scaling * (integration_time_ms / 100.0)
            )

            # Source photons (before camera QE) = photons_per_100ms × power_scaling × (integration_time / 100ms)
            # This is what we pass to gen_camera_image_stack, which applies camera QE internally
            source_photons = photons_per_100ms * excitation_power_scaling * (integration_time_ms / 100.0)

            expected_photons[dye_name] = source_photons

            # Check threshold using photons at detector
            if exp_photons_at_detector >= min_photons_per_100ms:
                viable_dyes.append(dye_name)

        return {
            "viable_dyes": viable_dyes,
            "expected_photons": expected_photons,  # Source photons for simulation
            "expected_photons_at_detector": {dye: phot * eff_qe * excitation_power_scaling * (integration_time_ms / 100.0)
                                            for dye, phot in photons_per_100ms_dict.items()},
            "effective_qe": effective_qe,
            "photons_per_100ms": photons_per_100ms_dict,
        }

    def _fit_dye_gaussian(self, color_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Fit 2D Gaussian to (A_R, A_G) color coordinates.

        Args:
            color_data: Output from _simulate_dye_color_distributions

        Returns:
            dict: {
                'mean': np.ndarray of shape (2,) - [mean_A_R, mean_A_G],
                'covariance': np.ndarray of shape (2, 2) - covariance matrix,
                'std_A_R': float,
                'std_A_G': float,
                'correlation': float - correlation coefficient,
                'n_valid': int - number of valid (non-NaN) points used
            }
        """
        A_R = color_data['A_R']
        A_G = color_data['A_G']

        # Filter out NaN values (failed fits)
        valid_mask = ~(np.isnan(A_R) | np.isnan(A_G))
        A_R_valid = A_R[valid_mask]
        A_G_valid = A_G[valid_mask]

        if len(A_R_valid) == 0:
            raise ValueError("No valid fits - all simulations failed!")

        # Stack into (N, 2) array
        X = np.vstack([A_R_valid, A_G_valid]).T

        # Calculate mean and covariance
        mean = np.mean(X, axis=0)
        covariance = np.cov(X.T)

        # Add regularization to ensure positive definiteness
        # This is necessary for very high photon counts where variance becomes extremely small
        # We add a small diagonal term (ridge regularization)
        min_variance = 1e-5
        covariance[0, 0] += min_variance
        covariance[1, 1] += min_variance

        std_A_R = np.sqrt(covariance[0, 0])
        std_A_G = np.sqrt(covariance[1, 1])
        correlation = covariance[0, 1] / (std_A_R * std_A_G) if std_A_R * std_A_G > 0 else 0

        return {
            'mean': mean,
            'covariance': covariance,
            'std_A_R': std_A_R,
            'std_A_G': std_A_G,
            'correlation': correlation,
            'n_valid': len(A_R_valid)
        }

    def _calculate_dye_separability(
        self,
        dye_gaussians: Dict[str, Dict[str, Any]],
        dye_names: List[str],
        n_monte_carlo: int = 10000
    ) -> Dict[str, Any]:
        """
        Calculate pairwise and overall misidentification rates for dye combinations.

        Uses the analytical approach from SM_extractionfunctions.calculate_analytical_misidentification.

        Args:
            dye_gaussians: Dict mapping dye_name -> gaussian_params from _fit_dye_gaussian
            dye_names: Ordered list of dye names
            n_monte_carlo: Number of Monte Carlo samples for overlap calculation

        Returns:
            dict: {
                'confusion_matrix': np.ndarray of shape (n_dyes, n_dyes),
                    Entry [i, j] = P(classified as j | true dye i),
                'accuracy_per_dye': np.ndarray of shape (n_dyes,),
                'overall_accuracy': float,
                'pairwise_separability': dict mapping (dye_i, dye_j) -> accuracy
            }
        """
        import SM_extractionfunctions

        n_dyes = len(dye_names)

        # Prepare arrays for GMM
        means = np.array([dye_gaussians[dye]['mean'] for dye in dye_names])
        covariances = np.array([dye_gaussians[dye]['covariance'] for dye in dye_names])
        weights = np.ones(n_dyes) / n_dyes  # Equal prior

        # Use SM_extractionfunctions analytical method
        SM_E = SM_extractionfunctions.extract_SMs()
        stats = SM_E.calculate_analytical_misidentification(
            fixed_means=means,
            covariances=covariances,
            weights=weights,
            n_samples=n_monte_carlo,
            random_state=42
        )

        # Calculate per-dye accuracy (diagonal of confusion matrix)
        accuracy_per_dye = np.diag(stats['confusion_matrix'])

        # Calculate pairwise separability
        pairwise_separability = {}
        for i in range(n_dyes):
            for j in range(i + 1, n_dyes):
                # For pair (i, j), accuracy = mean of diagonal elements in 2x2 submatrix
                conf = stats['confusion_matrix'][[i, j], :][:, [i, j]]
                pairwise_acc = np.mean(np.diag(conf))
                pairwise_separability[(dye_names[i], dye_names[j])] = pairwise_acc

        stats['accuracy_per_dye'] = accuracy_per_dye
        stats['pairwise_separability'] = pairwise_separability
        stats['dye_names'] = dye_names

        return stats

    def _simulate_dye_color_distributions(
        self,
        dye_name: str,
        filters: List[str],
        camera_parameters: Dict[str, Any],
        wavelength: np.ndarray,
        expected_photons: float,
        n_simulations: int = 1000,
        background_photons: float = 4,
        NA: float = 1.49,
        pixel_size: float = 69,
        image_dims: int = 12,
        smoothing_function=None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate camera images for a single dye and extract (A_R, A_G) coordinates.

        This method uses the existing gen_camera_image_stack to simulate realistic
        camera images with proper Bayer pattern and noise, then extracts color
        coordinates for separability analysis.

        **Stochastic Photon Sampling (Nov 2025):**
        Two levels of realistic shot noise for each frame:
        1. Poisson photon count variation - each molecule emits a different
           number of photons drawn from Poisson(expected_photons)
        2. Spectral sampling - for each frame's photon count, sample which
           wavelengths are emitted from the dye's emission spectrum

        This generates realistic variation in:
        - Total photon count per frame (Poisson noise)
        - Per-frame average wavelength (affects PSF width variation)
        - Per-frame BGR color ratios (shot noise in detected color)

        This accounts for the full stochastic nature of photon emission,
        beyond just spatial Poisson noise in the PSF.

        Args:
            dye_name: Name of the dye
            filters: Filter names
            camera_parameters: Camera parameters
            wavelength: Wavelength array
            expected_photons: Expected photon count for this dye
            n_simulations: Number of simulated molecules (default: 1000)
            background_photons: Background photon count (default: 4)
            NA: Numerical aperture (default: 1.49)
            pixel_size: Pixel size in nm (default: 69)
            image_dims: Image dimension in pixels (default: 12)
            smoothing_function: PSF smoothing function (default: None)

        Returns:
            dict: {
                'A_R': np.ndarray of shape (n_simulations,),
                'A_G': np.ndarray of shape (n_simulations,),
                'A_R_err': np.ndarray of shape (n_simulations,),
                'A_G_err': np.ndarray of shape (n_simulations,),
                'photons': np.ndarray of shape (n_simulations,)
            }
        """
        # Get dye spectral properties
        filter_spectrum = np.prod(
            self.spectral.get_dye_or_filter_data(
                names=filters, wavelength=wavelength, dye_or_filter=False
            ),
            axis=0,
        )

        dye_spectrum = self.spectral.get_dye_or_filter_data([dye_name], wavelength=wavelength)[0]
        dye_at_detector = dye_spectrum * filter_spectrum / np.sum(dye_spectrum * filter_spectrum)

        # Generate random positions in center region
        max_pos = pixel_size * image_dims
        center = max_pos / 2
        position_range = pixel_size  # Keep within central 2x2 pixels

        np.random.seed(42)  # Reproducibility
        x0 = np.random.uniform(center - position_range, center + position_range, n_simulations)
        y0 = np.random.uniform(center - position_range, center + position_range, n_simulations)

        # STOCHASTIC PHOTON SAMPLING:
        # Two levels of stochasticity for realistic shot noise:
        # 1. Poisson photon count variation (each molecule emits different # photons)
        # 2. Spectral sampling (which wavelengths are emitted for those photons)

        rng = np.random.default_rng(42)  # Reproducible random state

        # Step 1: Poisson photon count sampling (vectorized, single call)
        # Each frame gets a different photon count drawn from Poisson distribution
        photon_counts_per_frame = rng.poisson(expected_photons, size=n_simulations)

        # Step 2: FAST spectral sampling using bulk bootstrap method
        # Strategy: Use generate_bootstrap_colour_ratios() for expected photon count,
        # which samples all photons at once (~N× faster than loop).
        # The Poisson variation in photon counts is handled by gen_camera_image_stack.

        pixel_QYs = camera_parameters["pixel_QYs"]
        pixel_order = camera_parameters["pixel_order"]

        # Bulk sample wavelengths and color ratios (FAST: single call)
        average_emission_wavelengths, bgr_ratios = self.spectral.generate_bootstrap_colour_ratios(
            dye_at_detector,
            wavelength,
            pixel_QYs,
            n_photons_per_image=int(expected_photons),
            n_bootstrap=n_simulations,
            pixel_order=pixel_order,
            pixel_order_indices={'B': 0, 'G': 1, 'R': 2},
            random_state=rng
        )

        # Note: We use expected_photons for spectral sampling, but actual
        # photon_counts_per_frame for image generation. This is accurate because:
        # 1. Spectral properties (wavelength, BGR ratios) depend on which photons
        #    are emitted, not how many (central limit theorem applies)
        # 2. Shot noise in color ratios scales as ~1/√N, accurately captured
        # 3. Poisson variation in total count is applied in gen_camera_image_stack

        # Prepare inputs for gen_camera_image_stack
        n_photons = {dye_name: photon_counts_per_frame}  # Array of Poisson-sampled counts
        x0y0 = {dye_name: np.zeros([n_simulations, 2, 1])}
        x0y0[dye_name][:, 0, 0] = x0
        x0y0[dye_name][:, 1, 0] = y0

        # Convert BGR ratios to pixel efficiencies (quantum efficiencies per channel)
        # These represent the effective QE for each channel for this specific set of photons
        dye_pixel_efficiency = bgr_ratios  # Shape: (n_simulations, 3)

        # Generate images with per-frame wavelengths and color ratios
        data, _, _ = self.gen_camera_image_stack(
            camera_parameters,
            wavelength,
            average_emission_wavelengths,  # Array: per-frame wavelengths
            dye_pixel_efficiency,           # Array: per-frame BGR ratios
            n_photons,
            x0y0,
            smoothing_function,
            background_photons=background_photons,
            NA=NA,
            pixel_size=pixel_size,
            return_normal_image=False,
        )

        # Extract color coordinates (A_R, A_G, A_B) using actual fitting pipeline
        # This is critical for realistic separability estimates

        masks_3d_dict = camera_parameters["masks"]

        # Create 3D mask array for fitting (B, G, R channels)
        masks_3d = np.dstack([masks_3d_dict[x] for x in ["B", "G", "R"]])

        # Smooth all images at once (gaussian_filter_stack handles 3D arrays)
        smoothed_data = smoothing_function.smoothing_function(
            **{smoothing_function.data_arg: data, **smoothing_function.args}
        )

        # Prepare data for fitting
        puncta_tofit, smoothed_puncta_tofit, masks_tofit, weights_tofit = [], [], [], []
        relative_coords, planes = [], []

        for i in range(n_simulations):
            puncta_tofit.append(data[i, :, :])
            smoothed_puncta_tofit.append(smoothed_data[i, :, :])
            masks_tofit.append(masks_3d)
            weights_tofit.append(np.ones_like(data[i, :, :]))  # Uniform weights
            relative_coords.append((0, 0))
            planes.append(i)

        # Perform parallel fitting
        fit_results, fit_errors = self.image_analysis.fit_puncta_parallel_method(
            puncta_tofit,
            smoothed_puncta_tofit,
            weights_tofit,
            relative_coords,
            planes,
            IAF_FittingStrategy.STANDARD,
            masks=masks_tofit,
        )

        # Extract A_R, A_G, A_B from fit results
        # fit_results columns: [xc, yc, s_x, s_y, bg_B, bg_G, bg_R, A_B, A_G, A_R, chi_sqr, frame]
        # fit_errors columns: [xc_err, yc_err, s_x_err, s_y_err, bg_B_err, bg_G_err, bg_R_err, A_B_err, A_G_err, A_R_err]

        A_R = np.zeros(n_simulations)
        A_G = np.zeros(n_simulations)
        A_B = np.zeros(n_simulations)
        A_R_err = np.zeros(n_simulations)
        A_G_err = np.zeros(n_simulations)
        A_B_err = np.zeros(n_simulations)
        photons = np.zeros(n_simulations)

        for i in range(n_simulations):
            # Fit results: [..., bg_B, bg_G, bg_R, A_B, A_G, A_R, chi_sqr, frame]
            # Indices:     [ 0,   1,    2,    3,   4,   5,   6,   7,   8,    9,    10,       11]
            if not np.isnan(fit_results[i, 9]):  # Check A_R validity
                amp_B = fit_results[i, 7]
                amp_G = fit_results[i, 8]
                amp_R = fit_results[i, 9]

                total_amp = amp_R + amp_G + amp_B
                if total_amp > 0:
                    A_R[i] = amp_R / total_amp
                    A_G[i] = amp_G / total_amp
                    A_B[i] = amp_B / total_amp

                    # Propagate fitted errors (already corrected for sqrt transform in main code)
                    A_R_err[i] = fit_errors[i, 9] / total_amp if fit_errors[i, 9] > 0 else 0.01
                    A_G_err[i] = fit_errors[i, 8] / total_amp if fit_errors[i, 8] > 0 else 0.01
                    A_B_err[i] = fit_errors[i, 7] / total_amp if fit_errors[i, 7] > 0 else 0.01
                    photons[i] = total_amp
                else:
                    # Fit succeeded but amplitudes are zero - mark as failed
                    A_R[i] = np.nan
                    A_G[i] = np.nan
                    A_B[i] = np.nan
                    A_R_err[i] = np.nan
                    A_G_err[i] = np.nan
                    A_B_err[i] = np.nan
                    photons[i] = np.nan
            else:
                # Fit failed - mark as NaN to exclude from analysis
                A_R[i] = np.nan
                A_G[i] = np.nan
                A_B[i] = np.nan
                A_R_err[i] = np.nan
                A_G_err[i] = np.nan
                A_B_err[i] = np.nan
                photons[i] = np.nan

        return {
            'A_R': A_R,
            'A_G': A_G,
            'A_B': A_B,
            'A_R_err': A_R_err,
            'A_G_err': A_G_err,
            'A_B_err': A_B_err,
            'photons': photons
        }

    def plot_dye_selection_results(
        self,
        result: Dict[str, Any],
        save_path: str = None,
        show: bool = True,
        n_std: float = 2.0,
        figsize: tuple = (12, 10)
    ):
        """
        Plot results from optimal_dye_selector_simulated with confusion matrix.

        Creates a combined visualization showing:
        1. Color distribution scatter plot with Gaussian fits
        2. Confusion matrix heatmap

        Args:
            result: Output dict from optimal_dye_selector_simulated
            save_path: Path to save figure (default: None, don't save)
            show: Whether to display the figure (default: True)
            n_std: Number of standard deviations for ellipse (default: 2.0)
            figsize: Figure size (width, height) in inches

        Returns:
            fig, axes: Matplotlib figure and axes objects

        Example:
            >>> result = MSF.optimal_dye_selector_simulated(...)
            >>> MSF.plot_dye_selection_results(result, save_path='dye_selection.png')
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from PlottingBase import PublicationPlotter

        # Initialize plotter for consistent styling
        plotter = PublicationPlotter()

        # Create figure with subplots
        fig = plt.figure(figsize=figsize, dpi=plotter.config.DEFAULT_DPI)
        gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        ax_scatter = fig.add_subplot(gs[0, :])
        ax_conf = fig.add_subplot(gs[1, 0])
        ax_acc = fig.add_subplot(gs[1, 1])

        # Extract data
        dye_names = result['selected_dyes']
        dye_simulations = result['dye_simulations']
        dye_gaussians = result['dye_gaussians']
        conf_matrix = result['confusion_matrix']

        n_dyes = len(dye_names)
        colors = plt.cm.tab10(np.linspace(0, 1, n_dyes))

        # --- PLOT 1: Color distributions with Gaussian fits ---
        for idx, dye_name in enumerate(dye_names):
            color = colors[idx]

            # Get simulation data
            A_R = dye_simulations[dye_name]['A_R']
            A_G = dye_simulations[dye_name]['A_G']

            # Remove NaNs
            valid = ~(np.isnan(A_R) | np.isnan(A_G))
            A_R_valid = A_R[valid]
            A_G_valid = A_G[valid]

            # Plot scatter points
            ax_scatter.scatter(A_R_valid, A_G_valid, s=10, alpha=0.3, color=color,
                              label=f'{dye_name} (n={len(A_R_valid)})')

            # Get Gaussian parameters
            mean = dye_gaussians[dye_name]['mean']
            cov = dye_gaussians[dye_name]['covariance']

            # Plot mean
            ax_scatter.plot(mean[0], mean[1], 'o', color=color, markersize=8,
                           markeredgecolor='black', markeredgewidth=1.5)

            # Plot confidence ellipse
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * n_std * np.sqrt(eigenvalues)

            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                            edgecolor=color, facecolor='none', linewidth=2,
                            linestyle='--')
            ax_scatter.add_patch(ellipse)

        ax_scatter.set_xlabel('A_R (Red Amplitude Fraction)', fontsize=12)
        ax_scatter.set_ylabel('A_G (Green Amplitude Fraction)', fontsize=12)
        ax_scatter.set_title(f'Dye Color Distributions (Overall Accuracy: {result["overall_accuracy"]:.1%})',
                            fontsize=14, fontweight='bold')
        ax_scatter.legend(loc='best', fontsize=9)
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.set_aspect('equal', adjustable='box')

        # --- PLOT 2: Confusion matrix heatmap ---
        im = ax_conf.imshow(conf_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)

        # Add text annotations
        for i in range(n_dyes):
            for j in range(n_dyes):
                text_color = 'white' if conf_matrix[i, j] > 0.5 else 'black'
                text = ax_conf.text(j, i, f'{conf_matrix[i, j]:.2f}',
                                   ha="center", va="center", color=text_color, fontsize=9)

        # Axis labels
        ax_conf.set_xticks(np.arange(n_dyes))
        ax_conf.set_yticks(np.arange(n_dyes))
        ax_conf.set_xticklabels([dye[:10] for dye in dye_names], rotation=45, ha='right', fontsize=9)
        ax_conf.set_yticklabels([dye[:10] for dye in dye_names], fontsize=9)
        ax_conf.set_xlabel('Predicted Dye', fontsize=10)
        ax_conf.set_ylabel('True Dye', fontsize=10)
        ax_conf.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax_conf, fraction=0.046, pad=0.04)
        cbar.set_label('Classification Probability', rotation=270, labelpad=15, fontsize=9)

        # --- PLOT 3: Per-dye accuracy bar chart ---
        accuracy_per_dye = result['separability_stats']['accuracy_per_dye']
        bars = ax_acc.barh(range(n_dyes), accuracy_per_dye, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracy_per_dye)):
            ax_acc.text(acc + 0.01, i, f'{acc:.1%}', va='center', fontsize=9)

        ax_acc.set_yticks(range(n_dyes))
        ax_acc.set_yticklabels([dye[:10] for dye in dye_names], fontsize=9)
        ax_acc.set_xlabel('Classification Accuracy', fontsize=10)
        ax_acc.set_title('Per-Dye Accuracy', fontsize=12, fontweight='bold')
        ax_acc.set_xlim(0, 1.1)
        ax_acc.grid(True, alpha=0.3, axis='x')
        ax_acc.axvline(x=0.95, color='red', linestyle='--', linewidth=1, alpha=0.5, label='95% threshold')
        ax_acc.legend(fontsize=8)

        plt.tight_layout()

        # Use PlottingBase save/show methods for consistency (600 DPI for publication)
        plotter.save_or_show(fig, save_path=save_path, show=show, dpi=600)

        return fig, (ax_scatter, ax_conf, ax_acc)

    def plot_dye_color_distributions(
        self,
        dye_simulations: Dict[str, Dict[str, np.ndarray]],
        dye_gaussians: Dict[str, Dict[str, Any]],
        dye_names: List[str],
        save_path: str = None,
        show: bool = True,
        n_std: float = 2.0
    ):
        """
        Plot color coordinate distributions with fitted Gaussians overlaid.

        Creates scatter plots of (A_R, A_G) points with fitted Gaussian ellipses
        to visualize dye separability and validate Gaussian assumption.

        Args:
            dye_simulations: Dict mapping dye_name -> color_data from _simulate_dye_color_distributions
            dye_gaussians: Dict mapping dye_name -> gaussian_params from _fit_dye_gaussian
            dye_names: List of dye names to plot
            save_path: Path to save figure (default: None, don't save)
            show: Whether to display the figure (default: True)
            n_std: Number of standard deviations for ellipse (default: 2.0)

        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from PlottingBase import PublicationPlotter

        # Initialize plotter for consistent styling
        plotter = PublicationPlotter()

        n_dyes = len(dye_names)
        colors = plt.cm.tab10(np.linspace(0, 1, n_dyes))

        # Use plotter's one_column_plot for consistent publication styling
        fig, ax = plotter.one_column_plot(npanels=1, height=6)

        for idx, dye_name in enumerate(dye_names):
            color = colors[idx]

            # Get simulation data
            A_R = dye_simulations[dye_name]['A_R']
            A_G = dye_simulations[dye_name]['A_G']

            # Remove NaNs
            valid = ~(np.isnan(A_R) | np.isnan(A_G))
            A_R_valid = A_R[valid]
            A_G_valid = A_G[valid]

            # Plot scatter points
            ax.scatter(A_R_valid, A_G_valid, s=10, alpha=0.3, color=color,
                      label=f'{dye_name} (n={len(A_R_valid)})')

            # Get Gaussian parameters
            mean = dye_gaussians[dye_name]['mean']
            cov = dye_gaussians[dye_name]['covariance']

            # Plot mean
            ax.plot(mean[0], mean[1], 'o', color=color, markersize=8,
                   markeredgecolor='black', markeredgewidth=1.5)

            # Plot confidence ellipse
            # Calculate eigenvalues and eigenvectors for ellipse
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * n_std * np.sqrt(eigenvalues)

            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                            edgecolor=color, facecolor='none', linewidth=2,
                            linestyle='--', label=f'{dye_name} ({n_std}σ)')

            ax.add_patch(ellipse)

        ax.set_xlabel('A_R (Red Amplitude Fraction)', fontsize=12)
        ax.set_ylabel('A_G (Green Amplitude Fraction)', fontsize=12)
        ax.set_title('Dye Color Distributions with Gaussian Fits', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        # Use PlottingBase save/show methods for consistency (600 DPI for publication)
        plotter.save_or_show(fig, save_path=save_path, show=show, dpi=600)

        return fig, ax

    def optimal_dye_selector_simulated(
        self,
        potential_dyes: List[str],
        single_molecule_dyes: np.ndarray,
        filters: List[str],
        camera_parameters: Dict[str, Any],
        wavelength: np.ndarray,
        n_dyes_desired: int,
        min_photons_per_100ms: int = 500,
        n_simulations: int = 1000,
        background_photons: float = 4,
        NA: float = 1.49,
        pixel_size: float = 69,
        image_dims: int = 12,
        smoothing_function=None,
        integration_time_ms: float = 100,
        excitation_power_scaling: float = 1.0,
        exhaustive_search: bool = False,
        return_all_simulations: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Select optimal dye combination based on simulated separability.

        This function implements a simulation-based approach to dye selection:
        1. Filters dyes by minimum photon threshold
        2. Simulates camera images for each viable dye
        3. Extracts (A_R, A_G) color distributions
        4. Fits Gaussian models to each dye's color cloud
        5. Computes analytical misidentification rates for all combinations
        6. Returns the most separable combination

        Args:
            potential_dyes: List of candidate dye names
            single_molecule_dyes: Array with columns [name, photons_per_100ms]
                Column 1 contains photons per 100ms under standard conditions
            filters: Filter names for spectral calculations
            camera_parameters: Camera calibration parameters
            wavelength: Wavelength array for spectral calculations
            n_dyes_desired: Number of dyes to select
            min_photons_per_100ms: Minimum photon threshold at detector (default: 500)
            n_simulations: Number of simulated molecules per dye (default: 1000)
            background_photons: Background photon count (default: 4)
            NA: Numerical aperture (default: 1.49)
            pixel_size: Pixel size in nm (default: 69)
            image_dims: Image dimension in pixels (default: 12)
            smoothing_function: PSF smoothing function (default: None)
            integration_time_ms: Integration time in ms (default: 100)
            excitation_power_scaling: Photon count scaling factor (default: 1.0)
            exhaustive_search: If True, test all combinations (slow!).
                              If False, use greedy selection (default: False)
            return_all_simulations: If True, return simulation data for ALL viable dyes,
                                   not just the selected ones (default: False)
            verbose: Print progress and results (default: True)

        Returns:
            dict: {
                'selected_dyes': list of n_dyes_desired dye names,
                'overall_accuracy': float - overall classification accuracy,
                'confusion_matrix': np.ndarray - confusion matrix for selected dyes,
                'dye_gaussians': dict - Gaussian parameters (selected or all dyes),
                'dye_simulations': dict - Simulation data (selected or all dyes),
                'expected_photons': dict - expected photon counts (selected or all dyes),
                'viable_dyes': list - all viable dyes that passed photon threshold,
                'all_combinations_tested': list of dicts (if exhaustive_search=True),
                'separability_stats': dict - full separability statistics
            }
        """
        from itertools import combinations

        if verbose:
            print("="*60)
            print("OPTIMAL DYE SELECTION VIA SIMULATION")
            print("="*60)

        # Step 1: Filter by photon threshold
        if verbose:
            print(f"\nStep 1: Filtering dyes (min {min_photons_per_100ms} photons/100ms)...")

        filtered = self._filter_dyes_by_photons(
            potential_dyes,
            single_molecule_dyes,
            filters,
            camera_parameters,
            wavelength,
            min_photons_per_100ms=min_photons_per_100ms,
            integration_time_ms=integration_time_ms,
            excitation_power_scaling=excitation_power_scaling
        )

        viable_dyes = filtered['viable_dyes']

        if verbose:
            print(f"  {len(potential_dyes)} candidates -> {len(viable_dyes)} viable dyes")
            rejected = set(potential_dyes) - set(viable_dyes)
            if rejected:
                print(f"  Rejected: {rejected}")

        if len(viable_dyes) < n_dyes_desired:
            raise ValueError(f"Only {len(viable_dyes)} viable dyes, but {n_dyes_desired} requested!")

        # Step 2 & 3: Simulate all viable dyes
        if verbose:
            print(f"\nStep 2-3: Simulating {n_simulations} molecules per dye...")

        dye_simulations = {}
        dye_gaussians = {}

        for dye_name in viable_dyes:
            if verbose:
                source = filtered['expected_photons'][dye_name]
                detector = filtered['expected_photons_at_detector'][dye_name]
                print(f"  Simulating {dye_name} ({source:.0f} source / {detector:.0f} detector photons)...")

            color_data = self._simulate_dye_color_distributions(
                dye_name,
                filters,
                camera_parameters,
                wavelength,
                filtered['expected_photons'][dye_name],
                n_simulations=n_simulations,
                background_photons=background_photons,
                NA=NA,
                pixel_size=pixel_size,
                image_dims=image_dims,
                smoothing_function=smoothing_function
            )

            dye_simulations[dye_name] = color_data

            # Step 4: Fit Gaussian
            gaussian_params = self._fit_dye_gaussian(color_data)
            dye_gaussians[dye_name] = gaussian_params

            if verbose:
                success_rate = gaussian_params['n_valid'] / n_simulations * 100
                print(f"    Fit success: {gaussian_params['n_valid']}/{n_simulations} ({success_rate:.1f}%)")
                print(f"    Mean (A_R, A_G): ({gaussian_params['mean'][0]:.3f}, {gaussian_params['mean'][1]:.3f})")
                print(f"    Std  (A_R, A_G): ({gaussian_params['std_A_R']:.3f}, {gaussian_params['std_A_G']:.3f})")

        # Step 5-6: Find optimal combination
        if verbose:
            print(f"\nStep 4-6: Searching for optimal {n_dyes_desired}-dye combination...")

        if exhaustive_search:
            # Test all combinations
            all_combos = list(combinations(viable_dyes, n_dyes_desired))

            if verbose:
                print(f"  Testing all {len(all_combos)} combinations (exhaustive search)...")

            results = []
            for combo in all_combos:
                stats = self._calculate_dye_separability(
                    dye_gaussians,
                    list(combo),
                    n_monte_carlo=10000
                )

                results.append({
                    'dyes': list(combo),
                    'accuracy': stats['overall_accuracy'],
                    'stats': stats
                })

            # Sort by accuracy (descending)
            results.sort(key=lambda x: x['accuracy'], reverse=True)
            best_result = results[0]

        else:
            # Greedy selection: iteratively add the dye that maximizes separability
            if verbose:
                print(f"  Using greedy selection algorithm...")

            selected = []
            remaining = viable_dyes.copy()

            # Start with the two most separable dyes
            best_pair_acc = 0
            best_pair = None

            for dye1, dye2 in combinations(remaining, 2):
                stats = self._calculate_dye_separability(
                    dye_gaussians,
                    [dye1, dye2],
                    n_monte_carlo=10000
                )
                if stats['overall_accuracy'] > best_pair_acc:
                    best_pair_acc = stats['overall_accuracy']
                    best_pair = [dye1, dye2]

            selected = best_pair
            remaining = [d for d in remaining if d not in selected]

            if verbose:
                print(f"    Initial pair: {selected} (accuracy: {best_pair_acc:.3f})")

            # Iteratively add dyes
            while len(selected) < n_dyes_desired:
                best_acc = 0
                best_dye = None

                for dye in remaining:
                    candidate = selected + [dye]
                    stats = self._calculate_dye_separability(
                        dye_gaussians,
                        candidate,
                        n_monte_carlo=10000
                    )

                    if stats['overall_accuracy'] > best_acc:
                        best_acc = stats['overall_accuracy']
                        best_dye = dye

                selected.append(best_dye)
                remaining.remove(best_dye)

                if verbose:
                    print(f"    Added {best_dye} -> {selected} (accuracy: {best_acc:.3f})")

            # Get final statistics
            final_stats = self._calculate_dye_separability(
                dye_gaussians,
                selected,
                n_monte_carlo=10000
            )

            best_result = {
                'dyes': selected,
                'accuracy': final_stats['overall_accuracy'],
                'stats': final_stats
            }
            results = None

        # Prepare return dict
        if verbose:
            print("\n" + "="*60)
            print("RESULTS")
            print("="*60)
            print(f"Selected dyes: {best_result['dyes']}")
            print(f"Overall accuracy: {best_result['accuracy']:.3f}")
            print(f"\nConfusion Matrix:")
            print(best_result['stats']['confusion_matrix'])
            print(f"\nExpected photons (source/detector):")
            for dye in best_result['dyes']:
                source = filtered['expected_photons'][dye]
                detector = filtered['expected_photons_at_detector'][dye]
                print(f"  {dye}: {source:.0f} / {detector:.0f}")

        # Decide which dyes to include in returned data
        if return_all_simulations:
            # Return data for ALL viable dyes
            dyes_to_return = viable_dyes
        else:
            # Return data only for selected dyes (original behavior)
            dyes_to_return = best_result['dyes']

        return {
            'selected_dyes': best_result['dyes'],
            'overall_accuracy': best_result['accuracy'],
            'confusion_matrix': best_result['stats']['confusion_matrix'],
            'dye_gaussians': {dye: dye_gaussians[dye] for dye in dyes_to_return},
            'dye_simulations': {dye: dye_simulations[dye] for dye in dyes_to_return},
            'expected_photons': {dye: filtered['expected_photons'][dye] for dye in dyes_to_return},
            'viable_dyes': viable_dyes,  # Always include list of all viable dyes
            'all_combinations_tested': results if exhaustive_search else None,
            'separability_stats': best_result['stats']
        }
