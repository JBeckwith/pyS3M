#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image analysis functions for multicolour single-molecule localization microscopy.

This module provides comprehensive functionality for:
- Gaussian fitting with various colour strategies  
- Parallel processing for large datasets
- Weighted least squares optimization
- Multiple fitting strategies for different analysis needs

The module uses a strategy pattern for handling different types of fitting approaches
(colour, no-colour, just-colour, raw-colour, post-colour) with unified interfaces.

Created on Tue Dec 10 08:59:38 2024
@author: jbeckwith
jsb92, 2024/01/02 - Refactored August 15, 2025
"""

from typing import Optional, List, Tuple, Union, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.optimize import leastsq
import os
import sys
from numba import jit
import multiprocessing
from concurrent import futures
from tqdm import tqdm
import ProgressUtils
import logging

# Set up module paths
module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import IOFunctions
import sCMOSFunctions
import PSFFunctions
import gaussoptfuncs


class FittingStrategy(Enum):
    """Enumeration for different fitting strategies.

    Defines the various approaches to Gaussian fitting with different
    colour channel handling strategies.
    """

    STANDARD = "standard"  # Full colour fitting with all channels
    NOCOLOUR = "nocolour"  # No colour information, intensity only
    JUSTCOLOUR = "justcolour"  # Colour channels only, no intensity
    RAWCOLOUR = "rawcolour"  # Raw colour data without processing
    POSTHENCOLOUR = "posthencolour"  # Post-processing colour enhancement


class FittingConstants:
    """Constants for fitting operations."""

    # Fitting tolerances
    DEFAULT_FTOL = 1e-2
    DEFAULT_XTOL = 1e-2

    # Parallel processing limits
    MAX_WORKERS = 60  # Python crashes when using >64 cores
    WORKER_RATIO = 0.9
    TASKS_PER_WORKER = 100

    # Array dimensions for different strategies
    PARAM_DIMENSIONS = {
        FittingStrategy.STANDARD: {"fit": 12, "error": 10},
        FittingStrategy.NOCOLOUR: {"fit": 8, "error": 6},
        FittingStrategy.JUSTCOLOUR: {"fit": 10, "error": 8},
        FittingStrategy.RAWCOLOUR: {"fit": 10, "error": 8},
        FittingStrategy.POSTHENCOLOUR: {"fit": 10, "error": 8},
    }


@dataclass
class FittingParameters:
    """Configuration parameters for fitting operations.

    Encapsulates all parameters needed for puncta fitting with validation.

    Attributes:
        puncta: List of 2D arrays containing puncta data.
        smoothed_puncta: List of 2D arrays containing smoothed puncta data.
        masks: Optional list of 3D arrays containing colour masks.
        weights: List of 2D arrays containing fitting weights.
        relative_coords: List of relative coordinate offsets.
        planes: List of plane indices for each punctum.
        strategy: Fitting strategy to use.
    """

    puncta: List[np.ndarray]
    smoothed_puncta: List[np.ndarray]
    weights: List[np.ndarray]
    relative_coords: List[List[float]]
    planes: List[int]
    strategy: FittingStrategy
    masks: Optional[List[np.ndarray]] = None

    def __post_init__(self):
        """Validate fitting parameters after initialization."""
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate that all input parameters are consistent.

        Raises:
            FittingValidationError: If parameters are inconsistent or invalid.
        """
        n_puncta = len(self.puncta)

        # Check array length consistency
        if not all(
            len(arr) == n_puncta
            for arr in [
                self.smoothed_puncta,
                self.weights,
                self.relative_coords,
                self.planes,
            ]
        ):
            raise FittingValidationError(
                "All input arrays must have the same length as puncta"
            )

        # Check masks requirement for colour strategies
        colour_strategies = {
            FittingStrategy.STANDARD,
            FittingStrategy.JUSTCOLOUR,
            FittingStrategy.RAWCOLOUR,
            FittingStrategy.POSTHENCOLOUR,
        }

        if self.strategy in colour_strategies and self.masks is None:
            raise FittingValidationError(
                f"Strategy {self.strategy.value} requires masks parameter"
            )

        if self.masks is not None and len(self.masks) != n_puncta:
            raise FittingValidationError("Masks array must have same length as puncta")


class FittingValidationError(Exception):
    """Custom exception for fitting parameter validation errors."""

    pass


class FittingResultProcessor:
    """Handles processing and validation of fitting results."""

    @staticmethod
    def calculate_errors(pcov: np.ndarray, strategy: FittingStrategy) -> List[float]:
        """Calculate parameter errors from covariance matrix.

        Args:
            pcov: Parameter covariance matrix from fitting.
            strategy: Fitting strategy to determine expected error array size.

        Returns:
            List of parameter errors (standard deviations).
        """
        # Get expected error array size
        expected_size = FittingConstants.PARAM_DIMENSIONS[strategy]["error"]

        if pcov is None:
            return [np.nan] * expected_size

        # Handle scalar pcov (e.g., np.inf from failed fits)
        if np.isscalar(pcov) or pcov.ndim == 0:
            return [np.nan] * expected_size

        # Handle inf values in matrix
        if np.any(np.isinf(pcov)):
            return [np.nan] * expected_size

        try:
            # Vectorized error calculation - more efficient than loops
            diagonal = np.diag(pcov)
            errors = np.where(diagonal >= 0, np.sqrt(np.abs(diagonal)), 0.0)
            error_list = errors.tolist()

            # Ensure we return the correct number of errors
            if len(error_list) < expected_size:
                error_list.extend([np.nan] * (expected_size - len(error_list)))
            elif len(error_list) > expected_size:
                error_list = error_list[:expected_size]

            return error_list
        except (IndexError, ValueError):
            return [np.nan] * expected_size

    @staticmethod
    def process_fit_results(
        pfit: np.ndarray,
        pcov: np.ndarray,
        size: int,
        relative_coords: List[float],
        strategy: FittingStrategy,
        chisqr: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process raw fitting results into standardized format.

        Args:
            pfit: Raw fitting parameters.
            pcov: Parameter covariance matrix.
            relative_coords: Coordinate offsets to add.
            size: Image size in fitting.
            strategy: Fitting strategy used.
            chisqr: Chi-squared value from fitting (default 1.0).

        Returns:
            Tuple of (processed_parameters, parameter_errors).
        """
        if pfit is None or not hasattr(pfit, "__len__") or len(pfit) == 0:
            dims = FittingConstants.PARAM_DIMENSIONS[strategy]
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

        # Chi-squared is now passed as parameter from the fitting processor

        # Process parameters based on strategy
        # For STANDARD: pfit has [x, y, sy, sx, bg_B, bg_G, bg_R, A_B, A_G, A_R] (10 parameters from gaussoptfuncs)
        # But output needs [x, y, sx, sy, bg_B, bg_G, bg_R, A_B, A_G, A_R] (sx/sy order corrected)

        if strategy == FittingStrategy.STANDARD:
            # Reorder sx/sy and keep everything else: [x, y, sx, sy, bg_B, bg_G, bg_R, A_B, A_G, A_R]
            pfit_processed = np.array(
                [
                    pfit[0],
                    pfit[1],
                    pfit[3],
                    pfit[2],  # x, y, sx, sy (note: sx/sy swapped)
                    pfit[4],
                    pfit[5],
                    pfit[6],  # bg_B, bg_G, bg_R
                    pfit[7],
                    pfit[8],
                    pfit[9],  # A_B, A_G, A_R
                ]
            )
        else:
            # For other strategies, use as-is for now
            pfit_processed = pfit.copy()

        if np.any(pfit_processed[:4] < 0) | np.any(pfit_processed[:4] > size):
            return (
                np.full(len(pfit_processed), np.nan),
                np.full(len(pfit_processed), np.nan),
            )

        # Add relative coordinates to position parameters (first two elements)
        if (
            relative_coords is not None
            and hasattr(relative_coords, "__len__")
            and len(relative_coords) >= 2
        ):
            pfit_processed[:2] += relative_coords[:2]

        # Square amplitude and background parameters for storage (after error calculation)
        # leastsq returns optimized square-root values, but we store squared values as photon counts
        if strategy == FittingStrategy.STANDARD:
            # For STANDARD output: [x, y, sx, sy, bg_B, bg_G, bg_R, A_B, A_G, A_R]
            # Square backgrounds (positions 4-6) and amplitudes (positions 7-9)
            pfit_processed[4:10] = np.square(pfit_processed[4:10])
        elif strategy == FittingStrategy.NOCOLOUR:
            # For NOCOLOUR: [x, y, sx, sy, bg, A, ...]
            # Square background and amplitude
            if len(pfit_processed) >= 6:
                pfit_processed[4:6] = np.square(pfit_processed[4:6])

        # Append chi-squared
        pfit_final = np.append(pfit_processed, chisqr)

        # Calculate errors
        errors = FittingResultProcessor.calculate_errors(pcov, strategy)

        return pfit_final, np.array(errors)


class FittingProcessor(ABC):
    """Abstract base class for fitting strategy processors."""

    @abstractmethod
    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit a single punctum with this strategy.

        Args:
            punctum: 2D array containing punctum data.
            smoothed_punctum: 2D array containing smoothed punctum data.
            weights: 2D array containing fitting weights.
            relative_coords: Relative coordinate offsets.
            masks: Optional 3D array containing colour masks.

        Returns:
            Tuple of (fit_parameters, parameter_errors).
        """
        pass


class StandardFittingProcessor(FittingProcessor):
    """Processor for standard colour fitting strategy."""

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit single punctum using standard colour fitting.

        Uses weighted least squares with colour channel information.

        Args:
            punctum: 2D array containing punctum data.
            smoothed_punctum: 2D array containing smoothed punctum data.
            weights: 2D array containing fitting weights.
            relative_coords: Relative coordinate offsets.
            masks: 3D array containing colour masks (required).

        Returns:
            Tuple of (fit_parameters, parameter_errors).

        Raises:
            FittingValidationError: If masks are not provided.
        """
        if masks is None:
            raise FittingValidationError("Standard fitting requires masks")

        # Get initial guess from smoothed and raw data
        initial_guess = self._generate_initial_guess(smoothed_punctum, punctum, masks)

        # Perform weighted least squares fit
        return self._perform_wls_fit(
            punctum, initial_guess, masks, weights, relative_coords
        )

    def _generate_initial_guess(
        self, smoothed_punctum: np.ndarray, raw_punctum: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """Generate initial parameter guess for fitting using gaussoptfuncs.

        Args:
            smoothed_punctum: Smoothed punctum data.
            raw_punctum: Raw punctum data.
            masks: Colour masks.

        Returns:
            Initial parameter guess array.
        """
        # Use the proper initial_guess function from gaussoptfuncs
        # Returns: [x, y, sy, sx, bg_B, bg_G, bg_R, A_B, A_G, A_R]
        return gaussoptfuncs.initial_guess(smoothed_punctum, raw_punctum, masks)

    def _perform_wls_fit(
        self,
        data: np.ndarray,
        initial_guess: np.ndarray,
        masks: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform weighted least squares fitting.

        Args:
            data: Data to fit.
            initial_guess: Initial parameter guess.
            masks: Colour masks.
            weights: Fitting weights.
            relative_coords: Coordinate offsets.

        Returns:
            Tuple of (fit_parameters, parameter_errors).
        """
        size = int(data.shape[0])
        ravelsize = int(np.prod(data.shape))

        try:
            pfit, pcov, infodict, errmsg, success = leastsq(
                gaussoptfuncs.WLS_chi_nobounds,
                x0=initial_guess,
                args=(data, masks, weights, size, ravelsize),
                full_output=True,
                ftol=FittingConstants.DEFAULT_FTOL,
                xtol=FittingConstants.DEFAULT_XTOL,
            )

            if success not in np.array([1, 2, 3, 4]):
                dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.STANDARD]
                return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

            # Calculate chi-squared
            chisqr = np.sum(
                np.square(
                    gaussoptfuncs.WLS_chi_nobounds(
                        pfit, data, masks, weights, size, ravelsize
                    )
                )
            ) / (len(data.ravel()) - len(initial_guess))

            # Process covariance matrix
            if (len(data.ravel()) > len(initial_guess)) and pcov is not None:
                s_sq = chisqr
                pcov = pcov * s_sq
            else:
                pcov = np.inf

            return FittingResultProcessor.process_fit_results(
                pfit, pcov, size, relative_coords, FittingStrategy.STANDARD, chisqr
            )

        except Exception as e:
            logging.warning(f"Fitting failed: {e}")
            dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.STANDARD]
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))


class NoColourFittingProcessor(FittingProcessor):
    """Processor for no-colour fitting strategy."""

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit single punctum using no-colour strategy.

        Fits intensity information only, ignoring colour channels.

        Args:
            punctum: 2D array containing punctum data.
            smoothed_punctum: 2D array containing smoothed punctum data.
            weights: 2D array containing fitting weights.
            relative_coords: Relative coordinate offsets.
            masks: Not used for this strategy.

        Returns:
            Tuple of (fit_parameters, parameter_errors).
        """
        # Get initial guess (simplified for no-colour)
        initial_guess = self._generate_initial_guess(smoothed_punctum)

        # Perform fitting without colour information
        return self._perform_nocolour_fit(
            punctum, initial_guess, weights, relative_coords
        )

    def _generate_initial_guess(self, smoothed_punctum: np.ndarray) -> np.ndarray:
        """Generate initial guess for no-colour fitting.

        Args:
            smoothed_punctum: Smoothed punctum data.

        Returns:
            Initial parameter guess (8 parameters for no-colour).
        """
        center = np.array(smoothed_punctum.shape) // 2
        max_val = np.max(smoothed_punctum)

        # 8-parameter guess: [x, y, sx, sy, A, bg, theta, offset]
        initial_guess = np.array(
            [
                center[1],
                center[0],  # x, y centers
                1.0,
                1.0,  # sigma_x, sigma_y
                max_val,  # Amplitude
                np.min(smoothed_punctum),  # Background
                0.0,
                0.0,  # theta, offset
            ]
        )

        return initial_guess

    def _perform_nocolour_fit(
        self,
        data: np.ndarray,
        initial_guess: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform no-colour fitting.

        Args:
            data: Data to fit.
            initial_guess: Initial parameter guess.
            weights: Fitting weights.
            relative_coords: Coordinate offsets.

        Returns:
            Tuple of (fit_parameters, parameter_errors).
        """
        try:
            # This would call the appropriate no-colour fitting function
            # For now, using a placeholder implementation
            pfit, pcov, infodict, errmsg, success = leastsq(
                gaussoptfuncs.WLS_chi_nocolour_nobounds,  # Placeholder function name
                x0=initial_guess,
                args=(data, weights),
                full_output=True,
                ftol=FittingConstants.DEFAULT_FTOL,
                xtol=FittingConstants.DEFAULT_XTOL,
            )

            if success not in np.array([1, 2, 3, 4]):
                dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.NOCOLOUR]
                return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

            # Calculate chi-squared
            chisqr = np.sum(
                np.square(
                    gaussoptfuncs.WLS_chi_nocolour_nobounds(
                        pfit, data, weights, size, ravelsize
                    )
                )
            ) / (len(data.ravel()) - len(initial_guess))

            # Process covariance matrix
            if (len(data.ravel()) > len(initial_guess)) and pcov is not None:
                s_sq = chisqr
                pcov = pcov * s_sq
            else:
                pcov = np.inf

            return FittingResultProcessor.process_fit_results(
                pfit,
                pcov,
                int(data.shape[0]),
                relative_coords,
                FittingStrategy.NOCOLOUR,
                chisqr,
            )

        except Exception as e:
            logging.warning(f"No-colour fitting failed: {e}")
            dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.NOCOLOUR]
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))


class JustColourFittingProcessor(FittingProcessor):
    """Processor for just-colour fitting strategy."""

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit single punctum using just-colour strategy.

        Args:
            punctum: 2D array containing punctum data.
            smoothed_punctum: 2D array containing smoothed punctum data.
            weights: 2D array containing fitting weights.
            relative_coords: Relative coordinate offsets.
            masks: 3D array containing colour masks (required).

        Returns:
            Tuple of (fit_parameters, parameter_errors).
        """
        if masks is None:
            raise FittingValidationError("Just-colour fitting requires masks")

        # Implementation similar to other processors but for just-colour strategy
        initial_guess = self._generate_initial_guess(smoothed_punctum, masks)
        return self._perform_justcolour_fit(
            punctum, initial_guess, masks, weights, relative_coords
        )

    def _generate_initial_guess(
        self, smoothed_punctum: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """Generate initial guess for just-colour fitting."""
        # 10-parameter guess for just-colour
        center = np.array(smoothed_punctum.shape) // 2
        max_val = np.max(smoothed_punctum)

        return np.array(
            [
                center[1],
                center[0],  # x, y
                1.0,
                1.0,  # sigmas
                max_val * 0.4,
                max_val * 0.6,  # A_G, A_R ratios
                np.min(smoothed_punctum),
                np.min(smoothed_punctum),  # backgrounds
                0.0,
                0.0,  # theta, offset
            ]
        )

    def _perform_justcolour_fit(
        self,
        data: np.ndarray,
        initial_guess: np.ndarray,
        masks: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform just-colour fitting."""
        size = int(data.shape[0])
        locparams = relative_coords  # For justcolour fitting

        try:
            pfit, pcov, infodict, errmsg, success = leastsq(
                gaussoptfuncs.WLS_chi_justcolour_nobounds,
                x0=initial_guess,
                args=(data, weights, size, locparams),
                full_output=True,
                ftol=FittingConstants.DEFAULT_FTOL,
                xtol=FittingConstants.DEFAULT_XTOL,
            )

            if success not in np.array([1, 2, 3, 4]):
                dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.JUSTCOLOUR]
                return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

            # Calculate chi-squared
            chisqr = np.sum(
                np.square(
                    gaussoptfuncs.WLS_chi_justcolour_nobounds(
                        pfit, data, weights, size, locparams
                    )
                )
            ) / (len(data.ravel()) - len(initial_guess))

            # Process covariance matrix
            if (len(data.ravel()) > len(initial_guess)) and pcov is not None:
                s_sq = chisqr
                pcov = pcov * s_sq
            else:
                pcov = np.inf

            return FittingResultProcessor.process_fit_results(
                pfit, pcov, size, relative_coords, FittingStrategy.JUSTCOLOUR, chisqr
            )

        except Exception as e:
            logging.warning(f"Just-colour fitting failed: {e}")
            dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.JUSTCOLOUR]
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))


class RawColourFittingProcessor(FittingProcessor):
    """Processor for raw-colour fitting strategy."""

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit single punctum using raw-colour strategy."""
        if masks is None:
            raise FittingValidationError("Raw-colour fitting requires masks")

        initial_guess = self._generate_initial_guess(smoothed_punctum, masks)
        return self._perform_rawcolour_fit(
            punctum, initial_guess, masks, weights, relative_coords
        )

    def _generate_initial_guess(
        self, smoothed_punctum: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """Generate initial guess for raw-colour fitting."""
        center = np.array(smoothed_punctum.shape) // 2
        max_val = np.max(smoothed_punctum)

        return np.array(
            [
                center[1],
                center[0],  # x, y
                1.0,
                1.0,  # sigmas
                max_val * 0.3,
                max_val * 0.4,
                max_val * 0.3,  # Raw colour amplitudes
                np.min(smoothed_punctum),  # background
                0.0,
                0.0,  # theta, offset
            ]
        )

    def _perform_rawcolour_fit(
        self,
        data: np.ndarray,
        initial_guess: np.ndarray,
        masks: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform raw-colour fitting."""
        size = int(data.shape[0])
        ravelsize = int(np.prod(data.shape))
        locparams = relative_coords  # For rawcolour fitting

        try:
            pfit, pcov, infodict, errmsg, success = leastsq(
                gaussoptfuncs.WLS_rawcolour_chi_nobounds,
                x0=initial_guess,
                args=(data, masks, weights, size, ravelsize, locparams),
                full_output=True,
                ftol=FittingConstants.DEFAULT_FTOL,
                xtol=FittingConstants.DEFAULT_XTOL,
            )

            if success not in np.array([1, 2, 3, 4]):
                dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.RAWCOLOUR]
                return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

            # Calculate chi-squared
            chisqr = np.sum(
                np.square(
                    gaussoptfuncs.WLS_rawcolour_chi_nobounds(
                        pfit, data, masks, weights, size, ravelsize, locparams
                    )
                )
            ) / (len(data.ravel()) - len(initial_guess))

            # Process covariance matrix
            if (len(data.ravel()) > len(initial_guess)) and pcov is not None:
                s_sq = chisqr
                pcov = pcov * s_sq
            else:
                pcov = np.inf

            return FittingResultProcessor.process_fit_results(
                pfit, pcov, size, relative_coords, FittingStrategy.RAWCOLOUR, chisqr
            )

        except Exception as e:
            logging.warning(f"Raw-colour fitting failed: {e}")
            dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.RAWCOLOUR]
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))


class PosthenColourFittingProcessor(FittingProcessor):
    """Processor for post-colour enhancement fitting strategy."""

    def fit_single_punctum(
        self,
        punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        relative_coords: List[float],
        masks: Optional[np.ndarray] = None,
        raw_punctum: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit single punctum using post-colour enhancement strategy.

        The posthencolour strategy performs a two-stage fit:
        1. First fits position using no-colour (greyscale) data
        2. Then fits colour using raw data with fixed positions
        """
        if masks is None:
            raise FittingValidationError("Post-colour fitting requires masks")

        # Use raw_punctum if provided, otherwise use punctum
        if raw_punctum is None:
            raw_punctum = punctum

        return self._perform_posthencolour_fit(
            punctum, raw_punctum, smoothed_punctum, weights, masks, relative_coords
        )

    def _generate_initial_guess(
        self, smoothed_punctum: np.ndarray, masks: np.ndarray
    ) -> np.ndarray:
        """Generate initial guess for post-colour fitting."""
        center = np.array(smoothed_punctum.shape) // 2
        max_val = np.max(smoothed_punctum)

        return np.array(
            [
                center[1],
                center[0],  # x, y
                1.0,
                1.0,  # sigmas
                max_val * 0.4,
                max_val * 0.6,  # Enhanced colour ratios
                np.min(smoothed_punctum),
                np.min(smoothed_punctum),  # backgrounds
                0.0,
                0.0,  # theta, offset
            ]
        )

    def _perform_posthencolour_fit(
        self,
        grayscale_punctum: np.ndarray,
        raw_punctum: np.ndarray,
        smoothed_punctum: np.ndarray,
        weights: np.ndarray,
        masks: np.ndarray,
        relative_coords: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform post-colour enhancement fitting.

        This implements the two-stage fitting approach from the original:
        1. First fit position parameters using greyscale data (no-colour)
        2. Then fit colour parameters using raw data with fixed positions
        """
        try:
            # Generate initial guess from smoothed data
            initial_guess = np.empty(10, dtype=np.float32)
            initial_guess[:] = gaussoptfuncs.initial_guess(
                smoothed_punctum, raw_punctum, masks
            )

            # Split into position and colour components
            initial_guess_position = initial_guess[:6]
            initial_guess_position[-1] = initial_guess[-1] * 3
            initial_guess_position[-2] = initial_guess_position[-2] * 3
            initial_guess_colour = initial_guess[6:]

            # Stage 1: Fit position using no-colour (greyscale) fitting
            nocolour_processor = NoColourFittingProcessor()
            pfit_pos_leastsq, perr_pos_leastsq = (
                nocolour_processor._perform_nocolour_fit(
                    grayscale_punctum, initial_guess_position, weights, relative_coords
                )
            )

            # Extract location parameters for colour fitting
            locparams = np.array(
                [
                    pfit_pos_leastsq[0],
                    pfit_pos_leastsq[1],
                    pfit_pos_leastsq[2],
                    pfit_pos_leastsq[3],
                    pfit_pos_leastsq[4],
                    pfit_pos_leastsq[5],
                ]
            )

            # Stage 2: Fit colour using raw data with fixed positions
            rawcolour_processor = RawColourFittingProcessor()

            # Create a mock WLS_fit_rawcolour_nobounds call
            size = int(raw_punctum.shape[0])
            ravelsize = int(np.prod(raw_punctum.shape))

            pfit_colour, pcov_colour, infodict, errmsg, success = leastsq(
                gaussoptfuncs.WLS_rawcolour_chi_nobounds,
                x0=initial_guess_colour,
                args=(raw_punctum, masks, weights, size, ravelsize, locparams),
                full_output=True,
                ftol=FittingConstants.DEFAULT_FTOL,
                xtol=FittingConstants.DEFAULT_XTOL,
            )

            if success not in np.array([1, 2, 3, 4]):
                dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.POSTHENCOLOUR]
                return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))

            # Calculate chi-squared for colour component
            chisqr_colour = np.sum(
                np.square(
                    gaussoptfuncs.WLS_rawcolour_chi_nobounds(
                        pfit_colour,
                        raw_punctum,
                        masks,
                        weights,
                        size,
                        ravelsize,
                        locparams,
                    )
                )
            ) / (len(raw_punctum.ravel()) - len(initial_guess_colour))

            # Process covariance matrix
            if (
                len(raw_punctum.ravel()) > len(initial_guess_colour)
            ) and pcov_colour is not None:
                s_sq = chisqr_colour
                pcov_colour = pcov_colour * s_sq
            else:
                pcov_colour = np.inf

            # Calculate errors for colour component
            perr_colour_leastsq = FittingResultProcessor.calculate_errors(pcov_colour)

            # Combine position and colour results
            pfit_leastsq = np.concatenate(
                [pfit_pos_leastsq[:-1], pfit_colour]
            )  # Exclude position chi-squared
            perr_leastsq = np.concatenate(
                [perr_pos_leastsq[:-1], perr_colour_leastsq]
            )  # Exclude position error

            return pfit_leastsq, np.array(perr_leastsq)

        except Exception as e:
            logging.warning(f"Post-colour fitting failed: {e}")
            dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.POSTHENCOLOUR]
            return (np.full(dims["fit"], np.nan), np.full(dims["error"], np.nan))


class Image_Analysis_Functions:
    """A class for image analysis and Gaussian fitting in multicolour SMLM.

    This class provides comprehensive functionality for:
    - Multiple fitting strategies (colour, no-colour, just-colour, etc.)
    - Parallel processing for large datasets
    - Weighted least squares optimization
    - Unified interface for all fitting approaches

    The class uses a strategy pattern to handle different fitting approaches
    while providing a consistent API and optimized performance.

    Example:
        ```python
        analyzer = Image_Analysis_Functions()

        # Standard colour fitting
        fit_params, errors = analyzer.fit_puncta_method(
            puncta_data, smoothed_data, masks, weights, coords, planes,
            strategy=FittingStrategy.STANDARD
        )

        # Parallel processing
        fit_params, errors = analyzer.fit_puncta_parallel_method(
            puncta_data, smoothed_data, masks, weights, coords, planes,
            strategy=FittingStrategy.NOCOLOUR
        )
        ```
    """

    def __init__(self):
        """Initialize the Image_Analysis_Functions class.

        Sets up strategy processors and loads required dependencies.
        """
        # Initialize strategy processors
        self.processors = {
            FittingStrategy.STANDARD: StandardFittingProcessor(),
            FittingStrategy.NOCOLOUR: NoColourFittingProcessor(),
            FittingStrategy.JUSTCOLOUR: JustColourFittingProcessor(),
            FittingStrategy.RAWCOLOUR: RawColourFittingProcessor(),
            FittingStrategy.POSTHENCOLOUR: PosthenColourFittingProcessor(),
        }

        # Initialize dependencies
        self.io = IOFunctions.IO_Functions()
        self.scmos = sCMOSFunctions.sCMOS_Functions()
        self.psf_f = PSFFunctions.PSF_Functions()

    def fit_puncta_method(
        self,
        puncta: List[np.ndarray],
        smoothed_puncta: List[np.ndarray],
        weights: List[np.ndarray],
        relative_coords: List[List[float]],
        planes: List[int],
        strategy: FittingStrategy,
        masks: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit puncta using specified strategy.

        Unified method that replaces the 19 duplicate fitting functions from
        the original implementation. Uses strategy pattern to handle different
        fitting approaches efficiently.

        Args:
            puncta: List of 2D arrays containing puncta data.
            smoothed_puncta: List of 2D arrays containing smoothed puncta data.
            weights: List of 2D arrays containing fitting weights.
            relative_coords: List of relative coordinate offsets for each punctum.
            planes: List of plane indices for each punctum.
            strategy: Fitting strategy to use (STANDARD, NOCOLOUR, etc.).
            masks: Optional list of 3D arrays containing colour masks.
                  Required for colour-based strategies.

        Returns:
            Tuple containing:
                - pfit_leastsq: Array of shape (n_puncta, n_params) with fit parameters
                - perr_leastsq: Array of shape (n_puncta, n_errors) with parameter errors

        Raises:
            FittingValidationError: If input parameters are invalid or inconsistent.

        Example:
            ```python
            # Standard colour fitting
            fit_params, errors = analyzer.fit_puncta_method(
                puncta, smoothed, weights, coords, planes,
                strategy=FittingStrategy.STANDARD, masks=colour_masks
            )

            # No-colour fitting
            fit_params, errors = analyzer.fit_puncta_method(
                puncta, smoothed, weights, coords, planes,
                strategy=FittingStrategy.NOCOLOUR
            )
            ```
        """
        # Create and validate parameters
        params = FittingParameters(
            puncta=puncta,
            smoothed_puncta=smoothed_puncta,
            weights=weights,
            relative_coords=relative_coords,
            planes=planes,
            strategy=strategy,
            masks=masks,
        )

        # Get appropriate processor
        processor = self.processors[strategy]

        # Get array dimensions for this strategy
        dims = FittingConstants.PARAM_DIMENSIONS[strategy]

        # Pre-allocate result arrays
        n_puncta = len(puncta)
        pfit_leastsq = np.empty((n_puncta, dims["fit"]), dtype=np.float32)
        perr_leastsq = np.empty((n_puncta, dims["error"]), dtype=np.float32)

        # Initialize with NaN
        pfit_leastsq.fill(np.nan)
        perr_leastsq.fill(np.nan)

        # Process each punctum
        for i, punctum in enumerate(puncta):
            try:
                # Get masks for this punctum if provided
                punctum_masks = masks[i] if masks is not None else None

                # Fit single punctum
                fit_params, fit_errors = processor.fit_single_punctum(
                    punctum=punctum,
                    smoothed_punctum=smoothed_puncta[i],
                    weights=weights[i],
                    relative_coords=relative_coords[i],
                    masks=punctum_masks,
                )

                # Store results (fit_params includes chi-squared at the end)
                # fit_params contains: [param1, param2, ..., paramN, chi_squared]
                # We need: [param1, param2, ..., paramN, chi_squared, plane_index]

                # Store fit parameters including chi-squared, then add plane index
                # fit_params contains: [fit_param1, ..., fit_paramN, chi_squared] (N+1 elements)
                # Final array: [fit_param1, ..., fit_paramN, chi_squared, plane_index] (N+2 elements)

                # Store all fit parameters including chi-squared, then add plane index
                # fit_params should contain [param1, ..., param10, chi_squared] = 11 elements
                # Final array should be [param1, ..., param10, chi_squared, plane_index] = 12 elements

                # Store fit_params in positions 0 to len(fit_params)-1
                fit_param_count = len(fit_params)
                max_fit_params = dims["fit"] - 1  # Leave last position for plane index

                if fit_param_count <= max_fit_params:
                    # Store all fit parameters including chi-squared
                    pfit_leastsq[i, :fit_param_count] = fit_params
                else:
                    # fit_params is too long, store what fits but preserve chi-squared
                    # This should not happen if dimensions are correct
                    pfit_leastsq[i, :max_fit_params] = fit_params[:max_fit_params]

                # Store plane index in the last position (don't overwrite chi-squared!)
                pfit_leastsq[i, -1] = planes[i]
                perr_leastsq[i, :] = fit_errors[: dims["error"]]

            except Exception as e:
                logging.warning(f"Failed to fit punctum {i}: {e}")
                # Results already initialized with NaN
                continue

        return pfit_leastsq, perr_leastsq

    def fit_puncta_parallel_method(
        self,
        puncta: List[np.ndarray],
        smoothed_puncta: List[np.ndarray],
        weights: List[np.ndarray],
        relative_coords: List[List[float]],
        planes: List[int],
        strategy: FittingStrategy,
        masks: Optional[List[np.ndarray]] = None,
        asynch: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit puncta in parallel using specified strategy.

        Parallel version of fit_puncta_method that distributes work across
        multiple CPU cores for improved performance on large datasets.

        Args:
            puncta: List of 2D arrays containing puncta data.
            smoothed_puncta: List of 2D arrays containing smoothed puncta data.
            weights: List of 2D arrays containing fitting weights.
            relative_coords: List of relative coordinate offsets for each punctum.
            planes: List of plane indices for each punctum.
            strategy: Fitting strategy to use.
            masks: Optional list of 3D arrays containing colour masks.
            asynch: If True, return futures for asynchronous processing.

        Returns:
            Tuple containing fit parameters and errors arrays.
            If asynch=True, returns futures that can be processed later.

        Example:
            ```python
            # Parallel processing with optimal batch size
            fit_params, errors = analyzer.fit_puncta_parallel_method(
                large_puncta_list, smoothed_list, weights_list,
                coords_list, planes_list,
                strategy=FittingStrategy.STANDARD, masks=masks_list
            )
            ```
        """
        # Calculate optimal parallelization parameters
        n_workers = min(
            FittingConstants.MAX_WORKERS,
            max(1, int(FittingConstants.WORKER_RATIO * multiprocessing.cpu_count())),
        )
        n_puncta = len(puncta)
        n_tasks = FittingConstants.TASKS_PER_WORKER * n_workers

        # Calculate puncta per task with load balancing
        puncta_per_task = [
            (
                int(n_puncta / n_tasks + 1)
                if _ < n_puncta % n_tasks
                else int(n_puncta / n_tasks)
            )
            for _ in range(n_tasks)
        ]

        start_indices = np.cumsum([0] + puncta_per_task[:-1])

        # Submit tasks to process pool
        fs = []
        executor = futures.ProcessPoolExecutor(n_workers)

        for i, n_puncta_task in zip(start_indices, puncta_per_task):
            if n_puncta_task == 0:
                continue

            # Slice data for this task
            task_puncta = puncta[i : i + n_puncta_task]
            task_smoothed = smoothed_puncta[i : i + n_puncta_task]
            task_weights = weights[i : i + n_puncta_task]
            task_coords = relative_coords[i : i + n_puncta_task]
            task_planes = planes[i : i + n_puncta_task]
            task_masks = masks[i : i + n_puncta_task] if masks is not None else None

            # Submit task
            fs.append(
                executor.submit(
                    self.fit_puncta_method,
                    task_puncta,
                    task_smoothed,
                    task_weights,
                    task_coords,
                    task_planes,
                    strategy,
                    task_masks,
                )
            )

        if asynch:
            return fs

        # Wait for completion and combine results
        return self.fits_from_futures(fs, strategy)

    def fits_from_futures(
        self, fs: List[futures.Future], strategy: FittingStrategy
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine results from parallel fitting futures.

        Args:
            fs: List of futures from parallel fitting operations.
            strategy: Fitting strategy used (determines array dimensions).

        Returns:
            Tuple of combined (fit_parameters, parameter_errors) arrays.
        """
        # Get dimensions for this strategy
        dims = FittingConstants.PARAM_DIMENSIONS[strategy]

        # Collect all results
        all_fits = []
        all_errors = []

        with ProgressUtils.fitting_progress_bar(total=len(fs), desc="Collecting fitting results") as pbar:
            for f in fs:
                try:
                    fit_params, fit_errors = f.result()
                    all_fits.append(fit_params)
                    all_errors.append(fit_errors)
                except Exception as e:
                    logging.warning(f"Future failed: {e}")
                finally:
                    pbar.update(1)

        # Concatenate results
        if all_fits:
            combined_fits = np.vstack(all_fits)
            combined_errors = np.vstack(all_errors)
        else:
            # No successful results
            combined_fits = np.empty((0, dims["fit"]), dtype=np.float32)
            combined_errors = np.empty((0, dims["error"]), dtype=np.float32)

        return combined_fits, combined_errors
