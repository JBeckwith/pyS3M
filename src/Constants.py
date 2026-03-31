#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Constants for pyBayerSMLM

This module contains all magic numbers and constants used throughout the codebase
to improve maintainability and avoid hard-coded values scattered across files.

Created on August 29, 2025
@author: Claude Code
"""

# Standard logging for constants module
from LoggingFramework import setup_logger

logger = setup_logger(__name__, console_output=False)
logger.info("Constants module loaded with standardised values")

# Camera and Image Processing Constants
DEFAULT_PIXEL_SIZE = 3.45  # μm - standard pixel size for many sCMOS cameras
DEFAULT_SMOOTHING_SIZE = 10  # pixels - size for uniform filtering
DEFAULT_N_BINS_FALLBACK = 10  # fallback number of bins for histograms

# Time Constants (seconds)
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600

# Default Camera Parameters
DEFAULT_CAMERA_OFFSET = 100.0  # ADU
DEFAULT_CAMERA_VARIANCE = 8.0  # ADU²
DEFAULT_N_PHOTONS = 1000  # photons per molecule
DEFAULT_N_FRAMES = 100  # frames per experiment

# Numerical Constants
SMALL_NUMBER = 1e-100  # small number to avoid division by zero
KERNEL_WIDTH_MULTIPLIER = 10  # multiplier for blur kernel width calculation

# File I/O Constants
DEFAULT_ENCODING = "utf-8"
TIFF_FLOAT_BIT_DEPTH = float

# Error Handling Constants
DEFAULT_TIMEOUT_SECONDS = 30
MAX_RETRY_ATTEMPTS = 3


class CalibrationConstants:
    """Constants specific to camera calibration routines."""

    PIXEL_SIZE = DEFAULT_PIXEL_SIZE
    SMOOTHING_SIZE = DEFAULT_SMOOTHING_SIZE
    TIME_DISPLAY_THRESHOLD_MINUTES = SECONDS_PER_MINUTE
    TIME_DISPLAY_THRESHOLD_HOURS = SECONDS_PER_HOUR


class ProcessingConstants:
    """Constants for image and data processing."""

    N_BINS_FALLBACK = DEFAULT_N_BINS_FALLBACK
    SMALL_EPSILON = SMALL_NUMBER
    KERNEL_MULTIPLIER = KERNEL_WIDTH_MULTIPLIER


class DefaultParameters:
    """Default parameter values for various functions."""

    CAMERA_OFFSET = DEFAULT_CAMERA_OFFSET
    CAMERA_VARIANCE = DEFAULT_CAMERA_VARIANCE
    N_PHOTONS = DEFAULT_N_PHOTONS
    N_FRAMES = DEFAULT_N_FRAMES


class ResultColumns:
    """Column names for localization fit results.

    Defines standard column names for fitting results and their error estimates
    to ensure consistency across all fitting workflows.
    """

    # Standard fit parameter columns
    STANDARD_FIT_PARAMS = [
        "xc",  # X center coordinate
        "yc",  # Y center coordinate
        "s_x",  # X sigma (width)
        "s_y",  # Y sigma (width)
        "bg_B",  # Background (Blue channel)
        "bg_G",  # Background (Green channel)
        "bg_R",  # Background (Red channel)
        "A_B",  # Amplitude (Blue channel)
        "A_G",  # Amplitude (Green channel)
        "A_R",  # Amplitude (Red channel)
        "chi_sqr",  # Chi-squared goodness of fit
        "frame",  # Frame number
    ]

    # Standard fit error columns
    STANDARD_FIT_ERRORS = [
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

    # Elliptical fit parameter columns (adds theta to STANDARD layout)
    ELLIPTICAL_FIT_PARAMS = [
        "xc",     # X center coordinate
        "yc",     # Y center coordinate
        "s_x",    # X sigma (width, rotated)
        "s_y",    # Y sigma (width, rotated)
        "theta",  # Rotation angle (radians)
        "bg_B",   # Background (Blue channel)
        "bg_G",   # Background (Green channel)
        "bg_R",   # Background (Red channel)
        "A_B",    # Amplitude (Blue channel)
        "A_G",    # Amplitude (Green channel)
        "A_R",    # Amplitude (Red channel)
        "chi_sqr",  # Chi-squared goodness of fit
        "frame",    # Frame number
    ]

    # Elliptical fit error columns
    ELLIPTICAL_FIT_ERRORS = [
        "xc_err",
        "yc_err",
        "s_x_err",
        "s_y_err",
        "theta_err",
        "bg_B_err",
        "bg_G_err",
        "bg_R_err",
        "A_B_err",
        "A_G_err",
        "A_R_err",
    ]

    @classmethod
    def get_all_columns(cls):
        """Get all column names (parameters + errors).

        Returns:
            list: Combined list of all parameter and error column names
        """
        return cls.STANDARD_FIT_PARAMS + cls.STANDARD_FIT_ERRORS

    @classmethod
    def get_elliptical_columns(cls):
        """Get all column names for elliptical fitting (parameters + errors).

        Returns:
            list: Combined list of elliptical parameter and error column names
        """
        return cls.ELLIPTICAL_FIT_PARAMS + cls.ELLIPTICAL_FIT_ERRORS
