"""
drift_correction/_base.py

Base types and abstract interfaces for drift correction.
Extracted from DriftCorrectionFunctions.py for better code organisation.

:authors: Claude Code (based on Joerg Schnitzbauer, Maximilian Thomas Strauss, Hongqiang Ma, Maomao Chen)
:copyright: Copyright (c) 2025 pyBayerSMLM
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple, Union, Dict, Any, List
import warnings

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from Constants import DriftConstants


class DriftMethod(Enum):
    """Enumeration of available drift correction methods."""

    AIM = "aim"
    FIDUCIAL = "fiducial"  # Fiducial-based drift correction using picked localisations
    AUTO = "auto"  # Automatically select based on data characteristics


class DriftCorrectionError(Exception):
    """Custom exception for drift correction errors."""

    pass


@dataclass
class DriftParameters:
    """Parameters for drift correction algorithms.

    AIM distances (`intersect_d`, `roi_r`) are in camera pixels and are computed
    automatically from `pixel_size_nm` if left as None.  Pass `pixel_size_nm` for
    your camera (e.g. ``DriftConstants.ZWO_PIXEL_SIZE_NM`` for ZWO) and the AIM
    parameters scale correctly without any other changes.

    Attributes:
        segmentation: Time interval for drift tracking (frames)
        pixel_size_nm: Camera pixel size in nm; used to scale AIM distances.
            Defaults to Ximea (69 nm). Use DriftConstants.ZWO_PIXEL_SIZE_NM for ZWO.
        intersect_d: Intersection distance for AIM (camera pixels).
            If None, computed as DriftConstants.AIM_INTERSECT_DISTANCE_NM / pixel_size_nm.
        roi_r: Search region radius for AIM (camera pixels).
            If None, computed as DriftConstants.AIM_ROI_RADIUS_NM / pixel_size_nm.
        progress_callback: Optional progress callback function
        display: Whether to display drift plots
        fiducial_threshold_percentile: Histogram percentile threshold for fiducial detection
        fiducial_box_size_nm: Box size for fiducial detection in nanometers
        fiducial_min_frames_fraction: Minimum fraction of frames for valid fiducial
        fiducial_histogram_bins: Number of bins for histogram analysis
        auto_detect_fiducials: Whether to automatically detect fiducials if no group field exists
    """

    segmentation: int = DriftConstants.DEFAULT_SEGMENTATION_FRAMES
    pixel_size_nm: float = DriftConstants.XIMEA_PIXEL_SIZE_NM
    intersect_d: Optional[float] = None  # computed in __post_init__ from pixel_size_nm
    roi_r: Optional[float] = None        # computed in __post_init__ from pixel_size_nm
    progress_callback: Optional[Callable[[int], None]] = None
    display: bool = False
    # Fiducial detection parameters with sensible defaults
    fiducial_threshold_percentile: float = 99.0
    fiducial_box_size_nm: float = DriftConstants.FIDUCIAL_BOX_SIZE_NM
    fiducial_min_frames_fraction: float = 0.8
    fiducial_histogram_bins: int = 256
    auto_detect_fiducials: bool = True

    def __post_init__(self):
        if self.intersect_d is None:
            self.intersect_d = DriftConstants.AIM_INTERSECT_DISTANCE_NM / self.pixel_size_nm
        if self.roi_r is None:
            self.roi_r = DriftConstants.AIM_ROI_RADIUS_NM / self.pixel_size_nm

    def validate(self) -> None:
        """Validate parameter values."""
        if self.segmentation <= 0:
            raise DriftCorrectionError("Segmentation must be positive")
        if self.intersect_d <= 0:
            raise DriftCorrectionError("Intersection distance must be positive")
        # Fiducial validation
        if not (0 < self.fiducial_threshold_percentile <= 100):
            raise DriftCorrectionError(
                "Fiducial threshold percentile must be between 0 and 100"
            )
        if self.fiducial_box_size_nm <= 0:
            raise DriftCorrectionError("Fiducial box size must be positive")
        if not (0 < self.fiducial_min_frames_fraction <= 1):
            raise DriftCorrectionError(
                "Fiducial minimum frames fraction must be between 0 and 1"
            )
        if self.fiducial_histogram_bins <= 0:
            raise DriftCorrectionError("Fiducial histogram bins must be positive")
        if self.roi_r <= 0:
            raise DriftCorrectionError("ROI radius must be positive")


@dataclass
class DriftResult:
    """Result of drift correction operation.

    Attributes:
        drift_x: Drift in x-direction for each frame
        drift_y: Drift in y-direction for each frame
        drift_z: Drift in z-direction for each frame (if 3D)
        method_used: Which method was actually used
        metadata: Additional metadata about the correction
    """

    drift_x: np.ndarray
    drift_y: np.ndarray
    drift_z: Optional[np.ndarray] = None
    method_used: DriftMethod = DriftMethod.AIM
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_rec_array(self) -> np.recarray:
        """Convert drift to numpy record array format."""
        if self.drift_z is not None:
            return np.rec.array(
                (self.drift_x, self.drift_y, self.drift_z),
                dtype=[("xc", "f"), ("yc", "f"), ("z", "f")],
            )
        else:
            return np.rec.array(
                (self.drift_x, self.drift_y), dtype=[("x", "f"), ("y", "f")]
            )


@dataclass
class FiducialDetectionResult:
    """Results from fiducial detection process.

    Attributes:
        picks: List of (x, y) coordinates for detected fiducials
        picked_localisations: List of localisation arrays, one per fiducial
        detection_image: Rendered image used for detection
        locs_with_groups: Original localisations with group field added
        n_fiducials: Number of detected fiducials
        detection_params: Parameters used for detection
        metadata: Additional detection metadata
    """

    picks: List[Tuple[float, float]]
    picked_localisations: List[np.recarray]
    detection_image: np.ndarray
    locs_with_groups: np.recarray
    n_fiducials: int
    detection_params: Dict[str, Any]
    metadata: Dict[str, Any]


# Note: SegmentationHandler and CoordinateProcessor are now imported from CoordinateProcessing.py
# This reduces duplication and improves code organization


class DriftCorrector(ABC):
    """Abstract base class for drift correction strategies."""

    @abstractmethod
    def calculate_drift(
        self, locs: np.recarray, info: list, params: DriftParameters
    ) -> DriftResult:
        """Calculate drift correction for localisations.

        Args:
            locs: Localisation data
            info: Metadata list
            params: Drift correction parameters

        Returns:
            Drift correction result
        """
        pass

    @abstractmethod
    def supports_3d(self) -> bool:
        """Return whether this corrector supports 3D drift correction."""
        pass

    def correct_drift(
        self, locs: np.recarray, info: list, params: DriftParameters
    ) -> Tuple[np.recarray, DriftResult]:
        """Complete drift correction workflow.

        Args:
            locs: Localisation data
            info: Metadata list
            params: Drift correction parameters

        Returns:
            Tuple of (corrected_locs, drift_result)
        """
        from CoordinateProcessing import CoordinateProcessor

        # Validate inputs
        params.validate()
        CoordinateProcessor.validate_localisations(locs)

        # Calculate drift
        drift_result = self.calculate_drift(locs, info, params)

        # Apply correction
        corrected_locs = CoordinateProcessor.apply_drift_correction(
            locs, drift_result.drift_x, drift_result.drift_y, drift_result.drift_z
        )

        return corrected_locs, drift_result
