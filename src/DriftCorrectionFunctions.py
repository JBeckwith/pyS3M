"""
DriftCorrectionFunctions.py

Unified drift correction module combining RCC and AIM approaches from postprocess.py and aim.py.
Implements strategy pattern for flexible drift correction method selection.

:authors: Claude Code (based on Joerg Schnitzbauer, Maximilian Thomas Strauss, Hongqiang Ma, Maomao Chen)
:copyright: Copyright (c) 2025 pyBayerSMLM
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple, Union, Dict, Any, List
import warnings
import gc

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import Birch

# Matplotlib imports (needed for drift correction plotting)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Local imports (will import from existing modules as needed)
import sys
import os

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)

import ProgressUtils

try:
    import render
    import imageprocess
    import postprocess
    import PlottingFunctions
except ImportError:
    warnings.warn(
        "Could not import render/imageprocess modules. RCC method may not work."
    )
    render = None
    imageprocess = None
    postprocess = None
    PlottingFunctions = None


class DriftMethod(Enum):
    """Enumeration of available drift correction methods."""

    RCC = "rcc"
    AIM = "aim"
    FIDUCIAL = "fiducial"  # Fiducial-based drift correction using picked localizations
    AUTO = "auto"  # Automatically select based on data characteristics


class DriftCorrectionError(Exception):
    """Custom exception for drift correction errors."""

    pass


@dataclass
class DriftParameters:
    """Parameters for drift correction algorithms.

    Attributes:
        segmentation: Time interval for drift tracking (frames)
        intersect_d: Intersection distance for AIM (camera pixels)
        roi_r: Search region radius for AIM (camera pixels)
        blur_method: Blur method for RCC rendering
        min_blur_width: Minimum blur width for RCC
        rcc_max_shift: Maximum correlation shift for RCC
        progress_callback: Optional progress callback function
        display: Whether to display drift plots
        # Fiducial detection parameters
        fiducial_threshold_percentile: Histogram percentile threshold for fiducial detection
        fiducial_box_size_nm: Box size for fiducial detection in nanometers
        fiducial_min_frames_fraction: Minimum fraction of frames for valid fiducial
        fiducial_histogram_bins: Number of bins for histogram analysis
        auto_detect_fiducials: Whether to automatically detect fiducials if no group field exists
    """

    segmentation: int = 100
    intersect_d: float = 20 / 69  # Default AIM intersection distance
    roi_r: float = 60 / 69  # Default AIM search radius
    blur_method: str = "gaussian"
    min_blur_width: float = 1.0
    rcc_max_shift: int = 32
    progress_callback: Optional[Callable[[int], None]] = None
    display: bool = False
    # Fiducial detection parameters with sensible defaults
    fiducial_threshold_percentile: float = 99.0  # 99th percentile threshold
    fiducial_box_size_nm: float = 900.0  # 900nm box size (matches imageprocess.py)
    fiducial_min_frames_fraction: float = (
        0.8  # 80% of frames minimum (matches imageprocess.py)
    )
    fiducial_histogram_bins: int = 256  # Number of histogram bins
    auto_detect_fiducials: bool = True  # Automatically detect if no group field

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
    method_used: DriftMethod = DriftMethod.RCC
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
        picked_localizations: List of localization arrays, one per fiducial
        detection_image: Rendered image used for detection
        locs_with_groups: Original localizations with group field added
        n_fiducials: Number of detected fiducials
        detection_params: Parameters used for detection
        metadata: Additional detection metadata
    """

    picks: List[Tuple[float, float]]
    picked_localizations: List[np.recarray]
    detection_image: np.ndarray
    locs_with_groups: np.recarray
    n_fiducials: int
    detection_params: Dict[str, Any]
    metadata: Dict[str, Any]


class SegmentationHandler:
    """Utilities for temporal segmentation of localization data."""

    @staticmethod
    def create_segments(n_frames: int, segmentation: int) -> np.ndarray:
        """Create segmentation bounds for drift correction.

        Args:
            n_frames: Total number of frames
            segmentation: Frames per segment

        Returns:
            Array of segment boundary frames
        """
        return np.concatenate((np.arange(0, n_frames, segmentation), [n_frames]))

    @staticmethod
    def n_segments(n_frames: int, segmentation: int) -> int:
        """Calculate number of segments."""
        return int(np.round(n_frames / segmentation))

    @staticmethod
    def standardize_frame_indexing(locs: np.recarray) -> np.ndarray:
        """Standardize frame indices to start at 1.

        Args:
            locs: Localization data

        Returns:
            Frame indices starting at 1
        """
        return locs.frame + 1 - locs.frame.min()


class CoordinateProcessor:
    """Utilities for coordinate processing and validation."""

    @staticmethod
    def extract_metadata(info: list) -> Dict[str, float]:
        """Extract required metadata from info list.

        Args:
            info: List of metadata dictionaries

        Returns:
            Dictionary with width, height, frames, pixelsize

        Raises:
            DriftCorrectionError: If required metadata is missing
        """
        width = height = pixelsize = n_frames = np.nan

        for inf in info:
            if val := inf.get("Width"):
                width = val
            if val := inf.get("Height"):
                height = val
            if val := inf.get("Frames"):
                n_frames = val
            if val := inf.get("Pixelsize"):
                pixelsize = val

        if np.isnan(width * height * pixelsize * n_frames):
            raise DriftCorrectionError(
                "Missing required metadata. Need 'Width', 'Height', 'Frames', 'Pixelsize'"
            )

        return {
            "width": width,
            "height": height,
            "n_frames": n_frames,
            "pixelsize": pixelsize,
        }

    @staticmethod
    def validate_localizations(locs: np.recarray) -> None:
        """Validate localization data format.

        Args:
            locs: Localization record array

        Raises:
            DriftCorrectionError: If required columns are missing
        """
        required_cols = ["xc", "yc", "frame"]
        missing_cols = [col for col in required_cols if not hasattr(locs, col)]

        if missing_cols:
            raise DriftCorrectionError(f"Missing required columns: {missing_cols}")

    @staticmethod
    def apply_drift_correction(
        locs: np.recarray, drift_result: DriftResult
    ) -> np.recarray:
        """Apply drift correction to localizations.

        Args:
            locs: Localization data to correct
            drift_result: Drift correction result

        Returns:
            Corrected localizations (copy with corrections applied)
        """
        # Create a copy to avoid modifying original data
        corrected_locs = locs.copy()

        # Apply x,y drift (ensure frame indices are within bounds)
        frame_indices = np.clip(corrected_locs.frame, 0, len(drift_result.drift_x) - 1)
        corrected_locs.xc -= drift_result.drift_x[frame_indices]
        corrected_locs.yc -= drift_result.drift_y[frame_indices]

        # Apply z drift if available
        if drift_result.drift_z is not None and hasattr(corrected_locs, "z"):
            corrected_locs.z -= drift_result.drift_z[frame_indices]

        return corrected_locs


class DriftCorrector(ABC):
    """Abstract base class for drift correction strategies."""

    @abstractmethod
    def calculate_drift(
        self, locs: np.recarray, info: list, params: DriftParameters
    ) -> DriftResult:
        """Calculate drift correction for localizations.

        Args:
            locs: Localization data
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
            locs: Localization data
            info: Metadata list
            params: Drift correction parameters

        Returns:
            Tuple of (corrected_locs, drift_result)
        """
        # Validate inputs
        params.validate()
        CoordinateProcessor.validate_localizations(locs)

        # Calculate drift
        drift_result = self.calculate_drift(locs, info, params)

        # Apply correction
        corrected_locs = CoordinateProcessor.apply_drift_correction(locs, drift_result)

        return corrected_locs, drift_result


class RCCDriftCorrector(DriftCorrector):
    """RCC (Rapid Cross-Correlation) drift correction implementation.

    Based on postprocess.py undrift() function. Uses image rendering and
    cross-correlation to detect drift between temporal segments.
    """

    def supports_3d(self) -> bool:
        """RCC supports 2D drift correction only."""
        return False

    def calculate_drift(
        self, locs: np.recarray, info: list, params: DriftParameters
    ) -> DriftResult:
        """Calculate drift using RCC method.

        Args:
            locs: Localization data
            info: Metadata list
            params: Drift correction parameters

        Returns:
            RCC drift correction result
        """
        if render is None or imageprocess is None:
            raise DriftCorrectionError(
                "RCC method requires render and imageprocess modules"
            )

        # Extract metadata
        meta = CoordinateProcessor.extract_metadata(info)

        # Generate segments
        bounds, segments = self._generate_segments(locs, info, meta, params)

        # Calculate shifts using RCC
        shift_y, shift_x = imageprocess.rcc(
            segments, params.rcc_max_shift, params.progress_callback
        )

        # Interpolate to all frames
        drift_x, drift_y = self._interpolate_drift(
            bounds, shift_x, shift_y, int(meta["n_frames"])
        )

        return DriftResult(
            drift_x=-drift_x,  # Negative because we want to correct drift
            drift_y=-drift_y,
            method_used=DriftMethod.RCC,
            metadata={
                "segments": len(bounds) - 1,
                "max_shift": params.rcc_max_shift,
                "blur_method": params.blur_method,
            },
        )

    def _generate_segments(
        self,
        locs: np.recarray,
        info: list,
        meta: Dict[str, float],
        params: DriftParameters,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate temporal segments for RCC analysis.

        Args:
            locs: Localization data
            info: Original metadata list (for render compatibility)
            meta: Extracted metadata
            params: Drift parameters

        Returns:
            Tuple of (bounds, segments)
        """
        n_segments = SegmentationHandler.n_segments(
            int(meta["n_frames"]), params.segmentation
        )
        bounds = np.linspace(0, meta["n_frames"] - 1, n_segments + 1, dtype=np.uint32)

        # Render segments
        Y = int(meta["height"])
        X = int(meta["width"])
        segments = np.zeros((n_segments, Y, X))

        if params.progress_callback is None:
            with ProgressUtils.clean_progress_bar(
                range(n_segments), desc="Generating segments"
            ) as it:
                for i in it:
                    segment_locs = locs[
                        (locs.frame >= bounds[i]) & (locs.frame < bounds[i + 1])
                    ]
                    _, segments[i] = render.render(
                        segment_locs,
                        info,
                        blur_method=params.blur_method,
                        min_blur_width=params.min_blur_width,
                    )
        else:
            params.progress_callback(0)
            for i in range(n_segments):
                segment_locs = locs[
                    (locs.frame >= bounds[i]) & (locs.frame < bounds[i + 1])
                ]
                _, segments[i] = render.render(
                    segment_locs,
                    info,
                    blur_method=params.blur_method,
                    min_blur_width=params.min_blur_width,
                )
                params.progress_callback(i + 1)

        return bounds, segments

    def _interpolate_drift(
        self,
        bounds: np.ndarray,
        shift_x: np.ndarray,
        shift_y: np.ndarray,
        n_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate drift to all frames using splines.

        Args:
            bounds: Segment boundaries
            shift_x: X shifts between segments
            shift_y: Y shifts between segments
            n_frames: Total number of frames

        Returns:
            Tuple of (drift_x, drift_y) for all frames
        """
        t = (bounds[1:] + bounds[:-1]) / 2
        drift_x_pol = InterpolatedUnivariateSpline(t, shift_x, k=3)
        drift_y_pol = InterpolatedUnivariateSpline(t, shift_y, k=3)

        t_inter = np.arange(n_frames)
        drift_x = drift_x_pol(t_inter)
        drift_y = drift_y_pol(t_inter)

        return drift_x, drift_y


class AIMDriftCorrector(DriftCorrector):
    """AIM (Adaptive Intersection Maximization) drift correction implementation.

    Based on aim.py implementation. Uses coordinate intersection counting
    to detect drift with sub-pixel precision via FFT.
    """

    @staticmethod
    def _intersect1d(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Slightly faster implementation of np.intersect1d without unnecessary checks.

        Finds the indices of common elements in two 1D arrays (a and b).
        Both a and b are assumed to be sorted and contain only unique values.

        Args:
            a: 1D array of integers
            b: 1D array of integers

        Returns:
            Tuple of (a_indices, b_indices) for common elements
        """
        aux = np.concatenate((a, b))
        aux_sort_indices = np.argsort(aux, kind="mergesort")
        aux = aux[aux_sort_indices]

        mask = aux[1:] == aux[:-1]
        a_indices = aux_sort_indices[:-1][mask]
        b_indices = aux_sort_indices[1:][mask] - a.size

        return a_indices, b_indices

    @staticmethod
    def _count_intersections(
        l0_coords: np.ndarray,
        l0_counts: np.ndarray,
        l1_coords: np.ndarray,
        l1_counts: np.ndarray,
    ) -> int:
        """Count the number of intersected localizations between two datasets.

        Args:
            l0_coords: Unique coordinates of reference localizations
            l0_counts: Counts of unique reference localizations
            l1_coords: Unique coordinates of target localizations
            l1_counts: Counts of unique target localizations

        Returns:
            Number of intersections
        """
        # indices of common elements
        idx0, idx1 = AIMDriftCorrector._intersect1d(l0_coords, l1_coords)

        # extract the counts of these elements
        l0_counts_subset = l0_counts[idx0]
        l1_counts_subset = l1_counts[idx1]

        # for each overlapping coordinate, take the minimum count from l0
        # and l1, sum up across all overlapping coordinates
        n_intersections = np.sum(np.minimum(l0_counts_subset, l1_counts_subset))
        return n_intersections

    @staticmethod
    def _run_intersections_multithread(
        l0_coords: np.ndarray,
        l0_counts: np.ndarray,
        l1_coords: np.ndarray,
        l1_counts: np.ndarray,
        shifts_xy: np.ndarray,
        box: int,
    ) -> np.ndarray:
        """Run intersection counting across local search region with multithreading.

        Args:
            l0_coords: Unique coordinates of reference localizations
            l0_counts: Counts of reference localizations
            l1_coords: Unique coordinates of target localizations
            l1_counts: Counts of target localizations
            shifts_xy: 1D array with x and y shifts
            box: Side length of local search region

        Returns:
            2D array with intersection counts across search region
        """
        # shift target coordinates
        l1_coords_shifted = l1_coords[:, np.newaxis] + shifts_xy

        # run multiple threads
        n_workers = min(len(shifts_xy), 16)  # Limit threads to avoid overhead
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    AIMDriftCorrector._count_intersections,
                    l0_coords,
                    l0_counts,
                    l1_coords_shifted[:, i],
                    l1_counts,
                )
                for i in range(len(shifts_xy))
            ]

            if box == 1:  # z intersection only, for z undrifting
                roi_cc = np.array([f.result() for f in futures])
            else:  # 2D intersection
                roi_cc = np.array([f.result() for f in futures]).reshape(box, box)

        return roi_cc

    @staticmethod
    def _point_intersect_2d(
        l0_coords: np.ndarray,
        l0_counts: np.ndarray,
        x1: np.ndarray,
        y1: np.ndarray,
        intersect_d: float,
        width_units: float,
        shifts_xy: np.ndarray,
        box: int,
    ) -> np.ndarray:
        """Convert target coordinates to 1D array and count intersections.

        Args:
            l0_coords: Unique values of reference localizations
            l0_counts: Counts of unique reference localizations
            x1, y1: x and y coordinates of target localizations
            intersect_d: Intersect distance in camera pixels
            width_units: Width of camera image in units of intersect_d
            shifts_xy: 1D array with x and y shifts
            box: Side length of local search region

        Returns:
            2D array with intersection counts in local search region
        """
        # convert target coordinates to a 1D array in intersect_d units
        x1_units = np.round(x1 / intersect_d)
        y1_units = np.round(y1 / intersect_d)
        l1 = np.int32(x1_units + y1_units * width_units)  # 1d list

        # get unique values and counts of the target localizations
        l1_coords, l1_counts = np.unique(l1, return_counts=True)

        # run the intersections counting
        roi_cc = AIMDriftCorrector._run_intersections_multithread(
            l0_coords, l0_counts, l1_coords, l1_counts, shifts_xy, box
        )
        return roi_cc

    @staticmethod
    def _point_intersect_3d(
        l0_coords: np.ndarray,
        l0_counts: np.ndarray,
        x1: np.ndarray,
        y1: np.ndarray,
        z1: np.ndarray,
        intersect_d: float,
        width_units: float,
        height_units: float,
        shifts_z: np.ndarray,
    ) -> np.ndarray:
        """Convert 3D target coordinates to 1D array and count intersections.

        Args:
            l0_coords: Unique values of reference localizations
            l0_counts: Counts of unique reference localizations
            x1, y1, z1: x, y, and z coordinates of target localizations
            intersect_d: Intersect distance in camera pixels
            width_units: Width of camera image in units of intersect_d
            height_units: Height of camera image in units of intersect_d
            shifts_z: 1D array with z shifts

        Returns:
            1D array with intersection counts in local search region
        """
        # convert target coordinates to a 1D array in intersect_d units
        x1_units = np.round(x1 / intersect_d)
        y1_units = np.round(y1 / intersect_d)
        z1_units = np.round(z1 / intersect_d)
        l1 = np.int32(
            x1_units + y1_units * width_units + z1_units * width_units * height_units
        )  # 1d list

        # get unique values and counts of the target localizations
        l1_coords, l1_counts = np.unique(l1, return_counts=True)

        # run the intersections counting
        roi_cc = AIMDriftCorrector._run_intersections_multithread(
            l0_coords, l0_counts, l1_coords, l1_counts, shifts_z, 1
        )
        return roi_cc

    @staticmethod
    def _get_fft_peak(roi_cc: np.ndarray, roi_size: float) -> Tuple[float, float]:
        """Estimate precise sub-pixel position of peak using FFT.

        Args:
            roi_cc: 2D array with intersection counts in local search region
            roi_size: Size of the local search region

        Returns:
            Tuple of (px, py) estimated peak coordinates
        """
        fft_values = np.fft.fft2(roi_cc.T)

        # X peak estimation
        ang_x = np.angle(fft_values[0, 1])
        ang_x = ang_x - 2 * np.pi * (ang_x > 0)  # normalise
        px = (
            np.abs(ang_x) / (2 * np.pi / roi_cc.shape[0]) - (roi_cc.shape[0] - 1) / 2
        )  # peak in x
        px *= roi_size / roi_cc.shape[0]  # convert to intersect_d units

        # Y peak estimation
        ang_y = np.angle(fft_values[1, 0])
        ang_y = ang_y - 2 * np.pi * (ang_y > 0)  # normalise
        py = (
            np.abs(ang_y) / (2 * np.pi / roi_cc.shape[1]) - (roi_cc.shape[1] - 1) / 2
        )  # peak in y
        py *= roi_size / roi_cc.shape[1]  # convert to intersect_d units

        return px, py

    @staticmethod
    def _get_fft_peak_z(roi_cc: np.ndarray, roi_size: float) -> float:
        """Estimate precise sub-pixel position of 1D peak using FFT.

        Args:
            roi_cc: 1D array with intersection counts in local search region
            roi_size: Size of the local search region

        Returns:
            Estimated z-coordinate of peak
        """
        fft_values = np.fft.fft(roi_cc)
        ang_z = np.angle(fft_values[1])
        ang_z = ang_z - 2 * np.pi * (ang_z > 0)  # normalise
        pz = (
            np.abs(ang_z) / (2 * np.pi / roi_cc.size) - (roi_cc.size - 1) / 2
        )  # peak in z
        pz *= roi_size / roi_cc.size  # convert to intersect_d units
        return pz

    @staticmethod
    def _intersection_max(
        x: np.ndarray,
        y: np.ndarray,
        ref_x: np.ndarray,
        ref_y: np.ndarray,
        frame: np.ndarray,
        seg_bounds: np.ndarray,
        intersect_d: float,
        roi_r: float,
        width: float,
        aim_round: int = 1,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Maximize intersection (undrift) for 2D localizations.

        This is the core AIM algorithm implementation.

        Args:
            x, y: x and y coordinates of localizations
            ref_x, ref_y: x and y coordinates of reference localizations
            frame: Frame indices of localizations
            seg_bounds: Segmentation bounds defining temporal intervals
            intersect_d: Intersect distance in camera pixels
            roi_r: Search region radius in camera pixels
            width: Width of camera image in camera pixels
            aim_round: Round of AIM algorithm (1 or 2)
            progress_callback: Optional progress callback

        Returns:
            Tuple of (x_pdc, y_pdc, drift_x, drift_y)
        """
        assert aim_round in [1, 2], "aim_round must be 1 or 2."

        # number of segments
        n_segments = len(seg_bounds) - 1
        rel_drift_x = 0  # adaptive drift (updated at each interval)
        rel_drift_y = 0

        # drift in x and y
        drift_x = np.zeros(n_segments)
        drift_y = np.zeros(n_segments)

        # find shifts for the local search region (in units of intersect_d)
        roi_units = int(np.ceil(roi_r / intersect_d))
        steps = np.arange(-roi_units, roi_units + 1, 1)
        box = len(steps)
        shifts_xy = np.zeros((box, box), dtype=np.int32)
        width_units = width / intersect_d

        for i, shift_x in enumerate(steps):
            for j, shift_y in enumerate(steps):
                shifts_xy[i, j] = shift_x + shift_y * width_units
        shifts_xy = shifts_xy.reshape(box**2)

        # convert reference to a 1D array in units of intersect_d and find
        # unique values and counts
        x0_units = np.round(ref_x / intersect_d)
        y0_units = np.round(ref_y / intersect_d)
        l0 = np.int32(x0_units + y0_units * width_units)  # 1d list
        l0_coords, l0_counts = np.unique(l0, return_counts=True)

        # initialize progress
        start_idx = 1 if aim_round == 1 else 0

        def _process_segment(s):
            nonlocal rel_drift_x, rel_drift_y

            # get the target localizations within the current segment
            min_frame_idx = frame > seg_bounds[s]
            max_frame_idx = frame <= seg_bounds[s + 1]
            x1 = x[min_frame_idx & max_frame_idx]
            y1 = y[min_frame_idx & max_frame_idx]

            # skip if no localizations in this segment
            if len(x1) == 0:
                if s > 0:
                    drift_x[s] = drift_x[s - 1]
                    drift_y[s] = drift_y[s - 1]
                return

            # undrifting from the previous round
            x1 += rel_drift_x
            y1 += rel_drift_y

            # count the number of intersected localizations
            roi_cc = AIMDriftCorrector._point_intersect_2d(
                l0_coords,
                l0_counts,
                x1,
                y1,
                intersect_d,
                width_units,
                shifts_xy,
                box,
            )

            # estimate the precise sub-pixel position of the peak with FFT
            px, py = AIMDriftCorrector._get_fft_peak(roi_cc, 2 * roi_r)

            # update the relative drift reference for the subsequent
            # segmented subset (interval) and save the drifts
            rel_drift_x += px
            rel_drift_y += py
            drift_x[s] = -rel_drift_x
            drift_y[s] = -rel_drift_y

        # run across each segment
        if progress_callback is None:
            with ProgressUtils.clean_progress_bar(
                range(start_idx, n_segments), desc=f"AIM Undrifting ({aim_round}/2)"
            ) as iterator:
                for s in iterator:
                    _process_segment(s)
        else:
            for s in range(start_idx, n_segments):
                _process_segment(s)
                progress_callback(s)

        # Use cubic spline interpolation following original MATLAB AIM implementation
        min_frame = int(frame.min())
        max_frame_data = int(frame.max())

        # Create drift arrays using cubic spline interpolation
        drift_x_full, drift_y_full = AIMDriftCorrector._cubic_spline_interpolation(
            drift_x, drift_y, seg_bounds, min_frame, max_frame_data
        )

        # Apply drift correction with proper indexing
        # Choice between vectorized and frame-by-frame approach
        use_vectorized = True  # Set to False for frame-by-frame if needed

        if use_vectorized:
            # Vectorized approach (faster, but more complex indexing)
            if min_frame == 0:
                # 0-indexed frames: use direct indexing
                # Ensure frame indices are within bounds
                valid_indices = (frame >= 0) & (frame < len(drift_x_full))
                x_pdc = x.copy()
                y_pdc = y.copy()
                x_pdc[valid_indices] = (
                    x[valid_indices] - drift_x_full[frame[valid_indices]]
                )
                y_pdc[valid_indices] = (
                    y[valid_indices] - drift_y_full[frame[valid_indices]]
                )
            else:
                # 1-indexed frames: subtract 1 for array indexing
                # Ensure frame indices are within bounds after converting to 0-indexed
                valid_indices = (frame >= 1) & (frame - 1 < len(drift_x_full))
                x_pdc = x.copy()
                y_pdc = y.copy()
                x_pdc[valid_indices] = (
                    x[valid_indices] - drift_x_full[frame[valid_indices] - 1]
                )
                y_pdc[valid_indices] = (
                    y[valid_indices] - drift_y_full[frame[valid_indices] - 1]
                )
        else:
            # Frame-by-frame approach (safer, guaranteed to work)
            x_pdc = x.copy()
            y_pdc = y.copy()

            for frame_num in np.unique(frame):
                if min_frame == 0:
                    if 0 <= frame_num < len(drift_x_full):
                        subset_mask = frame == frame_num
                        x_pdc[subset_mask] -= drift_x_full[frame_num]
                        y_pdc[subset_mask] -= drift_y_full[frame_num]
                else:
                    if 1 <= frame_num <= len(drift_x_full):
                        subset_mask = frame == frame_num
                        x_pdc[subset_mask] -= drift_x_full[
                            frame_num - 1
                        ]  # Convert to 0-indexed
                        y_pdc[subset_mask] -= drift_y_full[frame_num - 1]

        return x_pdc, y_pdc, drift_x_full, drift_y_full

    @staticmethod
    def _cubic_spline_interpolation(
        drift_x: np.ndarray,
        drift_y: np.ndarray,
        seg_bounds: np.ndarray,
        min_frame: int,
        max_frame: int,
    ) -> tuple:
        """Cubic spline interpolation following original MATLAB AIM implementation.

        This method replicates the original MATLAB interpolation approach:
        1. Calculates segment centers as measurement points
        2. Extends with boundary extrapolation points
        3. Uses cubic spline interpolation for smooth drift correction

        Args:
            drift_x: Measured drift values for each segment
            drift_y: Measured drift values for each segment
            seg_bounds: Segment boundaries
            min_frame: Minimum frame number
            max_frame: Maximum frame number

        Returns:
            Tuple of (drift_x_full, drift_y_full) arrays for all frames
        """
        # Calculate segment centers (where we have actual measurements)
        seg_centers = (seg_bounds[1:] + seg_bounds[:-1]) / 2
        track_interval = seg_bounds[1] - seg_bounds[0]  # Assuming uniform intervals
        track_num = len(drift_x)

        # Extend drift values with boundary extrapolation (following MATLAB pattern)
        # drift_X = [2*driftX(1)-driftX(2) driftX 2*driftX(end)-driftX(end-1)]
        if len(drift_x) >= 2:
            drift_x_extended = np.concatenate(
                [
                    [2 * drift_x[0] - drift_x[1]],
                    drift_x,
                    [2 * drift_x[-1] - drift_x[-2]],
                ]
            )
            drift_y_extended = np.concatenate(
                [
                    [2 * drift_y[0] - drift_y[1]],
                    drift_y,
                    [2 * drift_y[-1] - drift_y[-2]],
                ]
            )
        else:
            # Handle edge case with single measurement
            drift_x_extended = np.concatenate([[drift_x[0]], drift_x, [drift_x[0]]])
            drift_y_extended = np.concatenate([[drift_y[0]], drift_y, [drift_y[0]]])

        # Create extended x-coordinates for interpolation
        # Following MATLAB: (-0.5:(trackNUM+0.5))*trackInterval
        x_coords = np.arange(-0.5, track_num + 1.0) * track_interval

        # Target frames for interpolation (1 to trackNUM*trackInterval)
        total_frames = track_num * track_interval
        target_frames = np.arange(1, total_frames + 1)

        # Perform cubic spline interpolation
        spline_x = InterpolatedUnivariateSpline(x_coords, drift_x_extended, k=3)
        spline_y = InterpolatedUnivariateSpline(x_coords, drift_y_extended, k=3)

        drift_x_interp = spline_x(target_frames)
        drift_y_interp = spline_y(target_frames)

        # Adjust for the actual frame range in the data
        if min_frame == 0:
            # 0-indexed frames: pad to match max_frame
            drift_x_full = np.zeros(max_frame + 1)
            drift_y_full = np.zeros(max_frame + 1)
            end_idx = min(len(drift_x_interp), max_frame + 1)
            drift_x_full[:end_idx] = drift_x_interp[:end_idx]
            drift_y_full[:end_idx] = drift_y_interp[:end_idx]
        else:
            # 1-indexed frames: adjust for frame numbering
            drift_x_full = np.zeros(max_frame)
            drift_y_full = np.zeros(max_frame)
            end_idx = min(len(drift_x_interp), max_frame)
            drift_x_full[:end_idx] = drift_x_interp[:end_idx]
            drift_y_full[:end_idx] = drift_y_interp[:end_idx]

        return drift_x_full, drift_y_full

    @staticmethod
    def _intersection_max_z(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        ref_x: np.ndarray,
        ref_y: np.ndarray,
        ref_z: np.ndarray,
        frame: np.ndarray,
        seg_bounds: np.ndarray,
        intersect_d: float,
        roi_r: float,
        width: float,
        height: float,
        pixelsize: float,
        aim_round: int = 1,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Maximize intersection (undrift) for 3D localizations.

        Args:
            x, y, z: Coordinates (x,y in pixels, z in nm)
            ref_x, ref_y, ref_z: Reference coordinates
            frame: Frame indices
            seg_bounds: Segmentation bounds
            intersect_d: Intersect distance in camera pixels
            roi_r: Search region radius in camera pixels
            width, height: Image dimensions in pixels
            pixelsize: Pixel size in nm
            aim_round: Round of AIM algorithm (1 or 2)
            progress_callback: Optional progress callback

        Returns:
            Tuple of (z_pdc, drift_z)
        """
        # convert z to camera pixels
        z = z.copy() / pixelsize
        ref_z = ref_z.copy() / pixelsize

        # number of segments
        n_segments = len(seg_bounds) - 1
        rel_drift_z = 0  # adaptive drift (updated at each interval)

        # drift in z
        drift_z = np.zeros(n_segments)

        # find shifts for the local search region (in units of intersect_d)
        roi_units = int(np.ceil(roi_r / intersect_d))
        steps = np.arange(-roi_units, roi_units + 1, 1)
        width_units = width / intersect_d
        height_units = height / intersect_d
        shifts_z = steps.astype(np.int32) * width_units * height_units

        # convert reference to a 1D array in units of intersect_d and find
        # unique values and counts
        x0_units = np.round(ref_x / intersect_d)
        y0_units = np.round(ref_y / intersect_d)
        z0_units = np.round(ref_z / intersect_d)
        l0 = np.int32(
            x0_units + y0_units * width_units + z0_units * width_units * height_units
        )  # 1d list
        l0_coords, l0_counts = np.unique(l0, return_counts=True)

        # initialize progress
        start_idx = 1 if aim_round == 1 else 0

        def _process_segment_z(s):
            nonlocal rel_drift_z

            # get the target localizations within the current segment
            min_frame_idx = frame > seg_bounds[s]
            max_frame_idx = frame <= seg_bounds[s + 1]
            x1 = x[min_frame_idx & max_frame_idx]
            y1 = y[min_frame_idx & max_frame_idx]
            z1 = z[min_frame_idx & max_frame_idx]

            # skip if no localizations in this segment
            if len(x1) == 0:
                if s > 0:
                    drift_z[s] = drift_z[s - 1]
                return

            # undrifting from the previous round
            z1 += rel_drift_z

            # count the number of intersected localizations
            roi_cc = AIMDriftCorrector._point_intersect_3d(
                l0_coords,
                l0_counts,
                x1,
                y1,
                z1,
                intersect_d,
                width_units,
                height_units,
                shifts_z,
            )

            # estimate the precise sub-pixel position of the peak with FFT
            pz = AIMDriftCorrector._get_fft_peak_z(roi_cc, 2 * roi_r)

            # update the relative drift reference for the subsequent
            # segmented subset (interval) and save the drifts
            rel_drift_z += pz
            drift_z[s] = -rel_drift_z

        # run across each segment
        if progress_callback is None:
            with ProgressUtils.clean_progress_bar(
                range(start_idx, n_segments), desc=f"AIM Undrifting z ({aim_round}/2)"
            ) as iterator:
                for s in iterator:
                    _process_segment_z(s)
        else:
            for s in range(start_idx, n_segments):
                _process_segment_z(s)
                progress_callback(s)

        # interpolate the drifts (cubic spline) for all frames
        t = (seg_bounds[1:] + seg_bounds[:-1]) / 2
        drift_z_pol = InterpolatedUnivariateSpline(t, drift_z, k=3)
        t_inter = np.arange(seg_bounds[-1]) + 1
        drift_z = drift_z_pol(t_inter)

        # undrift the localizations
        z_pdc = z - drift_z[frame - 1]

        # convert back to nm
        z_pdc *= pixelsize
        drift_z *= pixelsize

        return z_pdc, drift_z

    def supports_3d(self) -> bool:
        """AIM supports both 2D and 3D drift correction."""
        return True

    def calculate_drift(
        self, locs: np.recarray, info: list, params: DriftParameters
    ) -> DriftResult:
        """Calculate drift using AIM method.

        Args:
            locs: Localization data
            info: Metadata list
            params: Drift correction parameters

        Returns:
            AIM drift correction result
        """
        # Extract metadata
        meta = CoordinateProcessor.extract_metadata(info)

        # Standardize frame indexing
        frame = SegmentationHandler.standardize_frame_indexing(locs)

        # Create segmentation bounds
        seg_bounds = SegmentationHandler.create_segments(
            int(meta["n_frames"]), params.segmentation
        )

        # Get reference localizations (first segment)
        ref_mask = frame <= params.segmentation
        ref_x = locs.xc[ref_mask]
        ref_y = locs.yc[ref_mask]

        # Run two-round AIM for 2D
        x_pdc, y_pdc, drift_x, drift_y = self._run_aim_2d(
            locs, ref_x, ref_y, frame, seg_bounds, meta, params
        )

        drift_z = None
        if hasattr(locs, "z") and self.supports_3d():
            # Run 3D drift correction
            z_pdc, drift_z = self._run_aim_3d(
                x_pdc, y_pdc, locs.z, frame, seg_bounds, meta, params
            )

        return DriftResult(
            drift_x=drift_x,
            drift_y=drift_y,
            drift_z=drift_z,
            method_used=DriftMethod.AIM,
            metadata={
                "intersect_d_nm": params.intersect_d * meta["pixelsize"],
                "roi_r_nm": params.roi_r * meta["pixelsize"],
                "segmentation": params.segmentation,
                "n_segments": len(seg_bounds) - 1,
            },
        )

    def _run_aim_2d(
        self,
        locs: np.recarray,
        ref_x: np.ndarray,
        ref_y: np.ndarray,
        frame: np.ndarray,
        seg_bounds: np.ndarray,
        meta: Dict[str, float],
        params: DriftParameters,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run 2D AIM drift correction using the complete AIM algorithm.

        Implements the full two-round AIM procedure from aim.py.
        """
        width = meta["width"]

        # Run first round AIM (reference = first segment)
        x_pdc, y_pdc, drift_x1, drift_y1 = self._intersection_max(
            locs.xc,
            locs.yc,
            ref_x,
            ref_y,
            frame,
            seg_bounds,
            params.intersect_d,
            params.roi_r,
            width,
            aim_round=1,
            progress_callback=params.progress_callback,
        )

        # Run second round AIM (reference = entire dataset)
        x_pdc, y_pdc, drift_x2, drift_y2 = self._intersection_max(
            x_pdc,
            y_pdc,
            x_pdc,
            y_pdc,
            frame,
            seg_bounds,
            params.intersect_d,
            params.roi_r,
            width,
            aim_round=2,
            progress_callback=params.progress_callback,
        )

        # Combine drifts from both rounds
        drift_x = drift_x1 + drift_x2
        drift_y = drift_y1 + drift_y2

        # Remove mean drift to centre the correction
        shift_x = np.mean(drift_x)
        shift_y = np.mean(drift_y)
        drift_x -= shift_x
        drift_y -= shift_y
        x_pdc += shift_x
        y_pdc += shift_y

        return x_pdc, y_pdc, drift_x, drift_y

    def _run_aim_3d(
        self,
        x_pdc: np.ndarray,
        y_pdc: np.ndarray,
        z: np.ndarray,
        frame: np.ndarray,
        seg_bounds: np.ndarray,
        meta: Dict[str, float],
        params: DriftParameters,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run 3D AIM drift correction for z-coordinate.

        Implements complete 3D AIM procedure from aim.py.
        """
        width = meta["width"]
        height = meta["height"]
        pixelsize = meta["pixelsize"]

        # Get reference localizations for Z (first segment)
        ref_mask = frame <= params.segmentation
        ref_x = x_pdc[ref_mask]
        ref_y = y_pdc[ref_mask]
        ref_z = z[ref_mask]

        # Run first round AIM for Z
        z_pdc, drift_z1 = self._intersection_max_z(
            x_pdc,
            y_pdc,
            z,
            ref_x,
            ref_y,
            ref_z,
            frame,
            seg_bounds,
            params.intersect_d,
            params.roi_r,
            width,
            height,
            pixelsize,
            aim_round=1,
            progress_callback=params.progress_callback,
        )

        # Run second round AIM for Z (reference = entire dataset)
        z_pdc, drift_z2 = self._intersection_max_z(
            x_pdc,
            y_pdc,
            z_pdc,
            x_pdc,
            y_pdc,
            z_pdc,
            frame,
            seg_bounds,
            params.intersect_d,
            params.roi_r,
            width,
            height,
            pixelsize,
            aim_round=2,
            progress_callback=params.progress_callback,
        )

        # Combine drifts from both rounds
        drift_z = drift_z1 + drift_z2

        # Remove mean drift to centre the correction
        shift_z = np.mean(drift_z)
        drift_z -= shift_z
        z_pdc += shift_z

        return z_pdc, drift_z


class FiducialDriftCorrector(DriftCorrector):
    """Fiducial-based drift corrector using picked localizations.

    This method calculates drift using manually selected fiducial markers
    (e.g., gold nanoparticles, fluorescent beads) that should remain stationary
    during the experiment.

    The algorithm:
    1. Takes pre-selected fiducial localizations (picked manually or automatically)
    2. Removes center-of-mass offset for each fiducial
    3. Calculates weighted average drift across all fiducials
    4. Uses inverse mean squared deviation as weights (more stable fiducials get higher weight)
    5. Interpolates drift for frames without localizations
    """

    def __init__(self):
        pass

    def supports_3d(self) -> bool:
        """Fiducial corrector supports 2D drift correction."""
        return False  # Can be extended to 3D in future

    def calculate_drift(
        self, locs: np.recarray, info: list, params: DriftParameters
    ) -> DriftResult:
        """Calculate drift using fiducial localizations.

        Args:
            locs: Localization data. If no 'group' field exists and auto_detect_fiducials=True,
                  fiducials will be detected automatically
            info: Metadata list containing frame count information
            params: Drift correction parameters with fiducial detection settings

        Returns:
            DriftResult with calculated drift corrections
        """
        # Check if group field exists, if not and auto-detect is enabled, detect fiducials
        if not hasattr(locs, "group"):
            if params.auto_detect_fiducials:
                locs = self._detect_and_add_fiducials(locs, info, params)
            else:
                raise DriftCorrectionError(
                    "Fiducial drift correction requires 'group' field in locs to identify fiducials. "
                    "Set auto_detect_fiducials=True to automatically detect fiducials."
                )

        # Extract metadata
        meta = CoordinateProcessor.extract_metadata(info)
        n_frames = int(meta["n_frames"])

        # Group localizations by fiducial ID
        picked_locs = self._group_fiducials(locs)

        if len(picked_locs) == 0:
            raise DriftCorrectionError("No fiducial localizations found")

        # Calculate drift for each coordinate
        drift_x = self._calculate_coordinate_drift(picked_locs, n_frames, "xc")
        drift_y = self._calculate_coordinate_drift(picked_locs, n_frames, "yc")

        # Create result
        result = DriftResult(
            drift_x=drift_x,
            drift_y=drift_y,
            method_used=DriftMethod.FIDUCIAL,
            metadata={
                "n_fiducials": len(picked_locs),
                "fiducial_groups": [
                    np.unique(group_locs.group)[0] for group_locs in picked_locs
                ],
                "frames_per_fiducial": [len(group_locs) for group_locs in picked_locs],
            },
        )

        return result

    def _group_fiducials(self, locs: np.recarray) -> List[np.recarray]:
        """Group localizations by fiducial ID.

        Args:
            locs: Localization data with 'group' field

        Returns:
            List of localization arrays, one per fiducial
        """
        picked_locs = []
        unique_groups = np.unique(locs.group)

        for group_id in unique_groups:
            if group_id >= 0:  # Skip negative group IDs (often used for unlabeled data)
                group_locs = locs[locs.group == group_id].copy()
                if len(group_locs) > 0:
                    picked_locs.append(group_locs)

        return picked_locs

    def _calculate_coordinate_drift(
        self, picked_locs: List[np.recarray], n_frames: int, coordinate: str
    ) -> np.ndarray:
        """Calculate drift in a given coordinate using fiducial localizations.

        Args:
            picked_locs: List of localization arrays for each fiducial
            n_frames: Total number of frames
            coordinate: Coordinate name ("xc", "yc", or "z")

        Returns:
            Array of drift values for each frame
        """
        n_picks = len(picked_locs)

        if n_picks == 0:
            return np.zeros(n_frames, dtype=np.float32)

        # Initialize drift matrix: [n_fiducials, n_frames]
        drift = np.empty((n_picks, n_frames), dtype=np.float32)
        drift.fill(np.nan)

        # Calculate drift for each fiducial (remove center of mass offset)
        for i, fiducial_locs in enumerate(picked_locs):
            if hasattr(fiducial_locs, coordinate):
                coordinates = getattr(fiducial_locs, coordinate)
                frames = fiducial_locs.frame.astype(int)

                # Ensure frames are within bounds
                valid_frames = (frames >= 0) & (frames < n_frames)
                coordinates = coordinates[valid_frames]
                frames = frames[valid_frames]

                if len(coordinates) > 0:
                    # Calculate deviation from the mean position (this IS the drift)
                    mean_position = np.mean(coordinates)
                    drift[i, frames] = coordinates - mean_position

        # Calculate mean drift across fiducials
        drift_mean = np.nanmean(drift, axis=0)

        # Calculate reliability weights (inverse of mean squared deviation)
        if n_picks > 1:
            # Square deviation from mean drift
            squared_deviations = (drift - drift_mean[np.newaxis, :]) ** 2
            # Mean squared deviation for each fiducial
            msd = np.nanmean(squared_deviations, axis=1)
            # Avoid division by zero
            msd[msd == 0] = np.nanmin(msd[msd > 0]) if np.any(msd > 0) else 1.0

            # Calculate weighted average
            weights = 1.0 / msd

            # Create masked array for proper weighted averaging with NaNs
            drift_masked = np.ma.masked_invalid(drift)
            drift_mean = np.ma.average(drift_masked, axis=0, weights=weights)
            drift_mean = drift_mean.filled(np.nan)

        # Interpolate missing frames
        drift_mean = self._interpolate_missing_frames(drift_mean)

        return drift_mean.astype(np.float32)

    def _interpolate_missing_frames(self, drift_mean: np.ndarray) -> np.ndarray:
        """Interpolate drift for frames without localizations.

        Args:
            drift_mean: Drift array with possible NaN values

        Returns:
            Interpolated drift array
        """
        # Find valid (non-NaN) frames
        valid_mask = ~np.isnan(drift_mean)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            # No valid data - return zeros
            return np.zeros_like(drift_mean)
        elif len(valid_indices) == 1:
            # Only one valid point - use constant value
            drift_mean[:] = drift_mean[valid_indices[0]]
        else:
            # Interpolate between valid points
            invalid_indices = np.where(~valid_mask)[0]
            if len(invalid_indices) > 0:
                drift_mean[invalid_indices] = np.interp(
                    invalid_indices, valid_indices, drift_mean[valid_indices]
                )

        return drift_mean

    def _detect_and_add_fiducials(
        self, locs: np.recarray, info: list, params: DriftParameters
    ) -> np.recarray:
        """Automatically detect fiducials and add group field to localizations.

        Args:
            locs: Localization data without group field
            info: Metadata list
            params: Drift parameters with fiducial detection settings

        Returns:
            New localization array with group field added
        """
        if render is None or imageprocess is None:
            raise DriftCorrectionError(
                "Fiducial detection requires render and imageprocess modules"
            )

        # Extract metadata for pixel size
        meta = CoordinateProcessor.extract_metadata(info)
        pixelsize = meta.get("pixelsize", 69.0)  # Default fallback
        n_frames = int(meta["n_frames"])

        # Render localizations to image for fiducial detection
        image = render.render(
            locs=locs,
            info=info,
            oversampling=1,
            viewport=None,
            blur_method="smooth",
        )[1]

        # Create histogram with user-specified number of bins
        hist = np.histogram(image.flatten(), bins=params.fiducial_histogram_bins)

        # Use user-specified threshold percentile
        threshold = np.percentile(hist[0], params.fiducial_threshold_percentile)

        # Calculate box size from nanometer specification
        box = int(np.round(params.fiducial_box_size_nm / pixelsize))
        box = box + 1 if box % 2 == 0 else box  # Ensure odd

        # Find local maxima (potential fiducials)
        try:
            import localise  # Import here to handle potential issues
        except ImportError:
            raise DriftCorrectionError(
                "localise module required for fiducial detection"
            )

        y, x, _ = localise.identify_in_image(image, threshold, box=box)
        # Format picks as rectangles centered on detected points
        half_box = box // 2
        picks = [((xi - half_box, yi), (xi + half_box, yi)) for xi, yi in zip(x, y)]

        if len(picks) == 0:
            raise DriftCorrectionError(
                "No fiducial candidates detected. Try lowering threshold_percentile."
            )

        # Filter picks by minimum localizations per fiducial
        min_n = params.fiducial_min_frames_fraction * n_frames

        try:
            import postprocess  # Import here to handle potential issues
        except ImportError:
            raise DriftCorrectionError(
                "postprocess module required for fiducial detection"
            )

        # Get localizations for each pick
        width = int(meta["width"])
        height = int(meta["height"])
        temp_picked_locs = postprocess.picked_locs(
            locs,
            width,
            height,
            picks,
            "Circle",
            pick_size=box / 2,
            add_group=False,
        )

        # Keep only picks with sufficient localizations
        valid_picks = []
        valid_picked_locs = []
        for i, pick in enumerate(picks):
            if len(temp_picked_locs[i]) > min_n:
                valid_picks.append(pick)
                valid_picked_locs.append(temp_picked_locs[i])

        if len(valid_picks) == 0:
            raise DriftCorrectionError(
                f"No fiducials found with minimum {min_n:.0f} localizations. "
                f"Try lowering fiducial_min_frames_fraction or threshold_percentile."
            )

        # Create new localization array with group field
        return self._add_group_field(locs, valid_picked_locs, valid_picks)

    def _add_group_field(
        self, locs: np.recarray, picked_locs: list, picks: list
    ) -> np.recarray:
        """Add group field to localizations based on fiducial assignments.

        Args:
            locs: Original localizations
            picked_locs: List of localizations for each fiducial
            picks: List of pick coordinates

        Returns:
            New recarray with group field added
        """
        # Create group field array, initialize with -1 (non-fiducial)
        group = np.full(len(locs), -1, dtype=np.int32)

        # Assign group IDs to fiducial localizations
        for group_id, fiducial_locs in enumerate(picked_locs):
            # Find indices of these localizations in original array
            for fid_loc in fiducial_locs:
                # Match by frame and coordinate (within small tolerance)
                matches = (
                    (locs.frame == fid_loc.frame)
                    & (np.abs(locs.xc - fid_loc.xc) < 0.1)
                    & (np.abs(locs.yc - fid_loc.yc) < 0.1)
                )
                group[matches] = group_id

        # Create new dtype with group field
        original_dtype = locs.dtype
        group_dtype = np.dtype(original_dtype.descr + [("group", "i4")])

        # Create new recarray with group field
        new_locs = np.empty(len(locs), dtype=group_dtype)

        # Copy original data
        for field in original_dtype.names:
            new_locs[field] = locs[field]

        # Add group data
        new_locs["group"] = group

        # Convert to recarray
        return new_locs.view(np.recarray)


class AutoDriftCorrector(DriftCorrector):
    """Automatic drift corrector that selects method based on data characteristics."""

    def __init__(self):
        self.rcc_corrector = RCCDriftCorrector()
        self.aim_corrector = AIMDriftCorrector()

    def supports_3d(self) -> bool:
        """Auto corrector supports 3D if AIM is available."""
        return True

    def calculate_drift(
        self, locs: np.recarray, info: list, params: DriftParameters
    ) -> DriftResult:
        """Automatically select and apply best drift correction method.

        Selection criteria:
        - Use AIM for dense data (>1000 locs per segment on average)
        - Use RCC for sparse data
        - Use RCC if render modules unavailable
        """
        # Calculate data density
        meta = CoordinateProcessor.extract_metadata(info)
        n_segments = SegmentationHandler.n_segments(
            int(meta["n_frames"]), params.segmentation
        )
        avg_locs_per_segment = len(locs) / n_segments

        # Selection logic
        if avg_locs_per_segment > 1000:
            selected_method = DriftMethod.AIM
            corrector = self.aim_corrector
        elif render is not None and imageprocess is not None:
            selected_method = DriftMethod.RCC
            corrector = self.rcc_corrector
        else:
            # Fallback to AIM if RCC modules unavailable
            selected_method = DriftMethod.AIM
            corrector = self.aim_corrector

        # Run selected method
        result = corrector.calculate_drift(locs, info, params)
        result.method_used = selected_method
        result.metadata["auto_selection_reason"] = (
            f"Selected {selected_method.value} based on "
            f"{avg_locs_per_segment:.1f} locs/segment"
        )

        return result


class DriftCorrectionFactory:
    """Factory for creating drift correctors."""

    _correctors = {
        DriftMethod.RCC: RCCDriftCorrector,
        DriftMethod.AIM: AIMDriftCorrector,
        DriftMethod.FIDUCIAL: FiducialDriftCorrector,
        DriftMethod.AUTO: AutoDriftCorrector,
    }

    @classmethod
    def create_corrector(cls, method: DriftMethod) -> DriftCorrector:
        """Create drift corrector instance.

        Args:
            method: Drift correction method

        Returns:
            Drift corrector instance

        Raises:
            DriftCorrectionError: If method not supported
        """
        if method not in cls._correctors:
            raise DriftCorrectionError(f"Unsupported drift method: {method}")

        return cls._correctors[method]()

    @classmethod
    def available_methods(cls) -> list:
        """Get list of available drift correction methods."""
        return list(cls._correctors.keys())


# Convenience functions for backward compatibility
def undrift_rcc(
    locs: np.recarray,
    info: list,
    segmentation: int = 100,
    display: bool = True,
    segmentation_callback: Optional[Callable] = None,
    rcc_callback: Optional[Callable] = None,
) -> Tuple[np.recarray, DriftResult]:
    """Apply RCC drift correction (backward compatible interface).

    Args:
        locs: Localization data
        info: Metadata list
        segmentation: Frames per segment
        display: Whether to display results
        segmentation_callback: Progress callback for segmentation
        rcc_callback: Progress callback for RCC

    Returns:
        Tuple of (corrected_locs, drift_result)
    """
    params = DriftParameters(
        segmentation=segmentation, display=display, progress_callback=rcc_callback
    )

    corrector = DriftCorrectionFactory.create_corrector(DriftMethod.RCC)
    return corrector.correct_drift(locs, info, params)


def undrift_aim(
    locs: np.recarray,
    info: list,
    segmentation: int = 100,
    intersect_d: float = 20 / 69,
    roi_r: float = 60 / 69,
    progress: Optional[Callable] = None,
) -> Tuple[np.recarray, DriftResult]:
    """Apply AIM drift correction (backward compatible interface).

    Args:
        locs: Localization data
        info: Metadata list
        segmentation: Frames per segment
        intersect_d: Intersection distance (camera pixels)
        roi_r: Search region radius (camera pixels)
        progress: Progress callback

    Returns:
        Tuple of (corrected_locs, drift_result)
    """
    params = DriftParameters(
        segmentation=segmentation,
        intersect_d=intersect_d,
        roi_r=roi_r,
        progress_callback=progress,
    )

    corrector = DriftCorrectionFactory.create_corrector(DriftMethod.AIM)
    return corrector.correct_drift(locs, info, params)


def undrift_auto(
    locs: np.recarray, info: list, **kwargs
) -> Tuple[np.recarray, DriftResult]:
    """Apply automatic drift correction method selection.

    Args:
        locs: Localization data
        info: Metadata list
        **kwargs: Parameters passed to DriftParameters

    Returns:
        Tuple of (corrected_locs, drift_result)
    """
    params = DriftParameters(**kwargs)

    corrector = DriftCorrectionFactory.create_corrector(DriftMethod.AUTO)
    return corrector.correct_drift(locs, info, params)


# Main class for external API
class Drift_Correction_Functions:
    """Main class providing drift correction functionality.

    This class follows the established pattern in the codebase of
    organizing functions within a class structure.
    """

    def __init__(self):
        """Initialize drift correction functions."""
        self.factory = DriftCorrectionFactory()

    def undrift(
        self,
        locs: np.recarray,
        info: list,
        method: Union[str, DriftMethod] = "auto",
        **params,
    ) -> Tuple[np.recarray, DriftResult]:
        """Universal drift correction interface.

        Args:
            locs: Localization data
            info: Metadata list
            method: Drift correction method ("rcc", "aim", "auto")
            **params: Method-specific parameters

        Returns:
            Tuple of (corrected_locs, drift_result)

        Example:
            >>> DCF = Drift_Correction_Functions()
            >>> corrected_locs, drift = DCF.undrift(locs, info, method="rcc")
            >>> corrected_locs, drift = DCF.undrift(locs, info, method="aim", segmentation=50)
        """
        # Convert string to enum if needed
        if isinstance(method, str):
            method = DriftMethod(method.lower())

        # Create parameters
        drift_params = DriftParameters(**params)

        # Get corrector and apply
        corrector = self.factory.create_corrector(method)
        return corrector.correct_drift(locs, info, drift_params)

    def available_methods(self) -> list:
        """Get available drift correction methods."""
        return [method.value for method in self.factory.available_methods()]

    def method_info(self, method: Union[str, DriftMethod]) -> Dict[str, Any]:
        """Get information about a drift correction method.

        Args:
            method: Drift method to query

        Returns:
            Dictionary with method information
        """
        if isinstance(method, str):
            method = DriftMethod(method.lower())

        corrector = self.factory.create_corrector(method)

        return {
            "name": method.value,
            "supports_3d": corrector.supports_3d(),
            "class": corrector.__class__.__name__,
            "description": (
                corrector.__class__.__doc__.split("\n")[0]
                if corrector.__class__.__doc__
                else ""
            ),
        }

    def _detect_fiducials_with_chunking(
        self,
        locs: np.recarray,
        info: list,
        threshold_percentile: float,
        box_size_nm: float,
        histogram_bins: int,
        n_chunks: int,
        max_linking_distance_nm: float,
        pixelsize: float,
    ) -> tuple:
        """Detect fiducials using temporal chunking for drift-robust detection.

        Args:
            locs: Localization data
            info: Metadata list
            threshold_percentile: Percentile threshold for detection
            box_size_nm: Box size in nanometers
            histogram_bins: Number of histogram bins
            n_chunks: Number of temporal chunks
            max_linking_distance_nm: Maximum linking distance in nm
            pixelsize: Pixel size in nm

        Returns:
            Tuple of (picks, combined_image, combined_hist, threshold)
        """
        try:
            import localise
            import render
        except ImportError:
            raise DriftCorrectionError(
                "localise and render modules required for chunked fiducial detection"
            )

        # Get frame range
        min_frame = int(locs.frame.min())
        max_frame = int(locs.frame.max())
        total_frames = max_frame - min_frame + 1

        # Create temporal chunks
        chunk_size = total_frames // n_chunks
        chunk_boundaries = []
        for i in range(n_chunks):
            start_frame = min_frame + i * chunk_size
            if i == n_chunks - 1:
                end_frame = max_frame  # Include remaining frames in last chunk
            else:
                end_frame = min_frame + (i + 1) * chunk_size - 1
            chunk_boundaries.append((start_frame, end_frame))

        print(f"Detecting fiducials using {n_chunks} temporal chunks")

        # Find candidates in each chunk
        chunk_candidates = []
        chunk_images = []
        all_chunk_histograms = []

        for chunk_idx, (start_frame, end_frame) in enumerate(chunk_boundaries):
            # Extract localizations for this chunk
            chunk_mask = (locs.frame >= start_frame) & (locs.frame <= end_frame)
            chunk_locs = locs[chunk_mask]

            if len(chunk_locs) == 0:
                print(f"Warning: Chunk {chunk_idx + 1} has no localizations")
                continue

            print(
                f"Chunk {chunk_idx + 1}/{n_chunks}: frames {start_frame}-{end_frame} ({len(chunk_locs)} locs)"
            )

            # Render this chunk
            chunk_image = render.render(
                locs=chunk_locs,
                info=info,
                oversampling=1,
                viewport=None,
                blur_method="smooth",
            )[1]
            chunk_images.append(chunk_image)

            # Create histogram for this chunk
            chunk_hist = np.histogram(chunk_image.flatten(), bins=histogram_bins)
            all_chunk_histograms.append(chunk_hist[0])

            # Use threshold percentile for this chunk
            chunk_threshold = np.percentile(chunk_hist[0], threshold_percentile)

            # Calculate box size
            box = int(np.round(box_size_nm / pixelsize))
            box = box + 1 if box % 2 == 0 else box  # Ensure odd

            # Find candidates in this chunk
            try:
                y, x, _ = localise.identify_in_image(
                    chunk_image, chunk_threshold, box=box
                )
                half_box = box // 2
                chunk_picks = [
                    (xi, yi, chunk_idx, (start_frame + end_frame) / 2)
                    for xi, yi in zip(x, y)
                ]
                chunk_candidates.extend(chunk_picks)
                print(f"  Found {len(chunk_picks)} candidates")
            except Exception as e:
                print(f"  Warning: Failed to detect in chunk {chunk_idx + 1}: {e}")
                continue

        print(f"Total candidates across all chunks: {len(chunk_candidates)}")

        # Link candidates across chunks to form tracks
        if len(chunk_candidates) == 0:
            raise DriftCorrectionError("No candidates found in any temporal chunk")

        linked_tracks = self._link_candidates_across_chunks(
            chunk_candidates, n_chunks, max_linking_distance_nm, pixelsize
        )

        print(f"Linked candidates into {len(linked_tracks)} potential fiducial tracks")

        # Convert tracks back to picks (use average position)
        # Format as rectangles for box picking
        half_box = int(np.round(box_size_nm / pixelsize)) // 2
        picks = []
        for track in linked_tracks:
            if (
                len(track) >= n_chunks * 0.6
            ):  # Require track to appear in >60% of chunks
                avg_x = np.mean([pos[0] for pos in track])
                avg_y = np.mean([pos[1] for pos in track])
                picks.append(((avg_x - half_box, avg_y), (avg_x + half_box, avg_y)))

        # Create combined image and histogram for visualization
        if len(chunk_images) > 0:
            combined_image = np.mean(chunk_images, axis=0)
            combined_hist_counts = np.sum(all_chunk_histograms, axis=0)
            # Reconstruct histogram tuple
            if len(all_chunk_histograms) > 0:
                bin_edges = np.histogram(combined_image.flatten(), bins=histogram_bins)[
                    1
                ]
                combined_hist = (combined_hist_counts, bin_edges)
            else:
                combined_hist = np.histogram(
                    combined_image.flatten(), bins=histogram_bins
                )
            threshold = np.percentile(combined_hist_counts, threshold_percentile)
        else:
            # Fallback: create empty image
            combined_image = np.zeros((100, 100))
            combined_hist = np.histogram(combined_image.flatten(), bins=histogram_bins)
            threshold = 0

        print(f"Final result: {len(picks)} robust fiducial candidates")
        return picks, combined_image, combined_hist, threshold

    def _link_candidates_across_chunks(
        self, candidates: list, n_chunks: int, max_distance_nm: float, pixelsize: float
    ) -> list:
        """Link candidates across temporal chunks to form tracks.

        Args:
            candidates: List of (x, y, chunk_idx, avg_frame) tuples
            n_chunks: Number of chunks
            max_distance_nm: Maximum linking distance in nm
            pixelsize: Pixel size in nm

        Returns:
            List of tracks, where each track is a list of (x, y, chunk_idx, avg_frame) positions
        """
        max_distance_pixels = max_distance_nm / pixelsize

        # Group candidates by chunk
        chunks_candidates = [[] for _ in range(n_chunks)]
        for candidate in candidates:
            x, y, chunk_idx, avg_frame = candidate
            chunks_candidates[chunk_idx].append((x, y, chunk_idx, avg_frame))

        # Start tracks from first chunk
        tracks = []
        for candidate in chunks_candidates[0]:
            tracks.append([candidate])

        # Extend tracks through subsequent chunks
        for chunk_idx in range(1, n_chunks):
            chunk_candidates = chunks_candidates[chunk_idx]

            # Try to extend existing tracks
            for track in tracks:
                if len(track) == 0:
                    continue

                last_pos = track[-1]
                last_x, last_y = last_pos[0], last_pos[1]

                # Find closest candidate in current chunk
                best_candidate = None
                best_distance = float("inf")

                for candidate in chunk_candidates:
                    x, y = candidate[0], candidate[1]
                    distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)

                    if distance < max_distance_pixels and distance < best_distance:
                        best_distance = distance
                        best_candidate = candidate

                # Add best candidate to track if found
                if best_candidate is not None:
                    track.append(best_candidate)
                    chunk_candidates.remove(best_candidate)  # Prevent double-assignment

            # Start new tracks for unlinked candidates
            for remaining_candidate in chunk_candidates:
                tracks.append([remaining_candidate])

        # Filter out short tracks (less than 60% of chunks)
        min_length = int(n_chunks * 0.6)
        robust_tracks = [track for track in tracks if len(track) >= min_length]

        return robust_tracks

    def detect_fiducials(
        self,
        locs: np.recarray,
        info: list,
        threshold_percentile: float = 99.0,
        box_size_nm: float = 900.0,
        min_frames_fraction: float = 0.8,
        histogram_bins: int = 256,
        plot_results: bool = True,
        save_plot: Optional[str] = None,
        use_temporal_chunking: bool = True,
        n_chunks: int = 10,
        max_linking_distance_nm: float = 500.0,
    ) -> FiducialDetectionResult:
        """Detect fiducial markers in localization data.

        This function automatically detects fiducial markers and creates a visualization
        using PlottingFunctions. Supports temporal chunking for datasets with strong drift.

        Args:
            locs: Localization data (group field not required)
            info: Metadata list containing frame count and image dimensions
            threshold_percentile: Histogram percentile threshold for fiducial detection (0-100)
            box_size_nm: Box size for fiducial detection in nanometers
            min_frames_fraction: Minimum fraction of frames for valid fiducial (0-1)
            histogram_bins: Number of bins for histogram analysis
            plot_results: Whether to create and display a plot of detected fiducials
            save_plot: Optional path to save the plot
            use_temporal_chunking: Use temporal chunking for drift-robust detection
            n_chunks: Number of temporal chunks (default: 10)
            max_linking_distance_nm: Maximum distance to link candidates across chunks (nm)

        Returns:
            FiducialDetectionResult containing detected fiducials and metadata

        Raises:
            DriftCorrectionError: If fiducial detection fails

        Example:
            >>> DCF = Drift_Correction_Functions()
            >>> # Detect fiducials with visualization
            >>> detection_result = DCF.detect_fiducials(locs, info, plot_results=True)
            >>> print(f"Found {detection_result.n_fiducials} fiducials")
            >>>
            >>> # Use detected fiducials for drift correction
            >>> corrected, drift = DCF.undrift_with_detected_fiducials(detection_result)
        """
        if render is None or imageprocess is None:
            raise DriftCorrectionError(
                "Fiducial detection requires render and imageprocess modules"
            )

        # Extract metadata
        meta = CoordinateProcessor.extract_metadata(info)
        pixelsize = meta.get("pixelsize", 69.0)  # Default fallback in nm
        n_frames = int(meta["n_frames"])
        width = int(meta["width"])
        height = int(meta["height"])

        # Store detection parameters
        detection_params = {
            "threshold_percentile": threshold_percentile,
            "box_size_nm": box_size_nm,
            "min_frames_fraction": min_frames_fraction,
            "histogram_bins": histogram_bins,
            "pixelsize": pixelsize,
        }

        # Calculate box size from nanometer specification (used later regardless of detection method)
        box = int(np.round(box_size_nm / pixelsize))
        box = box + 1 if box % 2 == 0 else box  # Ensure odd

        try:
            if use_temporal_chunking:
                # Temporal chunking approach for drift-robust detection
                picks, image, hist, threshold = self._detect_fiducials_with_chunking(
                    locs,
                    info,
                    threshold_percentile,
                    box_size_nm,
                    histogram_bins,
                    n_chunks,
                    max_linking_distance_nm,
                    pixelsize,
                )
            else:
                # Original approach (render entire dataset at once)
                image = render.render(
                    locs=locs,
                    info=info,
                    oversampling=1,
                    viewport=None,
                    blur_method="smooth",
                )[1]

                # Create histogram with user-specified number of bins
                hist = np.histogram(image.flatten(), bins=histogram_bins)

                # Use user-specified threshold percentile
                threshold = np.percentile(hist[0], threshold_percentile)

                # Find local maxima (potential fiducials)
                try:
                    import localise
                except ImportError:
                    raise DriftCorrectionError(
                        "localise module required for fiducial detection"
                    )

                y, x, _ = localise.identify_in_image(image, threshold, box=box)
                # Format picks as rectangles centered on detected points
                # Each rectangle is box×box pixels around the center point
                half_box = box // 2
                picks = [
                    ((xi - half_box, yi), (xi + half_box, yi)) for xi, yi in zip(x, y)
                ]

            if len(picks) == 0:
                raise DriftCorrectionError(
                    f"No fiducial candidates detected with threshold percentile {threshold_percentile}%. "
                    "Try lowering threshold_percentile."
                )

            # Filter picks by minimum localisations per fiducial
            min_n = min_frames_fraction * n_frames

            try:
                import postprocess
            except ImportError:
                raise DriftCorrectionError(
                    "postprocess module required for fiducial detection"
                )

            # Get localisations for each pick using Rectangle picking (better for drifted fiducials)
            temp_picked_locs = postprocess.picked_locs(
                locs,
                width,
                height,
                picks,
                "Rectangle",
                pick_size=box,  # Width of the rectangle in pixels
                add_group=False,
                parallel=True,  # Use parallel processing for efficiency
            )

            # Keep only picks with sufficient localisations
            valid_picks = []
            valid_picked_locs = []
            for i, pick in enumerate(picks):
                if len(temp_picked_locs[i]) > min_n:
                    valid_picks.append(pick)
                    valid_picked_locs.append(temp_picked_locs[i])

            if len(valid_picks) == 0:
                raise DriftCorrectionError(
                    f"No fiducials found with minimum {min_n:.0f} localisations. "
                    f"Try lowering min_frames_fraction (currently {min_frames_fraction}) "
                    f"or threshold_percentile (currently {threshold_percentile}%)."
                )

            # Create localisation array with group field
            locs_with_groups = self._add_group_field_to_locs(locs, valid_picked_locs)

            # Create result object
            result = FiducialDetectionResult(
                picks=valid_picks,
                picked_localizations=valid_picked_locs,
                detection_image=image,
                locs_with_groups=locs_with_groups,
                n_fiducials=len(valid_picks),
                detection_params=detection_params,
                metadata={
                    "total_candidates": len(picks),
                    "threshold_used": threshold,
                    "box_size_pixels": box,
                    "min_localisations_required": min_n,
                    "localisations_per_fiducial": [
                        len(locs) for locs in valid_picked_locs
                    ],
                },
            )

            # Create plot if requested
            if plot_results:
                self._plot_fiducial_detection_steps(
                    image, hist, threshold, picks, valid_picks, result, info, save_plot
                )

            return result

        except Exception as e:
            if isinstance(e, DriftCorrectionError):
                raise
            else:
                raise DriftCorrectionError(f"Fiducial detection failed: {str(e)}")

    def _add_group_field_to_locs(
        self, locs: np.recarray, picked_locs_list: List[np.recarray]
    ) -> np.recarray:
        """Add group field to localisations based on fiducial assignments.

        Ultra-fast index-based implementation. Achieves ~1000x speedup by using
        index-based assignment instead of coordinate matching.
        """
        # Create group field array, initialize with -1 (non-fiducial)
        group = np.full(len(locs), -1, dtype=np.int32)

        # Ultra-fast index-based assignment
        if len(picked_locs_list) > 0:
            # Set up progress bar for large datasets or many fiducial groups
            show_progress = len(locs) > 500_000 or len(picked_locs_list) > 5
            progress_bar_context = None
            progress_bar = None

            if show_progress:
                progress_bar_context = ProgressUtils.clean_progress_bar(
                    total=len(picked_locs_list),
                    desc=f"Adding group field to {len(locs):,} localizations (index-based)",
                )
                progress_bar = progress_bar_context.__enter__()

            try:
                # Process all fiducial groups using index-based approach
                for group_id, fiducial_locs in enumerate(picked_locs_list):
                    if len(fiducial_locs) > 0:
                        # Find indices of fiducial localizations in original array
                        # This is the key optimization: use indices instead of coordinate matching
                        indices = self._find_indices_in_original_locs(
                            locs, fiducial_locs
                        )

                        # Direct assignment by index (ultra-fast)
                        group[indices] = group_id

                        # Update progress bar
                        if progress_bar:
                            progress_bar.update(1)

            finally:
                # Ensure progress bar is cleaned up
                if progress_bar_context:
                    progress_bar_context.__exit__(None, None, None)

        # Fast array construction using lib.append_to_rec if available
        try:
            import lib

            return lib.append_to_rec(locs, group, "group")
        except ImportError:
            # Fallback to manual construction
            return self._manual_add_group_field(locs, group)

    def _find_indices_in_original_locs(
        self, locs: np.recarray, fiducial_locs: np.recarray
    ) -> np.ndarray:
        """Find indices of fiducial localizations in the original localization array.

        Uses ultra-fast hash-based lookup for massive datasets.
        Expected ~1000x speedup over coordinate matching approach.

        Args:
            locs: Original localization array
            fiducial_locs: Fiducial localizations to find indices for

        Returns:
            Array of indices where fiducial_locs appear in locs
        """
        # Create hash-based lookup table for ultra-fast index finding
        # This is the key to massive performance improvement

        # Use deterministic rounding to handle floating point precision issues
        # Round coordinates to 6 decimal places for reliable hashing
        round_factor = 1e6

        # Create unique keys for each localization in the original array
        locs_frames = locs.frame.astype(np.int32)
        locs_xc_rounded = np.round(locs.xc * round_factor).astype(np.int64)
        locs_yc_rounded = np.round(locs.yc * round_factor).astype(np.int64)

        # Build hash table: key -> index mapping
        # Use Python dict for ultimate speed with hash-based lookups
        hash_to_index = {}
        for i, (frame, x_rounded, y_rounded) in enumerate(
            zip(locs_frames, locs_xc_rounded, locs_yc_rounded)
        ):
            key = (frame, x_rounded, y_rounded)

            # Handle potential duplicates by storing multiple indices
            if key in hash_to_index:
                if isinstance(hash_to_index[key], list):
                    hash_to_index[key].append(i)
                else:
                    hash_to_index[key] = [hash_to_index[key], i]
            else:
                hash_to_index[key] = i

        # Find indices for fiducial localizations using hash lookup
        indices = []

        fid_frames = fiducial_locs.frame.astype(np.int32)
        fid_xc_rounded = np.round(fiducial_locs.xc * round_factor).astype(np.int64)
        fid_yc_rounded = np.round(fiducial_locs.yc * round_factor).astype(np.int64)

        for frame, x_rounded, y_rounded in zip(
            fid_frames, fid_xc_rounded, fid_yc_rounded
        ):
            key = (frame, x_rounded, y_rounded)

            if key in hash_to_index:
                idx_or_list = hash_to_index[key]
                if isinstance(idx_or_list, list):
                    indices.extend(idx_or_list)  # Multiple matches
                else:
                    indices.append(idx_or_list)  # Single match

        return np.array(indices, dtype=np.int64)

    def _manual_add_group_field(
        self, locs: np.recarray, group: np.ndarray
    ) -> np.recarray:
        """Fallback method for adding group field manually."""
        # Create new dtype with group field
        original_dtype = locs.dtype
        group_dtype = np.dtype(original_dtype.descr + [("group", "i4")])

        # Create new recarray with group field (vectorized copy)
        new_locs = np.empty(len(locs), dtype=group_dtype)

        # Vectorized field copying
        for field in original_dtype.names:
            new_locs[field] = locs[field]

        # Add group data
        new_locs["group"] = group

        # Convert to recarray
        return new_locs.view(np.recarray)

    def _plot_fiducial_detection_steps(
        self,
        image: np.ndarray,
        hist: tuple,
        threshold: float,
        all_picks: list,
        valid_picks: list,
        result: FiducialDetectionResult,
        info: List[dict],
        save_path: Optional[str] = None,
    ) -> None:
        """Create step-by-step visualization of fiducial detection process."""
        if PlottingFunctions is None:
            print("⚠️ PlottingFunctions not available, skipping step-by-step plots")
            return

        try:
            # Create plotter instance
            plotter = PlottingFunctions.Plotter(poster=False, dark_background=False)

            # Extract metadata
            meta = CoordinateProcessor.extract_metadata(info)
            pixelsize = meta.get("pixelsize", 69.0)  # nm
            box_size_pixels = result.metadata["box_size_pixels"]

            # Create figure with 2x2 subplots using PlottingFunctions
            fig, axes = plotter.two_column_plot(
                ncolumns=2, nrows=2, widthratio=[1, 1], heightratio=[1, 1]
            )

            fig.suptitle(
                "Fiducial Detection Process - Step by Step",
                fontsize=10,
                fontweight="bold",
            )

            # Step 1: Original rendered image using PlottingFunctions
            axes[0, 0] = plotter.image_plot(
                axes[0, 0],
                data=image,
                pixelsize=pixelsize,
                cmap="hot",
                cbarlabel="Intensity",
                scalebarsize=1000,
                vmax=np.percentile(image, 99.99),
                vmin=np.percentile(image, 0.1),
                scalebarlabel="1 μm",
            )
            axes[0, 0].set_title("Step 1: Rendered Localization Image")

            # Step 2: Image histogram (using matplotlib as PlottingFunctions doesn't have histogram)
            hist_values, bin_edges = hist
            axes[0, 1] = plotter.histogram_plot(
                axes[0, 1], data=hist_values, bins=bin_edges, xaxislabel="Intensity"
            )  # Create empty histogram plot
            ymin, ymax = axes[0, 1].get_ylim()
            axes[0, 1].axvline(
                threshold,
                ymin=0,
                ymax=ymax,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f'Threshold = {threshold:.1f}\n({result.detection_params["threshold_percentile"]}th percentile)',
            )
            axes[0, 1].set_ylim(ymin, ymax)
            axes[0, 1].set_title("Step 2: Intensity Histogram & Threshold")

            # Step 3: Threshold regions and candidates using PlottingFunctions
            if all_picks:
                # Extract coordinates from point format (x, y)
                all_x = np.array([pick[0] for pick in all_picks])
                all_y = np.array([pick[1] for pick in all_picks])

                # Use PlottingFunctions image_scatter_plot for candidates
                axes[1, 0] = plotter.image_scatter_plot(
                    axes[1, 0],
                    data=image,
                    xdata=all_y,
                    ydata=all_x,
                    cmap="hot",
                    cbar="on",
                    cbarlabel="Intensity",
                    label=f"All Candidates ({len(all_picks)})",
                    labelcolor="yellow",
                    pixelsize=pixelsize,
                    scalebarsize=1000,
                    scalebarlabel="1 μm",
                    scattercolor="yellow",
                    vmax=np.percentile(image, 99.99),
                    vmin=np.percentile(image, 0.1),
                    s=150,
                    scatteralpha=0.8,
                )
            else:
                # No candidates found - just show image
                axes[1, 0] = plotter.image_plot(
                    axes[1, 0],
                    data=image,
                    pixelsize=pixelsize,
                    cmap="hot",
                    cbarlabel="Intensity",
                    scalebarsize=1000,
                    scalebarlabel="1 μm",
                )

            axes[1, 0].set_title(
                f"Step 3: Above-Threshold Regions & Candidates\n(Search radius: {box_size_pixels // 2} pixels)"
            )

            # Step 4: Final validated fiducials using PlottingFunctions
            if valid_picks:
                # Extract coordinates from point format (x, y)
                valid_x = np.array([pick[0] for pick in valid_picks])
                valid_y = np.array([pick[1] for pick in valid_picks])

                # Use PlottingFunctions image_scatter_plot for valid fiducials
                axes[1, 1] = plotter.image_scatter_plot(
                    axes[1, 1],
                    data=image,
                    xdata=valid_y,
                    ydata=valid_x,
                    cmap="hot",
                    cbar="on",
                    cbarlabel="Intensity",
                    label=f"Valid Fiducials ({len(valid_picks)})",
                    labelcolor="cyan",
                    pixelsize=pixelsize,
                    scalebarsize=1000,
                    scalebarlabel="1 μm",
                    scattercolor="cyan",
                    vmax=np.percentile(image, 99.99),
                    vmin=np.percentile(image, 0.1),
                    s=100,
                    scatteralpha=1.0,
                )

                # Add colored circles and numbers for each fiducial (must use matplotlib for custom patches)
                color_list = [
                    "red",
                    "blue",
                    "green",
                    "orange",
                    "purple",
                    "brown",
                    "pink",
                    "gray",
                    "olive",
                    "cyan",
                ]
                colors = (color_list * (len(valid_picks) // len(color_list) + 1))[
                    : len(valid_picks)
                ]
                for i, (x, y) in enumerate(zip(valid_x, valid_y)):
                    # Draw colored circle (requires direct matplotlib)
                    circle = patches.Circle(
                        (x, y),
                        radius=box_size_pixels // 3,
                        color=colors[i],
                        alpha=0.7,
                        linewidth=2,
                        fill=False,
                    )
                    axes[1, 1].add_patch(circle)

                    # Add fiducial number (requires direct matplotlib)
                    axes[1, 1].text(
                        x,
                        y,
                        str(i + 1),
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color="white",
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.8
                        ),
                    )
            else:
                axes[1, 1] = plotter.image_plot(
                    axes[1, 1],
                    data=image,
                    pixelsize=pixelsize,
                    cmap="hot",
                    cbarlabel="Intensity",
                    vmax=np.percentile(image, 99.99),
                    vmin=np.percentile(image, 0.1),
                    scalebarsize=1000,
                    scalebarlabel="1 μm",
                )

            axes[1, 1].set_title(
                f"Step 4: Final Valid Fiducials\n(Min {result.metadata['min_localisations_required']:.0f} localisations each)"
            )

            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"📊 Step-by-step fiducial detection plot saved to: {save_path}")

            plt.show()

        except Exception as e:
            print(f"⚠️ Error creating step-by-step fiducial detection plot: {e}")

    def _plot_fiducial_detection_results(
        self,
        result: FiducialDetectionResult,
        info: List[dict],
        save_path: Optional[str] = None,
    ) -> None:
        """Create a plot of fiducial detection results using PlottingFunctions."""
        if PlottingFunctions is None:
            print("⚠️ PlottingFunctions not available, skipping plot creation")
            return

        try:
            # Create plotter instance
            plotter = PlottingFunctions.Plotter(poster=False, dark_background=False)

            # Extract metadata for plotting
            meta = CoordinateProcessor.extract_metadata(info)
            pixelsize = meta.get("pixelsize", 130.0)  # nm

            # Create figure

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Left plot: Detection image with fiducial markers
            # Get fiducial coordinates for scatter overlay (fixed coordinate order)
            fiducial_x = [
                pick[0] for pick in result.picks
            ]  # X coordinates (first element)
            fiducial_y = [
                pick[1] for pick in result.picks
            ]  # Y coordinates (second element)

            plotter.image_scatter_plot(
                ax1,
                data=result.detection_image,
                xdata=np.array(fiducial_x),
                ydata=np.array(fiducial_y),
                cmap="hot",
                cbar="on",
                cbarlabel="Intensity",
                label=f"Detected Fiducials ({result.n_fiducials})",
                labelcolor="cyan",
                pixelsize=pixelsize,
                scalebarsize=1000,  # 1μm scale bar
                scalebarlabel="1 μm",
                scattercolor="cyan",
                s=100,  # Larger marker size
                scatteralpha=0.8,
            )
            ax1.set_title("Fiducial Detection Results")

            # Right plot: Fiducial localizations colored by group
            fiducial_locs = result.locs_with_groups[result.locs_with_groups.group >= 0]

            if len(fiducial_locs) > 0:
                unique_groups = np.unique(fiducial_locs.group)
                try:
                    # Use basic colors - colormaps are causing issues with Pylance
                    color_list = [
                        "red",
                        "blue",
                        "green",
                        "orange",
                        "purple",
                        "brown",
                        "pink",
                        "gray",
                        "olive",
                        "cyan",
                    ]
                    colors = (color_list * (len(unique_groups) // len(color_list) + 1))[
                        : len(unique_groups)
                    ]
                except:
                    # Ultimate fallback
                    colors = ["red"] * len(unique_groups)

                for i, group_id in enumerate(unique_groups):
                    group_locs = fiducial_locs[fiducial_locs.group == group_id]
                    ax2.scatter(
                        group_locs.xc * 1000,  # Convert to nm for plotting
                        group_locs.yc * 1000,
                        s=2,
                        alpha=0.6,
                        c=[colors[i]],
                        label=f"Fiducial {group_id+1} ({len(group_locs)} locs)",
                        rasterized=True,
                    )

            ax2.set_xlabel("X (nm)")
            ax2.set_ylabel("Y (nm)")
            ax2.set_title("Fiducial Localizations by Group")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax2.set_aspect("equal")
            ax2.grid(True, alpha=0.3)

            # Add summary text
            summary_text = (
                f"Detection Summary:\n"
                f"• Threshold: {result.detection_params['threshold_percentile']:.1f}%\n"
                f"• Box size: {result.detection_params['box_size_nm']:.0f} nm\n"
                f"• Min frames: {result.detection_params['min_frames_fraction']:.1%}\n"
                f"• Candidates found: {result.metadata['total_candidates']}\n"
                f"• Valid fiducials: {result.n_fiducials}"
            )

            fig.text(
                0.02,
                0.98,
                summary_text,
                transform=fig.transFigure,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            )

            plt.tight_layout()
            plt.subplots_adjust(left=0.15)  # Make room for summary text

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"✅ Fiducial detection plot saved to: {save_path}")

            plt.show()

        except Exception as e:
            print(f"⚠️ Failed to create fiducial detection plot: {e}")

    def undrift_with_detected_fiducials(
        self, detection_result: FiducialDetectionResult, **params
    ) -> Tuple[np.recarray, DriftResult]:
        """Perform drift correction using previously detected fiducials.

        Args:
            detection_result: Result from detect_fiducials()
            **params: Additional drift correction parameters

        Returns:
            Tuple of (corrected_locs, drift_result)

        Example:
            >>> DCF = Drift_Correction_Functions()
            >>> # First detect fiducials
            >>> detection_result = DCF.detect_fiducials(locs, info)
            >>> # Then perform drift correction
            >>> corrected, drift = DCF.undrift_with_detected_fiducials(detection_result)
        """
        # Use the localizations with group field for drift correction
        return self.undrift(
            locs=detection_result.locs_with_groups,
            info=[
                {
                    "Width": detection_result.detection_image.shape[1],
                    "Height": detection_result.detection_image.shape[0],
                    "Frames": len(np.unique(detection_result.locs_with_groups.frame)),
                    "Pixelsize": detection_result.detection_params["pixelsize"]
                    / 1000,  # Convert nm to μm
                }
            ],
            method="fiducial",
            **params,
        )

    def detect_high_density_regions_from_image(
        self,
        smoothed_image: np.ndarray,
        histogram_bins: int = 256,
        threshold_percentile: float = 99.0,
        pixelsize: float = 100.0,
        output_figure_path: Optional[str] = None,
        title: str = "High-Density Region Detection",
        create_plot: bool = True,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, float, Dict[str, Any]]:
        """Detect high-density regions from a smoothed image using histogram analysis.

        This function takes a pre-smoothed/rendered image and identifies high-density
        regions based on histogram analysis. It provides clear visualization of the
        detection process and outputs region coordinates for downstream processing.

        Args:
            smoothed_image: Pre-smoothed 2D image array (e.g., from render functions)
            histogram_bins: Number of bins for histogram analysis
            threshold_percentile: Percentile threshold for region detection (0-100)
            pixelsize: Pixel size in nm for scale bar visualization
            output_figure_path: Optional path to save the detection figure
            title: Title for the detection plot
            create_plot: Whether to create visualization plots

        Returns:
            Tuple containing:
            - List of (y, x) coordinates of detected high-density region centers
            - Binary mask of detected regions
            - Threshold value used for detection
            - Metadata dictionary with detection statistics
        """
        try:
            import PlottingFunctions
        except ImportError:
            warnings.warn(
                "PlottingFunctions not available. Visualization will be limited."
            )
            PlottingFunctions = None

        # Calculate histogram and threshold
        image_flat = smoothed_image.ravel()
        image_flat = image_flat[image_flat > 0]  # Exclude zero values

        if len(image_flat) == 0:
            raise DriftCorrectionError("Image contains no non-zero values")

        hist, bin_edges = np.histogram(image_flat, bins=histogram_bins)
        threshold = np.percentile(image_flat, threshold_percentile)

        # Create binary mask of high-density regions
        binary_mask = smoothed_image > threshold

        # Find connected components / regions
        from scipy import ndimage

        labeled_regions, n_regions = ndimage.label(binary_mask)

        # Calculate region centers and properties
        region_centers = []
        region_stats = []

        for region_id in range(1, n_regions + 1):
            region_mask = labeled_regions == region_id
            region_coords = np.where(region_mask)

            if len(region_coords[0]) > 0:
                # Calculate center of mass
                center_y = np.mean(region_coords[0])
                center_x = np.mean(region_coords[1])
                region_centers.append((int(center_y), int(center_x)))

                # Calculate region statistics
                region_area = np.sum(region_mask)
                region_intensity = np.sum(smoothed_image[region_mask])
                region_max_intensity = np.max(smoothed_image[region_mask])

                region_stats.append(
                    {
                        "center": (center_y, center_x),
                        "area_pixels": region_area,
                        "total_intensity": region_intensity,
                        "max_intensity": region_max_intensity,
                        "mean_intensity": (
                            region_intensity / region_area if region_area > 0 else 0
                        ),
                    }
                )

        # Create visualization using PlottingFunctions (if requested)
        if create_plot:
            self._plot_density_detection_results(
                smoothed_image,
                binary_mask,
                region_centers,
                hist,
                bin_edges,
                threshold,
                pixelsize,
                output_figure_path,
                title,
                PlottingFunctions,
            )

        # Prepare metadata
        metadata = {
            "n_regions_detected": n_regions,
            "threshold_value": threshold,
            "threshold_percentile": threshold_percentile,
            "histogram_bins": histogram_bins,
            "image_shape": smoothed_image.shape,
            "image_max": np.max(smoothed_image),
            "image_mean": np.mean(smoothed_image[smoothed_image > 0]),
            "region_statistics": region_stats,
            "total_region_area": np.sum(binary_mask),
            "region_area_fraction": np.sum(binary_mask) / binary_mask.size,
        }

        return region_centers, binary_mask, threshold, metadata

    def select_puncta_from_regions(
        self,
        locs: np.recarray,
        region_centers: List[Tuple[int, int]],
        binary_mask: np.ndarray,
        pixelsize: float = 100.0,
        selection_box_size_nm: float = 600.0,
        min_localizations_per_region: int = 10,
        output_figure_path: Optional[str] = None,
        title: str = "Puncta Selection from Regions",
        create_plot: bool = True,
        plot_individual_regions: bool = True,
        use_datashader_threshold: int = 1000,
        memory_optimize: bool = True,
    ) -> Tuple[List[np.recarray], Dict[str, Any]]:
        """Select puncta (localizations) from detected high-density regions using postprocess.picked_locs.

        This function takes the output from detect_high_density_regions_from_image
        and selects localizations within rectangular boxes around each detected region center
        to create potential fiducial candidates. Uses the optimized postprocess.picked_locs
        function with Rectangle shape, creating axis-aligned boxes by using diagonal picks
        with appropriate width parameters. Automatically enables parallelization for 8+ regions
        for improved performance on large datasets.

        Args:
            locs: Localization data with xc, yc, frame fields
            region_centers: List of (y, x) coordinates from density detection
            binary_mask: Binary mask from density detection
            pixelsize: Pixel size in nm for coordinate conversion
            selection_box_size_nm: Size of square selection box around each region center (nm)
            min_localizations_per_region: Minimum number of localizations required for a valid region
            output_figure_path: Optional path to save selection visualization
            title: Title for visualization plots
            create_plot: Whether to create visualization plots
            plot_individual_regions: Whether to plot individual region details (all regions shown)
            use_datashader_threshold: Use datashader for scatter plots with more than this many points

        Returns:
            Tuple containing:
            - List of localization arrays, one per valid region
            - Metadata dictionary with selection statistics
        """

        # Check if postprocess module is available
        if postprocess is None:
            raise RuntimeError(
                "postprocess module not available - cannot use picked_locs function"
            )

        # Handle empty region centers
        if not region_centers:
            metadata = {
                "n_regions_input": 0,
                "n_regions_selected": 0,
                "selection_criteria": {
                    "min_localizations": min_localizations_per_region,
                    "selection_box_size_nm": selection_box_size_nm,
                    "selection_box_size_pixels": 0.0,
                },
                "rejection_reasons": {"too_few_localizations": 0, "accepted": 0},
                "region_statistics": [],
            }
            return [], metadata

        # Convert box size from nm to pixels
        box_size_pixels = selection_box_size_nm / pixelsize
        half_box = box_size_pixels / 2.0

        # Create horizontal line picks for Rectangle shape (following existing pattern)
        # Rectangle implementation creates boxes around lines defined by two points
        picks = []
        for center_y, center_x in region_centers:
            # Create horizontal line through center - much simpler!
            picks.append(
                ((center_x - half_box, center_y), (center_x + half_box, center_y))
            )

        # Use postprocess.picked_locs with parallelization if we have 8+ picks
        width = max(locs.xc.max() + 10, 100)
        height = max(locs.yc.max() + 10, 100)

        picked_locs_arrays = postprocess.picked_locs(
            locs=locs,
            width=width,
            height=height,
            picks=picks,
            pick_shape="Rectangle",
            pick_size=box_size_pixels,  # Width of the rectangle in pixels (like existing code)
            add_group=False,
            callback="console",  # Show progress bar for puncta selection
            parallel=len(picks) >= 8,  # Enable parallelization for 8+ picks
        )

        # Memory cleanup: clear picks list immediately after use
        if memory_optimize:
            del picks
            gc.collect()

        # Filter results based on minimum localization count and build statistics
        selected_puncta = []
        region_stats = []

        # Ensure picked_locs_arrays is not None
        if picked_locs_arrays is None:
            picked_locs_arrays = []

        # Memory-optimized processing: stream through regions, immediate filtering and cleanup
        rejected_count = 0
        for region_id, (region_locs, (center_y, center_x)) in enumerate(
            zip(picked_locs_arrays, region_centers)
        ):
            n_locs = len(region_locs)

            # Apply localization count filter FIRST (Option C: Lazy statistics)
            if n_locs >= min_localizations_per_region:
                selected_puncta.append(region_locs)

                # Only calculate statistics for regions that passed the filter
                region_stat = {
                    "region_id": region_id,
                    "center_y": center_y,
                    "center_x": center_x,
                    "n_localizations": n_locs,
                    "mean_x": np.mean(region_locs.xc),
                    "mean_y": np.mean(region_locs.yc),
                    "std_x": np.std(region_locs.xc),
                    "std_y": np.std(region_locs.yc),
                    "frame_range": [int(region_locs.frame.min()), int(region_locs.frame.max())],
                    "frame_span": int(region_locs.frame.max() - region_locs.frame.min() + 1),
                    "selection_box_size_nm": selection_box_size_nm,
                    "selection_box_size_pixels": box_size_pixels,
                    "box_boundaries": {
                        "x_min": center_x - half_box,
                        "x_max": center_x + half_box,
                        "y_min": center_y - half_box,
                        "y_max": center_y + half_box,
                    },
                }

                # Add photon statistics if available (only for accepted regions)
                if hasattr(region_locs, "photons"):
                    region_stat["mean_photons"] = np.mean(region_locs.photons)
                    region_stat["std_photons"] = np.std(region_locs.photons)

                region_stats.append(region_stat)
            else:
                # Region rejected - no statistics calculated, immediate cleanup
                rejected_count += 1
                if memory_optimize:
                    # Explicitly delete rejected region data to free memory immediately
                    del region_locs

            # Periodic memory cleanup during processing (Option D: Aggressive cleanup)
            if memory_optimize and region_id % 100 == 0 and region_id > 0:
                gc.collect()
                print(f"Processed {region_id + 1}/{len(picked_locs_arrays)} regions "
                      f"({len(selected_puncta)} accepted, {rejected_count} rejected)")

        # Final memory cleanup (Option D)
        if memory_optimize:
            del picked_locs_arrays
            gc.collect()
            print(f"Memory optimization: Freed intermediate arrays after region processing")

        # Create visualization if requested
        if create_plot:
            self._plot_puncta_selection_results(
                locs,
                selected_puncta,
                region_centers,
                binary_mask,
                region_stats,
                box_size_pixels,
                pixelsize,
                output_figure_path,
                title,
                plot_individual_regions,
                use_datashader_threshold,
            )

            # Memory optimization: clear plot data if requested
            if memory_optimize:
                plt.close('all')  # Close all figure windows to free memory
                gc.collect()

        # Prepare metadata with memory-optimized calculations
        total_locs_selected = sum(len(puncta) for puncta in selected_puncta)

        metadata = {
            "n_regions_input": len(region_centers),
            "n_regions_selected": len(selected_puncta),
            "n_regions_rejected": rejected_count,
            "selection_rate": (
                len(selected_puncta) / len(region_centers) if region_centers else 0
            ),
            "selection_criteria": {
                "min_localizations": min_localizations_per_region,
                "selection_box_size_nm": selection_box_size_nm,
                "selection_box_size_pixels": box_size_pixels,
            },
            "region_statistics": region_stats,
            "total_selected_localizations": total_locs_selected,
            "memory_optimized": memory_optimize,
            "rejection_reasons": {
                "too_few_localizations": rejected_count,
                "accepted": len(selected_puncta),
            },
        }

        return selected_puncta, metadata

    def identify_real_fiducials_with_clustering(
        self,
        selected_puncta: List[np.recarray],
        precision_factor: float = 3.0,
        min_samples_factor: float = 0.7,
        frame_count: int = 100000,
        output_figure_path: Optional[str] = None,
        title: str = "Fiducial Clustering Analysis",
        create_plot: bool = True,
    ) -> Tuple[List[np.recarray], Dict[str, Any]]:
        """Identify real fiducials from selected puncta using BIRCH clustering.

        This function takes puncta (localizations) from select_puncta_from_regions
        and applies BIRCH clustering to identify real fiducial markers by requiring
        spatial clustering of localizations. BIRCH is much more memory-efficient than DBSCAN
        and uses a sample-then-predict strategy for large datasets.

        Args:
            selected_puncta: List of localization arrays from select_puncta_from_regions
            precision_factor: Multiplier for localization precision to set BIRCH threshold parameter
            min_samples_factor: Fraction of frame_count (unused in BIRCH but kept for compatibility)
            frame_count: Total number of frames (for reference)
            output_figure_path: Optional path to save clustering visualization
            title: Title for visualization plots
            create_plot: Whether to create visualization plots

        Returns:
            Tuple containing:
            - List of localization arrays for validated fiducials
            - Metadata dictionary with clustering statistics
        """

        validated_fiducials = []
        clustering_metadata = []

        # Process each puncta region
        for region_id, puncta_locs in enumerate(selected_puncta):
            n_locs = len(puncta_locs)

            if n_locs < 10:  # Skip regions with too few localizations for clustering
                continue

            # Prepare data for DBSCAN
            X = np.vstack([puncta_locs["xc"], puncta_locs["yc"]]).T

            # Calculate localization precision-based eps parameter
            if hasattr(puncta_locs, "xc_err") and hasattr(puncta_locs, "yc_err"):
                loc_precision = precision_factor * (
                    np.mean(puncta_locs["xc_err"]) + np.mean(puncta_locs["yc_err"])
                )
            else:
                # Fallback: estimate precision from localization spread
                loc_precision = (
                    precision_factor
                    * (np.std(puncta_locs["xc"]) + np.std(puncta_locs["yc"]))
                    / 10
                )

            # Calculate minimum samples requirement
            min_samples = max(
                int(min_samples_factor * frame_count / 1000), 5
            )  # Scale by 1000, minimum 5

            # Apply memory-efficient BIRCH clustering with sampling strategy
            try:
                # Step 1: Sample data for BIRCH training if dataset is large
                n_points = len(X)
                sample_size = min(2000, n_points)  # Use up to 2000 points for training

                if n_points > sample_size:
                    # Randomly sample points for BIRCH training
                    np.random.seed(42)  # For reproducibility
                    sample_indices = np.random.choice(n_points, sample_size, replace=False)
                    X_sample = X[sample_indices]
                    print(f"  BIRCH training on {sample_size} sampled points from {n_points} total")
                else:
                    X_sample = X
                    sample_indices = np.arange(n_points)
                    print(f"  BIRCH training on all {n_points} points")

                # Step 2: Configure BIRCH parameters
                # threshold: Maximum distance between a point and cluster centroid to assign it
                # branching_factor: Maximum number of subclusters in each CF node
                # n_clusters: Auto-determine clusters or set to None for natural clustering
                threshold_distance = loc_precision * 1.5  # Slightly larger than DBSCAN eps

                birch = Birch(
                    threshold=threshold_distance,
                    branching_factor=50,  # Balance memory vs accuracy
                    n_clusters=None,  # Let BIRCH determine clusters naturally
                    compute_labels=True
                )

                # Step 3: Train BIRCH on sample data
                birch.fit(X_sample)
                sample_labels = birch.labels_

                # Count valid clusters in sample (excluding noise points if any)
                sample_cluster_ids = set(sample_labels)
                if -1 in sample_cluster_ids:
                    sample_cluster_ids.remove(-1)  # Remove noise label if present
                n_sample_clusters = len(sample_cluster_ids)

                print(f"  BIRCH found {n_sample_clusters} clusters in sample data")

                # Step 4: Predict on full dataset using trained BIRCH model
                cluster_labels = birch.predict(X)

                # Clear sample data to free memory
                del X, X_sample, sample_labels
                gc.collect()

                # Analyze clustering results
                cluster_ids = set(cluster_labels)
                if -1 in cluster_ids:
                    cluster_ids.remove(-1)  # Remove noise label if present
                n_clusters = len(cluster_ids)
                n_noise = np.sum(cluster_labels == -1)

                # Consider this a valid fiducial if we have at least one significant cluster
                if n_clusters >= 1 and n_noise < 0.8 * n_locs:  # Less than 80% noise
                    # Keep only the largest cluster (main fiducial core)
                    if n_clusters > 0:
                        cluster_sizes = [
                            (label, np.sum(cluster_labels == label))
                            for label in set(cluster_labels)
                            if label != -1
                        ]
                        largest_cluster_label = max(cluster_sizes, key=lambda x: x[1])[
                            0
                        ]

                        # Extract localizations from the largest cluster
                        main_cluster_mask = cluster_labels == largest_cluster_label
                        validated_locs = puncta_locs[main_cluster_mask]
                        validated_fiducials.append(validated_locs)

                        # Store clustering metadata
                        cluster_metadata = {
                            "region_id": region_id,
                            "original_n_locs": n_locs,
                            "validated_n_locs": len(validated_locs),
                            "n_clusters": n_clusters,
                            "n_noise": n_noise,
                            "noise_fraction": n_noise / n_locs,
                            "largest_cluster_size": np.sum(main_cluster_mask),
                            "threshold_used": threshold_distance,
                            "clustering_method": "BIRCH",
                            "sample_size": sample_size if n_points > sample_size else n_points,
                            "precision_factor": precision_factor,
                            "min_samples_factor": min_samples_factor,
                            "cluster_labels": cluster_labels,
                            "cluster_center_x": np.mean(validated_locs["xc"]),
                            "cluster_center_y": np.mean(validated_locs["yc"]),
                            "cluster_std_x": np.std(validated_locs["xc"]),
                            "cluster_std_y": np.std(validated_locs["yc"]),
                        }
                        clustering_metadata.append(cluster_metadata)

                        # Memory monitoring for large datasets (before cleanup)
                        if n_locs > 10000:
                            print(f"Processed large region {region_id}: {len(validated_locs)}/{n_locs} locs validated")

                        # Clean up intermediate arrays to free memory
                        del cluster_labels, validated_locs
                        gc.collect()

                        # Extra cleanup for large regions
                        if n_locs > 10000:
                            gc.collect()

            except Exception as e:
                # Skip this region if clustering fails
                print(f"Warning: BIRCH clustering failed for region {region_id}: {e}")
                continue

        # Create visualization if requested
        if create_plot and len(validated_fiducials) > 0:
            self._plot_clustering_results(
                selected_puncta,
                validated_fiducials,
                clustering_metadata,
                output_figure_path,
                title,
            )

        # Prepare summary metadata
        summary_metadata = {
            "n_input_regions": len(selected_puncta),
            "n_validated_fiducials": len(validated_fiducials),
            "validation_rate": (
                len(validated_fiducials) / len(selected_puncta)
                if selected_puncta
                else 0
            ),
            "clustering_parameters": {
                "precision_factor": precision_factor,
                "min_samples_factor": min_samples_factor,
                "frame_count": frame_count,
            },
            "region_details": clustering_metadata,
            "total_input_locs": sum(len(puncta) for puncta in selected_puncta),
            "total_validated_locs": sum(
                len(fiducial) for fiducial in validated_fiducials
            ),
        }

        return validated_fiducials, summary_metadata

    def _plot_puncta_selection_results(
        self,
        all_locs: np.recarray,
        selected_puncta: List[np.recarray],
        region_centers: List[Tuple[int, int]],
        binary_mask: np.ndarray,
        region_stats: List[Dict[str, Any]],
        box_size_pixels: float,
        pixelsize: float,
        output_figure_path: Optional[str],
        title: str,
        plot_individual_regions: bool = True,
        use_datashader_threshold: int = 1000,
    ) -> None:
        """Create visualization of puncta selection results."""

        try:
            import PlottingFunctions

            plotter = PlottingFunctions.Plotter(poster=False)
            use_plotting_functions = True
        except ImportError:
            use_plotting_functions = False
            plotter = None

        # Get file base name for multiple plots
        if output_figure_path:
            base_path = (
                output_figure_path.rsplit(".", 1)[0]
                if "." in output_figure_path
                else output_figure_path
            )
        else:
            base_path = "puncta_selection"

        # Plot 1: Overview with all localizations and detected regions
        fig, axs = (
            plotter.one_column_plot()
            if use_plotting_functions and plotter
            else plt.subplots(1, 1, figsize=(8, 8))
        )
        ax1 = axs

        # Use datashader for large datasets, regular plotting for smaller ones
        if len(all_locs) > use_datashader_threshold:
            try:
                import datashader as ds
                import pandas as pd
                import colorcet as cc

                # Create DataFrame from localization data
                df = pd.DataFrame(
                    {"x": np.array(all_locs.xc), "y": np.array(all_locs.yc)}
                )

                # Create datashader canvas with proper aspect ratio
                x_range = all_locs.xc.max() - all_locs.xc.min()
                y_range = all_locs.yc.max() - all_locs.yc.min()
                aspect_ratio = x_range / y_range if y_range > 0 else 1.0

                if aspect_ratio > 1:
                    plot_width, plot_height = 500, int(500 / aspect_ratio)
                else:
                    plot_width, plot_height = int(500 * aspect_ratio), 500

                cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)

                # Aggregate points
                agg = cvs.points(df, "x", "y")

                # Create rasterized image
                img = ds.tf.set_background(
                    ds.tf.shade(agg, how="log", cmap=cc.fire), "black"
                ).to_pil()

                # Display with imshow
                ax1.imshow(
                    img,
                    extent=[
                        all_locs.xc.min(),
                        all_locs.xc.max(),
                        all_locs.yc.min(),
                        all_locs.yc.max(),
                    ],
                    aspect="auto",
                    origin="lower",
                )
                ax1.text(
                    0.02,
                    0.98,
                    f"Datashader: {len(all_locs)} locs",
                    transform=ax1.transAxes,
                    va="top",
                    fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

                # Clean up datashader objects to free memory
                del df, agg, img
                gc.collect()

            except ImportError:
                print(
                    "Warning: datashader not available, falling back to subsampled points"
                )
                # Fallback to subsampled plotting
                if len(all_locs) > 5000:
                    indices = np.random.choice(len(all_locs), 5000, replace=False)
                    bg_locs = all_locs[indices]
                else:
                    bg_locs = all_locs
                ax1.plot(
                    bg_locs.xc,
                    bg_locs.yc,
                    ".",
                    color="lightgray",
                    markersize=0.5,
                    alpha=0.5,
                    label="Background",
                )
        else:
            # Regular plotting for smaller datasets
            ax1.plot(
                all_locs.xc,
                all_locs.yc,
                ".",
                color="lightgray",
                markersize=0.5,
                alpha=0.5,
                label="Background",
            )

        # Overlay binary mask as contours (keep as is - efficient)
        ax1.contour(
            binary_mask,
            levels=[0.5],
            colors="blue",
            linewidths=1,
            alpha=0.5,
            label="Detected regions",
        )

        # Plot region centers and selection boxes (optimized)
        half_box = box_size_pixels / 2.0
        if region_centers:
            # Fast center plotting
            centers_array = np.array(region_centers)
            ax1.plot(
                centers_array[:, 1],
                centers_array[:, 0],
                "ro",
                markersize=5,
                label="Centers",
            )

            # Draw selection boxes (faster line plots instead of patches)
            for center_y, center_x in region_centers:
                box_x = [
                    center_x - half_box,
                    center_x + half_box,
                    center_x + half_box,
                    center_x - half_box,
                    center_x - half_box,
                ]
                box_y = [
                    center_y - half_box,
                    center_y - half_box,
                    center_y + half_box,
                    center_y + half_box,
                    center_y - half_box,
                ]
                ax1.plot(box_x, box_y, "r--", linewidth=1, alpha=0.7)

        # Highlight selected puncta (optimized)
        if selected_puncta:
            colors = plt.cm.tab10(
                np.linspace(0, 1, min(len(selected_puncta), 10))
            )  # Limit colors for speed
            for i, puncta in enumerate(selected_puncta):
                color = colors[i % len(colors)]
                # Subsample large puncta for display
                if len(puncta) > 200:
                    indices = np.random.choice(len(puncta), 200, replace=False)
                    display_puncta = puncta[indices]
                else:
                    display_puncta = puncta
                # Ultra-fast plot points instead of scatter
                ax1.plot(
                    display_puncta.xc,
                    display_puncta.yc,
                    ".",
                    color=color,
                    markersize=3,
                    alpha=0.8,
                )

        ax1.set_xlabel("X (pixels)")
        ax1.set_ylabel("Y (pixels)")
        ax1.set_title(f"{title} - Overview")
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)
        plt.show()
        if output_figure_path:
            plt.savefig(f"{base_path}_1_overview.png", dpi=300, bbox_inches="tight")
            plt.close()
            # Force garbage collection after plot generation to free memory
            gc.collect()

        # Plot 2: Individual region details (all regions, no limit)
        if selected_puncta and plot_individual_regions:
            n_regions = len(selected_puncta)  # Show ALL regions
            n_cols = min(6, n_regions)  # Max 6 columns for readability
            n_rows = (n_regions + n_cols - 1) // n_cols

            fig2, axes = (
                plotter.two_column_plot(
                    ncolumns=n_cols,
                    nrows=n_rows,
                    widthratio=np.ones(n_cols),
                    heightratio=np.ones(n_rows),
                    height=n_rows,
                    width=n_cols,
                )
                if use_plotting_functions and plotter
                else plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
            )
            if n_regions == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            # Optimize plotting by batching operations
            for i in range(n_regions):
                puncta = selected_puncta[i]
                stats = region_stats[i]
                ax = axes[i] if i < len(axes) else None
                if ax is None:
                    break

                # Use datashader for dense regions, regular plot for sparse
                if len(puncta) > use_datashader_threshold:
                    try:
                        import datashader as ds
                        import pandas as pd
                        import colorcet as cc

                        # Create DataFrame for this region
                        df = pd.DataFrame(
                            {"x": np.array(puncta.xc), "y": np.array(puncta.yc)}
                        )

                        # Create datashader canvas with proper aspect ratio for individual regions
                        x_range = puncta.xc.max() - puncta.xc.min()
                        y_range = puncta.yc.max() - puncta.yc.min()
                        aspect_ratio = x_range / y_range if y_range > 0 else 1.0

                        if aspect_ratio > 1:
                            plot_width, plot_height = 200, int(200 / aspect_ratio)
                        else:
                            plot_width, plot_height = int(200 * aspect_ratio), 200

                        cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)
                        agg = cvs.points(df, "x", "y")
                        img = ds.tf.set_background(
                            ds.tf.shade(agg, how="log", cmap=cc.blues), "white"
                        ).to_pil()

                        # Display with imshow
                        ax.imshow(
                            img,
                            extent=[
                                puncta.xc.min(),
                                puncta.xc.max(),
                                puncta.yc.min(),
                                puncta.yc.max(),
                            ],
                            aspect="auto",
                            origin="lower",
                        )
                        ax.text(
                            0.98,
                            0.02,
                            f"{len(puncta)} locs",
                            transform=ax.transAxes,
                            ha="right",
                            fontsize=8,
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                        )

                    except ImportError:
                        # Fallback to regular plotting
                        ax.plot(
                            puncta.xc,
                            puncta.yc,
                            ".",
                            color="blue",
                            markersize=1,
                            alpha=0.7,
                        )
                        ax.text(
                            0.98,
                            0.02,
                            f"{len(puncta)} locs",
                            transform=ax.transAxes,
                            ha="right",
                            fontsize=8,
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                        )
                else:
                    # Regular plotting for sparse regions
                    ax.plot(
                        puncta.xc, puncta.yc, ".", color="blue", markersize=2, alpha=0.7
                    )

                # Mark centers (no alpha for speed)
                ax.plot(
                    stats["center_x"],
                    stats["center_y"],
                    "ro",
                    markersize=1,
                    label="Center",
                )
                ax.plot(
                    stats["mean_x"],
                    stats["mean_y"],
                    "go",
                    markersize=1,
                    label="Centroid",
                )

                # Simple box outline (faster than Rectangle patch)
                half_box_val = stats["box_boundaries"]["x_max"] - stats["center_x"]
                box_x = [
                    stats["center_x"] - half_box_val,
                    stats["center_x"] + half_box_val,
                    stats["center_x"] + half_box_val,
                    stats["center_x"] - half_box_val,
                    stats["center_x"] - half_box_val,
                ]
                box_y = [
                    stats["center_y"] - half_box_val,
                    stats["center_y"] - half_box_val,
                    stats["center_y"] + half_box_val,
                    stats["center_y"] + half_box_val,
                    stats["center_y"] - half_box_val,
                ]
                ax.plot(box_x, box_y, "r--", linewidth=1)

                # Batch axis settings for speed
                ax.set_title(f"R{i+1}: {stats['n_localizations']} locs", fontsize=10)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for j in range(n_regions, len(axes)):
                axes[j].set_visible(False)

            # Add title showing all regions
            fig2.suptitle(
                f"Individual Regions (all {len(selected_puncta)} regions)", fontsize=12
            )

            plt.show()
            if output_figure_path:
                plt.savefig(f"{base_path}_2_regions.png", dpi=300, bbox_inches="tight")
                plt.close()
                # Force garbage collection after region plots to free memory
                gc.collect()

        # Plot 3: Statistics summary
        if region_stats:
            fig3, axes = (
                plotter.two_column_plot(
                    nrows=2, ncolumns=2, widthratio=[1, 1], heightratio=[1, 1]
                )
                if use_plotting_functions and plotter
                else plt.subplots(2, 2, figsize=(10, 8))
            )

            # Histogram of localizations per region
            n_locs_list = [stats["n_localizations"] for stats in region_stats]
            bins = np.histogram_bin_edges(n_locs_list, bins="fd")
            if use_plotting_functions and plotter:
                axes[0, 0].hist(
                    n_locs_list, bins=bins, alpha=0.7, color="skyblue", density=True
                )
            else:
                axes[0, 0].hist(
                    n_locs_list, bins=bins, alpha=0.7, color="skyblue", density=True
                )
            axes[0, 0].set_xlabel("localisations per region")
            axes[0, 0].set_ylabel("probability density")
            axes[0, 0].set_title("distribution of localisations per region")
            axes[0, 0].grid(True, alpha=0.3)

            # Frame spans
            frame_spans = [stats["frame_span"] for stats in region_stats]
            bins = np.histogram_bin_edges(frame_spans, bins="fd")
            if use_plotting_functions and plotter:
                axes[0, 1].hist(
                    frame_spans, bins=bins, alpha=0.7, color="lightgreen", density=True
                )
            else:
                axes[0, 1].hist(
                    frame_spans, bins=bins, alpha=0.7, color="lightgreen", density=True
                )
            axes[0, 1].set_xlabel("frame span")
            axes[0, 1].set_ylabel("probability density")
            axes[0, 1].set_title("distribution of frame spans")
            axes[0, 1].grid(True, alpha=0.3)

            # Position spreads
            std_x_list = [stats["std_x"] for stats in region_stats]
            std_y_list = [stats["std_y"] for stats in region_stats]
            if use_plotting_functions and plotter:
                axes[1, 0].scatter(std_x_list, std_y_list, alpha=0.7, color="orange")
            else:
                axes[1, 0].scatter(std_x_list, std_y_list, alpha=0.7, color="orange")
            axes[1, 0].set_xlabel("X std (pixels)")
            axes[1, 0].set_ylabel("Y std (pixels)")
            axes[1, 0].set_title("Localization Spreads")
            axes[1, 0].grid(True, alpha=0.3)

            # Photon statistics (if available)
            if region_stats and "mean_photons" in region_stats[0]:
                photons_list = [stats["mean_photons"] for stats in region_stats]
                bins = np.histogram_bin_edges(photons_list, bins="fd")
                if use_plotting_functions and plotter:
                    axes[1, 1].hist(
                        photons_list, bins=bins, alpha=0.7, color="pink", density=True
                    )
                else:
                    axes[1, 1].hist(
                        photons_list, bins=bins, alpha=0.7, color="pink", density=True
                    )
                axes[1, 1].set_xlabel("mean photons per localisation")
                axes[1, 1].set_ylabel("probability density")
                axes[1, 1].set_title("distribution of photon counts")
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "No photon data available",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title("Photon Statistics")
            axes[1, 1].grid(True, alpha=0.3)

            plt.show()
            if output_figure_path:
                plt.savefig(
                    f"{base_path}_3_statistics.png", dpi=300, bbox_inches="tight"
                )
                plt.close()
                # Force garbage collection after statistics plots
                gc.collect()

        if output_figure_path:
            print(f"Puncta selection results saved:")
            print(f"  - {base_path}_1_overview.png")
            if selected_puncta:
                print(f"  - {base_path}_2_regions.png")
            if region_stats:
                print(f"  - {base_path}_3_statistics.png")
        else:
            plt.show()

    def _plot_density_detection_results(
        self,
        smoothed_image: np.ndarray,
        binary_mask: np.ndarray,
        region_centers: List[Tuple[int, int]],
        hist: np.ndarray,
        bin_edges: np.ndarray,
        threshold: float,
        pixelsize: float,
        output_figure_path: Optional[str],
        title: str,
        PlottingFunctions_module,
    ) -> None:
        """Create detailed visualization using PlottingFunctions module."""

        # Create four separate figures to work within PlottingFunctions constraints
        self._create_separate_plots(
            smoothed_image,
            binary_mask,
            region_centers,
            hist,
            bin_edges,
            threshold,
            pixelsize,
            output_figure_path,
            title,
            PlottingFunctions_module,
        )

    def _create_separate_plots(
        self,
        smoothed_image: np.ndarray,
        binary_mask: np.ndarray,
        region_centers: List[Tuple[int, int]],
        hist: np.ndarray,
        bin_edges: np.ndarray,
        threshold: float,
        pixelsize: float,
        output_figure_path: Optional[str],
        title: str,
        PlottingFunctions_module,
    ) -> None:
        """Create separate plots using PlottingFunctions to avoid layout conflicts."""

        # Create plotter instance
        plotter = PlottingFunctions_module.Plotter(poster=False)

        # Get file base name for multiple plots
        if output_figure_path:
            base_path = (
                output_figure_path.rsplit(".", 1)[0]
                if "." in output_figure_path
                else output_figure_path
            )
        else:
            base_path = "density_detection"

        # Plot 1: Original smoothed image
        fig, axs = plotter.two_column_plot(
            nrows=2, ncolumns=2, widthratio=[1, 1], heightratio=[1, 1]
        )
        ax1 = axs[0, 0]
        plotter.image_plot(
            ax1,
            smoothed_image,
            cmap="hot",
            cbar="on",
            cbarlabel="Intensity",
            label="Smoothed Image",
            pixelsize=pixelsize,
            sbar="on",
        )
        ax1.set_title(f"{title} - Smoothed Image")

        # Plot 2: Binary mask with detected regions
        ax2 = axs[0, 1]
        plotter.image_plot(
            ax2,
            smoothed_image,
            cmap="hot",
            cbar="off",
            label="Detected Regions",
            pixelsize=pixelsize,
            sbar="on",
        )
        # Overlay region centers
        if region_centers:
            centers_y, centers_x = zip(*region_centers)
            ax2.scatter(centers_x, centers_y, c="red", s=25, marker="x", linewidths=0.5)
        ax2.set_title(f"{title} - Detected Regions (n={len(region_centers)})")

        # Plot 3: Image with overlaid detections
        ax3 = axs[1, 0]
        plotter.image_plot(
            ax3,
            smoothed_image,
            cmap="gray",
            cbar="on",
            cbarlabel="Intensity",
            label="Detection Overlay",
            pixelsize=pixelsize,
            sbar="on",
        )
        # Overlay detection mask as contours
        ax3.contour(binary_mask, levels=[0.5], colors="red", linewidths=0.5, alpha=0.8)
        if region_centers:
            centers_y, centers_x = zip(*region_centers)
            ax3.scatter(
                centers_x, centers_y, c="cyan", s=40, marker="+", linewidths=0.5
            )
        ax3.set_title(f"{title} - Detection Overlay")

        # Plot 4: Histogram - use basic matplotlib since PlottingFunctions histogram_plot expects different input
        ax4 = axs[1, 1]

        # Create histogram plot manually to match our data format
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax4.plot(bin_centers, hist, "b-", linewidth=2, label="Histogram")

        # Add threshold line and fill
        ax4.axvline(
            threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold:.1f})",
        )
        ax4.fill_between(
            bin_centers[bin_centers >= threshold],
            hist[bin_centers >= threshold],
            alpha=0.3,
            color="red",
            label="Selected Region",
        )

        ax4.set_xlabel("Intensity")
        ax4.set_ylabel("Frequency")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_title(f"{title} - Intensity Distribution")

        if output_figure_path:
            plt.savefig(f"{base_path}_Figure.png", dpi=300, bbox_inches="tight")
            plt.close()

        if output_figure_path:
            print(f"Detection results saved as png:")
            print(f"  - {base_path}_Figure.png")
        else:
            plt.show()

    def _plot_clustering_results(
        self,
        selected_puncta: List[np.recarray],
        validated_fiducials: List[np.recarray],
        clustering_metadata: List[Dict[str, Any]],
        output_figure_path: Optional[str],
        title: str,
    ) -> None:
        """Create visualization of DBSCAN clustering results using PlottingFunctions."""

        try:
            import PlottingFunctions

            plotter = PlottingFunctions.Plotter(poster=False)
            use_plotting_functions = True
        except ImportError:
            print(
                "Warning: PlottingFunctions not available, skipping clustering visualization"
            )
            return

        # Get file base name for multiple plots
        if output_figure_path:
            base_path = output_figure_path.rsplit(".", 1)[0]
            base_path += f"_clustering"
        else:
            base_path = "clustering"

        n_regions = len(selected_puncta)
        n_validated = len(validated_fiducials)

        # Create column of 2x2 plots, one for each validated region
        n_plots_per_region = 4  # 2x2 grid per region
        total_cols = 2
        total_rows = max(1, n_validated * 2)  # 2 rows per validated region

        fig, axes = plotter.two_column_plot(
            ncolumns=total_cols,
            nrows=total_rows,
            widthratio=[1.0, 1.0],
            heightratio=[1.0] * total_rows,
            width=12,
            height=6 * n_validated,
        )

        # Handle case where axes might be 1D
        if total_rows == 1:
            axes = axes.reshape(1, -1) if hasattr(axes, 'reshape') else np.array([axes])
        elif total_cols == 1:
            axes = axes.reshape(-1, 1) if hasattr(axes, 'reshape') else axes

        fig.suptitle(
            f"{title} - Individual Region Analysis (Validated: {n_validated})",
            fontsize=16,
        )

        # Create individual 2x2 plots for each validated region
        try:
            import matplotlib.cm as cm
            colors = cm.tab10(np.linspace(0, 1, max(n_regions, 10)))
        except:
            colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

        # Plot each validated region in a 2x2 grid
        for region_idx, (fiducial_locs, meta) in enumerate(zip(validated_fiducials, clustering_metadata)):
            region_id = meta["region_id"]
            original_puncta = selected_puncta[region_id]

            # Calculate row positions for this region (2 rows per region)
            start_row = region_idx * 2

            # Get the 4 axes for this region's 2x2 grid
            if total_rows > 1:
                ax_tl = axes[start_row, 0]      # Top-left
                ax_tr = axes[start_row, 1]      # Top-right
                ax_bl = axes[start_row + 1, 0]  # Bottom-left
                ax_br = axes[start_row + 1, 1]  # Bottom-right
            else:
                # Single row case
                ax_tl = axes[0]
                ax_tr = axes[1] if len(axes) > 1 else axes[0]
                ax_bl = axes[0]
                ax_br = axes[1] if len(axes) > 1 else axes[0]

            region_color = colors[region_id % len(colors)]

            # Plot 1: Original puncta for this region
            self._plot_region_data_with_datashader(ax_tl, [original_puncta], [region_color],
                                                  f"Region {region_id+1}: Original Puncta\n({len(original_puncta):,} points)")

            # Plot 2: Validated fiducial points only
            self._plot_region_data_with_datashader(ax_tr, [fiducial_locs], [region_color],
                                                  f"Region {region_id+1}: Validated Fiducial\n({len(fiducial_locs):,} points)")

            # Plot 3: Clustering overlay (original + validated)
            all_x = np.concatenate([original_puncta['xc'], fiducial_locs['xc']])
            all_y = np.concatenate([original_puncta['yc'], fiducial_locs['yc']])
            types = ['original'] * len(original_puncta) + ['validated'] * len(fiducial_locs)

            self._plot_clustering_overlay(ax_bl, all_x, all_y, types,
                                        f"Region {region_id+1}: Clustering Overlay")

            # Plot 4: Clustering statistics
            ax_br.axis('off')
            stats_text = f"Region {region_id+1} Statistics:\n\n"
            stats_text += f"• Original Points: {len(original_puncta):,}\n"
            stats_text += f"• Validated Points: {len(fiducial_locs):,}\n"
            stats_text += f"• Retention Rate: {100*len(fiducial_locs)/len(original_puncta):.1f}%\n"
            stats_text += f"• Noise Fraction: {meta['noise_fraction']:.3f}\n"
            stats_text += f"• N Clusters: {meta['n_clusters']}\n"
            if 'eps' in meta:
                stats_text += f"• DBSCAN eps: {meta['eps']:.2f}\n"
                stats_text += f"• Min Samples: {meta['min_samples']}\n"

            # Quality indicator
            if meta['noise_fraction'] < 0.2:
                quality = "Excellent ✓"
                color = "green"
            elif meta['noise_fraction'] < 0.5:
                quality = "Good ~"
                color = "orange"
            else:
                quality = "Poor ✗"
                color = "red"

            stats_text += f"\nQuality: {quality}"

            ax_br.text(0.05, 0.95, stats_text, transform=ax_br.transAxes,
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.1))

        if output_figure_path:
            fig.savefig(f"{base_path}_overview.png", dpi=300, bbox_inches="tight")

        import matplotlib.pyplot as plt
        plt.show()

        # Print summary
        if output_figure_path:
            print(f"Clustering results saved as:")
            print(f"  - {base_path}_overview.png")

    def _plot_region_data_with_datashader(self, ax, data_list, color_list, title):
        """Plot region data using datashader for large datasets, regular plotting for small ones."""
        total_points = sum(len(data) for data in data_list)

        if total_points > 1000:  # Use datashader for large datasets
            try:
                import datashader as ds
                import pandas as pd
                import colorcet as cc

                # Combine all data
                all_data = []
                for i, data in enumerate(data_list):
                    df_part = pd.DataFrame({
                        'x': data['xc'],
                        'y': data['yc'],
                        'group': f'group_{i}'
                    })
                    all_data.append(df_part)

                if all_data:
                    df = pd.concat(all_data, ignore_index=True)

                    # Create datashader canvas
                    canvas = ds.Canvas(plot_width=400, plot_height=400)
                    if len(data_list) > 1:
                        df['group'] = df['group'].astype('category')
                        agg = canvas.points(df, 'x', 'y', ds.count_cat('group'))
                        img = ds.tf.shade(agg, color_key=color_list, how='eq_hist')
                    else:
                        agg = canvas.points(df, 'x', 'y', ds.count())
                        img = ds.tf.shade(agg, cmap=cc.fire, how='eq_hist')

                    # Display the image
                    extent = [df.x.min(), df.x.max(), df.y.min(), df.y.max()]
                    ax.imshow(img.to_pil(), extent=extent, aspect='equal', origin='lower')
                    ax.set_xlim(extent[0], extent[1])
                    ax.set_ylim(extent[2], extent[3])

            except ImportError:
                # Fallback to subsampled regular plotting
                for i, data in enumerate(data_list):
                    color = color_list[i % len(color_list)]
                    # Heavy subsampling for display
                    max_points = 500
                    if len(data) > max_points:
                        indices = np.random.choice(len(data), max_points, replace=False)
                        display_data = data[indices]
                    else:
                        display_data = data

                    ax.plot(display_data['xc'], display_data['yc'], '.',
                           color=color, markersize=2, alpha=0.6)
        else:
            # Standard plotting for smaller datasets
            for i, data in enumerate(data_list):
                color = color_list[i % len(color_list)]
                ax.plot(data['xc'], data['yc'], '.', color=color, markersize=2, alpha=0.6)

        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    def _plot_clustering_overlay(self, ax, all_x, all_y, types, title):
        """Plot clustering overlay showing original vs validated points."""
        total_points = len(all_x)

        if total_points > 1000:  # Use datashader for large datasets
            try:
                import datashader as ds
                import pandas as pd
                import colorcet as cc

                df = pd.DataFrame({
                    'x': all_x,
                    'y': all_y,
                    'type': types
                })
                df['type'] = df['type'].astype('category')

                # Create datashader canvas
                canvas = ds.Canvas(plot_width=400, plot_height=400)
                agg = canvas.points(df, 'x', 'y', ds.count_cat('type'))

                # Custom color key: light gray for original, bright color for validated
                color_key = ['lightgray', 'red']
                img = ds.tf.shade(agg, color_key=color_key, how='eq_hist')

                # Display the image
                extent = [df.x.min(), df.x.max(), df.y.min(), df.y.max()]
                ax.imshow(img.to_pil(), extent=extent, aspect='equal', origin='lower')
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])

            except ImportError:
                # Fallback to subsampled regular plotting
                original_mask = np.array(types) == 'original'
                validated_mask = np.array(types) == 'validated'

                # Background points (heavily subsampled)
                original_x, original_y = all_x[original_mask], all_y[original_mask]
                if len(original_x) > 200:
                    indices = np.random.choice(len(original_x), 200, replace=False)
                    original_x, original_y = original_x[indices], original_y[indices]

                ax.plot(original_x, original_y, '.', color='lightgray',
                       markersize=1, alpha=0.3, label='Original')

                # Validated points (less subsampling)
                validated_x, validated_y = all_x[validated_mask], all_y[validated_mask]
                if len(validated_x) > 500:
                    indices = np.random.choice(len(validated_x), 500, replace=False)
                    validated_x, validated_y = validated_x[indices], validated_y[indices]

                ax.plot(validated_x, validated_y, '.', color='red',
                       markersize=3, alpha=0.9, label='Validated')
        else:
            # Standard plotting for smaller datasets
            original_mask = np.array(types) == 'original'
            validated_mask = np.array(types) == 'validated'

            ax.plot(all_x[original_mask], all_y[original_mask], '.',
                   color='lightgray', markersize=1, alpha=0.3, label='Original')
            ax.plot(all_x[validated_mask], all_y[validated_mask], '.',
                   color='red', markersize=3, alpha=0.9, label='Validated')

        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.legend()

    def _plot_individual_clustering_details(
        self,
        selected_puncta: List[np.recarray],
        validated_fiducials: List[np.recarray],
        clustering_metadata: List[Dict[str, Any]],
        base_path: str,
        title: str,
    ) -> None:
        """Create detailed plots for individual clustering results using PlottingFunctions."""

        try:
            import PlottingFunctions

            plotter = PlottingFunctions.Plotter(poster=False)
        except ImportError:
            print(
                "Warning: PlottingFunctions not available, skipping detailed clustering plots"
            )
            return

        n_validated = len(validated_fiducials)
        if n_validated == 0:
            return

        # Calculate subplot layout
        cols = min(3, n_validated)
        rows = (n_validated + cols - 1) // cols

        # Create figure layout with PlottingFunctions
        if cols == 1:
            widthratio = [1.0]
        elif cols == 2:
            widthratio = [1.0, 1.0]
        else:
            widthratio = [1.0, 1.0, 1.0]

        if rows == 1:
            heightratio = [1.0]
        elif rows == 2:
            heightratio = [1.0, 1.0]
        else:
            heightratio = [1.0] * rows

        fig, axes = plotter.two_column_plot(
            ncolumns=cols,
            nrows=rows,
            widthratio=widthratio,
            heightratio=heightratio,
            width=6 * cols,
            height=5 * rows,
        )
        fig.suptitle(f"{title} - Individual Clustering Details", fontsize=16)

        # Handle different axes configurations
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        else:
            axes = axes.flatten()

        # Define colors for clusters
        try:
            import matplotlib.cm as cm

            cluster_colormap = cm.tab10
        except:
            cluster_colormap = None

        for i, (fiducial_locs, meta) in enumerate(
            zip(validated_fiducials, clustering_metadata)
        ):
            if i >= len(axes):
                break

            ax = axes[i]
            region_id = meta["region_id"]
            original_puncta = selected_puncta[region_id]
            cluster_labels = meta["cluster_labels"]

            # Plot all original points with cluster colors
            unique_labels = set(cluster_labels)
            if cluster_colormap:
                colors = [
                    cluster_colormap(j / max(len(unique_labels), 1))
                    for j in range(len(unique_labels))
                ]
            else:
                colors = [
                    "blue",
                    "red",
                    "green",
                    "orange",
                    "purple",
                    "brown",
                    "pink",
                    "gray",
                ]

            color_map = {}
            for j, label in enumerate(unique_labels):
                if label == -1:  # Noise points
                    color_map[label] = "black"
                else:
                    color_map[label] = colors[j % len(colors)]

            # Use datashader for large datasets with multiple colors, regular plotting for smaller ones
            if len(original_puncta) > 1000 and len(validated_fiducials) > 8:
                try:
                    import datashader as ds
                    import pandas as pd
                    import colorcet as cc

                    # Create DataFrame with cluster labels as categorical data
                    df = pd.DataFrame(
                        {
                            "x": np.array(original_puncta.xc),
                            "y": np.array(original_puncta.yc),
                            "cluster": pd.Categorical(cluster_labels),
                        }
                    )

                    # Create datashader canvas with proper aspect ratio for clustering regions
                    x_range = original_puncta.xc.max() - original_puncta.xc.min()
                    y_range = original_puncta.yc.max() - original_puncta.yc.min()
                    aspect_ratio = x_range / y_range if y_range > 0 else 1.0

                    if aspect_ratio > 1:
                        plot_width, plot_height = 300, int(300 / aspect_ratio)
                    else:
                        plot_width, plot_height = int(300 * aspect_ratio), 300

                    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)

                    # Use categorical aggregation with ds.by()
                    agg = cvs.points(df, "x", "y", agg=ds.by("cluster", ds.count()))

                    # Convert cluster color_map to datashader color_key format
                    # Map cluster IDs to colors, handling noise (-1) separately
                    color_key = {}
                    for cluster_id in unique_labels:
                        if cluster_id == -1:
                            color_key[cluster_id] = "black"
                        else:
                            color_key[cluster_id] = color_map[cluster_id]

                    # Create shaded image with categorical colors
                    img = ds.tf.shade(agg, color_key=color_key, how="eq_hist")
                    img_pil = img.to_pil()

                    # Display with imshow
                    ax.imshow(
                        img_pil,
                        extent=[
                            original_puncta.xc.min(),
                            original_puncta.xc.max(),
                            original_puncta.yc.min(),
                            original_puncta.yc.max(),
                        ],
                        aspect="auto",
                        origin="lower",
                    )

                    # Add text annotation for datashader
                    ax.text(
                        0.98,
                        0.98,
                        f"Datashader\n{len(original_puncta)} locs\n{len(unique_labels)} clusters",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

                    # Create custom legend for clusters using small scatter plots
                    legend_elements = []
                    for k in unique_labels:
                        if k == -1:
                            label = "Noise"
                        else:
                            label = f"Cluster {k}"
                        legend_elements.append(
                            plt.Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="w",
                                markerfacecolor=color_map[k],
                                markersize=5,
                                label=label,
                            )
                        )
                    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

                except ImportError:
                    print(
                        "Warning: datashader not available for clustering plots, falling back to regular plotting"
                    )
                    # Fallback to regular plotting - plot each cluster separately
                    for k in unique_labels:
                        class_mask = cluster_labels == k
                        if np.any(class_mask):
                            if k == -1:  # Noise points
                                alpha = 0.3
                                size = 0.5
                                label = "Noise"
                            else:
                                alpha = 0.8
                                size = 2
                                label = f"Cluster {k}"
                            plotter.scatter_plot(
                                ax,
                                original_puncta["xc"][class_mask],
                                original_puncta["yc"][class_mask],
                                c=color_map[k],
                                s=size,
                                alpha=alpha,
                                label=label,
                            )
                    # Highlight the main cluster (validated fiducial)
                    plotter.scatter_plot(
                        ax,
                        fiducial_locs["xc"],
                        fiducial_locs["yc"],
                        c="red",
                        s=4,
                        alpha=1.0,
                        edgecolors="white",
                        linewidths=0.2,
                        label="Validated",
                    )
            else:
                # Regular plotting for smaller datasets - plot each cluster separately
                for k in unique_labels:
                    class_mask = cluster_labels == k
                    if np.any(class_mask):
                        if k == -1:  # Noise points
                            alpha = 0.3
                            size = 0.5
                            label = "Noise"
                        else:
                            alpha = 0.8
                            size = 2
                            label = f"Cluster {k}"

                        plotter.scatter_plot(
                            ax,
                            original_puncta["xc"][class_mask],
                            original_puncta["yc"][class_mask],
                            c=color_map[k],
                            s=size,
                            alpha=alpha,
                            label=label,
                        )

                # Highlight the main cluster (validated fiducial)
                plotter.scatter_plot(
                    ax,
                    fiducial_locs["xc"],
                    fiducial_locs["yc"],
                    c="red",
                    s=4,
                    alpha=1.0,
                    edgecolors="white",
                    linewidths=0.2,
                    label="Validated",
                )

            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            ax.set_title(
                f'Region {region_id} - {len(fiducial_locs)}/{meta["original_n_locs"]} locs\n'
                f'Noise: {meta["noise_fraction"]*100:.1f}%'
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axis("equal")

        # Hide unused subplots
        for i in range(n_validated, len(axes)):
            if i < len(axes):
                axes[i].set_visible(False)

        plotter.save_plot(f"{base_path}_details.png", dpi=300, bbox_inches="tight")

    def undrift_with_fiducial_detection(
        self,
        locs: np.recarray,
        info: list,
        threshold_percentile: float = 99.0,
        box_size_nm: float = 900.0,
        min_frames_fraction: float = 0.8,
        histogram_bins: int = 256,
        **params,
    ) -> Tuple[np.recarray, DriftResult, Dict[str, Any]]:
        """Complete fiducial drift correction workflow with automatic fiducial detection.

        This is a user-friendly wrapper that combines automatic fiducial detection
        with drift correction. It allows fine-tuning of detection parameters while
        providing sensible defaults.

        Args:
            locs: Localization data (group field not required)
            info: Metadata list containing frame count and image dimensions
            threshold_percentile: Histogram percentile threshold for fiducial detection (0-100)
            box_size_nm: Box size for fiducial detection in nanometers
            min_frames_fraction: Minimum fraction of frames for valid fiducial (0-1)
            histogram_bins: Number of bins for histogram analysis
            **params: Additional drift correction parameters

        Returns:
            Tuple of (corrected_locs, drift_result, detection_info)
            - corrected_locs: Drift-corrected localizations
            - drift_result: Drift correction results with metadata
            - detection_info: Information about fiducial detection process

        Raises:
            DriftCorrectionError: If fiducial detection fails or no valid fiducials found

        Example:
            >>> DCF = Drift_Correction_Functions()
            >>> # Basic usage with defaults
            >>> corrected, drift, info = DCF.undrift_with_fiducial_detection(locs, metadata)
            >>>
            >>> # Fine-tune detection parameters
            >>> corrected, drift, info = DCF.undrift_with_fiducial_detection(
            ...     locs, metadata,
            ...     threshold_percentile=95.0,  # Lower threshold for more candidates
            ...     box_size_nm=1200.0,         # Larger search box
            ...     min_frames_fraction=0.6     # Allow fiducials with fewer localizations
            ... )
            >>>
            >>> print(f"Found {info['n_fiducials']} fiducials")
            >>> print(f"Drift range: X [{drift.drift_x.min():.2f}, {drift.drift_x.max():.2f}]")
        """
        # Store original parameters for reporting
        detection_params = {
            "threshold_percentile": threshold_percentile,
            "box_size_nm": box_size_nm,
            "min_frames_fraction": min_frames_fraction,
            "histogram_bins": histogram_bins,
        }

        try:
            # Step 1: Detect fiducials using the new separated function
            detection_result = self.detect_fiducials(
                locs=locs,
                info=info,
                threshold_percentile=threshold_percentile,
                box_size_nm=box_size_nm,
                min_frames_fraction=min_frames_fraction,
                histogram_bins=histogram_bins,
                plot_results=False,  # No plot for the combined workflow
                save_plot=None,
            )

            # Step 2: Apply drift correction using detected fiducials
            corrected_locs, drift_result = self.undrift_with_detected_fiducials(
                detection_result=detection_result, **params
            )

            # Create detection info for backward compatibility
            detection_info = {
                "detection_params": detection_params,
                "n_fiducials": detection_result.n_fiducials,
                "fiducial_groups": list(range(detection_result.n_fiducials)),
                "frames_per_fiducial": detection_result.metadata[
                    "localizations_per_fiducial"
                ],
                "success": True,
                "message": f"Successfully detected {detection_result.n_fiducials} fiducials",
                "total_candidates": detection_result.metadata["total_candidates"],
                "threshold_used": detection_result.metadata["threshold_used"],
            }

            return corrected_locs, drift_result, detection_info

        except DriftCorrectionError as e:
            # Provide detailed error information for troubleshooting
            detection_info = {
                "detection_params": detection_params,
                "n_fiducials": 0,
                "fiducial_groups": [],
                "frames_per_fiducial": [],
                "success": False,
                "message": str(e),
                "troubleshooting": {
                    "try_lower_threshold": "Reduce threshold_percentile (e.g., 95.0 or 90.0)",
                    "try_larger_box": "Increase box_size_nm (e.g., 1200.0)",
                    "try_lower_min_frames": "Reduce min_frames_fraction (e.g., 0.6 or 0.5)",
                    "check_data": "Ensure localizations contain bright, stationary markers",
                },
            }
            raise DriftCorrectionError(f"Fiducial detection failed: {e}") from e
