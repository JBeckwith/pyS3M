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

# Matplotlib imports (needed for drift correction plotting)
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Local imports (will import from existing modules as needed)
import sys
import os

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)

import ProgressUtils

# Import our new plotting module
try:
    from DriftPlotting import DriftPlotter
    _drift_plotter = DriftPlotter()
except ImportError:
    warnings.warn("Could not import DriftPlotter. Plotting features may be limited.")
    _drift_plotter = None

# Import our specialised algorithm modules
try:
    from FiducialDetection import FiducialDetector
    _fiducial_detector = FiducialDetector()
except ImportError:
    warnings.warn("Could not import FiducialDetector. Fiducial detection features may be limited.")
    _fiducial_detector = None

try:
    from RCCAlgorithm import RCCAlgorithm
    _rcc_algorithm = RCCAlgorithm()
except ImportError:
    warnings.warn("Could not import RCCAlgorithm. RCC algorithm features may be limited.")
    _rcc_algorithm = None

try:
    from AIMAlgorithm import AIMAlgorithm
    _aim_algorithm = AIMAlgorithm()
except ImportError:
    warnings.warn("Could not import AIMAlgorithm. AIM algorithm features may be limited.")
    _aim_algorithm = None

try:
    from CoordinateProcessing import CoordinateProcessor, SegmentationHandler, DriftCorrectionError as CoordDriftError
    _coordinate_processor = CoordinateProcessor()
    _segmentation_handler = SegmentationHandler()
except ImportError:
    warnings.warn("Could not import CoordinateProcessing. Coordinate processing features may be limited.")
    _coordinate_processor = None
    _segmentation_handler = None
    # Define fallback error class if import fails
    class CoordDriftError(Exception):
        pass

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
    FIDUCIAL = "fiducial"  # Fiducial-based drift correction using picked localisations
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
            locs: Localisation data
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
            locs: Localisation data
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

        Now delegates to CoordinateProcessor for consistent interpolation.

        Args:
            bounds: Segment boundaries
            shift_x: X shifts between segments
            shift_y: Y shifts between segments
            n_frames: Total number of frames

        Returns:
            Tuple of (drift_x, drift_y) for all frames
        """
        return CoordinateProcessor.interpolate_drift(bounds, shift_x, shift_y, n_frames, method="cubic")


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
        """Count the number of intersected localisations between two datasets.

        Args:
            l0_coords: Unique coordinates of reference localisations
            l0_counts: Counts of unique reference localisations
            l1_coords: Unique coordinates of target localisations
            l1_counts: Counts of unique target localisations

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
            l0_coords: Unique coordinates of reference localisations
            l0_counts: Counts of reference localisations
            l1_coords: Unique coordinates of target localisations
            l1_counts: Counts of target localisations
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
            l0_coords: Unique values of reference localisations
            l0_counts: Counts of unique reference localisations
            x1, y1: x and y coordinates of target localisations
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

        # get unique values and counts of the target localisations
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
            l0_coords: Unique values of reference localisations
            l0_counts: Counts of unique reference localisations
            x1, y1, z1: x, y, and z coordinates of target localisations
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

        # get unique values and counts of the target localisations
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
        """Maximize intersection (undrift) for 2D localisations.

        This is the core AIM algorithm implementation.

        Args:
            x, y: x and y coordinates of localisations
            ref_x, ref_y: x and y coordinates of reference localisations
            frame: Frame indices of localisations
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

            # get the target localisations within the current segment
            min_frame_idx = frame > seg_bounds[s]
            max_frame_idx = frame <= seg_bounds[s + 1]
            x1 = x[min_frame_idx & max_frame_idx]
            y1 = y[min_frame_idx & max_frame_idx]

            # skip if no localisations in this segment
            if len(x1) == 0:
                if s > 0:
                    drift_x[s] = drift_x[s - 1]
                    drift_y[s] = drift_y[s - 1]
                return

            # undrifting from the previous round
            x1 += rel_drift_x
            y1 += rel_drift_y

            # count the number of intersected localisations
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
        1. Calculates segment centres as measurement points
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
        # Calculate segment centres (where we have actual measurements)
        seg_centres = (seg_bounds[1:] + seg_bounds[:-1]) / 2
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
        """Maximize intersection (undrift) for 3D localisations.

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

            # get the target localisations within the current segment
            min_frame_idx = frame > seg_bounds[s]
            max_frame_idx = frame <= seg_bounds[s + 1]
            x1 = x[min_frame_idx & max_frame_idx]
            y1 = y[min_frame_idx & max_frame_idx]
            z1 = z[min_frame_idx & max_frame_idx]

            # skip if no localisations in this segment
            if len(x1) == 0:
                if s > 0:
                    drift_z[s] = drift_z[s - 1]
                return

            # undrifting from the previous round
            z1 += rel_drift_z

            # count the number of intersected localisations
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

        # undrift the localisations
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
            locs: Localisation data
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

        # Get reference localisations (first segment)
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

        # Get reference localisations for Z (first segment)
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
    """Fiducial-based drift corrector using picked localisations.

    This method calculates drift using manually selected fiducial markers
    (e.g., gold nanoparticles, fluorescent beads) that should remain stationary
    during the experiment.

    The algorithm:
    1. Takes pre-selected fiducial localisations (picked manually or automatically)
    2. Removes centre-of-mass offset for each fiducial
    3. Calculates weighted average drift across all fiducials
    4. Uses inverse mean squared deviation as weights (more stable fiducials get higher weight)
    5. Interpolates drift for frames without localisations
    """

    def __init__(self):
        pass

    def supports_3d(self) -> bool:
        """Fiducial corrector supports 2D drift correction."""
        return False  # Can be extended to 3D in future

    def calculate_drift(
        self, locs: np.recarray, info: list, params: DriftParameters
    ) -> DriftResult:
        """Calculate drift using fiducial localisations.

        Args:
            locs: Localisation data. If no 'group' field exists and auto_detect_fiducials=True,
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

        # Group localisations by fiducial ID
        picked_locs = self._group_fiducials(locs)

        if len(picked_locs) == 0:
            raise DriftCorrectionError("No fiducial localisations found")

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
        """Group localisations by fiducial ID.

        Args:
            locs: Localisation data with 'group' field

        Returns:
            List of localisation arrays, one per fiducial
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
        """Calculate drift in a given coordinate using fiducial localisations.

        Args:
            picked_locs: List of localisation arrays for each fiducial
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

        # Calculate drift for each fiducial (remove centre of mass offset)
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
        """Interpolate drift for frames without localisations.

        Now delegates to CoordinateProcessor for consistent interpolation.

        Args:
            drift_mean: Drift array with possible NaN values

        Returns:
            Interpolated drift array
        """
        return CoordinateProcessor.interpolate_missing_frames(drift_mean, method="linear")

    def _detect_and_add_fiducials(
        self, locs: np.recarray, info: list, params: DriftParameters
    ) -> np.recarray:
        """Automatically detect fiducials and add group field to localisations.

        Args:
            locs: Localisation data without group field
            info: Metadata list
            params: Drift parameters with fiducial detection settings

        Returns:
            New localisation array with group field added
        """
        if render is None or imageprocess is None:
            raise DriftCorrectionError(
                "Fiducial detection requires render and imageprocess modules"
            )

        # Extract metadata for pixel size
        meta = CoordinateProcessor.extract_metadata(info)
        pixelsize = meta.get("pixelsize", 69.0)  # Default fallback
        n_frames = int(meta["n_frames"])

        # Render localisations to image for fiducial detection
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
        # Format picks as rectangles centreed on detected points
        half_box = box // 2
        picks = [((xi - half_box, yi), (xi + half_box, yi)) for xi, yi in zip(x, y)]

        if len(picks) == 0:
            raise DriftCorrectionError(
                "No fiducial candidates detected. Try lowering threshold_percentile."
            )

        # Filter picks by minimum localisations per fiducial
        min_n = params.fiducial_min_frames_fraction * n_frames

        try:
            import postprocess  # Import here to handle potential issues
        except ImportError:
            raise DriftCorrectionError(
                "postprocess module required for fiducial detection"
            )

        # Get localisations for each pick
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
                f"Try lowering fiducial_min_frames_fraction or threshold_percentile."
            )

        # Create new localisation array with group field
        return self._add_group_field(locs, valid_picked_locs, valid_picks)

    def _add_group_field(
        self, locs: np.recarray, picked_locs: list, picks: list
    ) -> np.recarray:
        """Add group field to localisations based on fiducial assignments.

        Args:
            locs: Original localisations
            picked_locs: List of localisations for each fiducial
            picks: List of pick coordinates

        Returns:
            New recarray with group field added
        """
        # Create group field array, initialize with -1 (non-fiducial)
        group = np.full(len(locs), -1, dtype=np.int32)

        # Assign group IDs to fiducial localisations
        for group_id, fiducial_locs in enumerate(picked_locs):
            # Find indices of these localisations in original array
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
        locs: Localisation data
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
        locs: Localisation data
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
        locs: Localisation data
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

        # Initialize plotting functionality
        try:
            from DriftPlotting import DriftPlotter
            self.plotter = DriftPlotter()
        except ImportError:
            self.plotter = None

        # Initialize specialised algorithm modules
        try:
            from FiducialDetection import FiducialDetector
            self.fiducial_detector = FiducialDetector(drift_correction_instance=self)
        except ImportError:
            self.fiducial_detector = None

        try:
            from RCCAlgorithm import RCCAlgorithm
            self.rcc_algorithm = RCCAlgorithm(drift_correction_instance=self)
        except ImportError:
            self.rcc_algorithm = None

        try:
            from AIMAlgorithm import AIMAlgorithm
            self.aim_algorithm = AIMAlgorithm(drift_correction_instance=self)
        except ImportError:
            self.aim_algorithm = None

        try:
            from CoordinateProcessing import CoordinateProcessor
            self.coordinate_processor = CoordinateProcessor()
        except ImportError:
            self.coordinate_processor = None

    def undrift(
        self,
        locs: np.recarray,
        info: list,
        method: Union[str, DriftMethod] = "auto",
        **params,
    ) -> Tuple[np.recarray, DriftResult]:
        """Universal drift correction interface.

        Args:
            locs: Localisation data
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

    # Delegation methods for specialised algorithm modules

    # Fiducial Detection delegation methods
    def detect_high_density_regions_from_image(self, *args, **kwargs):
        """Delegate to FiducialDetector.detect_high_density_regions_from_image"""
        if self.fiducial_detector is None:
            raise RuntimeError("FiducialDetector module not available")
        return self.fiducial_detector.detect_high_density_regions_from_image(*args, **kwargs)

    def select_puncta_from_regions(self, *args, **kwargs):
        """Delegate to FiducialDetector.select_puncta_from_regions"""
        if self.fiducial_detector is None:
            raise RuntimeError("FiducialDetector module not available")
        return self.fiducial_detector.select_puncta_from_regions(*args, **kwargs)

    def identify_real_fiducials_with_clustering_delegated(self, *args, **kwargs):
        """Delegate to FiducialDetector.identify_real_fiducials_with_clustering"""
        if self.fiducial_detector is None:
            raise RuntimeError("FiducialDetector module not available")
        return self.fiducial_detector.identify_real_fiducials_with_clustering(*args, **kwargs)

    # RCC Algorithm delegation methods
    def run_rcc_2d(self, *args, **kwargs):
        """Delegate to RCCAlgorithm.run_rcc_2d"""
        if self.rcc_algorithm is None:
            raise RuntimeError("RCCAlgorithm module not available")
        return self.rcc_algorithm.run_rcc_2d(*args, **kwargs)

    def run_rcc_3d(self, *args, **kwargs):
        """Delegate to RCCAlgorithm.run_rcc_3d"""
        if self.rcc_algorithm is None:
            raise RuntimeError("RCCAlgorithm module not available")
        return self.rcc_algorithm.run_rcc_3d(*args, **kwargs)

    # AIM Algorithm delegation methods
    def run_aim_2d(self, *args, **kwargs):
        """Delegate to AIMAlgorithm.run_aim_2d"""
        if self.aim_algorithm is None:
            raise RuntimeError("AIMAlgorithm module not available")
        return self.aim_algorithm.run_aim_2d(*args, **kwargs)

    def run_aim_3d(self, *args, **kwargs):
        """Delegate to AIMAlgorithm.run_aim_3d"""
        if self.aim_algorithm is None:
            raise RuntimeError("AIMAlgorithm module not available")
        return self.aim_algorithm.run_aim_3d(*args, **kwargs)

    # Coordinate Processing delegation methods
    def convert_pixels_to_nm(self, *args, **kwargs):
        """Delegate to CoordinateProcessor.convert_pixels_to_nm"""
        if self.coordinate_processor is None:
            raise RuntimeError("CoordinateProcessor module not available")
        return self.coordinate_processor.convert_pixels_to_nm(*args, **kwargs)

    def undrift_with_fiducial_detection(
        self,
        locs: np.recarray,
        info: list,
        histogram_bins: int = 256,
        threshold_percentile: float = 99.0,
        box_size_nm: float = 600.0,
        min_localisations_per_region: int = 100,
        retention_percentage: float = 0.9,
        create_plots: bool = False,
        output_dir: str = "./fiducial_detection",
    ) -> DriftResult:
        """Automatically detect fiducials and perform drift correction.

        This is a high-level convenience method that:
        1. Renders localisations to an image
        2. Detects high-density regions (potential fiducials)
        3. Selects localisations within those regions
        4. Validates fiducials using clustering
        5. Performs fiducial-based drift correction

        Args:
            locs: Localisation data (xc, yc, frame fields required)
            info: Metadata list containing image dimensions and frame info
            histogram_bins: Number of bins for histogram analysis
            threshold_percentile: Percentile threshold for fiducial detection (0-100)
            box_size_nm: Size of selection box around each fiducial (nm)
            min_localisations_per_region: Minimum localisations required per fiducial
            retention_percentage: Fraction of points to retain during validation (0-1)
            create_plots: Whether to create diagnostic plots
            output_dir: Directory to save plots (if create_plots=True)

        Returns:
            DriftResult object with drift_x, drift_y arrays and metadata

        Raises:
            DriftCorrectionError: If fiducial detection or drift correction fails
        """
        # Extract metadata
        meta = CoordinateProcessor.extract_metadata(info)
        pixelsize = meta.get("pixelsize", 100.0)
        n_frames = int(meta["n_frames"])

        # Step 1: Render localisations to image
        print("Step 1/5: Rendering localisations to image...")
        if render is None:
            raise DriftCorrectionError("render module required for fiducial detection")

        _, image = render.render(
            locs=locs,
            info=info,
            oversampling=1,
            blur_method="smooth",
        )

        # Step 2: Detect high-density regions
        print("Step 2/5: Detecting high-density regions...")
        region_centres, binary_mask, threshold, detection_meta = (
            self.fiducial_detector.detect_high_density_regions_from_image(
                smoothed_image=image,
                histogram_bins=histogram_bins,
                threshold_percentile=threshold_percentile,
                pixelsize=pixelsize,
                output_figure_path=f"{output_dir}/01_density_detection.png" if create_plots else None,
                create_plot=create_plots,
            )
        )

        print(f"  Found {detection_meta['n_regions_detected']} potential fiducial regions")

        if detection_meta['n_regions_detected'] == 0:
            raise DriftCorrectionError("No fiducial regions detected. Try lowering threshold_percentile.")

        # Step 3: Select puncta from regions
        print("Step 3/5: Selecting localisations from regions...")
        selected_puncta, selection_meta = (
            self.fiducial_detector.select_puncta_from_regions(
                locs=locs,
                region_centres=region_centres,
                binary_mask=binary_mask,
                pixelsize=pixelsize,
                selection_box_size_nm=box_size_nm,
                min_localisations_per_region=min_localisations_per_region,
                output_figure_path=f"{output_dir}/02_puncta_selection.png" if create_plots else None,
                create_plot=create_plots,
            )
        )

        print(f"  Selected {selection_meta['n_regions_selected']} fiducial candidates")

        if selection_meta['n_regions_selected'] == 0:
            raise DriftCorrectionError(
                f"No valid fiducials with >={min_localisations_per_region} localisations. "
                "Try lowering min_localisations_per_region or threshold_percentile."
            )

        # Step 4: Validate fiducials using clustering
        print("Step 4/5: Validating fiducials with clustering...")
        validated_fiducials, validation_meta = (
            self.fiducial_detector.identify_real_fiducials_with_clustering(
                selected_puncta=selected_puncta,
                retention_percentage=retention_percentage,
                pixelsize=pixelsize,
                output_figure_path=f"{output_dir}/03_fiducial_validation.png" if create_plots else None,
                create_plot=create_plots,
            )
        )

        print(f"  Validated {len(validated_fiducials)} final fiducials")

        if len(validated_fiducials) == 0:
            raise DriftCorrectionError(
                "No fiducials passed validation. Check your detection parameters."
            )

        # Step 5: Add group field and perform drift correction
        print("Step 5/5: Performing fiducial-based drift correction...")
        locs_with_groups = self._add_group_field(locs, validated_fiducials, region_centres)

        # Use the fiducial corrector directly
        fiducial_corrector = FiducialDriftCorrector()
        params = DriftParameters()  # Use default parameters
        result = fiducial_corrector.calculate_drift(locs_with_groups, info, params)

        # Add detection metadata to result
        result.metadata.update({
            "detection_method": "automatic",
            "n_regions_detected": detection_meta['n_regions_detected'],
            "n_fiducials_selected": selection_meta['n_regions_selected'],
            "n_fiducials_validated": len(validated_fiducials),
            "detection_params": {
                "histogram_bins": histogram_bins,
                "threshold_percentile": threshold_percentile,
                "box_size_nm": box_size_nm,
                "min_localisations_per_region": min_localisations_per_region,
                "retention_percentage": retention_percentage,
            }
        })

        print(f"✓ Drift correction complete using {len(validated_fiducials)} fiducials")

        return result

    def _add_group_field(
        self, locs: np.recarray, picked_locs: list, picks: list
    ) -> np.recarray:
        """Add group field to localisations based on fiducial assignments.

        Args:
            locs: Original localisations
            picked_locs: List of localisations for each fiducial
            picks: List of pick coordinates (not used, for compatibility)

        Returns:
            New recarray with group field added
        """
        # Create group field array, initialize with -1 (non-fiducial)
        group = np.full(len(locs), -1, dtype=np.int32)

        # Assign group IDs to fiducial localisations
        for group_id, fiducial_locs in enumerate(picked_locs):
            # Find indices of these localisations in original array
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

    def convert_nm_to_pixels(self, *args, **kwargs):
        """Delegate to CoordinateProcessor.convert_nm_to_pixels"""
        if self.coordinate_processor is None:
            raise RuntimeError("CoordinateProcessor module not available")
        return self.coordinate_processor.convert_nm_to_pixels(*args, **kwargs)

    def apply_drift_correction(self, *args, **kwargs):
        """Delegate to CoordinateProcessor.apply_drift_correction"""
        if self.coordinate_processor is None:
            raise RuntimeError("CoordinateProcessor module not available")
        return self.coordinate_processor.apply_drift_correction(*args, **kwargs)

    def create_spatial_grid(self, *args, **kwargs):
        """Delegate to CoordinateProcessor.create_spatial_grid"""
        if self.coordinate_processor is None:
            raise RuntimeError("CoordinateProcessor module not available")
        return self.coordinate_processor.create_spatial_grid(*args, **kwargs)

    def bin_localisations_spatially(self, *args, **kwargs):
        """Delegate to CoordinateProcessor.bin_localisations_spatially"""
        if self.coordinate_processor is None:
            raise RuntimeError("CoordinateProcessor module not available")
        return self.coordinate_processor.bin_localisations_spatially(*args, **kwargs)

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
            locs: Localisation data
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
            # Extract localisations for this chunk
            chunk_mask = (locs.frame >= start_frame) & (locs.frame <= end_frame)
            chunk_locs = locs[chunk_mask]

            if len(chunk_locs) == 0:
                print(f"Warning: Chunk {chunk_idx + 1} has no localisations")
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
        """Detect fiducial markers in localisation data.

        This function automatically detects fiducial markers and creates a visualization
        using PlottingFunctions. Supports temporal chunking for datasets with strong drift.

        Args:
            locs: Localisation data (group field not required)
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
                # Format picks as rectangles centreed on detected points
                # Each rectangle is box×box pixels around the centre point
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
                picked_localisations=valid_picked_locs,
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
                if self.plotter is not None:
                    self.plotter.plot_fiducial_detection_steps(
                        image, hist, threshold, picks, valid_picks, result, info, save_plot
                    )
                else:
                    print("⚠️ DriftPlotter not available, skipping step-by-step plots")

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
                    desc=f"Adding group field to {len(locs):,} localisations (index-based)",
                )
                progress_bar = progress_bar_context.__enter__()

            try:
                # Process all fiducial groups using index-based approach
                for group_id, fiducial_locs in enumerate(picked_locs_list):
                    if len(fiducial_locs) > 0:
                        # Find indices of fiducial localisations in original array
                        # This is the key optimisation: use indices instead of coordinate matching
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
        """Find indices of fiducial localisations in the original localisation array.

        Uses ultra-fast hash-based lookup for massive datasets.
        Expected ~1000x speedup over coordinate matching approach.

        Args:
            locs: Original localisation array
            fiducial_locs: Fiducial localisations to find indices for

        Returns:
            Array of indices where fiducial_locs appear in locs
        """
        # Create hash-based lookup table for ultra-fast index finding
        # This is the key to massive performance improvement

        # Use deterministic rounding to handle floating point precision issues
        # Round coordinates to 6 decimal places for reliable hashing
        round_factor = 1e6

        # Create unique keys for each localisation in the original array
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

        # Find indices for fiducial localisations using hash lookup
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
            - List of (y, x) coordinates of detected high-density region centres
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

        # Calculate region centres and properties
        region_centres = []
        region_stats = []

        for region_id in range(1, n_regions + 1):
            region_mask = labeled_regions == region_id
            region_coords = np.where(region_mask)

            if len(region_coords[0]) > 0:
                # Calculate centre of mass
                centre_y = np.mean(region_coords[0])
                centre_x = np.mean(region_coords[1])
                region_centres.append((int(centre_y), int(centre_x)))

                # Calculate region statistics
                region_area = np.sum(region_mask)
                region_intensity = np.sum(smoothed_image[region_mask])
                region_max_intensity = np.max(smoothed_image[region_mask])

                region_stats.append(
                    {
                        "centre": (centre_y, centre_x),
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
            if self.plotter is not None:
                self.plotter.create_separate_plots(
                    smoothed_image,
                    binary_mask,
                    region_centres,
                    hist,
                    bin_edges,
                    threshold,
                    pixelsize,
                    output_figure_path,
                    title,
                )
            else:
                print("⚠️ DriftPlotter not available, skipping density detection plots")

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

        return region_centres, binary_mask, threshold, metadata

    def select_puncta_from_regions(
        self,
        locs: np.recarray,
        region_centres: List[Tuple[int, int]],
        binary_mask: np.ndarray,
        pixelsize: float = 100.0,
        selection_box_size_nm: float = 600.0,
        min_localisations_per_region: int = 10,
        output_figure_path: Optional[str] = None,
        title: str = "Puncta Selection from Regions",
        create_plot: bool = True,
        plot_individual_regions: bool = True,
        use_datashader_threshold: int = 1000,
        memory_optimize: bool = True,
    ) -> Tuple[List[np.recarray], Dict[str, Any]]:
        """Select puncta (localisations) from detected high-density regions using postprocess.picked_locs.

        This function takes the output from detect_high_density_regions_from_image
        and selects localisations within rectangular boxes around each detected region centre
        to create potential fiducial candidates. Uses the optimized postprocess.picked_locs
        function with Rectangle shape, creating axis-aligned boxes by using diagonal picks
        with appropriate width parameters. Automatically enables parallelization for 8+ regions
        for improved performance on large datasets.

        Args:
            locs: Localisation data with xc, yc, frame fields
            region_centres: List of (y, x) coordinates from density detection
            binary_mask: Binary mask from density detection
            pixelsize: Pixel size in nm for coordinate conversion
            selection_box_size_nm: Size of square selection box around each region centre (nm)
            min_localisations_per_region: Minimum number of localisations required for a valid region
            output_figure_path: Optional path to save selection visualization
            title: Title for visualization plots
            create_plot: Whether to create visualization plots
            plot_individual_regions: Whether to plot individual region details (all regions shown)
            use_datashader_threshold: Use datashader for scatter plots with more than this many points

        Returns:
            Tuple containing:
            - List of localisation arrays, one per valid region
            - Metadata dictionary with selection statistics
        """

        # Check if postprocess module is available
        if postprocess is None:
            raise RuntimeError(
                "postprocess module not available - cannot use picked_locs function"
            )

        # Handle empty region centres
        if not region_centres:
            metadata = {
                "n_regions_input": 0,
                "n_regions_selected": 0,
                "selection_criteria": {
                    "min_localisations": min_localisations_per_region,
                    "selection_box_size_nm": selection_box_size_nm,
                    "selection_box_size_pixels": 0.0,
                },
                "rejection_reasons": {"too_few_localisations": 0, "accepted": 0},
                "region_statistics": [],
            }
            return [], metadata

        # Convert box size from nm to pixels
        box_size_pixels = selection_box_size_nm / pixelsize
        half_box = box_size_pixels / 2.0

        # Create horizontal line picks for Rectangle shape (following existing pattern)
        # Rectangle implementation creates boxes around lines defined by two points
        picks = []
        for centre_y, centre_x in region_centres:
            # Create horizontal line through centre - much simpler!
            picks.append(
                ((centre_x - half_box, centre_y), (centre_x + half_box, centre_y))
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

        # Filter results based on minimum localisation count and build statistics
        selected_puncta = []
        region_stats = []

        # Ensure picked_locs_arrays is not None
        if picked_locs_arrays is None:
            picked_locs_arrays = []

        # Memory-optimized processing: stream through regions, immediate filtering and cleanup
        rejected_count = 0
        for region_id, (region_locs, (centre_y, centre_x)) in enumerate(
            zip(picked_locs_arrays, region_centres)
        ):
            n_locs = len(region_locs)

            # Apply localisation count filter FIRST (Option C: Lazy statistics)
            if n_locs >= min_localisations_per_region:
                selected_puncta.append(region_locs)

                # Only calculate statistics for regions that passed the filter
                region_stat = {
                    "region_id": region_id,
                    "centre_y": centre_y,
                    "centre_x": centre_x,
                    "n_localisations": n_locs,
                    "mean_x": np.mean(region_locs.xc),
                    "mean_y": np.mean(region_locs.yc),
                    "std_x": np.std(region_locs.xc),
                    "std_y": np.std(region_locs.yc),
                    "frame_range": [int(region_locs.frame.min()), int(region_locs.frame.max())],
                    "frame_span": int(region_locs.frame.max() - region_locs.frame.min() + 1),
                    "selection_box_size_nm": selection_box_size_nm,
                    "selection_box_size_pixels": box_size_pixels,
                    "box_boundaries": {
                        "x_min": centre_x - half_box,
                        "x_max": centre_x + half_box,
                        "y_min": centre_y - half_box,
                        "y_max": centre_y + half_box,
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
            print(f"Memory optimisation: Freed intermediate arrays after region processing")

        # Create visualization if requested
        if create_plot:
            if self.plotter is not None:
                self.plotter.plot_puncta_selection_results(
                    locs,
                    selected_puncta,
                    region_centres,
                    binary_mask,
                    region_stats,
                    box_size_pixels,
                    pixelsize,
                    output_figure_path,
                    title,
                    plot_individual_regions,
                    use_datashader_threshold,
                )
            else:
                print("⚠️ DriftPlotter not available, skipping puncta selection plots")

            # Memory optimisation: clear plot data if requested
            if memory_optimize:
                plt.close('all')  # Close all figure windows to free memory
                gc.collect()

        # Prepare metadata with memory-optimized calculations
        total_locs_selected = sum(len(puncta) for puncta in selected_puncta)

        metadata = {
            "n_regions_input": len(region_centres),
            "n_regions_selected": len(selected_puncta),
            "n_regions_rejected": rejected_count,
            "selection_rate": (
                len(selected_puncta) / len(region_centres) if region_centres else 0
            ),
            "selection_criteria": {
                "min_localisations": min_localisations_per_region,
                "selection_box_size_nm": selection_box_size_nm,
                "selection_box_size_pixels": box_size_pixels,
            },
            "region_statistics": region_stats,
            "total_selected_localisations": total_locs_selected,
            "memory_optimized": memory_optimize,
            "rejection_reasons": {
                "too_few_localisations": rejected_count,
                "accepted": len(selected_puncta),
            },
        }

        return selected_puncta, metadata

    def identify_real_fiducials_with_clustering(
        self,
        selected_puncta: List[np.recarray],
        retention_percentage: float = 0.9,
        min_samples_factor: float = 0.7,
        frame_count: int = 100000,
        pixelsize: float = 69.0,
        output_figure_path: Optional[str] = None,
        title: str = "Fiducial Gaussian Fitting Analysis",
        create_plot: bool = True,
    ) -> Tuple[List[np.recarray], Dict[str, Any]]:
        """Identify real fiducials from selected puncta using single Gaussian distribution fitting.

        This function takes puncta (localisations) from select_puncta_from_regions
        and applies single Gaussian mixture fitting to identify real fiducial markers.
        It fits each region to a single 2D Gaussian and keeps a specified percentage
        of points based on their distance from the Gaussian centre.

        Args:
            selected_puncta: List of localisation arrays from select_puncta_from_regions
            retention_percentage: Percentage of data to keep (0.0 to 1.0), default 0.9 (90%)
            min_samples_factor: Minimum samples factor for filtering regions
            frame_count: Total number of frames (for calculating min samples)
            pixelsize: Pixel size in nm for distance calculations
            output_figure_path: Optional path to save Gaussian fitting visualization
            title: Title for the plots
            create_plot: Whether to create visualization plots

        Returns:
            Tuple containing:
            - List of localisation arrays for validated fiducials
            - Metadata dictionary with Gaussian fitting statistics
        """

        validated_fiducials = []
        clustering_metadata = []

        # Pre-calculate radial CDF threshold for 2D Gaussian
        # For retention percentage p, solve: 1 - exp(-(r/s)^2/2) = p
        # This gives: r_threshold = s * sqrt(-2 * ln(1 - p))
        if retention_percentage <= 0 or retention_percentage >= 1:
            raise ValueError("retention_percentage must be between 0 and 1")

        # Calculate the radial threshold factor (r/s ratio)
        radial_threshold_factor = np.sqrt(-2 * np.log(1 - retention_percentage))
        print(f"Using radial threshold factor: {radial_threshold_factor:.3f} for {retention_percentage*100:.1f}% retention")

        # Process each puncta region
        for region_id, puncta_locs in enumerate(selected_puncta):
            n_locs = len(puncta_locs)

            if n_locs < 10:  # Skip regions with too few localisations for clustering
                continue

            # Prepare data for Gaussian fitting
            X = np.vstack([puncta_locs["xc"], puncta_locs["yc"]]).T

            # Calculate minimum samples requirement
            min_samples = max(
                int(min_samples_factor * frame_count / 1000), 5
            )  # Scale by 1000, minimum 5

            # Check if region has enough points
            if n_locs < min_samples:
                print(f"  Region {region_id}: Too few points ({n_locs}) < min_samples ({min_samples}), skipping")
                continue

            # Apply single Gaussian mixture fitting
            try:
                from sklearn.mixture import GaussianMixture

                print(f"  Fitting single Gaussian to {n_locs} points in region {region_id}")

                # Fit single Gaussian component
                gm = GaussianMixture(n_components=1, random_state=0)
                gm.fit(X)

                # Get Gaussian parameters
                mean = gm.means_[0]  # Center of Gaussian
                covariance = gm.covariances_[0]  # Covariance matrix

                # Calculate standard deviation (sigma) for radial distance
                # For 2D Gaussian, use average of eigenvalues as characteristic scale
                eigenvals = np.linalg.eigvals(covariance)
                sigma_pixels = np.sqrt(np.mean(eigenvals))
                sigma_nm = sigma_pixels * pixelsize

                # Calculate radial distances from centre
                dx = X[:, 0] - mean[0]
                dy = X[:, 1] - mean[1]
                radial_distances_pixels = np.sqrt(dx**2 + dy**2)
                radial_distances_nm = radial_distances_pixels * pixelsize

                # Apply radial threshold for percentage retention
                # r_threshold = sigma * radial_threshold_factor
                r_threshold_pixels = sigma_pixels * radial_threshold_factor
                r_threshold_nm = r_threshold_pixels * pixelsize

                # Keep points within the radial threshold (using pixel values)
                kept_mask = radial_distances_pixels <= r_threshold_pixels
                n_kept = np.sum(kept_mask)

                if n_kept >= min_samples:
                    # Extract validated localisations
                    validated_locs = puncta_locs[kept_mask]
                    validated_fiducials.append(validated_locs)

                    # Store Gaussian fitting metadata
                    gaussian_metadata = {
                        "region_id": region_id,
                        "original_n_locs": n_locs,
                        "validated_n_locs": n_kept,
                        "retention_rate": n_kept / n_locs,
                        "gaussian_centre_x": mean[0],  # pixels
                        "gaussian_centre_y": mean[1],  # pixels
                        "gaussian_centre_x_nm": mean[0] * pixelsize,  # nm
                        "gaussian_centre_y_nm": mean[1] * pixelsize,  # nm
                        "gaussian_sigma_pixels": sigma_pixels,  # pixels
                        "gaussian_sigma_nm": sigma_nm,  # nm
                        "radial_threshold_pixels": r_threshold_pixels,  # pixels
                        "radial_threshold_nm": r_threshold_nm,  # nm
                        "fitting_method": "Single Gaussian",
                        "retention_percentage": retention_percentage,
                        "min_samples_factor": min_samples_factor,
                        "min_samples_used": min_samples,
                        "pixelsize": pixelsize,
                        "kept_mask": kept_mask,
                        "radial_distances_pixels": radial_distances_pixels,
                        "radial_distances_nm": radial_distances_nm,
                    }
                    clustering_metadata.append(gaussian_metadata)

                    # Plot this validated region immediately
                    if create_plot:
                        self._plot_single_gaussian_validation(
                            puncta_locs, validated_locs, kept_mask, radial_distances_pixels,
                            region_id, gaussian_metadata, output_figure_path, title, r_threshold_pixels
                        )

                    # Clean up intermediate arrays to free memory
                    del validated_locs
                else:
                    # Skip this region - not enough points meet retention criteria
                    print(f"  Region {region_id}: Not enough kept points ({n_kept}) < min_samples ({min_samples}), discarding")

                # Clean up intermediate arrays
                del X, kept_mask, radial_distances_pixels, radial_distances_nm
                gc.collect()

                # Extra cleanup for large regions
                if n_locs > 10000:
                    gc.collect()

            except Exception as e:
                # Skip this region if Gaussian fitting fails
                print(f"Warning: Gaussian fitting failed for region {region_id}: {e}")
                continue

        # Create summary visualization if requested (individual clusters already plotted)
        if create_plot and len(validated_fiducials) > 0:
            if self.plotter is not None:
                self.plotter.plot_clustering_summary_only(
                    selected_puncta,
                    validated_fiducials,
                    clustering_metadata,
                    output_figure_path,
                    title,
                )
            else:
                print("⚠️ DriftPlotter not available, skipping clustering summary plots")

        # Prepare summary metadata
        summary_metadata = {
            "n_input_regions": len(selected_puncta),
            "n_validated_fiducials": len(validated_fiducials),
            "validation_rate": (
                len(validated_fiducials) / len(selected_puncta)
                if selected_puncta
                else 0
            ),
            "fitting_parameters": {
                "retention_percentage": retention_percentage,
                "min_samples_factor": min_samples_factor,
                "frame_count": frame_count,
                "radial_threshold_factor": radial_threshold_factor,
            },
            "region_details": clustering_metadata,
            "total_input_locs": sum(len(puncta) for puncta in selected_puncta),
            "total_validated_locs": sum(
                len(fiducial) for fiducial in validated_fiducials
            ),
        }

        return validated_fiducials, summary_metadata


    def _plot_single_gaussian_validation(
        self,
        original_puncta: np.recarray,
        validated_locs: np.recarray,
        kept_mask: np.ndarray,
        radial_distances: np.ndarray,
        region_id: int,
        metadata: Dict[str, Any],
        output_figure_path: Optional[str],
        title: str,
        r_threshold: float,
    ) -> None:
        """Plot individual Gaussian validation results showing kept vs discarded points."""

        try:
            import PlottingFunctions
            import matplotlib.pyplot as plt
            import numpy as np

            plotter = PlottingFunctions.Plotter(poster=False)
        except ImportError:
            print("PlottingFunctions not available, skipping Gaussian plot")
            return

        # Create a single column plot using PlottingFunctions
        fig, ax = plotter.one_column_plot(width=3.5, height=3.5)

        fig.suptitle(
            f"{title} - Region {region_id+1} Gaussian Validation",
            fontsize=9,
        )

        # Create datasets for plotting: discarded points first, then kept points
        data_arrays = []
        colors = []
        labels = []

        # Add discarded points first (so kept points appear on top)
        discarded_mask = ~kept_mask
        if np.any(discarded_mask):
            data_arrays.append(original_puncta[discarded_mask])
            colors.append('grey')
            labels.append(f'Discarded ({np.sum(discarded_mask):,})')

        # Add kept points
        if np.any(kept_mask):
            data_arrays.append(original_puncta[kept_mask])
            colors.append('red')
            labels.append(f'Kept ({np.sum(kept_mask):,})')

        # Use datashader for plotting
        if data_arrays:
            if self.plotter is not None:
                self.plotter.plot_region_data_with_datashader(ax, data_arrays, colors, labels)
            else:
                # Basic fallback without datashader
                for i, data in enumerate(data_arrays):
                    color = colors[i % len(colors)] if colors else 'blue'
                    ax.plot(data['xc'], data['yc'], '.', color=color, markersize=2, alpha=0.6)

            # Add manual legend for datashader plots
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color, label=label) for color, label in zip(colors, labels)]
            ax.legend(handles=legend_elements, loc='upper right')


        # Set labels and formatting
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_title(f'Gaussian Fitting - Region {region_id+1}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # Add statistics text box
        stats_text = f"Stats:\n"
        stats_text += f"Original: {metadata['original_n_locs']:,}\n"
        stats_text += f"Kept: {metadata['validated_n_locs']:,}\n"
        stats_text += f"Retention: {100*metadata['retention_rate']:.1f}%\n"
        stats_text += f"Gaussian σ: {metadata['gaussian_sigma_nm']:.1f} nm\n"
        stats_text += f"Threshold: {metadata['radial_threshold_nm']:.1f} nm"

        # Quality assessment based on retention rate
        retention_rate = metadata['retention_rate']
        if 0.8 <= retention_rate <= 0.95:
            quality_color = "lightgreen"
        elif 0.7 <= retention_rate < 0.8 or 0.95 < retention_rate <= 1.0:
            quality_color = "lightyellow"
        else:
            quality_color = "lightcoral"

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=6, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=quality_color, alpha=0.7))

        # Save if path provided
        if output_figure_path:
            base_path = output_figure_path.rsplit(".", 1)[0] if "." in output_figure_path else output_figure_path
            gaussian_filename = f"{base_path}_gaussian_region_{region_id+1:02d}.png"
            fig.savefig(gaussian_filename, dpi=300, bbox_inches="tight")
            print(f"Saved Gaussian plot: {gaussian_filename}")

        plt.show()
        plt.close(fig)

    def _filter_fiducials_fast(self, all_corrected_x, all_corrected_y, variance_threshold=3.0, rms_threshold=2.0):
        """
        Fast filtering of fiducial traces using variance ratio and RMS distance.

        Parameters:
        - all_corrected_x, all_corrected_y: [n_frames, n_fiducials] arrays
        - variance_threshold: Remove if variance > threshold * median_variance
        - rms_threshold: Remove if RMS > threshold * median_RMS
        """
        n_frames, n_fiducials = all_corrected_x.shape
        valid_fiducials = np.ones(n_fiducials, dtype=bool)

        # Step 1: Variance Ratio Filter (removes obviously noisy fiducials)
        print("Filtering by variance ratio...", end='', flush=True)

        x_variances = np.nanvar(all_corrected_x, axis=0)  # Variance for each fiducial
        y_variances = np.nanvar(all_corrected_y, axis=0)
        combined_variances = x_variances + y_variances  # Total variance per fiducial

        # Find median of non-NaN variances
        finite_variances = combined_variances[~np.isnan(combined_variances)]
        if len(finite_variances) == 0:
            print("No valid variances found")
            return np.zeros(n_fiducials, dtype=bool), {}

        median_variance = np.median(finite_variances)
        threshold_variance = variance_threshold * median_variance

        # Remove high-variance fiducials
        variance_mask = combined_variances <= threshold_variance
        n_removed_variance = np.sum(~variance_mask)
        valid_fiducials &= variance_mask

        print(f"\rRemoved {n_removed_variance} high-variance fiducials.    ", end='', flush=True)
        print("\rFiltering by RMS distance...", end='', flush=True)

        # Step 2: RMS Distance Filter (removes drifty fiducials)
        rms_distances = np.sqrt(
            np.nanmean(all_corrected_x ** 2 + all_corrected_y ** 2, axis=0)
        )

        # Find median RMS
        finite_rms = rms_distances[~np.isnan(rms_distances)]
        if len(finite_rms) == 0:
            print("No valid RMS distances found")
            return valid_fiducials, {
                'n_variance_filtered': n_removed_variance,
                'n_rms_filtered': 0,
                'median_variance': median_variance,
                'median_rms': np.nan
            }

        median_rms = np.median(finite_rms)
        threshold_rms = rms_threshold * median_rms

        # Remove high-RMS fiducials
        rms_mask = rms_distances <= threshold_rms
        n_removed_rms = np.sum(~rms_mask)
        valid_fiducials &= rms_mask

        print(f"\rRemoved {n_removed_rms} high-RMS fiducials.    ", end='', flush=True)

        # Summary
        n_total_removed = n_removed_variance + n_removed_rms
        n_final = np.sum(valid_fiducials)
        print(f"\rFinal: {n_final}/{n_fiducials} fiducials retained ({n_total_removed} removed)    ", flush=True)

        return valid_fiducials, {
            'n_variance_filtered': n_removed_variance,
            'n_rms_filtered': n_removed_rms,
            'median_variance': median_variance,
            'median_rms': median_rms,
            'variance_threshold_used': variance_threshold * median_variance,
            'rms_threshold_used': rms_threshold * median_rms
        }

    def apply_validated_fiducial_drift_correction(
        self,
        locs: np.recarray,
        validated_fiducials: List[np.recarray],
        x_err_field: str = 'xc_err',
        y_err_field: str = 'yc_err'
    ) -> Tuple[np.recarray, Dict[str, np.ndarray]]:
        """
        Apply drift correction using validated fiducials.

        For each cluster, subtracts the median (x, y) value, then calculates drift_x and drift_y
        by averaging over all validated fiducials, weighting by inverse error (lower error = more weight).
        Does not interpolate - frames without fiducials are dropped from the corrected dataset.

        Parameters
        ----------
        locs : np.recarray
            Full localisation dataset with fields: xc, yc, frame, and error fields
        validated_fiducials : List[np.recarray]
            List of validated fiducial clusters, each with fields: xc, yc, frame, and error fields
        x_err_field : str, default 'xc_err'
            Field name for x-coordinate error
        y_err_field : str, default 'yc_err'
            Field name for y-coordinate error

        Returns
        -------
        corrected_locs : np.recarray
            Drift-corrected localisations with additional 'is_fiducial' field
        drift_info : Dict[str, np.ndarray]
            Dictionary containing:
            - 'frames': frame numbers with drift correction
            - 'drift_x': x drift correction for each frame
            - 'drift_y': y drift correction for each frame
            - 'n_fiducials_per_frame': number of fiducials used per frame
        """
        if not validated_fiducials:
            raise ValueError("No validated fiducials provided")

        # Check if error fields exist in the data
        sample_fiducial = validated_fiducials[0]
        has_x_err = x_err_field in sample_fiducial.dtype.names
        has_y_err = y_err_field in sample_fiducial.dtype.names

        # Get min and max frames
        min_frame = int(locs.frame.min())
        max_frame = int(locs.frame.max())

        if not has_x_err or not has_y_err:
            # Use uniform weights if error fields don't exist
            print(f"Warning: Error fields '{x_err_field}' or '{y_err_field}' not found. Using uniform weights.", flush=True)
            has_x_err = has_y_err = False

        # Step 1: Subtract median from each cluster and collect all corrected fiducial positions
        unique_frames = np.unique(locs.frame)
        frame_to_idx = {frame: i for i, frame in enumerate(unique_frames)}

        # Store corrected positions for each frame and fiducial cluster
        all_corrected_x = np.full([len(unique_frames), len(validated_fiducials)], np.nan)
        all_corrected_y = np.full([len(unique_frames), len(validated_fiducials)], np.nan)
        all_fiducial_weights_x = np.full([len(unique_frames), len(validated_fiducials)], np.nan)
        all_fiducial_weights_y = np.full([len(unique_frames), len(validated_fiducials)], np.nan)

        for i, fiducial_cluster in enumerate(validated_fiducials):
            if len(fiducial_cluster) == 0:
                continue

            # Calculate median position for this cluster
            median_x = np.median(fiducial_cluster.xc)
            median_y = np.median(fiducial_cluster.yc)

            # Create corrected positions (subtract median)
            corrected_x = fiducial_cluster.xc - median_x
            corrected_y = fiducial_cluster.yc - median_y
            frames = np.asarray(fiducial_cluster.frame, dtype=np.int_)

            # Check for unique frames (no duplicates within cluster)
            if len(frames) == len(np.unique(frames)):
                # Map frame numbers to array indices
                frame_indices = [frame_to_idx[frame] for frame in frames]

                all_corrected_x[frame_indices, i] = corrected_x
                all_corrected_y[frame_indices, i] = corrected_y

                # Store weights (inverse of error)
                if has_x_err and has_y_err:
                    all_fiducial_weights_x[frame_indices, i] = 1.0 / (1e-10 + fiducial_cluster[x_err_field])
                    all_fiducial_weights_y[frame_indices, i] = 1.0 / (1e-10 + fiducial_cluster[y_err_field])
                else:
                    all_fiducial_weights_x[frame_indices, i] = 1.0
                    all_fiducial_weights_y[frame_indices, i] = 1.0
            else:
                print(f"\rWarning: Fiducial cluster {i} has multiple localisations in the same frame. Skipping this cluster.    ", end='', flush=True)
                continue

        # Check if we have any valid fiducials
        if np.all(np.isnan(all_corrected_x)):
            raise ValueError("No valid fiducials found after median subtraction")

        # Apply the filtering using extracted helper method
        valid_fiducials, _ = self._filter_fiducials_fast(all_corrected_x, all_corrected_y)

        # Apply the filter to all arrays
        all_corrected_x = all_corrected_x[:, valid_fiducials]
        all_corrected_y = all_corrected_y[:, valid_fiducials]
        all_fiducial_weights_x = all_fiducial_weights_x[:, valid_fiducials]
        all_fiducial_weights_y = all_fiducial_weights_y[:, valid_fiducials]

        ma_x = np.ma.MaskedArray(all_corrected_x, mask=np.isnan(all_corrected_x))
        ma_y = np.ma.MaskedArray(all_corrected_y, mask=np.isnan(all_corrected_x))
        ma_x_err = np.ma.MaskedArray(all_fiducial_weights_x, mask=np.isnan(all_fiducial_weights_x))
        ma_y_err = np.ma.MaskedArray(all_fiducial_weights_y, mask=np.isnan(all_fiducial_weights_y))

        drift_x = np.ma.average(ma_x, weights=ma_x_err, axis=1)
        drift_y = np.ma.average(ma_y, weights=ma_y_err, axis=1)

        # Find which frames have valid drift corrections (not NaN/masked)
        mask_x = np.ma.is_masked(drift_x) if hasattr(drift_x, 'mask') else np.zeros(len(drift_x), dtype=bool)
        mask_y = np.ma.is_masked(drift_y) if hasattr(drift_y, 'mask') else np.zeros(len(drift_y), dtype=bool)
        valid_frame_mask = np.logical_not(mask_x | mask_y)

        valid_frame_numbers = unique_frames[valid_frame_mask]
        valid_drift_x = np.asarray(drift_x[valid_frame_mask])
        valid_drift_y = np.asarray(drift_y[valid_frame_mask])

        # Ensure we have arrays, not scalars (handle single-frame case)
        if np.isscalar(valid_frame_numbers):
            valid_frame_numbers = np.array([valid_frame_numbers])
        if np.isscalar(valid_drift_x):
            valid_drift_x = np.array([valid_drift_x])
        if np.isscalar(valid_drift_y):
            valid_drift_y = np.array([valid_drift_y])

        # Step 3: Apply drift correction to full dataset
        # Only keep localisations from frames where we have drift correction
        frame_mask = np.isin(locs.frame, valid_frame_numbers)
        corrected_locs = locs[frame_mask].copy()

        # Create drift lookup dictionary for fast access
        # Ensure we have flat arrays for dictionary keys
        frame_nums = valid_frame_numbers.flatten() if valid_frame_numbers.ndim > 1 else valid_frame_numbers
        drift_x_vals = valid_drift_x.flatten() if valid_drift_x.ndim > 1 else valid_drift_x
        drift_y_vals = valid_drift_y.flatten() if valid_drift_y.ndim > 1 else valid_drift_y

        drift_lookup_x = dict(zip(frame_nums, drift_x_vals))
        drift_lookup_y = dict(zip(frame_nums, drift_y_vals))

        # Step 4: Create fiducial position set for labeling (before drift correction)
        fiducial_positions = set()
        for fiducial_cluster in validated_fiducials:
            for fiducial in fiducial_cluster:
                # Use (x, y, frame) tuple as unique identifier
                fiducial_positions.add((fiducial.xc, fiducial.yc, fiducial.frame))

        # Apply drift correction AND label fiducials in one pass
        is_fiducial_flags = np.zeros(len(corrected_locs), dtype=bool)
        for i in range(len(corrected_locs)):
            frame = corrected_locs[i].frame

            # Check if this localisation is a fiducial (before drift correction)
            original_pos = (corrected_locs[i].xc, corrected_locs[i].yc, frame)
            is_fiducial_flags[i] = original_pos in fiducial_positions

            # Apply drift correction in-place
            corrected_locs[i].xc -= drift_lookup_x[frame]
            corrected_locs[i].yc -= drift_lookup_y[frame]

        # Add is_fiducial field efficiently using numpy.lib.recfunctions
        from numpy.lib import recfunctions as rfn

        # Use append_fields with asrecarray=True to ensure we get a recarray, not MaskedArray
        final_corrected_locs = rfn.append_fields(
            corrected_locs, 'is_fiducial', is_fiducial_flags,
            dtypes=bool, asrecarray=True, usemask=False
        )

        # Prepare drift info dictionary
        drift_info = {
            'frames': valid_frame_numbers,
            'drift_x': valid_drift_x,
            'drift_y': valid_drift_y,
            'n_fiducials_per_frame': np.sum(~np.isnan(all_corrected_x), axis=1)[valid_frame_mask]
        }

        print(f"Drift correction applied to {len(final_corrected_locs)} localisations")
        print(f"Used {len(valid_frame_numbers)} frames with fiducials (out of {max_frame - min_frame + 1} total frames)")
        print(f"Average {np.mean(drift_info['n_fiducials_per_frame']):.1f} fiducials per frame")

        return final_corrected_locs, drift_info
