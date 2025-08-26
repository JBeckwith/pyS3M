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
from typing import Callable, Optional, Tuple, Union, Dict, Any
import warnings

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from concurrent.futures import ThreadPoolExecutor

# Local imports (will import from existing modules as needed)
import sys
import os

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)

import ProgressUtils

try:
    import render as _render
    import imageprocess as _imageprocess
except ImportError:
    warnings.warn(
        "Could not import render/imageprocess modules. RCC method may not work."
    )
    _render = None
    _imageprocess = None


class DriftMethod(Enum):
    """Enumeration of available drift correction methods."""

    RCC = "rcc"
    AIM = "aim"
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
    """

    segmentation: int = 100
    intersect_d: float = 20 / 69  # Default AIM intersection distance
    roi_r: float = 60 / 69  # Default AIM search radius
    blur_method: str = "gaussian"
    min_blur_width: float = 1.0
    rcc_max_shift: int = 32
    progress_callback: Optional[Callable[[int], None]] = None
    display: bool = False

    def validate(self) -> None:
        """Validate parameter values."""
        if self.segmentation <= 0:
            raise DriftCorrectionError("Segmentation must be positive")
        if self.intersect_d <= 0:
            raise DriftCorrectionError("Intersection distance must be positive")
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
        """Apply drift correction to localizations in-place.

        Args:
            locs: Localization data to correct
            drift_result: Drift correction result

        Returns:
            Corrected localizations (modified in-place)
        """
        # Apply x,y drift (ensure frame indices are within bounds)
        frame_indices = np.clip(locs.frame - 1, 0, len(drift_result.drift_x) - 1)
        locs.xc -= drift_result.drift_x[frame_indices]
        locs.yc -= drift_result.drift_y[frame_indices]

        # Apply z drift if available
        if drift_result.drift_z is not None and hasattr(locs, "z"):
            locs.z -= drift_result.drift_z[frame_indices]

        return locs


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
        if _render is None or _imageprocess is None:
            raise DriftCorrectionError(
                "RCC method requires render and imageprocess modules"
            )

        # Extract metadata
        meta = CoordinateProcessor.extract_metadata(info)

        # Generate segments
        bounds, segments = self._generate_segments(locs, meta, params)

        # Calculate shifts using RCC
        shift_y, shift_x = _imageprocess.rcc(
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
        self, locs: np.recarray, meta: Dict[str, float], params: DriftParameters
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate temporal segments for RCC analysis.

        Args:
            locs: Localization data
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
                    _, segments[i] = _render.render(
                        segment_locs,
                        [meta],
                        blur_method=params.blur_method,
                        min_blur_width=params.min_blur_width,
                    )
        else:
            params.progress_callback(0)
            for i in range(n_segments):
                segment_locs = locs[
                    (locs.frame >= bounds[i]) & (locs.frame < bounds[i + 1])
                ]
                _, segments[i] = _render.render(
                    segment_locs,
                    [meta],
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

        # interpolate the drifts (cubic spline) for all frames
        t = (seg_bounds[1:] + seg_bounds[:-1]) / 2
        drift_x_pol = InterpolatedUnivariateSpline(t, drift_x, k=3)
        drift_y_pol = InterpolatedUnivariateSpline(t, drift_y, k=3)
        t_inter = np.arange(seg_bounds[-1]) + 1
        drift_x = drift_x_pol(t_inter)
        drift_y = drift_y_pol(t_inter)

        # undrift the localizations
        x_pdc = x - drift_x[frame - 1]
        y_pdc = y - drift_y[frame - 1]

        return x_pdc, y_pdc, drift_x, drift_y

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
        elif _render is not None and _imageprocess is not None:
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
