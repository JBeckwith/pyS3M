"""
drift_correction/aim.py

AIM (Adaptive Intersection Maximization) drift corrector.
Extracted from DriftCorrectionFunctions.py for better code organisation.

:authors: Claude Code (based on Joerg Schnitzbauer, Maximilian Thomas Strauss, Hongqiang Ma, Maomao Chen)
:copyright: Copyright (c) 2025 pyBayerSMLM
"""

import warnings
from typing import Callable, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from ._base import DriftCorrector, DriftParameters, DriftResult, DriftMethod

import ProgressUtils


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
        from CoordinateProcessing import CoordinateProcessor, SegmentationHandler

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
