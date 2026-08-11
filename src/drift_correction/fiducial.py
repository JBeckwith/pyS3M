"""
drift_correction/fiducial.py

Fiducial-based drift corrector.
Extracted from DriftCorrectionFunctions.py for better code organisation.

:authors: Claude Code (based on Joerg Schnitzbauer, Maximilian Thomas Strauss, Hongqiang Ma, Maomao Chen)
:copyright: Copyright (c) 2025 pyS3M
"""

import warnings
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from ._base import (
    DriftCorrector,
    DriftCorrectionError,
    DriftParameters,
    DriftResult,
    DriftMethod,
    FiducialDetectionResult,
)
from pyS3M.Constants import DriftConstants

try:
    from pyS3M.FiducialDetection import FiducialDetector, DriftPlotter
    _drift_plotter = DriftPlotter()
    _fiducial_detector = FiducialDetector()
except ImportError:
    warnings.warn(
        "Could not import FiducialDetector/DriftPlotter. Plotting/detection features may be limited."
    )
    _drift_plotter = None
    _fiducial_detector = None

try:
    import pyS3M.render as render
    import pyS3M.postprocess as postprocess
except ImportError:
    warnings.warn("Could not import render/postprocess modules.")
    render = None
    postprocess = None


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
        from pyS3M.CoordinateProcessing import CoordinateProcessor

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
        from pyS3M.CoordinateProcessing import CoordinateProcessor
        return CoordinateProcessor.interpolate_missing_frames(
            drift_mean, method="linear"
        )

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
        from pyS3M.CoordinateProcessing import CoordinateProcessor

        if render is None:
            raise DriftCorrectionError(
                "Fiducial detection requires render module"
            )

        # Extract metadata for pixel size
        meta = CoordinateProcessor.extract_metadata(info)
        pixelsize = meta.get("pixelsize", DriftConstants.XIMEA_PIXEL_SIZE_NM)
        n_frames = int(meta["n_frames"])

        # Render localisations to image for fiducial detection
        image = render.render(
            locs=locs,
            info=info,
            oversampling=1,
            viewport=None,
            blur_method="smooth",
        )[1]

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

        # `identify_in_image`'s threshold filters on each local maximum's net
        # gradient (a spot-sharpness measure), not on raw pixel intensity or
        # (the previous, dimensionally wrong) histogram bin counts -- so the
        # percentile must be taken over that same net-gradient distribution.
        # A first, unthresholded pass collects every candidate's gradient;
        # the real threshold is then this distribution's percentile.
        _, _, all_ng = localise.identify_in_image(image, -np.inf, box=box)
        threshold = (
            np.percentile(all_ng, params.fiducial_threshold_percentile)
            if len(all_ng) > 0
            else np.inf
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
            "Rectangle",
            pick_size=box,
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
