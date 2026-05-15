"""
Coordinate Processing and Conversion Module

Contains coordinate transformation and processing utilities.
Extracted from DriftCorrectionFunctions.py for better code organisation.

This module handles:
- Coordinate conversion between pixel and nanometre units
- Frame-based temporal coordinate processing
- Coordinate system transformations
- Spatial binning and gridding operations
- Temporal segmentation
- Drift interpolation

:authors: Claude Code (refactored from DriftCorrectionFunctions.py)
:copyright: Copyright (c) 2025 pyBayerSMLM
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from scipy.interpolate import InterpolatedUnivariateSpline
import warnings

from drift_correction._base import DriftCorrectionError  # noqa: F401 — re-exported for callers


class SegmentationHandler:
    """Utilities for temporal segmentation of localisation data."""

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
            locs: Localisation data

        Returns:
            Frame indices starting at 1
        """
        return locs.frame + 1 - locs.frame.min()

    @staticmethod
    def temporal_coordinate_segmentation(
        locs: np.recarray,
        segment_size_frames: int = 1000,
        overlap_frames: int = 100,
    ) -> List[np.recarray]:
        """Segment localisation data temporally with optional overlap.

        Args:
            locs: Localisation data with frame field
            segment_size_frames: Size of each segment in frames
            overlap_frames: Number of overlapping frames between segments

        Returns:
            List of localisation segments
        """
        if not hasattr(locs, "frame"):
            raise DriftCorrectionError("Localisation data must have 'frame' field")

        min_frame = locs.frame.min()
        max_frame = locs.frame.max()
        n_frames = max_frame - min_frame + 1

        segments = []
        start_frame = min_frame

        while start_frame <= max_frame:
            end_frame = min(start_frame + segment_size_frames, max_frame + 1)

            # Extract localisations in this segment
            mask = (locs.frame >= start_frame) & (locs.frame < end_frame)
            segment_locs = locs[mask]

            if len(segment_locs) > 0:
                segments.append(segment_locs)

            # Move to next segment with overlap
            start_frame += segment_size_frames - overlap_frames

            # Prevent infinite loop if overlap >= segment_size
            if overlap_frames >= segment_size_frames:
                start_frame = end_frame

        return segments


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
    def validate_localisations(locs: np.recarray) -> None:
        """Validate localisation data format.

        Args:
            locs: Localisation record array

        Raises:
            DriftCorrectionError: If required columns are missing
        """
        required_cols = ["xc", "yc", "frame"]
        missing_cols = [col for col in required_cols if not hasattr(locs, col)]

        if missing_cols:
            raise DriftCorrectionError(f"Missing required columns: {missing_cols}")

    @staticmethod
    def apply_drift_correction(
        locs: np.recarray,
        drift_x: np.ndarray,
        drift_y: np.ndarray,
        drift_z: Optional[np.ndarray] = None,
    ) -> np.recarray:
        """Apply drift correction to localisations.

        Args:
            locs: Localisation data to correct
            drift_x: X-axis drift values per frame
            drift_y: Y-axis drift values per frame
            drift_z: Optional Z-axis drift values per frame

        Returns:
            Corrected localisations (copy with corrections applied)
        """
        # Create a copy to avoid modifying original data
        corrected_locs = locs.copy()

        # Apply x,y drift (ensure frame indices are within bounds)
        frame_indices = np.clip(corrected_locs.frame, 0, len(drift_x) - 1)
        corrected_locs.xc -= drift_x[frame_indices]
        corrected_locs.yc -= drift_y[frame_indices]

        # Apply z drift if available
        if drift_z is not None and hasattr(corrected_locs, "z"):
            corrected_locs.z -= drift_z[frame_indices]

        return corrected_locs

    @staticmethod
    def convert_pixels_to_nm(
        pixel_coords: np.ndarray,
        pixelsize_nm: float = 100.0,
        offset: Tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Convert pixel coordinates to nanometre coordinates.

        Args:
            pixel_coords: Array of pixel coordinates (N, 2) for (x, y)
            pixelsize_nm: Size of each pixel in nanometres
            offset: Offset to apply (x_offset, y_offset) in nm

        Returns:
            Array of coordinates in nanometres
        """
        nm_coords = pixel_coords * pixelsize_nm
        nm_coords[:, 0] += offset[0]
        nm_coords[:, 1] += offset[1]
        return nm_coords

    @staticmethod
    def convert_nm_to_pixels(
        nm_coords: np.ndarray,
        pixelsize_nm: float = 100.0,
        offset: Tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Convert nanometre coordinates to pixel coordinates.

        Args:
            nm_coords: Array of coordinates in nanometres (N, 2) for (x, y)
            pixelsize_nm: Size of each pixel in nanometres
            offset: Offset to apply (x_offset, y_offset) in nm

        Returns:
            Array of pixel coordinates
        """
        # Subtract offset first
        coords = nm_coords.copy()
        coords[:, 0] -= offset[0]
        coords[:, 1] -= offset[1]
        # Convert to pixels
        pixel_coords = coords / pixelsize_nm
        return pixel_coords

    @staticmethod
    def create_spatial_grid(
        locs: np.recarray,
        grid_size_nm: float = 100.0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create spatial grid for localisation binning.

        Args:
            locs: Localisation data with xc, yc fields
            grid_size_nm: Size of grid cells in nanometres
            bounds: Optional bounds (x_min, x_max, y_min, y_max) in nm

        Returns:
            Tuple of (x_edges, y_edges, grid_centres)
        """
        if bounds is None:
            x_min, x_max = locs.xc.min(), locs.xc.max()
            y_min, y_max = locs.yc.min(), locs.yc.max()
        else:
            x_min, x_max, y_min, y_max = bounds

        # Create edges with grid_size spacing
        x_edges = np.arange(x_min, x_max + grid_size_nm, grid_size_nm)
        y_edges = np.arange(y_min, y_max + grid_size_nm, grid_size_nm)

        # Calculate grid centres
        x_centres = (x_edges[:-1] + x_edges[1:]) / 2
        y_centres = (y_edges[:-1] + y_edges[1:]) / 2

        return x_edges, y_edges, (x_centres, y_centres)

    @staticmethod
    def bin_localisations_spatially(
        locs: np.recarray,
        x_edges: np.ndarray,
        y_edges: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Bin localisations into spatial histogram.

        Args:
            locs: Localisation data with xc, yc fields
            x_edges: X-axis bin edges
            y_edges: Y-axis bin edges
            weights: Optional weights for each localisation

        Returns:
            2D histogram of localisation counts
        """
        hist, _, _ = np.histogram2d(
            locs.xc, locs.yc, bins=[x_edges, y_edges], weights=weights
        )
        return hist

    @staticmethod
    def calculate_centre_of_mass(
        locs: np.recarray,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, Optional[float]]:
        """Calculate centre of mass of localisation data.

        Args:
            locs: Localisation data with xc, yc fields (and optionally zc)
            weights: Optional weights for each localisation

        Returns:
            Tuple of (centre_x, centre_y, centre_z) where centre_z is None for 2D data
        """
        if weights is None:
            weights = np.ones(len(locs))

        total_weight = weights.sum()
        if total_weight == 0:
            return 0.0, 0.0, None

        centre_x = np.sum(locs.xc * weights) / total_weight
        centre_y = np.sum(locs.yc * weights) / total_weight

        centre_z = None
        if hasattr(locs, "zc"):
            centre_z = np.sum(locs.zc * weights) / total_weight

        return centre_x, centre_y, centre_z

    @staticmethod
    def interpolate_coordinates(
        source_frames: np.ndarray,
        source_coords: np.ndarray,
        target_frames: np.ndarray,
        method: str = "linear",
        extrapolate: bool = False,
    ) -> np.ndarray:
        """Interpolate coordinates to new frame positions.

        Args:
            source_frames: Frame numbers for known coordinates
            source_coords: Known coordinate values
            target_frames: Frame numbers where interpolation is needed
            method: Interpolation method ('linear', 'cubic', 'nearest')
            extrapolate: Whether to extrapolate beyond known range

        Returns:
            Interpolated coordinates at target frames
        """
        if method == "linear":
            # Use numpy linear interpolation
            interpolated = np.interp(target_frames, source_frames, source_coords)
        elif method == "cubic":
            # Use scipy cubic spline
            if len(source_frames) < 4:
                warnings.warn(
                    "Cubic interpolation requires at least 4 points, falling back to linear"
                )
                interpolated = np.interp(target_frames, source_frames, source_coords)
            else:
                spline = InterpolatedUnivariateSpline(source_frames, source_coords, k=3)
                interpolated = spline(target_frames)
        elif method == "nearest":
            # Nearest neighbor interpolation
            indices = np.searchsorted(source_frames, target_frames)
            indices = np.clip(indices, 0, len(source_frames) - 1)
            interpolated = source_coords[indices]
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        # Handle extrapolation
        if not extrapolate:
            # Set values outside range to boundary values
            min_frame, max_frame = source_frames.min(), source_frames.max()
            interpolated[target_frames < min_frame] = source_coords[0]
            interpolated[target_frames > max_frame] = source_coords[-1]

        return interpolated

    @staticmethod
    def interpolate_missing_frames(
        drift_values: np.ndarray, method: str = "linear"
    ) -> np.ndarray:
        """Interpolate drift for frames without localisations (NaN values).

        Args:
            drift_values: Drift array with possible NaN values
            method: Interpolation method ('linear', 'cubic')

        Returns:
            Interpolated drift array
        """
        # Find valid (non-NaN) frames
        valid_mask = ~np.isnan(drift_values)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            # No valid data - return zeros
            return np.zeros_like(drift_values)
        elif len(valid_indices) == 1:
            # Only one valid point - use constant value
            result = np.full_like(drift_values, drift_values[valid_indices[0]])
            return result
        else:
            # Interpolate between valid points
            invalid_indices = np.where(~valid_mask)[0]
            if len(invalid_indices) > 0:
                interpolated = CoordinateProcessor.interpolate_coordinates(
                    valid_indices,
                    drift_values[valid_indices],
                    invalid_indices,
                    method=method,
                    extrapolate=False,
                )
                drift_values[invalid_indices] = interpolated

            return drift_values

    @staticmethod
    def interpolate_drift(
        bounds: np.ndarray,
        shift_x: np.ndarray,
        shift_y: np.ndarray,
        n_frames: int,
        method: str = "cubic",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate drift to all frames using splines.

        Args:
            bounds: Segment boundaries
            shift_x: X shifts between segments
            shift_y: Y shifts between segments
            n_frames: Total number of frames
            method: Interpolation method ('linear', 'cubic')

        Returns:
            Tuple of (drift_x, drift_y) for all frames
        """
        # Calculate segment centres
        t = (bounds[1:] + bounds[:-1]) / 2

        # Interpolate to all frames
        t_inter = np.arange(n_frames)

        if method == "cubic" and len(t) >= 4:
            # Use cubic spline interpolation
            drift_x_pol = InterpolatedUnivariateSpline(t, shift_x, k=3)
            drift_y_pol = InterpolatedUnivariateSpline(t, shift_y, k=3)

            drift_x = drift_x_pol(t_inter)
            drift_y = drift_y_pol(t_inter)
        else:
            # Fall back to linear interpolation
            drift_x = np.interp(t_inter, t, shift_x)
            drift_y = np.interp(t_inter, t, shift_y)

        return drift_x, drift_y

    @staticmethod
    def cubic_spline_interpolation(
        x: np.ndarray, y: np.ndarray, x_new: np.ndarray, k: int = 3
    ) -> np.ndarray:
        """Perform cubic spline interpolation.

        Args:
            x: Known x coordinates
            y: Known y values
            x_new: New x coordinates for interpolation
            k: Spline order (default 3 for cubic)

        Returns:
            Interpolated y values at x_new
        """
        if len(x) < k + 1:
            # Not enough points for requested order, use linear
            return np.interp(x_new, x, y)

        spline = InterpolatedUnivariateSpline(x, y, k=k)
        return spline(x_new)

    @staticmethod
    def _validate_coordinate_arrays(
        coords: np.ndarray,
        expected_dimensions: int = 2,
    ) -> bool:
        """Validate coordinate array format and dimensions.

        Args:
            coords: Coordinate array to validate
            expected_dimensions: Expected number of spatial dimensions

        Returns:
            True if coordinates are valid, False otherwise
        """
        if not isinstance(coords, np.ndarray):
            return False

        if coords.ndim == 1 and expected_dimensions == 1:
            return True

        if coords.ndim == 2 and coords.shape[1] == expected_dimensions:
            return True

        return False

    @staticmethod
    def _apply_coordinate_transformation(
        coords: np.ndarray,
        transformation_matrix: np.ndarray,
        translation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply linear transformation to coordinates.

        Args:
            coords: Input coordinates (N, D) where D is dimensionality
            transformation_matrix: Transformation matrix (D, D)
            translation: Optional translation vector (D,)

        Returns:
            Transformed coordinates
        """
        # Apply matrix transformation
        transformed = coords @ transformation_matrix.T

        # Apply translation if provided
        if translation is not None:
            transformed += translation

        return transformed
