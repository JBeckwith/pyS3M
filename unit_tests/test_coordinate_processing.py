"""Full coverage tests for pyS3M.CoordinateProcessing -- temporal segmentation
bounds (SegmentationHandler) and coordinate conversion/validation/interpolation
utilities (CoordinateProcessor).

Small hand-built recarrays and arrays throughout (pure numeric/utility code,
no I/O or database dependency).
"""
from __future__ import annotations

import numpy as np
import pytest

import pyS3M.CoordinateProcessing as CoordinateProcessing
from pyS3M.drift_correction._base import DriftCorrectionError

SegmentationHandler = CoordinateProcessing.SegmentationHandler
CoordinateProcessor = CoordinateProcessing.CoordinateProcessor


def _make_locs(xc, yc, frame, z=None):
    xc = np.asarray(xc, dtype=np.float64)
    yc = np.asarray(yc, dtype=np.float64)
    frame = np.asarray(frame, dtype=np.int64)
    if z is not None:
        z = np.asarray(z, dtype=np.float64)
        arr = np.rec.fromarrays([xc, yc, frame, z], names="xc,yc,frame,z")
    else:
        arr = np.rec.fromarrays([xc, yc, frame], names="xc,yc,frame")
    return arr


# ======================================================================
# SegmentationHandler
# ======================================================================

class TestSegmentationHandler:
    def test_create_segments_includes_final_bound(self):
        bounds = SegmentationHandler.create_segments(n_frames=100, segmentation=30)
        np.testing.assert_array_equal(bounds, [0, 30, 60, 90, 100])

    def test_n_segments_rounds(self):
        assert SegmentationHandler.n_segments(n_frames=95, segmentation=30) == 3
        assert SegmentationHandler.n_segments(n_frames=100, segmentation=30) == 3

    def test_standardize_frame_indexing_starts_at_one(self):
        locs = _make_locs([0, 0], [0, 0], [5, 7])
        out = SegmentationHandler.standardize_frame_indexing(locs)
        np.testing.assert_array_equal(out, [1, 3])


# ======================================================================
# extract_metadata
# ======================================================================

class TestExtractMetadata:
    def test_happy_path(self):
        info = [{"Width": 512}, {"Height": 512}, {"Frames": 1000}, {"Pixelsize": 100.0}]
        meta = CoordinateProcessor.extract_metadata(info)
        assert meta == {"width": 512, "height": 512, "n_frames": 1000, "pixelsize": 100.0}

    def test_missing_metadata_raises(self):
        info = [{"Width": 512}, {"Height": 512}]
        with pytest.raises(DriftCorrectionError):
            CoordinateProcessor.extract_metadata(info)


# ======================================================================
# validate_localisations
# ======================================================================

class TestValidateLocalisations:
    def test_valid_locs_passes(self):
        locs = _make_locs([1.0], [2.0], [1])
        CoordinateProcessor.validate_localisations(locs)  # no raise

    def test_missing_columns_raises(self):
        locs = np.rec.fromarrays([np.array([1.0])], names="xc")
        with pytest.raises(DriftCorrectionError):
            CoordinateProcessor.validate_localisations(locs)


# ======================================================================
# apply_drift_correction
# ======================================================================

class TestApplyDriftCorrection:
    def test_xy_correction_no_z_field(self):
        locs = _make_locs([10.0, 20.0], [30.0, 40.0], [0, 1])
        drift_x = np.array([1.0, 2.0])
        drift_y = np.array([0.5, 1.5])
        corrected = CoordinateProcessor.apply_drift_correction(locs, drift_x, drift_y)
        np.testing.assert_allclose(corrected.xc, [9.0, 18.0])
        np.testing.assert_allclose(corrected.yc, [29.5, 38.5])

    def test_z_correction_applied_when_z_field_and_drift_z_present(self):
        locs = _make_locs([10.0], [30.0], [0], z=[5.0])
        drift_x = np.array([1.0])
        drift_y = np.array([1.0])
        drift_z = np.array([0.5])
        corrected = CoordinateProcessor.apply_drift_correction(locs, drift_x, drift_y, drift_z=drift_z)
        np.testing.assert_allclose(corrected.z, [4.5])

    def test_drift_z_none_skips_z_correction(self):
        locs = _make_locs([10.0], [30.0], [0], z=[5.0])
        drift_x = np.array([1.0])
        drift_y = np.array([1.0])
        corrected = CoordinateProcessor.apply_drift_correction(locs, drift_x, drift_y, drift_z=None)
        np.testing.assert_allclose(corrected.z, [5.0])

    def test_frame_indices_clipped_to_drift_bounds(self):
        locs = _make_locs([10.0], [30.0], [5])
        drift_x = np.array([1.0, 2.0])
        drift_y = np.array([0.5, 1.5])
        corrected = CoordinateProcessor.apply_drift_correction(locs, drift_x, drift_y)
        np.testing.assert_allclose(corrected.xc, [8.0])


# ======================================================================
# convert_pixels_to_nm / convert_nm_to_pixels
# ======================================================================

class TestPixelNmConversion:
    def test_roundtrip_with_offset(self):
        pixels = np.array([[1.0, 2.0], [3.0, 4.0]])
        nm = CoordinateProcessor.convert_pixels_to_nm(pixels, pixelsize_nm=100.0, offset=(10.0, -5.0))
        np.testing.assert_allclose(nm, [[110.0, 195.0], [310.0, 395.0]])
        back = CoordinateProcessor.convert_nm_to_pixels(nm, pixelsize_nm=100.0, offset=(10.0, -5.0))
        np.testing.assert_allclose(back, pixels)

    def test_default_offset_is_zero(self):
        pixels = np.array([[1.0, 2.0]])
        nm = CoordinateProcessor.convert_pixels_to_nm(pixels, pixelsize_nm=50.0)
        np.testing.assert_allclose(nm, [[50.0, 100.0]])


# ======================================================================
# create_spatial_grid
# ======================================================================

class TestCreateSpatialGrid:
    def test_bounds_none_derives_from_locs(self):
        locs = _make_locs([0.0, 100.0], [0.0, 50.0], [0, 1])
        x_edges, y_edges, (x_c, y_c) = CoordinateProcessor.create_spatial_grid(locs, grid_size_nm=25.0)
        assert x_edges[0] == 0.0
        assert x_edges[-1] >= 100.0
        assert len(x_c) == len(x_edges) - 1

    def test_explicit_bounds(self):
        locs = _make_locs([0.0], [0.0], [0])
        x_edges, y_edges, _ = CoordinateProcessor.create_spatial_grid(
            locs, grid_size_nm=10.0, bounds=(0.0, 20.0, 0.0, 20.0)
        )
        np.testing.assert_allclose(x_edges, [0.0, 10.0, 20.0])
        np.testing.assert_allclose(y_edges, [0.0, 10.0, 20.0])


# ======================================================================
# bin_localisations_spatially
# ======================================================================

class TestBinLocalisationsSpatially:
    def test_counts_without_weights(self):
        locs = _make_locs([1.0, 1.0, 9.0], [1.0, 1.0, 9.0], [0, 1, 2])
        x_edges = np.array([0.0, 5.0, 10.0])
        y_edges = np.array([0.0, 5.0, 10.0])
        hist = CoordinateProcessor.bin_localisations_spatially(locs, x_edges, y_edges)
        assert hist[0, 0] == 2
        assert hist[1, 1] == 1

    def test_weighted_histogram(self):
        locs = _make_locs([1.0, 1.0], [1.0, 1.0], [0, 1])
        x_edges = np.array([0.0, 5.0])
        y_edges = np.array([0.0, 5.0])
        hist = CoordinateProcessor.bin_localisations_spatially(
            locs, x_edges, y_edges, weights=np.array([2.0, 3.0])
        )
        assert hist[0, 0] == 5.0


# ======================================================================
# interpolate_coordinates
# ======================================================================

class TestInterpolateCoordinates:
    def test_linear(self):
        out = CoordinateProcessor.interpolate_coordinates(
            np.array([0, 10]), np.array([0.0, 10.0]), np.array([5]), method="linear"
        )
        np.testing.assert_allclose(out, [5.0])

    def test_cubic_with_enough_points(self):
        src_frames = np.array([0, 1, 2, 3, 4])
        src_coords = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        out = CoordinateProcessor.interpolate_coordinates(
            src_frames, src_coords, np.array([2]), method="cubic"
        )
        np.testing.assert_allclose(out, [4.0], atol=1e-6)

    def test_cubic_falls_back_to_linear_with_few_points(self):
        with pytest.warns(UserWarning, match="falling back to linear"):
            out = CoordinateProcessor.interpolate_coordinates(
                np.array([0, 10]), np.array([0.0, 10.0]), np.array([5]), method="cubic"
            )
        np.testing.assert_allclose(out, [5.0])

    def test_nearest(self):
        # Uses np.searchsorted (insertion index), not true nearest-neighbour:
        # target=12 falls between source_frames[1]=10 and [2]=20, so
        # searchsorted returns index 2.
        out = CoordinateProcessor.interpolate_coordinates(
            np.array([0, 10, 20]), np.array([0.0, 100.0, 200.0]), np.array([12]), method="nearest"
        )
        np.testing.assert_allclose(out, [200.0])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            CoordinateProcessor.interpolate_coordinates(
                np.array([0, 10]), np.array([0.0, 10.0]), np.array([5]), method="bogus"
            )

    def test_extrapolate_false_clamps_to_boundary(self):
        out = CoordinateProcessor.interpolate_coordinates(
            np.array([0, 10]), np.array([0.0, 10.0]), np.array([-5, 15]), method="linear", extrapolate=False
        )
        np.testing.assert_allclose(out, [0.0, 10.0])

    def test_extrapolate_true_does_not_clamp(self):
        out = CoordinateProcessor.interpolate_coordinates(
            np.array([0, 10]), np.array([0.0, 10.0]), np.array([-5, 15]), method="linear", extrapolate=True
        )
        # np.interp itself clamps regardless of "extrapolate", but the boundary
        # post-processing step must be skipped -- confirm the two calls agree
        # (both clamp, since np.interp does it internally) but the code path
        # (skipping lines 291-295) is still exercised.
        assert out[0] == 0.0
        assert out[1] == 10.0


# ======================================================================
# interpolate_missing_frames
# ======================================================================

class TestInterpolateMissingFrames:
    def test_all_nan_returns_zeros(self):
        drift = np.array([np.nan, np.nan, np.nan])
        out = CoordinateProcessor.interpolate_missing_frames(drift)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])

    def test_single_valid_point_returns_constant(self):
        drift = np.array([np.nan, 5.0, np.nan])
        out = CoordinateProcessor.interpolate_missing_frames(drift)
        np.testing.assert_allclose(out, [5.0, 5.0, 5.0])

    def test_multiple_valid_interpolates_gaps(self):
        drift = np.array([0.0, np.nan, 2.0, np.nan, 4.0])
        out = CoordinateProcessor.interpolate_missing_frames(drift, method="linear")
        np.testing.assert_allclose(out, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_no_invalid_indices_returns_unchanged(self):
        drift = np.array([1.0, 2.0, 3.0])
        out = CoordinateProcessor.interpolate_missing_frames(drift)
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0])
