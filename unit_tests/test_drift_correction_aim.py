#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.drift_correction.aim — AIMDriftCorrector.

Part of the drift_correction/ package coverage push (claude/TODO.md PRIORITY 1).
The 2D AIM path's core intersection-counting/FFT-peak machinery is already
exercised by unit_tests/test_drift_correction.py and
unit_tests/test_drift_correction_simple.py -- this file fills the remaining
gaps found by the real coverage baseline:

- The 3D AIM path (_intersection_max_z / _point_intersect_3d / _get_fft_peak_z /
  _run_aim_3d). Confirmed unused anywhere else in this codebase (not called by
  the GUI drift panel, AnalysisPipeline, or any notebook) but it is real,
  reachable code, not dead code -- per user decision (2026-08-11), tested
  properly rather than excluded.
- The 1-indexed-frames branch of _intersection_max and _cubic_spline_interpolation
  (existing tests only exercise 0-indexed frames).
- The explicit progress_callback branch of _intersection_max (existing tests
  only exercise the default clean_progress_bar path).
- The sparse-segment carry-forward branch (a segment with zero localisations
  that isn't the first one processed).

Note: the previously-dead `use_vectorized=False` frame-by-frame branch has
been deleted from aim.py entirely (2026-08-11, per user decision) rather than
tested -- it was hardcoded unreachable (use_vectorized was a local literal,
never False), so there was nothing to preserve.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyS3M.drift_correction._base import DriftParameters
from pyS3M.drift_correction.aim import AIMDriftCorrector


def _small_2d_locs_and_info(n_locs=300, n_frames=20, width=100.0, height=100.0, frame_start=0):
    rng = np.random.default_rng(0)
    frames = rng.integers(frame_start, n_frames + frame_start, n_locs)
    x = rng.uniform(0, width, n_locs)
    y = rng.uniform(0, height, n_locs)
    locs = np.rec.fromarrays([x, y, frames], names=["xc", "yc", "frame"])
    info = [{"Width": width, "Height": height, "Frames": n_frames, "Pixelsize": 69}]
    return locs, info


def _small_3d_locs_and_info(n_locs=400, n_frames=20, width=100.0, height=100.0):
    rng = np.random.default_rng(2)
    frames = rng.integers(0, n_frames, n_locs)
    x = rng.uniform(0, width, n_locs)
    y = rng.uniform(0, height, n_locs)
    z = rng.uniform(-200, 200, n_locs)  # nm
    locs = np.rec.fromarrays([x, y, z, frames], names=["xc", "yc", "z", "frame"])
    info = [{"Width": width, "Height": height, "Frames": n_frames, "Pixelsize": 69}]
    return locs, info


class TestSupports3D:
    def test_supports_3d_is_true(self):
        assert AIMDriftCorrector().supports_3d() is True


class Test2DFrameIndexing:
    def test_one_indexed_frames_via_calculate_drift(self):
        """calculate_drift() always standardizes frames to 1-indexed
        internally (SegmentationHandler.standardize_frame_indexing), so this
        is the only frame-indexing branch reachable through the public API
        -- regardless of whether the caller's raw locs.frame started at 0
        or 1, min_frame is 1 by the time it reaches _intersection_max."""
        locs, info = _small_2d_locs_and_info(frame_start=0)
        aim = AIMDriftCorrector()
        params = DriftParameters(segmentation=5, intersect_d=1.0, roi_r=2.0)
        result = aim.calculate_drift(locs, info, params)
        assert np.all(np.isfinite(result.drift_x))
        assert np.all(np.isfinite(result.drift_y))

    def test_zero_indexed_frames_direct_staticmethod_call(self):
        """_intersection_max/_cubic_spline_interpolation support a
        min_frame == 0 path in their own logic, but it's unreachable through
        calculate_drift() (which always standardizes to 1-indexed first) --
        only reachable by calling the staticmethods directly with a raw
        0-indexed frame array, as done here."""
        locs, info = _small_2d_locs_and_info(frame_start=0)
        assert locs.frame.min() == 0
        width = info[0]["Width"]

        ref_mask = locs.frame <= 5
        x_pdc, y_pdc, drift_x, drift_y = AIMDriftCorrector._intersection_max(
            locs.xc, locs.yc,
            locs.xc[ref_mask], locs.yc[ref_mask],
            locs.frame, np.array([0, 5, 10, 15, 20]),
            intersect_d=1.0, roi_r=2.0, width=width, aim_round=1,
        )
        assert np.all(np.isfinite(drift_x))
        assert np.all(np.isfinite(drift_y))
        assert len(drift_x) == int(locs.frame.max()) + 1  # 0-indexed padding branch


class Test3DAIM:
    """3D AIM is unused elsewhere in the codebase but is real, reachable
    code -- tested properly rather than excluded (user decision 2026-08-11)."""

    def test_calculate_drift_with_z_field_runs_3d_path(self):
        locs, info = _small_3d_locs_and_info()
        aim = AIMDriftCorrector()
        params = DriftParameters(segmentation=5, intersect_d=1.0, roi_r=2.0)
        result = aim.calculate_drift(locs, info, params)

        assert result.drift_z is not None
        assert len(result.drift_z) >= 20
        assert np.all(np.isfinite(result.drift_z))

    def test_without_z_field_drift_z_is_none(self):
        locs, info = _small_2d_locs_and_info()
        assert not hasattr(locs, "z")
        aim = AIMDriftCorrector()
        params = DriftParameters(segmentation=5, intersect_d=1.0, roi_r=2.0)
        result = aim.calculate_drift(locs, info, params)
        assert result.drift_z is None

    def test_point_intersect_3d_and_fft_peak_z_directly(self):
        """Direct, deterministic exercise of the low-level 3D helpers."""
        rng = np.random.default_rng(3)
        intersect_d, width_units, height_units = 1.0, 100.0, 100.0

        ref = rng.integers(-50, 50, (30, 3))
        l0 = np.int32(
            ref[:, 0] + ref[:, 1] * width_units + ref[:, 2] * width_units * height_units
        )
        l0_coords, l0_counts = np.unique(l0, return_counts=True)

        x1 = rng.uniform(-50, 50, 30)
        y1 = rng.uniform(-50, 50, 30)
        z1 = rng.uniform(-50, 50, 30)

        shifts_z = np.arange(-2, 3, dtype=np.int32) * width_units * height_units

        roi_cc = AIMDriftCorrector._point_intersect_3d(
            l0_coords, l0_counts, x1, y1, z1,
            intersect_d, width_units, height_units, shifts_z,
        )
        assert roi_cc.shape == (5,)

        pz = AIMDriftCorrector._get_fft_peak_z(roi_cc, roi_size=4.0)
        assert np.isfinite(pz)


class TestSparseSegmentCarryForward:
    def test_empty_middle_segment_carries_forward_previous_drift(self, monkeypatch):
        """Segment 2 (frames (10,15]) has zero localisations and isn't the
        first segment processed (aim_round=1 starts at s=1) -- exercises the
        `if s > 0: drift_x[s] = drift_x[s - 1]` carry-forward branch.

        _intersection_max returns the *cubic-spline-interpolated* per-frame
        drift, not the raw per-segment array, so the carry-forward is no
        longer visible verbatim in the return value -- bypass the spline
        (identity passthrough) to inspect the raw per-segment drift directly.
        """
        monkeypatch.setattr(
            AIMDriftCorrector,
            "_cubic_spline_interpolation",
            staticmethod(lambda drift_x, drift_y, seg_bounds, min_frame, max_frame: (drift_x, drift_y)),
        )

        rng = np.random.default_rng(1)
        width = 50.0
        ref_x = rng.uniform(0, width, 20)
        ref_y = rng.uniform(0, width, 20)

        frame = np.concatenate([
            rng.integers(1, 6, 15),    # segment 0: frames (0,5]
            rng.integers(6, 11, 15),   # segment 1: frames (5,10]
            # segment 2: frames (10,15] -- deliberately empty
            rng.integers(16, 21, 15),  # segment 3: frames (15,20]
        ]).astype(np.int64)
        x = rng.uniform(0, width, len(frame))
        y = rng.uniform(0, width, len(frame))
        seg_bounds = np.array([0, 5, 10, 15, 20])

        x_pdc, y_pdc, drift_x, drift_y = AIMDriftCorrector._intersection_max(
            x, y, ref_x, ref_y, frame, seg_bounds,
            intersect_d=1.0, roi_r=2.0, width=width, aim_round=1,
        )
        assert len(drift_x) == 4  # raw per-segment array, spline bypassed
        assert drift_x[2] == drift_x[1]
        assert drift_y[2] == drift_y[1]


class TestExplicitProgressCallback:
    def test_progress_callback_invoked_per_segment(self):
        locs, info = _small_2d_locs_and_info()
        calls = []
        aim = AIMDriftCorrector()
        params = DriftParameters(
            segmentation=5, intersect_d=1.0, roi_r=2.0,
            progress_callback=lambda s: calls.append(s),
        )
        aim.calculate_drift(locs, info, params)
        assert len(calls) > 0

    def test_progress_callback_invoked_per_segment_3d(self):
        """Same explicit-callback branch, but in the Z-axis sibling method
        _intersection_max_z (only reachable via a locs.z field)."""
        locs, info = _small_3d_locs_and_info()
        calls = []
        aim = AIMDriftCorrector()
        params = DriftParameters(
            segmentation=5, intersect_d=1.0, roi_r=2.0,
            progress_callback=lambda s: calls.append(s),
        )
        aim.calculate_drift(locs, info, params)
        assert len(calls) > 0


class TestSingleSegmentEdgeCase:
    def test_cubic_spline_interpolation_single_measurement(self):
        """len(drift_x) < 2 (a single segment) hits the edge-case branch in
        _cubic_spline_interpolation that pads instead of extrapolating from
        a neighbour."""
        drift_x = np.array([0.5])
        drift_y = np.array([-0.3])
        seg_bounds = np.array([0, 20])
        drift_x_full, drift_y_full = AIMDriftCorrector._cubic_spline_interpolation(
            drift_x, drift_y, seg_bounds, min_frame=1, max_frame=20
        )
        assert np.all(np.isfinite(drift_x_full))
        assert np.all(np.isfinite(drift_y_full))


class Test3DSparseSegment:
    def test_empty_middle_segment_3d_does_not_crash(self):
        """Z-axis sibling of TestSparseSegmentCarryForward -- a middle
        segment with zero localisations exercises the
        `if s > 0: drift_z[s] = drift_z[s - 1]` branch in
        _intersection_max_z's inner _process_segment_z."""
        rng = np.random.default_rng(4)
        width = height = 50.0
        pixelsize = 100.0
        ref = rng.uniform(-25, 25, (20, 3))

        frame = np.concatenate([
            rng.integers(1, 6, 15),    # segment 0
            rng.integers(6, 11, 15),   # segment 1
            # segment 2: deliberately empty
            rng.integers(16, 21, 15),  # segment 3
        ]).astype(np.int64)
        x = rng.uniform(-25, 25, len(frame))
        y = rng.uniform(-25, 25, len(frame))
        z = rng.uniform(-25, 25, len(frame)) * pixelsize
        seg_bounds = np.array([0, 5, 10, 15, 20])

        z_pdc, drift_z = AIMDriftCorrector._intersection_max_z(
            x, y, z, ref[:, 0], ref[:, 1], ref[:, 2] * pixelsize,
            frame, seg_bounds,
            intersect_d=1.0, roi_r=2.0, width=width, height=height,
            pixelsize=pixelsize, aim_round=1,
        )
        assert np.all(np.isfinite(drift_z))
