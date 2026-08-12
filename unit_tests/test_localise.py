#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.localise -- the net-gradient local-maxima
detector (local_maxima / gradient_at / net_gradient / identify_in_image).

`identify_in_image` (and its three numba helpers) is called from
`drift_correction/`'s fiducial peak-finding.

All four functions are `@numba.jit(nopython=True)` -- once JIT-compiled they
run as machine code that bypasses Python's trace hooks entirely, so
coverage.py cannot see line hits inside them no matter how many times
they're actually called. Each function exposes the original, uncompiled
Python implementation via `.py_func`, which *is* traceable -- tests call
both the normal (JIT) path, to verify real runtime behaviour, and the
`.py_func` path, purely so coverage.py can see the body executed.
"""
from __future__ import annotations

import numpy as np
import pytest

import pyS3M.localise as loc


def _spot_frame(size=20, cy=10, cx=10, amplitude=100.0):
    # Zero background (not noise): with a noisy background, many pixels can
    # be a "brightest in its own 5x5 window" local maximum by chance, making
    # exact-match assertions on which pixels get found flaky/nondeterministic.
    frame = np.zeros((size, size), dtype=np.float32)
    frame[cy, cx] = amplitude
    return frame


def _uv_kernels(box):
    box_half = int(box / 2)
    ux = np.zeros((box, box), dtype=np.float32)
    uy = np.zeros((box, box), dtype=np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux**2 + uy**2)
    with np.errstate(invalid="ignore"):
        ux = ux / unorm
        uy = uy / unorm
    return ux, uy


class TestLocalMaxima:
    def test_finds_single_bright_pixel(self):
        frame = _spot_frame()
        y, x = loc.local_maxima(frame, 5)
        assert list(zip(y.tolist(), x.tolist())) == [(10, 10)]

    def test_flat_frame_finds_nothing(self):
        frame = np.ones((20, 20), dtype=np.float32)
        y, x = loc.local_maxima(frame, 5)
        assert len(y) == 0 and len(x) == 0

    def test_py_func_matches_jit(self):
        frame = _spot_frame()
        y_jit, x_jit = loc.local_maxima(frame, 5)
        y_py, x_py = loc.local_maxima.py_func(frame, 5)
        np.testing.assert_array_equal(y_jit, y_py)
        np.testing.assert_array_equal(x_jit, x_py)


class TestGradientAt:
    def test_symmetric_peak_zero_gradient_at_centre(self):
        frame = _spot_frame()
        gy, gx = loc.gradient_at(frame, 10, 10, 0)
        assert gy == 0.0 and gx == 0.0

    def test_off_centre_nonzero_gradient(self):
        frame = _spot_frame()
        gy, gx = loc.gradient_at(frame, 9, 10, 0)
        assert gy != 0.0

    def test_py_func_matches_jit(self):
        frame = _spot_frame()
        assert loc.gradient_at(frame, 9, 10, 0) == loc.gradient_at.py_func(frame, 9, 10, 0)


class TestNetGradient:
    def test_matches_jit_and_py_func(self):
        frame = _spot_frame()
        y, x = loc.local_maxima(frame, 5)
        ux, uy = _uv_kernels(5)
        ng_jit = loc.net_gradient(frame, y, x, 5, uy, ux)
        ng_py = loc.net_gradient.py_func(frame, y, x, 5, uy, ux)
        np.testing.assert_allclose(ng_jit, ng_py)
        assert ng_jit[0] > 0

    def test_empty_candidates(self):
        frame = _spot_frame()
        ux, uy = _uv_kernels(5)
        ng = loc.net_gradient(frame, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), 5, uy, ux)
        assert len(ng) == 0


class TestIdentifyInImage:
    def test_finds_bright_spot(self):
        frame = _spot_frame()
        y, x, ng = loc.identify_in_image(frame, 1.0, 5)
        assert list(zip(y.tolist(), x.tolist())) == [(10, 10)]
        assert ng[0] > 1.0

    def test_high_threshold_filters_everything_out(self):
        frame = _spot_frame()
        y, x, ng = loc.identify_in_image(frame, 1e9, 5)
        assert len(y) == 0

    def test_flat_image_finds_nothing(self):
        frame = np.ones((20, 20), dtype=np.float32)
        y, x, ng = loc.identify_in_image(frame, 1.0, 5)
        assert len(y) == 0

    def test_py_func_matches_jit(self):
        frame = _spot_frame()
        y_jit, x_jit, ng_jit = loc.identify_in_image(frame, 1.0, 5)
        y_py, x_py, ng_py = loc.identify_in_image.py_func(frame, 1.0, 5)
        np.testing.assert_array_equal(y_jit, y_py)
        np.testing.assert_array_equal(x_jit, x_py)
        np.testing.assert_allclose(ng_jit, ng_py)

    def test_multiple_spots(self):
        frame = np.zeros((40, 40), dtype=np.float32)
        frame[10, 10] = 100
        frame[25, 30] = 100
        y, x, ng = loc.identify_in_image(frame, 1.0, 5)
        found = set(zip(y.tolist(), x.tolist()))
        assert found == {(10, 10), (25, 30)}
