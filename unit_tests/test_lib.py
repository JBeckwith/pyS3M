"""Full coverage tests for pyS3M.lib -- recarray column add/remove, radius/
polygon/rectangle spatial-selection helpers (picasso-derived), and pick-corner
geometry helpers.

Small hand-built recarrays and point sets throughout (pure numeric/geometry
code, no I/O or database dependency). `check_if_in_polygon`/
`check_if_in_rectangle` are `@numba.jit(nopython=True)`-decorated, so each is
called both compiled and via `.py_func(...)` (same pattern as
`gaussoptfuncs.py`/`localise.py`/`render.py` elsewhere in this codebase)
since coverage.py cannot see inside JIT-compiled machine code.
"""
from __future__ import annotations

import numpy as np
import pytest

import pyS3M.lib as lib


def _make_locs(xc, yc):
    return np.rec.fromarrays(
        [np.asarray(xc, dtype=np.float64), np.asarray(yc, dtype=np.float64)],
        names="xc,yc",
    )


class TestAppendToRec:
    def test_appends_new_column(self):
        rec = _make_locs([1.0, 2.0], [3.0, 4.0])
        data = np.array([10.0, 20.0])
        out = lib.append_to_rec(rec, data, "photons")
        np.testing.assert_allclose(out.photons, [10.0, 20.0])
        np.testing.assert_allclose(out.xc, [1.0, 2.0])

    def test_overwrites_existing_column(self):
        rec = _make_locs([1.0, 2.0], [3.0, 4.0])
        rec = lib.append_to_rec(rec, np.array([100.0, 200.0]), "photons")
        # Re-appending the same field name triggers remove-then-append.
        out = lib.append_to_rec(rec, np.array([5.0, 6.0]), "photons")
        np.testing.assert_allclose(out.photons, [5.0, 6.0])
        assert list(out.dtype.names).count("photons") == 1


class TestRemoveFromRec:
    def test_removes_column(self):
        rec = _make_locs([1.0, 2.0], [3.0, 4.0])
        rec = lib.append_to_rec(rec, np.array([10.0, 20.0]), "photons")
        out = lib.remove_from_rec(rec, "photons")
        assert "photons" not in out.dtype.names
        assert "xc" in out.dtype.names


class TestIsLocAt:
    def test_marks_points_within_radius(self):
        locs = _make_locs([0.0, 5.0, 0.5], [0.0, 5.0, 0.5])
        mask = lib.is_loc_at(0.0, 0.0, locs, r=1.0)
        np.testing.assert_array_equal(mask, [True, False, True])


class TestLocsAt:
    def test_filters_locs_within_radius(self):
        locs = _make_locs([0.0, 5.0, 0.5], [0.0, 5.0, 0.5])
        picked = lib.locs_at(0.0, 0.0, locs, r=1.0)
        assert len(picked) == 2
        np.testing.assert_allclose(sorted(picked.xc), [0.0, 0.5])


class TestCheckIfInPolygon:
    # A unit square polygon: (0,0), (10,0), (10,10), (0,10).
    X = np.array([0.0, 10.0, 10.0, 0.0])
    Y = np.array([0.0, 0.0, 10.0, 10.0])

    def test_compiled_call(self):
        x = np.array([5.0, 50.0])
        y = np.array([5.0, 50.0])
        result = lib.check_if_in_polygon(x, y, self.X, self.Y)
        np.testing.assert_array_equal(result, [True, False])

    def test_py_func_call(self):
        x = np.array([5.0, 50.0])
        y = np.array([5.0, 50.0])
        result = lib.check_if_in_polygon.py_func(x, y, self.X, self.Y)
        np.testing.assert_array_equal(result, [True, False])


class TestLocsInPolygon:
    def test_filters_locs_in_polygon(self):
        locs = _make_locs([5.0, 50.0], [5.0, 50.0])
        X = [0.0, 10.0, 10.0, 0.0]
        Y = [0.0, 0.0, 10.0, 10.0]
        picked = lib.locs_in_polygon(locs, X, Y)
        assert len(picked) == 1
        np.testing.assert_allclose(picked.xc, [5.0])


class TestCheckIfInRectangle:
    X = np.array([0.0, 10.0, 10.0, 0.0])
    Y = np.array([0.0, 0.0, 10.0, 10.0])

    def test_compiled_call(self):
        x = np.array([5.0, 50.0])
        y = np.array([5.0, 50.0])
        result = lib.check_if_in_rectangle(x, y, self.X, self.Y)
        np.testing.assert_array_equal(result, [True, False])

    def test_py_func_call(self):
        x = np.array([5.0, 50.0])
        y = np.array([5.0, 50.0])
        result = lib.check_if_in_rectangle.py_func(x, y, self.X, self.Y)
        np.testing.assert_array_equal(result, [True, False])


class TestLocsInRectangle:
    def test_filters_locs_in_rectangle(self):
        locs = _make_locs([5.0, 50.0], [5.0, 50.0])
        X = [0.0, 10.0, 10.0, 0.0]
        Y = [0.0, 0.0, 10.0, 10.0]
        picked = lib.locs_in_rectangle(locs, X, Y)
        assert len(picked) == 1
        np.testing.assert_allclose(picked.xc, [5.0])


class TestGetPickPolygonCorners:
    def test_valid_closed_polygon(self):
        pick = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)]
        X, Y = lib.get_pick_polygon_corners(pick)
        assert X == [0.0, 10.0, 10.0, 0.0]
        assert Y == [0.0, 0.0, 10.0, 0.0]

    def test_too_few_points_returns_none(self):
        pick = [(0.0, 0.0), (10.0, 0.0)]
        X, Y = lib.get_pick_polygon_corners(pick)
        assert X is None
        assert Y is None

    def test_not_closed_returns_none(self):
        pick = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        X, Y = lib.get_pick_polygon_corners(pick)
        assert X is None
        assert Y is None


class TestGetPickRectangleCorners:
    def test_diagonal_line(self):
        X, Y = lib.get_pick_rectangle_corners(0.0, 0.0, 10.0, 10.0, width=2.0)
        assert len(X) == 4
        assert len(Y) == 4

    def test_vertical_line_uses_pi_over_2(self):
        # end_x == start_x triggers the alpha = pi/2 branch directly (no
        # arctan division-by-zero).
        X, Y = lib.get_pick_rectangle_corners(5.0, 0.0, 5.0, 10.0, width=2.0)
        assert len(X) == 4
        assert len(Y) == 4
        # For a vertical line, corners should be offset purely in x.
        np.testing.assert_allclose(X, [4.0, 6.0, 6.0, 4.0])
