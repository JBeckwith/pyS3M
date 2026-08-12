#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.StepDetector -- Gaussian/Poisson likelihood-
ratio change-point detection (Watkins & Yang 2005 / Jiang's findcp.m).

Checked usage first (same workflow as the rest of this session's coverage
push): zero callers anywhere in src/, unit_tests/, or main-branch notebooks,
but one real, working call site on the developer branch
(notebooks/figures/SI/SI_Single_Dye_Photobleaching.ipynb):
`StepDetector(win_size=CP_WIN_SIZE, alpha=CP_ALPHA, estimator='gaussian')`.
Nothing here is dead -- a small, self-contained algorithmic module that
simply never had a dedicated test file. No deletions.

Uses deliberately tiny hand-built arrays (tens of points, not real
acquisitions) -- these are unit tests for branch coverage, not statistical
benchmarks.
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

import pyS3M.StepDetector as SD
from pyS3M.StepDetector import (
    StepDetector,
    _gaussian_threshold,
    _lr_test,
    _lr_test_gaussian,
    _PoissonCost,
    _segment,
    _vost_threshold,
)


# ======================================================================
# _vost_threshold
# ======================================================================

class TestVostThreshold:
    def test_n_le_2_returns_inf(self):
        assert _vost_threshold(2) == np.inf
        assert _vost_threshold(1) == np.inf

    def test_normal_case_finite(self):
        v = _vost_threshold(20, alpha=0.05, d=1)
        assert np.isfinite(v)
        assert v > 0

    def test_brentq_value_error_returns_inf(self):
        # No sign change in thresh_func over [1, 20] for this (n, alpha, d)
        # combination -- brentq raises ValueError, caught and mapped to inf.
        assert _vost_threshold(3, alpha=0.5, d=5) == np.inf


# ======================================================================
# _lr_test (Poisson LR)
# ======================================================================

class TestLrTest:
    def test_n_lt_2_returns_zero(self):
        assert _lr_test(np.array([5.0])) == (0, 0.0)
        assert _lr_test(np.array([])) == (0, 0.0)

    def test_ss_le_zero_returns_zero(self):
        assert _lr_test(np.zeros(5)) == (0, 0.0)

    def test_partial_invalid_mask(self):
        # Negative prefix makes the first candidate split invalid (sk <= 0)
        # while the overall sum stays positive.
        data = np.array([-5.0, 10.0, 10.0, 10.0])
        best, lm = _lr_test(data)
        assert lm > 0.0
        assert 0 <= best < len(data) - 1

    def test_clear_step_detected(self):
        data = np.array([10.0, 10.0, 10.0, 10.0, 50.0, 50.0, 50.0, 50.0])
        best, lm = _lr_test(data)
        assert lm > 0.0
        # Best split should land at/near the true break (index 3).
        assert 2 <= best <= 4


# ======================================================================
# _lr_test_gaussian
# ======================================================================

class TestLrTestGaussian:
    def test_n_lt_4_returns_zero(self):
        assert _lr_test_gaussian(np.array([1.0, 2.0, 3.0])) == (0, 0.0)

    def test_zero_variance_returns_zero(self):
        assert _lr_test_gaussian(np.full(6, 5.0)) == (0, 0.0)

    def test_partial_invalid_mask(self):
        # Constant run at the start gives lvar == 0 for the smallest k.
        data = np.array([5.0, 5.0, 5.0, 5.0, 1.0, 9.0, 2.0, 8.0])
        best, lm = _lr_test_gaussian(data)
        assert lm >= 0.0

    def test_clear_step_detected(self):
        rng = np.random.default_rng(0)
        data = np.concatenate([
            rng.normal(5.0, 0.5, 10), rng.normal(20.0, 0.5, 10),
        ])
        best, lm = _lr_test_gaussian(data)
        assert lm > 0.0
        assert 7 <= best <= 12


# ======================================================================
# _gaussian_threshold
# ======================================================================

class TestGaussianThreshold:
    def test_n_lt_4_returns_inf(self):
        assert _gaussian_threshold(3) == np.inf

    def test_normal_case_finite(self):
        v = _gaussian_threshold(20, alpha=0.05)
        assert np.isfinite(v)


# ======================================================================
# _segment (recursive binary segmentation, exercised directly)
# ======================================================================

class TestSegment:
    def test_too_short_no_split(self):
        cp_set = set()
        signal = np.array([1.0, 2.0, 3.0])
        _segment(signal, 0, 3, cp_set, win_size=5,
                 threshold_fn=lambda n: 0.0, lr_fn=_lr_test)
        assert cp_set == set()

    def test_split_accepted_recurses(self):
        signal = np.array([10.0] * 10 + [50.0] * 10)
        cp_set = set()
        _segment(signal, 0, 20, cp_set, win_size=5,
                 threshold_fn=lambda n: 0.0, lr_fn=_lr_test)
        assert len(cp_set) >= 1

    def test_split_rejected_below_threshold(self):
        rng = np.random.default_rng(1)
        signal = rng.normal(10.0, 0.1, 20)
        cp_set = set()
        _segment(signal, 0, 20, cp_set, win_size=5,
                 threshold_fn=lambda n: np.inf, lr_fn=_lr_test)
        assert cp_set == set()

    def test_split_rejected_too_close_to_edge(self):
        # Minimal segment length (n == 2*win_size); force the best LR split
        # (via a fake lr_fn) to land one sample from the right edge, so the
        # win_size margin check on the right side rejects it.
        win_size = 5
        signal = np.zeros(2 * win_size)

        def fake_lr(data):
            return len(data) - 2, 100.0  # local_cp near the very end

        cp_set = set()
        _segment(signal, 0, 2 * win_size, cp_set, win_size=win_size,
                 threshold_fn=lambda n: 0.0, lr_fn=fake_lr)
        assert cp_set == set()


# ======================================================================
# _PoissonCost (ruptures optional cost function)
# ======================================================================

class TestPoissonCost:
    def test_fit_and_error_normal(self):
        cost = _PoissonCost()
        cost.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        e = cost.error(0, 3)
        assert e < 0.0  # -SS*log(SS/n), positive signal -> negative cost

    def test_error_ss_le_zero_returns_zero(self):
        cost = _PoissonCost()
        cost.fit(np.zeros(5))
        assert cost.error(0, 3) == 0.0

    def test_error_not_enough_points_raises(self):
        from ruptures.costs import NotEnoughPoints
        cost = _PoissonCost()
        cost.fit(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(NotEnoughPoints):
            cost.error(0, 1)


# ======================================================================
# StepDetector class
# ======================================================================

class TestStepDetectorInit:
    def test_defaults(self):
        sd = StepDetector()
        assert sd.win_size == 10
        assert sd.alpha == 0.05
        assert sd.d == 1
        assert sd.backend == "binseg"
        assert sd.estimator == "poisson"

    def test_estimator_lowercased(self):
        sd = StepDetector(estimator="Gaussian")
        assert sd.estimator == "gaussian"


class TestStepDetectorThreshold:
    def test_poisson_threshold_dispatch(self):
        sd = StepDetector(estimator="poisson")
        v = sd._threshold(20)
        assert np.isfinite(v)

    def test_gaussian_threshold_dispatch(self):
        sd = StepDetector(estimator="gaussian")
        v = sd._threshold(20)
        assert np.isfinite(v)

    def test_threshold_is_cached(self):
        sd = StepDetector()
        v1 = sd._threshold(20)
        assert 20 in sd._cache
        v2 = sd._threshold(20)  # cache-hit branch
        assert v1 == v2


class TestStepDetectorLrFn:
    def test_lr_fn_poisson(self):
        sd = StepDetector(estimator="poisson")
        assert sd._lr_fn() is _lr_test

    def test_lr_fn_gaussian(self):
        sd = StepDetector(estimator="gaussian")
        assert sd._lr_fn() is _lr_test_gaussian


class TestStepDetectorDetect:
    def test_binseg_default_backend(self):
        signal = [10.0] * 15 + [50.0] * 15
        sd = StepDetector(win_size=5, alpha=0.05, estimator="poisson")
        cps = sd.detect(signal)
        assert cps[-1] == len(signal)
        assert len(cps) >= 2  # at least one real split plus the terminal

    def test_binseg_gaussian_estimator(self):
        rng = np.random.default_rng(2)
        signal = np.concatenate([
            rng.normal(5.0, 0.3, 15), rng.normal(25.0, 0.3, 15),
        ])
        sd = StepDetector(win_size=5, alpha=0.05, estimator="gaussian")
        cps = sd.detect(signal)
        assert cps[-1] == len(signal)

    def test_no_step_flat_signal_only_terminal_cp(self):
        rng = np.random.default_rng(3)
        signal = rng.normal(10.0, 0.05, 20)
        sd = StepDetector(win_size=5, alpha=0.001)
        cps = sd.detect(signal)
        assert cps == [len(signal)]

    def test_too_short_signal_only_terminal_cp(self):
        sd = StepDetector(win_size=10)
        cps = sd.detect([1.0, 2.0, 3.0])
        assert cps == [3]

    def test_pelt_backend(self):
        signal = [10.0] * 15 + [50.0] * 15
        sd = StepDetector(win_size=5, alpha=0.05, backend="pelt")
        cps = sd.detect(signal)
        assert cps[-1] == len(signal)

    def test_pelt_backend_without_ruptures_raises(self, monkeypatch):
        monkeypatch.setattr(SD, "_RUPTURES_AVAILABLE", False)
        sd = StepDetector(backend="pelt")
        with pytest.raises(ImportError, match="ruptures is required"):
            sd.detect([1.0, 2.0, 3.0, 4.0])


class TestStepDetectorSegmentMeans:
    def test_segment_means(self):
        signal = [1.0, 1.0, 1.0, 5.0, 5.0]
        sd = StepDetector()
        means = sd.segment_means(signal, cps=[3, 5])
        np.testing.assert_allclose(means, [1.0, 1.0, 1.0, 5.0, 5.0])


# ======================================================================
# Module-level ruptures-unavailable fallback (top-of-file try/except)
# ======================================================================

class TestRuptuesImportFallback:
    def test_module_reload_without_ruptures(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "ruptures.base", None)
        monkeypatch.setitem(sys.modules, "ruptures.costs", None)
        try:
            reloaded = importlib.reload(SD)
            assert reloaded._RUPTURES_AVAILABLE is False
            # Note: reload() re-executes the module body in-place but does
            # not clear its namespace first, so a `_PoissonCost` name
            # defined by an earlier successful import is still present here
            # -- only _RUPTURES_AVAILABLE (and, on a truly fresh interpreter
            # import, the class definition itself) reflects the fallback.
        finally:
            # sys.modules must be restored *before* re-importing for real --
            # monkeypatch only undoes it after the test function returns.
            monkeypatch.undo()
            importlib.reload(SD)
            assert SD._RUPTURES_AVAILABLE is True
