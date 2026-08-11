#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.drift_correction.auto — AutoDriftCorrector,
DriftCorrectionFactory, and the undrift_aim/undrift_auto convenience functions.

Part of the drift_correction/ package coverage push (claude/TODO.md PRIORITY 1).
AutoDriftCorrector.calculate_drift's body is already exercised indirectly by
unit_tests/test_drift_correction*.py's method="auto" tests -- this file fills
the remaining gaps: supports_3d(), the factory's error branch, and the two
backward-compatible module-level functions.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyS3M.drift_correction._base import DriftCorrectionError, DriftMethod
from pyS3M.drift_correction.auto import (
    AutoDriftCorrector,
    DriftCorrectionFactory,
    undrift_aim,
    undrift_auto,
)


def _small_locs_and_info(n_locs=300, n_frames=20, width=128, height=128):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, n_frames, n_locs)
    x = rng.uniform(0, width, n_locs)
    y = rng.uniform(0, height, n_locs)
    locs = np.rec.fromarrays([x, y, frames], names=["xc", "yc", "frame"])
    info = [{"Width": width, "Height": height, "Frames": n_frames, "Pixelsize": 69}]
    return locs, info


class TestAutoDriftCorrector:
    def test_supports_3d_is_true(self):
        assert AutoDriftCorrector().supports_3d() is True


class TestDriftCorrectionFactory:
    def test_create_corrector_unsupported_method_raises(self):
        with pytest.raises(DriftCorrectionError, match="Unsupported drift method"):
            DriftCorrectionFactory.create_corrector("not_a_method")

    def test_available_methods_lists_all_three(self):
        methods = DriftCorrectionFactory.available_methods()
        assert set(methods) == {DriftMethod.AIM, DriftMethod.FIDUCIAL, DriftMethod.AUTO}


class TestConvenienceFunctions:
    def test_undrift_aim(self):
        locs, info = _small_locs_and_info()
        corrected, result = undrift_aim(locs, info, segmentation=5, intersect_d=1.0, roi_r=2.0)
        assert len(corrected) == len(locs)
        assert result.method_used == DriftMethod.AIM

    def test_undrift_auto(self):
        locs, info = _small_locs_and_info()
        corrected, result = undrift_auto(locs, info, segmentation=5, intersect_d=1.0, roi_r=2.0)
        assert len(corrected) == len(locs)
        assert result.method_used == DriftMethod.AIM
        assert "auto_selection_reason" in result.metadata
