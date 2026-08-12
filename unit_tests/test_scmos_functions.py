"""Full coverage tests for pyS3M.sCMOSFunctions -- variance-aware demosaicing,
plain Bayer demosaicing (single-frame and parallel multi-frame), Gaussian/
variance-weighted smoothing filters, and the standalone multiprocessing worker.

Small synthetic Bayer images throughout ('bilinear' strategy, the fastest, is
used except where `test_demosaic_strategies.py` already exercises all four
strategies). Real `ProcessPoolExecutor` for the multi-frame parallel path,
matching the pattern used elsewhere in this coverage push (e.g.
`test_spot_detection_functions.py`).
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

import pyS3M.sCMOSFunctions as sCMOSFunctions

sCMOS_Functions = sCMOSFunctions.sCMOS_Functions


def _bayer_image(h=16, w=16, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w)) * 50 + 100).astype(np.float32)


# ======================================================================
# module import fallback (ProgressUtils unavailable)
# ======================================================================

class TestProgressUtilsImportFallback:
    def test_import_error_sets_progressutils_none(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyS3M.ProgressUtils", None)
        try:
            importlib.reload(sCMOSFunctions)
            assert sCMOSFunctions.ProgressUtils is None
        finally:
            monkeypatch.undo()
            importlib.reload(sCMOSFunctions)


# ======================================================================
# variance_aware_demosaic
# ======================================================================

class TestVarianceAwareDemosaic:
    def test_invalid_strategy_raises(self):
        scmos = sCMOS_Functions()
        image = _bayer_image()
        with pytest.raises(ValueError, match="Unknown demosaicing strategy"):
            scmos.variance_aware_demosaic(image, np.ones_like(image), strategy="bogus")

    def test_variance_map_ndim_gt_2_is_squeezed(self):
        scmos = sCMOS_Functions()
        image = _bayer_image()
        variance_map = np.ones((1,) + image.shape, dtype=np.float32)
        result = scmos.variance_aware_demosaic(image, variance_map, grayscale=True)
        assert result.shape == image.shape

    def test_variance_map_transposed_shape_is_fixed(self):
        scmos = sCMOS_Functions()
        image = _bayer_image(h=12, w=20)
        variance_map = np.ones((20, 12), dtype=np.float32)  # transposed shape
        result = scmos.variance_aware_demosaic(image, variance_map, grayscale=True)
        assert result.shape == image.shape

    def test_variance_map_incompatible_shape_raises(self):
        scmos = sCMOS_Functions()
        image = _bayer_image(h=12, w=20)
        variance_map = np.ones((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="variance_map shape"):
            scmos.variance_aware_demosaic(image, variance_map)

    def test_gain_array_matching_shape(self):
        scmos = sCMOS_Functions()
        image = _bayer_image()
        gain = np.full(image.shape, 2.0, dtype=np.float32)
        result = scmos.variance_aware_demosaic(image, np.ones_like(image), gain=gain, grayscale=True)
        assert result.shape == image.shape

    def test_gain_array_ndim_gt_2_is_squeezed(self):
        scmos = sCMOS_Functions()
        image = _bayer_image()
        gain = np.full((1,) + image.shape, 2.0, dtype=np.float32)
        result = scmos.variance_aware_demosaic(image, np.ones_like(image), gain=gain, grayscale=True)
        assert result.shape == image.shape

    def test_gain_array_transposed_shape_is_fixed(self):
        scmos = sCMOS_Functions()
        image = _bayer_image(h=12, w=20)
        gain = np.full((20, 12), 2.0, dtype=np.float32)
        result = scmos.variance_aware_demosaic(image, np.ones_like(image), gain=gain, grayscale=True)
        assert result.shape == image.shape

    def test_gain_array_incompatible_shape_raises(self):
        scmos = sCMOS_Functions()
        image = _bayer_image(h=12, w=20)
        gain = np.full((5, 5), 2.0, dtype=np.float32)
        with pytest.raises(ValueError, match="gain shape"):
            scmos.variance_aware_demosaic(image, np.ones_like(image), gain=gain)

    def test_offset_map_ndim_gt_2_is_squeezed(self):
        scmos = sCMOS_Functions()
        image = _bayer_image()
        offset_map = np.full((1,) + image.shape, 10.0, dtype=np.float32)
        result = scmos.variance_aware_demosaic(image, np.ones_like(image), offset_map=offset_map, grayscale=True)
        assert result.shape == image.shape

    def test_offset_map_transposed_shape_is_fixed(self):
        scmos = sCMOS_Functions()
        image = _bayer_image(h=12, w=20)
        offset_map = np.full((20, 12), 10.0, dtype=np.float32)
        result = scmos.variance_aware_demosaic(image, np.ones_like(image), offset_map=offset_map, grayscale=True)
        assert result.shape == image.shape

    def test_offset_map_incompatible_shape_raises(self):
        scmos = sCMOS_Functions()
        image = _bayer_image(h=12, w=20)
        offset_map = np.full((5, 5), 10.0, dtype=np.float32)
        with pytest.raises(ValueError, match="offset_map shape"):
            scmos.variance_aware_demosaic(image, np.ones_like(image), offset_map=offset_map)

    def test_3d_cfa_stack_with_offset_and_scalar_gain(self):
        scmos = sCMOS_Functions()
        frame = _bayer_image()
        stack = np.stack([frame, frame, frame], axis=0)
        offset_map = np.full(frame.shape, 10.0, dtype=np.float32)
        result = scmos.variance_aware_demosaic(
            stack, np.ones_like(frame), offset_map=offset_map, gain=2.0, grayscale=True
        )
        assert result.shape == stack.shape

    def test_3d_cfa_stack_with_array_gain(self):
        scmos = sCMOS_Functions()
        frame = _bayer_image()
        stack = np.stack([frame, frame], axis=0)
        gain = np.full(frame.shape, 2.0, dtype=np.float32)
        result = scmos.variance_aware_demosaic(stack, np.ones_like(frame), gain=gain, grayscale=True)
        assert result.shape == stack.shape

    def test_grayscale_false_returns_rgb_result(self):
        scmos = sCMOS_Functions()
        image = _bayer_image()
        result = scmos.variance_aware_demosaic(image, np.ones_like(image), grayscale=False)
        assert result.shape == image.shape + (3,)


# ======================================================================
# bayer_demosaic_stack_grayscale
# ======================================================================

class TestBayerDemosaicStackGrayscale:
    def test_invalid_strategy_raises(self):
        scmos = sCMOS_Functions()
        image = _bayer_image()
        with pytest.raises(ValueError, match="Unknown demosaicing strategy"):
            scmos.bayer_demosaic_stack_grayscale(image, strategy="bogus")

    def test_single_frame_2d(self):
        scmos = sCMOS_Functions()
        image = _bayer_image()
        gray = scmos.bayer_demosaic_stack_grayscale(image)
        assert gray.shape == image.shape

    def test_multiframe_parallel_with_progressbar(self):
        scmos = sCMOS_Functions()
        frame = _bayer_image()
        stack = np.stack([frame, frame, frame], axis=0)
        gray = scmos.bayer_demosaic_stack_grayscale(stack)
        assert gray.shape == stack.shape

    def test_multiframe_parallel_without_progressbar_fallback(self, monkeypatch):
        monkeypatch.setattr(sCMOSFunctions, "ProgressUtils", None)
        scmos = sCMOS_Functions()
        frame = _bayer_image()
        stack = np.stack([frame, frame], axis=0)
        gray = scmos.bayer_demosaic_stack_grayscale(stack)
        assert gray.shape == stack.shape


# ======================================================================
# bayer_demosaic_stack
# ======================================================================

class TestBayerDemosaicStack:
    def test_invalid_strategy_raises(self):
        scmos = sCMOS_Functions()
        image = _bayer_image()
        with pytest.raises(ValueError, match="Unknown demosaicing strategy"):
            scmos.bayer_demosaic_stack(image, strategy="bogus")

    def test_2d_grayscale_false_returns_none_for_gray(self):
        scmos = sCMOS_Functions()
        image = _bayer_image()
        rgb, gray = scmos.bayer_demosaic_stack(image, grayscale=False)
        assert rgb.shape == image.shape + (3,)
        assert gray is None

    def test_3d_grayscale_true(self):
        scmos = sCMOS_Functions()
        frame = _bayer_image()
        stack = np.stack([frame, frame], axis=0)
        rgb, gray = scmos.bayer_demosaic_stack(stack, grayscale=True)
        assert rgb.shape == stack.shape + (3,)
        assert gray.shape == stack.shape


# ======================================================================
# gaussian_filter_stack
# ======================================================================

class TestGaussianFilterStack:
    def test_2d_image(self):
        scmos = sCMOS_Functions()
        image = _bayer_image()
        out = scmos.gaussian_filter_stack(image, sigma=1.0)
        assert out.shape == image.shape

    def test_3d_stack(self):
        scmos = sCMOS_Functions()
        frame = _bayer_image()
        stack = np.stack([frame, frame, frame], axis=0)
        out = scmos.gaussian_filter_stack(stack, sigma=1.0)
        assert out.shape == stack.shape


# ======================================================================
# var_weighted_uniform_filter
# ======================================================================

class TestVarWeightedUniformFilter:
    def test_2d_image_with_2d_variance(self):
        scmos = sCMOS_Functions()
        image = _bayer_image()
        variance_map = np.full(image.shape, 2.0, dtype=np.float32)
        out = scmos.var_weighted_uniform_filter(image, variance_map, kernel_size=3)
        assert out.shape == image.shape

    def test_3d_stack_with_2d_variance_replicated(self):
        scmos = sCMOS_Functions()
        frame = _bayer_image()
        stack = np.stack([frame, frame, frame], axis=0)
        variance_map = np.full(frame.shape, 2.0, dtype=np.float32)
        out = scmos.var_weighted_uniform_filter(stack, variance_map, kernel_size=3)
        assert out.shape == stack.shape


# ======================================================================
# _demosaic_frames_standalone (module-level, pickleable worker function)
# ======================================================================

class TestDemosaicFramesStandalone:
    def test_invalid_strategy_raises(self):
        frame = _bayer_image()
        chunk = np.stack([frame], axis=0)
        with pytest.raises(ValueError, match="Unknown demosaicing strategy"):
            sCMOSFunctions._demosaic_frames_standalone(chunk, strategy="bogus")

    def test_processes_chunk_of_frames(self):
        frame = _bayer_image()
        chunk = np.stack([frame, frame], axis=0)
        results = sCMOSFunctions._demosaic_frames_standalone(chunk, strategy="bilinear")
        assert results.shape == chunk.shape
