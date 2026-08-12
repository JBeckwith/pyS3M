"""Full coverage tests for pyS3M.PSFFunctions -- diffraction-limit/PSF-width
formulas, spatial PSF rendering (full and crop-accelerated), and the photon ->
photoelectron -> noisy-image simulation chain.

Small hand-built arrays throughout (this is pure numeric/PSF code, no I/O or
database dependency). Every `@jit(nopython=True, ...)` staticmethod is called
both compiled and via `.py_func` (same pattern as `gaussoptfuncs.py`/
`localise.py`/`render.py` elsewhere in this codebase) since coverage.py cannot
see inside JIT-compiled machine code.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyS3M.PSFFunctions import PSF_Functions


@pytest.fixture
def psf():
    return PSF_Functions()


# ======================================================================
# diffraction_limit / sigma_PSF
# ======================================================================

class TestDiffractionLimit:
    def test_jit_and_py_func_agree(self, psf):
        jit_val = psf.diffraction_limit(0.52, 1.49)
        py_val = psf.diffraction_limit.py_func(0.52, 1.49)
        assert jit_val == pytest.approx(py_val)
        assert jit_val == pytest.approx(0.52 / (2.0 * 1.49))


class TestSigmaPSF:
    def test_jit_and_py_func_agree(self, psf):
        jit_val = psf.sigma_PSF(0.52, 1.49)
        py_val = psf.sigma_PSF.py_func(0.52, 1.49)
        assert jit_val == pytest.approx(py_val)
        assert jit_val > 0


# ======================================================================
# gen_photons_hitting_detector
# ======================================================================

class TestGenPhotonsHittingDetector:
    def test_scalar_background(self, psf):
        pdf = np.full((6, 6), 5.0)
        out = psf.gen_photons_hitting_detector(pdf, background=2.0)
        assert out.shape == (6, 6)
        assert np.all(out >= 0)

    def test_array_background_py_func(self, psf):
        pdf = np.full((6, 6), 5.0)
        bg = np.full((6, 6), 1.0)
        out = psf.gen_photons_hitting_detector.py_func(pdf, background=bg)
        assert out.shape == (6, 6)

    def test_negative_pdf_clipped(self, psf):
        pdf = np.full((4, 4), -5.0)
        out = psf.gen_photons_hitting_detector(pdf, background=0)
        np.testing.assert_allclose(out, 0.0)


# ======================================================================
# gen_spatial_PSF
# ======================================================================

class TestGenSpatialPSF:
    def test_normal_multi_spot(self, psf):
        x = np.arange(16, dtype=np.float64)
        y = np.arange(16, dtype=np.float64)
        x0 = np.array([4.0, 10.0])
        y0 = np.array([4.0, 10.0])
        n_photons = np.array([500.0, 800.0])
        relative_QE = np.ones((16, 16), dtype=np.float32)
        out = psf.gen_spatial_PSF(x, y, 1.3, 1.3, x0, y0, n_photons, relative_QE)
        assert out.shape == (16, 16)
        assert out.sum() > 0

    def test_spot_far_outside_grid_underflows_to_zero_total(self, psf):
        # A spot placed far enough outside the grid that the Gaussian
        # underflows to exact 0.0 everywhere exercises the total<=0 fallback
        # (temp.fill(0)) instead of the normal normalisation branch.
        x = np.arange(16, dtype=np.float64)
        y = np.arange(16, dtype=np.float64)
        x0 = np.array([1e6])
        y0 = np.array([1e6])
        n_photons = np.array([500.0])
        relative_QE = np.ones((16, 16), dtype=np.float32)
        out = psf.gen_spatial_PSF(x, y, 1.3, 1.3, x0, y0, n_photons, relative_QE)
        np.testing.assert_allclose(out, 0.0)


# ======================================================================
# gen_spatial_PSF_fast
# ======================================================================

class TestGenSpatialPSFFast:
    def test_default_crop_radius(self, psf):
        x = np.arange(32, dtype=np.float64)
        y = np.arange(32, dtype=np.float64)
        x0 = np.array([16.0])
        y0 = np.array([16.0])
        n_photons = np.array([500.0])
        relative_QE = np.ones((32, 32), dtype=np.float32)
        out = psf.gen_spatial_PSF_fast(x, y, 1.3, 1.3, x0, y0, n_photons, relative_QE)
        assert out.shape == (32, 32)
        assert out.sum() > 0

    def test_explicit_crop_radius(self, psf):
        x = np.arange(32, dtype=np.float64)
        y = np.arange(32, dtype=np.float64)
        x0 = np.array([16.0])
        y0 = np.array([16.0])
        n_photons = np.array([500.0])
        relative_QE = np.ones((32, 32), dtype=np.float32)
        out = psf.gen_spatial_PSF_fast(
            x, y, 1.3, 1.3, x0, y0, n_photons, relative_QE, crop_radius=4,
        )
        assert out.shape == (32, 32)
        assert out.sum() > 0

    def test_out_of_bounds_spot_is_skipped(self, psf):
        # A spot whose crop window falls entirely outside the image (x1>=x2)
        # exercises the `continue` branch, leaving the image untouched.
        x = np.arange(16, dtype=np.float64)
        y = np.arange(16, dtype=np.float64)
        x0 = np.array([1000.0])
        y0 = np.array([1000.0])
        n_photons = np.array([500.0])
        relative_QE = np.ones((16, 16), dtype=np.float32)
        out = psf.gen_spatial_PSF_fast(
            x, y, 1.3, 1.3, x0, y0, n_photons, relative_QE, crop_radius=4,
        )
        np.testing.assert_allclose(out, 0.0)

    def test_matches_full_version_for_interior_spot(self, psf):
        x = np.arange(24, dtype=np.float64)
        y = np.arange(24, dtype=np.float64)
        x0 = np.array([12.0])
        y0 = np.array([12.0])
        n_photons = np.array([1000.0])
        relative_QE = np.ones((24, 24), dtype=np.float32)
        full = psf.gen_spatial_PSF(x, y, 1.2, 1.2, x0, y0, n_photons, relative_QE)
        fast = psf.gen_spatial_PSF_fast(
            x, y, 1.2, 1.2, x0, y0, n_photons, relative_QE, crop_radius=12,
        )
        np.testing.assert_allclose(full, fast, atol=1e-3)


# ======================================================================
# gen_photoelectrons
# ======================================================================

class TestGenPhotoelectrons:
    def test_normal(self, psf):
        n_photons = np.full((6, 6), 100, dtype=np.int32)
        out = psf.gen_photoelectrons(n_photons, abs_QE=0.9)
        assert out.shape == (6, 6)
        assert np.all(out <= 100)

    def test_negative_photons_clipped(self, psf):
        n_photons = np.full((4, 4), -5, dtype=np.int32)
        out = psf.gen_photoelectrons(n_photons, abs_QE=0.9)
        np.testing.assert_allclose(out, 0)

    def test_nan_qe_treated_as_zero(self, psf):
        n_photons = np.full((4, 4), 100, dtype=np.int32)
        abs_QE = np.full((4, 4), np.nan)
        out = psf.gen_photoelectrons(n_photons, abs_QE=abs_QE)
        np.testing.assert_allclose(out, 0)

    def test_inf_qe_treated_as_one(self, psf):
        n_photons = np.full((4, 4), 100, dtype=np.int32)
        abs_QE = np.full((4, 4), np.inf)
        out = psf.gen_photoelectrons(n_photons, abs_QE=abs_QE)
        np.testing.assert_allclose(out, 100)

    def test_qe_clipped_to_valid_range(self, psf):
        n_photons = np.full((4, 4), 100, dtype=np.int32)
        abs_QE = np.array([[-1.0, 2.0, 0.5, 0.5]] * 4)
        out = psf.gen_photoelectrons(n_photons, abs_QE=abs_QE)
        np.testing.assert_allclose(out[:, 0], 0)
        np.testing.assert_allclose(out[:, 1], 100)


# ======================================================================
# gen_photoelectrons_vectorized_frames
# ======================================================================

class TestGenPhotoelectronsVectorizedFrames:
    def test_uniform_qe_fast_path(self, psf):
        n_frames, w, h, n_dyes, n_channels = 3, 5, 5, 1, 3
        n_photons_all = np.full((n_frames, w, h, n_dyes), 50, dtype=np.int32)
        QE_per_channel_all = np.full((n_frames, n_dyes, n_channels), 0.8)
        mask_stack = np.ones((w, h, n_channels), dtype=bool)
        out = psf.gen_photoelectrons_vectorized_frames(
            n_photons_all, QE_per_channel_all, mask_stack,
        )
        assert out.shape == (n_frames, w, h, n_dyes)
        assert np.all(out <= 50)

    def test_bayer_qe_static_mask(self, psf):
        n_frames, w, h, n_dyes, n_channels = 2, 4, 4, 1, 3
        n_photons_all = np.full((n_frames, w, h, n_dyes), 30, dtype=np.int32)
        QE_per_channel_all = np.zeros((n_frames, n_dyes, n_channels))
        QE_per_channel_all[:, 0, :] = [0.1, 0.6, 0.3]
        idx = np.arange(w * h).reshape(w, h) % n_channels
        mask_stack = np.zeros((w, h, n_channels), dtype=bool)
        for c in range(n_channels):
            mask_stack[:, :, c] = idx == c
        out = psf.gen_photoelectrons_vectorized_frames(
            n_photons_all, QE_per_channel_all, mask_stack,
        )
        assert out.shape == (n_frames, w, h, n_dyes)

    def test_bayer_qe_per_frame_mask(self, psf):
        # mask_stack.ndim == 4 -> per-frame Bayer pattern branch.
        n_frames, w, h, n_dyes, n_channels = 2, 4, 4, 1, 3
        n_photons_all = np.full((n_frames, w, h, n_dyes), 30, dtype=np.int32)
        QE_per_channel_all = np.zeros((n_frames, n_dyes, n_channels))
        QE_per_channel_all[:, 0, :] = [0.1, 0.6, 0.3]
        idx = np.arange(w * h).reshape(w, h) % n_channels
        base_mask = np.zeros((w, h, n_channels), dtype=bool)
        for c in range(n_channels):
            base_mask[:, :, c] = idx == c
        mask_stack = np.broadcast_to(base_mask, (n_frames, w, h, n_channels)).copy()
        out = psf.gen_photoelectrons_vectorized_frames(
            n_photons_all, QE_per_channel_all, mask_stack,
        )
        assert out.shape == (n_frames, w, h, n_dyes)

    def test_multiple_dyes(self, psf):
        n_frames, w, h, n_dyes, n_channels = 2, 4, 4, 2, 3
        n_photons_all = np.full((n_frames, w, h, n_dyes), 20, dtype=np.int32)
        QE_per_channel_all = np.full((n_frames, n_dyes, n_channels), 0.5)
        mask_stack = np.ones((w, h, n_channels), dtype=bool)
        out = psf.gen_photoelectrons_vectorized_frames(
            n_photons_all, QE_per_channel_all, mask_stack,
        )
        assert out.shape == (n_frames, w, h, n_dyes)


# ======================================================================
# photoelectrons_to_image
# ======================================================================

class TestPhotoelectronsToImage:
    def test_jit_and_py_func_agree_in_shape(self, psf):
        n_photoelectrons = np.full((6, 6), 50.0)
        gain = np.full((6, 6), 2.0)
        offset = np.full((6, 6), 100.0)
        variance = np.full((6, 6), 4.0)
        jit_out = psf.photoelectrons_to_image(n_photoelectrons, gain, offset, variance)
        py_out = psf.photoelectrons_to_image.py_func(n_photoelectrons, gain, offset, variance)
        assert jit_out.shape == py_out.shape == (6, 6)
        assert np.all(jit_out >= 0)
        assert np.all(py_out >= 0)


# ======================================================================
# generate_noisy_image_matrix
# ======================================================================

class TestGenerateNoisyImageMatrix:
    def test_jit_and_py_func_agree_in_shape(self, psf):
        jit_out = psf.generate_noisy_image_matrix((8, 8), 5.0, 100.0, 10.0)
        py_out = psf.generate_noisy_image_matrix.py_func((8, 8), 5.0, 100.0, 10.0)
        assert jit_out.shape == py_out.shape == (8, 8)
