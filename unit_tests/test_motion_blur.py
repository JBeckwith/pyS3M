"""Unit tests for motion blur simulation (_apply_motion_blur in multicolour.py).

Tests cover:
  - No-blur identity: _apply_motion_blur with displacement_px=0 matches direct PSF call
  - Blur reduces peak intensity (energy is conserved but spread out)
  - Blur is direction-dependent: orthogonal directions produce different asymmetric PSFs
  - Blur symmetry: opposite directions (θ, θ+π) produce identical maps (symmetric kernel)
  - Blur magnitude: larger displacement → broader PSF (larger FWHM in blur direction)
  - n_samples convergence: result stabilises quickly with increasing quadrature points
  - SimulationConfig accepts and stores new motion blur fields
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "simulation"))

from multicolour import MultiC_Sim_Funcs_Refactored, SimulationConfig


def _make_gaussian_fn(sigma: float, image_size: int = 64):
    """Return a simple isotropic Gaussian PSF callable compatible with _apply_motion_blur."""
    xx, yy = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing="ij")

    def _fn(x0_pixels, y0_pixels, n_photons_array):
        x0 = float(np.asarray(x0_pixels).flat[0])
        y0 = float(np.asarray(y0_pixels).flat[0])
        n = float(np.asarray(n_photons_array).flat[0])
        g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))
        total = g.sum()
        return (g * n / total).astype(np.float32)

    return _fn


class TestApplyMotionBlur:
    """Tests for MultiC_Sim_Funcs_Refactored._apply_motion_blur."""

    SIZE = 64
    CENTRE = 32.0
    SIGMA = 3.0

    @pytest.fixture
    def fn(self):
        return _make_gaussian_fn(self.SIGMA, self.SIZE)

    @pytest.fixture
    def centre_args(self):
        return (
            np.array([self.CENTRE]),
            np.array([self.CENTRE]),
            np.array([1000]),
        )

    def test_zero_displacement_matches_direct_call(self, fn, centre_args):
        x0, y0, n = centre_args
        direct = fn(x0, y0, n)
        blurred = MultiC_Sim_Funcs_Refactored._apply_motion_blur(fn, x0, y0, n, 0.0, 0.0)
        np.testing.assert_allclose(blurred, direct, rtol=1e-3)

    def test_blur_reduces_peak(self, fn, centre_args):
        x0, y0, n = centre_args
        direct = fn(x0, y0, n)
        blurred = MultiC_Sim_Funcs_Refactored._apply_motion_blur(fn, x0, y0, n, 8.0, 0.0)
        assert blurred.max() < direct.max(), "Motion blur must reduce the peak value"

    def test_energy_conservation(self, fn, centre_args):
        x0, y0, n = centre_args
        direct = fn(x0, y0, n)
        blurred = MultiC_Sim_Funcs_Refactored._apply_motion_blur(fn, x0, y0, n, 8.0, 0.0)
        np.testing.assert_allclose(blurred.sum(), direct.sum(), rtol=5e-3,
                                   err_msg="Total photons must be conserved under motion blur")

    def test_opposite_directions_equal(self, fn, centre_args):
        """Blur along θ and θ+π must give the same result (symmetric kernel)."""
        x0, y0, n = centre_args
        theta = np.pi / 4
        b1 = MultiC_Sim_Funcs_Refactored._apply_motion_blur(fn, x0, y0, n, 5.0, theta)
        b2 = MultiC_Sim_Funcs_Refactored._apply_motion_blur(fn, x0, y0, n, 5.0, theta + np.pi)
        np.testing.assert_allclose(b1, b2, rtol=1e-3,
                                   err_msg="Blur along ±θ should be symmetric")

    def test_orthogonal_blur_directions_differ(self, fn, centre_args):
        """Blur along x-axis and y-axis should produce different (rotated) maps."""
        x0, y0, n = centre_args
        disp = 10.0
        bx = MultiC_Sim_Funcs_Refactored._apply_motion_blur(fn, x0, y0, n, disp, 0.0)
        by = MultiC_Sim_Funcs_Refactored._apply_motion_blur(fn, x0, y0, n, disp, np.pi / 2)
        assert not np.allclose(bx, by), "Perpendicular blur directions should differ"

    def test_larger_displacement_broader_psf(self, fn, centre_args):
        """Bigger displacement → bigger FWHM in the blur direction.

        direction=0 means cos(0)=1, sin(0)=0, so blur moves x0 (axis-0/row index).
        We therefore slice along axis 0 (column at centre) to measure the blurred width.
        """
        x0, y0, n = centre_args
        b_small = MultiC_Sim_Funcs_Refactored._apply_motion_blur(fn, x0, y0, n, 2.0, 0.0)
        b_large = MultiC_Sim_Funcs_Refactored._apply_motion_blur(fn, x0, y0, n, 12.0, 0.0)
        # Profile along axis 0 (the blur direction), at the central column
        col_s = b_small[:, self.SIZE // 2]
        col_l = b_large[:, self.SIZE // 2]
        fwhm_s = np.sum(col_s >= col_s.max() / 2)
        fwhm_l = np.sum(col_l >= col_l.max() / 2)
        assert fwhm_l > fwhm_s, "Larger displacement should produce a wider blur profile"

    def test_n_samples_convergence(self, fn, centre_args):
        """Result should change by <1% when doubling from 25 to 51 samples."""
        x0, y0, n = centre_args
        b25 = MultiC_Sim_Funcs_Refactored._apply_motion_blur(fn, x0, y0, n, 6.0, np.pi / 3,
                                                       n_samples=25)
        b51 = MultiC_Sim_Funcs_Refactored._apply_motion_blur(fn, x0, y0, n, 6.0, np.pi / 3,
                                                       n_samples=51)
        rel_diff = np.abs(b25 - b51).sum() / (np.abs(b51).sum() + 1e-12)
        assert rel_diff < 0.01, f"Quadrature not converged: rel diff = {rel_diff:.4f}"


class TestSimulationConfigMotionBlur:
    """SimulationConfig should expose and default the new motion blur fields."""

    def test_defaults(self):
        cfg = SimulationConfig()
        assert cfg.motion_velocity_nm_per_s == 0.0
        assert cfg.frame_exposure_ms == 100.0

    def test_custom_values(self):
        cfg = SimulationConfig(motion_velocity_nm_per_s=200.0, frame_exposure_ms=50.0)
        assert cfg.motion_velocity_nm_per_s == 200.0
        assert cfg.frame_exposure_ms == 50.0

    def test_displacement_calculation(self):
        """200 nm/s × 50 ms = 10 nm displacement."""
        cfg = SimulationConfig(motion_velocity_nm_per_s=200.0, frame_exposure_ms=50.0)
        displacement_nm = cfg.motion_velocity_nm_per_s * (cfg.frame_exposure_ms / 1000.0)
        assert displacement_nm == pytest.approx(10.0)
