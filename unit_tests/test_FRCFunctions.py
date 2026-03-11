# -*- coding: utf-8 -*-
"""Unit tests for FRCFunctions.py."""

import sys
import os
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from FRCFunctions import (
    _radial_sum,
    _tukey_window,
    _intersect,
    bin_localisations,
    frc,
    frc_to_resolution,
    fire,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# _radial_sum
# ---------------------------------------------------------------------------

def test_radial_sum_total():
    """Sum of all rings equals sum of image."""
    img = RNG.random((64, 64))
    assert np.isclose(_radial_sum(img).sum(), img.sum())


def test_radial_sum_ones_pixel_count():
    """_radial_sum(ones) counts every pixel exactly once."""
    sz = 64
    nr = _radial_sum(np.ones((sz, sz)))
    assert int(nr.sum()) == sz * sz


def test_radial_sum_length():
    """Length equals round(r_max)+1 where r_max is corner distance."""
    sz = 128
    nr = _radial_sum(np.ones((sz, sz)))
    # centre at (64, 64), corner at (0, 0): dist = 64*sqrt(2) ≈ 90.5 → 91
    expected_len = int(round(np.sqrt(2) * (sz // 2))) + 1
    assert len(nr) == expected_len


# ---------------------------------------------------------------------------
# _tukey_window
# ---------------------------------------------------------------------------

def test_tukey_range():
    w = _tukey_window(128, 128)
    assert w.min() >= 0.0
    assert w.max() <= 1.0 + 1e-6


def test_tukey_centre_is_one():
    w = _tukey_window(128, 128)
    assert w[64, 64] == pytest.approx(1.0)


def test_tukey_corners_near_zero():
    w = _tukey_window(128, 128)
    assert w[0, 0] < 1e-3
    assert w[-1, -1] < 1e-3


# ---------------------------------------------------------------------------
# _intersect
# ---------------------------------------------------------------------------

def test_intersect_sine():
    x = np.linspace(0, 2 * np.pi, 2000)
    crossings = _intersect(x, np.sin(x), np.zeros_like(x))
    # should find crossings at 0, pi (2pi is edge — not always detected)
    assert len(crossings) >= 2
    assert crossings[0] == pytest.approx(0.0, abs=0.01)
    assert crossings[1] == pytest.approx(np.pi, abs=0.01)


def test_intersect_no_duplicates():
    x = np.linspace(0, 2 * np.pi, 500)
    crossings = _intersect(x, np.sin(x), np.zeros_like(x))
    assert len(crossings) == len(np.unique(crossings))


def test_intersect_no_crossing():
    x = np.linspace(0, 1, 100)
    # a always above b
    crossings = _intersect(x, np.ones_like(x), np.zeros_like(x))
    assert len(crossings) == 0


# ---------------------------------------------------------------------------
# bin_localisations
# ---------------------------------------------------------------------------

def test_bin_localisations_shape():
    pos = RNG.uniform(0, 50, (500, 2))
    im = bin_localisations(pos, nx=100, ny=80, zoom=2.0)
    assert im.shape == (80, 100)


def test_bin_localisations_count():
    """All localisations should be binned (none clipped)."""
    pos = RNG.uniform(1, 49, (1000, 2))   # strictly inside 50x50 px FOV
    im = bin_localisations(pos, nx=500, ny=500, zoom=10.0)
    assert im.sum() == pytest.approx(1000, abs=5)   # tiny floating-point clip tolerance


def test_bin_localisations_zoom():
    """Zooming changes image size but not count."""
    pos = RNG.uniform(1, 49, (500, 2))
    im1 = bin_localisations(pos, nx=500,  ny=500,  zoom=10.0)
    im2 = bin_localisations(pos, nx=1000, ny=1000, zoom=20.0)
    assert im1.shape == (500, 500)
    assert im2.shape == (1000, 1000)
    assert im1.sum() == pytest.approx(im2.sum(), rel=0.02)


# ---------------------------------------------------------------------------
# frc
# ---------------------------------------------------------------------------

def test_frc_identical_images():
    """FRC of image with itself must be exactly 1 everywhere."""
    im = RNG.random((128, 128)).astype(np.float32) * 100
    curve = frc(im, im)
    assert np.allclose(curve, 1.0, atol=1e-6)


def test_frc_independent_noise_near_zero():
    """FRC of two independent noise images should be close to 0."""
    n1 = RNG.random((256, 256)).astype(np.float32)
    n2 = RNG.random((256, 256)).astype(np.float32)
    curve = frc(n1, n2)
    sz = 256
    q = np.arange(len(curve)) / sz
    assert np.abs(curve[q < 0.5].mean()) < 0.1


def test_frc_non_square_pad():
    """Non-square images are zero-padded; result should still be length r_max+1."""
    im1 = RNG.random((128, 192)).astype(np.float32)
    im2 = RNG.random((128, 192)).astype(np.float32)
    curve = frc(im1, im2)
    # padded to 192×192; corner ring ≈ round(96*sqrt(2)) + 1
    expected = int(round(96 * np.sqrt(2))) + 1
    assert len(curve) == expected


def test_frc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        frc(np.zeros((64, 64)), np.zeros((64, 128)))


# ---------------------------------------------------------------------------
# frc_to_resolution
# ---------------------------------------------------------------------------

def test_frc_to_resolution_self_is_nan():
    """Self-FRC (all 1s) never crosses 1/7 → resolution = NaN."""
    im = RNG.random((128, 128)).astype(np.float32)
    curve = frc(im, im)
    res, _, _ = frc_to_resolution(curve, 128)
    assert np.isnan(res)


def test_frc_to_resolution_finite():
    """Shared low-freq signal should yield a finite resolution."""
    sz = 256
    signal = gaussian_filter(
        RNG.random((sz, sz)).astype(np.float32) * 500, sigma=8
    )
    n1 = signal + RNG.random((sz, sz)).astype(np.float32) * 10
    n2 = signal + RNG.random((sz, sz)).astype(np.float32) * 10
    curve = frc(n1, n2)
    res, hi, lo = frc_to_resolution(curve, sz)
    assert np.isfinite(res)
    assert res > 0
    assert hi <= res <= lo or hi == 0   # hi ≤ res ≤ lo (or hi=0 if not found)


def test_frc_to_resolution_length_mismatch_raises():
    curve = np.ones(50)   # wrong length for sz=128
    with pytest.raises(ValueError):
        frc_to_resolution(curve, 128)


# ---------------------------------------------------------------------------
# fire (integration)
# ---------------------------------------------------------------------------

def test_fire_finite_result():
    """FIRE on structured data returns finite positive result."""
    sz = 64
    signal = gaussian_filter(
        RNG.random((sz, sz)).astype(np.float32) * 1000, sigma=4
    )
    # Simulate localisations from Poisson-sampled SR image
    y_idx, x_idx = np.where(signal > signal.mean())
    positions = np.column_stack([
        x_idx.astype(float) / 1.0,
        y_idx.astype(float) / 1.0,
        np.arange(len(x_idx), dtype=float),   # frame = index
    ])
    res_nm, curve, hi, lo = fire(
        positions, nx=sz, ny=sz, zoom=1.0,
        n_blocks=10, reps=3, pixel_size_nm=100.0,
    )
    assert np.isfinite(res_nm)
    assert res_nm > 0
    assert len(curve) > 0


def test_fire_accepts_dataframe():
    """fire() accepts a pandas DataFrame with xc, yc, frame columns."""
    import pandas as pd
    sz = 64
    signal = gaussian_filter(
        RNG.random((sz, sz)).astype(np.float32) * 1000, sigma=4
    )
    y_idx, x_idx = np.where(signal > signal.mean())
    df = pd.DataFrame({
        'xc': x_idx.astype(float),
        'yc': y_idx.astype(float),
        'frame': np.arange(len(x_idx), dtype=float),
    })
    res_nm, _, _, _ = fire(df, nx=sz, ny=sz, zoom=1.0, n_blocks=5, reps=2)
    assert np.isfinite(res_nm)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
