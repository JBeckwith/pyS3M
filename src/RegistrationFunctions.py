#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coordinate-transform registration between two independent point sets.

Ported from the transform-fitting core of
``pyFRETMiSeq/src/PreProcessingFunctions.py`` (mutual-nearest-neighbour
matching + RANSAC-fit affine transform), which there registers the two
channels of one beamsplitter camera. Here it is used to register two
completely independent cameras (different sensor, different pixel size,
different FOV) against each other via matched single-molecule positions,
rather than via raw bead images — so only the matching + transform-fitting
core is needed, not pyFRETMiSeq's own spot-detection/fitting layer.

Fit points in the same physical units (e.g. nm) for both point sets, not raw
pixel coordinates — otherwise the fitted transform conflates genuine
rotation/shear/translation with the trivial pixel-size ratio between the two
cameras.

jsb92, 2026
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from skimage.transform import AffineTransform
from skimage.measure import ransac


def match_spot_pairs(
    pts_src: NDArray, pts_dst: NDArray, max_distance: float = 5.0
) -> tuple[NDArray, NDArray]:
    """Match two point sets using mutual nearest-neighbours with a distance gate.

    A pair (i, j) is retained only when:
    - j is the nearest neighbour of i in pts_dst, AND
    - i is the nearest neighbour of j in pts_src, AND
    - the distance between them is <= max_distance (same units as the points).

    This avoids the one-to-many assignments that plague simple greedy
    nearest-neighbour matching when the two point sets are substantially
    offset or rotated relative to each other.

    Args:
        pts_src: (N, 2) positions from the source set.
        pts_dst: (M, 2) positions from the destination set.
        max_distance: maximum distance for a valid pair (same units as points).

    Returns:
        Tuple of (matched_src, matched_dst), each (K, 2).
    """
    pts_src = np.asarray(pts_src)
    pts_dst = np.asarray(pts_dst)
    if len(pts_src) == 0 or len(pts_dst) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    D = cdist(pts_src, pts_dst)
    nn_src = np.argmin(D, axis=1)   # for each src: index of nearest dst
    nn_dst = np.argmin(D, axis=0)   # for each dst: index of nearest src

    matched_src, matched_dst = [], []
    for i, j in enumerate(nn_src):
        if nn_dst[j] == i and D[i, j] <= max_distance:
            matched_src.append(pts_src[i])
            matched_dst.append(pts_dst[j])

    if not matched_src:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.array(matched_src), np.array(matched_dst)


def match_spot_pairs_indexed(
    pts_src: NDArray, pts_dst: NDArray, max_distance: float = 5.0
) -> tuple[list[int], list[int]]:
    """Like match_spot_pairs but returns index arrays instead of coordinates.

    Useful when the source coordinates have been pre-transformed and the
    caller needs to recover original (un-transformed) coordinates by index.

    Args:
        pts_src: (N, 2) positions from the source set.
        pts_dst: (M, 2) positions from the destination set.
        max_distance: maximum distance for a valid pair (same units as points).

    Returns:
        Tuple of (src_idx, dst_idx) index lists into pts_src / pts_dst.
    """
    pts_src = np.asarray(pts_src)
    pts_dst = np.asarray(pts_dst)
    if len(pts_src) == 0 or len(pts_dst) == 0:
        return [], []

    D = cdist(pts_src, pts_dst)
    nn_src = np.argmin(D, axis=1)
    nn_dst = np.argmin(D, axis=0)

    src_idx, dst_idx = [], []
    for i, j in enumerate(nn_src):
        if nn_dst[j] == i and D[i, j] <= max_distance:
            src_idx.append(i)
            dst_idx.append(j)
    return src_idx, dst_idx


def fit_affine_transform(
    pts_src: NDArray,
    pts_dst: NDArray,
    residual_threshold: float = 2.0,
    min_samples: int = 3,
    max_trials: int = 1000,
) -> tuple[AffineTransform, NDArray]:
    """Fit a robust affine transform mapping pts_src -> pts_dst.

    Uses RANSAC to reject outlier pairs (mismatches, multiply-registered
    beads, etc.); falls back to a plain (non-robust) least-squares affine fit
    over all supplied pairs if RANSAC fails to converge (e.g. too few pairs).

    Args:
        pts_src: (K, 2) matched source positions (same units as pts_dst).
        pts_dst: (K, 2) matched destination positions.
        residual_threshold: RANSAC inlier distance threshold (same units as points).
        min_samples: minimum point pairs per RANSAC trial (3 for an affine transform).
        max_trials: maximum RANSAC iterations.

    Returns:
        Tuple of (tform, inliers) where tform is a fitted
        skimage.transform.AffineTransform and inliers is a boolean mask into
        pts_src/pts_dst (all-True if RANSAC fell back to the plain fit).
    """
    pts_src = np.asarray(pts_src)
    pts_dst = np.asarray(pts_dst)
    if len(pts_src) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} matched pairs to fit an affine "
            f"transform, got {len(pts_src)}."
        )

    try:
        tform, inliers = ransac(
            (pts_src, pts_dst),
            AffineTransform,
            min_samples=min_samples,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
        )
        if tform is None:
            raise RuntimeError("RANSAC did not converge")
    except Exception:
        tform = AffineTransform()
        tform.estimate(pts_src, pts_dst)
        inliers = np.ones(len(pts_src), dtype=bool)

    return tform, inliers


def save_transform(tform: AffineTransform, filepath: str | Path) -> None:
    """Save an AffineTransform to a CSV file (the 3x3 homogeneous matrix).

    Args:
        tform: skimage AffineTransform (or any transform with a .params matrix).
        filepath: destination path, e.g. 'ximea_to_prime95b_transform.csv'.
    """
    np.savetxt(filepath, tform.params, delimiter=",")


def load_transform(filepath: str | Path) -> AffineTransform:
    """Load an AffineTransform from a CSV file saved by save_transform.

    Args:
        filepath: path to the CSV file.

    Returns:
        AffineTransform reconstructed from the saved 3x3 matrix.
    """
    matrix = np.loadtxt(filepath, delimiter=",")
    return AffineTransform(matrix=matrix)
