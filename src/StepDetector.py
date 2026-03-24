"""
StepDetector.py
~~~~~~~~~~~~~~~

Gaussian likelihood-ratio change-point detector based on:
    Watkins & Yang, J. Phys. Chem. B, 109, 617-628 (2005)
    Boudjellaba et al., Commun. Statist. Theory Meth., 30(3), 407-434 (2001)

Original MATLAB implementation by Yan Jiang (2007-2008).
Refactored to a vectorised Python class by jsb92.

The output convention matches ruptures: detect() returns a sorted list of
change-point indices where the last element is always len(signal).
"""

import numpy as np


def _lr_test(data: np.ndarray):
    """Log-likelihood ratio test for a single segment.

    Tests whether splitting the segment at any interior point gives a
    statistically significant improvement over the single-segment model.

    Args:
        data: 1D array of signal values (must be positive for the log to
              be valid; negative values are guarded by the validity mask).

    Returns:
        best_local (int): 0-based index of the best split point within data.
        lm (float): Maximum log-likelihood ratio score (0 if no valid split).
    """
    n = len(data)
    if n < 2:
        return 0, 0.0

    S  = np.cumsum(data)
    SS = S[-1]

    if SS <= 0:
        return 0, 0.0

    # k runs from 1..n-1 (number of points in the left sub-segment)
    k  = np.arange(1, n)
    sk = S[:-1]          # cumulative sum after k points
    sm = SS - sk         # cumulative sum of remaining points

    valid = (sk > 0) & (sm > 0)
    L = np.zeros(n - 1)
    kv, skv, smv = k[valid], sk[valid], sm[valid]
    L[valid] = (
        skv * np.log(skv / kv)
        + smv * np.log(smv / (n - kv))
        - SS  * np.log(SS  / n)
    )

    best = int(np.argmax(L))
    return best, float(L[best])


def _segment(signal, left, right, cp_set, win_size, threshold):
    """Recursive binary segmentation.

    Splits [left, right) if the best LR split exceeds threshold and the
    segment is longer than win_size, then recurses on both halves.
    """
    if right - left <= win_size:
        return
    local_cp, lm = _lr_test(signal[left:right])
    if lm > threshold:
        global_cp = left + local_cp + 1
        cp_set.add(global_cp)
        _segment(signal, left,      global_cp, cp_set, win_size, threshold)
        _segment(signal, global_cp, right,     cp_set, win_size, threshold)


class StepDetector:
    """Gaussian likelihood-ratio change-point detector (Watkins & Yang 2005).

    Uses vectorised numpy operations for the inner LR test (~100x faster than
    the original Python loop) and a clean recursive binary segmentation.

    Output convention matches ruptures.Pelt.predict: detect() returns a
    sorted list of change-point indices ending with len(signal).

    Args:
        win_size:  Minimum segment length to attempt a split (default 10).
        threshold: LR acceptance threshold (default 400). Increase to be more
                   conservative; decrease to catch smaller steps.
                   Guidance for photobleaching traces (~50-200 pe/frame):
                     100-200 : very sensitive, may over-segment noise
                     400     : original paper default, good starting point
                     800+    : conservative, only large unambiguous steps
    """

    def __init__(self, win_size: int = 10, threshold: float = 400.0):
        self.win_size  = win_size
        self.threshold = threshold

    def detect(self, signal) -> list:
        """Detect change points in a 1D signal.

        Args:
            signal: 1D array-like of intensity values (photoelectrons).

        Returns:
            Sorted list of change-point indices; last element is len(signal).
        """
        signal = np.asarray(signal, dtype=float)
        n = len(signal)
        cp_set = set()
        _segment(signal, 0, n, cp_set, self.win_size, self.threshold)
        return sorted(cp_set) + [n]

    def segment_means(self, signal, cps: list) -> np.ndarray:
        """Return a per-sample array of segment means for plotting.

        Args:
            signal: 1D array-like of intensity values.
            cps:    Change-point list as returned by detect().

        Returns:
            Array of same length as signal with each sample replaced by
            the mean of its segment.
        """
        signal = np.asarray(signal, dtype=float)
        out  = np.empty(len(signal))
        prev = 0
        for cp in cps:
            out[prev:cp] = np.nanmean(signal[prev:cp])
            prev = cp
        return out
