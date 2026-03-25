"""
StepDetector.py
~~~~~~~~~~~~~~~

Gaussian likelihood-ratio change-point detector based on:
    Watkins & Yang, J. Phys. Chem. B, 109, 617-628 (2005)
    Boudjellaba et al., Commun. Statist. Theory Meth., 30(3), 407-434 (2001)

Threshold calculation follows:
    Vostrikova, Theory Probab. Its Appl., 26, 356-362 (1982)
    Gombay & Horvath, J. Multivar. Anal., 56, 120-152 (1996)

as implemented in MULLR by Hugh Wilson
(Wilson et al., J. Phys. Chem. B, 2022,
https://pubs.acs.org/doi/full/10.1021/acs.jpcb.1c08869).

Original MATLAB implementation by Yan Jiang (2007-2008).
Refactored to a vectorised Python class by jsb92.

The output convention matches ruptures: detect() returns a sorted list of
change-point indices where the last element is always len(signal).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma as _sp_gamma

try:
    from ruptures.base import BaseCost
    from ruptures.costs import NotEnoughPoints
    _RUPTURES_AVAILABLE = True
except ImportError:
    _RUPTURES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Threshold — Vostrikova (1982) / Gombay-Horváth (1996)
# ---------------------------------------------------------------------------

def _vost_threshold(n: int, alpha: float = 0.05, d: int = 1) -> float:
    """Asymptotically correct LLR threshold for a segment of length n.

    Implements the Vostrikova (1982) result for the supremum of the
    log-likelihood ratio process, following the derivation in:
        Gombay & Horváth, J. Multivar. Anal., 56, 120-152 (1996)
    and ported from MULLR (Wilson et al., J. Phys. Chem. B, 2022).

    Args:
        n:     Segment length.
        alpha: False-positive rate (probability of a spurious split
               under H₀). Default 0.05 (5%).
        d:     Dimensionality of the signal (1 for scalar traces).

    Returns:
        LLR threshold; accept a split iff max_k LR_k > threshold.
    """
    if n <= 2:
        return np.inf

    log_n = np.log(n)
    h = log_n ** 1.5 / n
    T = np.log(((1.0 - h) ** 2) / (h ** 2))

    prefactor = 2.0 ** (d / 2.0) * _sp_gamma(d / 2.0)

    def thresh_func(x):
        return (
            x ** d * np.exp(-x ** 2 / 2.0) / prefactor
            * (T - d * T / x ** 2 + 4.0 / x ** 2)
            - alpha
        )

    # f(1.0) > 0 and f(20.0) < 0 for all practical (n, alpha, d).
    # brentq on [1, 20] recovers the larger of the two roots (the critical
    # value), matching the MATLAB max(xrootVec) convention.
    try:
        z_n = brentq(thresh_func, 1.0, 20.0)
    except ValueError:
        return np.inf

    return z_n ** 2 / 2.0


# ---------------------------------------------------------------------------
# Core LR test (vectorised)
# ---------------------------------------------------------------------------

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

    k  = np.arange(1, n)
    sk = S[:-1]
    sm = SS - sk

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


# ---------------------------------------------------------------------------
# Gaussian LR test (vectorised) — ported from findcp.m (Jiang 2007-2008)
# ---------------------------------------------------------------------------

def _lr_test_gaussian(data: np.ndarray):
    """Log-likelihood ratio test for a variance-change model (Gaussian noise).

    Tests whether splitting the segment at any interior point gives a
    statistically significant improvement.  The statistic is:

        LRT(k) = |√( N·log σ²_whole − k·log σ²_left − (N−k)·log σ²_right )|

    where all variances use ddof=1.  Valid split points require ≥ 2 samples
    on each side so that the sample variance is defined.

    Args:
        data: 1D array of signal values.

    Returns:
        best_local (int): 0-based index of the best split point within data.
        lm (float): Maximum LRT score (0 if no valid split).
    """
    n = len(data)
    if n < 4:
        return 0, 0.0

    # Precompute cumulative sums for fast mean/variance via ddof=1
    S  = np.cumsum(data)
    SS = np.cumsum(data ** 2)

    wvar = np.var(data, ddof=1)
    if wvar <= 0:
        return 0, 0.0

    wlog = n * np.log(wvar)

    # k: number of points in left segment, ranges 2 … n-2
    k = np.arange(2, n - 1)
    sk  = S[k - 1]
    ssk = SS[k - 1]

    # Unbiased sample variance via Welford / algebraic identity
    # var = (Σx² - (Σx)²/k) / (k-1)
    lvar = (ssk - sk ** 2 / k) / (k - 1)
    rn   = n - k
    rsk  = S[-1] - sk
    rssk = SS[-1] - ssk
    rvar = (rssk - rsk ** 2 / rn) / (rn - 1)

    valid = (lvar > 0) & (rvar > 0)
    L = np.zeros(len(k))
    kv, lv, rv = k[valid], lvar[valid], rvar[valid]
    L[valid] = np.abs(np.sqrt(np.maximum(
        wlog - kv * np.log(lv) - (n - kv) * np.log(rv), 0.0
    )))

    best = int(np.argmax(L))
    return best + 1, float(L[best])   # +1 because k starts at 2 (offset from index 0)


def _gaussian_threshold(n: int, alpha: float = 0.05) -> float:
    """Gumbel extreme-value threshold for the Gaussian LRT.

    Direct port of ``CriVal(N, alpha)`` from findcp.m (Jiang 2007-2008).

    Args:
        n:     Segment length.
        alpha: False-positive rate. Default 0.05.

    Returns:
        Critical value; accept a split iff max_k LRT_k > threshold.
    """
    if n < 4:
        return np.inf
    # findcp.m calls CriVal(N, 1-alpha) — i.e. the argument is a confidence
    # level, not a tail probability.  We follow the same convention here.
    p  = 1.0 - alpha
    yd = -np.log(-0.5 * np.log(p))
    x  = np.log(n)
    a  = np.sqrt(2.0 * np.log(x))
    b  = 2.0 * np.log(x) + np.log(np.log(x))
    return (yd + b) / a


# ---------------------------------------------------------------------------
# Recursive binary segmentation
# ---------------------------------------------------------------------------

def _segment(signal, left, right, cp_set, win_size, threshold_fn, lr_fn):
    """Recursive binary segmentation.

    Splits [left, right) if the best LR split exceeds the Vostrikova
    threshold for this segment length and the segment is longer than
    win_size, then recurses on both halves.

    Args:
        threshold_fn: callable(n) -> float giving the LLR threshold for a
                      segment of length n.
        lr_fn:        callable(data) -> (best_local, lm) — the LR test to use.
    """
    n = right - left
    if n < 2 * win_size:
        return
    local_cp, lm = lr_fn(signal[left:right])
    global_cp = left + local_cp + 1
    if (lm > threshold_fn(n)
            and (global_cp - left) >= win_size
            and (right - global_cp) >= win_size):
        cp_set.add(global_cp)
        _segment(signal, left,      global_cp, cp_set, win_size, threshold_fn, lr_fn)
        _segment(signal, global_cp, right,     cp_set, win_size, threshold_fn, lr_fn)


# ---------------------------------------------------------------------------
# Optional PELT cost function (requires ruptures)
# ---------------------------------------------------------------------------

if _RUPTURES_AVAILABLE:
    class _PoissonCost(BaseCost):
        """Negative Poisson log-likelihood cost for ruptures PELT.

        cost(s, e) = -SS · log(SS / n)  where SS = Σ signal[s:e], n = e - s

        This is the single-segment term in the Watkins & Yang LR statistic,
        making it additive and compatible with the PELT dynamic programme.
        """

        model = "poisson_lr"
        min_size = 2

        def fit(self, signal):
            self.signal = np.asarray(signal, dtype=float).ravel()
            self._cum = np.concatenate(([0.0], np.cumsum(self.signal)))
            return self

        def error(self, start: int, end: int) -> float:
            n  = end - start
            if n < 2:
                raise NotEnoughPoints
            SS = self._cum[end] - self._cum[start]
            if SS <= 0.0:
                return 0.0
            return -SS * np.log(SS / n)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class StepDetector:
    """Gaussian likelihood-ratio change-point detector (Watkins & Yang 2005).

    Uses the Vostrikova (1982) / Gombay-Horváth (1996) asymptotic threshold
    so that the acceptance criterion is a statistically principled false-
    positive rate rather than an arbitrary fixed value.

    Output convention matches ruptures.Pelt.predict: detect() returns a
    sorted list of change-point indices ending with len(signal).

    Args:
        win_size:  Minimum segment length to attempt a split (default 10).
        alpha:     False-positive rate per segment under H₀ (default 0.05).
                   Decrease to be more conservative (fewer detections);
                   increase to be more sensitive (more detections).
        d:         Dimensionality of the signal (1 for scalar intensity
                   traces). Only used by the Poisson estimator.
        backend:   'binseg' (default) — recursive binary segmentation, no
                   extra dependencies.
                   'pelt' — globally optimal PELT with Poisson cost;
                   requires the ruptures package.
        estimator: 'poisson' (default) — Watkins & Yang (2005) LR based on
                   Poisson photon-counting statistics; Vostrikova threshold.
                   'gaussian' — Gaussian variance-change LR from findcp.m
                   (Jiang 2007-2008); Gumbel extreme-value threshold.
                   Use 'gaussian' when background-subtracted traces have
                   both positive and negative values, or when signal-to-noise
                   is dominated by read noise rather than shot noise.
    """

    def __init__(self, win_size: int = 10, alpha: float = 0.05,
                 d: int = 1, backend: str = "binseg",
                 estimator: str = "poisson"):
        self.win_size  = win_size
        self.alpha     = alpha
        self.d         = d
        self.backend   = backend
        self.estimator = estimator.lower()
        self._cache: dict = {}

    def _threshold(self, n: int) -> float:
        """Cached threshold for segment length n (dispatches on estimator)."""
        if n not in self._cache:
            if self.estimator == "gaussian":
                self._cache[n] = _gaussian_threshold(n, self.alpha)
            else:
                self._cache[n] = _vost_threshold(n, self.alpha, self.d)
        return self._cache[n]

    def _lr_fn(self):
        """Return the appropriate LR test function for the current estimator."""
        if self.estimator == "gaussian":
            return _lr_test_gaussian
        return _lr_test

    def detect(self, signal) -> list:
        """Detect change points in a 1D signal.

        Args:
            signal: 1D array-like of intensity values (photoelectrons).

        Returns:
            Sorted list of change-point indices; last element is len(signal).
        """
        signal = np.asarray(signal, dtype=float)
        if self.backend == "pelt":
            return self._detect_pelt(signal)
        return self._detect_binseg(signal)

    def _detect_binseg(self, signal) -> list:
        n      = len(signal)
        cp_set = set()
        _segment(signal, 0, n, cp_set, self.win_size, self._threshold,
                 self._lr_fn())
        return sorted(cp_set) + [n]

    def _detect_pelt(self, signal) -> list:
        if not _RUPTURES_AVAILABLE:
            raise ImportError(
                "ruptures is required for backend='pelt'. "
                "Install with: pip install ruptures"
            )
        import ruptures as rpt
        cost = _PoissonCost()
        algo = rpt.Pelt(custom_cost=cost, min_size=self.win_size, jump=1)
        algo.fit(signal)
        pen = _vost_threshold(len(signal), self.alpha, self.d)
        return algo.predict(pen=pen)

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
