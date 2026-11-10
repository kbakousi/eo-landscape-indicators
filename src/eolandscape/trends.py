"""Per-pixel trend analysis of annual series (e.g. growing-season NDVI).

Arrays are time-first: ``values`` has shape ``(T, ...)`` and ``t`` holds the
``T`` time coordinates (e.g. years).

Greening and browning should be judged by significance (Mann-Kendall,
corrected for multiple testing), not by a fixed slope threshold
(de Jong et al. 2011; Wilks 2016).

References
----------
Mann, H.B. (1945). Nonparametric tests against trend. Econometrica 13,
    245-259. https://doi.org/10.2307/1907187
Kendall, M.G. (1975). Rank Correlation Methods, 4th ed. Griffin, London.
Sen, P.K. (1968). Estimates of the regression coefficient based on Kendall's
    tau. Journal of the American Statistical Association 63, 1379-1389.
    https://doi.org/10.1080/01621459.1968.10480934
Yue, S., Pilon, P., Phinney, B., Cavadias, G. (2002). The influence of
    autocorrelation on the ability to detect trend in hydrological series.
    Hydrological Processes 16, 1807-1829. https://doi.org/10.1002/hyp.1095
Benjamini, Y., Hochberg, Y. (1995). Controlling the false discovery rate: a
    practical and powerful approach to multiple testing. Journal of the
    Royal Statistical Society B 57, 289-300.
    https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
Wilks, D.S. (2016). "The stippling shows statistically significant grid
    points". Bulletin of the American Meteorological Society 97, 2263-2273.
    https://doi.org/10.1175/BAMS-D-15-00267.1
de Jong, R., de Bruin, S., de Wit, A., Schaepman, M.E., Dent, D.L. (2011).
    Analysis of monotonic greening and browning trends from global NDVI
    time-series. Remote Sensing of Environment 115, 692-702.
    https://doi.org/10.1016/j.rse.2010.10.011
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "annual_composite",
    "ols_trend",
    "theil_sen",
    "mann_kendall_1d",
    "mann_kendall",
    "benjamini_hochberg",
]


def annual_composite(values, dates, months: tuple[int, int], min_obs: int = 1, reducer: str = "mean"):
    """Aggregate observations into one value per year within a month window.

    Parameters
    ----------
    values : array_like, shape (T, ...)
        Quality-masked observations (NaN = unusable).
    dates : array_like, length T
    months : (first_month, last_month)
        Inclusive, within one calendar year (e.g. ``(4, 10)``).
    min_obs : int
        A pixel's annual value is NaN with fewer valid observations.
    reducer : {"mean", "median"}

    Returns
    -------
    years : numpy.ndarray
    composite : numpy.ndarray, shape (Y, ...)
    """
    first, last = months
    if not 1 <= first <= last <= 12:
        raise ValueError("months must satisfy 1 <= first <= last <= 12")
    values = np.asarray(values, dtype="float64")
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    in_window = (idx.month >= first) & (idx.month <= last)
    years = np.unique(idx.year[in_window])
    out = []
    for y in years:
        block = values[in_window & (idx.year == y)]
        n = np.sum(np.isfinite(block), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            agg = np.nanmean(block, axis=0) if reducer == "mean" else np.nanmedian(block, axis=0)
        out.append(np.where(n >= min_obs, agg, np.nan))
    return years, np.stack(out) if out else np.empty((0,) + values.shape[1:])


def _prep(values, t):
    y = np.asarray(values, dtype="float64")
    t = np.asarray(t, dtype="float64")
    if t.ndim != 1 or t.shape[0] != y.shape[0]:
        raise ValueError("t must be 1-D with the same length as values' first axis")
    tb = t.reshape((-1,) + (1,) * (y.ndim - 1))
    return y, np.broadcast_to(tb, y.shape)


def ols_trend(values, t, min_n: int = 10):
    """Ordinary least-squares slope, intercept and R^2 per pixel.

    Pixels with fewer than ``min_n`` valid values return NaN. R^2 is NaN for
    a constant series.

    Returns
    -------
    slope, intercept, r2 : numpy.ndarray
    """
    y, tb = _prep(values, t)
    v = np.isfinite(y)
    n = v.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = np.where(v, tb, 0).sum(axis=0) / n
        ym = np.where(v, y, 0).sum(axis=0) / n
        dt = np.where(v, tb - tm, 0)
        dy = np.where(v, y - ym, 0)
        sxx = (dt**2).sum(axis=0)
        slope = (dt * dy).sum(axis=0) / sxx
        intercept = ym - slope * tm
        resid = np.where(v, y - (intercept + slope * tb), 0)
        ss_tot = (dy**2).sum(axis=0)
        r2 = np.where(ss_tot > 0, 1 - (resid**2).sum(axis=0) / ss_tot, np.nan)
    bad = (n < max(min_n, 2)) | (sxx == 0)
    return (np.where(bad, np.nan, slope), np.where(bad, np.nan, intercept), np.where(bad, np.nan, r2))


def theil_sen(values, t, min_n: int = 10):
    """Theil-Sen slope (median of pairwise slopes; Sen 1968) and intercept.

    The intercept is ``median(y - slope * t)``. Pixels with fewer than
    ``min_n`` valid values return NaN.
    """
    y, _ = _prep(values, t)
    t = np.asarray(t, dtype="float64")
    i, j = np.triu_indices(len(t), k=1)
    dt = (t[j] - t[i]).reshape((-1,) + (1,) * (y.ndim - 1))
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        slopes = (y[j] - y[i]) / dt
        slope = np.nanmedian(slopes, axis=0)
        tb = t.reshape((-1,) + (1,) * (y.ndim - 1))
        intercept = np.nanmedian(y - slope * tb, axis=0)
    n = np.isfinite(y).sum(axis=0)
    bad = n < max(min_n, 2)
    return np.where(bad, np.nan, slope), np.where(bad, np.nan, intercept)


def _sen_slope_1d(y, t):
    i, j = np.triu_indices(len(t), k=1)
    return np.median((y[j] - y[i]) / (t[j] - t[i]))


def mann_kendall_1d(y, t=None, prewhiten: bool = False):
    """Two-sided Mann-Kendall test for one series.

    NaN values are dropped. The variance includes the correction for tied
    values (Kendall 1975). With ``prewhiten=True`` the trend-free
    pre-whitening of Yue et al. (2002) removes lag-1 autocorrelation first;
    it assumes equally spaced observations.

    Returns
    -------
    s : float
        The Mann-Kendall S statistic.
    z : float
    p : float
        Two-sided p-value from the normal approximation. NaN when fewer than
        3 values remain or the variance is zero.
    """
    y = np.asarray(y, dtype="float64")
    t = np.arange(len(y), dtype="float64") if t is None else np.asarray(t, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(t)
    y, t = y[ok], t[ok]
    order = np.argsort(t)
    y, t = y[order], t[order]
    if prewhiten and len(y) >= 4:
        b = _sen_slope_1d(y, t)
        x = y - b * t
        xc = x - x.mean()
        denom = np.sum(xc**2)
        r1 = np.sum(xc[1:] * xc[:-1]) / denom if denom > 0 else 0.0
        y = (x[1:] - r1 * x[:-1]) + b * t[1:]
    n = len(y)
    if n < 3:
        return np.nan, np.nan, np.nan
    i, j = np.triu_indices(n, k=1)
    s = float(np.sum(np.sign(y[j] - y[i])))
    _, counts = np.unique(y, return_counts=True)
    ties = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var = (n * (n - 1) * (2 * n + 5) - ties) / 18.0
    if var <= 0:
        return s, np.nan, np.nan
    if s > 0:
        z = (s - 1) / np.sqrt(var)
    elif s < 0:
        z = (s + 1) / np.sqrt(var)
    else:
        z = 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return s, z, p


def mann_kendall(values, t=None, prewhiten: bool = False, min_n: int = 10):
    """Apply :func:`mann_kendall_1d` to every pixel.

    Returns ``(z, p)`` arrays with the pixel shape. Pixels with fewer than
    ``min_n`` valid values are NaN. Correct the p-values for multiple testing
    with :func:`benjamini_hochberg` before reporting significant areas.
    """
    y = np.asarray(values, dtype="float64")
    flat = y.reshape(y.shape[0], -1)
    z = np.full(flat.shape[1], np.nan)
    p = np.full(flat.shape[1], np.nan)
    for k in range(flat.shape[1]):
        col = flat[:, k]
        if np.sum(np.isfinite(col)) < max(min_n, 3):
            continue
        _, z[k], p[k] = mann_kendall_1d(col, t, prewhiten)
    return z.reshape(y.shape[1:]), p.reshape(y.shape[1:])


def benjamini_hochberg(pvalues, alpha: float = 0.05):
    """Return a boolean array: ``True`` where the test is significant under
    false-discovery-rate control at level ``alpha`` (Benjamini & Hochberg 1995).

    NaN p-values are ignored and returned as ``False``.
    """
    p = np.asarray(pvalues, dtype="float64")
    flat = p.ravel()
    finite = np.isfinite(flat)
    out = np.zeros(flat.shape, dtype=bool)
    pv = flat[finite]
    m = pv.size
    if m == 0:
        return out.reshape(p.shape)
    order = np.argsort(pv)
    ranked = pv[order]
    below = ranked <= alpha * np.arange(1, m + 1) / m
    if below.any():
        cutoff = ranked[np.max(np.nonzero(below))]
        out[np.flatnonzero(finite)] = pv <= cutoff
    return out.reshape(p.shape)
