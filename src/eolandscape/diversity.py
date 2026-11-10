"""Land-cover composition diversity in a moving window.

This measures the diversity of land-cover classes (a structural landscape
proxy), not species diversity. Natural logarithms are used throughout.

References
----------
Shannon, C.E. (1948). A mathematical theory of communication. Bell System
    Technical Journal 27, 379-423.
    https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
Simpson, E.H. (1949). Measurement of diversity. Nature 163, 688.
    https://doi.org/10.1038/163688a0
Pielou, E.C. (1966). The measurement of diversity in different types of
    biological collections. Journal of Theoretical Biology 13, 131-144.
    https://doi.org/10.1016/0022-5193(66)90013-0
Hill, M.O. (1973). Diversity and evenness: a unifying notation and its
    consequences. Ecology 54, 427-432. https://doi.org/10.2307/1934352
McGarigal, K., Marks, B.J. (1995). FRAGSTATS: spatial pattern analysis
    program for quantifying landscape structure. USDA Forest Service General
    Technical Report PNW-GTR-351. (SHDI, SIDI, PR, SHEI)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

__all__ = [
    "disk_kernel",
    "class_proportions",
    "shannon",
    "simpson",
    "richness",
    "pielou_evenness",
    "hill_numbers",
]


def disk_kernel(radius_px: float):
    """Circular 0/1 kernel containing pixels whose centre is within ``radius_px``."""
    if radius_px < 0:
        raise ValueError("radius_px must be >= 0")
    r = int(np.floor(radius_px))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (xx**2 + yy**2 <= radius_px**2).astype("float64")


def class_proportions(classes, class_values, radius_m: float, pixel_size_m: float):
    """Proportion of each class among the selected classes in a circular window.

    Parameters
    ----------
    classes : array_like of int, 2-D
    class_values : sequence of int
        The classes (or class groups, after remapping) to include. Pixels of
        other values count neither in the numerator nor in the denominator.
    radius_m, pixel_size_m : float
        Window radius and pixel size in metres (square pixels).

    Returns
    -------
    proportions : numpy.ndarray, shape (K, H, W)
        NaN where the window contains no selected pixel. Proportions sum to 1.
    coverage : numpy.ndarray, shape (H, W)
        Share of the window covered by the selected classes. Use it to mask
        windows with too little cover (the minimum is a user choice).
    """
    c = np.asarray(classes)
    kernel = disk_kernel(radius_m / pixel_size_m)
    def wsum(a):  # zero-padded window sum; FFT keeps large windows fast; counts are integers
        return np.rint(np.clip(fftconvolve(a, kernel, mode="same"), 0, None))

    counts = np.stack([wsum((c == v).astype("float64")) for v in class_values])
    total = counts.sum(axis=0)
    window = wsum(np.ones(c.shape))
    with np.errstate(invalid="ignore", divide="ignore"):
        props = np.where(total > 0, counts / total, np.nan)
        coverage = total / window
    return props, coverage


def _p(p):
    p = np.asarray(p, dtype="float64")
    return np.clip(p, 0.0, None)


def shannon(p, axis: int = 0):
    """Shannon index H = -sum(p ln p), with 0 ln 0 = 0. NaN if all p are NaN."""
    p = _p(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log(p), 0.0)
    h = terms.sum(axis=axis)
    return np.where(np.all(np.isnan(p), axis=axis), np.nan, h)


def simpson(p, axis: int = 0):
    """Simpson diversity 1 - sum(p^2) (the Gini-Simpson form; FRAGSTATS SIDI)."""
    p = _p(p)
    d = 1.0 - np.nansum(p**2, axis=axis)
    return np.where(np.all(np.isnan(p), axis=axis), np.nan, d)


def richness(p, min_share: float = 0.0, axis: int = 0):
    """Number of classes with a proportion above ``min_share``."""
    p = np.asarray(p, dtype="float64")
    r = np.sum(np.nan_to_num(p, nan=0.0) > min_share, axis=axis).astype("float64")
    return np.where(np.all(np.isnan(p), axis=axis), np.nan, r)


def pielou_evenness(p, axis: int = 0):
    """Pielou evenness J = H / ln(m), m = number of classes with p > 0.

    Undefined (NaN) when fewer than two classes are present.
    """
    h = shannon(p, axis=axis)
    m = richness(p, 0.0, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m > 1, h / np.log(m), np.nan)


def hill_numbers(p, axis: int = 0):
    """Hill numbers of order 1 and 2: ``exp(H)`` and ``1 / sum(p^2)``.

    Both are effective numbers of equally common classes (Hill 1973).
    """
    p = _p(p)
    n1 = np.exp(shannon(p, axis=axis))
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = np.nansum(p**2, axis=axis)
        n2 = np.where(s2 > 0, 1.0 / s2, np.nan)
    return n1, np.where(np.all(np.isnan(p), axis=axis), np.nan, n2)
