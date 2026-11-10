"""Structural fragmentation of a focal class (e.g. forest, or any class group).

Only published metrics are produced. There is no composite "integrity" or
"condition" index.

References
----------
Riitters, K., Wickham, J., O'Neill, R., Jones, B., Smith, E. (2000).
    Global-scale patterns of forest fragmentation. Conservation Ecology 4(2):
    3. (Pf and Pff in a moving window)
Vogt, P., Riitters, K.H., Estreguil, C., Kozak, J., Wade, T.G., Wickham, J.D.
    (2007). Mapping spatial patterns with morphological image processing.
    Landscape Ecology 22, 171-177. https://doi.org/10.1007/s10980-006-9013-2
Haddad, N.M. et al. (2015). Habitat fragmentation and its lasting impact on
    Earth's ecosystems. Science Advances 1, e1500052.
    https://doi.org/10.1126/sciadv.1500052 (distance to edge)
McGarigal, K., Marks, B.J. (1995). FRAGSTATS. USDA Forest Service General
    Technical Report PNW-GTR-351. (edge density)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["distance_to_edge", "core_edge", "riitters_pf_pff", "edge_density"]


def _bool(a):
    return np.asarray(a, dtype=bool)


def distance_to_edge(focal, pixel_size_m, valid=None, invalid_is_edge: bool = True):
    """Distance (m) from each focal pixel to the nearest non-focal pixel.

    Distances are between pixel centres and use the real pixel size
    (``pixel_size_m`` is a scalar or ``(dy, dx)``), so a focal pixel next to
    a non-focal one has distance equal to one pixel size. Non-focal pixels
    get 0.

    If ``invalid_is_edge`` is ``False``, invalid (no-data) pixels do not
    create edges. The outer border of the array is not an edge. If no pixel
    can act as an edge, focal pixels get ``inf`` (no edge within the array).
    """
    f = _bool(focal)
    v = np.ones(f.shape, dtype=bool) if valid is None else _bool(valid)
    inside = f & v
    if not invalid_is_edge:
        inside = inside | ~v
    if inside.all():
        return np.where(f & v, np.inf, 0.0)
    d = ndimage.distance_transform_edt(inside, sampling=pixel_size_m)
    return np.where(f & v, d, 0.0)


def core_edge(focal, pixel_size_m, edge_width_m: float, valid=None, invalid_is_edge: bool = True):
    """Split focal pixels into core and edge (morphological approach, Vogt et al. 2007).

    Core pixels are farther than ``edge_width_m`` from any non-focal pixel;
    the remaining focal pixels are edge. ``edge_width_m`` is required.

    Returns
    -------
    core, edge : numpy.ndarray of bool
    """
    if edge_width_m <= 0:
        raise ValueError("edge_width_m must be > 0")
    f = _bool(focal)
    v = np.ones(f.shape, dtype=bool) if valid is None else _bool(valid)
    d = distance_to_edge(f, pixel_size_m, v, invalid_is_edge)
    core = f & v & (d > edge_width_m)
    edge = f & v & ~core
    return core, edge


def riitters_pf_pff(focal, window_px: int, valid=None):
    """Pf and Pff in a square moving window (Riitters et al. 2000).

    Pf is the share of valid pixels in the window that are focal. Pff is the
    share of cardinal-direction pixel pairs with at least one focal pixel
    in which both pixels are focal. Each pair is counted at its left or upper
    pixel.

    Parameters
    ----------
    focal : array_like of bool, 2-D
    window_px : int
        Odd window size in pixels (e.g. 3, 9, 27, 81).
    valid : array_like of bool, optional

    Returns
    -------
    pf, pff : numpy.ndarray
        NaN where the denominator is zero.
    """
    if window_px < 1 or window_px % 2 == 0:
        raise ValueError("window_px must be a positive odd integer")
    f = _bool(focal)
    v = np.ones(f.shape, dtype=bool) if valid is None else _bool(valid)
    f = f & v
    k = np.ones((window_px, window_px))

    def wsum(a):
        return ndimage.convolve(a.astype("float64"), k, mode="constant", cval=0.0)

    any_pair = np.zeros(f.shape)
    both_pair = np.zeros(f.shape)
    # horizontal pairs (r, c)-(r, c+1)
    hv = v[:, :-1] & v[:, 1:]
    any_pair[:, :-1] += hv & (f[:, :-1] | f[:, 1:])
    both_pair[:, :-1] += hv & f[:, :-1] & f[:, 1:]
    # vertical pairs (r, c)-(r+1, c)
    vv = v[:-1, :] & v[1:, :]
    any_pair[:-1, :] += vv & (f[:-1, :] | f[1:, :])
    both_pair[:-1, :] += vv & f[:-1, :] & f[1:, :]

    n_valid = wsum(v)
    n_focal = wsum(f)
    n_any = wsum(any_pair)
    n_both = wsum(both_pair)
    with np.errstate(invalid="ignore", divide="ignore"):
        pf = np.where(n_valid > 0, n_focal / n_valid, np.nan)
        pff = np.where(n_any > 0, n_both / n_any, np.nan)
    return pf, pff


def edge_density(focal, pixel_size_m: float, valid=None):
    """Edge density (m/ha) between focal and non-focal valid pixels.

    Edge length counts shared sides of cardinal neighbours (square pixels of
    side ``pixel_size_m``); the area is the valid area. Edges with no-data
    pixels and the outer boundary are not counted.
    """
    f = _bool(focal)
    v = np.ones(f.shape, dtype=bool) if valid is None else _bool(valid)
    h = v[:, :-1] & v[:, 1:] & (f[:, :-1] != f[:, 1:])
    vt = v[:-1, :] & v[1:, :] & (f[:-1, :] != f[1:, :])
    edge_m = (h.sum() + vt.sum()) * pixel_size_m
    area_ha = v.sum() * pixel_size_m**2 / 10_000.0
    return edge_m / area_ha if area_ha > 0 else np.nan
