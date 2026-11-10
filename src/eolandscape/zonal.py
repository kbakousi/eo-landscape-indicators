"""Statistics of a raster per zone (e.g. per parcel or administrative unit).

Zones are given as an integer label raster (0 = no zone), for example from
rasterising polygons. Pixels are assigned to a zone by their centre; partial
pixel coverage is not weighted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["zonal_stats"]


def zonal_stats(values, labels, valid=None, pixel_area_m2=None, background: int = 0):
    """Count, mean, median, std, min and max of ``values`` per label.

    Parameters
    ----------
    values : array_like, 2-D
    labels : array_like of int, same shape
    valid : array_like of bool, optional
        Extra mask; non-finite values are always excluded.
    pixel_area_m2 : float or array_like, optional
        If given, ``area_ha`` (valid area) is added and ``mean`` is
        area-weighted.
    background : int
        Label value that is not a zone.

    Returns
    -------
    pandas.DataFrame indexed by label with columns ``n_total`` (all pixels of
    the zone), ``n_valid``, ``valid_fraction``, ``mean``, ``median``, ``std``
    (ddof=1), ``min``, ``max`` and, optionally, ``area_ha``. Zones without a
    valid pixel keep a row with NaN statistics.
    """
    v = np.asarray(values, dtype="float64").ravel()
    lab = np.asarray(labels).ravel()
    if v.shape != lab.shape:
        raise ValueError("values and labels must have the same shape")
    ok = np.isfinite(v) & (lab != background)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool).ravel()
    zone = lab != background
    n_total = pd.Series(lab[zone]).value_counts().rename("n_total")

    df = pd.DataFrame({"label": lab[ok], "v": v[ok]})
    if pixel_area_m2 is not None:
        area = np.broadcast_to(np.asarray(pixel_area_m2, dtype="float64"), np.shape(values)).ravel()
        df["a"] = area[ok]
        df["va"] = df["v"] * df["a"]
    g = df.groupby("label")
    stats = g["v"].agg(n_valid="count", median="median", std="std", min="min", max="max")
    if pixel_area_m2 is not None:
        sums = g[["a", "va"]].sum()
        stats["mean"] = sums["va"] / sums["a"]
        stats["area_ha"] = sums["a"] / 10_000.0
    else:
        stats["mean"] = g["v"].mean()
    out = pd.concat([n_total, stats], axis=1)
    out.index.name = "label"
    out["n_valid"] = out["n_valid"].fillna(0).astype(int)
    out["valid_fraction"] = out["n_valid"] / out["n_total"]
    cols = ["n_total", "n_valid", "valid_fraction", "mean", "median", "std", "min", "max"]
    if pixel_area_m2 is not None:
        cols.append("area_ha")
    return out[cols].sort_index()
