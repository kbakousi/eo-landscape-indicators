"""Burned-area frequency from the MODIS MCD64A1 monthly burned-area product.

This module is an independent implementation of the burned-area use of the
MCD64A1 product described in Giglio et al. (2018).

``BurnDate`` codes: 1-366 = day of burn; 0 = unburned; -1 = unmapped
(insufficient data); -2 = water. A month is "burned" only where
``BurnDate > 0``. Day-of-year values are never summed.

Limits: at 500 m resolution small burns are under-detected (Giglio et al.
2018). The record starts in November 2000.

References
----------
Giglio, L., Boschetti, L., Roy, D.P., Humber, M.L., Justice, C.O. (2018). The
    Collection 6 MODIS burned area mapping algorithm and product. Remote
    Sensing of Environment 217, 72-85.
    https://doi.org/10.1016/j.rse.2018.08.005
Giglio, L., Justice, C., Boschetti, L., Roy, D. (2021). MCD64A1 MODIS/Terra+
    Aqua Burned Area Monthly L3 Global 500m SIN Grid V061 [Dataset]. NASA LP
    DAAC. https://doi.org/10.5067/MODIS/MCD64A1.061
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["monthly_flags", "annual_burned", "burn_frequency", "burned_area"]


def monthly_flags(burn_date):
    """Return ``(burned, mapped)`` boolean arrays for monthly BurnDate layers.

    ``burned`` is ``BurnDate > 0``. ``mapped`` is ``BurnDate >= 0``: water
    (-2) and unmapped (-1) pixels are not mapped. NaN counts as not mapped.
    """
    bd = np.asarray(burn_date, dtype="float64")
    with np.errstate(invalid="ignore"):
        burned = bd > 0
        mapped = bd >= 0
    return burned, mapped


def annual_burned(burn_date, dates):
    """Collapse monthly layers into one burned / not-burned layer per year.

    Parameters
    ----------
    burn_date : array_like, shape (T, ...)
        Monthly MCD64A1 ``BurnDate`` layers.
    dates : array_like, length T

    Returns
    -------
    years : numpy.ndarray
    annual : numpy.ndarray of float, shape (Y, ...)
        1.0 if any month of that year burned, 0.0 if at least one month was
        mapped and none burned, NaN if no month of that year was mapped.
    """
    burned, mapped = monthly_flags(burn_date)
    year = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates))).year.to_numpy()
    years = np.unique(year)
    out = []
    for y in years:
        sel = year == y
        any_burn = burned[sel].any(axis=0)
        any_mapped = mapped[sel].any(axis=0)
        out.append(np.where(any_mapped | any_burn, any_burn.astype(float), np.nan))
    return years, np.stack(out)


def burn_frequency(annual):
    """Number of burned years per pixel, and the number of mapped years.

    Returns ``(frequency, n_years_mapped)``. ``frequency`` is NaN where no
    year was mapped.
    """
    a = np.asarray(annual, dtype="float64")
    n = np.sum(np.isfinite(a), axis=0)
    freq = np.where(n > 0, np.nansum(a, axis=0), np.nan)
    return freq, n


def burned_area(flags, pixel_area_m2):
    """Burned area in hectares for each layer of a 0/1 (or boolean) array.

    Parameters
    ----------
    flags : array_like, shape (Y, ...) or (...)
        Burned flags; NaN is treated as not burned.
    pixel_area_m2 : float or array_like
        Area of each pixel in m^2. Use true areas (or an equal-area grid such
        as EPSG:3035 for Europe), not degrees.
    """
    f = np.nan_to_num(np.asarray(flags, dtype="float64"), nan=0.0)
    if f.ndim < 2:
        raise ValueError("flags must have at least two (spatial) dimensions")
    area = np.broadcast_to(np.asarray(pixel_area_m2, dtype="float64"), f.shape[-2:])
    return np.sum(f * area, axis=(-2, -1)) / 10_000.0
