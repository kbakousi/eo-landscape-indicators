"""Seasonal composites, climatologies, anomalies and standardized anomalies.

Arrays are time-first: ``values`` has shape ``(T, ...)`` and ``dates`` holds
``T`` dates (anything ``numpy.datetime64`` accepts). Works for any monthly
variable: ERA5-Land temperature or soil moisture, GRACE water storage,
CHIRPS precipitation, or SPEI.

Seasons are meteorological: DJF, MAM, JJA, SON. December belongs to the
*next* year's winter (December 2020 is in winter 2021).

References
----------
WMO (2017). WMO Guidelines on the Calculation of Climate Normals.
    WMO-No. 1203.
McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought
    frequency and duration to time scales. 8th Conference on Applied
    Climatology, AMS, 179-184.
Vicente-Serrano, S.M., Begueria, S., Lopez-Moreno, J.I. (2010). A multiscalar
    drought index sensitive to global warming: the Standardized
    Precipitation Evapotranspiration Index. Journal of Climate 23,
    1696-1718. https://doi.org/10.1175/2009JCLI2909.1
Thomas, A.C., Reager, J.T., Famiglietti, J.S., Rodell, M. (2014). A
    GRACE-based water storage deficit approach for hydrological drought
    characterization. Geophysical Research Letters 41, 1537-1545.
    https://doi.org/10.1002/2014GL059323
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "season_of",
    "seasonal_composites",
    "Climatology",
    "climatology",
    "anomalies",
    "standardized_anomalies",
    "period_mean_range",
    "drought_category",
]

_SEASON_BY_MONTH = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def _months_years(dates):
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    return idx.month.to_numpy(), idx.year.to_numpy()


def season_of(dates):
    """Return ``(season, season_year)`` arrays for each date.

    December is assigned to the following year's DJF season.
    """
    months, years = _months_years(dates)
    season = np.array([_SEASON_BY_MONTH[m] for m in months])
    season_year = np.where(months == 12, years + 1, years)
    return season, season_year


def seasonal_composites(values, dates, min_months: int = 3, reducer: str = "mean"):
    """Aggregate a monthly series into seasonal composites.

    Parameters
    ----------
    values : array_like, shape (T, ...)
        Monthly values; NaN marks missing data.
    dates : array_like, length T
        One date per month.
    min_months : int
        A pixel's seasonal value is NaN unless at least this many of the
        season's months are valid (1-3). Seasons with fewer time steps in
        the input are dropped, so a season cut by the start or end of the
        series is never reported as complete.
    reducer : {"mean", "median"}

    Returns
    -------
    table : pandas.DataFrame
        Columns ``season`` and ``season_year``, one row per composite.
    composites : numpy.ndarray, shape (S, ...)
    """
    if not 1 <= min_months <= 3:
        raise ValueError("min_months must be 1, 2 or 3")
    if reducer not in ("mean", "median"):
        raise ValueError("reducer must be 'mean' or 'median'")
    values = np.asarray(values, dtype="float64")
    season, season_year = season_of(dates)
    rows, out = [], []
    keys = pd.DataFrame({"season": season, "season_year": season_year})
    for (s, y), grp in keys.groupby(["season", "season_year"], sort=False):
        idx = grp.index.to_numpy()
        if len(idx) < min_months:
            continue
        block = values[idx]
        n_valid = np.sum(np.isfinite(block), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            agg = np.nanmean(block, axis=0) if reducer == "mean" else np.nanmedian(block, axis=0)
        agg = np.where(n_valid >= min_months, agg, np.nan)
        rows.append((s, y))
        out.append(agg)
    table = pd.DataFrame(rows, columns=["season", "season_year"])
    if not out:
        return table, np.empty((0,) + values.shape[1:])
    return table, np.stack(out)


@dataclass
class Climatology:
    """Mean, standard deviation and sample size per key over a baseline."""

    by: str
    keys: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n: np.ndarray
    baseline: tuple[int, int]


def _keys(dates, by):
    if by == "month":
        months, _ = _months_years(dates)
        return months
    if by == "season":
        return season_of(dates)[0]
    raise ValueError("by must be 'month' or 'season'")


def climatology(values, dates, baseline: tuple[int, int], by: str = "month"):
    """Compute the baseline mean and standard deviation per calendar key.

    Parameters
    ----------
    values : array_like, shape (T, ...)
    dates : array_like, length T
    baseline : (first_year, last_year)
        Inclusive. Required: there is no default period. WMO recommends a
        30-year normal (currently 1991-2020); a shorter baseline triggers a
        warning.
    by : {"month", "season"}
        Group by calendar month or by meteorological season. For seasonal
        input pass the composites' representative dates.

    Notes
    -----
    The standard deviation uses ``ddof=1``. Missing values are ignored and
    never interpolated (e.g. the GRACE / GRACE-FO gap, Jul 2017 - May 2018).
    """
    _check_baseline(baseline, stacklevel=3)
    keys = _keys(dates, by)
    _, years = _months_years(dates)
    uniq, mean, std, n = _climatology_arrays(values, keys, years, baseline)
    return Climatology(by, uniq, mean, std, n, tuple(baseline))


def _check_baseline(baseline, stacklevel: int = 2):
    first, last = baseline
    if last < first:
        raise ValueError("baseline must be (first_year, last_year)")
    n_years = last - first + 1
    if n_years < 30:
        warnings.warn(
            f"baseline covers {n_years} years; WMO (2017) recommends 30-year normals",
            UserWarning,
            stacklevel=stacklevel,
        )


def _climatology_arrays(values, keys, years, baseline):
    """Mean, std (ddof=1) and count per key over the baseline years (no checks)."""
    first, last = baseline
    values = np.asarray(values, dtype="float64")
    in_base = (years >= first) & (years <= last)
    uniq = np.unique(keys)
    means, stds, ns = [], [], []
    for k in uniq:
        sel = values[(keys == k) & in_base]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means.append(np.nanmean(sel, axis=0))
            stds.append(np.nanstd(sel, axis=0, ddof=1))
        ns.append(np.sum(np.isfinite(sel), axis=0))
    return uniq, np.stack(means), np.stack(stds), np.stack(ns)


def _lookup(clim: Climatology, dates, field):
    keys = _keys(dates, clim.by)
    pos = {k: i for i, k in enumerate(clim.keys)}
    missing = set(np.unique(keys)) - set(pos)
    if missing:
        raise ValueError(f"no baseline values for keys {sorted(missing)}")
    arr = getattr(clim, field)
    return arr[[pos[k] for k in keys]]


def anomalies(values, dates, clim: Climatology):
    """Return ``values - baseline mean`` for each time step's key."""
    values = np.asarray(values, dtype="float64")
    return values - _lookup(clim, dates, "mean")


def standardized_anomalies(values, dates, clim: Climatology, min_std: float = 0.0, min_n: int = 2):
    """Return ``(values - mean) / std`` per key.

    Pixels with ``std <= min_std`` or fewer than ``min_n`` baseline values are
    NaN, never divided by a small constant.

    Do not apply this to SPEI: SPEI is already standardized.
    """
    mean = _lookup(clim, dates, "mean")
    std = _lookup(clim, dates, "std")
    n = _lookup(clim, dates, "n")
    return _standardize(values, mean, std, n, min_std, min_n)


def _standardize(values, mean, std, n, min_std: float = 0.0, min_n: int = 2):
    """``(values - mean) / std`` where ``std > min_std`` and ``n >= min_n``, else NaN."""
    values = np.asarray(values, dtype="float64")
    std = np.asarray(std, dtype="float64")
    ok = np.isfinite(std) & (std > min_std) & (np.asarray(n) >= min_n)
    out = np.full(np.broadcast(values, std).shape, np.nan)
    np.divide(values - mean, std, out=out, where=ok)
    return out


def period_mean_range(values, select=None):
    """Return the mean and the range (max - min) along the time axis.

    ``select`` is an optional boolean array of length T choosing time steps.
    All-NaN pixels give NaN.
    """
    values = np.asarray(values, dtype="float64")
    if select is not None:
        values = values[np.asarray(select, dtype=bool)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=0), np.nanmax(values, axis=0) - np.nanmin(values, axis=0)


def drought_category(index_values):
    """Classify SPI/SPEI values into the McKee et al. (1993) dry categories.

    Returns 0 (no drought, > -1.0), 1 (moderately dry, -1.5 < x <= -1.0),
    2 (severely dry, -2.0 < x <= -1.5), 3 (extremely dry, <= -2.0), and NaN
    where the input is NaN.
    """
    x = np.asarray(index_values, dtype="float64")
    out = np.zeros(x.shape)
    out[x <= -1.0] = 1
    out[x <= -1.5] = 2
    out[x <= -2.0] = 3
    out[~np.isfinite(x)] = np.nan
    return out
