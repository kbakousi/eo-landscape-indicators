"""Labelled interface: the eolandscape methods on ``xarray`` objects.

Every function here takes and returns xarray objects and keeps their labels:
the time, y and x coordinates and the scalar ``spatial_ref`` coordinate in
which odc-stac and rioxarray store the coordinate reference system. Inputs
backed by dask arrays stay lazy and are processed chunk by chunk.

Nothing is re-implemented: each function applies the numpy function of the
same name from the other modules with :func:`xarray.apply_ufunc`, so the
results are identical (this is tested).

Conventions
-----------
- Time dimension ``"time"`` (datetime64); spatial dimensions ``"y"`` and
  ``"x"``. Other names can be passed with ``dim=``.
- Functions that regroup or reduce along time need the whole time axis of a
  pixel in one chunk. They rechunk time automatically; space stays chunked.
- Results on different grids are never combined silently: inputs that must
  share a grid are aligned with ``join="exact"`` and raise otherwise.
- The neighbourhood metrics (:mod:`eolandscape.diversity`,
  :mod:`eolandscape.fragmentation`) are not wrapped: they need overlapping
  chunks and stay numpy-only for now.

Install with ``pip install eolandscape[xarray]``. Import as
``from eolandscape import xr as exr`` to keep ``xr`` free for xarray itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray

from . import anomalies as _an
from . import fire as _fire
from . import indices as _ix
from . import masking as _mk
from . import trends as _tr

__all__ = [
    # indices
    "normalized_difference",
    "ndvi",
    "ndwi_mcfeeters",
    "ndwi_gao",
    "ndmi",
    "nddi",
    "ndre",
    "spectral_indices",
    # masking
    "landsat_c2_reflectance",
    "landsat_qa_clear",
    "s2_l2a_reflectance",
    "s2_scl_clear",
    "modis_vi_scale",
    "modis_vi_clear",
    # anomalies
    "seasonal_composites",
    "climatology",
    "anomalies",
    "standardized_anomalies",
    "period_mean_range",
    "drought_category",
    # trends
    "annual_composite",
    "decimal_years",
    "ols_trend",
    "theil_sen",
    "mann_kendall",
    "benjamini_hochberg",
    # fire
    "annual_burned",
    "burn_frequency",
    "burned_area",
]

_OUT = "__eolandscape_out__"


# --------------------------------------------------------------------------- helpers


def _map(func, *args, dtype="float64", **kwargs):
    """Apply an element-wise numpy function; coordinates kept, lazy on dask."""
    # Shallow copies (data shared, attrs dicts not): older xarray clears
    # coordinate attributes on the objects it is given, see below.
    copies = [a.copy(deep=False) if isinstance(a, (xarray.DataArray, xarray.Dataset)) else a for a in args]
    out = xarray.apply_ufunc(func, *copies, kwargs=kwargs, dask="parallelized", output_dtypes=[dtype])
    return _restore_coord_attrs(out, args)


def _restore_coord_attrs(out, inputs):
    """Give coordinates back the attributes they had on the inputs.

    With more than one labelled input, xarray before 2025.7 drops coordinate
    attributes when it merges the inputs' coordinates (in place, on the inputs
    too), which loses the CRS stored on ``spatial_ref``. Newer versions keep
    them; this makes both agree. For each coordinate, the attributes of the
    first input that has any win.
    """
    labelled = [o for o in inputs if isinstance(o, (xarray.DataArray, xarray.Dataset))]
    if len(labelled) < 2:
        return out
    fixed = {}
    for name in out.coords:
        if out.coords[name].attrs:
            continue
        attrs = next((o.coords[name].attrs for o in labelled if name in o.coords and o.coords[name].attrs), None)
        if attrs:
            var = out.coords[name].variable.copy(deep=False)
            var.attrs = dict(attrs)
            fixed[name] = var
    return out.assign_coords(fixed) if fixed else out


def _first(a):
    return np.moveaxis(a, -1, 0)


def _last(a):
    return np.moveaxis(a, 0, -1)


def _whole_dim(da, dim):
    if dim not in da.dims:
        raise ValueError(f"dimension {dim!r} not found; dims are {da.dims}")
    return da.chunk({dim: -1}) if da.chunks is not None else da


def _apply_along(func, da, dim, out_dim=None, out_size=None, dtypes=("float64",)):
    """Apply ``func`` to ``da`` with ``dim`` as the last axis of each block.

    ``func`` receives an array whose last axis is ``dim`` and returns one
    array (or a tuple, one per entry of ``dtypes``) whose last axis is the
    new dimension ``out_dim`` of length ``out_size``, or with ``dim``
    removed when ``out_dim`` is None.
    """
    da = _whole_dim(da, dim)
    n_out = len(dtypes)
    core_out = [[_OUT] if out_dim else []] * n_out
    gufunc_kwargs = {"output_sizes": {_OUT: out_size}} if out_dim else {}
    res = xarray.apply_ufunc(
        func,
        da,
        input_core_dims=[[dim]],
        output_core_dims=core_out,
        dask="parallelized",
        output_dtypes=list(dtypes),
        dask_gufunc_kwargs=gufunc_kwargs,
    )
    res = res if isinstance(res, tuple) else (res,)
    if out_dim:
        res = tuple(r.rename({_OUT: out_dim}).transpose(out_dim, ...) for r in res)
    return res if n_out > 1 else res[0]


def _exact(*objs):
    try:
        return xarray.align(*objs, join="exact")
    except ValueError as exc:
        raise ValueError("inputs are not on the same grid/time axis (coordinates differ)") from exc


# --------------------------------------------------------------------------- indices


def normalized_difference(a, b):
    """``(a - b) / (a + b)``; see :func:`eolandscape.indices.normalized_difference`."""
    return _map(_ix.normalized_difference, a, b)


def ndvi(nir, red):
    """NDVI; see :func:`eolandscape.indices.ndvi`."""
    return _map(_ix.ndvi, nir, red).rename("ndvi")


def ndwi_mcfeeters(green, nir):
    """Open-water NDWI; see :func:`eolandscape.indices.ndwi_mcfeeters`."""
    return _map(_ix.ndwi_mcfeeters, green, nir).rename("ndwi_mcfeeters")


def ndwi_gao(nir, swir1):
    """Vegetation-water NDWI; see :func:`eolandscape.indices.ndwi_gao`."""
    return _map(_ix.ndwi_gao, nir, swir1).rename("ndwi_gao")


def ndmi(nir, swir1):
    """NDMI; see :func:`eolandscape.indices.ndmi`."""
    return _map(_ix.ndmi, nir, swir1).rename("ndmi")


def nddi(ndvi_values, ndwi_gao_values):
    """NDDI from NDVI and Gao's NDWI; see :func:`eolandscape.indices.nddi`."""
    return _map(_ix.nddi, ndvi_values, ndwi_gao_values).rename("nddi")


def ndre(nir, rededge):
    """Red-edge index; see :func:`eolandscape.indices.ndre`."""
    return _map(_ix.ndre, nir, rededge).rename("ndre")


#: Band roles per index, in the argument order of the numpy function.
#: ``nir_narrow`` falls back to ``nir`` for sensors without a narrow NIR band.
INDEX_BANDS = {
    "ndvi": ("nir", "red"),
    "ndwi_mcfeeters": ("green", "nir"),
    "ndwi_gao": ("nir_narrow", "swir1"),
    "ndmi": ("nir_narrow", "swir1"),
    "ndre": ("nir_narrow", "rededge"),
}


def spectral_indices(reflectance, names, sensor: str = "sentinel2", bands: dict | None = None):
    """Compute several indices from a Dataset of reflectance bands.

    Parameters
    ----------
    reflectance : xarray.Dataset
        One variable per band, named as in
        :data:`eolandscape.indices.SENSOR_BANDS` for ``sensor`` (e.g.
        ``B04``, ``B08``, ``B8A`` and ``B11`` for Sentinel-2).
    names : list of str
        Any of ``ndvi``, ``ndwi_mcfeeters``, ``ndwi_gao``, ``ndmi``,
        ``ndre`` and ``nddi``.
    sensor : str
        Key of :data:`~eolandscape.indices.SENSOR_BANDS`.
    bands : dict, optional
        Overrides for band roles, e.g. ``{"nir": "nir08"}``.

    Returns
    -------
    xarray.Dataset with one variable per index.
    """
    roles = dict(_ix.SENSOR_BANDS[sensor])
    roles.update(bands or {})
    roles.setdefault("nir_narrow", roles.get("nir"))

    def band(role):
        name = roles.get(role)
        if name is None or name not in reflectance:
            raise KeyError(f"band for role {role!r} ({name!r}) not found in the dataset")
        return reflectance[name]

    out = {}
    for name in names:
        if name == "nddi":
            out[name] = nddi(ndvi(band("nir"), band("red")), ndwi_gao(band("nir_narrow"), band("swir1")))
        elif name in INDEX_BANDS:
            a, b = INDEX_BANDS[name]
            out[name] = _map(getattr(_ix, name), band(a), band(b))
        else:
            raise ValueError(f"unknown index {name!r}")
    return xarray.Dataset(out)


# --------------------------------------------------------------------------- masking


def landsat_c2_reflectance(dn, fill_value: int = 0):
    """See :func:`eolandscape.masking.landsat_c2_reflectance`."""
    return _map(_mk.landsat_c2_reflectance, dn, fill_value=fill_value)


def landsat_qa_clear(qa_pixel, mask_snow: bool = True):
    """See :func:`eolandscape.masking.landsat_qa_clear`."""
    return _map(_mk.landsat_qa_clear, qa_pixel, dtype=bool, mask_snow=mask_snow)


def s2_l2a_reflectance(dn, processing_baseline):
    """Sentinel-2 L2A DN to reflectance, with one baseline per time step.

    Parameters
    ----------
    dn : xarray.DataArray or xarray.Dataset
        Band(s) with a time dimension.
    processing_baseline : float, str or xarray.DataArray
        One value for all data, or a DataArray along ``time`` (e.g. from
        each STAC item's ``s2:processing_baseline``). Strings such as
        ``"05.11"`` are accepted.

    See :func:`eolandscape.masking.s2_l2a_reflectance`.
    """
    if isinstance(processing_baseline, xarray.DataArray):
        baseline = processing_baseline.astype("float64")
    else:
        baseline = float(processing_baseline)
    return _map(_mk.s2_l2a_reflectance, dn, baseline)


def s2_scl_clear(scl, invalid_classes=_mk.S2_SCL_INVALID_DEFAULT):
    """See :func:`eolandscape.masking.s2_scl_clear`."""
    return _map(_mk.s2_scl_clear, scl, dtype=bool, invalid_classes=tuple(invalid_classes)).rename("clear")


def modis_vi_scale(dn, fill_value: int = -3000):
    """See :func:`eolandscape.masking.modis_vi_scale`."""
    return _map(_mk.modis_vi_scale, dn, fill_value=fill_value)


def modis_vi_clear(summary_qa, accept_marginal: bool = False):
    """See :func:`eolandscape.masking.modis_vi_clear`."""
    return _map(_mk.modis_vi_clear, summary_qa, dtype=bool, accept_marginal=accept_marginal)


# --------------------------------------------------------------------------- anomalies

#: Representative month of each season, used to date seasonal composites.
SEASON_MID_MONTH = {"DJF": 1, "MAM": 4, "JJA": 7, "SON": 10}


def seasonal_composites(da, min_months: int = 3, reducer: str = "mean", dim: str = "time"):
    """Seasonal composites of a monthly series.

    Returns a DataArray whose ``dim`` holds one representative date per
    season (the 15th of its middle month; winter 2021 = 15 January 2021), with
    ``season`` and ``season_year`` as extra coordinates. Pass it to
    :func:`climatology` with ``by="season"``.

    See :func:`eolandscape.anomalies.seasonal_composites`.
    """
    dates = da[dim].values
    table, _ = _an.seasonal_composites(np.zeros((len(dates), 1)), dates, min_months, reducer)

    def f(arr):
        return _last(_an.seasonal_composites(_first(arr), dates, min_months, reducer)[1])

    out = _apply_along(f, da, dim, out_dim=dim, out_size=len(table))
    mid = [pd.Timestamp(int(y), SEASON_MID_MONTH[s], 15) for s, y in zip(table["season"], table["season_year"],
                                                                          strict=True)]
    return out.assign_coords(
        {dim: np.asarray(mid, dtype="datetime64[ns]"),
         "season": (dim, table["season"].to_numpy()),
         "season_year": (dim, table["season_year"].to_numpy())}
    )


def climatology(da, baseline: tuple[int, int], by: str = "month", dim: str = "time"):
    """Baseline mean, standard deviation and count per month or season.

    Returns a Dataset with variables ``mean``, ``std`` and ``n`` along a new
    dimension named ``by`` (``"month"`` or ``"season"``), and the baseline in
    its attributes. See :func:`eolandscape.anomalies.climatology`.
    """
    _an._check_baseline(baseline, stacklevel=3)
    dates = da[dim].values
    keys = _an._keys(dates, by)
    _, years = _an._months_years(dates)
    uniq = np.unique(keys)

    def f(arr):
        _, mean, std, n = _an._climatology_arrays(_first(arr), keys, years, baseline)
        return _last(mean), _last(std), _last(n)

    mean, std, n = _apply_along(f, da, dim, out_dim=by, out_size=len(uniq), dtypes=("float64", "float64", "int64"))
    ds = xarray.Dataset({"mean": mean, "std": std, "n": n}).assign_coords({by: uniq})
    ds.attrs.update(baseline_first=int(baseline[0]), baseline_last=int(baseline[1]))
    return ds


def _clim_for(da, clim, dim):
    by = "month" if "month" in clim.dims else "season" if "season" in clim.dims else None
    if by is None:
        raise ValueError("clim must come from eolandscape.xr.climatology (a 'month' or 'season' dimension)")
    keys = _an._keys(da[dim].values, by)
    missing = set(np.unique(keys)) - set(clim[by].values.tolist())
    if missing:
        raise ValueError(f"no baseline values for keys {sorted(missing)}")
    sel = clim.sel({by: xarray.DataArray(keys, dims=dim, coords={dim: da[dim]})}).drop_vars(by)
    return _exact(da, sel)


def anomalies(da, clim, dim: str = "time"):
    """``da`` minus the baseline mean of each time step's month or season."""
    da, sel = _clim_for(da, clim, dim)
    return _map(np.subtract, da.astype("float64"), sel["mean"])


def standardized_anomalies(da, clim, min_std: float = 0.0, min_n: int = 2, dim: str = "time"):
    """See :func:`eolandscape.anomalies.standardized_anomalies` (not for SPEI)."""
    da, sel = _clim_for(da, clim, dim)
    return _map(_an._standardize, da, sel["mean"], sel["std"], sel["n"], min_std=min_std, min_n=min_n)


def period_mean_range(da, select=None, dim: str = "time"):
    """Mean and range along ``dim``; returns a Dataset (``mean``, ``range``)."""
    if select is not None:
        da = da.isel({dim: np.asarray(select, dtype=bool)})

    def f(arr):
        m, r = _an.period_mean_range(_first(arr))
        return m, r

    mean, rng = _apply_along(f, da, dim, dtypes=("float64", "float64"))
    return xarray.Dataset({"mean": mean, "range": rng})


def drought_category(da):
    """McKee et al. (1993) dry categories; see :func:`eolandscape.anomalies.drought_category`."""
    return _map(_an.drought_category, da)


# --------------------------------------------------------------------------- trends


def annual_composite(da, months: tuple[int, int], min_obs: int = 1, reducer: str = "mean", dim: str = "time"):
    """One value per year within a month window; new dimension ``year``.

    See :func:`eolandscape.trends.annual_composite`.
    """
    dates = da[dim].values
    years, _ = _tr.annual_composite(np.zeros((len(dates), 1)), dates, months)

    def f(arr):
        return _last(_tr.annual_composite(_first(arr), dates, months, min_obs, reducer)[1])

    return _apply_along(f, da, dim, out_dim="year", out_size=len(years)).assign_coords(year=years)


def decimal_years(coord) -> np.ndarray:
    """Time coordinate as numbers: datetimes become decimal years, numbers pass through."""
    v = np.asarray(coord)
    if np.issubdtype(v.dtype, np.datetime64):
        idx = pd.DatetimeIndex(v)
        days = np.where(idx.is_leap_year, 366.0, 365.0)
        return idx.year.to_numpy() + (idx.day_of_year.to_numpy() - 1) / days
    return v.astype("float64")


def _trend(func, da, dim, names, **kwargs):
    t = decimal_years(da[dim].values)

    def f(arr):
        return func(_first(arr), t, **kwargs)

    res = _apply_along(f, da, dim, dtypes=("float64",) * len(names))
    out = xarray.Dataset(dict(zip(names, res, strict=True)))
    per_year = np.issubdtype(da[dim].dtype, np.datetime64) or dim == "year"
    out.attrs["slope_unit"] = "per year" if per_year else f"per unit of {dim}"
    return out


def ols_trend(da, dim: str = "year", min_n: int = 10):
    """OLS slope, intercept and R^2 per pixel; slope per unit of ``dim`` (years for datetimes)."""
    return _trend(_tr.ols_trend, da, dim, ("slope", "intercept", "r2"), min_n=min_n)


def theil_sen(da, dim: str = "year", min_n: int = 10):
    """Theil-Sen slope and intercept per pixel; see :func:`eolandscape.trends.theil_sen`."""
    return _trend(_tr.theil_sen, da, dim, ("slope", "intercept"), min_n=min_n)


def mann_kendall(da, dim: str = "year", prewhiten: bool = False, min_n: int = 10):
    """Mann-Kendall ``z`` and two-sided ``p`` per pixel; see :func:`eolandscape.trends.mann_kendall`."""
    return _trend(_tr.mann_kendall, da, dim, ("z", "p"), prewhiten=prewhiten, min_n=min_n)


def benjamini_hochberg(p, alpha: float = 0.05):
    """FDR-controlled significance mask (Benjamini & Hochberg 1995).

    The procedure ranks all p-values of the map together, so a dask-backed
    input is computed here.
    """
    return p.copy(data=_tr.benjamini_hochberg(np.asarray(p.values), alpha)).rename("significant")


# --------------------------------------------------------------------------- fire


def annual_burned(burn_date, dim: str = "time"):
    """Monthly MCD64A1 BurnDate to one burned/not-burned layer per ``year``.

    See :func:`eolandscape.fire.annual_burned`.
    """
    dates = burn_date[dim].values
    years = np.unique(pd.DatetimeIndex(dates).year)

    def f(arr):
        return _last(_fire.annual_burned(_first(arr), dates)[1])

    return _apply_along(f, burn_date, dim, out_dim="year", out_size=len(years)).assign_coords(year=years)


def burn_frequency(annual, dim: str = "year"):
    """Dataset with ``frequency`` and ``n_years_mapped``; see :func:`eolandscape.fire.burn_frequency`."""

    def f(arr):
        return _fire.burn_frequency(_first(arr))

    freq, n = _apply_along(f, annual, dim, dtypes=("float64", "int64"))
    return xarray.Dataset({"frequency": freq, "n_years_mapped": n})


def burned_area(flags, pixel_area_m2, spatial_dims: tuple[str, str] = ("y", "x")):
    """Burned hectares per layer; ``pixel_area_m2`` is a number or a DataArray on the same grid.

    See :func:`eolandscape.fire.burned_area`.
    """
    ydim, xdim = spatial_dims
    if isinstance(pixel_area_m2, xarray.DataArray):
        flags, area = _exact(flags, pixel_area_m2)
        area = area.transpose(ydim, xdim).values
    else:
        area = float(pixel_area_m2)
    if flags.chunks is not None:
        flags = flags.chunk({ydim: -1, xdim: -1})
    return xarray.apply_ufunc(
        _fire.burned_area,
        flags,
        input_core_dims=[[ydim, xdim]],
        kwargs={"pixel_area_m2": area},
        dask="parallelized",
        output_dtypes=["float64"],
    ).rename("burned_area_ha")
