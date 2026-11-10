"""The xarray layer must give exactly the numpy results and keep the labels."""

import numpy as np
import pandas as pd
import pytest

xarray = pytest.importorskip("xarray")

from eolandscape import anomalies as an  # noqa: E402
from eolandscape import fire, indices, masking, trends  # noqa: E402
from eolandscape import xr as exr  # noqa: E402

CRS_ATTRS = {"crs_wkt": 'PROJCS["Amersfoort / RD New"]', "spatial_ref": 'PROJCS["Amersfoort / RD New"]'}


def cube(values, dates, name="v"):
    """A (time, y, x) DataArray on a 10 m north-up grid with a CRS coordinate."""
    t, h, w = values.shape
    return xarray.DataArray(
        values,
        dims=("time", "y", "x"),
        coords={
            "time": pd.to_datetime(dates),
            "y": 100_005.0 - 10 * np.arange(h),
            "x": 50_005.0 + 10 * np.arange(w),
            "spatial_ref": xarray.DataArray(0, attrs=CRS_ATTRS),
        },
        name=name,
    )


def keeps_grid(out, ref):
    assert np.array_equal(out["y"].values, ref["y"].values)
    assert np.array_equal(out["x"].values, ref["x"].values)
    assert out["spatial_ref"].attrs == CRS_ATTRS


@pytest.fixture
def monthly():
    rng = np.random.default_rng(3)
    dates = pd.date_range("1990-01-01", "2021-12-01", freq="MS")
    v = rng.normal(10, 2, (len(dates), 4, 3))
    v[rng.random(v.shape) < 0.05] = np.nan
    return cube(v, dates)


# --------------------------------------------------------------------------- indices and masking


def test_indices_match_numpy_and_keep_grid():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-04-01", periods=3, freq="10D")
    nir, red = cube(rng.uniform(0, 0.5, (3, 4, 5)), dates), cube(rng.uniform(0, 0.5, (3, 4, 5)), dates)
    out = exr.ndvi(nir, red)
    assert out.name == "ndvi" and out.dims == ("time", "y", "x")
    np.testing.assert_array_equal(out.values, indices.ndvi(nir.values, red.values))
    keeps_grid(out, nir)


def test_spectral_indices_use_the_documented_bands():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-04-01", periods=2, freq="10D")
    bands = {b: cube(rng.uniform(0.01, 0.5, (2, 3, 3)), dates) for b in ["B03", "B04", "B05", "B08", "B8A", "B11"]}
    ds = xarray.Dataset(bands)
    out = exr.spectral_indices(ds, ["ndvi", "ndmi", "ndre", "nddi"])
    v = {k: a.values for k, a in bands.items()}
    np.testing.assert_array_equal(out["ndvi"].values, indices.ndvi(v["B08"], v["B04"]))
    np.testing.assert_array_equal(out["ndmi"].values, indices.ndmi(v["B8A"], v["B11"]))
    np.testing.assert_array_equal(out["ndre"].values, indices.ndre(v["B8A"], v["B05"]))
    nddi = indices.nddi(indices.ndvi(v["B08"], v["B04"]), indices.ndwi_gao(v["B8A"], v["B11"]))
    np.testing.assert_array_equal(out["nddi"].values, nddi)
    with pytest.raises(KeyError, match="swir1"):
        exr.spectral_indices(ds.drop_vars("B11"), ["ndmi"])


def test_s2_reflectance_uses_each_dates_baseline():
    dates = pd.to_datetime(["2021-12-20", "2022-02-10", "2022-03-01"])
    dn = cube(np.array([[[1200, 0]], [[1200, 0]], [[1200, 5000]]], dtype="uint16"), dates)
    baseline = xarray.DataArray(["03.01", "04.00", "05.11"], dims="time", coords={"time": dn.time})
    out = exr.s2_l2a_reflectance(dn, baseline)
    expected = np.stack([masking.s2_l2a_reflectance(dn.values[t], baseline.values[t]) for t in range(3)])
    np.testing.assert_array_equal(out.values, expected)
    assert out.values[0, 0, 0] == pytest.approx(0.12) and out.values[1, 0, 0] == pytest.approx(0.02)
    assert np.isnan(out.values[0, 0, 1])  # DN 0 = no data
    keeps_grid(out, dn)


def test_masks_are_boolean():
    dates = pd.date_range("2024-01-01", periods=1)
    scl = cube(np.array([[[4, 9, 6]]]), dates)
    clear = exr.s2_scl_clear(scl)
    assert clear.dtype == bool and clear.values.tolist() == [[[True, False, True]]]
    qa = cube(np.array([[[21824, 1 << 3]]]), dates)
    assert exr.landsat_qa_clear(qa).values.tolist() == [[[True, False]]]


# --------------------------------------------------------------------------- anomalies


def test_seasonal_composites_match_numpy_and_feed_season_climatology(monthly):
    out = exr.seasonal_composites(monthly)
    table, expected = an.seasonal_composites(monthly.values, monthly.time.values)
    np.testing.assert_array_equal(out.values, expected)
    assert out["season"].values.tolist() == table["season"].tolist()
    assert out["season_year"].values.tolist() == table["season_year"].tolist()
    first_winter = out.sel(time=out["season"] == "DJF").time.values[0]
    assert pd.Timestamp(first_winter) == pd.Timestamp("1991-01-15")  # Dec 1990-Feb 1991
    keeps_grid(out, monthly)

    clim = exr.climatology(out, baseline=(1991, 2020), by="season")
    ref = an.climatology(expected, out.time.values, baseline=(1991, 2020), by="season")
    assert clim["season"].values.tolist() == ref.keys.tolist()
    np.testing.assert_array_equal(clim["mean"].values, ref.mean)


def test_climatology_and_anomalies_match_numpy(monthly):
    dates = monthly.time.values
    clim = exr.climatology(monthly, baseline=(1991, 2020))
    ref = an.climatology(monthly.values, dates, baseline=(1991, 2020))
    np.testing.assert_array_equal(clim["mean"].values, ref.mean)
    np.testing.assert_array_equal(clim["std"].values, ref.std)
    np.testing.assert_array_equal(clim["n"].values, ref.n)
    assert clim.attrs == {"baseline_first": 1991, "baseline_last": 2020}

    np.testing.assert_array_equal(exr.anomalies(monthly, clim).values, an.anomalies(monthly.values, dates, ref))
    z = exr.standardized_anomalies(monthly, clim, min_n=20)
    np.testing.assert_array_equal(z.values, an.standardized_anomalies(monthly.values, dates, ref, min_n=20))
    keeps_grid(z, monthly)


def test_short_baseline_warns_once_and_grids_must_match(monthly):
    with pytest.warns(UserWarning, match="30-year"):
        clim = exr.climatology(monthly, baseline=(2001, 2020))
    shifted = monthly.assign_coords(x=monthly.x + 5)  # half a pixel off
    with pytest.raises(ValueError, match="same grid"):
        exr.anomalies(shifted, clim)
    winter_only = exr.climatology(monthly.sel(time=monthly.time.dt.month == 1), baseline=(1991, 2020))
    with pytest.raises(ValueError, match="no baseline values"):
        exr.anomalies(monthly, winter_only)


def test_period_mean_range_and_drought_category(monthly):
    out = exr.period_mean_range(monthly)
    m, r = an.period_mean_range(monthly.values)
    np.testing.assert_array_equal(out["mean"].values, m)
    np.testing.assert_array_equal(out["range"].values, r)
    spei = monthly - 10
    np.testing.assert_array_equal(exr.drought_category(spei).values, an.drought_category(spei.values))


# --------------------------------------------------------------------------- trends


@pytest.fixture
def ndvi_obs():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2001-01-01", "2020-12-31", freq="16D")
    years = dates.year.to_numpy() - 2000
    trend = np.array([0.0, 0.004, -0.006]).reshape(1, 1, 3)
    v = 0.5 + years[:, None, None] * trend + rng.normal(0, 0.02, (len(dates), 2, 3))
    return cube(v, dates)


def test_trend_pipeline_matches_numpy(ndvi_obs):
    annual = exr.annual_composite(ndvi_obs, months=(4, 10), min_obs=3)
    years, expected = trends.annual_composite(ndvi_obs.values, ndvi_obs.time.values, (4, 10), min_obs=3)
    assert annual.dims == ("year", "y", "x") and annual.year.values.tolist() == years.tolist()
    np.testing.assert_array_equal(annual.values, expected)

    ols = exr.ols_trend(annual)
    for got, ref in zip((ols["slope"], ols["intercept"], ols["r2"]), trends.ols_trend(expected, years),
                        strict=True):
        np.testing.assert_allclose(got.values, ref)
    assert ols.attrs["slope_unit"] == "per year"
    ts = exr.theil_sen(annual)
    np.testing.assert_allclose(ts["slope"].values, trends.theil_sen(expected, years)[0])
    mk = exr.mann_kendall(annual, prewhiten=True)
    z, p = trends.mann_kendall(expected, years, prewhiten=True)
    np.testing.assert_allclose(mk["z"].values, z)
    np.testing.assert_allclose(mk["p"].values, p)
    sig = exr.benjamini_hochberg(mk["p"])
    np.testing.assert_array_equal(sig.values, trends.benjamini_hochberg(p))
    keeps_grid(sig, ndvi_obs)


def test_decimal_years():
    t = exr.decimal_years(pd.to_datetime(["2020-01-01", "2020-07-02", "2021-01-01"]).values)
    np.testing.assert_allclose(t, [2020.0, 2020 + 183 / 366, 2021.0])


# --------------------------------------------------------------------------- fire


def test_fire_matches_numpy():
    rng = np.random.default_rng(8)
    dates = pd.date_range("2018-01-01", "2020-12-01", freq="MS")
    bd = rng.choice([0, 0, 0, 0, 120, -1, -2], size=(len(dates), 3, 4)).astype(float)
    da = cube(bd, dates)
    annual = exr.annual_burned(da)
    years, expected = fire.annual_burned(bd, dates)
    assert annual.year.values.tolist() == years.tolist()
    np.testing.assert_array_equal(annual.values, expected)
    freq = exr.burn_frequency(annual)
    ref_f, ref_n = fire.burn_frequency(expected)
    np.testing.assert_array_equal(freq["frequency"].values, ref_f)
    np.testing.assert_array_equal(freq["n_years_mapped"].values, ref_n)
    area = exr.burned_area(annual, 250_000.0)
    np.testing.assert_allclose(area.values, fire.burned_area(expected, 250_000.0))
    assert area.dims == ("year",)


# --------------------------------------------------------------------------- dask


def test_chunked_input_stays_lazy_and_gives_the_same_answer(monthly, ndvi_obs):
    pytest.importorskip("dask")
    lazy = monthly.chunk({"time": 60, "y": 2, "x": 2})
    comp = exr.seasonal_composites(lazy)
    assert comp.chunks is not None  # nothing computed yet
    np.testing.assert_array_equal(comp.values, exr.seasonal_composites(monthly).values)

    clim = exr.climatology(lazy, baseline=(1991, 2020))
    np.testing.assert_array_equal(exr.standardized_anomalies(lazy, clim).values,
                                  exr.standardized_anomalies(monthly, exr.climatology(monthly, (1991, 2020))).values)

    annual = exr.annual_composite(ndvi_obs.chunk({"time": 50, "x": 1}), months=(4, 10))
    mk = exr.mann_kendall(annual)
    assert mk["z"].chunks is not None
    ref = exr.mann_kendall(exr.annual_composite(ndvi_obs, months=(4, 10)))
    np.testing.assert_allclose(mk["z"].values, ref["z"].values)

    dates = pd.to_datetime(["2021-12-20", "2022-02-10"])
    dn = cube(np.full((2, 4, 4), 1500, dtype="uint16"), dates).chunk({"time": 1, "y": 2})
    pb = xarray.DataArray(["03.01", "05.11"], dims="time", coords={"time": dn.time})
    out = exr.s2_l2a_reflectance(dn, pb).values
    assert np.allclose(out[0], 0.15) and np.allclose(out[1], 0.05)
