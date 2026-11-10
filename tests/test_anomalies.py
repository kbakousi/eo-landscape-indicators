import warnings

import numpy as np
import pandas as pd
import pytest

from eolandscape import anomalies as an


def monthly(start, end):
    return pd.date_range(start, end, freq="MS")


def test_december_belongs_to_next_winter():
    s, y = an.season_of(np.array(["2020-12-01", "2021-01-01", "2021-03-01"], dtype="datetime64[D]"))
    assert s.tolist() == ["DJF", "DJF", "MAM"]
    assert y.tolist() == [2021, 2021, 2021]


def test_seasonal_composite_requires_min_months():
    dates = monthly("2020-12-01", "2021-05-01")  # DJF 2021 complete, MAM 2021 complete
    vals = np.arange(6, dtype=float).reshape(6, 1)
    vals[4, 0] = np.nan  # April missing
    table, comp = an.seasonal_composites(vals, dates, min_months=3)
    got = dict(zip(table["season"], comp[:, 0], strict=True))
    assert got["DJF"] == pytest.approx(1.0)
    assert np.isnan(got["MAM"])
    _, comp2 = an.seasonal_composites(vals, dates, min_months=2)
    assert comp2[1, 0] == pytest.approx((3 + 5) / 2)


def test_partial_season_at_series_edge_is_dropped():
    dates = monthly("2021-01-01", "2021-03-01")  # Jan, Feb (DJF partial) + Mar
    table, _ = an.seasonal_composites(np.ones((3, 1)), dates, min_months=3)
    assert table.empty


def test_climatology_warns_for_short_baseline_and_standardizes():
    dates = monthly("1991-01-01", "2020-12-01")
    rng = np.random.default_rng(1)
    vals = rng.normal(10, 2, size=(len(dates), 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clim = an.climatology(vals, dates, baseline=(1991, 2020))  # 30 years: no warning
    z = an.standardized_anomalies(vals, dates, clim)
    jan = pd.DatetimeIndex(dates).month == 1
    assert np.allclose(np.nanmean(z[jan], axis=0), 0, atol=1e-12)
    assert np.allclose(np.nanstd(z[jan], axis=0, ddof=1), 1, atol=1e-12)
    with pytest.warns(UserWarning, match="30-year"):
        an.climatology(vals, dates, baseline=(2001, 2020))


def test_zero_std_gives_nan_not_inflated_value():
    dates = monthly("2000-01-01", "2000-12-01")
    vals = np.ones((12, 1))
    with pytest.warns(UserWarning):
        clim = an.climatology(np.vstack([vals, vals]), monthly("2000-01-01", "2001-12-01"), baseline=(2000, 2001))
    z = an.standardized_anomalies(vals + 1, dates, clim)
    assert np.all(np.isnan(z))


def test_anomaly_against_month_mean():
    dates = monthly("2000-01-01", "2001-12-01")
    vals = np.tile(np.arange(12, dtype=float), 2).reshape(-1, 1)
    vals[12:] += 1.0
    with pytest.warns(UserWarning):
        clim = an.climatology(vals, dates, baseline=(2000, 2001))
    a = an.anomalies(vals, dates, clim)
    assert np.allclose(a[:12], -0.5) and np.allclose(a[12:], 0.5)


def test_drought_categories_mckee():
    cat = an.drought_category([0.0, -1.0, -1.49, -1.5, -1.99, -2.0, np.nan])
    assert cat[:6].tolist() == [0, 1, 1, 2, 2, 3]
    assert np.isnan(cat[6])


def test_period_mean_range():
    m, r = an.period_mean_range(np.array([[1.0], [3.0], [np.nan]]))
    assert m[0] == 2.0 and r[0] == 2.0
