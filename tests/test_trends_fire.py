import numpy as np
import pandas as pd
import pytest
from scipy import stats

from eolandscape import fire, trends


def test_ols_matches_scipy():
    rng = np.random.default_rng(2)
    t = np.arange(2005, 2025)
    y = 0.003 * (t - 2005) + 0.5 + rng.normal(0, 0.01, t.size)
    slope, intercept, r2 = trends.ols_trend(y.reshape(-1, 1), t)
    ref = stats.linregress(t, y)
    assert slope[0] == pytest.approx(ref.slope)
    assert intercept[0] == pytest.approx(ref.intercept)
    assert r2[0] == pytest.approx(ref.rvalue**2)


def test_ols_min_n_and_nan_handling():
    t = np.arange(12)
    y = np.where(t < 5, np.nan, t * 2.0).reshape(-1, 1)
    s, _, _ = trends.ols_trend(y, t, min_n=10)
    assert np.isnan(s[0])
    s, _, _ = trends.ols_trend(y, t, min_n=5)
    assert s[0] == pytest.approx(2.0)


def test_theil_sen_robust_to_outlier():
    t = np.arange(15, dtype=float)
    y = 2 * t + 1
    y[3] = 500
    s, b = trends.theil_sen(y.reshape(-1, 1), t)
    assert s[0] == pytest.approx(2.0)
    assert b[0] == pytest.approx(1.0)
    ref = stats.theilslopes(y, t)
    assert s[0] == pytest.approx(ref.slope)


def test_mann_kendall_against_reference_values():
    # Hand-checked: strictly increasing n=10 -> S=45, var=125, z=(45-1)/sqrt(125)
    s, z, p = trends.mann_kendall_1d(np.arange(10.0))
    assert s == 45
    assert z == pytest.approx(44 / np.sqrt(125))
    assert p == pytest.approx(2 * (1 - stats.norm.cdf(44 / np.sqrt(125))))


def test_mann_kendall_ties_reduce_variance():
    y = np.array([1, 1, 2, 3, 3, 3, 4, 5, 6, 6], dtype=float)
    s, z, _ = trends.mann_kendall_1d(y)
    n = 10
    ties = 2 * 1 * 9 + 3 * 2 * 11 + 2 * 1 * 9
    var = (n * (n - 1) * (2 * n + 5) - ties) / 18
    assert z == pytest.approx((s - 1) / np.sqrt(var))


def test_mann_kendall_no_trend_and_grid():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(20, 4, 4))
    z, p = trends.mann_kendall(y, np.arange(20))
    assert z.shape == (4, 4) and np.all((p >= 0) & (p <= 1))


def test_prewhitening_runs_and_keeps_strong_trend():
    t = np.arange(25, dtype=float)
    y = 0.5 * t + np.sin(t)
    _, _, p = trends.mann_kendall_1d(y, t, prewhiten=True)
    assert p < 0.01


def test_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, np.nan])
    sig = trends.benjamini_hochberg(p, alpha=0.05)
    # BH with m=8: thresholds k*0.05/8 -> 0.00625, 0.0125, 0.01875, 0.025 ...; only first two pass
    assert sig.tolist() == [True, True, False, False, False, False, False, False, False]


def test_annual_composite_window_and_min_obs():
    dates = pd.date_range("2020-01-01", "2021-12-01", freq="MS")
    vals = np.arange(len(dates), dtype=float).reshape(-1, 1)
    years, comp = trends.annual_composite(vals, dates, months=(4, 10), min_obs=7)
    assert years.tolist() == [2020, 2021]
    assert comp[0, 0] == pytest.approx(np.mean(np.arange(3, 10)))
    _, comp = trends.annual_composite(vals, dates, months=(4, 10), min_obs=8)
    assert np.all(np.isnan(comp))


def test_fire_flags_never_sum_doy_and_handle_unmapped():
    dates = pd.date_range("2001-01-01", "2002-12-01", freq="MS")
    bd = np.zeros((24, 1, 3))
    bd[5, 0, 0] = 180        # burned in 2001
    bd[:, 0, 1] = -1         # unmapped all period
    bd[:12, 0, 2] = -1       # unmapped 2001, mapped & burned 2002
    bd[13, 0, 2] = 40
    bd[14, 0, 0] = -1        # an unmapped month must not cancel anything
    years, annual = fire.annual_burned(bd, dates)
    assert years.tolist() == [2001, 2002]
    assert annual[:, 0, 0].tolist() == [1.0, 0.0]
    assert np.all(np.isnan(annual[:, 0, 1]))
    assert np.isnan(annual[0, 0, 2]) and annual[1, 0, 2] == 1.0
    freq, n = fire.burn_frequency(annual)
    assert freq[0].tolist()[0] == 1 and np.isnan(freq[0, 1]) and freq[0, 2] == 1
    assert n[0].tolist() == [2, 0, 1]


def test_burned_area_hectares():
    flags = np.array([[[1, 0], [1, 1]], [[0, 0], [0, 1]]], dtype=float)
    ha = fire.burned_area(flags, 250_000.0)  # 500 m pixels
    assert ha.tolist() == [75.0, 25.0]
