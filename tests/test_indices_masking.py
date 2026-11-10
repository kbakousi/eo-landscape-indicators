import numpy as np
import pytest

from eolandscape import indices, masking


def test_normalized_difference_basic_and_zero_denominator():
    out = indices.normalized_difference([0.5, 0.0, np.nan], [0.1, 0.0, 0.2])
    assert out[0] == pytest.approx(0.4 / 0.6)
    assert np.isnan(out[1])  # zero denominator -> NaN, never inf
    assert np.isnan(out[2])


def test_ndvi_bounds():
    rng = np.random.default_rng(0)
    nir, red = rng.uniform(0, 1, 1000), rng.uniform(0, 1, 1000)
    v = indices.ndvi(nir, red)
    assert np.nanmin(v) >= -1 and np.nanmax(v) <= 1


def test_two_ndwis_are_different_formulas():
    green, nir, swir1 = 0.08, 0.35, 0.20
    assert indices.ndwi_mcfeeters(green, nir) == pytest.approx((green - nir) / (green + nir))
    assert indices.ndwi_gao(nir, swir1) == pytest.approx((nir - swir1) / (nir + swir1))
    assert indices.ndmi(nir, swir1) == indices.ndwi_gao(nir, swir1)


def test_nddi_uses_gao_ndwi():
    nir, red, swir1 = 0.35, 0.05, 0.20
    v = indices.ndvi(nir, red)
    w = indices.ndwi_gao(nir, swir1)
    assert indices.nddi(v, w) == pytest.approx((v - w) / (v + w))


def test_s2_offset_by_processing_baseline():
    dn = np.array([0, 1000, 3000])
    old = masking.s2_l2a_reflectance(dn, "03.01")
    new = masking.s2_l2a_reflectance(dn, "04.00")
    assert np.isnan(old[0]) and np.isnan(new[0])
    assert old[2] == pytest.approx(0.3)
    assert new[2] == pytest.approx(0.2)
    assert new[1] == pytest.approx(0.0)


def test_scl_mask_default_classes():
    scl = np.arange(12)
    clear = masking.s2_scl_clear(scl)
    assert clear.tolist() == [False, False, True, False, True, True, True, True, False, False, False, False]


def test_landsat_qa_bits():
    qa = np.array([0, 1 << 3, 1 << 4, 1 << 5, 1 << 6])  # clear, cloud, shadow, snow, clear-flag bit
    assert masking.landsat_qa_clear(qa).tolist() == [True, False, False, False, True]
    assert masking.landsat_qa_clear(qa, mask_snow=False).tolist() == [True, False, False, True, True]


def test_landsat_scaling_and_fill():
    out = masking.landsat_c2_reflectance(np.array([0, 7273 + 1]))
    assert np.isnan(out[0])
    assert out[1] == pytest.approx((7274) * 0.0000275 - 0.2)


def test_modis_vi_qa():
    assert masking.modis_vi_clear([0, 1, 2, 3]).tolist() == [True, False, False, False]
    assert masking.modis_vi_clear([0, 1, 2, 3], accept_marginal=True).tolist() == [True, True, False, False]
    assert np.isnan(masking.modis_vi_scale([-3000])[0])
