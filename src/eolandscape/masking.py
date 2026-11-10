"""Scaling and cloud masking for Landsat Collection 2, Sentinel-2 L2A and MODIS.

All functions return ``float64`` reflectance with NaN for fill values, and
boolean "clear" masks where ``True`` means the pixel is usable.

References
----------
USGS. Landsat 8-9 Collection 2 Level-2 Science Product Guide (scale factor
    0.0000275, offset -0.2; QA_PIXEL bit layout).
ESA. Sentinel-2 Level-2A Product Specification / Sen2Cor (Scene
    Classification Layer classes; BOA_ADD_OFFSET of -1000 from processing
    baseline 04.00, January 2022).
Didan, K. (2021). MOD13Q1 MODIS/Terra Vegetation Indices 16-Day L3 Global
    250m SIN Grid V061 [Dataset]. NASA LP DAAC.
    https://doi.org/10.5067/MODIS/MOD13Q1.061
Skakun, S. et al. (2022). Cloud Mask Intercomparison eXercise (CMIX).
    Remote Sensing of Environment 274, 112990.
    https://doi.org/10.1016/j.rse.2022.112990 (known limits of SCL masking)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "LANDSAT_QA_BITS",
    "S2_SCL_CLASSES",
    "S2_SCL_INVALID_DEFAULT",
    "landsat_c2_reflectance",
    "landsat_qa_clear",
    "s2_l2a_reflectance",
    "s2_scl_clear",
    "modis_vi_scale",
    "modis_vi_clear",
]

#: QA_PIXEL bit positions used for masking (Landsat Collection 2).
LANDSAT_QA_BITS = {
    "fill": 0,
    "dilated_cloud": 1,
    "cirrus": 2,
    "cloud": 3,
    "cloud_shadow": 4,
    "snow": 5,
}

#: Sentinel-2 L2A Scene Classification Layer classes.
S2_SCL_CLASSES = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "dark_area_pixels",
    3: "cloud_shadows",
    4: "vegetation",
    5: "not_vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud_medium_probability",
    9: "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow",
}

#: SCL classes treated as unusable by default.
S2_SCL_INVALID_DEFAULT = (0, 1, 3, 8, 9, 10, 11)


def landsat_c2_reflectance(dn, fill_value: int = 0):
    """Convert Landsat C2 L2 surface-reflectance DN to reflectance.

    ``reflectance = DN * 0.0000275 - 0.2``; ``fill_value`` becomes NaN.
    """
    dn = np.asarray(dn)
    out = dn.astype("float64") * 0.0000275 - 0.2
    out[dn == fill_value] = np.nan
    return out


def landsat_qa_clear(qa_pixel, mask_snow: bool = True):
    """Return ``True`` where QA_PIXEL flags no fill, cloud, cirrus or shadow.

    Bits tested: fill (0), dilated cloud (1), cirrus (2), cloud (3), cloud
    shadow (4) and, if ``mask_snow``, snow (5).
    """
    qa = np.asarray(qa_pixel).astype("int64")
    bits = ["fill", "dilated_cloud", "cirrus", "cloud", "cloud_shadow"]
    if mask_snow:
        bits.append("snow")
    bad = np.zeros(qa.shape, dtype=bool)
    for name in bits:
        bad |= (qa >> LANDSAT_QA_BITS[name]) & 1 == 1
    return ~bad


def s2_l2a_reflectance(dn, processing_baseline: float | str):
    """Convert Sentinel-2 L2A DN to bottom-of-atmosphere reflectance.

    From processing baseline 04.00 (25 January 2022) ESA adds an offset of
    1000 to L2A DN, so ``reflectance = (DN - 1000) / 10000``; before that
    ``reflectance = DN / 10000``. ``DN == 0`` is no data and becomes NaN.

    Parameters
    ----------
    dn : array_like
        Digital numbers of one band.
    processing_baseline : float, str or array_like
        The product's processing baseline, e.g. ``"04.00"`` or ``5.09``, or
        an array of baselines that broadcasts against ``dn`` (e.g. shape
        ``(T, 1, 1)`` for a time stack mixing products from before and after
        January 2022). Some catalogues already remove the offset
        ("harmonized" collections); in that case pass a baseline below 4 so
        no offset is applied.
    """
    baseline = np.asarray(processing_baseline).astype("float64")
    dn = np.asarray(dn)
    offset = np.where(baseline >= 4.0, -1000.0, 0.0)
    out = (dn.astype("float64") + offset) / 10000.0
    return np.where(dn == 0, np.nan, out)


def s2_scl_clear(scl, invalid_classes=S2_SCL_INVALID_DEFAULT):
    """Return ``True`` where the SCL class is not in ``invalid_classes``."""
    scl = np.asarray(scl)
    return ~np.isin(scl, np.asarray(invalid_classes))


def modis_vi_scale(dn, fill_value: int = -3000):
    """Scale MOD13 NDVI/EVI DN by 0.0001; ``fill_value`` becomes NaN."""
    dn = np.asarray(dn)
    out = dn.astype("float64") * 0.0001
    out[dn == fill_value] = np.nan
    return out


def modis_vi_clear(summary_qa, accept_marginal: bool = False):
    """Return ``True`` for usable MOD13 pixels from the SummaryQA layer.

    SummaryQA: 0 good, 1 marginal, 2 snow/ice, 3 cloudy. Only 0 is kept
    unless ``accept_marginal`` is ``True``.
    """
    qa = np.asarray(summary_qa)
    keep = (0, 1) if accept_marginal else (0,)
    return np.isin(qa, keep)
