"""Spectral indices from surface reflectance.

Every index is a normalized difference ``(a - b) / (a + b)``. Where the
denominator is zero or an input is not finite, the result is NaN; values are
never clipped.

References
----------
Rouse, J.W., Haas, R.H., Schell, J.A., Deering, D.W. (1974). Monitoring
    vegetation systems in the Great Plains with ERTS. Third ERTS Symposium,
    NASA SP-351, 309-317.
Tucker, C.J. (1979). Red and photographic infrared linear combinations for
    monitoring vegetation. Remote Sensing of Environment 8, 127-150.
    https://doi.org/10.1016/0034-4257(79)90013-0
McFeeters, S.K. (1996). The use of the Normalized Difference Water Index
    (NDWI) in the delineation of open water features. International Journal
    of Remote Sensing 17, 1425-1432. https://doi.org/10.1080/01431169608948714
Gao, B.-C. (1996). NDWI - a normalized difference water index for remote
    sensing of vegetation liquid water from space. Remote Sensing of
    Environment 58, 257-266. https://doi.org/10.1016/S0034-4257(96)00067-3
Wilson, E.H., Sader, S.A. (2002). Detection of forest harvest type using
    multiple dates of Landsat TM imagery. Remote Sensing of Environment 80,
    385-396. https://doi.org/10.1016/S0034-4257(01)00318-2
Gu, Y., Brown, J.F., Verdin, J.P., Wardlow, B. (2007). A five-year analysis
    of MODIS NDVI and NDWI for grassland drought assessment over the central
    Great Plains of the United States. Geophysical Research Letters 34,
    L06407. https://doi.org/10.1029/2006GL029127
Gitelson, A., Merzlyak, M.N. (1994). Spectral reflectance changes associated
    with autumn senescence of Aesculus hippocastanum L. and Acer platanoides
    L. leaves. Journal of Plant Physiology 143, 286-292.
    https://doi.org/10.1016/S0176-1617(11)81633-0
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SENSOR_BANDS",
    "normalized_difference",
    "ndvi",
    "ndwi_mcfeeters",
    "ndwi_gao",
    "ndmi",
    "nddi",
    "ndre",
]

#: Band names per sensor for the generic band roles used in this module.
#: Landsat names follow Collection 2 Level-2; MODIS names follow MOD09/MCD43.
SENSOR_BANDS: dict[str, dict[str, str]] = {
    "sentinel2": {
        "green": "B03",
        "red": "B04",
        "rededge": "B05",
        "nir": "B08",
        "nir_narrow": "B8A",
        "swir1": "B11",
    },
    "landsat89": {"green": "SR_B3", "red": "SR_B4", "nir": "SR_B5", "swir1": "SR_B6"},
    "landsat457": {"green": "SR_B2", "red": "SR_B3", "nir": "SR_B4", "swir1": "SR_B5"},
    "modis": {
        "red": "sur_refl_b01",
        "nir": "sur_refl_b02",
        "green": "sur_refl_b04",
        "swir1": "sur_refl_b06",
    },
}


def normalized_difference(a, b, eps: float = 1e-12):
    """Return ``(a - b) / (a + b)``, with NaN where ``|a + b| <= eps``.

    Parameters
    ----------
    a, b : array_like
        Reflectance arrays of the same shape (or broadcastable).
    eps : float
        Denominators with absolute value at or below ``eps`` give NaN.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    num = a - b
    den = a + b
    out = np.full(np.broadcast(a, b).shape, np.nan)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > eps)
    np.divide(num, den, out=out, where=ok)
    return out


def ndvi(nir, red):
    """Normalized Difference Vegetation Index, (NIR - Red) / (NIR + Red).

    Rouse et al. (1974); Tucker (1979).
    """
    return normalized_difference(nir, red)


def ndwi_mcfeeters(green, nir):
    """NDWI for open water, (Green - NIR) / (Green + NIR). McFeeters (1996).

    Measures open water. It is *not* the vegetation-water index used in NDDI.
    """
    return normalized_difference(green, nir)


def ndwi_gao(nir, swir1):
    """NDWI for vegetation liquid water, (NIR - SWIR1) / (NIR + SWIR1). Gao (1996).

    For Sentinel-2 use B8A and B11 (both 20 m); for Landsat 8-9 SR_B5 and
    SR_B6; for MODIS bands 2 and 6, as in Gu et al. (2007).
    """
    return normalized_difference(nir, swir1)


def ndmi(nir, swir1):
    """Normalized Difference Moisture Index; same formula as :func:`ndwi_gao`.

    The name NDMI is from Wilson & Sader (2002).
    """
    return ndwi_gao(nir, swir1)


def nddi(ndvi_values, ndwi_gao_values):
    """Normalized Difference Drought Index, (NDVI - NDWI) / (NDVI + NDWI).

    Gu et al. (2007). ``ndwi_gao_values`` must be Gao's NIR-SWIR NDWI, not
    McFeeters' Green-NIR NDWI. Returns NaN where NDVI + NDWI is zero.
    """
    return normalized_difference(ndvi_values, ndwi_gao_values)


def ndre(nir, rededge):
    """Normalized Difference Red-Edge index, (NIR - RE) / (NIR + RE).

    Gitelson & Merzlyak (1994). For Sentinel-2 use B8A (or B08) and B05.
    """
    return normalized_difference(nir, rededge)
