"""Land-cover composition: area and share per class or per user-defined group.

No class grouping (e.g. "habitat") ships with this module: groupings and the
domain used as denominator are user inputs.

References
----------
Kosztra, B., Buttner, G., Hazeu, G., Arnold, S. (2019). Updated CLC
    illustrated nomenclature guidelines. European Environment Agency.
Data: Copernicus Land Monitoring Service, CORINE Land Cover 2018. Minimum
    mapping unit 25 ha; the 100 m raster is a rasterised vector product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["CLC_NOMENCLATURE", "class_area", "class_share"]

#: The 44 CORINE Land Cover level-3 classes (reference table only).
CLC_NOMENCLATURE: dict[int, str] = {
    111: "Continuous urban fabric",
    112: "Discontinuous urban fabric",
    121: "Industrial or commercial units",
    122: "Road and rail networks and associated land",
    123: "Port areas",
    124: "Airports",
    131: "Mineral extraction sites",
    132: "Dump sites",
    133: "Construction sites",
    141: "Green urban areas",
    142: "Sport and leisure facilities",
    211: "Non-irrigated arable land",
    212: "Permanently irrigated land",
    213: "Rice fields",
    221: "Vineyards",
    222: "Fruit trees and berry plantations",
    223: "Olive groves",
    231: "Pastures",
    241: "Annual crops associated with permanent crops",
    242: "Complex cultivation patterns",
    243: "Land principally occupied by agriculture with significant areas of natural vegetation",
    244: "Agro-forestry areas",
    311: "Broad-leaved forest",
    312: "Coniferous forest",
    313: "Mixed forest",
    321: "Natural grasslands",
    322: "Moors and heathland",
    323: "Sclerophyllous vegetation",
    324: "Transitional woodland-shrub",
    331: "Beaches, dunes, sands",
    332: "Bare rocks",
    333: "Sparsely vegetated areas",
    334: "Burnt areas",
    335: "Glaciers and perpetual snow",
    411: "Inland marshes",
    412: "Peat bogs",
    421: "Salt marshes",
    422: "Salines",
    423: "Intertidal flats",
    511: "Water courses",
    512: "Water bodies",
    521: "Coastal lagoons",
    522: "Estuaries",
    523: "Sea and ocean",
}


def _area_grid(classes, pixel_area_m2):
    return np.broadcast_to(np.asarray(pixel_area_m2, dtype="float64"), np.shape(classes))


def class_area(classes, pixel_area_m2, groups: dict | None = None, nodata=None):
    """Area (ha) per class, or per group if ``groups`` maps class -> group.

    Parameters
    ----------
    classes : array_like of int
        Land-cover raster.
    pixel_area_m2 : float or array_like
        True pixel area in m^2 (use an equal-area grid such as EPSG:3035).
    groups : dict, optional
        ``{class_code: group_name}``. Classes missing from the dict are
        reported under ``"unassigned"``.
    nodata : int, optional
        Class value to ignore.

    Returns
    -------
    pandas.DataFrame with columns ``key`` and ``area_ha``.
    """
    c = np.asarray(classes).ravel()
    a = _area_grid(classes, pixel_area_m2).ravel()
    keep = np.ones(c.shape, dtype=bool) if nodata is None else c != nodata
    df = pd.DataFrame({"cls": c[keep], "area": a[keep]})
    if groups is not None:
        df["key"] = df["cls"].map(groups).fillna("unassigned")
    else:
        df["key"] = df["cls"]
    res = df.groupby("key", sort=True)["area"].sum() / 10_000.0
    return res.rename("area_ha").reset_index()


def class_share(classes, pixel_area_m2, domain_classes, groups: dict | None = None):
    """Share (%) of each class or group within an explicit domain.

    ``domain_classes`` lists the class codes forming the denominator (e.g.
    terrestrial classes only, or all classes including water). It is
    required, so the denominator is always stated.
    """
    c = np.asarray(classes)
    in_domain = np.isin(c, np.asarray(list(domain_classes)))
    if not in_domain.any():
        raise ValueError("no pixels of domain_classes present")
    area = _area_grid(c, pixel_area_m2)
    total_ha = area[in_domain].sum() / 10_000.0
    tab = class_area(np.where(in_domain, c, -999999), area, groups=groups, nodata=-999999)
    tab["share_pct"] = 100.0 * tab["area_ha"] / total_ha
    return tab
