"""Open satellite data through STAC catalogues (no Earth Engine account).

Optional dependencies: ``pip install eolandscape[stac]`` (pystac-client,
odc-stac, planetary-computer). These functions need network access and are
not covered by the offline test suite.

Catalogues
----------
- ``"planetary-computer"``: https://planetarycomputer.microsoft.com/api/stac/v1
  (e.g. ``sentinel-2-l2a``, ``landsat-c2-l2``, ``modis-13Q1-061``,
  ``modis-64A1-061``). Items are signed automatically.
- ``"earth-search"``: https://earth-search.aws.element84.com/v1
  (e.g. ``sentinel-2-l2a``).

Sentinel-2: check each item's ``s2:processing_baseline`` and pass it to
:func:`eolandscape.masking.s2_l2a_reflectance`, unless the collection
documents that the offset is already removed.
"""

from __future__ import annotations

CATALOGS = {
    "planetary-computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
    "earth-search": "https://earth-search.aws.element84.com/v1",
}


def _require(module: str):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(f"{module} is required: pip install 'eolandscape[stac]'") from exc


def search(collection: str, bbox, start: str, end: str, catalog: str = "planetary-computer", query=None):
    """Search a STAC catalogue and return a list of items.

    Parameters
    ----------
    collection : str
    bbox : (min_lon, min_lat, max_lon, max_lat)
    start, end : str
        ISO dates, inclusive.
    catalog : {"planetary-computer", "earth-search"}
    query : dict, optional
        STAC query, e.g. ``{"eo:cloud_cover": {"lt": 60}}``.
    """
    pystac_client = _require("pystac_client")
    kwargs = {}
    if catalog == "planetary-computer":
        pc = _require("planetary_computer")
        kwargs["modifier"] = pc.sign_inplace
    client = pystac_client.Client.open(CATALOGS[catalog], **kwargs)
    result = client.search(collections=[collection], bbox=bbox, datetime=f"{start}/{end}", query=query)
    return list(result.items())


def load(items, bands, crs: str, resolution: float, bbox=None, chunks=None, **kwargs):
    """Load items into an ``xarray.Dataset`` with odc-stac (lazy if ``chunks``).

    Extra keyword arguments go to ``odc.stac.load`` (e.g.
    ``groupby="solar_day"`` to merge tiles of the same overpass).
    """
    odc_stac = _require("odc.stac")
    return odc_stac.load(items, bands=bands, crs=crs, resolution=resolution, bbox=bbox, chunks=chunks, **kwargs)
