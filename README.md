# eo-landscape-indicators

[![Tests](https://github.com/kbakousi/eo-landscape-indicators/actions/workflows/ci.yml/badge.svg)](https://github.com/kbakousi/eo-landscape-indicators/actions/workflows/ci.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24118750.svg)](https://doi.org/10.5281/zenodo.24118750)

Earth-observation landscape indicators from open satellite and climate data: spectral indices, cloud masking, climate anomalies, trend tests, burned-area frequency, land-cover composition, landscape diversity and fragmentation.

- **Methods follow the literature.** Each module is an independent implementation of the methods described in the publications cited in its docstring (see [References](#references)).
- **numpy core, labelled interface.** Every calculation is a plain `numpy` function (time-first, `(T, ...)`). The `eolandscape.xr` module applies the same functions to `xarray` objects, keeps time, grid and CRS coordinates, and stays lazy on dask-backed data, so results are identical either way (tested).
- **No Earth Engine account needed.** Optional loaders read open STAC catalogues (Microsoft Planetary Computer, Element84 Earth Search).
- **No hidden choices.** Windows, thresholds, class groupings and baselines are explicit parameters. None are hard-coded.

## Install

```bash
pip install -e .            # core: numpy, pandas, scipy
pip install -e ".[xarray]"  # + labelled interface (xarray)
pip install -e ".[stac]"    # + STAC loaders (pystac-client, odc-stac, planetary-computer)
pip install -e ".[dev]"     # + pytest, ruff, xarray, dask
```

## Modules

| Module | What it does | Method references |
|---|---|---|
| `indices` | NDVI, NDWI (McFeeters, open water), NDWI/NDMI (Gao, vegetation water), NDDI, NDRE, with band maps for Sentinel-2, Landsat and MODIS | Rouse 1974; Tucker 1979; McFeeters 1996; Gao 1996; Wilson & Sader 2002; Gu et al. 2007; Gitelson & Merzlyak 1994 |
| `masking` | Landsat C2 scaling + QA_PIXEL; Sentinel-2 L2A offset by processing baseline + SCL; MODIS VI scaling + SummaryQA | USGS and ESA product guides; Didan 2021; Skakun et al. 2022 |
| `anomalies` | Seasonal composites (DJF/MAM/JJA/SON), baseline climatology, anomalies, standardized anomalies, SPI/SPEI drought categories | WMO 2017; McKee et al. 1993; Vicente-Serrano et al. 2010; Thomas et al. 2014 |
| `trends` | Annual composites, OLS, Theil–Sen, Mann–Kendall (with tie correction and optional pre-whitening), Benjamini–Hochberg FDR | Mann 1945; Kendall 1975; Sen 1968; Yue et al. 2002; Benjamini & Hochberg 1995; Wilks 2016; de Jong et al. 2011 |
| `fire` | Annual burned flags, burn frequency and burned area from MODIS MCD64A1 | Giglio et al. 2018 |
| `landcover` | Area and share per class or group within an explicit domain; CORINE nomenclature table | Kosztra et al. 2019 |
| `diversity` | Class proportions in a circular window; Shannon, Simpson, richness, Pielou evenness, Hill numbers | Shannon 1948; Simpson 1949; Pielou 1966; Hill 1973; McGarigal & Marks 1995 |
| `fragmentation` | Distance to edge (in metres), core/edge split, Riitters Pf/Pff, edge density | Riitters et al. 2000; Vogt et al. 2007; Haddad et al. 2015; McGarigal & Marks 1995 |
| `zonal` | Per-zone count, valid fraction, mean (optionally area-weighted), median, std, min, max | — |
| `xr` | The indices, masking, anomaly, trend and fire functions on `xarray` objects: labels kept, lazy with dask, grids checked | Same as the wrapped modules |
| `loaders.stac` | Search and load STAC items (optional, needs network) | — |

## Quick start

```python
import numpy as np
import pandas as pd
from eolandscape import indices, masking, trends, anomalies

# Sentinel-2 L2A: DN -> reflectance (offset depends on processing baseline) -> NDVI on clear pixels
red = masking.s2_l2a_reflectance(dn_b04, processing_baseline="05.09")
nir = masking.s2_l2a_reflectance(dn_b08, processing_baseline="05.09")
ndvi = np.where(masking.s2_scl_clear(scl), indices.ndvi(nir, red), np.nan)

# Growing-season NDVI trend: Theil-Sen slope + Mann-Kendall, FDR-corrected significance
years, annual = trends.annual_composite(ndvi_stack, dates, months=(4, 10), min_obs=3)
slope, _ = trends.theil_sen(annual, years, min_n=10)
z, p = trends.mann_kendall(annual, years, min_n=10)
significant = trends.benjamini_hochberg(p, alpha=0.05)

# Monthly soil-moisture standardized anomalies against a 30-year baseline
clim = anomalies.climatology(sm, sm_dates, baseline=(1991, 2020), by="month")
z_sm = anomalies.standardized_anomalies(sm, sm_dates, clim)
```

### With xarray (labelled, lazy)

```python
from eolandscape import xr as exr
from eolandscape.loaders import stac

items = stac.search("sentinel-2-l2a", bbox, "2025-03-01", "2025-10-31")
ds = stac.load(items, ["B04", "B08", "B8A", "B11", "SCL"], crs="EPSG:28992", resolution=10, bbox=bbox,
               chunks={"x": 1024, "y": 1024})                     # lazy (dask)

refl = exr.s2_l2a_reflectance(ds[["B04", "B08", "B8A", "B11"]], baseline)  # baseline: DataArray along time
cube = exr.spectral_indices(refl, ["ndvi", "ndmi"]).where(exr.s2_scl_clear(ds["SCL"]))

annual = exr.annual_composite(cube["ndvi"], months=(4, 10), min_obs=3)    # dims (year, y, x)
mk = exr.mann_kendall(annual)                                             # Dataset: z, p
significant = exr.benjamini_hochberg(mk["p"])
```

Coordinates (time, y, x and the `spatial_ref` CRS) are kept at every step. Inputs that must share a grid, such as values and a climatology, are aligned with `join="exact"`, so data on a shifted grid raises an error and is never combined silently. Functions that work along time rechunk time into one chunk and keep space chunked. The neighbourhood metrics (`diversity`, `fragmentation`) are numpy-only for now, because they need overlapping chunks.

## Real-data examples

Both examples read open MODIS data from Microsoft Planetary Computer. They need network access and `pip install -e ".[stac]"`, but no account.

| Script | Area | What it computes |
|---|---|---|
| [`examples/demo_modis_ndvi_trend.py`](examples/demo_modis_ndvi_trend.py) | Naoussa, northern Greece (about 13 × 12 km) | MOD13Q1 NDVI with pixel-reliability mask; April–October composites for 2001–2024; Theil–Sen slope; Mann–Kendall with FDR |
| [`examples/demo_mcd64_burned_area.py`](examples/demo_mcd64_burned_area.py) | Dadia forest, Evros (about 38 × 33 km) | MCD64A1 burned hectares per year and burn frequency on an equal-area grid. The 2023 fire is a built-in sanity check |

## Tutorial

[`tutorials/01_trends_and_anomalies.ipynb`](tutorials/01_trends_and_anomalies.ipynb) runs offline in about a minute.
It builds a synthetic NDVI record with known greening, browning and a drought, and shows that the methods recover them:

- Theil–Sen slopes match the true trends;
- raw p < 0.05 gives 8.5 % false detections, Benjamini–Hochberg 2.7 %;
- a 30-year climatology shows the drought as standardized anomalies;
- the chunked (dask) run gives identical results.

## Scientific notes

- **Two NDWIs.** NDDI uses Gao's NIR–SWIR NDWI (Gu et al. 2007). McFeeters' Green–NIR NDWI maps open water and must not be used in NDDI.
- **Greening and browning** are judged by significance (Mann–Kendall + FDR), not by a fixed slope threshold (de Jong et al. 2011; Wilks 2016).
- **SPEI is already standardized.** Use `drought_category`, not `standardized_anomalies`, on SPEI.
- **Baselines.** A baseline shorter than 30 years triggers a warning (WMO 2017). Missing months, e.g. the GRACE / GRACE-FO gap, stay missing and are never filled silently.
- **Areas** need true pixel areas or an equal-area grid (e.g. EPSG:3035 for Europe). Distances use the real pixel size.
- **Landscape diversity** here is land-cover diversity, a structural proxy, not species diversity.
- **MCD64A1** under-detects small burns at 500 m (Giglio et al. 2018).

## Tests

```bash
pytest
```

The tests run offline on small synthetic arrays and check the formulas against reference values: `scipy.stats.linregress` and `theilslopes`, hand-computed Mann–Kendall statistics, Benjamini–Hochberg, and the mathematical bounds of the diversity indices. The `xr` tests check that every wrapped function returns exactly the numpy result, keeps the grid and CRS coordinates, gives the same answer on dask-chunked input, and refuses misaligned grids.

The tutorial in `tutorials/` is executed as part of the test suite (`tests/test_tutorials.py`), so it cannot drift out of date with the code.

## References

- Benjamini, Y., Hochberg, Y. (1995). *JRSS B* 57, 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- de Jong, R. et al. (2011). *Remote Sensing of Environment* 115, 692–702. https://doi.org/10.1016/j.rse.2010.10.011
- Didan, K. (2021). MOD13Q1 V061 [Dataset]. https://doi.org/10.5067/MODIS/MOD13Q1.061
- Gao, B.-C. (1996). *Remote Sensing of Environment* 58, 257–266. https://doi.org/10.1016/S0034-4257(96)00067-3
- Giglio, L. et al. (2018). *Remote Sensing of Environment* 217, 72–85. https://doi.org/10.1016/j.rse.2018.08.005
- Gitelson, A., Merzlyak, M.N. (1994). *Journal of Plant Physiology* 143, 286–292. https://doi.org/10.1016/S0176-1617(11)81633-0
- Gu, Y. et al. (2007). *Geophysical Research Letters* 34, L06407. https://doi.org/10.1029/2006GL029127
- Haddad, N.M. et al. (2015). *Science Advances* 1, e1500052. https://doi.org/10.1126/sciadv.1500052
- Hill, M.O. (1973). *Ecology* 54, 427–432. https://doi.org/10.2307/1934352
- Kendall, M.G. (1975). *Rank Correlation Methods*, 4th ed. Griffin.
- Kosztra, B. et al. (2019). *Updated CLC illustrated nomenclature guidelines*. EEA.
- Mann, H.B. (1945). *Econometrica* 13, 245–259. https://doi.org/10.2307/1907187
- McFeeters, S.K. (1996). *Int. J. Remote Sensing* 17, 1425–1432. https://doi.org/10.1080/01431169608948714
- McGarigal, K., Marks, B.J. (1995). FRAGSTATS. USDA FS PNW-GTR-351.
- McKee, T.B., Doesken, N.J., Kleist, J. (1993). 8th Conf. on Applied Climatology, AMS, 179–184.
- Pielou, E.C. (1966). *J. Theoretical Biology* 13, 131–144. https://doi.org/10.1016/0022-5193(66)90013-0
- Riitters, K. et al. (2000). *Conservation Ecology* 4(2): 3.
- Rouse, J.W. et al. (1974). Third ERTS Symposium, NASA SP-351, 309–317.
- Sen, P.K. (1968). *JASA* 63, 1379–1389. https://doi.org/10.1080/01621459.1968.10480934
- Shannon, C.E. (1948). *Bell System Technical Journal* 27, 379–423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
- Simpson, E.H. (1949). *Nature* 163, 688. https://doi.org/10.1038/163688a0
- Skakun, S. et al. (2022). *Remote Sensing of Environment* 274, 112990. https://doi.org/10.1016/j.rse.2022.112990
- Thomas, A.C. et al. (2014). *Geophysical Research Letters* 41, 1537–1545. https://doi.org/10.1002/2014GL059323
- Tucker, C.J. (1979). *Remote Sensing of Environment* 8, 127–150. https://doi.org/10.1016/0034-4257(79)90013-0
- Vicente-Serrano, S.M. et al. (2010). *Journal of Climate* 23, 1696–1718. https://doi.org/10.1175/2009JCLI2909.1
- Vogt, P. et al. (2007). *Landscape Ecology* 22, 171–177. https://doi.org/10.1007/s10980-006-9013-2
- Wilks, D.S. (2016). *BAMS* 97, 2263–2273. https://doi.org/10.1175/BAMS-D-15-00267.1
- Wilson, E.H., Sader, S.A. (2002). *Remote Sensing of Environment* 80, 385–396. https://doi.org/10.1016/S0034-4257(01)00318-2
- WMO (2017). *Guidelines on the Calculation of Climate Normals*. WMO-No. 1203.
- Yue, S. et al. (2002). *Hydrological Processes* 16, 1807–1829. https://doi.org/10.1002/hyp.1095

## Licence and citation

Apache-2.0 (see `LICENSE` and `NOTICE`). To cite this software, use `CITATION.cff` ("Cite this repository" on GitHub).

Archived on Zenodo: [doi:10.5281/zenodo.24118750](https://doi.org/10.5281/zenodo.24118750) (all versions; each release also has its own DOI on that page).
