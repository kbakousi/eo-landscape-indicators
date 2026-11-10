# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-11-10

### Fixed
- `xr`: functions with two or more labelled inputs (the indices, `s2_l2a_reflectance` with per-date baselines,
  `anomalies`, `standardized_anomalies`) lost the CRS on `spatial_ref` with xarray < 2025.7 (the newest xarray
  for Python 3.10), and also cleared it on the caller's input arrays. Coordinate attributes are now kept and
  the inputs are left untouched on every supported xarray version.

## [0.1.0] - 2026-11-10

### Added
- `indices`: NDVI, NDWI (McFeeters and Gao), NDMI, NDDI, NDRE; sensor band maps.
- `masking`: Landsat C2, Sentinel-2 L2A (processing-baseline offset, SCL) and MODIS VI scaling and QA masks.
- `anomalies`: seasonal composites, climatology, anomalies, standardized anomalies, McKee drought categories.
- `trends`: annual composites, OLS, Theil-Sen, Mann-Kendall (ties, pre-whitening), Benjamini-Hochberg.
- `fire`: MCD64A1 annual burned flags, burn frequency, burned area.
- `landcover`, `diversity`, `fragmentation`, `zonal`.
- `xr`: labelled interface: the indices, masking, anomaly, trend and fire functions on `xarray` objects, keeping
  coordinates and CRS, lazy on dask arrays, with exact grid alignment (optional extra `[xarray]`).
- Optional STAC loaders.
- Real-data examples: MODIS NDVI trends (Naoussa) and MCD64A1 burned area (Dadia, Evros).
- Tutorial notebook `tutorials/01_trends_and_anomalies.ipynb`, executed by the test suite.

### Changed
- `masking.s2_l2a_reflectance` accepts an array of processing baselines (one per time step).
