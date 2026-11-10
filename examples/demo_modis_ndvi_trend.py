"""Real-data demo: growing-season NDVI trends from MODIS MOD13Q1 (250 m), Planetary Computer.

Area: a small box around Naoussa, northern Greece (Xinomavro vineyards and surrounding land).
Steps: quality mask (pixel reliability) -> April-October composites -> Theil-Sen slope ->
Mann-Kendall with Benjamini-Hochberg FDR. Data are read lazily and per chunk.

Requirements: network access and ``pip install -e ".[stac]"``; no account needed.
Run:  python examples/demo_modis_ndvi_trend.py [--start 2001] [--end 2024]
Writes outputs/modis_ndvi_trend.nc and prints a summary. Expect a few minutes for 24 years.

References: Didan (2021) MOD13Q1 V061, https://doi.org/10.5067/MODIS/MOD13Q1.061; Sen (1968); Mann (1945);
Benjamini & Hochberg (1995). See the README for the full list.
"""

import argparse
from pathlib import Path

import numpy as np

from eolandscape import xr as exr
from eolandscape.loaders import stac

BBOX = (22.05, 40.55, 22.20, 40.66)  # lon/lat, about 13 x 12 km
CRS, RES = "EPSG:3035", 250  # equal-area grid for Europe

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=2001)
ap.add_argument("--end", type=int, default=2024)
args = ap.parse_args()

items = stac.search("modis-13Q1-061", BBOX, f"{args.start}-01-01", f"{args.end}-12-31",
                    query={"platform": {"eq": "terra"}})  # MOD13Q1 (Terra) only: one sensor, one series
print(f"{len(items)} MOD13Q1 composites, {args.start}-{args.end}")
ds = stac.load(items, ["250m_16_days_NDVI", "250m_16_days_pixel_reliability"], crs=CRS, resolution=RES, bbox=BBOX,
               chunks={"x": 256, "y": 256}, resampling="nearest", groupby="solar_day")
ds = ds.sortby("time")

ndvi = exr.modis_vi_scale(ds["250m_16_days_NDVI"]).where(exr.modis_vi_clear(ds["250m_16_days_pixel_reliability"]))
annual = exr.annual_composite(ndvi, months=(4, 10), min_obs=3)
sen = exr.theil_sen(annual, min_n=10)
mk = exr.mann_kendall(annual, min_n=10)
significant = exr.benjamini_hochberg(mk["p"], alpha=0.05)

slope = sen["slope"].values
valid = np.isfinite(slope)
print(f"pixels with a trend estimate: {valid.sum()} of {valid.size}")
print(f"median Theil-Sen slope: {np.nanmedian(slope):+.4f} NDVI/yr")
for name, mask in [("greening", significant.values & (slope > 0)), ("browning", significant.values & (slope < 0))]:
    print(f"significant {name} (FDR 5 %): {100 * mask.sum() / max(valid.sum(), 1):.1f} % of pixels")

out = Path(__file__).resolve().parent.parent / "outputs"
out.mkdir(exist_ok=True)
result = sen.assign(z=mk["z"], p=mk["p"], significant=significant).drop_vars("spatial_ref", errors="ignore")
result = result.astype("float32")  # portable NetCDF types
result.attrs.update(source="MODIS MOD13Q1 V061 via Microsoft Planetary Computer", bbox_lonlat=str(BBOX), crs=CRS,
                    years=f"{args.start}-{args.end}", composite="April-October mean, >= 3 clear observations")
result.to_netcdf(out / "modis_ndvi_trend.nc")
print(f"Saved {out / 'modis_ndvi_trend.nc'}")
