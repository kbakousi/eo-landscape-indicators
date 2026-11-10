"""Real-data demo: burned area and burn frequency from MODIS MCD64A1 (500 m), Planetary Computer.

Area: Dadia forest and surroundings, Evros, north-eastern Greece, which burned extensively in August 2023.
That known event is a sanity check: 2023 should stand out in the annual burned-area table.

Steps: BurnDate -> monthly burned / mapped flags -> one layer per year -> burn frequency and burned hectares
on an equal-area grid (EPSG:3035).

Requirements: network access and ``pip install -e ".[stac]"``; no account needed.
Run:  python examples/demo_mcd64_burned_area.py [--start 2001] [--end 2024]
Writes outputs/mcd64_burn_frequency.nc and outputs/mcd64_burned_area.csv.

References: Giglio et al. (2018), https://doi.org/10.1016/j.rse.2018.08.005; Giglio et al. (2021) MCD64A1 V061,
https://doi.org/10.5067/MODIS/MCD64A1.061.
"""

import argparse
from pathlib import Path

from eolandscape import xr as exr
from eolandscape.loaders import stac

BBOX = (25.95, 40.95, 26.40, 41.25)  # lon/lat, about 38 x 33 km
CRS, RES = "EPSG:3035", 500  # equal-area: every pixel is 25 ha

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=2001)
ap.add_argument("--end", type=int, default=2024)
args = ap.parse_args()

items = stac.search("modis-64A1-061", BBOX, f"{args.start}-01-01", f"{args.end}-12-31")
print(f"{len(items)} monthly MCD64A1 layers, {args.start}-{args.end}")
ds = stac.load(items, ["Burn_Date"], crs=CRS, resolution=RES, bbox=BBOX, chunks={"x": 256, "y": 256},
               resampling="nearest", groupby="solar_day")
burn_date = ds["Burn_Date"].sortby("time")

annual = exr.annual_burned(burn_date)
freq = exr.burn_frequency(annual)
area = exr.burned_area(annual, pixel_area_m2=RES * RES).to_series().rename("burned_ha").round(0)

print(area.to_string())
print(f"largest year: {int(area.idxmax())} ({area.max():,.0f} ha)")
print(f"pixels burned at least once: {int((freq['frequency'] > 0).sum())}; "
      f"at least twice: {int((freq['frequency'] > 1).sum())}")

out = Path(__file__).resolve().parent.parent / "outputs"
out.mkdir(exist_ok=True)
area.to_csv(out / "mcd64_burned_area.csv")
result = freq.drop_vars("spatial_ref", errors="ignore").astype("float32")  # portable NetCDF types
result.attrs.update(source="MODIS MCD64A1 V061 via Microsoft Planetary Computer", bbox_lonlat=str(BBOX), crs=CRS,
                    years=f"{args.start}-{args.end}")
result.to_netcdf(out / "mcd64_burn_frequency.nc")
print(f"Saved {out / 'mcd64_burned_area.csv'} and {out / 'mcd64_burn_frequency.nc'}")
