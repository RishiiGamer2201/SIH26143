import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from shapely.geometry import shape, Point

from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/home/admin_wsl/SIH26143")

OCEAN = (
    ROOT /
    "data/incident_001/currents/ocean_opendrift.nc"
)

WIND = (
    ROOT /
    "data/incident_001/wind/era5_wind_10m_151h.nc"
)

SLICK = (
    ROOT /
    "data/incident_001/metadata/slick_3775938.geojson"
)

OUTDIR = ROOT / "results/incident_001"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTFILE = OUTDIR / "hindcast_48h.nc"


# ============================================================
# INCIDENT
# ============================================================

T0 = datetime(
    2023, 1, 3,
    0, 1, 42
)

N_PARTICLES = 2000
RNG_SEED = 26143


# ============================================================
# LOAD SLICK POLYGON
# ============================================================

geojson = json.loads(SLICK.read_text())

geom = shape(
    geojson["features"][0]["geometry"]
)

if not geom.is_valid:
    geom = geom.buffer(0)

print("=" * 80)
print("SLICK")
print("=" * 80)

print("Geometry :", geom.geom_type)
print("Bounds   :", geom.bounds)
print("Centroid :", geom.centroid.x, geom.centroid.y)


# ============================================================
# UNIFORM RANDOM POINTS INSIDE SLICK
# ============================================================

rng = np.random.default_rng(RNG_SEED)

minx, miny, maxx, maxy = geom.bounds

lons = []
lats = []

while len(lons) < N_PARTICLES:

    # generate batches to avoid one-at-a-time sampling
    n_need = N_PARTICLES - len(lons)
    n_try = max(1000, n_need * 3)

    xs = rng.uniform(minx, maxx, n_try)
    ys = rng.uniform(miny, maxy, n_try)

    for x, y in zip(xs, ys):

        if geom.contains(Point(x, y)):

            lons.append(x)
            lats.append(y)

            if len(lons) == N_PARTICLES:
                break

lons = np.asarray(lons)
lats = np.asarray(lats)

print("Particles:", len(lons))


# ============================================================
# READERS
# ============================================================

ocean_reader = reader_netCDF_CF_generic.Reader(
    str(OCEAN)
)

wind_reader = reader_netCDF_CF_generic.Reader(
    str(WIND)
)


# ============================================================
# OPENOIL
# ============================================================

o = OpenOil(loglevel=20)

o.add_reader([
    ocean_reader,
    wind_reader
])


# ============================================================
# TRANSPORT-ONLY HINDCAST SETTINGS
# ============================================================

# Backwards oil-weathering is not physically reversible.
# We want source-location reconstruction here.

o.set_config(
    "processes:evaporation",
    False
)

o.set_config(
    "processes:emulsification",
    False
)

o.set_config(
    "processes:dispersion",
    False
)

o.set_config(
    "processes:biodegradation",
    False
)

o.set_config(
    "drift:vertical_mixing",
    False
)

o.set_config(
    "drift:vertical_advection",
    False
)

# Deterministic baseline.
# Ensemble uncertainty comes later.
o.set_config(
    "drift:current_uncertainty",
    0
)

o.set_config(
    "drift:wind_uncertainty",
    0
)

# Incident is far offshore, so this is acceptable
# for our first transport-only test.
o.set_config(
    "environment:fallback:land_binary_mask",
    0
)


# ============================================================
# SEED OBSERVED SLICK
# ============================================================

o.seed_elements(
    lon=lons,
    lat=lats,
    time=T0,
    number=N_PARTICLES,

    # OpenOil default is 0.03.
    # Explicit for reproducibility.
    wind_drift_factor=0.03
)


# ============================================================
# RUN BACKWARDS
# ============================================================

print("\n" + "=" * 80)
print("STARTING 48-HOUR BACKWARD HINDCAST")
print("=" * 80)

print("Start:", T0)
print(
    "Target:",
    T0 - timedelta(hours=48)
)

o.run(
    duration=timedelta(hours=48),

    # 15 minute numerical timestep, backwards
    time_step=-900,

    # save hourly
    time_step_output=3600,

    outfile=str(OUTFILE)
)

print("\n" + "=" * 80)
print("HINDCAST COMPLETE")
print("=" * 80)

print(o)

print("\nSaved:")
print(OUTFILE)


# ============================================================
# AUDIT RESULT
# ============================================================

ds = xr.open_dataset(OUTFILE)

print("\nOUTPUT DATASET")
print(ds)

time_name = "time"

print("\nTIME")
print("First:", ds[time_name].values[0])
print("Last :", ds[time_name].values[-1])
print("Steps:", ds.sizes[time_name])

lon = ds["lon"]
lat = ds["lat"]

# Find earliest physical timestamp regardless of array ordering
tvals = ds[time_name].values

earliest_idx = int(np.argmin(tvals))

origin_lon = lon.isel(time=earliest_idx)
origin_lat = lat.isel(time=earliest_idx)

valid = (
    np.isfinite(origin_lon.values)
    &
    np.isfinite(origin_lat.values)
)

origin_lons = origin_lon.values[valid]
origin_lats = origin_lat.values[valid]

print("\n48-HOUR ORIGIN CLOUD")
print("Valid particles:", valid.sum())

print(
    "Longitude:",
    float(origin_lons.min()),
    "->",
    float(origin_lons.max())
)

print(
    "Latitude :",
    float(origin_lats.min()),
    "->",
    float(origin_lats.max())
)

print(
    "Mean:",
    float(origin_lons.mean()),
    float(origin_lats.mean())
)

print(
    "Median:",
    float(np.median(origin_lons)),
    float(np.median(origin_lats))
)

np.savetxt(
    OUTDIR / "origin_cloud_48h.csv",
    np.column_stack([
        origin_lons,
        origin_lats
    ]),
    delimiter=",",
    header="longitude,latitude",
    comments=""
)

print("\nSaved:")
print(
    OUTDIR /
    "origin_cloud_48h.csv"
)

ds.close()
