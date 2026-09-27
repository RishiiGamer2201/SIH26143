import json
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
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

OUT_PARTICLES = (
    OUTDIR /
    "hindcast_ensemble_particles.parquet"
)

OUT_SCENARIOS = (
    OUTDIR /
    "hindcast_ensemble_scenarios.csv"
)


# ============================================================
# INCIDENT
# ============================================================

T0 = datetime(
    2023, 1, 3,
    0, 1, 42
)

DURATION_HOURS = 48

PARTICLES_PER_SCENARIO = 400

BASE_SEED = 26143


# ============================================================
# UNCERTAINTY GRID
# ============================================================

WINDAGE = [
    0.02,
    0.03,
    0.04,
]

CURRENT_UNCERTAINTY = [
    0.00,
    0.05,
    0.10,
]

WIND_UNCERTAINTY = [
    0.00,
    0.50,
    1.00,
]

SCENARIOS = list(
    product(
        WINDAGE,
        CURRENT_UNCERTAINTY,
        WIND_UNCERTAINTY,
    )
)


# ============================================================
# SLICK
# ============================================================

geojson = json.loads(
    SLICK.read_text()
)

geom = shape(
    geojson["features"][0]["geometry"]
)

if not geom.is_valid:
    geom = geom.buffer(0)

minx, miny, maxx, maxy = geom.bounds


def sample_polygon(
    geometry,
    number,
    seed,
):
    rng = np.random.default_rng(seed)

    xs_out = []
    ys_out = []

    while len(xs_out) < number:

        need = number - len(xs_out)

        n_try = max(
            1000,
            need * 4
        )

        xs = rng.uniform(
            minx,
            maxx,
            n_try
        )

        ys = rng.uniform(
            miny,
            maxy,
            n_try
        )

        for x, y in zip(xs, ys):

            if geometry.contains(
                Point(x, y)
            ):
                xs_out.append(x)
                ys_out.append(y)

                if (
                    len(xs_out)
                    == number
                ):
                    break

    return (
        np.asarray(xs_out),
        np.asarray(ys_out),
    )


# ============================================================
# READERS
# ============================================================

ocean_reader = (
    reader_netCDF_CF_generic.Reader(
        str(OCEAN)
    )
)

wind_reader = (
    reader_netCDF_CF_generic.Reader(
        str(WIND)
    )
)


# ============================================================
# RUN ENSEMBLE
# ============================================================

all_particles = []
scenario_records = []

print("=" * 90)
print("48-HOUR HINDCAST UNCERTAINTY ENSEMBLE")
print("=" * 90)

print(
    "Scenarios:",
    len(SCENARIOS)
)

print(
    "Particles/scenario:",
    PARTICLES_PER_SCENARIO
)

print(
    "Total trajectories:",
    len(SCENARIOS)
    * PARTICLES_PER_SCENARIO
)


for scenario_id, (
    windage,
    current_unc,
    wind_unc,
) in enumerate(
    SCENARIOS,
    start=1,
):

    scenario_seed = (
        BASE_SEED
        + scenario_id * 1000
    )

    # Seed both our polygon sampler
    # and OpenDrift's stochastic terms.
    np.random.seed(
        scenario_seed
    )

    lons, lats = sample_polygon(
        geom,
        PARTICLES_PER_SCENARIO,
        scenario_seed,
    )

    print(
        f"\nScenario "
        f"{scenario_id:02d}/"
        f"{len(SCENARIOS)}"
    )

    print(
        " windage=",
        windage,
        " current_unc=",
        current_unc,
        " wind_unc=",
        wind_unc,
    )

    o = OpenOil(
        loglevel=50
    )

    o.add_reader([
        ocean_reader,
        wind_reader,
    ])

    # ------------------------------------
    # Transport-only backwards simulation
    # ------------------------------------

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

    o.set_config(
        "drift:current_uncertainty",
        current_unc
    )

    o.set_config(
        "drift:wind_uncertainty",
        wind_unc
    )

    o.set_config(
        "environment:fallback:land_binary_mask",
        0
    )

    # ------------------------------------
    # Seed complete observed slick
    # ------------------------------------

    o.seed_elements(
        lon=lons,
        lat=lats,
        time=T0,
        number=PARTICLES_PER_SCENARIO,
        wind_drift_factor=windage,
    )

    # ------------------------------------
    # Backward run
    # ------------------------------------

    result = o.run(
        duration=timedelta(
            hours=DURATION_HOURS
        ),
        time_step=-900,
        time_step_output=3600,
    )

    lon = result["lon"].values
    lat = result["lat"].values

    result_times = (
        pd.to_datetime(
            result["time"].values,
            utc=True
        )
    )

    ntraj, ntime = lon.shape

    scenario_df = pd.DataFrame({
        "scenario_id":
            np.repeat(
                scenario_id,
                ntraj * ntime
            ),

        "trajectory":
            np.repeat(
                np.arange(ntraj),
                ntime
            ),

        "time":
            np.tile(
                result_times,
                ntraj
            ),

        "lon":
            lon.reshape(-1),

        "lat":
            lat.reshape(-1),

        "windage":
            np.repeat(
                windage,
                ntraj * ntime
            ),

        "current_uncertainty":
            np.repeat(
                current_unc,
                ntraj * ntime
            ),

        "wind_uncertainty":
            np.repeat(
                wind_unc,
                ntraj * ntime
            ),
    })

    scenario_df = (
        scenario_df[
            np.isfinite(
                scenario_df["lon"]
            )
            &
            np.isfinite(
                scenario_df["lat"]
            )
        ]
        .copy()
    )

    all_particles.append(
        scenario_df
    )

    scenario_records.append({
        "scenario_id":
            scenario_id,

        "seed":
            scenario_seed,

        "windage":
            windage,

        "current_uncertainty":
            current_unc,

        "wind_uncertainty":
            wind_unc,

        "particles":
            ntraj,

        "times":
            ntime,

        "valid_rows":
            len(scenario_df),
    })


# ============================================================
# COMBINE
# ============================================================

particles = pd.concat(
    all_particles,
    ignore_index=True
)

scenarios = pd.DataFrame(
    scenario_records
)

particles.to_parquet(
    OUT_PARTICLES,
    index=False
)

scenarios.to_csv(
    OUT_SCENARIOS,
    index=False
)


# ============================================================
# AUDIT
# ============================================================

print("\n" + "=" * 90)
print("ENSEMBLE COMPLETE")
print("=" * 90)

print(
    "Scenarios:",
    scenarios.shape[0]
)

print(
    "Total particle-time rows:",
    len(particles)
)

print(
    "Unique model times:",
    particles["time"].nunique()
)

print(
    "Earliest:",
    particles["time"].min()
)

print(
    "Latest:",
    particles["time"].max()
)

print(
    "Unique windage:",
    sorted(
        particles[
            "windage"
        ].unique()
    )
)

print(
    "Unique current uncertainties:",
    sorted(
        particles[
            "current_uncertainty"
        ].unique()
    )
)

print(
    "Unique wind uncertainties:",
    sorted(
        particles[
            "wind_uncertainty"
        ].unique()
    )
)

print("\nROWS PER SCENARIO")
print(
    scenarios[
        [
            "scenario_id",
            "windage",
            "current_uncertainty",
            "wind_uncertainty",
            "particles",
            "times",
            "valid_rows",
        ]
    ].to_string(
        index=False
    )
)

print("\nSaved:")
print(OUT_PARTICLES)
print(OUT_SCENARIOS)
