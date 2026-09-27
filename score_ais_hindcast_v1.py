from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/home/admin_wsl/SIH26143")

HINDCAST = (
    ROOT /
    "results/incident_001/hindcast_48h.nc"
)

AIS_FILE = (
    ROOT /
    "data/incident_001/ais/ais_roi.parquet"
)

OUTDIR = ROOT / "results/incident_001"
OUTDIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUTDIR / "ais_hindcast_scores.csv"
TIME_OUT = OUTDIR / "ais_hindcast_time_scores.parquet"


# Cerulean vessel-source associations for validation only.
# They are NOT given any bonus in the score.
CERULEAN_CANDIDATES = {
    "368196410",
    "538005361",
    "538005314",
    "368166560",
    "367655260",
    "367507790",
}


# ============================================================
# PARAMETERS
# ============================================================

# At each hourly hindcast time, only use an AIS report
# if there is a real observation within this tolerance.
AIS_TOLERANCE = pd.Timedelta(minutes=20)

# Exploratory deterministic compatibility scale.
# This is NOT yet a calibrated probability.
SIGMA_KM = 5.0


# ============================================================
# HAVERSINE
# ============================================================

def haversine_km(lon1, lat1, lon2, lat2):
    """
    lon1,lat1 may be scalar.
    lon2,lat2 may be arrays.
    """
    R = 6371.0088

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return 2.0 * R * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )


# ============================================================
# LOAD HINDCAST
# ============================================================

ds = xr.open_dataset(HINDCAST)

times = pd.to_datetime(
    ds["time"].values,
    utc=True
).astype("datetime64[ns, UTC]")

# OpenDrift stored backwards:
# T0 -> T0-48h.
# Sort chronologically for AIS comparison.
order = np.argsort(times.values)

times = times[order]

particle_lon = (
    ds["lon"]
    .values[:, order]
)

particle_lat = (
    ds["lat"]
    .values[:, order]
)

print("=" * 80)
print("HINDCAST")
print("=" * 80)

print("Particles :", particle_lon.shape[0])
print("Times     :", particle_lon.shape[1])
print("Start     :", times[0])
print("End       :", times[-1])


# ============================================================
# LOAD AIS
# ============================================================

ais = pd.read_parquet(AIS_FILE)

ais["BaseDateTime"] = pd.to_datetime(
    ais["BaseDateTime"],
    utc=True,
    errors="coerce"
).astype("datetime64[ns, UTC]")

# Normalize MMSI
ais["MMSI"] = (
    ais["MMSI"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
)

ais["LAT"] = pd.to_numeric(
    ais["LAT"],
    errors="coerce"
)

ais["LON"] = pd.to_numeric(
    ais["LON"],
    errors="coerce"
)

ais = ais.dropna(
    subset=[
        "MMSI",
        "BaseDateTime",
        "LAT",
        "LON",
    ]
)

ais = ais[
    (ais["BaseDateTime"] >= times[0] - AIS_TOLERANCE)
    &
    (ais["BaseDateTime"] <= times[-1] + AIS_TOLERANCE)
].copy()

print("\n" + "=" * 80)
print("AIS")
print("=" * 80)

print("Rows       :", len(ais))
print("Unique MMSI:", ais["MMSI"].nunique())


# ============================================================
# MODEL TIME TABLE
# ============================================================

model_time_table = pd.DataFrame({
    "model_time": times
})


# ============================================================
# SCORE EACH MMSI
# ============================================================

time_rows = []
summary_rows = []

for mmsi, g in ais.groupby("MMSI"):

    g = (
        g.sort_values("BaseDateTime")
        .drop_duplicates(
            subset="BaseDateTime",
            keep="last"
        )
    )

    # Find the real AIS observation nearest each hourly
    # hindcast timestamp.
    matched = pd.merge_asof(
        model_time_table,
        g,
        left_on="model_time",
        right_on="BaseDateTime",
        direction="nearest",
        tolerance=AIS_TOLERANCE,
    )

    matched_valid = matched[
        matched["LAT"].notna()
        &
        matched["LON"].notna()
    ]

    if matched_valid.empty:
        continue

    vessel_name = ""

    if "VesselName" in g.columns:
        names = (
            g["VesselName"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        names = names[
            names != ""
        ]

        if not names.empty:
            vessel_name = names.mode().iloc[0]

    vessel_type = np.nan

    if "VesselType" in g.columns:
        vt = g["VesselType"].dropna()

        if not vt.empty:
            vessel_type = vt.mode().iloc[0]

    vessel_time_rows = []

    for _, row in matched_valid.iterrows():

        t = row["model_time"]

        # model-time index
        ti = model_time_table.index[
            model_time_table["model_time"] == t
        ][0]

        cloud_lon = particle_lon[:, ti]
        cloud_lat = particle_lat[:, ti]

        finite = (
            np.isfinite(cloud_lon)
            &
            np.isfinite(cloud_lat)
        )

        if finite.sum() == 0:
            continue

        distances = haversine_km(
            float(row["LON"]),
            float(row["LAT"]),
            cloud_lon[finite],
            cloud_lat[finite],
        )

        dmin = float(
            np.min(distances)
        )

        d10 = float(
            np.percentile(
                distances,
                10
            )
        )

        # Simple baseline spatial compatibility.
        # 1 = vessel directly intersects cloud;
        # falls smoothly as distance increases.
        compatibility = float(
            np.exp(
                -0.5
                * (dmin / SIGMA_KM) ** 2
            )
        )

        ais_offset_minutes = abs(
            (
                row["BaseDateTime"]
                -
                t
            ).total_seconds()
        ) / 60.0

        item = {
            "MMSI": mmsi,
            "VesselName": vessel_name,
            "VesselType": vessel_type,
            "cerulean_candidate":
                mmsi in CERULEAN_CANDIDATES,
            "model_time": t,
            "ais_time": row["BaseDateTime"],
            "ais_offset_minutes":
                ais_offset_minutes,
            "vessel_lon": float(row["LON"]),
            "vessel_lat": float(row["LAT"]),
            "min_cloud_distance_km": dmin,
            "p10_cloud_distance_km": d10,
            "compatibility": compatibility,
        }

        vessel_time_rows.append(item)
        time_rows.append(item)

    if not vessel_time_rows:
        continue

    vt = pd.DataFrame(
        vessel_time_rows
    )

    closest_idx = (
        vt["min_cloud_distance_km"]
        .idxmin()
    )

    closest = vt.loc[
        closest_idx
    ]

    coverage = (
        len(vt) /
        len(model_time_table)
    )

    mean_compatibility = float(
        vt["compatibility"].mean()
    )

    # Penalize poor AIS temporal coverage.
    coverage_adjusted_score = (
        mean_compatibility
        *
        coverage
    )

    summary_rows.append({
        "MMSI": mmsi,
        "VesselName": vessel_name,
        "VesselType": vessel_type,
        "cerulean_candidate":
            mmsi in CERULEAN_CANDIDATES,

        "matched_hours": len(vt),
        "coverage_fraction": coverage,

        "min_distance_km":
            float(
                vt[
                    "min_cloud_distance_km"
                ].min()
            ),

        "median_distance_km":
            float(
                vt[
                    "min_cloud_distance_km"
                ].median()
            ),

        "mean_distance_km":
            float(
                vt[
                    "min_cloud_distance_km"
                ].mean()
            ),

        "hours_within_2km":
            int(
                (
                    vt[
                        "min_cloud_distance_km"
                    ] <= 2
                ).sum()
            ),

        "hours_within_5km":
            int(
                (
                    vt[
                        "min_cloud_distance_km"
                    ] <= 5
                ).sum()
            ),

        "hours_within_10km":
            int(
                (
                    vt[
                        "min_cloud_distance_km"
                    ] <= 10
                ).sum()
            ),

        "mean_compatibility":
            mean_compatibility,

        "physics_score":
            coverage_adjusted_score,

        "closest_time":
            closest["model_time"],

        "closest_vessel_lon":
            closest["vessel_lon"],

        "closest_vessel_lat":
            closest["vessel_lat"],
    })


# ============================================================
# SAVE
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

time_scores = pd.DataFrame(
    time_rows
)

summary = summary.sort_values(
    [
        "physics_score",
        "hours_within_5km",
        "min_distance_km",
    ],
    ascending=[
        False,
        False,
        True,
    ]
)

summary.to_csv(
    SUMMARY_OUT,
    index=False
)

time_scores.to_parquet(
    TIME_OUT,
    index=False
)


# ============================================================
# REPORT
# ============================================================

cols = [
    "MMSI",
    "VesselName",
    "cerulean_candidate",
    "matched_hours",
    "coverage_fraction",
    "min_distance_km",
    "median_distance_km",
    "hours_within_2km",
    "hours_within_5km",
    "hours_within_10km",
    "physics_score",
    "closest_time",
]

print("\n" + "=" * 120)
print("TOP 25 — PHYSICS COMPATIBILITY")
print("=" * 120)

print(
    summary[
        cols
    ]
    .head(25)
    .to_string(
        index=False
    )
)

print("\n" + "=" * 120)
print("CERULEAN-LINKED VESSELS")
print("=" * 120)

known = summary[
    summary[
        "cerulean_candidate"
    ]
]

if len(known):
    print(
        known[
            cols
        ].to_string(
            index=False
        )
    )
else:
    print(
        "No Cerulean candidate "
        "had sufficient AIS coverage."
    )

print("\nSaved:")
print(SUMMARY_OUT)
print(TIME_OUT)

ds.close()
