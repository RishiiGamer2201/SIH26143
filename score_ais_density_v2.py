from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/home/admin_wsl/SIH26143")

HINDCAST = ROOT / "results/incident_001/hindcast_48h.nc"
AIS_FILE = ROOT / "data/incident_001/ais/ais_roi.parquet"
OUTDIR = ROOT / "results/incident_001"

TIME_OUT = OUTDIR / "ais_density_time_scores_v2.parquet"
SUMMARY_OUT = OUTDIR / "ais_density_scores_v2.csv"

CERULEAN = {
    "368196410",
    "538005361",
    "538005314",
    "368166560",
    "367655260",
    "367507790",
}

AIS_TOLERANCE = pd.Timedelta(minutes=20)

# Spatial bandwidth for the particle probability cloud.
SIGMA_KM = 5.0


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0088

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2 * R *
        np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )


# ============================================================
# HINDCAST
# ============================================================

ds = xr.open_dataset(HINDCAST)

times = (
    pd.to_datetime(ds.time.values, utc=True)
    .astype("datetime64[ns, UTC]")
)

order = np.argsort(times.values)

times = times[order]

plon = ds.lon.values[:, order]
plat = ds.lat.values[:, order]

model_times = pd.DataFrame({
    "model_time": times
})


# ============================================================
# AIS
# ============================================================

ais = pd.read_parquet(AIS_FILE)

ais["BaseDateTime"] = (
    pd.to_datetime(
        ais["BaseDateTime"],
        utc=True,
        errors="coerce"
    )
    .astype("datetime64[ns, UTC]")
)

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
    subset=["MMSI", "BaseDateTime", "LAT", "LON"]
)

ais = ais[
    (ais.BaseDateTime >= times[0] - AIS_TOLERANCE)
    &
    (ais.BaseDateTime <= times[-1] + AIS_TOLERANCE)
].copy()


# ============================================================
# SCORE
# ============================================================

time_records = []
summary_records = []

for mmsi, g in ais.groupby("MMSI"):

    g = (
        g.sort_values("BaseDateTime")
        .drop_duplicates("BaseDateTime")
    )

    matched = pd.merge_asof(
        model_times,
        g,
        left_on="model_time",
        right_on="BaseDateTime",
        direction="nearest",
        tolerance=AIS_TOLERANCE,
    )

    matched = matched.dropna(
        subset=["LAT", "LON"]
    )

    if matched.empty:
        continue

    vessel_name = ""

    if "VesselName" in g:
        names = (
            g.VesselName
            .dropna()
            .astype(str)
            .str.strip()
        )

        names = names[names != ""]

        if len(names):
            vessel_name = names.mode().iloc[0]

    rows = []

    for _, row in matched.iterrows():

        t = row["model_time"]

        ti = int(
            np.where(times == t)[0][0]
        )

        lon_cloud = plon[:, ti]
        lat_cloud = plat[:, ti]

        finite = (
            np.isfinite(lon_cloud)
            &
            np.isfinite(lat_cloud)
        )

        lon_cloud = lon_cloud[finite]
        lat_cloud = lat_cloud[finite]

        if len(lon_cloud) == 0:
            continue

        d = haversine_km(
            float(row.LON),
            float(row.LAT),
            lon_cloud,
            lat_cloud,
        )

        #
        # IMPORTANT DIFFERENCE FROM V1:
        #
        # Score against ALL particles, not just
        # the single nearest particle.
        #
        kernel = np.exp(
            -0.5 * (d / SIGMA_KM) ** 2
        )

        density_score = float(
            kernel.mean()
        )

        rec = {
            "MMSI": mmsi,
            "VesselName": vessel_name,
            "cerulean_candidate": mmsi in CERULEAN,

            "model_time": t,
            "ais_time": row.BaseDateTime,

            "vessel_lon": float(row.LON),
            "vessel_lat": float(row.LAT),

            "min_distance_km":
                float(np.min(d)),

            "p10_distance_km":
                float(np.percentile(d, 10)),

            "median_distance_km":
                float(np.median(d)),

            "fraction_particles_2km":
                float(np.mean(d <= 2)),

            "fraction_particles_5km":
                float(np.mean(d <= 5)),

            "fraction_particles_10km":
                float(np.mean(d <= 10)),

            "density_score":
                density_score,
        }

        rows.append(rec)
        time_records.append(rec)

    if not rows:
        continue

    v = pd.DataFrame(rows)

    peak_idx = v.density_score.idxmax()
    peak = v.loc[peak_idx]

    #
    # POINT-RELEASE HYPOTHESIS:
    # vessel only needs to intersect a likely
    # source region at one plausible release time.
    #
    point_release_score = float(
        v.density_score.max()
    )

    #
    # CONTINUOUS/EXTENDED RELEASE HYPOTHESIS:
    # rewards persistent overlap through time.
    #
    continuous_release_score = float(
        v.density_score.mean()
    )

    summary_records.append({
        "MMSI": mmsi,
        "VesselName": vessel_name,
        "cerulean_candidate": mmsi in CERULEAN,

        # Keep coverage as CONFIDENCE,
        # not part of the physics score.
        "matched_hours": len(v),
        "coverage_fraction":
            len(v) / len(times),

        "point_release_score":
            point_release_score,

        "continuous_release_score":
            continuous_release_score,

        "peak_time":
            peak.model_time,

        "peak_min_distance_km":
            float(peak.min_distance_km),

        "peak_p10_distance_km":
            float(peak.p10_distance_km),

        "peak_fraction_particles_5km":
            float(
                peak.fraction_particles_5km
            ),

        "peak_fraction_particles_10km":
            float(
                peak.fraction_particles_10km
            ),

        "best_min_distance_km":
            float(v.min_distance_km.min()),

        "best_p10_distance_km":
            float(v.p10_distance_km.min()),

        "hours_density_gt_005":
            int(
                (v.density_score >= 0.05).sum()
            ),

        "hours_density_gt_01":
            int(
                (v.density_score >= 0.10).sum()
            ),
    })


summary = pd.DataFrame(summary_records)
time_scores = pd.DataFrame(time_records)

summary["point_rank"] = (
    summary["point_release_score"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

summary["continuous_rank"] = (
    summary["continuous_release_score"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

summary = summary.sort_values(
    "point_release_score",
    ascending=False
)

summary.to_csv(
    SUMMARY_OUT,
    index=False
)

time_scores.to_parquet(
    TIME_OUT,
    index=False
)


cols = [
    "point_rank",
    "continuous_rank",
    "MMSI",
    "VesselName",
    "cerulean_candidate",
    "matched_hours",
    "coverage_fraction",
    "point_release_score",
    "continuous_release_score",
    "peak_time",
    "peak_min_distance_km",
    "peak_p10_distance_km",
    "peak_fraction_particles_5km",
    "peak_fraction_particles_10km",
]


print("=" * 150)
print("TOP 20 — POINT RELEASE")
print("=" * 150)

print(
    summary[cols]
    .head(20)
    .to_string(index=False)
)


print("\n" + "=" * 150)
print("CERULEAN-LINKED — V2")
print("=" * 150)

print(
    summary[
        summary.cerulean_candidate
    ][cols]
    .sort_values("point_rank")
    .to_string(index=False)
)


print("\n" + "=" * 150)
print("TOP 20 — CONTINUOUS RELEASE")
print("=" * 150)

print(
    summary
    .sort_values(
        "continuous_release_score",
        ascending=False
    )[cols]
    .head(20)
    .to_string(index=False)
)


print("\nSaved:")
print(SUMMARY_OUT)
print(TIME_OUT)

ds.close()
