from pathlib import Path

import pandas as pd
import numpy as np


ROOT = Path("/home/admin_wsl/SIH26143")

TIME_FILE = (
    ROOT /
    "results/incident_001/ais_density_time_scores_v2.parquet"
)

SUMMARY_FILE = (
    ROOT /
    "results/incident_001/ais_density_scores_v2.csv"
)

OUT = (
    ROOT /
    "results/incident_001/release_age_scores.csv"
)

T0 = pd.Timestamp("2023-01-03T00:01:42Z")


# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(TIME_FILE)

df["model_time"] = pd.to_datetime(
    df["model_time"],
    utc=True
)

df["hours_before_detection"] = (
    (
        T0 - df["model_time"]
    ).dt.total_seconds()
    / 3600.0
)


# ============================================================
# RELEASE AGE WINDOWS
# ============================================================

bins = [
    -0.001,
    6,
    12,
    24,
    36,
    48.01,
]

labels = [
    "0-6h",
    "6-12h",
    "12-24h",
    "24-36h",
    "36-48h",
]

df["release_age_bin"] = pd.cut(
    df["hours_before_detection"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)


# ============================================================
# AGGREGATE EACH VESSEL WITHIN EACH AGE WINDOW
# ============================================================

records = []

for (
    mmsi,
    vessel_name,
    cerulean,
    age_bin
), g in df.groupby(
    [
        "MMSI",
        "VesselName",
        "cerulean_candidate",
        "release_age_bin",
    ],
    observed=True
):

    if len(g) == 0:
        continue

    peak_idx = g["density_score"].idxmax()
    peak = g.loc[peak_idx]

    records.append({
        "MMSI": mmsi,
        "VesselName": vessel_name,
        "cerulean_candidate": cerulean,

        "release_age_bin": str(age_bin),

        "matched_hours_in_bin": len(g),

        "peak_density_score":
            float(g["density_score"].max()),

        "mean_density_score":
            float(g["density_score"].mean()),

        "peak_time":
            peak["model_time"],

        "peak_age_hours":
            float(
                peak["hours_before_detection"]
            ),

        "peak_min_distance_km":
            float(
                peak["min_distance_km"]
            ),

        "peak_p10_distance_km":
            float(
                peak["p10_distance_km"]
            ),

        "fraction_particles_5km":
            float(
                peak[
                    "fraction_particles_5km"
                ]
            ),

        "fraction_particles_10km":
            float(
                peak[
                    "fraction_particles_10km"
                ]
            ),
    })


out = pd.DataFrame(records)

out["rank_in_age_bin"] = (
    out.groupby(
        "release_age_bin"
    )["peak_density_score"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

out = out.sort_values(
    [
        "release_age_bin",
        "rank_in_age_bin",
    ]
)

out.to_csv(
    OUT,
    index=False
)


# ============================================================
# REPORT
# ============================================================

cols = [
    "rank_in_age_bin",
    "MMSI",
    "VesselName",
    "cerulean_candidate",
    "matched_hours_in_bin",
    "peak_density_score",
    "peak_age_hours",
    "peak_min_distance_km",
    "peak_p10_distance_km",
    "fraction_particles_5km",
    "fraction_particles_10km",
]


for age_bin in labels:

    print("\n")
    print("=" * 145)
    print(
        f"RELEASE AGE: {age_bin}"
    )
    print("=" * 145)

    sub = out[
        out["release_age_bin"] == age_bin
    ]

    print(
        sub[cols]
        .head(12)
        .to_string(index=False)
    )


print("\n")
print("=" * 145)
print("CERULEAN-LINKED ACROSS RELEASE AGES")
print("=" * 145)

known = out[
    out["cerulean_candidate"]
]

print(
    known[
        [
            "release_age_bin",
            *cols,
        ]
    ]
    .sort_values(
        [
            "release_age_bin",
            "rank_in_age_bin",
        ]
    )
    .to_string(index=False)
)


print("\nSaved:")
print(OUT)
