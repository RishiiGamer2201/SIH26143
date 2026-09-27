from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/admin_wsl/SIH26143")

ENSEMBLE_FILE = (
    ROOT /
    "results/incident_001/hindcast_ensemble_particles.parquet"
)

AIS_FILE = (
    ROOT /
    "data/incident_001/ais/ais_roi.parquet"
)

OUTDIR = ROOT / "results/incident_001"

SCENARIO_OUT = (
    OUTDIR /
    "ais_ensemble_scenario_scores.parquet"
)

ROBUST_OUT = (
    OUTDIR /
    "ais_ensemble_robust_scores.csv"
)


CERULEAN = {
    "368196410",
    "538005361",
    "538005314",
    "368166560",
    "367655260",
    "367507790",
}


T0 = pd.Timestamp(
    "2023-01-03T00:01:42Z"
)

AIS_TOLERANCE = pd.Timedelta(
    minutes=20
)

SIGMA_KM = 5.0


AGE_BINS = [
    -0.001,
    6,
    12,
    24,
    36,
    48.01,
]

AGE_LABELS = [
    "0-6h",
    "6-12h",
    "12-24h",
    "24-36h",
    "36-48h",
]


# ============================================================
# DISTANCE
# ============================================================

def haversine_km(
    lon1,
    lat1,
    lon2,
    lat2,
):
    R = 6371.0088

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)

    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return (
        2 * R *
        np.arcsin(
            np.sqrt(
                np.clip(a, 0, 1)
            )
        )
    )


# ============================================================
# LOAD ENSEMBLE
# ============================================================

print("=" * 100)
print("LOADING ENSEMBLE")
print("=" * 100)

ens = pd.read_parquet(
    ENSEMBLE_FILE
)

ens["time"] = (
    pd.to_datetime(
        ens["time"],
        utc=True
    )
    .astype("datetime64[ns, UTC]")
)

print(
    "Rows:",
    len(ens)
)

print(
    "Scenarios:",
    ens["scenario_id"].nunique()
)

print(
    "Times:",
    ens["time"].nunique()
)

print(
    "Time:",
    ens["time"].min(),
    "->",
    ens["time"].max()
)


# ============================================================
# BASIC ENSEMBLE SANITY CHECK
# ============================================================

earliest = ens["time"].min()

origin = (
    ens[
        ens["time"] == earliest
    ]
    .groupby("scenario_id")
    .agg(
        mean_lon=("lon", "mean"),
        mean_lat=("lat", "mean"),
        std_lon=("lon", "std"),
        std_lat=("lat", "std"),
    )
)

print("\n" + "=" * 100)
print("48-HOUR ENSEMBLE SPREAD")
print("=" * 100)

print(
    "Scenario centroid longitude range:",
    origin["mean_lon"].min(),
    "->",
    origin["mean_lon"].max()
)

print(
    "Scenario centroid latitude range :",
    origin["mean_lat"].min(),
    "->",
    origin["mean_lat"].max()
)

print(
    "Distinct centroids:",
    len(
        origin[
            ["mean_lon", "mean_lat"]
        ]
        .round(6)
        .drop_duplicates()
    )
)


# ============================================================
# INDEX CLOUDS
# ============================================================

print("\nIndexing particle clouds...")

clouds = {}

for (
    scenario_id,
    t
), g in ens.groupby(
    [
        "scenario_id",
        "time",
    ],
    sort=False,
):

    clouds[
        (
            int(scenario_id),
            t,
        )
    ] = (
        g[
            ["lon", "lat"]
        ]
        .to_numpy(
            dtype=float
        )
    )


scenario_ids = sorted(
    ens[
        "scenario_id"
    ].unique()
)

model_times = (
    pd.DatetimeIndex(
        sorted(
            ens["time"].unique()
        )
    )
)

model_time_df = pd.DataFrame({
    "model_time":
        model_times
})


# ============================================================
# AIS
# ============================================================

ais = pd.read_parquet(
    AIS_FILE
)

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
    .str.replace(
        r"\.0$",
        "",
        regex=True
    )
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
    (
        ais["BaseDateTime"]
        >= model_times.min()
        - AIS_TOLERANCE
    )
    &
    (
        ais["BaseDateTime"]
        <= model_times.max()
        + AIS_TOLERANCE
    )
].copy()


print("\n" + "=" * 100)
print("AIS")
print("=" * 100)

print(
    "Rows:",
    len(ais)
)

print(
    "MMSIs:",
    ais["MMSI"].nunique()
)


# ============================================================
# MATCH AIS ONCE
# ============================================================

matched_vessels = {}

for mmsi, g in ais.groupby(
    "MMSI"
):

    g = (
        g.sort_values(
            "BaseDateTime"
        )
        .drop_duplicates(
            "BaseDateTime"
        )
    )

    matched = pd.merge_asof(
        model_time_df,
        g,
        left_on="model_time",
        right_on="BaseDateTime",
        direction="nearest",
        tolerance=AIS_TOLERANCE,
    )

    matched = matched.dropna(
        subset=[
            "LAT",
            "LON",
        ]
    ).copy()

    if matched.empty:
        continue

    matched[
        "hours_before_detection"
    ] = (
        (
            T0
            -
            matched["model_time"]
        )
        .dt.total_seconds()
        /
        3600
    )

    matched[
        "release_age_bin"
    ] = pd.cut(
        matched[
            "hours_before_detection"
        ],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        include_lowest=True,
    )

    matched = matched.dropna(
        subset=[
            "release_age_bin"
        ]
    )

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
            vessel_name = (
                names.mode().iloc[0]
            )

    matched_vessels[mmsi] = (
        vessel_name,
        matched,
    )


print(
    "Matched MMSIs:",
    len(matched_vessels)
)


# ============================================================
# SCENARIO-BY-SCENARIO SCORE
# ============================================================

records = []

print("\nScoring scenarios...")

for scenario_id in scenario_ids:

    print(
        f"Scenario {int(scenario_id):02d}/"
        f"{len(scenario_ids)}"
    )

    for (
        mmsi,
        (
            vessel_name,
            matched
        )
    ) in matched_vessels.items():

        time_scores = []

        for _, row in matched.iterrows():

            key = (
                int(scenario_id),
                row["model_time"],
            )

            cloud = clouds.get(
                key
            )

            if cloud is None:
                continue

            d = haversine_km(
                float(row["LON"]),
                float(row["LAT"]),
                cloud[:, 0],
                cloud[:, 1],
            )

            density = float(
                np.exp(
                    -0.5
                    *
                    (
                        d
                        /
                        SIGMA_KM
                    ) ** 2
                )
                .mean()
            )

            time_scores.append({
                "release_age_bin":
                    str(
                        row[
                            "release_age_bin"
                        ]
                    ),

                "model_time":
                    row[
                        "model_time"
                    ],

                "hours_before_detection":
                    float(
                        row[
                            "hours_before_detection"
                        ]
                    ),

                "density_score":
                    density,

                "min_distance_km":
                    float(
                        np.min(d)
                    ),

                "p10_distance_km":
                    float(
                        np.percentile(
                            d,
                            10
                        )
                    ),

                "fraction_5km":
                    float(
                        np.mean(
                            d <= 5
                        )
                    ),

                "fraction_10km":
                    float(
                        np.mean(
                            d <= 10
                        )
                    ),
            })

        if not time_scores:
            continue

        ts = pd.DataFrame(
            time_scores
        )

        for age_bin, a in ts.groupby(
            "release_age_bin"
        ):

            peak_idx = (
                a[
                    "density_score"
                ]
                .idxmax()
            )

            peak = a.loc[
                peak_idx
            ]

            records.append({
                "scenario_id":
                    int(
                        scenario_id
                    ),

                "MMSI":
                    mmsi,

                "VesselName":
                    vessel_name,

                "cerulean_candidate":
                    mmsi in CERULEAN,

                "release_age_bin":
                    age_bin,

                "matched_hours":
                    len(a),

                "point_score":
                    float(
                        a[
                            "density_score"
                        ].max()
                    ),

                "mean_score":
                    float(
                        a[
                            "density_score"
                        ].mean()
                    ),

                "peak_time":
                    peak[
                        "model_time"
                    ],

                "peak_age_hours":
                    float(
                        peak[
                            "hours_before_detection"
                        ]
                    ),

                "peak_min_distance_km":
                    float(
                        peak[
                            "min_distance_km"
                        ]
                    ),

                "peak_p10_distance_km":
                    float(
                        peak[
                            "p10_distance_km"
                        ]
                    ),

                "peak_fraction_5km":
                    float(
                        peak[
                            "fraction_5km"
                        ]
                    ),

                "peak_fraction_10km":
                    float(
                        peak[
                            "fraction_10km"
                        ]
                    ),
            })


scenario_scores = pd.DataFrame(
    records
)


# ============================================================
# RANK INSIDE EACH PHYSICAL SCENARIO
# ============================================================

scenario_scores[
    "scenario_rank"
] = (
    scenario_scores
    .groupby(
        [
            "scenario_id",
            "release_age_bin",
        ]
    )[
        "point_score"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


scenario_scores.to_parquet(
    SCENARIO_OUT,
    index=False
)


# ============================================================
# ROBUSTNESS AGGREGATION
# ============================================================

def q10(x):
    return float(
        np.quantile(
            x,
            0.10
        )
    )


def q90(x):
    return float(
        np.quantile(
            x,
            0.90
        )
    )


robust = (
    scenario_scores
    .groupby(
        [
            "MMSI",
            "VesselName",
            "cerulean_candidate",
            "release_age_bin",
        ],
        as_index=False,
    )
    .agg(
        matched_hours=(
            "matched_hours",
            "first"
        ),

        scenarios_evaluated=(
            "scenario_id",
            "nunique"
        ),

        score_mean=(
            "point_score",
            "mean"
        ),

        score_median=(
            "point_score",
            "median"
        ),

        score_q10=(
            "point_score",
            q10
        ),

        score_q90=(
            "point_score",
            q90
        ),

        rank_mean=(
            "scenario_rank",
            "mean"
        ),

        rank_median=(
            "scenario_rank",
            "median"
        ),

        rank_q10=(
            "scenario_rank",
            q10
        ),

        rank_q90=(
            "scenario_rank",
            q90
        ),

        peak_age_median=(
            "peak_age_hours",
            "median"
        ),

        peak_age_q10=(
            "peak_age_hours",
            q10
        ),

        peak_age_q90=(
            "peak_age_hours",
            q90
        ),
    )
)


# ============================================================
# TOP-K STABILITY
# ============================================================

topk = (
    scenario_scores
    .assign(
        top1=lambda d:
            d[
                "scenario_rank"
            ] <= 1,

        top3=lambda d:
            d[
                "scenario_rank"
            ] <= 3,

        top5=lambda d:
            d[
                "scenario_rank"
            ] <= 5,

        top10=lambda d:
            d[
                "scenario_rank"
            ] <= 10,
    )
    .groupby(
        [
            "MMSI",
            "VesselName",
            "cerulean_candidate",
            "release_age_bin",
        ],
        as_index=False,
    )
    .agg(
        top1_fraction=(
            "top1",
            "mean"
        ),

        top3_fraction=(
            "top3",
            "mean"
        ),

        top5_fraction=(
            "top5",
            "mean"
        ),

        top10_fraction=(
            "top10",
            "mean"
        ),
    )
)


robust = robust.merge(
    topk,
    on=[
        "MMSI",
        "VesselName",
        "cerulean_candidate",
        "release_age_bin",
    ],
    how="left",
)


age_order = {
    x: i
    for i, x
    in enumerate(
        AGE_LABELS
    )
}

robust[
    "_age_order"
] = (
    robust[
        "release_age_bin"
    ]
    .map(
        age_order
    )
)


robust = robust.sort_values(
    [
        "_age_order",
        "rank_median",
        "score_median",
    ],
    ascending=[
        True,
        True,
        False,
    ]
).drop(
    columns=[
        "_age_order"
    ]
)


robust.to_csv(
    ROBUST_OUT,
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
    "scenarios_evaluated",
    "score_median",
    "score_q10",
    "score_q90",
    "rank_median",
    "rank_q10",
    "rank_q90",
    "top3_fraction",
    "top5_fraction",
    "top10_fraction",
    "peak_age_median",
]


for age_bin in AGE_LABELS:

    print("\n")
    print("=" * 155)
    print(
        "ROBUST RELEASE AGE:",
        age_bin
    )
    print("=" * 155)

    sub = robust[
        robust[
            "release_age_bin"
        ]
        == age_bin
    ]

    print(
        sub[
            cols
        ]
        .head(12)
        .to_string(
            index=False
        )
    )


print("\n")
print("=" * 155)
print(
    "CERULEAN-LINKED ROBUSTNESS"
)
print("=" * 155)

known = robust[
    robust[
        "cerulean_candidate"
    ]
]

print(
    known[
        [
            "release_age_bin",
            *cols,
        ]
    ]
    .to_string(
        index=False
    )
)


print("\nSaved:")
print(SCENARIO_OUT)
print(ROBUST_OUT)
