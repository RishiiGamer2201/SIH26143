"""Backward-drift x AIS compatibility scores.

The frozen functions reproduce score_ais_density_v2.py, analyze_release_age.py, score_ensemble_robustness.py and
scripts/score_ais_density_v2_1.py (same operations, same column names) so the parity stage can compare bit-exactly.
"""
import numpy as np
import pandas as pd

from .ais import positions_v2, positions_v2_1, ns, vessel_name
from .geo import haversine_km


def kernel_stats(vlon, vlat, clon, clat, sigma_km):
    d = haversine_km(float(vlon), float(vlat), clon, clat)
    return d, {"min_distance_km": float(np.min(d)), "p10_distance_km": float(np.percentile(d, 10)),
               "median_distance_km": float(np.median(d)), "fraction_particles_2km": float(np.mean(d <= 2)),
               "fraction_particles_5km": float(np.mean(d <= 5)), "fraction_particles_10km": float(np.mean(d <= 10)),
               "density_score": float(np.exp(-0.5 * (d / sigma_km) ** 2).mean())}


def age_bins(hours, bins, labels):
    return pd.cut(hours, bins=bins, labels=labels, include_lowest=True)


# ------------------------------------------------------------------ frozen V2 (deterministic hindcast)
def v2_time_scores(plon, plat, times, ais, ref_mmsi, tol, sigma_km):
    ais = ais[(ais.BaseDateTime >= times[0] - tol) & (ais.BaseDateTime <= times[-1] + tol)].copy()
    recs, summ = [], []
    for mmsi, g in ais.groupby("MMSI"):
        g = g.sort_values("BaseDateTime").drop_duplicates("BaseDateTime")
        matched = positions_v2(times, g, tol)
        if matched.empty:
            continue
        name = vessel_name(g)
        rows = []
        for _, row in matched.iterrows():
            ti = int(np.where(times == row["model_time"])[0][0])
            clon, clat = plon[:, ti], plat[:, ti]
            ok = np.isfinite(clon) & np.isfinite(clat)
            if not ok.any():
                continue
            d, st = kernel_stats(row.LON, row.LAT, clon[ok], clat[ok], sigma_km)
            rec = {"MMSI": mmsi, "VesselName": name, "cerulean_candidate": mmsi in ref_mmsi, "model_time": row["model_time"],
                   "ais_time": row.BaseDateTime, "vessel_lon": float(row.LON), "vessel_lat": float(row.LAT), **st}
            rows.append(rec)
            recs.append(rec)
        if not rows:
            continue
        v = pd.DataFrame(rows)
        peak = v.loc[v.density_score.idxmax()]
        summ.append({"MMSI": mmsi, "VesselName": name, "cerulean_candidate": mmsi in ref_mmsi, "matched_hours": len(v),
                     "coverage_fraction": len(v) / len(times), "point_release_score": float(v.density_score.max()),
                     "continuous_release_score": float(v.density_score.mean()), "peak_time": peak.model_time,
                     "peak_min_distance_km": float(peak.min_distance_km), "peak_p10_distance_km": float(peak.p10_distance_km),
                     "peak_fraction_particles_5km": float(peak.fraction_particles_5km),
                     "peak_fraction_particles_10km": float(peak.fraction_particles_10km),
                     "best_min_distance_km": float(v.min_distance_km.min()), "best_p10_distance_km": float(v.p10_distance_km.min()),
                     "hours_density_gt_005": int((v.density_score >= 0.05).sum()),
                     "hours_density_gt_01": int((v.density_score >= 0.10).sum())})
    summary = pd.DataFrame(summ)
    for s, r in (("point_release_score", "point_rank"), ("continuous_release_score", "continuous_rank")):
        summary[r] = summary[s].rank(ascending=False, method="min").astype(int)
    return pd.DataFrame(recs), summary.sort_values("point_release_score", ascending=False)


def release_age_table(ts, t0, bins, labels):
    """analyze_release_age.py."""
    df = ts.copy()
    df["hours_before_detection"] = (t0 - pd.to_datetime(df.model_time, utc=True)).dt.total_seconds() / 3600.0
    df["release_age_bin"] = age_bins(df.hours_before_detection, bins, labels)
    rows = []
    for (mmsi, name, cer, b), g in df.groupby(["MMSI", "VesselName", "cerulean_candidate", "release_age_bin"], observed=True):
        peak = g.loc[g.density_score.idxmax()]
        rows.append({"MMSI": mmsi, "VesselName": name, "cerulean_candidate": cer, "release_age_bin": str(b),
                     "matched_hours_in_bin": len(g), "peak_density_score": float(g.density_score.max()),
                     "mean_density_score": float(g.density_score.mean()), "peak_time": peak.model_time,
                     "peak_age_hours": float(peak.hours_before_detection), "peak_min_distance_km": float(peak.min_distance_km),
                     "peak_p10_distance_km": float(peak.p10_distance_km), "fraction_particles_5km": float(peak.fraction_particles_5km),
                     "fraction_particles_10km": float(peak.fraction_particles_10km)})
    out = pd.DataFrame(rows)
    out["rank_in_age_bin"] = out.groupby("release_age_bin").peak_density_score.rank(ascending=False, method="min").astype(int)
    return out.sort_values(["release_age_bin", "rank_in_age_bin"])


# ------------------------------------------------------------------ frozen ensemble robustness
def ensemble_scenario_scores(ens, ais, t0, ref_mmsi, tol, sigma_km, bins, labels):
    """score_ensemble_robustness.py, scenario part."""
    clouds = {(int(s), t): g[["lon", "lat"]].to_numpy(dtype=float) for (s, t), g in ens.groupby(["scenario_id", "time"], sort=False)}
    model_times = pd.DatetimeIndex(sorted(ens["time"].unique()))
    ais = ais[(ais.BaseDateTime >= model_times.min() - tol) & (ais.BaseDateTime <= model_times.max() + tol)].copy()
    matched_vessels = {}
    for mmsi, g in ais.groupby("MMSI"):
        g = g.sort_values("BaseDateTime").drop_duplicates("BaseDateTime")
        m = positions_v2(model_times, g, tol).copy()
        if m.empty:
            continue
        m["hours_before_detection"] = (t0 - m["model_time"]).dt.total_seconds() / 3600
        m["release_age_bin"] = age_bins(m["hours_before_detection"], bins, labels)
        m = m.dropna(subset=["release_age_bin"])
        matched_vessels[mmsi] = (vessel_name(g), m)
    recs = []
    for sid in sorted(ens["scenario_id"].unique()):
        for mmsi, (name, m) in matched_vessels.items():
            ts = []
            for _, row in m.iterrows():
                cloud = clouds.get((int(sid), row["model_time"]))
                if cloud is None:
                    continue
                d = haversine_km(float(row["LON"]), float(row["LAT"]), cloud[:, 0], cloud[:, 1])
                ts.append({"release_age_bin": str(row["release_age_bin"]), "model_time": row["model_time"],
                           "hours_before_detection": float(row["hours_before_detection"]),
                           "density_score": float(np.exp(-0.5 * (d / sigma_km) ** 2).mean()), "min_distance_km": float(np.min(d)),
                           "p10_distance_km": float(np.percentile(d, 10)), "fraction_5km": float(np.mean(d <= 5)),
                           "fraction_10km": float(np.mean(d <= 10))})
            if not ts:
                continue
            ts = pd.DataFrame(ts)
            for b, a in ts.groupby("release_age_bin"):
                peak = a.loc[a.density_score.idxmax()]
                recs.append({"scenario_id": int(sid), "MMSI": mmsi, "VesselName": name, "cerulean_candidate": mmsi in ref_mmsi,
                             "release_age_bin": b, "matched_hours": len(a), "point_score": float(a.density_score.max()),
                             "mean_score": float(a.density_score.mean()), "peak_time": peak["model_time"],
                             "peak_age_hours": float(peak["hours_before_detection"]),
                             "peak_min_distance_km": float(peak["min_distance_km"]), "peak_p10_distance_km": float(peak["p10_distance_km"]),
                             "peak_fraction_5km": float(peak["fraction_5km"]), "peak_fraction_10km": float(peak["fraction_10km"])})
    sc = pd.DataFrame(recs)
    sc["scenario_rank"] = sc.groupby(["scenario_id", "release_age_bin"])["point_score"].rank(ascending=False, method="min").astype(int)
    return sc


def q10(x):
    return float(np.quantile(x, 0.10))


def q90(x):
    return float(np.quantile(x, 0.90))


def robust_table(sc, labels, keys=("MMSI", "VesselName", "cerulean_candidate", "release_age_bin"),
                 member="scenario_id", score="point_score", rank="scenario_rank"):
    """score_ensemble_robustness.py, aggregation part (generic over key columns)."""
    keys = list(keys)
    robust = sc.groupby(keys, as_index=False).agg(
        matched_hours=("matched_hours", "first"), scenarios_evaluated=(member, "nunique"),
        score_mean=(score, "mean"), score_median=(score, "median"), score_q10=(score, q10), score_q90=(score, q90),
        rank_mean=(rank, "mean"), rank_median=(rank, "median"), rank_q10=(rank, q10), rank_q90=(rank, q90),
        peak_age_median=("peak_age_hours", "median"), peak_age_q10=("peak_age_hours", q10), peak_age_q90=("peak_age_hours", q90))
    topk = sc.assign(top1=lambda d: d[rank] <= 1, top3=lambda d: d[rank] <= 3, top5=lambda d: d[rank] <= 5,
                     top10=lambda d: d[rank] <= 10).groupby(keys, as_index=False).agg(
        top1_fraction=("top1", "mean"), top3_fraction=("top3", "mean"), top5_fraction=("top5", "mean"), top10_fraction=("top10", "mean"))
    robust = robust.merge(topk, on=keys, how="left")
    robust["_o"] = robust["release_age_bin"].map({x: i for i, x in enumerate(labels)})
    return robust.sort_values(["_o", "rank_median", "score_median"], ascending=[True, True, False]).drop(columns="_o")


# ------------------------------------------------------------------ frozen V2.1 (interpolated AIS)
def v2_1_time_scores(plon, plat, times, ais, cfg, t0):
    ais = ais[ais.LAT.between(-90, 90) & ais.LON.between(-180, 180)]
    margin = pd.Timedelta(minutes=max(cfg["max_bracket_gap_min"], cfg["edge_nearest_tol_min"]))
    ais = ais[(ais.BaseDateTime >= times[0] - margin) & (ais.BaseDateTime <= times[-1] + margin)]
    t_model = times.as_unit("ns").asi8
    recs = []
    for mmsi, g in ais.groupby("MMSI"):
        g = g.sort_values("BaseDateTime").drop_duplicates("BaseDateTime")
        name = vessel_name(g)
        lon, lat, method, gap, off = positions_v2_1(t_model, ns(g.BaseDateTime), g.LON.values, g.LAT.values, cfg)
        for k in range(len(t_model)):
            if method[k] == "unavailable":
                continue
            ok = np.isfinite(plon[:, k]) & np.isfinite(plat[:, k])
            d = haversine_km(lon[k], lat[k], plon[ok, k], plat[ok, k])
            recs.append({"MMSI": mmsi, "VesselName": name, "model_time": times[k], "method": method[k],
                         "bracket_gap_min": gap[k], "ais_offset_min": off[k], "vessel_lon": lon[k], "vessel_lat": lat[k],
                         "min_distance_km": float(d.min()), "p10_distance_km": float(np.percentile(d, 10)),
                         "median_distance_km": float(np.median(d)), "fraction_particles_5km": float(np.mean(d <= 5)),
                         "density_score": float(np.exp(-0.5 * (d / cfg["sigma_km"]) ** 2).mean())})
    ts = pd.DataFrame(recs)
    ts["hours_before_detection"] = (t0 - ts.model_time).dt.total_seconds() / 3600
    ts["release_age_bin"] = pd.cut(ts.hours_before_detection, bins=cfg["age_bins_h"], labels=cfg["age_labels"], include_lowest=True)
    return ts
