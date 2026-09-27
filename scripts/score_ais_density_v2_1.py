"""AIS density attribution V2.1 = frozen V2 score with interpolated AIS positions.

Only the vessel-position step differs from V2 (score_ais_density_v2.py, frozen):
  V2   : nearest AIS report within +/-20 min of each hourly model time (no interpolation)
  V2.1 : per model time t, per vessel
         - exact report at t                                  -> method "exact"
         - reports bracketing t with gap <= max_bracket_gap and implied speed <= max_implied_speed
                                                              -> linear interpolation ("interp")
         - otherwise nearest report within +/- edge_nearest_tol (one-sided / gap too long)
                                                              -> "edge_nearest"
         - otherwise                                          -> "unavailable" (not scored)
Kernel, sigma, release-age bins, point/continuous aggregation: identical to V2.

Usage: python scripts/score_ais_density_v2_1.py [--config configs/ais_v2_1.json] [--out results/incident_001_v2_1]
Writes only to --out (refuses results/incident_001). Then compares against the frozen V2 outputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def wrap180(x):
    return (np.asarray(x) + 180.0) % 360.0 - 180.0


def vessel_positions(t_model, t_obs, lon, lat, cfg):
    """t_* as int64 ns. Returns lon, lat, method, gap_min, offset_min (arrays over t_model)."""
    n = len(t_model)
    out_lon, out_lat = np.full(n, np.nan), np.full(n, np.nan)
    method = np.full(n, "unavailable", dtype=object)
    gap_min, off_min = np.full(n, np.nan), np.full(n, np.nan)
    max_gap = cfg["max_bracket_gap_min"] * 60e9
    edge_tol = cfg["edge_nearest_tol_min"] * 60e9
    j = np.searchsorted(t_obs, t_model, side="left")  # first obs >= t
    for k, t in enumerate(t_model):
        nxt, prv = j[k], j[k] - 1
        if nxt < len(t_obs) and t_obs[nxt] == t:
            out_lon[k], out_lat[k], method[k], gap_min[k], off_min[k] = lon[nxt], lat[nxt], "exact", 0.0, 0.0
            continue
        if 0 <= prv and nxt < len(t_obs):
            g = t_obs[nxt] - t_obs[prv]
            d_km = haversine_km(lon[prv], lat[prv], lon[nxt], lat[nxt])
            speed_kn = d_km / 1.852 / (g / 3.6e12)
            if g <= max_gap and speed_kn <= cfg["max_implied_speed_kn"]:
                f = (t - t_obs[prv]) / g
                out_lon[k] = wrap180(lon[prv] + f * wrap180(lon[nxt] - lon[prv]))  # antimeridian-safe
                out_lat[k] = lat[prv] + f * (lat[nxt] - lat[prv])
                method[k], gap_min[k] = "interp", g / 60e9
                off_min[k] = min(t - t_obs[prv], t_obs[nxt] - t) / 60e9
                continue
        cands = [i for i in (prv, nxt) if 0 <= i < len(t_obs)]
        if cands:
            i = min(cands, key=lambda i: abs(t_obs[i] - t))
            if abs(t_obs[i] - t) <= edge_tol:
                out_lon[k], out_lat[k], method[k] = lon[i], lat[i], "edge_nearest"
                off_min[k] = abs(t_obs[i] - t) / 60e9
    return out_lon, out_lat, method, gap_min, off_min


def age_bin_table(ts, labels):
    """Same aggregation as frozen analyze_release_age.py (peak density per vessel per bin)."""
    rows = []
    for (mmsi, name, age_bin), g in ts.groupby(["MMSI", "VesselName", "release_age_bin"], observed=True):
        peak = g.loc[g.density_score.idxmax()]
        rows.append({"MMSI": mmsi, "VesselName": name, "release_age_bin": str(age_bin),
                     "matched_hours_in_bin": len(g), "peak_density_score": float(g.density_score.max()),
                     "mean_density_score": float(g.density_score.mean()), "peak_age_hours": float(peak.hours_before_detection),
                     "peak_min_distance_km": float(peak.min_distance_km), "peak_method": peak.method})
    out = pd.DataFrame(rows)
    out["rank_in_age_bin"] = out.groupby("release_age_bin").peak_density_score.rank(ascending=False, method="min").astype(int)
    out["_o"] = out.release_age_bin.map({l: i for i, l in enumerate(labels)})
    return out.sort_values(["_o", "rank_in_age_bin"]).drop(columns="_o")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/ais_v2_1.json"))
    ap.add_argument("--out", default=str(ROOT / "results/incident_001_v2_1"))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out).resolve()
    assert out != (ROOT / "results/incident_001").resolve(), "refusing to write into frozen results/incident_001"
    out.mkdir(parents=True, exist_ok=True)
    T0 = pd.Timestamp(cfg["incident_t0_utc"])

    ds = xr.open_dataset(ROOT / cfg["hindcast_nc"])
    times = pd.to_datetime(ds.time.values, utc=True)
    order = np.argsort(times.values)
    times = times[order]
    plon, plat = ds.lon.values[:, order], ds.lat.values[:, order]
    ds.close()
    t_model = times.as_unit("ns").asi8

    ais = pd.read_parquet(ROOT / cfg["ais_parquet"])
    ais["BaseDateTime"] = pd.to_datetime(ais.BaseDateTime, utc=True, errors="coerce")
    ais["MMSI"] = ais.MMSI.astype(str).str.replace(r"\.0$", "", regex=True)
    ais["LAT"], ais["LON"] = pd.to_numeric(ais.LAT, errors="coerce"), pd.to_numeric(ais.LON, errors="coerce")
    ais = ais.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON"])
    ais = ais[ais.LAT.between(-90, 90) & ais.LON.between(-180, 180)]
    margin = pd.Timedelta(minutes=max(cfg["max_bracket_gap_min"], cfg["edge_nearest_tol_min"]))
    ais = ais[(ais.BaseDateTime >= times[0] - margin) & (ais.BaseDateTime <= times[-1] + margin)]

    recs, pos_diag = [], []
    for mmsi, g in ais.groupby("MMSI"):
        g = g.sort_values("BaseDateTime").drop_duplicates("BaseDateTime")
        names = g.VesselName.dropna().astype(str).str.strip()
        names = names[names != ""]
        name = names.mode().iloc[0] if len(names) else ""
        lon, lat, method, gap, off = vessel_positions(t_model, g.BaseDateTime.dt.tz_convert(None).values.astype("datetime64[ns]").astype("int64"),
                                                      g.LON.values, g.LAT.values, cfg)
        for k in range(len(t_model)):
            pos_diag.append((mmsi, times[k], method[k]))
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
    ts["hours_before_detection"] = (T0 - ts.model_time).dt.total_seconds() / 3600
    ts["release_age_bin"] = pd.cut(ts.hours_before_detection, bins=cfg["age_bins_h"], labels=cfg["age_labels"], include_lowest=True)
    ts.to_parquet(out / "ais_density_time_scores_v2_1.parquet", index=False)

    n_t = len(times)
    summ = ts.groupby(["MMSI", "VesselName"]).agg(
        matched_hours=("density_score", "size"), n_exact=("method", lambda m: (m == "exact").sum()),
        n_interp=("method", lambda m: (m == "interp").sum()), n_edge=("method", lambda m: (m == "edge_nearest").sum()),
        point_release_score=("density_score", "max"), continuous_release_score=("density_score", "mean"),
        best_min_distance_km=("min_distance_km", "min")).reset_index()
    summ["coverage_fraction"] = summ.matched_hours / n_t
    for s, r in (("point_release_score", "point_rank"), ("continuous_release_score", "continuous_rank")):
        summ[r] = summ[s].rank(ascending=False, method="min").astype(int)
    summ = summ.sort_values("point_release_score", ascending=False)
    summ.to_csv(out / "ais_density_scores_v2_1.csv", index=False)
    age = age_bin_table(ts, cfg["age_labels"])
    age.to_csv(out / "release_age_scores_v2_1.csv", index=False)

    diag = pd.DataFrame(pos_diag, columns=["MMSI", "model_time", "method"])
    method_counts = diag.method.value_counts().to_dict()

    # ------------------------------------------------------------ V2 vs V2.1 (V2 files read-only)
    v2 = pd.read_csv(ROOT / cfg["v2_summary_csv"], dtype={"MMSI": str})
    cmp = v2[["MMSI", "VesselName", "matched_hours", "coverage_fraction", "point_release_score", "continuous_release_score",
              "point_rank", "continuous_rank"]].merge(
        summ[["MMSI", "matched_hours", "coverage_fraction", "point_release_score", "continuous_release_score",
              "point_rank", "continuous_rank", "n_interp", "n_edge"]], on="MMSI", how="outer", suffixes=("_v2", "_v2_1"))
    for c in ("point_release_score", "continuous_release_score", "point_rank", "continuous_rank", "matched_hours", "coverage_fraction"):
        cmp[f"d_{c}"] = cmp[f"{c}_v2_1"] - cmp[f"{c}_v2"]
    cmp = cmp.sort_values("point_rank_v2")
    cmp.to_csv(out / "compare_v2_vs_v2_1_summary.csv", index=False)

    a2 = pd.read_csv(ROOT / cfg["v2_release_age_csv"], dtype={"MMSI": str})
    acmp = a2[["MMSI", "VesselName", "release_age_bin", "matched_hours_in_bin", "peak_density_score", "peak_age_hours", "rank_in_age_bin"]].merge(
        age[["MMSI", "release_age_bin", "matched_hours_in_bin", "peak_density_score", "peak_age_hours", "rank_in_age_bin", "peak_method"]],
        on=["MMSI", "release_age_bin"], how="outer", suffixes=("_v2", "_v2_1"))
    acmp["d_rank"] = acmp.rank_in_age_bin_v2_1 - acmp.rank_in_age_bin_v2
    acmp["d_score"] = acmp.peak_density_score_v2_1 - acmp.peak_density_score_v2
    acmp.to_csv(out / "compare_v2_vs_v2_1_release_age.csv", index=False)

    per_bin = []
    for b in cfg["age_labels"]:
        x = acmp[acmp.release_age_bin == b]
        both = x.dropna(subset=["rank_in_age_bin_v2", "rank_in_age_bin_v2_1"])
        top = lambda col, k: set(x.loc[x[col] <= k, "MMSI"])
        per_bin.append({"bin": b, "n_v2": int(x.rank_in_age_bin_v2.notna().sum()), "n_v2_1": int(x.rank_in_age_bin_v2_1.notna().sum()),
                        "spearman_score": float(spearmanr(both.peak_density_score_v2, both.peak_density_score_v2_1).statistic),
                        "top10_overlap": len(top("rank_in_age_bin_v2", 10) & top("rank_in_age_bin_v2_1", 10)),
                        "top6_identical_order": list(x.sort_values("rank_in_age_bin_v2").head(6).MMSI) ==
                                                list(x.sort_values("rank_in_age_bin_v2_1").head(6).MMSI),
                        "max_abs_rank_change_top10_v2": float(x[x.rank_in_age_bin_v2 <= 10].d_rank.abs().max())})
    per_bin = pd.DataFrame(per_bin)
    per_bin.to_csv(out / "compare_v2_vs_v2_1_per_bin.csv", index=False)

    diag_json = {"config": cfg, "vessel_time_positions": method_counts, "vessels_scored_v2": int(len(v2)),
                 "vessels_scored_v2_1": int(len(summ)), "model_times": n_t}
    (out / "v2_1_diagnostics.json").write_text(json.dumps(diag_json, indent=2, default=str))

    pd.set_option("display.width", 250)
    print("position methods (vessel x model-time):", method_counts)
    print("vessels scored  V2:", len(v2), " V2.1:", len(summ))
    print("\nTOP 15 point-release V2.1 vs V2")
    print(cmp.sort_values("point_rank_v2_1").head(15)[["MMSI", "VesselName", "point_rank_v2", "point_rank_v2_1", "point_release_score_v2",
          "point_release_score_v2_1", "matched_hours_v2", "matched_hours_v2_1", "n_interp", "n_edge"]].round(4).to_string(index=False))
    print("\nPER RELEASE-AGE BIN"); print(per_bin.round(3).to_string(index=False))
    for b in cfg["age_labels"]:
        x = acmp[acmp.release_age_bin == b].sort_values("rank_in_age_bin_v2_1").head(8)
        print(f"\n{b}: top 8 V2.1 (rank V2 -> V2.1)")
        print(x[["VesselName", "MMSI", "rank_in_age_bin_v2", "rank_in_age_bin_v2_1", "peak_density_score_v2", "peak_density_score_v2_1",
                 "peak_age_hours_v2_1", "peak_method"]].round(4).to_string(index=False))
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
