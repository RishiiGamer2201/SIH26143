"""Export one oilwatch run into compact, map-ready JSON for the analyst portal.

  .venv/Scripts/python portal/build_data.py incident_001 e008_v3_scene [--twin results/twin_v1]

Writes portal/web/public/data/<incident>/ : meta, events, reference (post-hoc layer), suspects, ais_tracks,
ships (SAR), platforms (BOEM), backward/<event>.json, forward/<event>.json, forecast/<event>.json, twin.json.
Particle sets are subsampled for the browser; the full-resolution data stays in runs/.
"""
import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oilwatch.ais import load_ais, ns, positions_v2_1  # noqa: E402
from oilwatch.config import Incident, load_json, load_params  # noqa: E402
from oilwatch.infrastructure import load_platforms  # noqa: E402

R4 = lambda a: np.round(np.asarray(a, float), 4)
SUSPECT_FIELDS = ["suspect_rank", "tier", "v31_decision", "direction_agreement", "MMSI", "VesselName", "source_type",
                  "source_type_basis", "motion_state", "combined", "bwd_point", "bwd_weighted", "bwd_rank_median",
                  "bwd_top3_fraction", "bwd_peak_age_median", "fwd_F", "fwd_precision", "fwd_recall", "fwd_member_support",
                  "fss_1km", "fss_2km", "fss_skilful", "fwd_window_start_h", "fwd_window_end_h", "t0_lon", "t0_lat",
                  "dist_to_slick_end_km", "ais_gaps_ge_60min", "ais_longest_gap_min", "ais_gap_intervals",
                  "flag_ais_gap_in_release_window", "flag_ais_silent_seen_by_sar_near_slick", "flag_fresh_discharge_match",
                  "flag_identity_anomaly", "flag_window_at_48h_boundary", "sar_label", "sar_dist_to_slick_km",
                  "nearest_boem_structure", "nearest_boem_km"]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":"), default=lambda o: None if o is pd.NaT else
                               (float(o) if isinstance(o, (np.floating, np.integer)) else str(o))))


def clean(v):
    if isinstance(v, (float, np.floating)) and not np.isfinite(v):
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("incident")
    ap.add_argument("run_id")
    ap.add_argument("--twin", default="results/twin_v1")
    ap.add_argument("--particles", type=int, default=350)
    a = ap.parse_args()
    inc, P = Incident.load(a.incident), load_params()
    run = ROOT / "runs" / inc.id / a.run_id
    out = ROOT / "portal/web/public/data" / inc.id
    rng = np.random.default_rng(0)
    t0 = inc.t0

    # ------------------------------------------------------------------ events + age + darkness + context
    cand = gpd.read_file(run / "seed/candidates.geojson")
    ages = pd.read_csv(run / "fuse/age_summary.csv").set_index("cluster_id")
    cues = pd.read_csv(run / "fuse/age_cues.csv")
    dark = pd.read_csv(run / "sar/darkness.csv").set_index("cluster_id") if (run / "sar/darkness.csv").exists() else None
    beach = pd.read_csv(run / "forecast/beaching.csv").set_index("cluster_id")
    fn = pd.read_csv(run / "backward/funnel.csv").set_index("cluster_id")
    feats = []
    for _, c in cand.iterrows():
        p = {k: clean(c[k]) for k in ["cluster_id", "potential_rank", "area_m2", "length_m", "width_proxy_m", "axis_bearing_deg",
                                      "confidence", "wind_speed_m_s", "wind_from_deg", "wind_class", "slick_potential",
                                      "eligible", "selected", "ineligible_reason", "n_parts", "centroid_lon", "centroid_lat",
                                      "end1_lon", "end1_lat", "end2_lon", "end2_lat"]}
        cid = int(c.cluster_id)
        if cid in ages.index:
            s = ages.loc[cid]
            p.update(age_lo_h=clean(s.age_lo_h), age_hi_h=clean(s.age_hi_h), age_basis=s.estimate_basis,
                     age_conflict=bool(s.age_conflict), age_cues=cues[cues.cluster_id == cid][["cue", "lo_h", "hi_h", "detail"]]
                     .replace({np.nan: None}).to_dict("records"))
        if dark is not None and cid in dark.index:
            d = dark.loc[cid]
            p.update(damping_db=clean(d.get("damping_ratio_db")), darkness_class=d.get("darkness_class"),
                     edge_sharpness=clean(d.get("edge_sharpness")))
        if cid in beach.index:
            p.update(p_beached_72h=clean(beach.loc[cid].p_beached_72h))
        if cid in fn.index:
            p.update(funnel={k: int(v) for k, v in fn.loc[cid].items()})
        feats.append({"type": "Feature", "properties": p, "geometry": json.loads(gpd.GeoSeries([c.geometry]).to_json())["features"][0]["geometry"]})
    dump(out / "events.geojson", {"type": "FeatureCollection", "features": feats})
    ref = json.loads((ROOT / inc.spec["reference"]["slick_geojson"]).read_text())
    dump(out / "reference.geojson", ref)

    # ------------------------------------------------------------------ suspects (per event)
    sus = pd.read_csv(run / "fuse/suspects.csv", dtype={"MMSI": str})
    top = sus[sus.suspect_rank <= 15]
    sj = {}
    for cid, g in top.groupby("cluster_id"):
        sj[int(cid)] = [{k: clean(r[k]) for k in SUSPECT_FIELDS if k in r} for _, r in g.sort_values("suspect_rank").iterrows()]
    dump(out / "suspects.json", sj)

    # ------------------------------------------------------------------ AIS tracks (suspects + SAR-matched), hourly -48..+6 h
    ais = load_ais(inc.ais_attribution())
    ships = pd.read_csv(run / "sar/ship_detections.csv")
    want = set(top.MMSI) | set(ships.match_id.dropna().astype(str).str.replace(r"\.0$", "", regex=True)) | \
        set(ships.silent_mmsi.dropna().astype(float).astype("int64").astype(str))
    cfg21 = load_json(ROOT / P["frozen"]["v2_1_config"])
    grid = pd.date_range(t0 - pd.Timedelta(hours=48), t0 + pd.Timedelta(hours=6), freq="15min")
    tracks = []
    for m, g in ais[ais.MMSI.isin(want)].groupby("MMSI"):
        g = g.sort_values("BaseDateTime").drop_duplicates("BaseDateTime")
        lon, lat, meth, _, _ = positions_v2_1(grid.as_unit("ns").asi8, ns(g.BaseDateTime), g.LON.values, g.LAT.values, cfg21)
        ok = meth != "unavailable"
        if ok.sum() < 2:
            continue
        name = g.VesselName.dropna().iloc[0] if g.VesselName.notna().any() else ""
        tracks.append({"mmsi": m, "name": name, "type": clean(g.VesselType.dropna().iloc[0]) if g.VesselType.notna().any() else None,
                       "path": R4(np.c_[lon[ok], lat[ok]]).tolist(),
                       "t": np.round(((grid[ok] - t0).total_seconds() / 3600).to_numpy(), 2).tolist(),
                       "reports_t": np.round(((g.BaseDateTime - t0).dt.total_seconds() / 3600).to_numpy(), 2).tolist()})
    dump(out / "ais_tracks.json", tracks)

    # ------------------------------------------------------------------ SAR ships, platforms
    ships = ships.replace({np.nan: None})
    dump(out / "ships.json", [{k: clean(r[k]) for k in ["lon", "lat", "label", "match_name", "match_km", "peak_db", "size_px",
                                                        "silent_name", "silent_last_report_min"] if k in r} for _, r in ships.iterrows()])
    plat = load_platforms(inc)
    b = cand.total_bounds
    if plat is not None:
        pl = plat[plat.lon.between(b[0] - 1.5, b[2] + 1.5) & plat.lat.between(b[1] - 1.5, b[3] + 1.5)]
        dump(out / "platforms.json", [{"lon": round(r.lon, 5), "lat": round(r.lat, 5), "name": r["name"], "type": r.STRUC_TYPE_CODE,
                                       "major": r.MAJ_STRUC_FLAG == "Y"} for _, r in pl.iterrows()])

    # ------------------------------------------------------------------ backward clouds (per event, per hour, pooled members)
    parts = pd.read_parquet(run / "backward/particles.parquet", columns=["cluster_id", "time", "lon", "lat", "member"])
    parts = parts[np.isfinite(parts.lon)]
    parts["tau"] = ((t0 - parts.time).dt.total_seconds() / 3600).round().astype(int)
    for cid, g in parts.groupby("cluster_id"):
        frames = {}
        for tau, h in g.groupby("tau"):
            k = rng.choice(len(h), min(a.particles, len(h)), replace=False)
            frames[int(tau)] = R4(h[["lon", "lat"]].to_numpy()[k]).ravel().tolist()
        dump(out / f"backward/{int(cid)}.json", frames)

    # ------------------------------------------------------------------ forward virtual oil at T0 for top suspects
    fpos = pd.read_parquet(run / "forward/virtual_oil_at_t0.parquet")
    for cid, g in top.groupby("cluster_id"):
        res = {}
        for _, s in g[g.fwd_F > 0].head(6).iterrows():
            v = fpos[(fpos.MMSI == s.MMSI) & (fpos.hour >= s.fwd_window_start_h) & (fpos.hour < s.fwd_window_end_h)]
            v = v[np.isfinite(v.lon_t0)]
            k = rng.choice(len(v), min(600, len(v)), replace=False) if len(v) else []
            res[s.MMSI] = R4(v[["lon_t0", "lat_t0"]].to_numpy()[k]).ravel().tolist()
        dump(out / f"forward/{int(cid)}.json", res)

    # ------------------------------------------------------------------ forecast (central oil, 3-hourly, surface flag)
    fc = pd.read_parquet(run / "forecast/forecast_particles.parquet", columns=["cluster_id", "oil_type", "lead_h", "lon", "lat", "z", "status"])
    fc = fc[(fc.oil_type == P["forecast_v1"]["oil_central"]) & (fc.lead_h % 3 == 0)]
    for cid, g in fc.groupby("cluster_id"):
        frames = {}
        for lead, h in g.groupby("lead_h"):
            k = rng.choice(len(h), min(a.particles, len(h)), replace=False)
            hh = h.iloc[k]
            frames[int(lead)] = {"xy": R4(hh[["lon", "lat"]].to_numpy()).ravel().tolist(),
                                 "surface": (hh.z > -0.5).astype(int).tolist()}
        dump(out / f"forecast/{int(cid)}.json", frames)
    mb = pd.read_csv(run / "forecast/mass_budget.csv")
    fate = mb.groupby(["oil_type", "lead_h"])[["frac_oil", "frac_evaporated", "frac_dispersed"]].mean().reset_index()

    # ------------------------------------------------------------------ twin skill + meta
    tw = ROOT / a.twin
    twin = {}
    if tw.exists():
        twin = {"skill": pd.read_csv(tw / "skill_table.csv").replace({np.nan: None}).to_dict("records"),
                "curve": pd.read_csv(tw / "operating_curve.csv").replace({np.nan: None}).to_dict("records"),
                "calibration": {k: v for k, v in json.loads((tw / "calibration.json").read_text()).items() if k != "split"},
                "summary": json.loads((tw / "summary.json").read_text())}
    dump(out / "twin.json", twin)
    geo = json.loads((run / "sar/geolocation.json").read_text()) if (run / "sar/geolocation.json").exists() else {}
    meta = {"incident": inc.id, "run_id": a.run_id, "t0": str(t0), "scene": inc.spec["sentinel1_scene_id"],
            "bounds": [float(x) for x in b], "n_events": int(len(cand)), "n_selected": int(cand.selected.sum()),
            "n_ais_vessels": int(ais.MMSI.nunique()), "n_ship_detections": int(len(ships)),
            "ship_labels": ships.label.value_counts().to_dict(), "geolocation": geo,
            "fate": fate.to_dict("records"), "sst_assumption_c": P["forecast_v1"]["sea_water_temperature_c"],
            "reference_name": inc.spec["reference"]["name"], "reference_vessels": sorted(inc.reference_mmsi),
            "disclaimer": "Rankings are evidence of spatio-temporal consistency between AIS tracks and modelled oil drift, "
                          "never proof of culpability. The reference (Cerulean) layer is post-hoc validation only.",
            "physics": P["physics_v3"], "v31_rule": twin.get("calibration", {}).get("rule")}
    dump(out / "meta.json", meta)
    print(f"portal data -> {out}")


if __name__ == "__main__":
    main()
