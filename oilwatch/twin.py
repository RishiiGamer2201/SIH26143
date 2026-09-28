"""Twin experiments: MEASURED attribution skill of the v3 pipeline (no published twin study reports this; E008 next step).

1. Truth: a real AIS vessel (real track, real forcing) secretly discharges along its track for d in {1,2,3,6} h, ending
   tau_end hours before T0 (strata 0-3 / 3-12 / 12-24 / 24-40 h). The virtual oil is drifted to T0 with TRUTH physics
   deliberately outside the attribution ensemble (windage 2.5 %, K 5 m2/s, current unc 0.07, wind unc 0.7, own seed),
   so the inverse problem is not solved by construction.
2. Observation: particles at T0 -> 100 m grid (>= 2 particles) -> closing -> polygon; then degraded like our real product:
   buffered +60 m (E007 strands are 2-3x too wide) and shifted by a random 40 m (measured geolocation median, E008 SAR).
   Cases below the E008 minimum area or outside the forcing domain are discarded (logged).
3. Blind attribution with the real scene AIS (all ~1,700 vessels are distractors): v3 backward + forward + fusion,
   identical code and parameters to the pipeline.
4. Scored: rank of the true vessel (fused / backward-only / forward-only), funnel pass, tier, release-window error.
   'Hidden culprit': the same case with the true vessel's AIS removed -> how often an innocent vessel is still
   labelled consistent/strongly consistent (false-accusation rate).
Limitation: truth and model share the same forcing fields (no forcing error) -> skill is an UPPER bound for
forcing-limited cases; parameter mismatch and observation degradation are represented.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import shapes
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely import affinity

from . import backward, drift, forward, fusion
from .ais import load_ais
from .config import ROOT, load_json
from .context import Forcing
from .geo import axis_endpoints, geodesic_area_m2, laea, to_metric
from .manifest import Stage

STRATA = [(0, 3), (3, 12), (12, 24), (24, 40)]


def sample_cases(pool, n, T, rng, bounds):
    """pool: release points (MMSI, time, lon0, lat0, tau_release_h) of every vessel. Returns case table."""
    cases, tried = [], 0
    by_v = {m: g for m, g in pool.groupby("MMSI")}
    vessels = np.array(sorted(by_v))
    step_h = T["release_step_min"] / 60
    while len(cases) < n and tried < 50 * n:
        tried += 1
        lo, hi = STRATA[len(cases) % len(STRATA)]
        d = int(rng.choice(T["durations_h"]))
        tau_end = float(rng.uniform(lo, hi))
        tau_end = round(tau_end / step_h) * step_h
        m = rng.choice(vessels)
        g = by_v[m]
        w = g[(g.tau_release_h > tau_end) & (g.tau_release_h <= tau_end + d)]
        need = d / step_h
        if len(w) < T["min_coverage"] * need:
            continue
        x0, y0, x1, y1 = bounds
        mg = T["domain_margin_deg"]
        if not (w.lon0.between(x0 + mg, x1 - mg).all() and w.lat0.between(y0 + mg, y1 - mg).all()):
            continue
        spread_km = float(np.hypot((w.lon0.max() - w.lon0.min()) * 99, (w.lat0.max() - w.lat0.min()) * 111))
        cases.append({"case_id": 1000 + len(cases), "MMSI": m, "tau_end_h": tau_end, "duration_h": d,
                      "tau_start_h": tau_end + d, "stratum": f"{lo}-{hi}h", "track_extent_km": spread_km,
                      "moving": spread_km > 1.0})
    return pd.DataFrame(cases), tried


def truth_slicks(cases, pool, t0, rdr, P, T, rng):
    """Drift the secret discharges with truth physics and turn the T0 particles into degraded slick polygons."""
    els = []
    for _, c in cases.iterrows():
        w = pool[(pool.MMSI == c.MMSI) & (pool.tau_release_h > c.tau_end_h) & (pool.tau_release_h <= c.tau_start_h)]
        e = w.loc[w.index.repeat(T["truth_particles_per_release"])].reset_index(drop=True)
        r = T["release_radius_m"]
        e["lon"] = e.lon0 + rng.normal(0, r, len(e)) / (111320 * np.cos(np.radians(e.lat0)))
        e["lat"] = e.lat0 + rng.normal(0, r, len(e)) / 110540
        e["windage"], e["case_id"] = T["truth"]["windage"], c.case_id
        els.append(e)
    els = pd.concat(els, ignore_index=True)
    from opendrift.readers.reader_global_landmask import Reader as Landmask
    els = els[~np.asarray(Landmask()._on_land(els.lon.values, els.lat.values), bool)].reset_index(drop=True)
    dur = (t0 - els.time.min()).total_seconds() / 3600
    sim = dict(sim_id=0, horizontal_diffusivity=T["truth"]["horizontal_diffusivity"],
               current_uncertainty=T["truth"]["current_uncertainty"])
    PT = {**P, "wind_uncertainty": T["truth"]["wind_uncertainty"], "output_s": P["dt_s"]}
    lon, lat, t = drift.simulate(els, dur, False, rdr, sim, PT, seed=T["truth"]["seed"])
    assert abs((t[-1] - t0).total_seconds()) < 1
    els["lon_t0"], els["lat_t0"] = lon[:, -1], lat[:, -1]
    rows, geoms = [], []
    g_m = T["grid_m"]
    for _, c in cases.iterrows():
        e = els[(els.case_id == c.case_id) & np.isfinite(els.lon_t0)]
        n_rel = int((els.case_id == c.case_id).sum())
        if len(e) < 0.8 * max(n_rel, 1):
            rows.append({"case_id": c.case_id, "status": f"discarded: {n_rel - len(e)} of {n_rel} particles stranded/left domain"})
            continue
        fwd, back = laea(float(e.lon_t0.mean()), float(e.lat_t0.mean()))
        x, y = fwd.transform(e.lon_t0.values, e.lat_t0.values)
        x0, y0 = x.min() - 5 * g_m, y.max() + 5 * g_m
        W, H = int((x.max() - x0) / g_m) + 6, int((y0 - y.min()) / g_m) + 6
        cnt = np.zeros((H, W), int)
        np.add.at(cnt, (((y0 - y) / g_m).astype(int), ((x - x0) / g_m).astype(int)), 1)
        mask = ndimage.binary_closing(cnt >= T["min_particles_per_cell"], iterations=1)
        tr = from_origin(x0, y0, g_m, g_m)
        polys = [shape(gm) for gm, v in shapes(mask.astype(np.uint8), mask=mask, transform=tr) if v == 1]
        if not polys:
            rows.append({"case_id": c.case_id, "status": "discarded: empty mask"})
            continue
        gm = unary_union(polys).buffer(T["degrade_buffer_m"])
        ang = rng.uniform(0, 2 * np.pi)
        gm = affinity.translate(gm, T["degrade_shift_m"] * np.cos(ang), T["degrade_shift_m"] * np.sin(ang))
        from shapely.ops import transform as sh_transform
        geom = sh_transform(back.transform, gm)
        area = geodesic_area_m2(geom)
        if area < T["min_area_m2"]:
            rows.append({"case_id": c.case_id, "status": f"discarded: area {area:.0f} m2 < {T['min_area_m2']}"})
            continue
        cen = geom.centroid
        f2, b2 = laea(cen.x, cen.y)
        p1, p2, length, bearing, short = axis_endpoints(to_metric(geom, f2))
        e1, e2 = b2.transform(p1.x, p1.y), b2.transform(p2.x, p2.y)
        rows.append({"case_id": c.case_id, "status": "ok", "cluster_id": int(c.case_id), "area_m2": area,
                     "centroid_lon": cen.x, "centroid_lat": cen.y, "length_m": length, "axis_bearing_deg": bearing,
                     "width_proxy_m": area / max(length, 1), "end1_lon": e1[0], "end1_lat": e1[1],
                     "end2_lon": e2[0], "end2_lat": e2[1]})
        geoms.append((c.case_id, geom))
    st = pd.DataFrame(rows)
    ok = st[st.status == "ok"].copy()
    gdf = gpd.GeoDataFrame(ok, geometry=[dict(geoms)[i] for i in ok.case_id], crs="EPSG:4326")
    return gdf, st


def _empty_evidence():
    geo = pd.DataFrame(columns=["cluster_id", "MMSI", "geo_fresh_discharge", "t0_lon", "t0_lat"])
    beh = pd.DataFrame(columns=["MMSI", "ais_gap_intervals", "mmsi_standard", "ais_speed_jumps_gt_50kn"])
    mo = pd.DataFrame(columns=["MMSI", "motion_state", "source_type"])
    for d in (geo, beh, mo):
        d["MMSI"] = d["MMSI"].astype(str)
    geo["cluster_id"] = geo["cluster_id"].astype(int)
    return geo, beh, mo


def score_cases(cases, rob, fwd_best, cands, P):
    geo, beh, mo = _empty_evidence()
    sus = fusion.fuse(rob, fwd_best, geo, beh, mo, cands, P)
    out = []
    for _, c in cases.iterrows():
        s = sus[sus.cluster_id == c.case_id]
        r = rob[(rob.cluster_id == c.case_id)]
        t = s[s.MMSI == c.MMSI]
        passers = r[r.passes_funnel]
        rank_b = passers.bwd_weighted.rank(ascending=False, method="min")
        fb = fwd_best[fwd_best.cluster_id == c.case_id]
        rank_f = fb.fwd_F.rank(ascending=False, method="min")
        rec = {"case_id": c.case_id, "n_suspects": len(s), "truth_passes_funnel": bool((passers.MMSI == c.MMSI).any()),
               "rank_fused": int(t.suspect_rank.iloc[0]) if len(t) else np.nan,
               "tier_truth": t.tier.iloc[0] if len(t) else "not in funnel",
               "rank_backward_only": float(rank_b[passers.MMSI == c.MMSI].iloc[0]) if (passers.MMSI == c.MMSI).any() else np.nan,
               "rank_forward_only": float(rank_f[fb.MMSI == c.MMSI].iloc[0]) if (fb.MMSI == c.MMSI).any() and fb[fb.MMSI == c.MMSI].fwd_F.iloc[0] > 0 else np.nan,
               "top_tier": s.tier.iloc[0] if len(s) else "none", "top_vessel": s.MMSI.iloc[0] if len(s) else None}
        if len(t) and t.fwd_F.iloc[0] > 0:
            a, b = t.fwd_window_start_h.iloc[0], t.fwd_window_end_h.iloc[0]
            ta, tb = c.tau_end_h, c.tau_start_h
            inter = max(0, min(b, tb) - max(a, ta))
            rec.update(est_window=f"{a:.0f}-{b:.0f}h", window_overlap_h=inter,
                       window_mid_error_h=abs((a + b) / 2 - (ta + tb) / 2))
        out.append(rec)
    return pd.DataFrame(out), sus


def summarise(res, cases, hid):
    r = res.merge(cases, on="case_id")
    def acc(x, k):
        return float((x <= k).mean()) if len(x) else np.nan
    rows = []
    for name, g in [("all", r), *[(s, r[r.stratum == s]) for s in [f"{a}-{b}h" for a, b in STRATA]],
                    ("moving", r[r.moving]), ("stationary", r[~r.moving])]:
        rows.append({"subset": name, "n": len(g), "funnel_recall": float(g.truth_passes_funnel.mean()) if len(g) else np.nan,
                     **{f"top{k}_{m}": acc(g[f"rank_{m}"].fillna(999), k) for m in ("fused", "backward_only", "forward_only") for k in (1, 3, 5)},
                     "truth_tier_consistent_or_better": float(g.tier_truth.isin(["strongly consistent", "consistent"]).mean()) if len(g) else np.nan,
                     "median_window_mid_error_h": float(g.window_mid_error_h.median()) if "window_mid_error_h" in g and g.window_mid_error_h.notna().any() else np.nan})
    tab = pd.DataFrame(rows)
    h = hid.merge(cases, on="case_id")
    false = {"n": len(h), "innocent_top_consistent_or_better": float(h.top_tier.isin(["strongly consistent", "consistent"]).mean()),
             "innocent_top_strong": float((h.top_tier == "strongly consistent").mean()),
             "present_top_consistent_or_better": float(res.top_tier.isin(["strongly consistent", "consistent"]).mean()),
             "present_top_is_truth": float((res.rank_fused == 1).mean())}
    return tab, false


def run(run_, inc, P, n):
    T, PP, B, Fp = P["twin_v1"], P["physics_v3"], P["backward_v3"], P["forward_v3"]
    cfg21 = load_json(ROOT / P["frozen"]["v2_1_config"])
    rng = np.random.default_rng(T["seed"])
    with Stage(run_, "twin", {"twin": T, "physics": PP}, [inc.ais_attribution(), inc.input("ocean_nc"), inc.input("wind_nc")]) as s:
        ais = load_ais(inc.ais_attribution())
        pool = forward.release_points(ais, sorted(ais.MMSI.unique()), inc.t0, Fp, cfg21)
        F = Forcing(inc.input("ocean_nc"), inc.input("wind_nc"))
        cases, tried = sample_cases(pool, n, {**T, "release_step_min": Fp["release_step_min"]}, rng, F.bounds())
        rdr = drift.readers(inc.input("ocean_nc"), inc.input("wind_nc"))
        cands, status = truth_slicks(cases, pool, inc.t0, rdr, PP, {**T, "release_radius_m": Fp["release_radius_m"]}, rng)
        cases = cases.merge(status[["case_id", "status"]], on="case_id")
        cases.to_csv(s.output(s.dir / "cases.csv"), index=False)
        cands.to_file(s.output(s.dir / "synthetic_slicks.geojson"), driver="GeoJSON")
        ok = cases[cases.status == "ok"].copy()
        print(f"  twin: {len(ok)} usable cases of {len(cases)} sampled ({tried} draws)", flush=True)
        # blind attribution: exactly the pipeline's backward -> forward -> fusion
        parts, _ = backward.run_ensemble(cands, inc.t0, rdr, PP)
        times = pd.DatetimeIndex(sorted(parts.time.unique()))
        pos = backward.hourly_positions(ais, times, cfg21)
        ts = backward.score(parts, pos, inc.t0, B)
        _, rob, _ = backward.aggregate(ts, pos, B, P["frozen"]["age_bins_h"], P["frozen"]["age_labels"])
        short = (rob[rob.passes_funnel].sort_values("bwd_weighted", ascending=False)
                 .groupby("cluster_id").head(T["forward_max_vessels"]))
        base = forward.release_points(ais, sorted(short.MMSI.unique()), inc.t0, Fp, cfg21)
        els = forward.jitter_elements(base, {**Fp, "windage": PP["windage"]}, np.random.default_rng(PP["seed"] + 7))
        fpos = forward.run(els, inc.t0, rdr, PP)
        simulated, passers, best = set(fpos.MMSI.unique()), rob[rob.passes_funnel], []
        for _, c in cands.iterrows():
            v = sorted(set(passers[passers.cluster_id == c.cluster_id].MMSI) & simulated)
            b, _ = forward.evaluate_cluster(c, fpos, v, Fp, np.random.default_rng(int(c.cluster_id)))
            best.append(b)
        fwd_best = pd.concat(best, ignore_index=True)
        res, sus = score_cases(ok, rob, fwd_best, cands, P)
        # hidden culprit: remove the true vessel from each case and re-rank
        drop = set(zip(ok.case_id, ok.MMSI))
        keep = ~pd.Series(list(zip(ts.cluster_id, ts.MMSI))).isin(drop).values
        _, rob_h, _ = backward.aggregate(ts[keep], pos, B, P["frozen"]["age_bins_h"], P["frozen"]["age_labels"])
        fb_h = fwd_best[~pd.Series(list(zip(fwd_best.cluster_id, fwd_best.MMSI))).isin(drop).values]
        hid, _ = score_cases(ok, rob_h, fb_h, cands, P)
        res.to_csv(s.output(s.dir / "results_present.csv"), index=False)
        hid.to_csv(s.output(s.dir / "results_hidden.csv"), index=False)
        sus.to_csv(s.output(s.dir / "suspects.csv"), index=False)
        tab, false = summarise(res, ok, hid)
        tab.to_csv(s.output(s.dir / "skill_table.csv"), index=False)
        summary = {"n_cases": int(len(ok)), "n_sampled": int(len(cases)), "false_accusation": false,
                   "skill": tab.to_dict("records"), "limitation": "truth and model share forcing fields: upper bound"}
        (s.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        s.output(s.dir / "summary.json")
        s.notes.update(n_cases=int(len(ok)), false_accusation=false)
        print(tab.round(2).to_string(index=False), flush=True)
        print(false, flush=True)
