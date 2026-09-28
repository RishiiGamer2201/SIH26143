"""Stage orchestration for one incident run. Every stage reads the previous stages' files (restartable) and writes a
manifest. Order: parity -> seed -> context -> backward -> forward -> evidence -> fuse -> report -> posthoc.
The reference (Cerulean) is read ONLY in parity (as the frozen seed) and in posthoc (after everything is written)."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from . import age, backward, drift, evidence, forward, fusion, seed
from .ais import load_ais
from .config import ROOT, load_json
from .context import Forcing
from .frozen import source_type as st1
from .manifest import Stage

STAGES = ["parity", "seed", "context", "sar", "backward", "forward", "evidence", "fuse", "forecast", "report", "posthoc"]


def physics_params(P):
    return {**P["physics_v3"]}


def st_seed(run, inc, P):
    with Stage(run, "seed", P["seed"], [inc.input("detected_components"), inc.ais_attribution()]) as s:
        comps = gpd.read_file(inc.input("detected_components"))
        F = Forcing(inc.input("ocean_nc"), inc.input("wind_nc"))
        ais = load_ais(inc.ais_attribution())
        ais_bbox = (ais.LON.min(), ais.LAT.min(), ais.LON.max(), ais.LAT.max())
        cand, comps = seed.build_candidates(comps, F, ais_bbox, inc.t0, P["seed"])
        cand = seed.apply_override(cand, inc.spec)
        cand.to_file(s.output(s.dir / "candidates.geojson"), driver="GeoJSON")
        cand.drop(columns="geometry").to_csv(s.output(s.dir / "candidates.csv"), index=False)
        comps.to_file(s.output(s.dir / "components_clustered.geojson"), driver="GeoJSON")
        s.notes.update(n_components=len(comps), n_events=len(cand), n_eligible=int(cand.eligible.sum()),
                       n_selected=int(cand.selected.sum()), ais_bbox=ais_bbox,
                       ais_coverage_caveat="AIS extract region was cut around the reference slick when the dataset was built; "
                                           "scene-wide autonomy needs a scene-wide AIS extract")


def selected(run):
    c = gpd.read_file(run.root / "seed/candidates.geojson")
    return c[c.selected].reset_index(drop=True)


def st_context(run, inc, P):
    with Stage(run, "context", {}, [inc.input("ocean_nc"), inc.input("wind_nc")]) as s:
        F = Forcing(inc.input("ocean_nc"), inc.input("wind_nc"))
        rows, hist = [], []
        for _, c in selected(run).iterrows():
            rows.append({"cluster_id": c.cluster_id, **F.wind(c.centroid_lon, c.centroid_lat, inc.t0),
                         **F.current(c.centroid_lon, c.centroid_lat, inc.t0)})
            hist.append(F.history(c.centroid_lon, c.centroid_lat, inc.t0, 48).assign(cluster_id=c.cluster_id))
        pd.DataFrame(rows).to_csv(s.output(s.dir / "context_t0.csv"), index=False)
        pd.concat(hist).to_parquet(s.output(s.dir / "context_history.parquet"), index=False)
        tr = F.time_range()
        s.notes.update(forcing_time_range=[str(tr[0]), str(tr[1])], forcing_bounds=F.bounds())


def st_sar(run, inc, P):
    """Darkness per slick event + scene-wide ship detection -> AIS / platform matching -> dark-vessel candidates."""
    from . import sar
    from .infrastructure import load_platforms
    S = P["sar_v1"]
    safe = inc.input("safe") if "safe" in inc.spec["inputs"] else None
    with Stage(run, "sar", S, [inc.ais_attribution()]) as s:
        try:
            scene = sar.Scene(safe, "vv")
        except (IndexError, RuntimeError) as e:
            s.notes["skipped"] = f"no measurement TIFF in SAFE ({e!r}); fetch with: python -m oilwatch.fetch s1 ..."
            print("  sar skipped: measurement TIFF missing", flush=True)
            return
        s.inputs.append(Path(scene.tif))
        s.notes.update(gcp_source=scene.gcp_source, n_gcps=scene.n_gcps, radar_size=[scene.H, scene.W])
        cands = selected(run)
        dk = sar.darkness(scene, cands, P=S["darkness"])
        dk.to_csv(s.output(s.dir / "darkness.csv"), index=False)
        raw = s.dir / "ship_detections_raw.csv"  # CFAR scan is the slow part; matching is re-run on cached detections
        if raw.exists() and s.notes.setdefault("cfar_cache", "reused") == "reused":
            det = pd.read_csv(raw)
        else:
            det = sar.ships(scene, S["ships"])
            det.to_csv(raw, index=False)
        s.output(raw)
        ais = load_ais(inc.ais_attribution())
        plat = load_platforms(inc)
        cfg21 = load_json(ROOT / P["frozen"]["v2_1_config"])
        det, st = sar.match_detections(det, ais, inc.t0, plat, S["match"], cfg21)
        det.to_csv(s.output(s.dir / "ship_detections.csv"), index=False)
        st.to_csv(s.output(s.dir / "ais_at_t0.csv"), index=False)
        dark = det[det.label.isin(["dark_vessel_candidate", "ais_silent_at_t0"])].copy()
        rows = []
        for _, c in cands.iterrows():
            from shapely.geometry import Point
            from .geo import laea, to_metric
            fwd, _ = laea(c.centroid_lon, c.centroid_lat)
            gm = to_metric(c.geometry, fwd)
            for _, d in dark.iterrows():
                km = gm.distance(to_metric(Point(d.lon, d.lat), fwd)) / 1000
                if km <= 25:
                    rows.append({"cluster_id": c.cluster_id, "label": d.label, "silent_name": d.silent_name,
                                 "silent_last_report_min": d.silent_last_report_min, "row": d.row, "col": d.col, "lon": d.lon, "lat": d.lat,
                                 "peak_db": d.peak_db, "size_px": d.size_px, "dist_to_slick_km": km,
                                 "near_slick": km <= S["dark_near_slick_km"]})
        pd.DataFrame(rows).to_csv(s.output(s.dir / "dark_vessels_near_slicks.csv"), index=False)
        geo_err = sar.geolocation_offset(det)
        (s.dir / "geolocation.json").write_text(json.dumps(geo_err, indent=1))
        s.output(s.dir / "geolocation.json")
        s.notes.update(n_detections=len(det), labels=det.label.value_counts().to_dict(), geolocation=geo_err,
                       platforms_loaded=0 if plat is None else len(plat))


def st_backward(run, inc, P):
    PP, B = physics_params(P), P["backward_v3"]
    cfg21 = load_json(ROOT / P["frozen"]["v2_1_config"])
    with Stage(run, "backward", {"physics": PP, "backward": B}, [inc.input("ocean_nc"), inc.input("wind_nc"), inc.ais_attribution()]) as s:
        cands = selected(run)
        rdr = drift.readers(inc.input("ocean_nc"), inc.input("wind_nc"))
        parts, els = backward.run_ensemble(cands, inc.t0, rdr, PP)
        parts.to_parquet(s.output(s.dir / "particles.parquet"), index=False)
        backward.cloud_summary(parts, inc.t0).to_parquet(s.output(s.dir / "cloud_summary.parquet"), index=False)
        ais = load_ais(inc.ais_attribution())
        times = pd.DatetimeIndex(sorted(parts.time.unique()))
        pos = backward.hourly_positions(ais, times, cfg21)
        ts = backward.score(parts, pos, inc.t0, B)
        ts.to_parquet(s.output(s.dir / "time_scores.parquet"), index=False)
        mem, rob, bins = backward.aggregate(ts, pos, B, P["frozen"]["age_bins_h"], P["frozen"]["age_labels"])
        mem.to_parquet(s.output(s.dir / "member_scores.parquet"), index=False)
        rob.to_csv(s.output(s.dir / "backward_robust.csv"), index=False)
        bins.to_csv(s.output(s.dir / "backward_age_bins.csv"), index=False)
        fn = backward.funnel(pos, rob, B)
        fn.to_csv(s.output(s.dir / "funnel.csv"), index=False)
        s.notes.update(n_elements=len(els), n_members=int(parts.member.nunique()), n_vessels_positioned=len(pos),
                       funnel=fn.to_dict("records"))


def st_forward(run, inc, P):
    PP, Fp = physics_params(P), P["forward_v3"]
    cfg21 = load_json(ROOT / P["frozen"]["v2_1_config"])
    with Stage(run, "forward", {"physics": PP, "forward": Fp}, [inc.input("ocean_nc"), inc.input("wind_nc"), inc.ais_attribution()]) as s:
        cands = selected(run)
        rob = pd.read_csv(run.root / "backward/backward_robust.csv", dtype={"MMSI": str})
        short = (rob[rob.passes_funnel].sort_values("bwd_weighted", ascending=False)
                 .groupby("cluster_id").head(Fp["max_vessels_per_cluster"]))
        short[["cluster_id", "MMSI"]].to_csv(s.output(s.dir / "shortlist.csv"), index=False)
        ais = load_ais(inc.ais_attribution())
        rng = np.random.default_rng(PP["seed"] + 7)
        els = forward.release_elements(ais, sorted(short.MMSI.unique()), inc.t0, {**Fp, "windage": PP["windage"]}, cfg21, rng)
        cache = s.dir / "virtual_oil_at_t0.parquet"
        cached = pd.read_parquet(cache) if cache.exists() else None
        if cached is not None and len(cached) == len(els) * len(drift.v3_members(PP)) and                 set(cached.MMSI.unique()) == set(els.MMSI.unique()):
            fpos, s.notes["simulation"] = cached, "reused cached simulation (same shortlist and element count)"
        else:
            rdr = drift.readers(inc.input("ocean_nc"), inc.input("wind_nc"))
            fpos = forward.run(els, inc.t0, rdr, PP)
            fpos.to_parquet(cache, index=False)
        s.output(cache)
        best, allw = [], []
        # evaluate every funnel vessel of each slick that has simulated virtual oil (the simulation shortlist is the
        # union of per-slick top-N, so a vessel cut from one slick's top-N is usually simulated for another slick)
        simulated = set(fpos.MMSI.unique())
        passers = rob[rob.passes_funnel]
        for _, c in cands.iterrows():
            v = sorted(set(passers[passers.cluster_id == c.cluster_id].MMSI) & simulated)
            b, a = forward.evaluate_cluster(c, fpos, v, Fp, np.random.default_rng(int(c.cluster_id)))
            best.append(b), allw.append(a)
            print(f"  forward eval cluster {c.cluster_id}: {len(v)} vessels", flush=True)
        pd.concat(best).to_csv(s.output(s.dir / "forward_best.csv"), index=False)
        pd.concat(allw).to_parquet(s.output(s.dir / "forward_windows.parquet"), index=False)
        s.notes.update(n_vessels=int(short.MMSI.nunique()), n_elements=len(els))


def motion_for(inc, mmsis, P):
    """Frozen E102 motion_state for any MMSI set (same metrics / thresholds / rules)."""
    cfg = load_json(ROOT / P["frozen"]["source_type_config"])
    st1._selftest(cfg)
    a = pd.read_parquet(inc.ais_attribution())
    a = a[(a.BaseDateTime >= inc.t0 - pd.Timedelta(hours=cfg["window_h_before_t0"])) & (a.BaseDateTime <= inc.t0)]
    a = a.sort_values("BaseDateTime").drop_duplicates(["MMSI", "BaseDateTime"])
    rows = []
    for m in mmsis:
        g = a[a.MMSI.astype(str).str.replace(r"\.0$", "", regex=True) == str(m)]
        if g.empty:
            rows.append({"MMSI": str(m), "motion_state": "undetermined", "source_type": "unknown"})
            continue
        beh, why = st1.classify(st1.track_metrics(g, cfg["stationary_radius_km"], cfg["slow_sog_kn"]), cfg)
        rows.append({"MMSI": str(m), "motion_state": beh, "motion_reason": why, "source_type": "unknown"})
    return pd.DataFrame(rows)


def st_evidence(run, inc, P):
    cfg21 = load_json(ROOT / P["frozen"]["v2_1_config"])
    with Stage(run, "evidence", {"geometry": P["geometry_v3"]}, [inc.ais_attribution()]) as s:
        cands = selected(run)
        short = pd.read_csv(run.root / "forward/shortlist.csv", dtype={"MMSI": str})
        ais = load_ais(inc.ais_attribution())
        geo = pd.concat([evidence.geometry(c, ais, short[short.cluster_id == c.cluster_id].MMSI.tolist(), inc.t0,
                                           P["geometry_v3"], cfg21) for _, c in cands.iterrows()])
        geo.to_csv(s.output(s.dir / "geometry.csv"), index=False)
        mm = sorted(short.MMSI.unique())
        evidence.behaviour(ais, mm, inc.t0).to_csv(s.output(s.dir / "behaviour.csv"), index=False)
        from .infrastructure import load_platforms, resolve_source_type
        plat = load_platforms(inc)
        mo = resolve_source_type(motion_for(inc, mm, P), ais, plat) if plat is not None else motion_for(inc, mm, P)
        mo.to_csv(s.output(s.dir / "motion.csv"), index=False)
        s.notes["platforms_loaded"] = 0 if plat is None else len(plat)
        s.notes["source_type_counts"] = mo.source_type.value_counts().to_dict()


def st_fuse(run, inc, P):
    with Stage(run, "fuse", {"fusion": P["fusion_v3"], "age": P["age_v3"]}) as s:
        rd = lambda p: pd.read_csv(run.root / p, dtype={"MMSI": str})
        sus = fusion.fuse(rd("backward/backward_robust.csv"), rd("forward/forward_best.csv"), rd("evidence/geometry.csv"),
                          rd("evidence/behaviour.csv"), rd("evidence/motion.csv"), selected(run), P,
                          pd.read_csv(run.root / "sar/ship_detections.csv") if (run.root / "sar/ship_detections.csv").exists() else None)
        sus.to_csv(s.output(s.dir / "suspects.csv"), index=False)
        ages, summ = [], []
        dkf = run.root / "sar/darkness.csv"
        dks = pd.read_csv(dkf).set_index("cluster_id") if dkf.exists() else None
        for _, c in selected(run).iterrows():
            dk = dks.loc[c.cluster_id].to_dict() if dks is not None and c.cluster_id in dks.index else None
            a, sm = age.cluster_age(c, sus[sus.cluster_id == c.cluster_id], P, P["physics_v3"]["horizontal_diffusivity_m2s"], dk)
            ages.append(a), summ.append({"cluster_id": c.cluster_id, **sm})
        pd.concat(ages).to_csv(s.output(s.dir / "age_cues.csv"), index=False)
        pd.DataFrame(summ).to_csv(s.output(s.dir / "age_summary.csv"), index=False)
        s.notes["tiers"] = sus.tier.value_counts().to_dict()


def st_forecast(run, inc, P):
    from . import forecast
    PP, F = physics_params(P), P["forecast_v1"]
    with Stage(run, "forecast", {"physics": PP, "forecast": F}, [inc.input("ocean_nc"), inc.input("wind_nc")]) as s:
        cands = selected(run)
        rdr = drift.readers(inc.input("ocean_nc"), inc.input("wind_nc"))
        fp = forecast.run(cands, inc.t0, rdr, PP, F)
        fp.to_parquet(s.output(s.dir / "forecast_particles.parquet"), index=False)
        oil = F["oil_central"]
        grids = [forecast.probability_grid(fp, c, h, oil, F["grid_cell_deg"]) for c in cands.cluster_id for h in F["lead_times_h"]]
        pd.concat(grids).to_csv(s.output(s.dir / "probability_grids.csv"), index=False)
        forecast.beaching(fp, oil).to_csv(s.output(s.dir / "beaching.csv"), index=False)
        forecast.mass_budget(fp[fp.sim_id == fp[fp.oil_type != oil].sim_id.iloc[0]] if (fp.oil_type != oil).any() else fp)             .to_csv(s.output(s.dir / "mass_budget.csv"), index=False)
        forecast.centroid_track(fp, oil).to_csv(s.output(s.dir / "centroid_track.csv"), index=False)
        s.notes["sst_assumption_c"] = F["sea_water_temperature_c"]


def st_report(run, inc, P):
    from .report import write_report
    with Stage(run, "report", {}) as s:
        for p in write_report(run, inc, P, s.dir):
            s.output(p)


def st_posthoc(run, inc, P):
    from .report import posthoc
    with Stage(run, "posthoc", {"note": "reference read here only, after all outputs were written"},
               [ROOT / inc.spec["reference"]["slick_geojson"]]) as s:
        for p in posthoc(run, inc, P, s.dir):
            s.output(p)


def run_all(run, inc, P, skip_parity=False, stages=None):
    todo = stages or STAGES
    if "parity" in todo and not skip_parity:
        from .parity import run_scoring_parity
        run_scoring_parity(run, inc, P)
    fn = {"seed": st_seed, "context": st_context, "sar": st_sar, "backward": st_backward, "forward": st_forward,
          "evidence": st_evidence, "fuse": st_fuse, "forecast": st_forecast, "report": st_report, "posthoc": st_posthoc}
    for name in STAGES[1:]:
        if name in todo:
            fn[name](run, inc, P)
    (run.root / "run.json").write_text(json.dumps({"incident": inc.id, "run_id": run.run_id, "stages": todo}, indent=1))
