"""Parity stage: the refactored code must reproduce every frozen Incident 001 output before anything new is trusted.

scoring parity (default): V2 time/summary, release-age, ensemble scenario/robust, V2.1 time/summary/age, E102 motion,
recomputed from the frozen hindcast/ensemble files and compared exactly (values after identical CSV/parquet round trip).
drift parity (--full): re-runs the deterministic hindcast and the 27-scenario ensemble (~5 min) and compares particles.
"""
import json
import shutil

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import shape

from . import drift, scoring
from .ais import load_ais
from .config import ROOT, load_json
from .frozen import source_type as st1, v2_1 as v21
from .manifest import Stage


def _read(p):
    return pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p, float_precision="round_trip")


ULP_RTOL = 1e-12  # float transcendental results differ in the last bits across numpy builds / CPUs (SIMD paths)


def compare(name, ours_path, frozen_path):
    """exact: bit-identical. ulp_equal: same shape/columns, every non-float cell (ranks, names, bins, flags) identical,
    every float within ULP_RTOL relative. On the machine that produced the frozen files the result is bit-exact;
    across platforms E000-E102 parity is ulp_equal (e.g. the frozen E102 code itself, imported verbatim, differs at 2e-16)."""
    a, b = _read(ours_path), _read(frozen_path)
    res = {"table": name, "rows_ours": len(a), "rows_frozen": len(b), "same_columns": list(a.columns) == list(b.columns)}
    try:
        pd.testing.assert_frame_equal(a, b, check_exact=True, check_dtype=False)
        res.update(exact=True, ulp_equal=True, max_rel_diff=0.0)
    except AssertionError:
        ok, rel = res["same_columns"] and len(a) == len(b), 0.0
        for c in (b.columns if ok else []):
            x, y = a[c], b[c]
            if pd.api.types.is_float_dtype(y):
                xv, yv = x.to_numpy(float), y.to_numpy(float)
                fin = np.isfinite(yv)
                ok &= np.array_equal(np.isnan(xv), np.isnan(yv))
                r = np.abs(xv[fin] - yv[fin]) / np.maximum(np.abs(yv[fin]), 1e-300)
                rel = max(rel, float(r.max()) if len(r) else 0.0)
            else:
                ok &= x.astype(str).equals(y.astype(str))
        res.update(exact=False, ulp_equal=bool(ok and rel <= ULP_RTOL), max_rel_diff=rel)
    tag = "EXACT" if res["exact"] else ("ULP  " if res["ulp_equal"] else "DIFF ")
    print(f"  {tag} {name}: {res['rows_ours']} rows, max rel diff {res['max_rel_diff']:.1e}", flush=True)
    return res


def run_scoring_parity(run, inc, P):
    fz = P["frozen"]
    cfg21 = load_json(ROOT / fz["v2_1_config"])
    cfg_st = load_json(ROOT / fz["source_type_config"])
    t0, ref = inc.t0, inc.reference_mmsi
    tol = pd.Timedelta(minutes=fz["v2"]["ais_tolerance_min"])
    with Stage(run, "parity_scoring", {"frozen": fz}, [inc.frozen("hindcast_nc"), inc.frozen("ensemble_particles"),
                                                        inc.input("ais_parquet")]) as s:
        out, results = s.dir, []
        ds = xr.open_dataset(inc.frozen("hindcast_nc"))
        times = pd.to_datetime(ds.time.values, utc=True).astype("datetime64[ns, UTC]")
        order = np.argsort(times.values)
        times, plon, plat = times[order], ds.lon.values[:, order], ds.lat.values[:, order]
        ds.close()
        ais = load_ais(inc.input("ais_parquet"))

        # V2 + release age
        ts, summ = scoring.v2_time_scores(plon, plat, times, ais, ref, tol, fz["v2"]["sigma_km"])
        ts.to_parquet(s.output(out / "ais_density_time_scores_v2.parquet"), index=False)
        summ.to_csv(s.output(out / "ais_density_scores_v2.csv"), index=False)
        age = scoring.release_age_table(pd.read_parquet(out / "ais_density_time_scores_v2.parquet"), t0, fz["age_bins_h"], fz["age_labels"])
        age.to_csv(s.output(out / "release_age_scores.csv"), index=False)
        results += [compare("v2_time_scores", out / "ais_density_time_scores_v2.parquet", inc.frozen("v2_time_scores")),
                    compare("v2_summary", out / "ais_density_scores_v2.csv", inc.frozen("v2_summary")),
                    compare("v2_release_age", out / "release_age_scores.csv", inc.frozen("v2_release_age"))]

        # ensemble robustness
        ens = pd.read_parquet(inc.frozen("ensemble_particles"))
        ens["time"] = pd.to_datetime(ens["time"], utc=True).astype("datetime64[ns, UTC]")
        sc = scoring.ensemble_scenario_scores(ens, ais, t0, ref, tol, fz["v2"]["sigma_km"], fz["age_bins_h"], fz["age_labels"])
        sc.to_parquet(s.output(out / "ais_ensemble_scenario_scores.parquet"), index=False)
        scoring.robust_table(sc, fz["age_labels"]).to_csv(s.output(out / "ais_ensemble_robust_scores.csv"), index=False)
        results += [compare("ensemble_scenario_scores", out / "ais_ensemble_scenario_scores.parquet", inc.frozen("ensemble_scenario_scores")),
                    compare("ensemble_robust_scores", out / "ais_ensemble_robust_scores.csv", inc.frozen("ensemble_robust_scores"))]

        # V2.1 (E101) + motion (E102)
        ts21 = scoring.v2_1_time_scores(plon, plat, times, ais, cfg21, pd.Timestamp(cfg21["incident_t0_utc"]))
        ts21.to_parquet(s.output(out / "ais_density_time_scores_v2_1.parquet"), index=False)
        v21.age_bin_table(ts21, cfg21["age_labels"]).to_csv(s.output(out / "release_age_scores_v2_1.csv"), index=False)
        results += [compare("v2_1_time_scores", out / "ais_density_time_scores_v2_1.parquet", inc.frozen("v2_1_time_scores")),
                    compare("v2_1_release_age", out / "release_age_scores_v2_1.csv", inc.frozen("v2_1_release_age"))]
        motion(inc, cfg_st, out / "track_motion.csv")
        s.output(out / "track_motion.csv")
        results.append(compare("track_motion", out / "track_motion.csv", inc.frozen("track_motion")))

        s.notes["results"] = results
        s.notes["all_exact"] = all(r["exact"] for r in results)
        s.notes["all_ulp_equal"] = all(r["ulp_equal"] for r in results)
        (out / "parity_report.json").write_text(json.dumps(results, indent=1, default=str))
        assert s.notes["all_ulp_equal"], "scoring parity FAILED - see parity_report.json"
    return results


def motion(inc, cfg, out_csv):
    """E102 motion_state for every scored MMSI, using the frozen track_metrics/classify (same selection as source_type_v1)."""
    st1._selftest(cfg)
    t0 = pd.Timestamp(cfg["incident_t0_utc"])
    ais = pd.read_parquet(ROOT / cfg["ais_parquet"])
    ais = ais[(ais.BaseDateTime >= t0 - pd.Timedelta(hours=cfg["window_h_before_t0"])) & (ais.BaseDateTime <= t0)]
    ais = ais.sort_values("BaseDateTime").drop_duplicates(["MMSI", "BaseDateTime"])
    tables = [pd.read_csv(ROOT / p, float_precision="round_trip") for p in st1.INPUTS.values()]
    mmsis = sorted(set().union(*(set(t.MMSI.astype("int64")) for t in tables)))
    rows = []
    for mmsi in mmsis:
        g = ais[ais.MMSI == mmsi]
        if g.empty:
            rows.append(dict(MMSI=mmsi, n_reports=0, track_span_h=0.0, motion_state="undetermined",
                             classification_reason="no AIS reports in window"))
            continue
        m = st1.track_metrics(g, cfg["stationary_radius_km"], cfg["slow_sog_kn"])
        beh, why = st1.classify(m, cfg)
        status, vtype, hints = g.Status.mode(), g.VesselType.mode(), []
        if len(status) and str(int(status.iloc[0])) in cfg["metadata_hint_status_codes"]:
            hints.append(cfg["metadata_hint_status_codes"][str(int(status.iloc[0]))])
        if len(vtype) and str(int(vtype.iloc[0])) in cfg["metadata_hint_vessel_types"]:
            hints.append(cfg["metadata_hint_vessel_types"][str(int(vtype.iloc[0]))])
        rows.append(dict(MMSI=mmsi, **m, motion_state=beh, classification_reason=why,
                         ais_status_mode=status.iloc[0] if len(status) else np.nan,
                         ais_vessel_type_mode=vtype.iloc[0] if len(vtype) else np.nan, metadata_hint=";".join(hints)))
    beh = pd.DataFrame(rows)
    beh["source_type"], beh["source_type_basis"] = "unknown", st1.UNKNOWN_BASIS
    beh["stationary_flag"] = beh.motion_state.eq("stationary")
    beh.to_csv(out_csv, index=False)
    return beh


def run_drift_parity(run, inc, P):
    fz = P["frozen"]
    geom = shape(json.loads((ROOT / inc.spec["reference"]["slick_geojson"]).read_text())["features"][0]["geometry"])
    geom = geom if geom.is_valid else geom.buffer(0)
    with Stage(run, "parity_drift", {"frozen": fz}, [inc.input("ocean_nc"), inc.input("wind_nc")]) as s:
        rdr = drift.readers(inc.input("ocean_nc"), inc.input("wind_nc"))
        det = s.output(s.dir / "hindcast_48h.nc")
        drift.frozen_deterministic(geom, inc.t0.tz_convert(None).to_pydatetime(), rdr, det, fz["deterministic"])
        a, b = xr.open_dataset(det), xr.open_dataset(inc.frozen("hindcast_nc"))
        res = {"deterministic_bit_exact": bool(all(np.array_equal(a[v].values, b[v].values, equal_nan=True) for v in ("lon", "lat")))}
        a.close(); b.close()
        parts, scen = drift.frozen_ensemble(geom, inc.t0.tz_convert(None).to_pydatetime(), rdr, fz["ensemble"])
        parts.to_parquet(s.output(s.dir / "hindcast_ensemble_particles.parquet"), index=False)
        scen.to_csv(s.output(s.dir / "hindcast_ensemble_scenarios.csv"), index=False)
        res["ensemble"] = compare("ensemble_particles", s.dir / "hindcast_ensemble_particles.parquet", inc.frozen("ensemble_particles"))
        res["scenarios"] = compare("ensemble_scenarios", s.dir / "hindcast_ensemble_scenarios.csv", inc.frozen("ensemble_scenarios"))
        print(f"  {'OK ' if res['deterministic_bit_exact'] else 'DIFF'} deterministic hindcast", flush=True)
        s.notes.update(res)
        assert res["deterministic_bit_exact"] and res["ensemble"]["ulp_equal"], "drift parity FAILED"
    return res
