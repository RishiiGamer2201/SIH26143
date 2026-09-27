"""E008b: the 42 E008a deterministic candidate hindcasts re-run with forcing v2 (expanded spatial coverage, same products /
recipe). Everything is the committed E008a code (scripts/e008a_candidate_hindcast.py, imported, not edited); only its OCEAN,
WIND and OUT module constants are replaced. No ensemble, no vessel attribution, no release-age scoring, no candidate fusion.

  python scripts/e008b_candidate_hindcast_v2.py preflight   # runtime config v1 == v2, reader info, v2 coverage preflight
  python scripts/e008b_candidate_hindcast_v2.py run         # 42 runs, component-id order (in-memory deactivation diagnostics)
  python scripts/e008b_candidate_hindcast_v2.py summary     # candidate_hindcast_summary.csv (+ realized margins)
  python scripts/e008b_candidate_hindcast_v2.py compare     # v1_v2_trajectory_comparison.csv for the 40 v1-valid candidates
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e008a_candidate_hindcast as A  # noqa: E402

ROOT = A.ROOT
V1 = dict(ocean=A.OCEAN, wind=A.WIND, out=A.OUT)
V2_OCEAN = ROOT / "data/incident_001/forcing_v2/ocean_opendrift.nc"
V2_WIND = ROOT / "data/incident_001/forcing_v2/era5_wind_10m_151h.nc"
OUT = ROOT / "results/E008b_candidate_hindcast_v2"
VALIDATION = ROOT / "results/E008b_forcing_expansion/forcing_overlap_validation.json"


def use(version):
    A.OCEAN, A.WIND, A.OUT = (V1["ocean"], V1["wind"], V1["out"]) if version == "v1" else (V2_OCEAN, V2_WIND, OUT)


def e008a_state():
    return {str(p.relative_to(V1["out"])): A.sha256(p) for p in sorted(V1["out"].rglob("*")) if p.is_file()}


def preflight():
    assert json.load(open(VALIDATION))["equivalence_pass"], "forcing v2 equivalence did not pass; stop"
    OUT.mkdir(parents=True, exist_ok=True)
    C, prov = A.load_candidates()
    use("v1")
    f1, o1 = A.full_config(A.build_model()), A.build_model()
    use("v2")
    o2 = A.build_model()
    f2 = A.full_config(o2)
    same = lambda a, b: json.dumps(a, default=str) == json.dumps(b, default=str)
    diff = {k: [f1.get(k), f2.get(k)] for k in sorted(set(f1) | set(f2)) if not same(f1.get(k), f2.get(k))}
    rd = lambda o: {t: A.reader_info(o.env.readers[n]) for t, n in zip(("current", "wind"), list(o.env.readers))}
    r1, r2 = rd(o1), rd(o2)
    readers = {"current": o2.env.readers[list(o2.env.readers)[0]], "wind": o2.env.readers[list(o2.env.readers)[1]]}
    pre = A.preflight_rows(C, readers)
    pre.to_csv(OUT / "forcing_coverage_preflight.csv", index=False)
    audit = dict(
        recipe="E008a code imported unchanged; only OCEAN/WIND (forcing v2) and OUT differ",
        e008a_script_sha256=A.sha256(A.__file__), e008b_script_sha256=A.sha256(__file__), provenance=prov,
        runtime_full_config_n_keys=len(f2), runtime_full_config_differences_v1_vs_v2=diff,
        set_config_calls=A.ast_recipe(A.__file__, func_scope={"build_model"})["set_config"],
        readers_v1=r1, readers_v2=r2,
        reader_differences={t: {k: [r1[t][k], r2[t][k]] for k in r1[t] if r1[t][k] != r2[t][k]} for t in r1},
        forcing_files={str(p.relative_to(ROOT)): A.sha256(p) for p in (V2_OCEAN, V2_WIND)},
        validity_rule=A.VALIDITY_RULE, risk_flag_km=A.RISK_KM, frozen_state_before=A.frozen_state())
    audit["config_equal"] = not diff
    A.dump(audit, OUT / "physics_config_audit.json")
    print(json.dumps(dict(config_equal=audit["config_equal"], reader_differences=audit["reader_differences"]), indent=1, default=str))
    print(pre[["candidate_id", "min_margin_km", "forcing_risk_flag"]].to_string(index=False))


def run():
    audit = json.load(open(OUT / "physics_config_audit.json"))
    assert audit["config_equal"]
    C, _ = A.load_candidates()
    use("v2")
    readers = A.coverage_readers()
    before, ea = A.frozen_state(), e008a_state()
    for c in C:  # component-id order; A.run_one refuses existing output dirs
        cfg, diag, _ = A.run_one(c, readers)
        cfg.update(forcing_version="v2", e008b_script_sha256=A.sha256(__file__))
        A.dump(cfg, OUT / c["candidate_id"] / "run_config.json")
        if diag is not None:
            diag.update(diagnostics_version="v1 (in-memory deactivation record; reached -48 h required)",
                        diagnostics_script_sha256=A.sha256(A.__file__))
            A.dump(diag, OUT / c["candidate_id"] / "coverage_diagnostics.json")
        print(c["candidate_id"], cfg["status"], round(cfg["runtime_s"], 1), "valid", diag and diag["physics_valid_for_attribution"],
              "dead", diag and diag["n_deactivated"], "outside cur/wind", diag and (diag["max_current_outside_frac"], diag["max_wind_outside_frac"]),
              "rerun==file", diag and diag["rerun_matches_saved_file"], flush=True)
    assert A.frozen_state() == before, "frozen incident outputs or root scripts changed"
    assert e008a_state() == ea, "E008a outputs changed"
    print("frozen state and E008a outputs unchanged")


def realized_margins(nc, readers):
    """Minimum signed distance (km) of any finite trajectory position to each reader edge (negative = outside)."""
    d = xr.open_dataset(nc)
    lon, lat = d.lon.values.ravel(), d.lat.values.ravel()
    d.close()
    ok = np.isfinite(lon) & np.isfinite(lat)
    lon, lat = lon[ok], lat[ok]
    r = {}
    for t, R in readers.items():
        side = dict(w=np.sign(lon - R.xmin) * A.GEOD.inv(np.full_like(lon, R.xmin), lat, lon, lat)[2],
                    e=np.sign(R.xmax - lon) * A.GEOD.inv(lon, lat, np.full_like(lon, R.xmax), lat)[2],
                    s=np.sign(lat - R.ymin) * A.GEOD.inv(lon, np.full_like(lat, R.ymin), lon, lat)[2],
                    n=np.sign(R.ymax - lat) * A.GEOD.inv(lon, lat, lon, np.full_like(lat, R.ymax))[2])
        for s, v in side.items():
            r[f"realized_{t}_margin_{s}_km"] = float(v.min() / 1000.0)
        r[f"realized_{t}_min_margin_km"] = min(r[f"realized_{t}_margin_{s}_km"] for s in side)
    r["realized_min_margin_km"] = min(r["realized_current_min_margin_km"], r["realized_wind_min_margin_km"])
    return r


def summary():
    use("v2")
    A.summary()
    readers = A.coverage_readers()
    S = pd.read_csv(OUT / "candidate_hindcast_summary.csv")
    pre = pd.read_csv(OUT / "forcing_coverage_preflight.csv").set_index("candidate_id")
    extra = []
    for cid in S.candidate_id:
        dg = json.load(open(OUT / cid / "coverage_diagnostics.json"))
        m = realized_margins(OUT / cid / "hindcast_48h.nc", readers)
        m.update(candidate_id=cid, forcing_min_margin_km=pre.loc[cid, "min_margin_km"], forcing_risk_flag=pre.loc[cid, "forcing_risk_flag"],
                 n_deactivated_outside_current=dg["n_deactivated_outside_current"], n_deactivated_outside_wind=dg["n_deactivated_outside_wind"],
                 deactivation_reasons=json.dumps(dg["deactivation_reasons"]), rerun_matches_saved_file=dg["rerun_matches_saved_file"])
        extra.append(m)
    S = S.merge(pd.DataFrame(extra), on="candidate_id")
    S.to_csv(OUT / "candidate_hindcast_summary.csv", index=False)
    print(S[["candidate_id", "run_status", "reached_minus48h", "n_active_minus48h", "n_deactivated", "physics_valid_for_attribution",
             "forcing_min_margin_km", "realized_current_min_margin_km", "realized_wind_min_margin_km", "runtime_s"]].round(2).to_string(index=False))
    print("completed", int((S.run_status == "completed").sum()), "valid", int(S.physics_valid_for_attribution.sum()),
          "current exits", int((S.max_current_outside_frac > 0).sum()), "wind exits", int((S.max_wind_outside_frac > 0).sum()),
          "runs with deactivations", int((S.n_deactivated > 0).sum()), "runtime_s", round(S.runtime_s.sum(), 1))


def compare():
    S1 = pd.read_csv(V1["out"] / "candidate_hindcast_summary.csv")
    valid = S1[S1.physics_valid_for_attribution].candidate_id.tolist()
    assert len(valid) == 40
    rows = []
    for cid in valid:
        a, b = xr.open_dataset(V1["out"] / cid / "hindcast_48h.nc"), xr.open_dataset(OUT / cid / "hindcast_48h.nc")
        g1 = json.load(open(V1["out"] / cid / "coverage_diagnostics.json"))
        g2 = json.load(open(OUT / cid / "coverage_diagnostics.json"))
        c1 = json.load(open(V1["out"] / cid / "run_config.json"))
        c2 = json.load(open(OUT / cid / "run_config.json"))
        same_shape = a.lon.shape == b.lon.shape and np.array_equal(a.time.values, b.time.values)
        r = dict(candidate_id=cid, run_status_v1=c1["status"], run_status_v2=c2["status"], status_match=c1["status"] == c2["status"],
                 valid_v1=g1["physics_valid_for_attribution"], valid_v2=g2["physics_valid_for_attribution"],
                 n_active_minus48h_v1=g1["n_active_minus48h"], n_active_minus48h_v2=g2["n_active_minus48h"],
                 active_count_match=[p["n_active"] for p in g1["per_output_time"]] == [p["n_active"] for p in g2["per_output_time"]],
                 same_shape_and_times=bool(same_shape))
        if same_shape:
            fa = np.isfinite(a.lon.values) & np.isfinite(a.lat.values)
            fb = np.isfinite(b.lon.values) & np.isfinite(b.lat.values)
            both = fa & fb
            dl, dt = np.abs(a.lon.values - b.lon.values)[both], np.abs(a.lat.values - b.lat.values)[both]
            dist = A.GEOD.inv(a.lon.values[:, -1], a.lat.values[:, -1], b.lon.values[:, -1], b.lat.values[:, -1])[2] / 1000.0
            r.update(status_arrays_equal=bool(np.array_equal(a.status.values, b.status.values)),
                     finite_mask_equal=bool(np.array_equal(fa, fb)), n_positions_compared=int(both.sum()),
                     exact_equal_where_finite=bool((dl == 0).all() and (dt == 0).all()),
                     max_abs_dlon=float(dl.max()), max_abs_dlat=float(dt.max()),
                     endpoint_distance_median_km=float(np.nanmedian(dist)), endpoint_distance_max_km=float(np.nanmax(dist)),
                     all_variables_bit_equal=bool(all(np.array_equal(a[v].values, b[v].values, equal_nan=a[v].dtype.kind == "f")
                                                      for v in a.data_vars)),
                     origin_cloud_file_identical=A.sha256(V1["out"] / cid / "origin_cloud_48h.csv") == A.sha256(OUT / cid / "origin_cloud_48h.csv"))
        a.close(), b.close()
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "v1_v2_trajectory_comparison.csv", index=False)
    print(D.to_string(index=False))
    print("exact where finite:", int(D.exact_equal_where_finite.sum()), "/", len(D), "| status match", int(D.status_match.sum()),
          "| active-count match", int(D.active_count_match.sum()), "| all vars bit-equal", int(D.all_variables_bit_equal.sum()),
          "| max dlon", D.max_abs_dlon.max(), "max dlat", D.max_abs_dlat.max(), "| endpoint max km", D.endpoint_distance_max_km.max())


if __name__ == "__main__":
    {"preflight": preflight, "run": run, "summary": summary, "compare": compare}[sys.argv[1]]()
