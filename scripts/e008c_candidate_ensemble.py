"""E008c: the frozen 27-scenario uncertainty ensemble (run_hindcast_ensemble.py, never edited or executed here) run
independently for every E008 policy v1 candidate with FROZEN forcing v2. No vessel attribution, no release-age scoring,
no candidate ranking or fusion. No reference-slick geometry is read.

  python scripts/e008c_candidate_ensemble.py audit                    # ensemble_config_audit.json (static + runtime diff)
  python scripts/e008c_candidate_ensemble.py pilot                    # E007-C1, 27 scenarios + pilot_acceptance.json
  python scripts/e008c_candidate_ensemble.py run-all --workers N      # remaining 41 candidates (+ C1 determinism re-run)
  python scripts/e008c_candidate_ensemble.py summary                  # scenario / candidate summaries, pairwise diagnostic
  python scripts/e008c_candidate_ensemble.py candidate --id E007-C7   # one candidate (used by pilot / run-all subprocesses)

Allowed differences from the baseline: input polygon (candidate instead of reference), forcing v1 -> frozen v2, output
location, candidate id, diagnostics. Scenario order within a candidate and reader sharing across its 27 scenarios are as
in the baseline; each candidate runs in its own process, so results do not depend on the worker count.
"""
import argparse
import ast
import json
import os
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Proj
from shapely.geometry import Point

from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e008a_candidate_hindcast as A  # noqa: E402  (hash helpers, candidate loading, coverage test; not its physics)

ROOT = Path("/home/admin_wsl/SIH26143")
OCEAN = ROOT / "data/incident_001/forcing_v2/ocean_opendrift.nc"
WIND = ROOT / "data/incident_001/forcing_v2/era5_wind_10m_151h.nc"
FORCING_FROZEN = ROOT / "data/incident_001/forcing_v2/FROZEN.json"
BASELINE = ROOT / "run_hindcast_ensemble.py"
OUT = ROOT / "results/E008c_candidate_ensemble"
PILOT = "E007-C1"
CHILD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
FORBIDDEN_TOKENS = ["slick_" + "3775938", "ceru" + "lean", "in_" + "roi", "incident" + ".json", "posthoc_" + "ceru"]
MAX_SERIAL_H = 4.0
AGE_BINS_H = [(0, 6), (6, 12), (12, 24), (24, 36), (36, 48)]

# ------------------------------------------------------------------ frozen ensemble constants (same names as the baseline)
T0 = datetime(2023, 1, 3, 0, 1, 42)
DURATION_HOURS = 48
PARTICLES_PER_SCENARIO = 400
BASE_SEED = 26143
WINDAGE = [0.02, 0.03, 0.04]
CURRENT_UNCERTAINTY = [0.00, 0.05, 0.10]
WIND_UNCERTAINTY = [0.00, 0.50, 1.00]
SCENARIOS = list(product(WINDAGE, CURRENT_UNCERTAINTY, WIND_UNCERTAINTY))


# ------------------------------------------------------------------ frozen recipe (mirrors run_hindcast_ensemble.py)
def sample_polygon(geometry, number, seed):
    """Baseline sampler; the baseline reads the bounds of the same polygon from module globals."""
    minx, miny, maxx, maxy = geometry.bounds
    rng = np.random.default_rng(seed)

    xs_out = []
    ys_out = []

    while len(xs_out) < number:

        need = number - len(xs_out)

        n_try = max(
            1000,
            need * 4
        )

        xs = rng.uniform(
            minx,
            maxx,
            n_try
        )

        ys = rng.uniform(
            miny,
            maxy,
            n_try
        )

        for x, y in zip(xs, ys):

            if geometry.contains(
                Point(x, y)
            ):
                xs_out.append(x)
                ys_out.append(y)

                if (
                    len(xs_out)
                    == number
                ):
                    break

    return (
        np.asarray(xs_out),
        np.asarray(ys_out),
    )


def candidate_geometry(geom):
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def make_readers():
    ocean_reader = reader_netCDF_CF_generic.Reader(str(OCEAN))
    wind_reader = reader_netCDF_CF_generic.Reader(str(WIND))
    return ocean_reader, wind_reader


def build_model(ocean_reader, wind_reader, current_unc, wind_unc):
    o = OpenOil(loglevel=50)
    o.add_reader([ocean_reader, wind_reader])
    o.set_config("processes:evaporation", False)
    o.set_config("processes:emulsification", False)
    o.set_config("processes:dispersion", False)
    o.set_config("processes:biodegradation", False)
    o.set_config("drift:vertical_mixing", False)
    o.set_config("drift:vertical_advection", False)
    o.set_config("drift:current_uncertainty", current_unc)
    o.set_config("drift:wind_uncertainty", wind_unc)
    o.set_config("environment:fallback:land_binary_mask", 0)
    return o


def run_scenario(ocean_reader, wind_reader, geom, scenario_id, windage, current_unc, wind_unc):
    scenario_seed = BASE_SEED + scenario_id * 1000
    np.random.seed(scenario_seed)
    lons, lats = sample_polygon(geom, PARTICLES_PER_SCENARIO, scenario_seed)
    o = build_model(ocean_reader, wind_reader, current_unc, wind_unc)
    o.seed_elements(lon=lons, lat=lats, time=T0, number=PARTICLES_PER_SCENARIO, wind_drift_factor=windage)
    result = o.run(duration=timedelta(hours=DURATION_HOURS), time_step=-900, time_step_output=3600)
    return o, result, scenario_seed, lons, lats


def scenario_frame(result, scenario_id, windage, current_unc, wind_unc, candidate_id):
    """Baseline particle table (same columns, dtypes and finite-row filter) + candidate_id, particle_id, age_hours."""
    lon = result["lon"].values
    lat = result["lat"].values
    result_times = pd.to_datetime(result["time"].values, utc=True)
    ntraj, ntime = lon.shape
    scenario_df = pd.DataFrame({
        "scenario_id": np.repeat(scenario_id, ntraj * ntime),
        "trajectory": np.repeat(np.arange(ntraj), ntime),
        "time": np.tile(result_times, ntraj),
        "lon": lon.reshape(-1),
        "lat": lat.reshape(-1),
        "windage": np.repeat(windage, ntraj * ntime),
        "current_uncertainty": np.repeat(current_unc, ntraj * ntime),
        "wind_uncertainty": np.repeat(wind_unc, ntraj * ntime),
    })
    scenario_df = scenario_df[np.isfinite(scenario_df["lon"]) & np.isfinite(scenario_df["lat"])].copy()
    scenario_df.insert(0, "candidate_id", candidate_id)
    scenario_df["particle_id"] = scenario_df["trajectory"]  # output trajectory index within the scenario (as baseline)
    scenario_df["age_hours"] = (scenario_df["time"] - pd.Timestamp(T0, tz="UTC")) / pd.Timedelta(hours=1)
    return scenario_df, ntraj, ntime


# ------------------------------------------------------------------ integrity
def forcing_check():
    fz = json.load(open(FORCING_FROZEN))
    bad = {k: v for k, v in fz["sha256"].items() if A.sha256(ROOT / k) != v}
    assert not bad, f"forcing v2 FROZEN mismatch: {bad}"
    assert str(OCEAN.relative_to(ROOT)) in fz["sha256"] and str(WIND.relative_to(ROOT)) in fz["sha256"]
    return dict(frozen_json_sha256=A.sha256(FORCING_FROZEN), files_verified=len(fz["sha256"]), sha256=fz["sha256"])


def integrity():
    C, prov = A.load_candidates()  # policy v1 sha, E007 FROZEN (all files), candidate output hashes
    return dict(forcing_v2=forcing_check(), policy_and_e007=prov, frozen=A.frozen_state())


# ------------------------------------------------------------------ config-diff audit vs run_hindcast_ensemble.py
def recipe(path, scope=None):
    tree = ast.parse(Path(path).read_text())
    nodes = tree.body if scope is None else [n for n in tree.body if isinstance(n, ast.Assign) or
                                             (isinstance(n, ast.FunctionDef) and n.name in scope)]
    r = dict(consts={}, scenarios=None, loop=None, scenario_seed=None, np_random_seed=None, sampler_call=None, sampler_rng=None,
             sampler_while=None, validity_fix=None, model=None, readers=[], add_reader=None, set_config=[], seed_elements=None,
             run=None, frame_columns=None, frame_filter=None)
    for top in nodes:
        for n in ast.walk(top):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                k, v = n.targets[0].id, ast.unparse(n.value)
                if k in ("T0", "DURATION_HOURS", "PARTICLES_PER_SCENARIO", "BASE_SEED", "WINDAGE", "CURRENT_UNCERTAINTY",
                         "WIND_UNCERTAINTY"):
                    r["consts"][k] = eval(v, {"datetime": datetime})  # trusted repo source; literals only
                elif k == "SCENARIOS":
                    r["scenarios"] = v
                elif k == "scenario_seed":
                    r["scenario_seed"] = v
                elif k == "rng":
                    r["sampler_rng"] = v
                elif k == "scenario_df" and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "pd.DataFrame":
                    d = n.value.args[0]
                    r["frame_columns"] = {ast.literal_eval(a): ast.unparse(b) for a, b in zip(d.keys, d.values)}
                elif k == "scenario_df":
                    r["frame_filter"] = v
            if isinstance(n, ast.For) and "SCENARIOS" in ast.unparse(n.iter):
                r["loop"] = dict(target=ast.unparse(n.target), iter=ast.unparse(n.iter))
            if isinstance(n, ast.While):
                r["sampler_while"] = ast.dump(n)
            if isinstance(n, ast.If) and "is_valid" in ast.unparse(n.test):
                r["validity_fix"] = ast.unparse(n)
            if isinstance(n, ast.Call):
                f = ast.unparse(n.func)
                if f == "np.random.seed":
                    r["np_random_seed"] = ast.unparse(n)
                elif f == "sample_polygon":
                    r["sampler_call"] = ast.unparse(n)
                elif f == "OpenOil":
                    r["model"] = ast.unparse(n)
                elif f.endswith("Reader"):
                    r["readers"].append(ast.unparse(n))
                elif f == "o.add_reader":
                    r["add_reader"] = ast.unparse(n.args[0])
                elif f == "o.set_config":
                    r["set_config"].append([ast.unparse(a) for a in n.args])
                elif f in ("o.seed_elements", "o.run"):
                    r[f[2:]] = {k.arg: ast.unparse(k.value) for k in n.keywords}
    return r


def scenario_table(consts, scen_expr, seed_expr):
    ns = dict(consts, product=product)
    rows = []
    for sid, (w, c, u) in enumerate(eval(scen_expr, ns), start=1):  # trusted repo source
        rows.append([sid, eval(seed_expr, dict(ns, scenario_id=sid)), w, c, u])
    return rows


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    integ = integrity()
    base = recipe(BASELINE)
    new = recipe(__file__, scope={"sample_polygon", "candidate_geometry", "make_readers", "build_model", "run_scenario",
                                  "scenario_frame", "run_candidate"})
    rows = []

    def cmp(field, b, n, allowed=False, note=None):
        rows.append(dict(field=field, baseline=b, e008c=n, equal=b == n, allowed_difference=allowed, note=note))

    for k in base["consts"]:
        cmp(f"const:{k}", str(base["consts"][k]), str(new["consts"].get(k)))
    cmp("scenario_grid_expression", base["scenarios"], new["scenarios"])
    tb = scenario_table(base["consts"], base["scenarios"], base["scenario_seed"])
    tn = scenario_table(new["consts"], new["scenarios"], new["scenario_seed"])
    cmp("scenario_table[id, seed, windage, current_unc, wind_unc] (ordering, ids, seeds, values)", tb, tn)
    cmp("scenario_loop", base["loop"], new["loop"])
    cmp("scenario_seed_formula", base["scenario_seed"], new["scenario_seed"])
    cmp("np_random_seed_call", base["np_random_seed"], new["np_random_seed"])
    cmp("sampler_call", base["sampler_call"], new["sampler_call"])
    cmp("sampler_rng", base["sampler_rng"], new["sampler_rng"])
    cmp("sampler_while_loop_ast", base["sampler_while"], new["sampler_while"])
    cmp("sampler_polygon_validity_fix", base["validity_fix"], new["validity_fix"])
    cmp("model_constructor", base["model"], new["model"])
    cmp("readers", base["readers"], new["readers"])
    cmp("add_reader", base["add_reader"], new["add_reader"])
    cmp("set_config_calls_ordered", base["set_config"], new["set_config"])
    for k in sorted(set(base["seed_elements"]) | set(new["seed_elements"])):
        cmp(f"seed_elements:{k}", base["seed_elements"].get(k), new["seed_elements"].get(k))
    for k in sorted(set(base["run"]) | set(new["run"])):
        cmp(f"run:{k}", base["run"].get(k), new["run"].get(k))
    cmp("output_frame_columns", base["frame_columns"], new["frame_columns"])
    cmp("output_frame_finite_filter", base["frame_filter"], new["frame_filter"])
    rows.append(dict(field="sampler_bounds", baseline="module globals = bounds of the (same) seed polygon",
                     e008c="geometry.bounds of the seed polygon passed in", equal=False, allowed_difference=True,
                     note="identical values for the same polygon; follows from the allowed input-polygon change"))
    rows.append(dict(field="input_polygon", baseline="reference polygon file", e008c="candidate polygon from "
                     "candidate_components.geojson", equal=False, allowed_difference=True))
    rows.append(dict(field="forcing_paths", baseline=["currents/ocean_opendrift.nc", "wind/era5_wind_10m_151h.nc"],
                     e008c=[str(OCEAN.relative_to(ROOT)), str(WIND.relative_to(ROOT))], equal=False, allowed_difference=True,
                     note="forcing v2 = v1 recipe with expanded coverage (E008b, FROZEN)"))
    rows.append(dict(field="output", baseline="results/incident_001/hindcast_ensemble_*", e008c="results/E008c_candidate_ensemble/"
                     "<candidate>/ensemble_*; extra columns candidate_id, particle_id, age_hours", equal=False, allowed_difference=True))
    rows.append(dict(field="reader_lifetime", baseline="2 readers created once, shared by all 27 scenarios in order",
                     e008c="2 readers created once per candidate (own process), shared by its 27 scenarios in order",
                     equal=True, allowed_difference=False))
    # Runtime: effective configuration for every scenario, E008c builder vs the baseline's parsed calls.
    oc, wd = make_readers()
    rt = {}
    stokes = {}
    for sid, (windage, current_unc, wind_unc) in enumerate(SCENARIOS, start=1):
        ob = OpenOil(loglevel=50)
        ob.add_reader([oc, wd])
        for k, v in base["set_config"]:
            ob.set_config(ast.literal_eval(k), eval(v, {}, dict(current_unc=current_unc, wind_unc=wind_unc)))
        on = build_model(oc, wd, current_unc, wind_unc)
        fb, fn = A.full_config(ob), A.full_config(on)
        same = lambda a, b: json.dumps(a, default=str) == json.dumps(b, default=str)
        d = {k: [fb.get(k), fn.get(k)] for k in sorted(set(fb) | set(fn)) if not same(fb.get(k), fn.get(k))}
        if d:
            rt[sid] = d
        stokes[sid] = {k: fn[k] for k in fn if "stokes" in k.lower()}
    provider = {v: [n for n, r in on.env.readers.items() if v in r.variables] for v in on.required_variables}
    src = Path(__file__).read_text().lower()
    hits = [t for t in FORBIDDEN_TOKENS if t in src]
    ok = all(r["equal"] or r["allowed_difference"] for r in rows) and not rt and not hits
    res = dict(all_equal_except_allowed=ok, allowed_differences=["input polygon", "forcing paths v1 -> frozen v2",
               "output location", "candidate ID", "diagnostics / instrumentation"], static_fields=rows,
               runtime_full_config_n_keys=len(fn), runtime_differences_per_scenario=rt,
               stokes_handling=dict(config_per_scenario_identical=len({json.dumps(v, default=str) for v in stokes.values()}) == 1,
                                    config=stokes[1], stokes_variables_provider={k: v for k, v in provider.items() if "stokes" in k}),
               required_variables_provider=provider, readers={t: A.reader_info(r) for t, r in (("current", oc), ("wind", wd))},
               static_forbidden_token_hits=hits, child_process_env=CHILD_ENV, integrity_before=integ,
               baseline_sha256=A.sha256(BASELINE), script_sha256=A.sha256(__file__), git_head=A.git("rev-parse", "HEAD"),
               opendrift_version=__import__("opendrift").__version__)
    A.dump(res, OUT / "ensemble_config_audit.json")
    for r in rows:
        if not (r["equal"] or r["allowed_difference"]):
            print("DIFF", r["field"], r["baseline"], r["e008c"])
    print(json.dumps(dict(all_equal_except_allowed=ok, runtime_diffs=rt, token_hits=hits,
                          n_static_fields=len(rows), stokes=res["stokes_handling"]), indent=1, default=str))


# ------------------------------------------------------------------ one candidate (runs in its own process)
def coverage(o, result, readers, lons0):
    t = result["time"].values
    hours = (t - np.datetime64(T0)) / np.timedelta64(1, "h")
    lon, lat, st = result["lon"].values, result["lat"].values, result["status"].values
    zero = {k: (result[a].values == 0) & (result[b].values == 0) for k, (a, b) in dict(
        current=("x_sea_water_velocity", "y_sea_water_velocity"), wind=("x_wind", "y_wind"),
        stokes=("sea_surface_wave_stokes_drift_x_velocity", "sea_surface_wave_stokes_drift_y_velocity")).items()}
    n0 = len(lons0)
    per = []
    for k in range(len(t)):
        act = np.isfinite(lon[:, k]) & np.isfinite(lat[:, k]) & (st[:, k] == 0)
        ic, iw = A.inside(readers["current"], lon[act, k], lat[act, k]), A.inside(readers["wind"], lon[act, k], lat[act, k])
        per.append(dict(hour=float(hours[k]), n_active=int(act.sum()), n_outside_current=int((~ic).sum()),
                        n_outside_wind=int((~iw).sum()), n_current_zero=int(zero["current"][act, k].sum()),
                        n_wind_zero=int(zero["wind"][act, k].sum()), n_stokes_zero=int(zero["stokes"][act, k].sum()),
                        n_active_nonfinite=int(((st[:, k] == 0) & ~(np.isfinite(lon[:, k]) & np.isfinite(lat[:, k]))).sum())))
    e = o.elements_deactivated
    nd = len(e.lon) if e is not None and e.lon is not None else 0
    dlon, dlat = np.asarray(e.lon if nd else [], float), np.asarray(e.lat if nd else [], float)
    dh = -np.asarray(e.age_seconds if nd else [], float) / 3600.0
    reasons = [o.status_categories[int(s)] for s in e.status] if nd else []
    out_c = ~A.inside(readers["current"], dlon, dlat) if nd else np.zeros(0, bool)
    out_w = ~A.inside(readers["wind"], dlon, dlat) if nd else np.zeros(0, bool)
    P = pd.DataFrame(per)
    reached = bool(len(t) == DURATION_HOURS + 1 and pd.Timestamp(t[-1]) == pd.Timestamp(T0 - timedelta(hours=DURATION_HOURS))
                   and o.time == T0 - timedelta(hours=DURATION_HOURS))

    def first_exit(col, dead_out):
        c = [float(-P.hour[P[col] > 0].iloc[0])] if (P[col] > 0).any() else []
        c += [float(dh[dead_out].min())] if dead_out.any() else []
        return min(c) if c else None

    fin = np.isfinite(lon) & np.isfinite(lat)
    xs, ys = lon[fin], lat[fin]
    marg = {}
    for tag, R in readers.items():
        sides = dict(w=np.sign(xs - R.xmin) * A.GEOD.inv(np.full_like(xs, R.xmin), ys, xs, ys)[2],
                     e=np.sign(R.xmax - xs) * A.GEOD.inv(xs, ys, np.full_like(xs, R.xmax), ys)[2],
                     s=np.sign(ys - R.ymin) * A.GEOD.inv(xs, np.full_like(ys, R.ymin), xs, ys)[2],
                     n=np.sign(R.ymax - ys) * A.GEOD.inv(xs, ys, xs, np.full_like(ys, R.ymax))[2])
        marg[f"realized_{tag}_min_margin_km"] = float(min(v.min() for v in sides.values()) / 1000.0)
        marg[f"realized_{tag}_min_margin_side"] = min(sides, key=lambda s: sides[s].min())
    d = dict(
        n_seeded=int(lon.shape[0]), n_seed_positions=n0, n_times=int(len(t)), reached_minus48h=reached,
        simulation_final_time=str(o.time), n_active_minus48h=int(n0 - nd) if reached else 0, n_deactivated=int(nd),
        deactivation_reasons={r: reasons.count(r) for r in sorted(set(reasons))},
        n_deactivated_outside_current=int(out_c.sum()), n_deactivated_outside_wind=int(out_w.sum()),
        first_deactivation_h=float(dh.min()) if nd else None,
        trajectory_min_lon=float(xs.min()), trajectory_max_lon=float(xs.max()),
        trajectory_min_lat=float(ys.min()), trajectory_max_lat=float(ys.max()),
        max_current_outside_frac=float(max((P.n_outside_current / n0).max(), out_c.sum() / n0)),
        max_wind_outside_frac=float(max((P.n_outside_wind / n0).max(), out_w.sum() / n0)),
        first_current_exit_h=first_exit("n_outside_current", out_c), first_wind_exit_h=first_exit("n_outside_wind", out_w),
        n_active_hours_current_zero=int(P.n_current_zero.sum()), n_active_hours_wind_zero=int(P.n_wind_zero.sum()),
        n_active_hours_stokes_zero=int(P.n_stokes_zero.sum()), n_active_nonfinite=int(P.n_active_nonfinite.sum()), **marg)
    d["physics_valid"] = bool(reached and d["n_seeded"] == PARTICLES_PER_SCENARIO and n0 == PARTICLES_PER_SCENARIO and nd == 0
                              and P.n_outside_current.sum() == 0 and P.n_outside_wind.sum() == 0
                              and d["n_active_hours_current_zero"] == 0 and d["n_active_hours_wind_zero"] == 0
                              and d["n_active_hours_stokes_zero"] == 0 and d["n_active_nonfinite"] == 0)
    d["per_output_time"] = per
    return d


def run_candidate(cid, outdir):
    t_start = time.perf_counter()
    C, _ = A.load_candidates()
    forcing_check()
    c = next(c for c in C if c["candidate_id"] == cid)
    cdir = Path(outdir) / cid
    assert not cdir.exists(), f"{cdir} exists; not overwritten"
    cdir.mkdir(parents=True)
    geom = candidate_geometry(c["geometry"])
    ocean_reader, wind_reader = make_readers()
    readers = {"current": ocean_reader, "wind": wind_reader}
    all_particles, scenario_records, cov = [], [], []
    for scenario_id, (windage, current_unc, wind_unc) in enumerate(SCENARIOS, start=1):
        t0 = time.perf_counter()
        o, result, scenario_seed, lons, lats = run_scenario(ocean_reader, wind_reader, geom, scenario_id, windage, current_unc, wind_unc)
        rt = time.perf_counter() - t0
        scenario_df, ntraj, ntime = scenario_frame(result, scenario_id, windage, current_unc, wind_unc, cid)
        t1 = time.perf_counter()
        d = coverage(o, result, readers, lons)
        all_particles.append(scenario_df)
        scenario_records.append(dict(candidate_id=cid, scenario_id=scenario_id, seed=scenario_seed, windage=windage,
                                     current_uncertainty=current_unc, wind_uncertainty=wind_unc, particles=ntraj, times=ntime,
                                     valid_rows=len(scenario_df), runtime_s=rt, diagnostics_s=time.perf_counter() - t1,
                                     physics_valid=d["physics_valid"]))
        cov.append(dict(candidate_id=cid, scenario_id=scenario_id, seed=scenario_seed, wind_drift_factor=windage,
                        current_uncertainty=current_unc, wind_uncertainty=wind_unc, runtime_s=rt, **d))
        del o, result
    particles = pd.concat(all_particles, ignore_index=True)
    particles.to_parquet(cdir / "ensemble_particles.parquet", index=False)
    pd.DataFrame(scenario_records).to_csv(cdir / "ensemble_scenarios.csv", index=False)
    A.dump(dict(candidate_id=cid, validity_rule="physics_valid = -48 h reached AND 400 seeded AND 0 deactivated AND 0 active "
                "particle-hours outside current or wind reader AND 0 active particle-hours with current, wind or Stokes exactly "
                "(0, 0) AND no non-finite active coordinates", scenarios=cov), cdir / "coverage_diagnostics.json")
    A.dump(dict(candidate_id=cid, component_id=c["component_id"], seed_area_m2=c["area_m2"],
                seed_geometry_valid=bool(c["geometry"].is_valid), n_scenarios=len(SCENARIOS),
                n_valid_scenarios=int(sum(r["physics_valid"] for r in scenario_records)), n_rows=len(particles),
                scenario_runtime_s=[r["runtime_s"] for r in scenario_records], total_runtime_s=time.perf_counter() - t_start,
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, pid=os.getpid(),
                env={k: os.environ.get(k) for k in CHILD_ENV}, forcing={str(p.relative_to(ROOT)): A.sha256(p) for p in (OCEAN, WIND)},
                script_sha256=A.sha256(__file__), baseline_sha256=A.sha256(BASELINE), git_head=A.git("rev-parse", "HEAD"),
                opendrift_version=__import__("opendrift").__version__,
                particle_table=dict(file="ensemble_particles.parquet", columns=list(particles.columns),
                                    note="all 49 hourly positions per particle retained (age_hours 0 .. -48) for later "
                                         "release-age binning; rows with non-finite lon/lat dropped as in the baseline"),
                age_bins_h_for_later_scoring=AGE_BINS_H), cdir / "ensemble_run_report.json")
    print(cid, "done", round(time.perf_counter() - t_start, 1), "s valid", sum(r["physics_valid"] for r in scenario_records), "/27",
          flush=True)


def spawn(cid, outdir):
    env = dict(os.environ, **CHILD_ENV)
    r = subprocess.run([sys.executable, __file__, "candidate", "--id", cid, "--out", str(outdir)], env=env,
                       capture_output=True, text=True)
    log = ROOT / "logs" / f"E008c_candidates/{cid}{'' if Path(outdir) == OUT else '_' + Path(outdir).name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + r.stderr)
    return cid, r.returncode


# ------------------------------------------------------------------ pilot
def pilot():
    audit_ = json.load(open(OUT / "ensemble_config_audit.json"))
    assert audit_["all_equal_except_allowed"], "config audit failed"
    C, _ = A.load_candidates()
    assert PILOT == min(C, key=lambda c: c["component_id"])["candidate_id"]
    before = integrity()
    t0 = time.perf_counter()
    cid, rc = spawn(PILOT, OUT)
    wall = time.perf_counter() - t0
    after = integrity()
    assert rc == 0, f"pilot process failed rc={rc}"
    cdir = OUT / PILOT
    cov = json.load(open(cdir / "coverage_diagnostics.json"))["scenarios"]
    rep = json.load(open(cdir / "ensemble_run_report.json"))
    sc = pd.read_csv(cdir / "ensemble_scenarios.csv")
    grid = [[i, BASE_SEED + i * 1000, w, c, u] for i, (w, c, u) in enumerate(SCENARIOS, start=1)]
    base_grid = [r for r in audit_["static_fields"] if r["field"].startswith("scenario_table")][0]["baseline"]
    checks = dict(
        n_scenarios_27=len(cov) == 27,
        seeded_400_all=all(d["n_seeded"] == 400 and d["n_seed_positions"] == 400 for d in cov),
        times_49_all=all(d["n_times"] == 49 for d in cov),
        reached_minus48h_all=all(d["reached_minus48h"] for d in cov),
        no_deactivations=all(d["n_deactivated"] == 0 for d in cov),
        no_current_exit=all(d["max_current_outside_frac"] == 0 for d in cov),
        no_wind_exit=all(d["max_wind_outside_frac"] == 0 for d in cov),
        no_fallback_zero=all(d["n_active_hours_current_zero"] == 0 and d["n_active_hours_wind_zero"] == 0
                             and d["n_active_hours_stokes_zero"] == 0 for d in cov),
        parameters_match_frozen_grid=sc[["scenario_id", "seed", "windage", "current_uncertainty", "wind_uncertainty"]].values.tolist() == grid,
        seeds_match_frozen_formula=[r[1] for r in base_grid] == sc.seed.tolist(),
        frozen_files_unchanged=before == after,
        all_physics_valid=all(d["physics_valid"] for d in cov))
    rts = np.asarray(rep["scenario_runtime_s"])
    est_h = float(rts.mean() * 27 * 42 / 3600.0)
    acc = dict(candidate_id=PILOT, selection_rule="smallest retained component_id (software smoke test, not an attribution choice)",
               passed=all(checks.values()), checks=checks, process_wall_s=wall, candidate_total_runtime_s=rep["total_runtime_s"],
               scenario_runtime_s=dict(sum=float(rts.sum()), mean=float(rts.mean()), median=float(np.median(rts)), min=float(rts.min()),
                                       max=float(rts.max()), p90=float(np.quantile(rts, 0.9)), per_scenario=rts.tolist()),
               peak_rss_mb=rep["peak_rss_mb"], estimated_serial_runtime_h=est_h,
               estimate_formula="C1 mean scenario runtime * 27 * 42", continue_automatically=bool(est_h <= MAX_SERIAL_H),
               integrity_before=before, integrity_after=after)
    A.dump(acc, OUT / "pilot_acceptance.json")
    print(json.dumps({k: v for k, v in acc.items() if not k.startswith("integrity")}, indent=1, default=str))


# ------------------------------------------------------------------ all candidates
def meminfo_mb(key):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 1024.0


def run_all(workers):
    acc = json.load(open(OUT / "pilot_acceptance.json"))
    assert acc["passed"] and acc["continue_automatically"], "pilot not passed or runtime estimate > 4 h"
    C, _ = A.load_candidates()
    todo = [c["candidate_id"] for c in C if not (OUT / c["candidate_id"]).exists()]  # component-id order
    before = integrity()
    stop = threading.Event()
    low = dict(min_available_mb=meminfo_mb("MemAvailable"), max_swap_used_mb=meminfo_mb("SwapTotal") - meminfo_mb("SwapFree"))
    swap0 = low["max_swap_used_mb"]

    def watch():
        while not stop.wait(1.0):
            low["min_available_mb"] = min(low["min_available_mb"], meminfo_mb("MemAvailable"))
            low["max_swap_used_mb"] = max(low["max_swap_used_mb"], meminfo_mb("SwapTotal") - meminfo_mb("SwapFree"))
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    det = OUT / "_determinism_check"
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(spawn, cid, OUT) for cid in todo] + [ex.submit(spawn, PILOT, det)]
        res = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    stop.set()
    after = integrity()
    a = pd.read_parquet(OUT / PILOT / "ensemble_particles.parquet")
    b = pd.read_parquet(det / PILOT / "ensemble_particles.parquet")
    same = bool(a.equals(b))
    (det / PILOT / "ensemble_particles.parquet").unlink()  # duplicate of the pilot table; equality result is kept
    rep = dict(workers=workers, cpu_count=os.cpu_count(), mem_total_mb=meminfo_mb("MemTotal"), wall_s=wall,
               return_codes=dict(res), n_candidates_run=len(todo), min_available_mb_during_run=low["min_available_mb"],
               swap_used_mb_before=swap0, max_swap_used_mb_during_run=low["max_swap_used_mb"],
               peak_rss_mb_per_candidate_process={cid: json.load(open(OUT / cid / "ensemble_run_report.json"))["peak_rss_mb"]
                                                  for cid in todo if (OUT / cid / "ensemble_run_report.json").exists()},
               determinism_check=dict(candidate=PILOT, rerun_in_parallel_pool=True, particle_table_identical_to_pilot=same),
               integrity_unchanged=before == after, integrity_before=before, integrity_after=after)
    A.dump(rep, OUT / "run_all_report.json")
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("integrity_")}, indent=1, default=str))
    assert before == after, "frozen files changed"


# ------------------------------------------------------------------ summaries (descriptive only; no score, no ranking)
def summary():
    C, _ = A.load_candidates()
    ids = [c["candidate_id"] for c in C]
    scen = pd.concat([pd.DataFrame(json.load(open(OUT / i / "coverage_diagnostics.json"))["scenarios"]).drop(columns=["per_output_time"])
                      for i in ids], ignore_index=True)
    scen.to_csv(OUT / "scenario_run_summary.csv", index=False)
    rows, ends = [], {}
    for i in ids:
        p = pd.read_parquet(OUT / i / "ensemble_particles.parquet", columns=["scenario_id", "trajectory", "age_hours", "lon", "lat"])
        e = p[p.age_hours == -DURATION_HOURS]
        sv = scen[scen.candidate_id == i]
        lon0, lat0 = e.lon.mean(), e.lat.mean()
        laea = Proj(proj="laea", lon_0=lon0, lat_0=lat0, ellps="WGS84")
        x, y = laea(e.lon.values.astype("f8"), e.lat.values.astype("f8"))
        x, y = x / 1000.0, y / 1000.0
        mx, my = np.median(x), np.median(y)
        cov = np.cov(np.vstack([x, y]))
        ev, evec = np.linalg.eigh(cov)
        r = np.hypot(x - mx, y - my)
        g = e.assign(x=x, y=y).groupby("scenario_id")[["x", "y"]].mean()
        gc = np.hypot(g.x - g.x.mean(), g.y - g.y.mean())
        rows.append(dict(candidate_id=i, component_id=int(i[6:]), n_scenarios=len(sv), n_valid_scenarios=int(sv.physics_valid.sum()),
                         n_endpoint_particles=len(e), endpoint_centroid_lon=lon0, endpoint_centroid_lat=lat0,
                         endpoint_median_lon=float(np.median(e.lon)), endpoint_median_lat=float(np.median(e.lat)),
                         rms_spread_km=float(np.sqrt(np.mean(x ** 2 + y ** 2))),
                         cov_xx_km2=cov[0, 0], cov_yy_km2=cov[1, 1], cov_xy_km2=cov[0, 1],
                         ellipse_1sigma_major_km=float(np.sqrt(ev[1])), ellipse_1sigma_minor_km=float(np.sqrt(ev[0])),
                         ellipse_major_azimuth_deg=float(np.degrees(np.arctan2(evec[0, 1], evec[1, 1])) % 180.0),
                         r50_km_about_median=float(np.quantile(r, 0.5)), r90_km_about_median=float(np.quantile(r, 0.9)),
                         scenario_centroid_rms_km=float(np.sqrt(np.mean(gc ** 2))), scenario_centroid_max_km=float(gc.max()),
                         min_realized_current_margin_km=float(sv.realized_current_min_margin_km.min()),
                         min_realized_wind_margin_km=float(sv.realized_wind_min_margin_km.min()),
                         max_current_outside_frac=float(sv.max_current_outside_frac.max()),
                         max_wind_outside_frac=float(sv.max_wind_outside_frac.max()), runtime_s=float(sv.runtime_s.sum())))
        ends[i] = e[["lon", "lat"]].values.astype("f8")
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "candidate_ensemble_summary.csv", index=False)
    # Optional pairwise diagnostic: descriptive only. No threshold, no clustering, no fusion.
    from scipy.spatial.distance import cdist
    lon0 = np.mean([c["geometry"].centroid.x for c in C])
    lat0 = np.mean([c["geometry"].centroid.y for c in C])
    laea = Proj(proj="laea", lon_0=lon0, lat_0=lat0, ellps="WGS84")
    xy = {i: np.column_stack(laea(v[:, 0], v[:, 1])) / 1000.0 for i, v in ends.items()}
    sub = {i: v[np.linspace(0, len(v) - 1, min(2000, len(v))).astype(int)] for i, v in xy.items()}  # deterministic subsample
    self_e = {i: cdist(v, v).mean() for i, v in sub.items()}
    edges = np.arange(-800, 800.001, 5.0)
    hist = {i: np.histogram2d(v[:, 0], v[:, 1], bins=[edges, edges])[0] / len(v) for i, v in xy.items()}
    Si = S.set_index("candidate_id")
    P = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            P.append(dict(candidate_a=i, candidate_b=j,
                          centroid_distance_km=A.GEOD.inv(Si.endpoint_centroid_lon[i], Si.endpoint_centroid_lat[i],
                                                          Si.endpoint_centroid_lon[j], Si.endpoint_centroid_lat[j])[2] / 1000,
                          median_distance_km=A.GEOD.inv(Si.endpoint_median_lon[i], Si.endpoint_median_lat[i],
                                                        Si.endpoint_median_lon[j], Si.endpoint_median_lat[j])[2] / 1000,
                          energy_distance_km=float(max(2 * cdist(sub[i], sub[j]).mean() - self_e[i] - self_e[j], 0.0)),
                          histogram_overlap_5km=float(np.minimum(hist[i], hist[j]).sum())))
    pd.DataFrame(P).to_parquet(OUT / "candidate_origin_pairwise.parquet", index=False)
    A.dump(dict(note="EXPLORATORY ONLY. Pairwise distances between -48 h ensemble endpoint clouds (all 27 scenarios pooled). "
                     "No threshold, no clustering, no fusion; overlapping origins do not show a common release.",
                laea_center=[lon0, lat0], energy_distance="2E|X-Y| - E|X-X'| - E|Y-Y'| (km), deterministic 2000-point subsample",
                histogram_overlap="sum over 5 km cells of min(p_a, p_b)", n_pairs=len(P)), OUT / "candidate_origin_pairwise_README.json")
    A.dump(dict(age_bins_h=AGE_BINS_H, note="Hourly positions (age_hours 0 .. -48) of every particle and scenario are kept in "
                "<candidate>/ensemble_particles.parquet so later release-age scoring needs no physics re-run. Nothing is scored here."),
           OUT / "release_age_products_README.json")
    print(S.round(3).to_string(index=False))
    print("scenarios", len(scen), "valid", int(scen.physics_valid.sum()), "candidates with invalid scenarios",
          sorted(scen[~scen.physics_valid].candidate_id.unique().tolist()),
          "min realized current/wind margin km", scen.realized_current_min_margin_km.min(), scen.realized_wind_min_margin_km.min())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["audit", "pilot", "run-all", "summary", "candidate"])
    ap.add_argument("--id")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    if a.stage == "candidate":
        run_candidate(a.id, a.out)
    elif a.stage == "run-all":
        run_all(a.workers)
    else:
        dict(audit=audit, pilot=pilot, summary=summary)[a.stage]()
