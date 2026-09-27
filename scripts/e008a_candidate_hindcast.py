"""E008a: candidate-driven deterministic 48 h backward hindcast with the FROZEN baseline physics recipe
(run_hindcast_48h.py, which is never edited or executed here). One run per E008 policy v1 candidate, each seeded
from its own predicted polygon. No ensemble, no AIS, no release-age scoring. No reference-slick geometry is read.

  python scripts/e008a_candidate_hindcast.py preflight               # input hashes, forcing bounds, per-candidate coverage,
                                                                     # physics config diff vs baseline, fallback probe
  python scripts/e008a_candidate_hindcast.py run --candidate E007-C1 # technical pilot (smallest component id) + acceptance
  python scripts/e008a_candidate_hindcast.py run --all               # remaining candidates, component-id order
  python scripts/e008a_candidate_hindcast.py summary                 # candidate_hindcast_summary.csv + pairwise origin diagnostic

Differences from the baseline allowed by review: seed geometry, output paths, candidate id, instrumentation.
Sampling: same rejection sampler (verbatim), same base seed RNG_SEED = 26143 for every candidate, N = 2000 each
(not area-proportional).
"""
import argparse
import ast
import hashlib
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod, Proj
from shapely.geometry import shape, Point

from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic

ROOT = Path("/home/admin_wsl/SIH26143")
OCEAN = ROOT / "data/incident_001/currents/ocean_opendrift.nc"
WIND = ROOT / "data/incident_001/wind/era5_wind_10m_151h.nc"
T0 = datetime(2023, 1, 3, 0, 1, 42)
N_PARTICLES = 2000
RNG_SEED = 26143

BASELINE = ROOT / "run_hindcast_48h.py"
FROZEN_ROOT_SCRIPTS = ["run_hindcast_48h.py", "run_hindcast_ensemble.py"]
FROZEN_INCIDENT_DIR = ROOT / "results/incident_001"
POLICY = ROOT / "configs/e008_candidate_policy_v1.json"
POLICY_SHA = ROOT / "configs/e008_candidate_policy_v1.json.sha256"
CAND_DIR = ROOT / "results/E008_candidate_policy_v1"
CANDIDATES = CAND_DIR / "candidate_components.geojson"
OUT = ROOT / "results/E008a_candidate_hindcast"
PILOT = "E007-C1"
# static self-check: tokens that must not appear anywhere in this file (split so the check does not match itself)
FORBIDDEN_TOKENS = ["slick_" + "3775938", "ceru" + "lean", "in_" + "roi", "incident" + ".json", "posthoc_" + "ceru"]
# Predeclared (before any candidate run) descriptive flag; does NOT exclude anything. 100 km ~ 0.58 m/s for 48 h.
RISK_KM = {"high": 50.0, "moderate": 100.0}
VALIDITY_RULE = ("physics_valid_for_attribution = run completed AND 2000 seeded AND 2000 active at -48 h AND every output time "
                 "inside both readers' time coverage AND zero active particle-hours outside current or wind reader horizontal "
                 "coverage AND zero active particle-hours with exported current, Stokes or wind exactly (0, 0) (fallback marker)")
GEOD = Geod(ellps="WGS84")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def dump(obj, p):
    Path(p).write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


# ------------------------------------------------------------------ inputs
def load_candidates():
    """Verify policy, E007 freeze and candidate files, then return candidates in component-id order."""
    pol = json.load(open(POLICY))
    assert sha256(POLICY) == POLICY_SHA.read_text().split()[0], "policy v1 JSON changed"
    fz_path = ROOT / pol["segmentation_source"]["frozen_manifest"]
    assert sha256(fz_path) == pol["segmentation_source"]["frozen_manifest_sha256"], "E007 FROZEN.json changed"
    fz = json.load(open(fz_path))
    bad = [k for k, v in fz["sha256"].items() if sha256(ROOT / k) != v["sha256"]]
    assert not bad, f"E007 frozen artefacts changed: {bad}"
    app = json.load(open(CAND_DIR / "application_report.json"))
    for f, h in app["output_sha256"].items():
        assert sha256(CAND_DIR / f) == h, f"{f} changed since policy application"
    fc = json.load(open(CANDIDATES))
    C = [dict(candidate_id=f["properties"]["candidate_id"], component_id=int(f["properties"]["component_id"]),
              area_m2=f["properties"]["area_m2"], geometry=shape(f["geometry"])) for f in fc["features"]]
    C.sort(key=lambda c: c["component_id"])
    assert len(C) == 42 and [c["candidate_id"] for c in C] == app["candidate_ids"]
    return C, dict(policy_sha256=sha256(POLICY), e007_frozen_sha256=sha256(fz_path), e007_frozen_files_verified=len(fz["sha256"]),
                   candidate_files_sha256=app["output_sha256"])


def frozen_state():
    return dict(incident_001={p.name: sha256(p) for p in sorted(FROZEN_INCIDENT_DIR.iterdir()) if p.is_file()},
                root_scripts={s: sha256(ROOT / s) for s in FROZEN_ROOT_SCRIPTS},
                root_scripts_git_clean=git("status", "--porcelain", "--", *FROZEN_ROOT_SCRIPTS) == "")


# ------------------------------------------------------------------ frozen recipe (mirrors run_hindcast_48h.py)
def sample_polygon(geom):
    """Baseline rejection sampler, verbatim (the While loop is AST-compared with the baseline in preflight)."""
    if not geom.is_valid:
        geom = geom.buffer(0)
    rng = np.random.default_rng(RNG_SEED)
    minx, miny, maxx, maxy = geom.bounds
    lons = []
    lats = []
    while len(lons) < N_PARTICLES:

        # generate batches to avoid one-at-a-time sampling
        n_need = N_PARTICLES - len(lons)
        n_try = max(1000, n_need * 3)

        xs = rng.uniform(minx, maxx, n_try)
        ys = rng.uniform(miny, maxy, n_try)

        for x, y in zip(xs, ys):

            if geom.contains(Point(x, y)):

                lons.append(x)
                lats.append(y)

                if len(lons) == N_PARTICLES:
                    break
    return np.asarray(lons), np.asarray(lats)


def build_model():
    ocean_reader = reader_netCDF_CF_generic.Reader(str(OCEAN))
    wind_reader = reader_netCDF_CF_generic.Reader(str(WIND))
    o = OpenOil(loglevel=20)
    o.add_reader([ocean_reader, wind_reader])
    o.set_config("processes:evaporation", False)
    o.set_config("processes:emulsification", False)
    o.set_config("processes:dispersion", False)
    o.set_config("processes:biodegradation", False)
    o.set_config("drift:vertical_mixing", False)
    o.set_config("drift:vertical_advection", False)
    o.set_config("drift:current_uncertainty", 0)
    o.set_config("drift:wind_uncertainty", 0)
    o.set_config("environment:fallback:land_binary_mask", 0)
    return o


def seed_and_run(o, lons, lats, outfile):
    o.seed_elements(lon=lons, lat=lats, time=T0, number=N_PARTICLES, wind_drift_factor=0.03)
    o.run(duration=timedelta(hours=48), time_step=-900, time_step_output=3600, outfile=str(outfile))


def write_origin_cloud(nc, dst):
    """Baseline origin-cloud extraction: earliest physical time, finite positions, same CSV format."""
    ds = xr.open_dataset(nc)
    earliest_idx = int(np.argmin(ds["time"].values))
    olon = ds["lon"].isel(time=earliest_idx).values
    olat = ds["lat"].isel(time=earliest_idx).values
    valid = np.isfinite(olon) & np.isfinite(olat)
    np.savetxt(dst, np.column_stack([olon[valid], olat[valid]]), delimiter=",", header="longitude,latitude", comments="")
    ds.close()


# ------------------------------------------------------------------ physics config comparison (static + runtime)
def ast_recipe(path, func_scope=None):
    """Physics-relevant calls/constants extracted from source without executing it."""
    tree = ast.parse(Path(path).read_text())
    if func_scope:
        tree = ast.Module(body=[n for n in tree.body if isinstance(n, ast.Assign) or
                                (isinstance(n, ast.FunctionDef) and n.name in func_scope)], type_ignores=[])
    r = dict(set_config=[], seed_elements=None, run=None, model=None, readers=[], add_reader=None, consts={}, rng=None,
             validity_fix=None, sampler_while=[])
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            k = n.targets[0].id
            if k in ("T0", "N_PARTICLES", "RNG_SEED", "OCEAN", "WIND", "ROOT"):
                r["consts"][k] = ast.unparse(n.value)
            if k == "rng":
                r["rng"] = ast.unparse(n.value)
        if isinstance(n, ast.If) and "is_valid" in ast.unparse(n.test):
            r["validity_fix"] = ast.unparse(n)
        if isinstance(n, ast.While):
            r["sampler_while"].append(ast.dump(n))
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == "set_config":
                r["set_config"].append([ast.literal_eval(a) for a in n.args])
            elif name in ("seed_elements", "run"):
                r[name] = {k.arg: ast.unparse(k.value) for k in n.keywords}
            elif name == "OpenOil":
                r["model"] = ast.unparse(n)
            elif name == "Reader":
                r["readers"].append(ast.unparse(n))
            elif name == "add_reader":
                r["add_reader"] = ast.unparse(n.args[0])
    return r


def eval_consts(c):
    ns = dict(Path=Path, datetime=datetime)
    for k in ("ROOT", "OCEAN", "WIND", "T0", "N_PARTICLES", "RNG_SEED"):
        ns[k] = eval(c[k], {}, ns)  # trusted repo source; constants only
    return {k: str(ns[k]) for k in ("OCEAN", "WIND", "T0", "N_PARTICLES", "RNG_SEED")}


def full_config(o):
    return {k: o.get_config(k) for k in sorted(o._config)}


def config_comparison():
    base = ast_recipe(BASELINE)
    new = ast_recipe(__file__, func_scope={"sample_polygon", "build_model", "seed_and_run"})
    allowed = {"run": {"outfile"}}
    rows = []

    def cmp(field, b, n, allowed_diff=False):
        rows.append(dict(field=field, baseline=b, e008a=n, equal=b == n, allowed_difference=allowed_diff))

    bc, nc = eval_consts(base["consts"]), eval_consts(new["consts"])
    for k in bc:
        cmp(f"const:{k}", bc[k], nc[k])
    cmp("model_constructor", base["model"], new["model"])
    cmp("readers", base["readers"], new["readers"])
    cmp("add_reader", base["add_reader"], new["add_reader"])
    cmp("set_config_calls_ordered", base["set_config"], new["set_config"])
    for k in sorted(set(base["seed_elements"]) | set(new["seed_elements"])):
        cmp(f"seed_elements:{k}", base["seed_elements"].get(k), new["seed_elements"].get(k))
    for k in sorted(set(base["run"]) | set(new["run"])):
        cmp(f"run:{k}", base["run"].get(k), new["run"].get(k), k in allowed["run"])
    cmp("sampler_rng", base["rng"], new["rng"])
    cmp("sampler_validity_fix", base["validity_fix"], new["validity_fix"])
    cmp("sampler_while_loop_ast", base["sampler_while"], new["sampler_while"])
    rows.append(dict(field="seed_geometry", baseline="reference polygon file (SLICK constant)", e008a="candidate polygon from "
                     "candidate_components.geojson", equal=False, allowed_difference=True))
    # Runtime: effective config of the E008a model vs a model built from the baseline's parsed calls.
    ob = OpenOil(loglevel=50)
    ob.add_reader([reader_netCDF_CF_generic.Reader(str(OCEAN)), reader_netCDF_CF_generic.Reader(str(WIND))])
    for k, v in base["set_config"]:
        ob.set_config(k, v)
    on = build_model()
    fb, fn = full_config(ob), full_config(on)
    same = lambda a, b: json.dumps(a, default=str) == json.dumps(b, default=str)  # NaN-safe
    rt_diff = {k: [fb.get(k), fn.get(k)] for k in sorted(set(fb) | set(fn)) if not same(fb.get(k), fn.get(k))}
    ok = all(r["equal"] or r["allowed_difference"] for r in rows) and not rt_diff
    return dict(all_equal_except_allowed=ok, allowed_differences=["seed geometry", "output directory / filenames",
                "candidate identifier", "instrumentation / coverage checks"], static_fields=rows,
                runtime_full_config_n_keys=len(fn), runtime_full_config_differences=rt_diff), fn, on


# ------------------------------------------------------------------ forcing coverage
def reader_info(r):
    zmid = 0.0 if (r.zmin is None or r.zmax is None or r.zmin <= 0 <= r.zmax) else r.zmin
    return dict(name=r.name, variables=sorted(r.variables), xmin=float(r.xmin), xmax=float(r.xmax), ymin=float(r.ymin),
                ymax=float(r.ymax), delta_x=r.delta_x, delta_y=r.delta_y, zmin=r.zmin, zmax=r.zmax, z_for_coverage=zmid,
                start_time=str(r.start_time), end_time=str(r.end_time), time_step=str(r.time_step), proj=str(r.proj4))


def inside(r, lon, lat):
    info = reader_info(r)
    m = np.zeros(len(lon), bool)
    ok = np.isfinite(lon) & np.isfinite(lat)
    if ok.any():
        idx = r.covers_positions(lon[ok], lat[ok], z=info["z_for_coverage"])[0]
        m[np.flatnonzero(ok)[idx]] = True
    return m


def margin_km(lon0, lat0, lon1, lat1):
    return GEOD.inv(lon0, lat0, lon1, lat1)[2] / 1000.0


def preflight_rows(C, readers):
    rows = []
    for c in C:
        w, s, e, n = c["geometry"].bounds
        cy, cx = (s + n) / 2, (w + e) / 2
        row = dict(candidate_id=c["candidate_id"], component_id=c["component_id"], polygon_west=w, polygon_east=e,
                   polygon_south=s, polygon_north=n, centroid_lon=c["geometry"].centroid.x, centroid_lat=c["geometry"].centroid.y)
        mins = []
        for tag, r in readers.items():
            mw = np.sign(w - r.xmin) * margin_km(r.xmin, cy, w, cy)
            me = np.sign(r.xmax - e) * margin_km(e, cy, r.xmax, cy)
            ms = np.sign(s - r.ymin) * margin_km(cx, r.ymin, cx, s)
            mn = np.sign(r.ymax - n) * margin_km(cx, n, cx, r.ymax)
            row.update({f"{tag}_margin_w_km": mw, f"{tag}_margin_e_km": me, f"{tag}_margin_s_km": ms, f"{tag}_margin_n_km": mn})
            ex = np.asarray(c["geometry"].exterior.coords) if c["geometry"].geom_type == "Polygon" else \
                np.vstack([np.asarray(g.exterior.coords) for g in c["geometry"].geoms])
            row[f"initial_polygon_{tag}_covered"] = bool(inside(r, ex[:, 0], ex[:, 1]).all() and min(mw, me, ms, mn) > 0)
            mins += [mw, me, ms, mn]
        m = min(mins)
        row["min_margin_km"] = m
        row["forcing_risk_flag"] = ("outside" if not (row["initial_polygon_current_covered"] and row["initial_polygon_wind_covered"])
                                    else "high" if m < RISK_KM["high"] else "moderate" if m < RISK_KM["moderate"] else "low")
        rows.append(row)
    return pd.DataFrame(rows)


def fallback_probe(o, readers):
    """Empirical check of out-of-domain behaviour: 4 probe elements, 1 h backward, frozen config. Not a hindcast."""
    oc, wd = readers["current"], readers["wind"]
    ds = xr.open_dataset(OCEAN)
    u = ds["x_sea_water_velocity"].isel(time=int(np.argmin(np.abs(ds.time.values - np.datetime64(T0)))), depth=0).values
    nan = ~np.isfinite(u)
    core = nan.copy()
    core[1:-1, 1:-1] = np.all([np.roll(np.roll(nan, i, 0), j, 1)[1:-1, 1:-1] for i in (-1, 0, 1) for j in (-1, 0, 1)], axis=0)
    core[[0, -1], :] = False
    core[:, [0, -1]] = False
    pts = {"inside_all": (-90.0, 27.0),
           "outside_wind_only": ((wd.xmax + oc.xmax) / 2, 27.5) if oc.xmax > wd.xmax else None,
           "outside_all_east": (max(oc.xmax, wd.xmax) + 0.5, 27.5)}
    if core.any():
        j, i = np.argwhere(core)[0]
        pts["current_nan_cell_inside_bbox"] = (float(ds.longitude.values[i]), float(ds.latitude.values[j]))
    ds.close()
    pts = {k: v for k, v in pts.items() if v is not None}
    names = list(pts)
    lon = np.array([pts[k][0] for k in names])
    lat = np.array([pts[k][1] for k in names])
    pdir = OUT / "fallback_probe"
    pdir.mkdir(parents=True, exist_ok=True)
    f = pdir / "fallback_probe.nc"
    if f.exists():
        f.unlink()
    o.seed_elements(lon=lon, lat=lat, time=T0, number=len(lon), wind_drift_factor=0.03)
    o.run(duration=timedelta(hours=1), time_step=-900, time_step_output=900, outfile=str(f))
    d = xr.open_dataset(f)
    vars_ = ["status", "lon", "lat", "x_sea_water_velocity", "y_sea_water_velocity", "sea_surface_wave_stokes_drift_x_velocity",
             "sea_surface_wave_stokes_drift_y_velocity", "x_wind", "y_wind", "land_binary_mask"]
    res = {}
    # OpenDrift does not keep seed order in the output file: match each probe by its first recorded position.
    first_lon, first_lat = d.lon.values[:, 0], d.lat.values[:, 0]
    for q, name in enumerate(names):
        k = int(np.nanargmin(np.hypot(first_lon - lon[q], first_lat - lat[q])))
        res[name] = dict(seed_lon=float(lon[q]), seed_lat=float(lat[q]), output_trajectory_index=k,
                         first_recorded_lon=float(first_lon[k]), first_recorded_lat=float(first_lat[k]),
                         in_current_reader=bool(inside(oc, lon[q:q + 1], lat[q:q + 1])[0]),
                         in_wind_reader=bool(inside(wd, lon[q:q + 1], lat[q:q + 1])[0]),
                         per_output_time={v: [None if not np.isfinite(x) else float(x) for x in d[v].values[k]] for v in vars_})
    flags = d.status.attrs.get("flag_meanings")
    d.close()
    od = xr.open_dataset(OCEAN)
    data_facts = {v: dict(n_nan=int(np.isnan(od[v].values).sum()), n_exact_zero=int((od[v].values == 0).sum()), n=int(od[v].size))
                  for v in od.data_vars}
    od.close()
    return dict(note="1 h backward from T0 with the frozen configuration; exported environment values are post-fallback",
                status_flag_meanings=flags, messages=[str(m) for m in getattr(o, "messages", [])][:50], probes=res,
                current_file_value_facts=data_facts,
                land_cell_probe=("skipped: no current cell with an all-NaN 3x3 neighbourhood (the file has only a few NaN cells)"
                                 if "current_nan_cell_inside_bbox" not in res else "included"))


def preflight():
    OUT.mkdir(parents=True, exist_ok=True)
    C, prov = load_candidates()
    src = Path(__file__).read_text()
    token_hits = [t for t in FORBIDDEN_TOKENS if t.lower() in src.lower()]
    comp, cfg, o = config_comparison()
    readers = {"current": o.env.readers[list(o.env.readers)[0]], "wind": o.env.readers[list(o.env.readers)[1]]}
    assert OCEAN.name in readers["current"].name and WIND.name in readers["wind"].name
    rinfo = {k: reader_info(r) for k, r in readers.items()}
    pre = preflight_rows(C, readers)
    pre.to_csv(OUT / "forcing_coverage_preflight.csv", index=False)
    req = {v: (o.required_variables[v].get("fallback")) for v in o.required_variables}
    fallbacks = {k.split(":", 2)[2]: v for k, v in cfg.items() if k.startswith("environment:fallback:")}
    probe = fallback_probe(build_model(), readers)
    fm = str(probe["status_flag_meanings"]).split()
    probe["observed_behaviour"] = {
        k: dict(in_current_reader=v["in_current_reader"], in_wind_reader=v["in_wind_reader"],
                deactivated=any(s not in (None, 0.0) for s in v["per_output_time"]["status"]),
                reason=[fm[int(s)] for s in v["per_output_time"]["status"] if s not in (None, 0.0)][:1],
                exported_at_first_record={x: v["per_output_time"][x][0] for x in (
                    "x_sea_water_velocity", "sea_surface_wave_stokes_drift_x_velocity", "x_wind", "land_binary_mask")})
        for k, v in probe["probes"].items()}
    # Stokes / wind / current provenance per required variable
    provider = {v: [n for n, r in o.env.readers.items() if v in r.variables] for v in o.required_variables}
    zero_fallback = sorted(v for v, fb in fallbacks.items() if fb == 0 and v in (
        "x_sea_water_velocity", "y_sea_water_velocity", "x_wind", "y_wind", "sea_surface_wave_stokes_drift_x_velocity",
        "sea_surface_wave_stokes_drift_y_velocity", "land_binary_mask"))
    audit = dict(
        opendrift_version=__import__("opendrift").__version__, provenance=prov, frozen_state_before=frozen_state(),
        forcing_files={str(p.relative_to(ROOT)): sha256(p) for p in (OCEAN, WIND)}, readers=rinfo,
        run_time_window=dict(start=str(T0), end=str(T0 - timedelta(hours=48)),
                             inside_current_time=bool(readers["current"].covers_time(T0 - timedelta(hours=48)) and readers["current"].covers_time(T0)),
                             inside_wind_time=bool(readers["wind"].covers_time(T0 - timedelta(hours=48)) and readers["wind"].covers_time(T0))),
        config_comparison=comp, required_variables_provider=provider, required_variables_default_fallback=req,
        effective_fallbacks=fallbacks,
        physically_important_zero_fallbacks=zero_fallback,
        relevant_config={k: cfg.get(k) for k in cfg if k.startswith(("drift:", "general:", "processes:", "environment:fallback:land",
                                                                      "seed:wind_drift_factor"))},
        opendrift_code_behaviour=dict(
            source="opendrift/models/basemodel/environment.py get_environment() and basemodel report_missing_variables()",
            variable_with_fallback="reader NotCoveredError or masked/NaN value -> element gets the configured fallback value "
                                   "(silent in output; logged at DEBUG only)",
            variable_without_fallback="masked value -> element deactivated with reason 'missing_data'",
            domain_deactivation="only if drift:deactivate_{north,south,east,west}_of are set",
            stokes="if drift:use_tabularised_stokes_drift is True and Stokes max over all elements == 0, Stokes is "
                   "parameterised from wind"),
        fallback_probe=probe,
        static_forbidden_token_hits=token_hits,
        predeclared=dict(validity_rule=VALIDITY_RULE, risk_flag_km=RISK_KM, pilot_rule="smallest retained component_id",
                         run_order="component_id"))
    dump(audit, OUT / "physics_config_audit.json")
    print(json.dumps(dict(config_equal=comp["all_equal_except_allowed"], runtime_diff=comp["runtime_full_config_differences"],
                          token_hits=token_hits, zero_fallbacks=zero_fallback, readers=rinfo), indent=1, default=str))
    print(pre[["candidate_id", "min_margin_km", "forcing_risk_flag", "initial_polygon_current_covered",
               "initial_polygon_wind_covered"]].to_string(index=False))
    for k, v in probe["probes"].items():
        pt = v["per_output_time"]
        print(k, v["in_current_reader"], v["in_wind_reader"], "status", pt["status"], "u", pt["x_sea_water_velocity"][:2],
              "stokes", pt["sea_surface_wave_stokes_drift_x_velocity"][:2], "wind", pt["x_wind"][:2], "land", pt["land_binary_mask"][:1])


# ------------------------------------------------------------------ per-candidate run + instrumentation
def sim_info(o, nc):
    """Deactivation record + final time from the model object; also checks that the object's trajectories equal the file."""
    e = o.elements_deactivated
    n = len(e.lon) if e is not None and e.lon is not None else 0
    d = xr.open_dataset(nc)
    r = o.result
    nt = d.sizes["time"]
    match = bool(r.lon.shape[0] == d.lon.shape[0] and r.lon.shape[1] >= nt
                 and np.array_equal(r.time.values[:nt], d.time.values)
                 and np.array_equal(r.lon.values[:, :nt], d.lon.values, equal_nan=True)
                 and np.array_equal(r.lat.values[:, :nt], d.lat.values, equal_nan=True))
    d.close()
    return dict(final_time=o.time, dead_lon=np.asarray(e.lon if n else [], float), dead_lat=np.asarray(e.lat if n else [], float),
                dead_reason=np.asarray([o.status_categories[int(s)] for s in e.status] if n else [], str),
                dead_hours_before_T0=-np.asarray(e.age_seconds if n else [], float) / 3600.0,
                messages=[str(m) for m in getattr(o, "_messages", getattr(o, "messages", []))],
                rerun_matches_saved_file=match)


def rerun_in_memory(c, nc):
    """Identical deterministic run without writing a file (the saved hindcast is never touched)."""
    lons, lats = sample_polygon(c["geometry"])
    o = build_model()
    o.seed_elements(lon=lons, lat=lats, time=T0, number=N_PARTICLES, wind_drift_factor=0.03)
    o.run(duration=timedelta(hours=48), time_step=-900, time_step_output=3600)
    return sim_info(o, nc)


def diagnostics(nc, readers, sim):
    d = xr.open_dataset(nc)
    t = d.time.values
    hours = (t - np.datetime64(T0)) / np.timedelta64(1, "h")
    lon, lat, st = d.lon.values, d.lat.values, d.status.values
    zero = {k: (d[a].values == 0) & (d[b].values == 0) for k, (a, b) in dict(
        current=("x_sea_water_velocity", "y_sea_water_velocity"), wind=("x_wind", "y_wind"),
        stokes=("sea_surface_wave_stokes_drift_x_velocity", "sea_surface_wave_stokes_drift_y_velocity")).items()}
    per = []
    for k in range(len(t)):
        act = np.isfinite(lon[:, k]) & np.isfinite(lat[:, k]) & (st[:, k] == 0)
        ic, iw = inside(readers["current"], lon[act, k], lat[act, k]), inside(readers["wind"], lon[act, k], lat[act, k])
        na = int(act.sum())
        per.append(dict(hour=float(hours[k]), n_active=na, n_deactivated=int(N_PARTICLES - na),
                        n_status_active_nonfinite=int(((st[:, k] == 0) & ~(np.isfinite(lon[:, k]) & np.isfinite(lat[:, k]))).sum()),
                        n_inside_current=int(ic.sum()), n_outside_current=int(na - ic.sum()),
                        n_inside_wind=int(iw.sum()), n_outside_wind=int(na - iw.sum()),
                        n_current_fallback_zero=int(zero["current"][act, k].sum()), n_wind_fallback_zero=int(zero["wind"][act, k].sum()),
                        n_stokes_fallback_zero=int(zero["stokes"][act, k].sum()),
                        in_current_time=bool(readers["current"].covers_time(pd.Timestamp(t[k]).to_pydatetime())),
                        in_wind_time=bool(readers["wind"].covers_time(pd.Timestamp(t[k]).to_pydatetime()))))
    # Deactivations come from OpenDrift's in-memory elements_deactivated (position, reason and age at deactivation).
    # The output file cannot be used for this: when every element is deactivated OpenDrift stops early and the final
    # deactivating step is never written (the file then ends before -48 h with all elements still 'active').
    dlon, dlat, dh = sim["dead_lon"], sim["dead_lat"], sim["dead_hours_before_T0"]
    dead_out_cur = ~inside(readers["current"], dlon, dlat) if len(dlon) else np.zeros(0, bool)
    dead_out_wind = ~inside(readers["wind"], dlon, dlat) if len(dlon) else np.zeros(0, bool)
    for r in per:  # cumulative outside counts incl. particles deactivated at a reader boundary up to this output time
        gone = dh <= -r["hour"] + 1e-6
        r["n_deactivated_outside_current_cum"] = int((gone & dead_out_cur).sum())
        r["n_deactivated_outside_wind_cum"] = int((gone & dead_out_wind).sum())
        r["n_outside_current_incl_deactivated"] = r["n_outside_current"] + r["n_deactivated_outside_current_cum"]
        r["n_outside_wind_incl_deactivated"] = r["n_outside_wind"] + r["n_deactivated_outside_wind_cum"]
    P = pd.DataFrame(per)
    frac = lambda col: (P[col] / N_PARTICLES)
    first = lambda col: (float(-P.hour[P[col] > 0].iloc[0]) if (P[col] > 0).any() else None)  # hours before T0

    def first_exit(col, dead_out):
        c = [x for x in [first(col), float(dh[dead_out].min()) if dead_out.any() else None] if x is not None]
        return min(c) if c else None

    reached = bool(len(t) == 49 and pd.Timestamp(t[-1]) == pd.Timestamp(T0 - timedelta(hours=48))
                   and sim["final_time"] == T0 - timedelta(hours=48))
    n_dead = len(dlon)
    reasons, counts = np.unique(sim["dead_reason"], return_counts=True)
    out = dict(
        n_times=len(t), time_first=str(t[0]), time_last=str(t[-1]), backward=bool(np.all(np.diff(t) < np.timedelta64(0))),
        reached_minus48h=reached, simulation_final_time=str(sim["final_time"]), stopped_early=not reached,
        n_seeded=int(d.sizes["trajectory"]), n_finite_at_T0=int((np.isfinite(lon[:, 0]) & np.isfinite(lat[:, 0])).sum()),
        n_active_minus48h=int(N_PARTICLES - n_dead) if reached else 0, n_deactivated=int(n_dead),
        n_active_at_last_output_record=int(P.n_active.iloc[-1]), last_output_hour=float(hours[-1]),
        deactivation_reasons={str(a): int(b) for a, b in zip(reasons, counts)},
        n_deactivated_outside_current=int(dead_out_cur.sum()), n_deactivated_outside_wind=int(dead_out_wind.sum()),
        n_deactivated_inside_both=int((~dead_out_cur & ~dead_out_wind).sum()),
        deactivation_lon_range=[float(dlon.min()), float(dlon.max())] if n_dead else None,
        deactivation_lat_range=[float(dlat.min()), float(dlat.max())] if n_dead else None,
        first_deactivation_h=float(dh.min()) if n_dead else None,
        max_current_outside_frac=float(max(frac("n_outside_current_incl_deactivated").max(), dead_out_cur.sum() / N_PARTICLES)),
        max_wind_outside_frac=float(max(frac("n_outside_wind_incl_deactivated").max(), dead_out_wind.sum() / N_PARTICLES)),
        current_outside_frac_minus48h=float(dead_out_cur.sum() / N_PARTICLES) if not reached else float(frac("n_outside_current_incl_deactivated").iloc[-1]),
        wind_outside_frac_minus48h=float(dead_out_wind.sum() / N_PARTICLES) if not reached else float(frac("n_outside_wind_incl_deactivated").iloc[-1]),
        first_current_exit_h=first_exit("n_outside_current", dead_out_cur), first_wind_exit_h=first_exit("n_outside_wind", dead_out_wind),
        first_current_fallback_zero_h=first("n_current_fallback_zero"), first_wind_fallback_zero_h=first("n_wind_fallback_zero"),
        max_current_fallback_zero_frac=float(frac("n_current_fallback_zero").max()),
        max_wind_fallback_zero_frac=float(frac("n_wind_fallback_zero").max()),
        max_stokes_fallback_zero_frac=float(frac("n_stokes_fallback_zero").max()),
        all_times_in_reader_time_coverage=bool(P.in_current_time.all() and P.in_wind_time.all()),
        trajectory_min_lon=float(np.nanmin(lon)), trajectory_max_lon=float(np.nanmax(lon)),
        trajectory_min_lat=float(np.nanmin(lat)), trajectory_max_lat=float(np.nanmax(lat)),
        note="inside = reader.covers_positions. Active-particle coverage from hourly output positions; deactivations (position, "
             "reason, time = -age) from OpenDrift elements_deactivated of an identical in-memory run checked bit-exact against "
             "the saved file. *_outside_frac = active outside + deactivated outside (cumulative). first_*_exit_h = hours before "
             "T0. Fallback marker = exported environment pair exactly (0, 0).",
        simulation_messages=sim["messages"], rerun_matches_saved_file=sim["rerun_matches_saved_file"],
        per_output_time=per)
    out["physics_valid_for_attribution"] = bool(
        reached and out["n_seeded"] == N_PARTICLES and out["n_active_minus48h"] == N_PARTICLES and n_dead == 0
        and out["all_times_in_reader_time_coverage"]
        and P.n_outside_current.sum() == 0 and P.n_outside_wind.sum() == 0 and P.n_current_fallback_zero.sum() == 0
        and P.n_wind_fallback_zero.sum() == 0 and P.n_stokes_fallback_zero.sum() == 0)
    schema = dict(dims=dict(d.sizes), variables=sorted(d.data_vars))
    d.close()
    return out, schema


def coverage_readers():
    return {"current": reader_netCDF_CF_generic.Reader(str(OCEAN)), "wind": reader_netCDF_CF_generic.Reader(str(WIND))}


def run_one(c, readers):
    cdir = OUT / c["candidate_id"]
    assert not cdir.exists(), f"{cdir} exists; E008a outputs are not overwritten"
    cdir.mkdir(parents=True)
    t0 = time.perf_counter()
    lons, lats = sample_polygon(c["geometry"])
    o = build_model()
    status = "completed"
    try:
        seed_and_run(o, lons, lats, cdir / "hindcast_48h.nc")
    except Exception as ex:  # keep the failure as a diagnostic
        status = f"failed: {ex!r}"
    runtime = time.perf_counter() - t0
    cfg = dict(candidate_id=c["candidate_id"], component_id=c["component_id"], status=status, runtime_s=runtime,
               seed_source=str(CANDIDATES.relative_to(ROOT)), seed_area_m2=c["area_m2"],
               seed_centroid_lon=c["geometry"].centroid.x, seed_centroid_lat=c["geometry"].centroid.y,
               seed_geometry_valid=bool(c["geometry"].is_valid), n_particles=N_PARTICLES, rng_seed=RNG_SEED, T0=str(T0),
               duration_h=48, time_step_s=-900, time_step_output_s=3600, wind_drift_factor=0.03,
               set_config=ast_recipe(__file__, func_scope={"build_model"})["set_config"],
               forcing={str(p.relative_to(ROOT)): sha256(p) for p in (OCEAN, WIND)},
               opendrift_version=__import__("opendrift").__version__, script_sha256=sha256(__file__), git_head=git("rev-parse", "HEAD"),
               seed_lon_min=float(lons.min()), seed_lon_max=float(lons.max()), seed_lat_min=float(lats.min()), seed_lat_max=float(lats.max()))
    dump(cfg, cdir / "run_config.json")
    if status != "completed":
        dump(dict(status=status, physics_valid_for_attribution=False), cdir / "coverage_diagnostics.json")
        return cfg, None, None
    write_origin_cloud(cdir / "hindcast_48h.nc", cdir / "origin_cloud_48h.csv")
    diag, schema = diagnostics(cdir / "hindcast_48h.nc", readers, rerun_in_memory(c, cdir / "hindcast_48h.nc"))
    diag["status"] = status
    dump(diag, cdir / "coverage_diagnostics.json")
    return cfg, diag, schema


def rediagnose():
    """Recompute coverage_diagnostics.json for existing runs (v0 inferred deactivations from the file's last records and
    missed early-stopped runs). Saved hindcasts / origin clouds are not touched; v0 files move to <cand>/superseded/."""
    C, _ = load_candidates()
    readers = coverage_readers()
    before = frozen_state()
    for c in C:
        cdir = OUT / c["candidate_id"]
        nc = cdir / "hindcast_48h.nc"
        h_nc = sha256(nc)
        sup = cdir / "superseded"
        sup.mkdir(exist_ok=True)
        old = cdir / "coverage_diagnostics.json"
        v0 = sup / "coverage_diagnostics.v0_file_last_record_inference.json"
        if not v0.exists():
            old.rename(v0)
        diag, _ = diagnostics(nc, readers, rerun_in_memory(c, nc))
        diag["status"] = json.load(open(cdir / "run_config.json"))["status"]
        diag["diagnostics_version"] = "v1 (in-memory deactivation record; reached -48 h required)"
        diag["diagnostics_script_sha256"] = sha256(__file__)
        dump(diag, old)
        assert sha256(nc) == h_nc
        print(c["candidate_id"], "reached", diag["reached_minus48h"], "dead", diag["n_deactivated"], diag["deactivation_reasons"],
              "out cur/wind", diag["n_deactivated_outside_current"], diag["n_deactivated_outside_wind"],
              "rerun==file", diag["rerun_matches_saved_file"], "valid", diag["physics_valid_for_attribution"], flush=True)
    assert frozen_state() == before
    print("frozen state unchanged")


def run(args):
    C, _ = load_candidates()
    audit = json.load(open(OUT / "physics_config_audit.json"))
    assert audit["config_comparison"]["all_equal_except_allowed"] and not audit["static_forbidden_token_hits"]
    readers = coverage_readers()
    if args.candidate:
        assert args.candidate == PILOT == min(C, key=lambda c: c["component_id"])["candidate_id"], "pilot = smallest component id"
        before = frozen_state()
        cfg, diag, schema = run_one(next(c for c in C if c["candidate_id"] == PILOT), readers)
        after = frozen_state()
        bd = xr.open_dataset(FROZEN_INCIDENT_DIR / "hindcast_48h.nc")  # schema only, read-only
        base_schema = dict(dims=dict(bd.sizes), variables=sorted(bd.data_vars))
        bd.close()
        checks = dict(
            completed=cfg["status"] == "completed",
            physics_config_equal_to_baseline=audit["config_comparison"]["all_equal_except_allowed"],
            seed_is_candidate_polygon=cfg["seed_source"].endswith("candidate_components.geojson") and cfg["candidate_id"] == PILOT,
            seeded_2000=bool(diag and diag["n_seeded"] == N_PARTICLES and diag["n_finite_at_T0"] == N_PARTICLES),
            times_49=bool(diag and diag["n_times"] == 49),
            backward_T0_to_minus48h=bool(diag and diag["backward"] and diag["time_first"].startswith("2023-01-03T00:01:42")
                                         and diag["time_last"].startswith("2023-01-01T00:01:42")),
            active_coordinates_finite=bool(diag and all(r["n_status_active_nonfinite"] == 0 for r in diag["per_output_time"])),
            output_schema_matches_baseline=schema == base_schema,
            frozen_incident_001_unchanged=before["incident_001"] == after["incident_001"],
            frozen_root_scripts_unchanged=before["root_scripts"] == after["root_scripts"] and after["root_scripts_git_clean"],
            reader_behaviour_audited=bool(audit.get("fallback_probe")),
            no_reference_geometry_dependency=not audit["static_forbidden_token_hits"])
        acc = dict(candidate_id=PILOT, selection_rule="smallest retained component_id (software smoke test, not an attribution choice)",
                   passed=all(checks.values()), checks=checks, runtime_s=cfg["runtime_s"],
                   physics_valid_for_attribution=bool(diag and diag["physics_valid_for_attribution"]),
                   schema_diff=None if schema == base_schema else dict(e008a=schema, baseline=base_schema))
        dump(acc, OUT / "pilot_acceptance.json")
        print(json.dumps({k: v for k, v in acc.items()}, indent=1, default=str))
        return
    acc = json.load(open(OUT / "pilot_acceptance.json"))
    assert acc["passed"], "pilot did not pass"
    before = frozen_state()
    for c in C:  # component-id order
        if (OUT / c["candidate_id"]).exists():
            continue
        cfg, diag, _ = run_one(c, readers)
        print(c["candidate_id"], cfg["status"], round(cfg["runtime_s"], 1), "valid", diag and diag["physics_valid_for_attribution"],
              "outside cur/wind", diag and (diag["max_current_outside_frac"], diag["max_wind_outside_frac"]), flush=True)
    after = frozen_state()
    assert before == after, "frozen incident outputs or root scripts changed"
    print("frozen state unchanged")


# ------------------------------------------------------------------ cross-candidate outputs (descriptive; no score, no ranking)
def summary():
    C, _ = load_candidates()
    lon0 = np.mean([c["geometry"].centroid.x for c in C])
    lat0 = np.mean([c["geometry"].centroid.y for c in C])
    laea = Proj(proj="laea", lon_0=lon0, lat_0=lat0, ellps="WGS84")
    rows, clouds = [], {}
    for c in C:
        cdir = OUT / c["candidate_id"]
        cfg = json.load(open(cdir / "run_config.json"))
        dg = json.load(open(cdir / "coverage_diagnostics.json"))
        r = dict(candidate_id=c["candidate_id"], component_id=c["component_id"], seed_area_m2=c["area_m2"],
                 seed_centroid_lon=cfg["seed_centroid_lon"], seed_centroid_lat=cfg["seed_centroid_lat"],
                 n_particles_seeded=dg.get("n_seeded"), n_active_minus48h=dg.get("n_active_minus48h"), n_deactivated=dg.get("n_deactivated"))
        if cfg["status"] == "completed":
            oc = pd.read_csv(cdir / "origin_cloud_48h.csv")
            x, y = laea(oc.longitude.values, oc.latitude.values)
            clouds[c["candidate_id"]] = np.column_stack([x, y]) / 1000.0
            mx, my = x.mean(), y.mean()
            r.update(origin_mean_lon=oc.longitude.mean(), origin_mean_lat=oc.latitude.mean(),
                     origin_median_lon=oc.longitude.median(), origin_median_lat=oc.latitude.median(),
                     origin_spread_km=float(np.sqrt(np.mean((x - mx) ** 2 + (y - my) ** 2)) / 1000.0))
        r.update({k: dg.get(k) for k in ("trajectory_min_lon", "trajectory_max_lon", "trajectory_min_lat", "trajectory_max_lat",
                                         "max_current_outside_frac", "max_wind_outside_frac", "first_current_exit_h", "first_wind_exit_h")})
        r.update(reached_minus48h=dg.get("reached_minus48h"), origin_cloud_hour=dg.get("last_output_hour"),
                 origin_cloud_n=len(clouds.get(c["candidate_id"], [])), first_deactivation_h=dg.get("first_deactivation_h"),
                 max_current_fallback_zero_frac=dg.get("max_current_fallback_zero_frac"),
                 max_wind_fallback_zero_frac=dg.get("max_wind_fallback_zero_frac"), runtime_s=cfg["runtime_s"],
                 run_status=cfg["status"], physics_valid_for_attribution=dg.get("physics_valid_for_attribution", False))
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "candidate_hindcast_summary.csv", index=False)
    # Pairwise origin-distribution distances: exploratory only. No threshold, no fusion, no clustering.
    from scipy.spatial.distance import cdist
    ids = list(clouds)
    self_e = {i: cdist(clouds[i], clouds[i]).mean() for i in ids}
    edges = np.arange(-600, 600.001, 2.0)  # 2 km grid in the common LAEA frame
    hist = {i: np.histogram2d(clouds[i][:, 0], clouds[i][:, 1], bins=[edges, edges])[0] / len(clouds[i]) for i in ids}
    med = {i: np.median(clouds[i], axis=0) for i in ids}
    P = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            ra, rb = S.set_index("candidate_id").loc[i], S.set_index("candidate_id").loc[j]
            cross = cdist(clouds[i], clouds[j]).mean()
            P.append(dict(candidate_a=i, candidate_b=j,
                          origin_mean_distance_km=GEOD.inv(ra.origin_mean_lon, ra.origin_mean_lat, rb.origin_mean_lon, rb.origin_mean_lat)[2] / 1000,
                          origin_median_distance_km=GEOD.inv(ra.origin_median_lon, ra.origin_median_lat, rb.origin_median_lon, rb.origin_median_lat)[2] / 1000,
                          origin_median_distance_laea_km=float(np.hypot(*(med[i] - med[j]))),
                          energy_distance_km=float(max(2 * cross - self_e[i] - self_e[j], 0.0)),
                          histogram_overlap_2km=float(np.minimum(hist[i], hist[j]).sum()),
                          seed_centroid_distance_km=GEOD.inv(ra.seed_centroid_lon, ra.seed_centroid_lat, rb.seed_centroid_lon, rb.seed_centroid_lat)[2] / 1000,
                          both_physics_valid=bool(ra.physics_valid_for_attribution and rb.physics_valid_for_attribution)))
    PW = pd.DataFrame(P)
    PW.attrs["note"] = "descriptive only"
    PW.to_parquet(OUT / "candidate_origin_pairwise.parquet", index=False)
    dump(dict(note="EXPLORATORY ONLY. Pairwise distances between deterministic -48 h origin clouds. No threshold, no fusion, no "
                   "clustering; close origins do not show a common release.",
              laea_center=[lon0, lat0], energy_distance="2E|X-Y| - E|X-X'| - E|Y-Y'| (km, LAEA)",
              histogram_overlap="sum over 2 km cells of min(p_a, p_b); 1 = identical binned distributions",
              n_pairs=len(PW)), OUT / "candidate_origin_pairwise_README.json")
    print(S.round(3).to_string(index=False))
    print(PW.describe().round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["preflight", "run", "rediagnose", "summary"])
    ap.add_argument("--candidate")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.stage == "run":
        assert bool(a.candidate) != bool(a.all), "run needs exactly one of --candidate / --all"
    {"preflight": lambda: preflight(), "run": lambda: run(a), "rediagnose": lambda: rediagnose(),
     "summary": lambda: summary()}[a.stage]()
