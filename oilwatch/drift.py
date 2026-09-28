"""OpenDrift wrappers.

frozen_*  : exact re-implementations of run_hindcast_48h.py / run_hindcast_ensemble.py (same RNG call order,
            same sampling loop, same config) so the parity stage reproduces the frozen particles bit-exactly.
simulate  : v3 engine used by the new stages. One OpenDrift run carries many elements with per-element
            windage and per-element release time, so an ensemble over windage and many release hypotheses
            (clusters, vessels, release hours) costs one simulation per (diffusivity, current-uncertainty) pair.
"""
import logging
from datetime import timedelta
from itertools import product

import numpy as np
import pandas as pd
from shapely import contains_xy
from shapely.geometry import Point

logging.getLogger("opendrift").setLevel(logging.ERROR)

TRANSPORT_ONLY = {  # weathering is not reversible; keep backward and forward physics symmetric
    "processes:evaporation": False, "processes:emulsification": False, "processes:dispersion": False,
    "processes:biodegradation": False, "drift:vertical_mixing": False, "drift:vertical_advection": False,
}


def readers(ocean_nc, wind_nc):
    from opendrift.readers import reader_netCDF_CF_generic
    return [reader_netCDF_CF_generic.Reader(str(ocean_nc)), reader_netCDF_CF_generic.Reader(str(wind_nc))]


# ------------------------------------------------------------------ frozen re-implementations (parity)
def _frozen_sample(geom, n, rng, try_factor):
    minx, miny, maxx, maxy = geom.bounds
    xs_out, ys_out = [], []
    while len(xs_out) < n:
        n_try = max(1000, (n - len(xs_out)) * try_factor)
        xs, ys = rng.uniform(minx, maxx, n_try), rng.uniform(miny, maxy, n_try)
        for x, y in zip(xs, ys):
            if geom.contains(Point(x, y)):
                xs_out.append(x)
                ys_out.append(y)
                if len(xs_out) == n:
                    break
    return np.asarray(xs_out), np.asarray(ys_out)


def _frozen_model(rdr, loglevel, current_unc, wind_unc):
    from opendrift.models.openoil import OpenOil
    o = OpenOil(loglevel=loglevel)
    o.add_reader(rdr)
    for k, v in TRANSPORT_ONLY.items():
        o.set_config(k, v)
    o.set_config("drift:current_uncertainty", current_unc)
    o.set_config("drift:wind_uncertainty", wind_unc)
    o.set_config("environment:fallback:land_binary_mask", 0)
    return o


def frozen_deterministic(geom, t0, rdr, outfile, p):
    """run_hindcast_48h.py: 2000 particles, windage 0.03, no uncertainty, 48 h back, dt 15 min, hourly output."""
    rng = np.random.default_rng(p["rng_seed"])
    lons, lats = _frozen_sample(geom, p["n_particles"], rng, try_factor=3)
    o = _frozen_model(rdr, 50, 0, 0)
    o.seed_elements(lon=lons, lat=lats, time=t0, number=p["n_particles"], wind_drift_factor=p["windage"])
    o.run(duration=timedelta(hours=p["duration_h"]), time_step=-p["dt_s"], time_step_output=p["output_s"],
          outfile=str(outfile))


def frozen_ensemble(geom, t0, rdr, p):
    """run_hindcast_ensemble.py: 27 scenarios (windage x current unc x wind unc), 400 particles each."""
    parts, recs = [], []
    for sid, (w, cu, wu) in enumerate(product(p["windage"], p["current_uncertainty"], p["wind_uncertainty"]), start=1):
        seed = p["base_seed"] + sid * 1000
        np.random.seed(seed)  # OpenDrift's perturbations draw from the global RNG
        lons, lats = _frozen_sample(geom, p["particles_per_scenario"], np.random.default_rng(seed), try_factor=4)
        o = _frozen_model(rdr, 50, cu, wu)
        o.seed_elements(lon=lons, lat=lats, time=t0, number=p["particles_per_scenario"], wind_drift_factor=w)
        res = o.run(duration=timedelta(hours=p["duration_h"]), time_step=-p["dt_s"], time_step_output=p["output_s"])
        lon, lat = res["lon"].values, res["lat"].values
        times = pd.to_datetime(res["time"].values, utc=True)
        ntraj, ntime = lon.shape
        df = pd.DataFrame({"scenario_id": np.repeat(sid, ntraj * ntime), "trajectory": np.repeat(np.arange(ntraj), ntime),
                           "time": np.tile(times, ntraj), "lon": lon.reshape(-1), "lat": lat.reshape(-1),
                           "windage": np.repeat(w, ntraj * ntime), "current_uncertainty": np.repeat(cu, ntraj * ntime),
                           "wind_uncertainty": np.repeat(wu, ntraj * ntime)})
        parts.append(df[np.isfinite(df.lon) & np.isfinite(df.lat)].copy())
        recs.append({"scenario_id": sid, "seed": seed, "windage": w, "current_uncertainty": cu, "wind_uncertainty": wu,
                     "particles": ntraj, "times": ntime, "valid_rows": len(parts[-1])})
        print(f"  frozen scenario {sid:02d}/27 done", flush=True)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(recs)


# ------------------------------------------------------------------ v3 engine
def sample_area(geom, n, rng):
    """Uniform-in-area sample inside a (Multi)Polygon, vectorised rejection sampling."""
    minx, miny, maxx, maxy = geom.bounds
    xs, ys = np.empty(0), np.empty(0)
    while len(xs) < n:
        m = max(4 * n, 2000)
        x, y = rng.uniform(minx, maxx, m), rng.uniform(miny, maxy, m)
        ok = contains_xy(geom, x, y)
        xs, ys = np.r_[xs, x[ok]], np.r_[ys, y[ok]]
    return xs[:n], ys[:n]


def v3_members(p):
    """Simulations = diffusivity x current uncertainty; windage is folded into each simulation per element."""
    return [dict(sim_id=i, horizontal_diffusivity=k, current_uncertainty=cu)
            for i, (k, cu) in enumerate(product(p["horizontal_diffusivity_m2s"], p["current_uncertainty"]), start=1)]


def simulate(elements, duration_h, backward, rdr, sim, p, seed):
    """Run one OpenDrift simulation.

    elements: DataFrame with lon, lat, time (tz-aware), windage (+ any tag columns, kept by row order).
    Returns (lon[n_elem, n_out], lat[...], output times) with rows in `elements` order.
    """
    from opendrift.models.oceandrift import OceanDrift
    np.random.seed(seed)
    o = OceanDrift(loglevel=50)
    o.add_reader(rdr)
    o.set_config("drift:vertical_mixing", False)
    o.set_config("drift:vertical_advection", False)
    o.set_config("drift:stokes_drift", p["stokes_drift"])
    o.set_config("drift:current_uncertainty", sim["current_uncertainty"])
    o.set_config("drift:wind_uncertainty", p["wind_uncertainty"])
    o.set_config("environment:constant:horizontal_diffusivity", sim["horizontal_diffusivity"])
    o.set_config("general:coastline_action", "stranding")
    times = [t.to_pydatetime() for t in pd.DatetimeIndex(elements["time"]).tz_convert(None)]
    kw = dict(lon=elements["lon"].to_numpy(float), lat=elements["lat"].to_numpy(float),
              wind_drift_factor=elements["windage"].to_numpy(float), z=0.0)
    if len(set(times)) == 1:
        o.seed_elements(time=times[0], number=len(elements), **kw)
    else:  # >2 distinct times -> OpenDrift seeds a time series, one time per element
        assert len(times) > 2
        o.seed_elements(time=times, number=len(elements), **kw)
    step = -p["dt_s"] if backward else p["dt_s"]
    res = o.run(time_step=step, time_step_output=p["output_s"], duration=timedelta(hours=duration_h))
    lon, lat = res["lon"].values, res["lat"].values
    t_out = pd.to_datetime(res["time"].values, utc=True)
    order = np.argsort(t_out.values)
    lon, lat, t_out = lon[:, order], lat[:, order], t_out[order]
    perm = _element_order(elements, lon, lat, t_out)
    return lon[perm], lat[perm], t_out


def _element_order(elements, lon, lat, t_out, tol=2e-5):  # OpenDrift stores float32 (~2e-6 deg)
    """Rows of the OpenDrift result -> rows of `elements`. OpenDrift 1.14 returns BACKWARD runs in reversed element order
    (forward keeps seeding order) and exposes no element ID, so the order is recovered from the data: every element
    released exactly at an output time sits at its seed position there. Raises unless the match is a clean permutation."""
    from scipy.spatial import cKDTree
    et = pd.DatetimeIndex(elements["time"])
    ex, ey = elements["lon"].to_numpy(float), elements["lat"].to_numpy(float)
    col = {t: k for k, t in enumerate(t_out)}
    # test the two orders OpenDrift produces (identity forward, reversed backward) on EVERY element released at an
    # output time; nearest-neighbour matching is ambiguous when seeds lie closer than float32 precision (~0.4 m)
    on = np.array([t in col for t in et])
    if on.any():
        ks = np.array([col[t] for t in et[on]])
        for cand in (np.arange(len(ex)), np.arange(len(ex))[::-1]):
            d = np.hypot(lon[cand[on], ks] - ex[on], lat[cand[on], ks] - ey[on])
            if np.all(np.isfinite(d)) and d.max() <= tol:
                return cand
    on_grid = np.array([t in col for t in et])
    perm = np.full(len(elements), -1)
    rows_left = np.ones(len(elements), bool)
    for t in pd.unique(et[on_grid]):
        idx = np.where(on_grid & (et == t))[0]
        k = col[t]
        cand = np.where(rows_left & np.isfinite(lon[:, k]))[0]
        tree = cKDTree(np.c_[lon[cand, k], lat[cand, k]])
        d, j = tree.query(np.c_[ex[idx], ey[idx]])
        if d.max() > tol or len(set(j)) != len(j):
            raise RuntimeError(f"cannot match OpenDrift rows to elements at {t} (max d {d.max():.2e})")
        perm[idx] = cand[j]
        rows_left[cand[j]] = False
    if (~on_grid).any():  # off-grid releases (forward time series): OpenDrift keeps seeding order among them
        rest_el, rest_rows = np.where(~on_grid)[0], np.where(rows_left)[0]
        if len(rest_el) != len(rest_rows):
            raise RuntimeError("element/row count mismatch")
        # on-grid elements matched their own rows in seeding order -> order is preserved; map the rest in order
        if on_grid.any() and not np.all(np.diff(perm[on_grid][np.argsort(np.where(on_grid)[0])]) > 0):
            raise RuntimeError("seeding order not preserved; off-grid elements cannot be mapped safely")
        perm[rest_el] = rest_rows
    assert (perm >= 0).all() and len(set(perm)) == len(perm)
    return perm
