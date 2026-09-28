"""Forward model (v3): 'what if this vessel discharged?' (Longepe et al. 2015; forward half of Luo et al. 2024).

Every funnel vessel releases virtual oil along its own interpolated AIS track every 15 min over T0-48 h..T0
(jittered 50 m, 3 windages x 2 particles). One forward simulation per ensemble simulation carries all vessels.
At T0 the virtual oil is compared with each observed slick event:

  window search  (all contiguous release windows of 1/2/3/6/12 h ending 0..47 h before T0):
      precision = share of the window's virtual oil within d_hit of the slick
      recall    = share of slick sample points with virtual oil from that window within d_hit
      F         = 2 P R / (P + R), 27 members pooled (probability of presence) -> best window = implied release time
                  member support = share of members with virtual oil on the slick in that window
      (bitmask trick: each slick point stores a 48-bit mask of release hours that reached it)
  skill          Fractions Skill Score of the best window (members pooled) at 0.5/1/2/5 km;
                 useful if FSS >= 0.5 + f_O/2 (Simecek-Beatty & Lehr 2021).
A stationary AIS track (platform, drillship) is tested the same way, as a point source.
"""
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree
from shapely import distance as sh_distance, points as sh_points

from . import drift
from .ais import ns, positions_v2_1
from .geo import laea, to_metric

N_HOURS = 48


def release_elements(ais, mmsis, t0, F, cfg21, rng):
    """Virtual discharge elements along each vessel's AIS track."""
    return jitter_elements(release_points(ais, mmsis, t0, F, cfg21), F, rng)


def release_points(ais, mmsis, t0, F, cfg21):
    """Vessel positions every release_step_min over T0-48 h..T0 (frozen V2.1 rule), off land."""
    step = pd.Timedelta(minutes=F["release_step_min"])
    rel_t = pd.DatetimeIndex([t0 - k * step for k in range(1, int(48 * 60 / F["release_step_min"]) + 1)])[::-1]
    t_model = rel_t.as_unit("ns").asi8
    rows = []
    for m, g in ais[ais.MMSI.isin(set(mmsis))].groupby("MMSI"):
        g = g.sort_values("BaseDateTime").drop_duplicates("BaseDateTime")
        lon, lat, meth, _, _ = positions_v2_1(t_model, ns(g.BaseDateTime), g.LON.values, g.LAT.values, cfg21)
        ok = meth != "unavailable"
        if not ok.any():
            continue
        rows.append(pd.DataFrame({"MMSI": m, "time": rel_t[ok], "lon0": lon[ok], "lat0": lat[ok]}))
    base = pd.concat(rows, ignore_index=True)
    # OpenDrift silently drops elements seeded on land (AIS in ports/rivers), which would break element identity:
    # remove them here with the same GSHHG land mask OpenDrift uses
    from opendrift.readers.reader_global_landmask import Reader as Landmask
    on_land = np.asarray(Landmask()._on_land(base.lon0.values, base.lat0.values), bool)
    base = base[~on_land].reset_index(drop=True)
    base.attrs["dropped_on_land"] = int(on_land.sum())
    base["tau_release_h"] = (t0 - base.time).dt.total_seconds() / 3600
    base["hour"] = np.clip(np.ceil(base.tau_release_h - 1e-9).astype(int) - 1, 0, N_HOURS - 1)  # (h, h+1] -> h
    return base


def jitter_elements(base, F, rng):
    """k particles per release point and windage, jittered by release_radius_m, re-checked against land."""
    from opendrift.readers.reader_global_landmask import Reader as Landmask
    els = []
    k = F["particles_per_release_per_windage"]
    for w in F["windage"]:
        e = base.loc[base.index.repeat(k)].reset_index(drop=True)
        r_m = F["release_radius_m"]
        e["lon"] = e.lon0 + rng.normal(0, r_m, len(e)) / (111320 * np.cos(np.radians(e.lat0)))
        e["lat"] = e.lat0 + rng.normal(0, r_m, len(e)) / 110540
        e["windage"] = w
        els.append(e)
    els = pd.concat(els, ignore_index=True)
    jitter_land = np.asarray(Landmask()._on_land(els.lon.values, els.lat.values), bool)  # jitter can cross a coastline
    return els[~jitter_land].reset_index(drop=True)


def run(els, t0, rdr, P):
    """Forward from T0-48 h to T0; returns final positions per element for every simulation."""
    out = []
    for sim in drift.v3_members(P):
        lon, lat, t = drift.simulate(els, P["duration_h"], False, rdr, sim, P, seed=P["seed"] + 500 + 1000 * sim["sim_id"])
        assert abs((t[-1] - t0).total_seconds()) < 1, (t[-1], t0)
        out.append(pd.DataFrame({"sim_id": sim["sim_id"], "MMSI": els.MMSI.values, "windage": els.windage.values,
                                 "hour": els.hour.values, "tau_release_h": els.tau_release_h.values,
                                 "lon_t0": lon[:, -1], "lat_t0": lat[:, -1]}))
        print(f"  forward sim {sim['sim_id']}/9: {len(els)} virtual-oil elements", flush=True)
    f = pd.concat(out, ignore_index=True)
    f["member"] = f.sim_id.astype(str) + "_w" + f.windage.map("{:.2f}".format)
    return f


def windows(durations):
    return [(a, d) for d in durations for a in range(0, N_HOURS - d + 1)]  # hours [a, a+d) before T0


def fss(obs, mod, n_cells):
    po, pm = uniform_filter(obs.astype(float), n_cells, mode="constant"), uniform_filter(mod.astype(float), n_cells, mode="constant")
    den = (po ** 2).sum() + (pm ** 2).sum()
    return float(1 - ((po - pm) ** 2).sum() / den) if den > 0 else np.nan


def evaluate_cluster(cl, fpos, vessels, F, rng):
    """Window search + FSS for one slick event against its funnel vessels."""
    fwd, back = laea(cl.centroid_lon, cl.centroid_lat)
    geom_m = to_metric(cl.geometry, fwd)
    d_hit = F["fss_primary_km"] * 1000
    sx, sy = drift.sample_area(geom_m, 400, rng)
    wins = windows(F["window_durations_h"])
    win_mask = np.array([sum(1 << h for h in range(a, a + d)) for a, d in wins], dtype=np.int64)
    win_hours = [(a, a + d) for a, d in wins]
    res, best_rows = [], []
    for m in vessels:
        pv = fpos[(fpos.MMSI == m) & np.isfinite(fpos.lon_t0)]
        if pv.empty:
            continue
        x, y = fwd.transform(pv.lon_t0.values, pv.lat_t0.values)
        near = sh_distance(geom_m, sh_points(np.c_[x, y])) <= d_hit
        pv = pv.assign(x=x, y=y, near=near)
        # Members pooled = ensemble probability-of-presence of the virtual slick. Per member a 1 h window holds only
        # ~8 particles, so member-wise medians are 0 by sparsity; member agreement is reported separately.
        cn = np.r_[0, np.cumsum(np.bincount(pv.hour, minlength=N_HOURS))]
        cnear = np.r_[0, np.cumsum(np.bincount(pv.hour[pv.near], minlength=N_HOURS))]
        hrs = pv.hour.values
        masks = np.array([np.bitwise_or.reduce(np.left_shift(1, hrs[idx]).astype(np.int64)) if idx else 0
                          for idx in cKDTree(np.c_[pv.x, pv.y]).query_ball_point(np.c_[sx, sy], d_hit)], dtype=np.int64)
        Pm = np.array([(cnear[b] - cnear[a]) / (cn[b] - cn[a]) if cn[b] > cn[a] else 0.0 for a, b in win_hours])
        Rm = ((masks[None, :] & win_mask[:, None]) != 0).mean(axis=1)
        Fm = np.where(Pm + Rm > 0, 2 * Pm * Rm / np.maximum(Pm + Rm, 1e-12), 0.0)
        i = int(np.argmax(Fm))
        a, b = win_hours[i]
        sel = pv[(pv.hour >= a) & (pv.hour < b)]
        support = sel.groupby("member").near.any().reindex(pv.member.unique(), fill_value=False).mean()
        fs = fss_scores(geom_m, sel.x.values, sel.y.values, F)
        best_rows.append({"cluster_id": cl.cluster_id, "MMSI": m, "fwd_F": float(Fm[i]), "fwd_precision": float(Pm[i]),
                          "fwd_recall": float(Rm[i]), "fwd_window_start_h": a, "fwd_window_end_h": b,
                          "fwd_member_support": float(support), "fwd_particles_in_window": len(sel), **fs})
        res.append(pd.DataFrame({"cluster_id": cl.cluster_id, "MMSI": m, "window_start_h": [w[0] for w in win_hours],
                                 "window_end_h": [w[1] for w in win_hours], "F_pooled": Fm, "precision_pooled": Pm,
                                 "recall_pooled": Rm}))
    return pd.DataFrame(best_rows), (pd.concat(res, ignore_index=True) if res else pd.DataFrame())


def fss_scores(geom_m, x, y, F):
    g = F["grid_m"]
    pad = max(F["fss_scales_km"]) * 1000 + 2000
    minx, miny, maxx, maxy = geom_m.bounds
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
    W, H = int(np.ceil((maxx - minx) / g)), int(np.ceil((maxy - miny) / g))
    tr = from_origin(minx, maxy, g, g)
    obs = rasterize([(geom_m, 1)], out_shape=(H, W), transform=tr, all_touched=True).astype(bool)
    mod = np.zeros((H, W), bool)
    ci, ri = ((x - minx) / g).astype(int), ((maxy - y) / g).astype(int)
    ok = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    mod[ri[ok], ci[ok]] = True
    f_o = obs.mean()
    out = {"fss_domain_obs_fraction": float(f_o), "fss_useful_threshold": float(0.5 + f_o / 2),
           "fwd_mass_in_domain": float(ok.mean()) if len(ok) else 0.0}
    for s in F["fss_scales_km"]:
        n = max(1, int(round(s * 1000 / g)) | 1)  # odd neighbourhood width in cells
        out[f"fss_{s:g}km"] = fss(obs, mod, n)
    out["fss_skilful"] = bool(out[f"fss_{F['fss_primary_km']:g}km"] >= out["fss_useful_threshold"])
    return out
