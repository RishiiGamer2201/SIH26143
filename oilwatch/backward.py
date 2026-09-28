"""Backward model (v3): every selected slick event drifted 48 h back in one ensemble, then AIS compatibility.

Ensemble member = (simulation, windage); simulation = (horizontal diffusivity, current uncertainty) -> 3 x 3 x 3 = 27.
Score per member, per vessel, per release age tau (hourly):
    density(tau) = mean_i exp(-d_i^2 / 2 sigma(tau)^2),  sigma(tau)^2 = sigma0^2 + (u_err * tau)^2
    S_point      = max_tau density                       (instantaneous release; frozen V2 analogue)
    S_weighted   = sum_tau w(tau) density(tau) / sum_tau w(tau),  w = exp(-tau / tau0)  (persistence prior;
                   hours without an AIS position count as 0 - the vessel cannot be placed there)
Vessel positions use the frozen V2.1 rule (interpolated AIS, E101).
Funnel (PS: 'irrelevant traffic filtered out'): AIS in window -> positioned >= min_positions hours ->
within gate_sigma * sigma(tau) of some particle in some member. Each drop is counted.
"""
import numpy as np
import pandas as pd

from . import drift
from .ais import ns, positions_v2_1, vessel_name
from .geo import haversine_km


def hourly_positions(ais, times, cfg21):
    """V2.1 positions of every vessel at the model times -> dict mmsi -> (name, lon[T], lat[T], method[T])."""
    margin = pd.Timedelta(minutes=max(cfg21["max_bracket_gap_min"], cfg21["edge_nearest_tol_min"]))
    a = ais[(ais.BaseDateTime >= times[0] - margin) & (ais.BaseDateTime <= times[-1] + margin)]
    a = a[a.LAT.between(-90, 90) & a.LON.between(-180, 180)]
    out, t_model = {}, times.as_unit("ns").asi8
    for mmsi, g in a.groupby("MMSI"):
        g = g.sort_values("BaseDateTime").drop_duplicates("BaseDateTime")
        lon, lat, m, _, _ = positions_v2_1(t_model, ns(g.BaseDateTime), g.LON.values, g.LAT.values, cfg21)
        out[mmsi] = (vessel_name(g), lon, lat, m)
    return out


def run_ensemble(cands, t0, rdr, P):
    """One backward simulation per (K, current_unc); every selected cluster x windage x N particles inside it."""
    rng = np.random.default_rng(P["seed"])
    els = []
    for _, c in cands.iterrows():
        for w in P["windage"]:
            x, y = drift.sample_area(c.geometry, P["backward_particles_per_windage"], rng)
            els.append(pd.DataFrame({"cluster_id": c.cluster_id, "windage": w, "lon": x, "lat": y, "time": t0}))
    els = pd.concat(els, ignore_index=True)
    parts = []
    for sim in drift.v3_members(P):
        lon, lat, t = drift.simulate(els, P["duration_h"], True, rdr, sim, P, seed=P["seed"] + 1000 * sim["sim_id"])
        n, T = lon.shape
        parts.append(pd.DataFrame({"sim_id": sim["sim_id"], "cluster_id": np.repeat(els.cluster_id.values, T),
                                   "windage": np.repeat(els.windage.values, T), "element": np.repeat(np.arange(n), T),
                                   "time": np.tile(t, n), "lon": lon.reshape(-1).astype("float32"),
                                   "lat": lat.reshape(-1).astype("float32")}))
        print(f"  backward sim {sim['sim_id']}/9 (K={sim['horizontal_diffusivity']}, cu={sim['current_uncertainty']}): "
              f"{n} elements", flush=True)
    parts = pd.concat(parts, ignore_index=True)
    parts["member"] = parts.sim_id.astype(str) + "_w" + parts.windage.map("{:.2f}".format)
    return parts, els


CUTOFF_SIGMA = 6.0


def positioned_hours(pos):
    return {m: int(np.isfinite(v[1]).sum()) for m, v in pos.items()}


def sigma_km(tau_h, B):
    return np.sqrt(B["sigma0_km"] ** 2 + (B["u_err_m_s"] * 3.6 * tau_h) ** 2)


def score(parts, pos, t0, B):
    """Per (cluster, member, vessel, tau): density + distances. Vectorised over vessels per cloud."""
    times = pd.DatetimeIndex(sorted(parts.time.unique()))
    tau = ((t0 - times).total_seconds() / 3600).to_numpy()
    mmsi = np.array(list(pos))
    V_lon = np.stack([pos[m][1] for m in mmsi])  # [V, T]
    V_lat = np.stack([pos[m][2] for m in mmsi])
    recs = []
    tidx = {t: i for i, t in enumerate(times)}
    p = parts[np.isfinite(parts.lon) & np.isfinite(parts.lat)]
    for (cid, member, t), c in p.groupby(["cluster_id", "member", "time"], sort=False):
        ti = tidx[t]
        s = sigma_km(tau[ti], B)
        # only vessels within CUTOFF_SIGMA*sigma of the cloud's bounding box can contribute (> exp(-18) ~ 1.5e-8)
        pad = CUTOFF_SIGMA * s
        la0, la1 = c.lat.min() - pad / 110.54, c.lat.max() + pad / 110.54
        dl = pad / (111.32 * np.cos(np.radians(float(c.lat.mean()))))
        ok = np.isfinite(V_lon[:, ti]) & (V_lat[:, ti] >= la0) & (V_lat[:, ti] <= la1) &             (V_lon[:, ti] >= c.lon.min() - dl) & (V_lon[:, ti] <= c.lon.max() + dl)
        if not ok.any():
            continue
        d = haversine_km(V_lon[ok, ti][:, None], V_lat[ok, ti][:, None], c.lon.values[None, :].astype(float),
                         c.lat.values[None, :].astype(float))
        recs.append(pd.DataFrame({"cluster_id": cid, "member": member, "MMSI": mmsi[ok], "tau_h": tau[ti],
                                  "density": np.exp(-0.5 * (d / s) ** 2).mean(axis=1), "min_distance_km": d.min(axis=1),
                                  "gate_hit": d.min(axis=1) <= B["funnel_gate_sigma"] * s}))
    return pd.concat(recs, ignore_index=True)


def aggregate(ts, pos, B, age_bins, age_labels):
    """Member-level S_point / S_weighted, member ranks, robust per (cluster, vessel), funnel counts."""
    tau_grid = np.arange(0, 49)
    w = np.exp(-tau_grid / B["persistence_tau0_h"])
    rows = []
    for (cid, member, m), g in ts.groupby(["cluster_id", "member", "MMSI"], sort=False):
        dens = np.zeros(len(tau_grid))
        dens[g.tau_h.round().astype(int).values] = g.density.values
        pk = g.loc[g.density.idxmax()]
        rows.append({"cluster_id": cid, "member": member, "MMSI": m, "positions": len(g), "gate_hit": bool(g.gate_hit.any()),
                     "S_point": float(dens.max()), "S_weighted": float((w * dens).sum() / w.sum()),
                     "peak_age_hours": float(pk.tau_h), "peak_min_distance_km": float(pk.min_distance_km)})
    mem = pd.DataFrame(rows)
    mem["rank_weighted"] = mem.groupby(["cluster_id", "member"]).S_weighted.rank(ascending=False, method="min").astype(int)
    mem["rank_point"] = mem.groupby(["cluster_id", "member"]).S_point.rank(ascending=False, method="min").astype(int)
    rob = mem.groupby(["cluster_id", "MMSI"], as_index=False).agg(
        positions=("positions", "max"), gate_hit=("gate_hit", "any"), members=("member", "nunique"),
        bwd_weighted=("S_weighted", "median"), bwd_weighted_q10=("S_weighted", lambda x: np.quantile(x, .1)),
        bwd_weighted_q90=("S_weighted", lambda x: np.quantile(x, .9)), bwd_point=("S_point", "median"),
        bwd_rank_median=("rank_weighted", "median"), bwd_point_rank_median=("rank_point", "median"),
        bwd_top1_fraction=("rank_weighted", lambda r: float((r <= 1).mean())),
        bwd_top3_fraction=("rank_weighted", lambda r: float((r <= 3).mean())),
        bwd_top5_fraction=("rank_weighted", lambda r: float((r <= 5).mean())),
        bwd_peak_age_median=("peak_age_hours", "median"), bwd_peak_min_distance_km=("peak_min_distance_km", "median"))
    rob["VesselName"] = rob.MMSI.map(lambda m: pos[m][0])
    rob["positions"] = rob.MMSI.map(positioned_hours(pos))  # hours with an AIS position (not only scored hours)
    rob["passes_funnel"] = rob.gate_hit & (rob.positions >= B["funnel_min_positions"])
    # per release-age bin (frozen bins) for the analyst table
    ts = ts.assign(release_age_bin=pd.cut(ts.tau_h, bins=age_bins, labels=age_labels, include_lowest=True).astype(str))
    pb = ts.groupby(["cluster_id", "member", "MMSI", "release_age_bin"], as_index=False).density.max()
    pb["rank"] = pb.groupby(["cluster_id", "member", "release_age_bin"]).density.rank(ascending=False, method="min")
    bins = pb.groupby(["cluster_id", "MMSI", "release_age_bin"], as_index=False).agg(
        density_median=("density", "median"), rank_median=("rank", "median"),
        top5_fraction=("rank", lambda r: float((r <= 5).mean())))
    return mem, rob, bins


def funnel(pos, rob, B):
    n_all = len(pos)
    n_pos = sum(v >= B["funnel_min_positions"] for v in positioned_hours(pos).values())
    rows = []
    for cid, g in rob.groupby("cluster_id"):
        rows.append({"cluster_id": cid, "ais_vessels_in_window": n_all, "positioned_ge_min_hours": int(n_pos),
                     "within_gate_of_backward_cloud": int(g.passes_funnel.sum())})
    return pd.DataFrame(rows)


def cloud_summary(parts, t0):
    """Per cluster / member / hour: centroid and spread (for maps and the drift theatre)."""
    p = parts[np.isfinite(parts.lon)]
    s = p.groupby(["cluster_id", "member", "time"]).agg(lon=("lon", "mean"), lat=("lat", "mean"),
                                                        lon_std=("lon", "std"), lat_std=("lat", "std"), n=("lon", "size")).reset_index()
    s["tau_h"] = (t0 - s.time).dt.total_seconds() / 3600
    return s
