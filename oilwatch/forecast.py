"""Forward forecast (PS clause b: 'predict the future flow of the slick'), T0 -> T0 + 72 h.

Unlike the attribution drift, weathering is physically valid forward in time, so this uses OpenOil WITH
evaporation / emulsification / dispersion / vertical mixing, the GSHHG coastline (stranding = beaching) and
the same v3 transport ensemble (diffusivity x current uncertainty, windage per element).
Oil type is unknown from SAR -> central members use GENERIC MEDIUM CRUDE; light and heavy crude are run on the
central physics member as a sensitivity. Sea temperature is an assumption (no SST file in the forcing).

Outputs: hourly particles, probability-of-presence grids at +24/48/72 h, beaching probability and earliest arrival,
mass budget (surface / evaporated / dispersed / stranded) per oil type.
"""
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from . import drift

logging.getLogger("opendrift").setLevel(logging.ERROR)


def _model(rdr, sim, P, oil):
    from opendrift.models.openoil import OpenOil
    o = OpenOil(loglevel=50, weathering_model="noaa")
    o.add_reader(rdr)
    for k, v in {"processes:evaporation": True, "processes:emulsification": True, "processes:dispersion": True,
                 "drift:vertical_mixing": True, "drift:stokes_drift": True,
                 "drift:current_uncertainty": sim["current_uncertainty"], "drift:wind_uncertainty": P["wind_uncertainty"],
                 "environment:constant:horizontal_diffusivity": sim["horizontal_diffusivity"],
                 "environment:constant:sea_water_temperature": P["sea_water_temperature_c"],
                 "environment:constant:sea_water_salinity": P["sea_water_salinity_psu"],
                 "general:coastline_action": "stranding"}.items():
        o.set_config(k, v)
    return o


def run(cands, t0, rdr, P, F):
    rng = np.random.default_rng(P["seed"] + 99)
    base = []
    for _, c in cands.iterrows():
        for w in P["windage"]:
            x, y = drift.sample_area(c.geometry, F["particles_per_windage"], rng)
            base.append(pd.DataFrame({"cluster_id": c.cluster_id, "windage": w, "lon": x, "lat": y}))
    base = pd.concat(base, ignore_index=True)
    sims = [(s, F["oil_central"]) for s in drift.v3_members(P)]
    central = [s for s in drift.v3_members(P) if s["horizontal_diffusivity"] == F["central_member"]["horizontal_diffusivity"]
               and s["current_uncertainty"] == F["central_member"]["current_uncertainty"]][0]
    sims += [(central, o) for o in F["oil_sensitivity"]]
    out = []
    for k, (sim, oil) in enumerate(sims, start=1):
        np.random.seed(P["seed"] + 2000 + k)
        o = _model(rdr, sim, {**P, **F}, oil)
        o.seed_elements(lon=base.lon.values, lat=base.lat.values, time=t0.tz_convert(None).to_pydatetime(), z=0.0,
                        number=len(base), wind_drift_factor=base.windage.values, oil_type=oil,
                        m3_per_hour=F["m3_total"] / 1.0)
        res = o.run(duration=timedelta(hours=F["hours"]), time_step=P["dt_s"], time_step_output=P["output_s"])
        t = pd.to_datetime(res["time"].values, utc=True)
        o_ = np.argsort(t.values)
        t = t[o_]
        perm = drift._element_order(base.assign(time=t0), res["lon"].values[:, o_], res["lat"].values[:, o_], t)
        n, T = res["lon"].shape
        d = {v: res[v].values[:, o_][perm].reshape(-1) for v in ("lon", "lat", "z", "mass_oil", "mass_evaporated", "mass_dispersed")
             if v in res}
        st = res["status"].values[:, o_][perm].reshape(-1)
        flags = res["status"].attrs.get("flag_meanings", "").split()
        df = pd.DataFrame({"run": k, "sim_id": sim["sim_id"], "oil_type": oil, "cluster_id": np.repeat(base.cluster_id.values, T),
                           "windage": np.repeat(base.windage.values, T), "element": np.repeat(np.arange(n), T),
                           "time": np.tile(t, n), **d})
        df["status"] = [flags[int(s)] if np.isfinite(s) and int(s) < len(flags) else "gone" for s in st]
        df["lead_h"] = (df.time - t0).dt.total_seconds() / 3600
        out.append(df.dropna(subset=["lon"]))
        print(f"  forecast run {k}/{len(sims)}: {oil}, K={sim['horizontal_diffusivity']}, cu={sim['current_uncertainty']}", flush=True)
    return pd.concat(out, ignore_index=True)


def probability_grid(fp, cluster_id, lead_h, oil, cell_deg):
    """Fraction of ensemble particles (surface + submerged, not stranded) per cell at a lead time."""
    g = fp[(fp.cluster_id == cluster_id) & (fp.lead_h == lead_h) & (fp.oil_type == oil) & (fp.status != "stranded")]
    if g.empty:
        return pd.DataFrame()
    ix, iy = np.floor(g.lon / cell_deg).astype(int), np.floor(g.lat / cell_deg).astype(int)
    c = pd.DataFrame({"ix": ix, "iy": iy}).value_counts().rename("n").reset_index()
    c["probability"] = c.n / len(g)
    c["lon"], c["lat"] = (c.ix + .5) * cell_deg, (c.iy + .5) * cell_deg
    return c.assign(cluster_id=cluster_id, lead_h=lead_h)


def beaching(fp, oil):
    rows = []
    for cid, g in fp[fp.oil_type == oil].groupby("cluster_id"):
        per_el = g.groupby(["run", "element"]).apply(lambda x: x.lead_h[x.status == "stranded"].min(), include_groups=False)
        rows.append({"cluster_id": cid, "p_beached_72h": float(per_el.notna().mean()),
                     "earliest_beaching_h": float(per_el.min()) if per_el.notna().any() else np.nan,
                     "median_beaching_h_if_beached": float(per_el.median()) if per_el.notna().any() else np.nan})
    return pd.DataFrame(rows)


def mass_budget(fp):
    """Share of initial oil mass by fate, per oil type (central physics member, all clusters pooled)."""
    g = fp.copy()
    tot = g.groupby(["run", "oil_type", "lead_h"])[["mass_oil", "mass_evaporated", "mass_dispersed"]].sum().reset_index()
    tot["initial"] = tot[["mass_oil", "mass_evaporated", "mass_dispersed"]].sum(axis=1).groupby([tot.run]).transform("first")
    for c in ("mass_oil", "mass_evaporated", "mass_dispersed"):
        tot[c.replace("mass_", "frac_")] = tot[c] / tot.initial
    return tot


def centroid_track(fp, oil):
    g = fp[(fp.oil_type == oil) & (fp.status != "stranded")]
    return g.groupby(["cluster_id", "lead_h"]).agg(lon=("lon", "mean"), lat=("lat", "mean"),
                                                   lon_p10=("lon", lambda x: np.quantile(x, .1)), lon_p90=("lon", lambda x: np.quantile(x, .9)),
                                                   lat_p10=("lat", lambda x: np.quantile(x, .1)), lat_p90=("lat", lambda x: np.quantile(x, .9))).reset_index()
