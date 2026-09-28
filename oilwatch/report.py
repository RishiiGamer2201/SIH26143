"""Run report (summary.json, report.md, map) and the post-hoc reference comparison (posthoc.*).

Wording rule (PROJECT_CONTEXT.md): rankings are evidence of spatio-temporal consistency, never proof of culpability.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

from .config import ROOT

DISCLAIMER = ("Rankings are evidence of spatio-temporal consistency between AIS tracks and modelled oil drift, "
              "never proof of culpability. source_type is 'unknown' for every AIS track until vessel metadata and an "
              "offshore-infrastructure database are joined; motion_state is not evidence of type.")
SHOW = ["suspect_rank", "tier", "v31_decision", "VesselName", "MMSI", "combined", "bwd_weighted", "bwd_point", "bwd_rank_median",
        "bwd_top3_fraction", "bwd_peak_age_median", "fwd_F", "fwd_member_support", "fss_1km", "fss_skilful", "fwd_window_start_h",
        "fwd_window_end_h", "direction_agreement", "dist_to_slick_end_km", "geo_fresh_discharge", "motion_state", "ais_gaps_ge_60min",
        "flag_fresh_discharge_match", "flag_ais_gap_in_release_window", "flag_identity_anomaly", "flag_window_at_48h_boundary", "sar_label", "sar_dist_to_slick_km", "flag_ais_silent_seen_by_sar_near_slick"]


def _load(run):
    rd = lambda p, **k: pd.read_csv(run.root / p, dtype={"MMSI": str}, **k)
    cand = gpd.read_file(run.root / "seed/candidates.geojson")
    return (cand, rd("fuse/suspects.csv"), rd("fuse/age_summary.csv"), rd("backward/funnel.csv"),
            rd("context/context_t0.csv"), rd("fuse/age_cues.csv"))


def md_table(df, cols, fmt="{:.3g}"):
    cols = [c for c in cols if c in df]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def write_report(run, inc, P, out):
    cand, sus, ages, fn, ctx, cues = _load(run)
    sel = cand[cand.selected]
    events = []
    for _, c in sel.iterrows():
        s = sus[sus.cluster_id == c.cluster_id]
        a = ages[ages.cluster_id == c.cluster_id].iloc[0]
        x = ctx[ctx.cluster_id == c.cluster_id].iloc[0]
        f = fn[fn.cluster_id == c.cluster_id].iloc[0]
        events.append({"cluster_id": int(c.cluster_id), "potential_rank": int(c.potential_rank),
                       "area_km2": c.area_m2 / 1e6, "length_km": c.length_m / 1e3, "width_proxy_m": c.width_proxy_m,
                       "axis_bearing_deg": c.axis_bearing_deg, "confidence": c.confidence,
                       "centroid": [c.centroid_lon, c.centroid_lat], "wind_speed_m_s": x.wind_speed_m_s, "wind_from_deg": x.wind_from_deg,
                       "wind_class": c.wind_class, "current_speed_m_s": x.current_speed_m_s, "current_to_deg": x.current_to_deg,
                       "funnel": {k: int(v) for k, v in f.items() if k != "cluster_id"},
                       "age_estimate_h": [a.age_lo_h, a.age_hi_h], "age_basis": a.estimate_basis, "age_conflict": bool(a.age_conflict), "age_conflict_note": a.age_conflict_note if isinstance(a.age_conflict_note, str) else "",
                       "tiers": s.tier.value_counts().to_dict(),
                       "top_suspects": s.head(5)[[c_ for c_ in SHOW if c_ in s]].replace({np.nan: None}).to_dict("records")})
    summary = {"incident": inc.id, "run_id": run.run_id, "t0_utc": str(inc.t0), "disclaimer": DISCLAIMER,
               "n_detected_events": int(len(cand)), "n_attributed": int(len(sel)), "events": events}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    L = [f"# oilwatch run {run.run_id}: {inc.id}", "", f"T0 {inc.t0} · scene {inc.spec['sentinel1_scene_id']}", "",
         f"> {DISCLAIMER}", "", "## Slick events (E008 policy v1)", "",
         md_table(cand.drop(columns="geometry"), ["potential_rank", "cluster_id", "component_ids", "area_m2", "confidence",
                                                  "wind_speed_m_s", "wind_class", "slick_potential", "eligible", "selected",
                                                  "ineligible_reason"]), ""]
    for e in events:
        s = sus[sus.cluster_id == e["cluster_id"]]
        L += [f"## Event {e['cluster_id']}: {e['area_km2']:.2f} km², {e['length_km']:.1f} km long, wind {e['wind_speed_m_s']:.1f} m/s ({e['wind_class']})",
              "", f"Funnel: {e['funnel']}. Age estimate {e['age_estimate_h'][0]:.1f}–{e['age_estimate_h'][1]:.1f} h ({e['age_basis']}).", "",
              md_table(s.head(8), SHOW), "",
              md_table(cues[cues.cluster_id == e["cluster_id"]], ["cue", "lo_h", "hi_h", "detail"]), ""]
    figs = [make_map(run, inc, sel, sus, out / "map.png")]
    fdir = run.root / "forecast"
    if (fdir / "beaching.csv").exists():
        mb = pd.read_csv(fdir / "mass_budget.csv")
        fate = mb[mb.lead_h.isin(P["forecast_v1"]["lead_times_h"])].groupby(["oil_type", "lead_h"])[
            ["frac_oil", "frac_evaporated", "frac_dispersed"]].mean().reset_index()
        L += ["## Forecast T0 → T0+72 h (OpenOil with weathering, GSHHG coastline)", "",
              f"Sea temperature {P['forecast_v1']['sea_water_temperature_c']} °C is an ASSUMPTION (no SST in forcing). "
              "Oil type unknown from SAR: medium crude central, light/heavy as sensitivity.", "",
              md_table(pd.read_csv(fdir / "beaching.csv"), ["cluster_id", "p_beached_72h", "earliest_beaching_h"]), "",
              md_table(fate, ["oil_type", "lead_h", "frac_oil", "frac_evaporated", "frac_dispersed"]), ""]
        summary["forecast"] = {"beaching": pd.read_csv(fdir / "beaching.csv").replace({np.nan: None}).to_dict("records"),
                               "fate": fate.to_dict("records")}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        figs.append(forecast_map(run, inc, sel, P, out / "forecast.png"))
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    return [out / "summary.json", out / "report.md", *figs]


def forecast_map(run, inc, sel, P, path, n_events=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fp = pd.read_parquet(run.root / "forecast/forecast_particles.parquet",
                         columns=["cluster_id", "oil_type", "lead_h", "lon", "lat", "z"])
    fp = fp[fp.oil_type == P["forecast_v1"]["oil_central"]]
    ev = sel.sort_values("potential_rank").head(n_events)
    fig, axs = plt.subplots(1, len(ev), figsize=(5.2 * len(ev), 5.2), squeeze=False)
    cols = {24: "#2563eb", 48: "#d97706", 72: "#dc2626"}
    for ax, (_, c) in zip(axs[0], ev.iterrows()):
        g = fp[fp.cluster_id == c.cluster_id]
        for h, col in cols.items():
            x = g[g.lead_h == h]
            surf = x[x.z > -0.5]
            ax.scatter(x.lon, x.lat, s=1, color=col, alpha=.08)
            ax.scatter(surf.lon, surf.lat, s=2, color=col, alpha=.5, label=f"+{h} h (solid = still at surface)")
        tr = g.groupby("lead_h")[["lon", "lat"]].mean()
        ax.plot(tr.lon, tr.lat, color="k", lw=1)
        gpd.GeoSeries([c.geometry]).plot(ax=ax, color="black")
        ax.set_title(f"event {c.cluster_id}: forecast (medium crude, 27 members)", fontsize=9)
        ax.legend(fontsize=7, markerscale=4)
        ax.set_aspect(1 / np.cos(np.radians(c.centroid_lat)))
    fig.suptitle(f"{inc.id}: where the slick goes next (probability cloud), T0 + 24/48/72 h", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def make_map(run, inc, sel, sus, path, n_events=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = pd.read_parquet(run.root / "backward/cloud_summary.parquet")
    fpos = pd.read_parquet(run.root / "forward/virtual_oil_at_t0.parquet")
    # events with the strongest two-direction support first (reference-independent choice of what to draw)
    best = sus[sus.direction_agreement == "forward+backward"].groupby("cluster_id").combined.max()
    order = sel.assign(_b=sel.cluster_id.map(best).fillna(-1)).sort_values(["_b", "potential_rank"], ascending=[False, True])
    top_ev = order.head(n_events)
    fig, axs = plt.subplots(1, len(top_ev), figsize=(5.2 * len(top_ev), 5.4), squeeze=False)
    for ax, (_, c) in zip(axs[0], top_ev.iterrows()):
        m = cs[cs.cluster_id == c.cluster_id].groupby("tau_h")[["lon", "lat"]].mean()
        sc = ax.scatter(m.lon, m.lat, c=m.index, cmap="viridis", vmin=0, vmax=48, s=12, label="backward cloud centre (mean of 27)")
        x0, y0, x1, y1 = c.geometry.bounds
        x0, x1 = min(x0, m.lon.min()) - .12, max(x1, m.lon.max()) + .12
        y0, y1 = min(y0, m.lat.min()) - .12, max(y1, m.lat.max()) + .12
        s = sus[(sus.cluster_id == c.cluster_id) & (sus.direction_agreement != "neither")].head(3)
        for (_, v), col in zip(s.iterrows(), ["#dc2626", "#2563eb", "#ea580c"]):
            lab = f"#{v.suspect_rank} {v.VesselName or v.MMSI} ({v.tier}, {v.direction_agreement})"
            if v.fwd_F > 0:
                p = fpos[(fpos.MMSI == v.MMSI) & (fpos.hour >= v.fwd_window_start_h) & (fpos.hour < v.fwd_window_end_h)]
                ax.scatter(p.lon_t0, p.lat_t0, s=2, color=col, alpha=.25)
                lab += f"; virtual oil released {v.fwd_window_start_h:.0f}-{v.fwd_window_end_h:.0f} h before T0"
            inside = x0 <= v.t0_lon <= x1 and y0 <= v.t0_lat <= y1
            ax.scatter([v.t0_lon if inside else np.nan], [v.t0_lat if inside else np.nan], marker="^", s=60, color=col,
                       edgecolor="k", lw=.5, label=lab + ("" if inside else " [at T0 outside view]"))
        gpd.GeoSeries([c.geometry]).plot(ax=ax, color="black")
        ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
        ax.set_title(f"event {c.cluster_id}: {c.area_m2 / 1e6:.2f} km²", fontsize=9)
        ax.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0, -0.08), frameon=False)
        ax.set_aspect(1 / np.cos(np.radians(c.centroid_lat)))
    fig.colorbar(sc, ax=axs[0].tolist(), label="hours before T0", fraction=.02)
    fig.suptitle(f"{inc.id}: backward clouds, top suspects at T0 (▲) and their best-matching virtual discharge at T0", fontsize=10)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def posthoc(run, inc, P, out):
    """Reference comparison AFTER all outputs exist: which events overlap the reference slick, how its associated
    vessels fare in our ranking, and how the v3 ranking relates to the frozen V2 0-6 h ranking."""
    ref = shape(json.loads((ROOT / inc.spec["reference"]["slick_geojson"]).read_text())["features"][0]["geometry"])
    ref = ref if ref.is_valid else ref.buffer(0)
    cand, sus, *_ = _load(run)
    from .geo import geodesic_area_m2
    rows = []
    for _, c in cand.iterrows():
        inter = c.geometry.intersection(ref)
        rows.append({"cluster_id": c.cluster_id, "selected": c.selected, "overlap_m2": geodesic_area_m2(inter) if not inter.is_empty else 0.0,
                     "iou": geodesic_area_m2(inter) / geodesic_area_m2(c.geometry.union(ref)) if not inter.is_empty else 0.0})
    ov = pd.DataFrame(rows)
    ov = ov[ov.overlap_m2 > 0].sort_values("overlap_m2", ascending=False)
    ref_m = inc.reference_mmsi
    rs = sus[sus.cluster_id.isin(ov.cluster_id) & sus.MMSI.isin(ref_m)][
        ["cluster_id", "MMSI", "VesselName", "suspect_rank", "tier", "bwd_point", "bwd_peak_age_median", "fwd_F", "fss_1km"]]
    frozen = pd.read_csv(inc.frozen("ensemble_robust_scores"), dtype={"MMSI": str})
    fz06 = frozen[frozen.release_age_bin == "0-6h"].head(6)[["MMSI", "VesselName", "rank_median"]]
    res = {"note": "POST-HOC only; the reference did not select, seed, tune or filter anything.",
           "events_overlapping_reference": ov.to_dict("records"),
           "reference_vessels_in_our_ranking": rs.to_dict("records"),
           "reference_vessels_passing_funnel": int(sus[sus.cluster_id.isin(ov.cluster_id)].MMSI.isin(ref_m).groupby(sus.cluster_id).sum().max()
                                                   if len(ov) else 0),
           "frozen_v2_0_6h_top6": fz06.to_dict("records"),
           "top3_per_overlapping_event": {int(c): sus[sus.cluster_id == c].head(3)[["VesselName", "tier", "combined"]].to_dict("records")
                                          for c in ov.cluster_id}}
    (out / "posthoc.json").write_text(json.dumps(res, indent=1, default=float))
    return [out / "posthoc.json"]
