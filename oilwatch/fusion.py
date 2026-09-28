"""Forward-backward fusion (Luo et al. 2024 average of forward and backward similarity) + transparent tiers.

combined = sqrt(backward_rel * forward_rel), each relative to the best vessel for that slick event.
Tiers use absolute criteria:
  strongly consistent : forward FSS skilful AND backward top-3 in >= 50% of members
  consistent          : forward FSS skilful OR backward top-5 in >= 50% of members
  weak                : some support (combined > 0)
  not supported       : none
Extra flags (never change the score): fresh-discharge geometry match, AIS gap during the implied release window,
AIS speed jumps, non-standard MMSI, motion_state. Rankings are evidence of spatio-temporal consistency, never proof.
"""
import numpy as np
import pandas as pd

from .evidence import gap_overlaps


def sar_evidence(sar_det, cands):
    """Per (cluster, MMSI): SAR position at T0 of AIS-matched / AIS-silent vessels and its distance to the slick."""
    if sar_det is None or not len(sar_det):
        return None
    from shapely.geometry import Point
    from .geo import laea, to_metric
    rows = []
    d = sar_det.copy()
    d["mmsi_any"] = d.silent_mmsi.where(d.label == "ais_silent_at_t0", d.match_id.where(d.label.str.startswith("ais_")))
    d = d.dropna(subset=["mmsi_any"])
    for _, c in cands.iterrows():
        fwd, _ = laea(c.centroid_lon, c.centroid_lat)
        gm = to_metric(c.geometry, fwd)
        for _, r in d.iterrows():
            rows.append({"cluster_id": c.cluster_id, "MMSI": str(int(float(r.mmsi_any))), "sar_label": r.label,
                         "sar_dist_to_slick_km": gm.distance(to_metric(Point(r.lon, r.lat), fwd)) / 1000})
    return pd.DataFrame(rows).sort_values("sar_dist_to_slick_km").drop_duplicates(["cluster_id", "MMSI"])


def fuse(rob, fwd, geo, beh, motion, cands, P, sar_det=None):
    Fz = P["fusion_v3"]
    df = rob[rob.passes_funnel].merge(fwd, on=["cluster_id", "MMSI"], how="left") \
        .merge(geo, on=["cluster_id", "MMSI"], how="left").merge(beh, on="MMSI", how="left")
    keep = [c for c in ("MMSI", "motion_state", "source_type", "source_type_basis", "nearest_boem_structure", "nearest_boem_km")
            if c in motion]
    mo = motion[keep].assign(MMSI=lambda x: x.MMSI.astype(str))
    df = df.merge(mo, on="MMSI", how="left")
    df["motion_state"] = df.motion_state.fillna("not in E102 table")
    se = sar_evidence(sar_det, cands)
    if se is not None:
        df = df.merge(se, on=["cluster_id", "MMSI"], how="left")
        df["flag_ais_silent_seen_by_sar_near_slick"] = (df.sar_label == "ais_silent_at_t0") & (df.sar_dist_to_slick_km <= 2.0)
    df["fwd_evaluated"] = df.fwd_F.notna()  # not simulated != no support
    df["fwd_F"] = df.fwd_F.fillna(0.0)
    out = []
    for cid, g in df.groupby("cluster_id"):
        g = g.copy()
        g["bwd_rel"] = g.bwd_weighted / g.bwd_weighted.max() if g.bwd_weighted.max() > 0 else 0.0
        g["fwd_rel"] = g.fwd_F / g.fwd_F.max() if g.fwd_F.max() > 0 else 0.0
        g["combined"] = np.sqrt(g.bwd_rel * g.fwd_rel)
        skil = g.fss_skilful.fillna(False).astype(bool)
        bwd_ok = g.bwd_point >= Fz["bwd_min_point_density"]  # absolute support, not just relative rank
        strong = skil & bwd_ok & (g.bwd_top3_fraction >= Fz["strong"]["bwd_top3_fraction"])
        cons = skil | (bwd_ok & (g.bwd_top5_fraction >= Fz["consistent"]["bwd_top5_fraction"]))
        g["tier"] = np.select([strong, cons, g.combined > 0], ["strongly consistent", "consistent", "weak"], "not supported")
        fresh_age = P["geometry_v3"]["fresh_age_h"]
        g["flag_fresh_discharge_match"] = (g.geo_fresh_discharge >= 0.5) & (g.fwd_window_end_h <= fresh_age)
        g["flag_ais_gap_in_release_window"] = [gap_overlaps(iv, a, b) if pd.notna(a) else False
                                               for iv, a, b in zip(g.ais_gap_intervals, g.fwd_window_start_h, g.fwd_window_end_h)]
        g["flag_identity_anomaly"] = (~g.mmsi_standard.fillna(True).astype(bool)) | (g.ais_speed_jumps_gt_50kn.fillna(0) > 0)
        # descriptive only (tiers above are pre-registered and unchanged)
        fw, bw = g.fwd_F > 0, g.bwd_point >= Fz["bwd_min_point_density"]
        g["direction_agreement"] = np.select([fw & bw, bw & ~g.fwd_evaluated, bw, fw],
                                             ["forward+backward", "backward only (forward not evaluated)", "backward only",
                                              "forward only"], "neither")
        g["flag_window_at_48h_boundary"] = fw & (g.fwd_window_end_h >= 47)
        order = {"strongly consistent": 0, "consistent": 1, "weak": 2, "not supported": 3}
        g = g.assign(_t=g.tier.map(order)).sort_values(["_t", "combined"], ascending=[True, False]).drop(columns="_t")
        g["suspect_rank"] = np.arange(1, len(g) + 1)
        out.append(g)
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return apply_v31(res, P) if len(res) else res


def apply_v31(sus, P):
    """v3.1 decision = absolute rule calibrated on twin experiments (oilwatch/calibrate.py); v3 tiers kept beside it.
    'accused' only if the vessel passes all absolute thresholds AND is the best supported vessel for that slick."""
    from .config import ROOT
    cfg = P.get("fusion_v31", {})
    f = ROOT / cfg.get("calibration_file", "__none__")
    if not f.exists():
        return sus
    import json
    rule = json.loads(f.read_text())["rule"]
    ok = (sus.fwd_F >= rule["fwd_F_min"]) & (sus.bwd_point >= rule["bwd_point_min"]) &          (sus.fwd_member_support.fillna(0) >= rule["member_support_min"]) & ((sus.fwd_F > 0) | (rule["fwd_F_min"] == 0))
    sus = sus.assign(v31_supported=ok)
    m = rule.get("margin_vs_runner_up", 1.0)
    acc = []
    for _, g in sus.groupby("cluster_id"):
        sup = g[g.v31_supported].sort_values("combined", ascending=False)
        if len(sup):
            top = sup.iloc[0]
            runner = g[g.MMSI != top.MMSI].fwd_F.max() if (g.MMSI != top.MMSI).any() else 0.0
            if top.fwd_F >= m * runner:
                acc.append(sup.index[0])
    sus["v31_decision"] = np.where(sus.index.isin(acc), "high-confidence flag (calibrated)",
                                   np.where(ok, "supported", "lead only"))
    sus.attrs["v31_rule"] = rule
    return sus
