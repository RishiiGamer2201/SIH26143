"""Independent evidence columns (never folded into the drift scores; shown beside them).

geometry  fresh-discharge test at T0 (Cerulean source scoring): a vessel discharging now sits at/near one END of a
          linear slick and heads along its axis. P = exp(-(d_end/d_ref)^5), d_ref = max(3 km/h * dt, 0.5 km);
          A = exp(-ln2 (dtheta/45)^2) for moving vessels. Relevant only for young slicks (<= ~3 h).
behaviour AIS anomalies in T0-48 h..T0: reporting gaps >= 60 min (possible AIS-off), implied-speed jumps > 50 kn
          (position glitch/spoofing), non-standard MMSI (not 9 digits / MID not 2xx-7xx), SOG < 2 kn in open water
          away from the vessel's own long stationary periods (loitering).
motion    frozen E102 motion_state (reused unchanged; source_type stays 'unknown' until an infrastructure DB is joined).
"""
import numpy as np
import pandas as pd

from .ais import ais_gaps, state_at
from .geo import angle_diff_axis, haversine_km


def geometry(cl, ais, mmsis, t0, G, cfg21):
    rows = []
    for m in mmsis:
        g = ais[ais.MMSI == m]
        s = state_at(g, t0, cfg21)
        if not np.isfinite(s["lon"]):
            # fall back to nearest report (Cerulean scores +/- hours around the image with a time penalty)
            s.update(lon=float(g.iloc[(g.BaseDateTime - t0).abs().argmin()].LON),
                     lat=float(g.iloc[(g.BaseDateTime - t0).abs().argmin()].LAT))
        dt_h = s["nearest_report_min"] / 60
        d_end = float(min(haversine_km(s["lon"], s["lat"], cl.end1_lon, cl.end1_lat),
                          haversine_km(s["lon"], s["lat"], cl.end2_lon, cl.end2_lat)))
        d_ref = max(G["spread_rate_km_h"] * dt_h, G["min_ref_km"])
        P = float(np.exp(-(d_end / d_ref) ** 5))
        moving = np.isfinite(s["sog"]) and s["sog"] >= G["moving_sog_kn"] and np.isfinite(s["cog"]) and s["cog"] < 360
        dth = angle_diff_axis(s["cog"], cl.axis_bearing_deg) if moving else np.nan
        A = float(np.exp(-np.log(2) * (dth / 45) ** 2)) if moving else np.nan
        rows.append({"cluster_id": cl.cluster_id, "MMSI": m, "t0_lon": s["lon"], "t0_lat": s["lat"], "t0_position_method": s["method"],
                     "t0_nearest_report_min": s["nearest_report_min"], "t0_sog_kn": s["sog"], "t0_cog_deg": s["cog"],
                     "dist_to_slick_end_km": d_end, "geo_head_proximity": P, "heading_vs_axis_deg": dth, "geo_heading_alignment": A,
                     "geo_fresh_discharge": float((2 * P + A) / 3) if moving else P})
    return pd.DataFrame(rows)


def valid_mmsi(m):
    s = str(m)
    return len(s) == 9 and s.isdigit() and s[0] in "234567"


def behaviour(ais, mmsis, t0, window_h=48, gap_min=60, max_kn=50):
    rows = []
    t1 = t0 - pd.Timedelta(hours=window_h)
    for m in mmsis:
        g = ais[(ais.MMSI == m) & (ais.BaseDateTime >= t1) & (ais.BaseDateTime <= t0)].sort_values("BaseDateTime")
        gaps = ais_gaps(ais[ais.MMSI == m], t1, t0, gap_min)
        # implied speeds between consecutive reports
        if len(g) > 1:
            dt_h = np.diff(g.BaseDateTime.values).astype("timedelta64[s]").astype(float) / 3600
            d = haversine_km(g.LON.values[:-1], g.LAT.values[:-1], g.LON.values[1:], g.LAT.values[1:])
            v = np.where(dt_h > 0, d / 1.852 / np.maximum(dt_h, 1e-9), 0)
            jumps = int((v > max_kn).sum())
        else:
            jumps = 0
        last = g.BaseDateTime.max() if len(g) else pd.NaT
        rows.append({"MMSI": m, "ais_reports_48h": len(g), "ais_gaps_ge_60min": len(gaps),
                     "ais_longest_gap_min": float(gaps.gap_min.max()) if len(gaps) else 0.0,
                     "ais_gap_intervals": ";".join(f"{(t0 - a).total_seconds() / 3600:.1f}-{(t0 - b).total_seconds() / 3600:.1f}h"
                                                   for a, b in zip(pd.to_datetime(gaps.gap_start, utc=True), pd.to_datetime(gaps.gap_end, utc=True))),
                     "ais_last_report_before_t0_min": float((t0 - last).total_seconds() / 60) if pd.notna(last) else np.nan,
                     "ais_speed_jumps_gt_50kn": jumps, "mmsi_standard": valid_mmsi(m),
                     "vessel_type": float(g.VesselType.mode().iloc[0]) if len(g) and g.VesselType.notna().any() else np.nan,
                     "length_m": float(g.Length.max()) if len(g) and g.Length.notna().any() else np.nan})
    return pd.DataFrame(rows)


def gap_overlaps(intervals, a, b):
    """True if any AIS gap 'x-yh' (hours before T0) overlaps the window [a, b) hours before T0."""
    if not isinstance(intervals, str):  # NaN: no gaps
        return False
    for iv in filter(None, intervals.split(";")):
        hi, lo = (float(x) for x in iv.rstrip("h").split("-"))  # gap from hi h before T0 to lo h before T0
        if lo < b and hi > a:
            return True
    return False
