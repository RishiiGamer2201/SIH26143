"""Source-type layer v1: post-score annotation of AIS candidates (motion_state and source_type kept separate).

Does NOT change any compatibility score. Reads the frozen V2 tables and the V2.1 tables,
adds motion_state / source_type columns, writes typed copies to a new directory.

motion_state (inferred from AIS reports in [T0 - window_h, T0], thresholds in configs/source_type_v1.json):
  undetermined : n_reports < min_reports or track_span_h < min_span_h, unless max excursion > loitering.max_excursion_km
                 (then moving: a short track can show movement but cannot establish station-keeping)
  stationary   : p90 spread <= stationary.max_p90_spread_km and max excursion <= stationary.max_excursion_km
                 and frac(SOG < slow_sog_kn) >= stationary.min_frac_slow
  loitering    : same test with the looser loitering.* thresholds
  mixed        : stationarity_score >= mixed_min_stationarity_score (parked part of the window, moved otherwise)
  moving       : everything else
source_type in {vessel, fixed_infrastructure, dark_vessel, unknown} is NOT inferred from motion_state:
a stationary AIS track can be a platform, a DP drillship or an anchored ship. It needs vessel metadata plus an
authoritative offshore-infrastructure database / spatial match, which is not joined yet, so every AIS track is
"unknown" for now (source_type_basis says why). "dark_vessel" is reserved for SAR detections without AIS.
Metadata (AIS Status / VesselType) is reported as metadata_hint only; it decides nothing.
No vessel names are used.

Usage: python scripts/source_type_v1.py [--config configs/source_type_v1.json] [--out results/incident_001_source_type_v1]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from score_ais_density_v2_1 import haversine_km

ROOT = Path(__file__).resolve().parents[1]
SOURCE_TYPE_ENUM = ("vessel", "fixed_infrastructure", "dark_vessel", "unknown")
UNKNOWN_BASIS = ("not determined: needs vessel metadata + offshore-infrastructure database match (not joined yet); "
                 "motion_state is not evidence of type")
INPUTS = {  # name -> path relative to ROOT (read only)
    "v2_summary": "results/incident_001/ais_density_scores_v2.csv",
    "v2_release_age": "results/incident_001/release_age_scores.csv",
    "v2_1_summary": "results/incident_001_v2_1/ais_density_scores_v2_1.csv",
    "v2_1_release_age": "results/incident_001_v2_1/release_age_scores_v2_1.csv",
}


def track_metrics(g, radius_km, slow_kn):
    lon, lat = g.LON.to_numpy(), g.LAT.to_numpy()
    d = haversine_km(lon, lat, np.median(lon), np.median(lat))
    t = g.BaseDateTime
    return dict(n_reports=len(g), track_span_h=(t.max() - t.min()).total_seconds() / 3600,
                p50_spread_km=float(np.percentile(d, 50)), position_spread_km=float(np.percentile(d, 90)),
                max_excursion_km=float(d.max()), stationarity_score=float((d <= radius_km).mean()),
                median_sog=float(g.SOG.median()),  # NaN SOG ignored; all-NaN -> NaN and geometry alone decides
                frac_sog_lt_slow=float((g.SOG.dropna() < slow_kn).mean()) if g.SOG.notna().any() else np.nan)


def classify(m, cfg):
    if m["n_reports"] < cfg["min_reports"] or m["track_span_h"] < cfg["min_span_h"]:
        # sparse/short track: enough to show movement, not enough to claim station-keeping
        if m["max_excursion_km"] > cfg["loitering"]["max_excursion_km"]:
            return "moving", (f"short track (n={m['n_reports']}, span {m['track_span_h']:.1f} h) but moved "
                              f"{m['max_excursion_km']:.1f} km > {cfg['loitering']['max_excursion_km']} km")
        return "undetermined", f"n_reports={m['n_reports']} or span={m['track_span_h']:.1f}h below minimum"
    for name in ("stationary", "loitering"):
        c = cfg[name]
        sog_ok = np.isnan(m["frac_sog_lt_slow"]) or m["frac_sog_lt_slow"] >= c["min_frac_slow"]
        if m["position_spread_km"] <= c["max_p90_spread_km"] and m["max_excursion_km"] <= c["max_excursion_km"] and sog_ok:
            return name, (f"p90 {m['position_spread_km']:.2f}<= {c['max_p90_spread_km']} km, max {m['max_excursion_km']:.2f}"
                          f"<= {c['max_excursion_km']} km, slow frac {m['frac_sog_lt_slow']:.2f}>= {c['min_frac_slow']}")
    if m["stationarity_score"] >= cfg["mixed_min_stationarity_score"]:
        return "mixed", (f"{m['stationarity_score']:.2f} of reports within {cfg['stationary_radius_km']} km of median "
                         f"position but max excursion {m['max_excursion_km']:.1f} km")
    return "moving", f"p90 spread {m['position_spread_km']:.2f} km, stationarity {m['stationarity_score']:.2f}"


def _selftest(cfg):
    base = dict(n_reports=100, track_span_h=48, p50_spread_km=0, stationarity_score=1.0, median_sog=0)
    cases = [(dict(base, position_spread_km=0.1, max_excursion_km=0.3, frac_sog_lt_slow=1.0), "stationary"),
             (dict(base, position_spread_km=1.5, max_excursion_km=2.5, frac_sog_lt_slow=0.95, stationarity_score=0.3), "loitering"),
             (dict(base, position_spread_km=0.2, max_excursion_km=40, frac_sog_lt_slow=0.9, stationarity_score=0.9), "mixed"),
             (dict(base, position_spread_km=20, max_excursion_km=60, frac_sog_lt_slow=0.1, stationarity_score=0.05), "moving"),
             (dict(base, n_reports=4, position_spread_km=0, max_excursion_km=0, frac_sog_lt_slow=1.0), "undetermined"),
             (dict(base, position_spread_km=0.01, max_excursion_km=0.02, frac_sog_lt_slow=np.nan), "stationary"),
             (dict(base, track_span_h=3, position_spread_km=30, max_excursion_km=60, frac_sog_lt_slow=0.0), "moving")]
    for m, want in cases:
        assert classify(m, cfg)[0] == want, (m, want, classify(m, cfg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/source_type_v1.json"))
    ap.add_argument("--out", default=str(ROOT / "results/incident_001_source_type_v1"))
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    out = Path(a.out).resolve()
    assert out != (ROOT / "results/incident_001").resolve(), "refusing to write into frozen results/incident_001"
    out.mkdir(parents=True, exist_ok=True)
    _selftest(cfg)

    t0 = pd.Timestamp(cfg["incident_t0_utc"])
    ais = pd.read_parquet(ROOT / cfg["ais_parquet"])
    ais = ais[(ais.BaseDateTime >= t0 - pd.Timedelta(hours=cfg["window_h_before_t0"])) & (ais.BaseDateTime <= t0)]
    ais = ais.sort_values("BaseDateTime").drop_duplicates(["MMSI", "BaseDateTime"])

    # round_trip parser: default fast parser drops last-ULP bits, typed copies must carry scores bit-exactly
    tables = {k: pd.read_csv(ROOT / p, float_precision="round_trip") for k, p in INPUTS.items()}
    mmsis = sorted(set().union(*(set(t.MMSI.astype("int64")) for t in tables.values())))

    rows = []
    for mmsi in mmsis:
        g = ais[ais.MMSI == mmsi]
        if g.empty:
            rows.append(dict(MMSI=mmsi, n_reports=0, track_span_h=0.0, motion_state="undetermined",
                             classification_reason="no AIS reports in window"))
            continue
        m = track_metrics(g, cfg["stationary_radius_km"], cfg["slow_sog_kn"])
        beh, why = classify(m, cfg)
        status = g.Status.mode()
        vtype = g.VesselType.mode()
        hints = []
        if len(status) and str(int(status.iloc[0])) in cfg["metadata_hint_status_codes"]:
            hints.append(cfg["metadata_hint_status_codes"][str(int(status.iloc[0]))])
        if len(vtype) and str(int(vtype.iloc[0])) in cfg["metadata_hint_vessel_types"]:
            hints.append(cfg["metadata_hint_vessel_types"][str(int(vtype.iloc[0]))])
        rows.append(dict(MMSI=mmsi, **m, motion_state=beh, classification_reason=why,
                         ais_status_mode=status.iloc[0] if len(status) else np.nan,
                         ais_vessel_type_mode=vtype.iloc[0] if len(vtype) else np.nan,
                         metadata_hint=";".join(hints)))
    beh = pd.DataFrame(rows)
    beh["source_type"], beh["source_type_basis"] = "unknown", UNKNOWN_BASIS
    beh["stationary_flag"] = beh.motion_state.eq("stationary")
    assert beh.source_type.isin(SOURCE_TYPE_ENUM).all() and beh.MMSI.is_unique
    beh.to_csv(out / "track_motion.csv", index=False)

    typed_cols = ["MMSI", "motion_state", "stationary_flag", "source_type", "source_type_basis", "stationarity_score",
                  "position_spread_km", "max_excursion_km", "median_sog", "frac_sog_lt_slow", "track_span_h", "n_reports",
                  "metadata_hint", "classification_reason"]  # classification_reason explains motion_state
    for k, t in tables.items():
        typed = t.assign(MMSI=t.MMSI.astype("int64")).merge(beh[typed_cols], on="MMSI", how="left", validate="many_to_one")
        # score columns are copied unchanged
        for c in t.columns:
            assert typed[c].equals(t[c]) or c == "MMSI", c
        typed.to_csv(out / f"{k}_typed.csv", index=False)

    json.dump(cfg, open(out / "config_used.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    show = beh.merge(tables["v2_summary"][["MMSI", "VesselName", "point_rank"]].assign(MMSI=lambda x: x.MMSI.astype("int64")),
                     on="MMSI", how="left").sort_values("position_spread_km")
    print(show[["VesselName", "MMSI", "n_reports", "track_span_h", "p50_spread_km", "position_spread_km", "max_excursion_km",
                "stationarity_score", "median_sog", "frac_sog_lt_slow", "motion_state", "metadata_hint", "point_rank"]]
          .round(3).to_string(index=False))
    print("\nmotion_state counts:", beh.motion_state.value_counts().to_dict())
    print("source_type counts:", beh.source_type.value_counts().to_dict())
    v2 = tables["v2_summary"].assign(MMSI=lambda x: x.MMSI.astype("int64")).merge(beh[["MMSI", "motion_state", "source_type"]], on="MMSI")
    print("\nV2 point-release top 15 with motion_state / source_type:")
    print(v2.sort_values("point_rank").head(15)[["point_rank", "VesselName", "point_release_score", "motion_state", "source_type"]]
          .to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
