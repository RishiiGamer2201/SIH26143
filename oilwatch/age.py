"""Slick age: several independent cues, reported as intervals (no validated SAR-only age estimator exists).

1 width diffusion   cross-track spread of a line source: sigma^2 = sigma0^2 + 2 K t, W ~ 4 sigma
                    -> t = ((W/4)^2 - (W0/4)^2) / (2K) for K in the ensemble diffusivity range.
2 Okubo 1971        oceanic diffusion diagram sigma_rc^2 [cm^2] = 0.0108 t[s]^2.34 (scale-dependent diffusivity).
3 forward drift     release window of the best-matching virtual discharge (per vessel; slick-level = best supported suspect).
4 backward drift    release age at which the suspect's AIS position best overlaps the backward cloud.
5 persistence       thin ship-discharge slicks mostly survive ~2-12 h (Liu B. et al. 2021; Cerulean 3 h drift cap).
6 SAR darkness      'colour' of oil in radar = damping ratio (sea sigma0 / slick sigma0) and edge sharpness: fresh = strong
                    contrast + sharp edges, weathered = weak/diffuse (KSAT labels, Bianchi et al. 2020). Needs the sigma0
                    raster (git-ignored on this machine) -> darkness_features() runs when the raster is supplied.
7 optical colour    Bonn Agreement Oil Appearance Code (thickness) when coincident optical imagery exists (none for Incident 001).
Caveat: E007 strands are 2-3x wider than the reference polygon (over-segmentation), so cues 1-2 are biased OLD.
"""
import numpy as np
import pandas as pd

BAOAC = [  # code, appearance, thickness um (Bonn Agreement 2007); thicker = fresher / larger volume
    (1, "sheen (silvery/grey)", 0.04, 0.30), (2, "rainbow", 0.30, 5.0), (3, "metallic", 5.0, 50.0),
    (4, "discontinuous true colour", 50.0, 200.0), (5, "continuous true colour", 200.0, np.inf)]


def width_age_h(width_m, w0_m, K):
    return max((width_m / 4) ** 2 - (w0_m / 4) ** 2, 0.0) / (2 * K) / 3600


def okubo_age_h(width_m):
    sigma_cm2 = (width_m / 4 * 100) ** 2
    return (sigma_cm2 / 0.0108) ** (1 / 2.34) / 3600


def darkness_features(vv_db, mask, ring_px=10):
    """Damping ratio (histogram medians, Quigley et al. 2023) and edge contrast from a sigma0 dB window + slick mask."""
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(mask, iterations=ring_px) & ~mask & np.isfinite(vv_db)
    inner = mask & np.isfinite(vv_db)
    edge = binary_dilation(mask, iterations=2) & ~mask & np.isfinite(vv_db)  # 2-px outer rim
    sea, oil = np.nanmedian(vv_db[ring]), np.nanmedian(vv_db[inner])
    return {"damping_ratio_db": float(sea - oil), "edge_contrast_db": float(np.nanmedian(vv_db[edge]) - oil)}


def darkness_cue(dk):
    """SAR 'colour' cue row from the sar stage (descriptive class, not a validated age)."""
    if dk is None or dk.get("darkness_status") != "ok":
        return {"cue": "sar_darkness", "lo_h": np.nan, "hi_h": np.nan,
                "detail": "not run: no sigma0 for this event" if dk is None else dk.get("darkness_status")}
    cls = dk["darkness_class"]
    lo, hi = {"strong/fresh-looking": (0.0, 6.0), "intermediate": (2.0, 24.0), "weak/weathered-looking": (6.0, 48.0)}[cls]
    return {"cue": "sar_darkness", "lo_h": lo, "hi_h": hi,
            "detail": f"damping {dk['damping_ratio_db']:.1f} dB, edge sharpness {dk['edge_sharpness']:.2f} -> {cls} "
                      "(heuristic interval; over-wide E007 polygons bias damping LOW)"}


def cluster_age(cl, suspects, P, K_range, dk=None):
    A = P["age_v3"]
    w = cl.width_proxy_m
    ages = {K: width_age_h(w, A["initial_width_m"], K) for K in K_range}
    rows = [{"cue": "width_diffusion", "lo_h": min(ages.values()), "hi_h": max(ages.values()),
             "detail": f"W={w:.0f} m, K {min(K_range)}-{max(K_range)} m2/s (biased old: E007 strands 2-3x too wide)"},
            {"cue": "okubo_1971", "lo_h": okubo_age_h(w), "hi_h": okubo_age_h(w), "detail": f"sigma=W/4={w / 4:.0f} m"},
            {"cue": "persistence_prior", "lo_h": 2.0, "hi_h": 12.0, "detail": "typical ship-discharge slick lifetime"}]
    sup = suspects[suspects.tier.isin(["strongly consistent", "consistent"])].head(3)
    for _, s in sup.iterrows():
        if s.fwd_F > 0:  # a window with no virtual oil on the slick says nothing about age
            rows.append({"cue": f"forward_best_window:{s.VesselName or s.MMSI}", "lo_h": float(s.fwd_window_start_h),
                         "hi_h": float(s.fwd_window_end_h), "detail": f"F={s.fwd_F:.2f}, FSS1km={s['fss_1km']:.2f}"})
        rows.append({"cue": f"backward_peak_age:{s.VesselName or s.MMSI}", "lo_h": float(s.bwd_peak_age_median),
                     "hi_h": float(s.bwd_peak_age_median), "detail": "median over members"})
    rows.append(darkness_cue(dk))
    rows.append({"cue": "optical_baoac", "lo_h": np.nan, "hi_h": np.nan, "detail": "no coincident optical imagery"})
    df = pd.DataFrame(rows).assign(cluster_id=cl.cluster_id)
    sup_f = sup[sup.fwd_F > 0]
    appearance_hi = max(r["hi_h"] for r in rows[:3])  # width, Okubo, persistence
    if len(sup_f):
        best = sup_f.iloc[0]
        summary = {"estimate_basis": f"forward best window of top supported suspect ({best.VesselName or best.MMSI})",
                   "age_lo_h": float(best.fwd_window_start_h), "age_hi_h": float(best.fwd_window_end_h)}
    else:
        summary = {"estimate_basis": "no drift-supported suspect: width/Okubo/persistence only",
                   "age_lo_h": float(min(r["lo_h"] for r in rows[:3])), "age_hi_h": float(appearance_hi)}
    summary["appearance_age_hi_h"] = float(appearance_hi)
    summary["age_conflict"] = bool(len(sup_f) and summary["age_lo_h"] > appearance_hi)
    summary["age_conflict_note"] = ("drift-based release age is older than slick appearance allows: either an old, persistent "
                                    "slick, or a young slick from a source not in AIS (platform/infrastructure, dark vessel, "
                                    "natural seep); check infrastructure DB and SAR ship detections") if summary["age_conflict"] else ""
    return df, summary
