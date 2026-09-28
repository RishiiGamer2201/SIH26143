# E008 — Slick-candidate selection policy v1 (reference-independent)

Status: **pre-registered 2026-09-29, before any E008 attribution result was computed.** Implemented in
`oilwatch/seed.py`, parameters in `configs/pipeline_v1.json` → `seed`. The reference (Cerulean) polygon, its bbox and its
vessel list are **not read** by this policy. They are read only by the `posthoc` stage, after all outputs are written.

## Problem
E007 produced 49 raw components (threshold 0.60, 4-connected) across the scene. Some are fragments of one slick and some are
unverified detections. Physics needs seed geometries, and choosing them by looking at Cerulean would be circular.

## Policy
1. **Input:** the frozen E007 components (`incident001_predicted_components.geojson`) with no filtering (min component area 0).
2. **Merge fragments into slick events:** components whose gap is ≤ **1 km** (metric LAEA) are single-linkage
   clustered. The value equals Cerulean's 500 m closing buffer (SkyTruth cerulean-cloud). Operational datasets show ~100 polygons per
   spill event (KSAT, Bianchi et al. 2020).
3. **Per event:**
   - geodesic area, number of parts, area-weighted E005 probability;
   - long axis from the minimum rotated rectangle: end points, length, bearing;
   - width proxy = area / length;
   - ERA5 10 m wind at the centroid at T0.
4. **Wind detectability class:**
   - valid 2.5–10 m/s;
   - marginal 1.5–2.5 or 10–12 m/s;
   - invalid otherwise;
   - factors 1 / 0.5 / 0.2.

   Sources: Topouzelis 2008; Fingas & Brown 2018. Below ~2–3 m/s the scene is dominated by look-alikes; above ~10 m/s slicks vanish.
5. `slick_potential = confidence × wind_factor × (1 − exp(−area / 0.1 km²))`.
6. **Attribution-eligible if all hold:**
   - area ≥ **23,000 m²** (64 px on the ×2 grid; E005 *validation* misses 73% of GT components ≤ 64 px);
   - inside the forcing domain (0.2° margin);
   - centroid inside AIS coverage.

   Ineligible events stay in the table with their reason.
7. **Selected** = the top **12** eligible events by `slick_potential`. An analyst may override with
   `"seed": {"clusters": [...]}` in the incident spec. The override is recorded as `selection_basis = analyst_override`.

## Known limitation
Incident 001's AIS extract (`ais_roi.parquet`) was cut around the reference slick when the dataset was built, so the
AIS-coverage gate inherits that region. For full scene-wide autonomy, extract AIS for the whole scene footprint from
the full NOAA file (`data/ais/processed/`, on the WSL machine). The four largest detections (north of 28.1° N) are
currently ineligible **only** for this reason.

## What E008 does with the selection
Each selected event is seeded into the v3 backward ensemble and the forward-hypothesis test (see
`EXPERIMENT_LOG.md` E008). Every event gets its own suspect table. Events are never merged post hoc, and none is chosen by its agreement with the reference.
