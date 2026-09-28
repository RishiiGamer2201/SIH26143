# SIH 2026 · PS 26143 — Project Context

Oil-spill detection → slick geometry → backward drift → release region/age → source attribution
(vessels / fixed infrastructure / dark vessels) → forward forecast → analyst dashboard.
This is a **forensic attribution pipeline**, not a segmentation project. Rankings are evidence of
spatio-temporal consistency, **never proof of culpability**.

Last audit: 2026-09-26 (see `SAR_DATA_AUDIT.md`, `EXPERIMENT_LOG.md`).

## Environment

- WSL Ubuntu, project root `~/SIH26143` (≈130 GB). 20 CPU cores, **15 GB RAM visible to WSL**, 418 GB free disk.
- GPU: RTX 5070 Ti 16 GB (Blackwell, sm_120 → needs PyTorch built for CUDA ≥ 12.8).
- Conda env `opendrift` (Python 3.12): OpenDrift 1.14.11, GDAL 3.10.3, numpy 2.5, pandas 3.0, shapely 2.1, scipy, sklearn 1.9. **No torch, no rasterio.**
- Conda env `sih-ml` (E001, 8.7 GB): Python 3.12, torch 2.11.0+cu128 (sm_120 in arch list), cuDNN 9.19, timm 1.0.30,
  segmentation-models-pytorch 0.5.0, GDAL 3.13, sklearn 1.9.1, matplotlib. No rasterio (GDAL covers reads). Use for all ML; `opendrift` for physics.
- Kaggle dataset `rishiikumarsingh/sih26143-complete-project`, mount `/kaggle/input/datasets/rishiikumarsingh/sih26143-complete-project` (read-only; write to `/kaggle/working`).

## Hard links — read before editing

`~/kaggle_upload/sih26143-complete-project/` is a **hard-link mirror** of the project data, results and
the 7 physics scripts (scripts appear there twice: root and `scripts/`, link count 3). Consequences:
- Editing a physics script in place changes the Kaggle staging copy too; an editor that saves via
  rename silently breaks the link and the two copies diverge.
- Files created on 2026-09-26 (`scripts/`, `results/sar_audit/`, the `.md` docs) are **not** in the mirror.
- The local project has no `scripts/` directory before 2026-09-26; the Kaggle upload does (duplicates of root scripts).

## Repository map

```
SIH26143/
├── run_hindcast_48h.py            deterministic 48 h OpenOil backward run (Cerulean polygon, 2000 particles)
├── run_hindcast_ensemble.py       27-scenario ensemble (windage × current unc × wind unc, 400 particles each)
├── score_ais_hindcast.py          V1 min-distance score (baseline only)
├── score_ais_hindcast_v1.py       byte-identical duplicate of the above
├── score_ais_density_v2.py        V2 Gaussian density score vs whole particle cloud
├── analyze_release_age.py         V2 scores split into release-age bins
├── score_ensemble_robustness.py   scenario × age-bin × vessel ranks, top-k stability
├── scripts/                       (new, 2026-09-26) SAR audit + S1 calibration
│   ├── sar_channel_evidence.py    VV/VH physics tests on sampled tiles
│   ├── s1_grd_calibrate.py        σ0 calibration + thermal-noise removal for GRD windows; labelled pol check
│   ├── sar_build_index.py         per-image index of all 3020 tiles (+64×64 thumbnails)
│   ├── sar_overlap_content.py     same-acquisition vs same-place detection for overlapping tiles
│   ├── sar_make_splits.py         leakage-safe split (policy acq|strict)
│   ├── sar_build_cache.py         E002 VV + label/validity cache (20 m, 40 m) -> data/sar/cache/v1
│   ├── sar_check_cache.py         E002 integrity checks + sample PNG -> results/sar_cache_v1
│   ├── e003_feature_baseline.py   E003 feature classifier + shortcut probes -> results/E003_feature_baseline
│   ├── e004_resnet18_cls.py       E004 ResNet-18 tile classifier (40 m) -> results/E004_resnet18_cls
│   ├── e005a_audit.py             E005-A mentor Kaggle dataset audit -> data/segformer_combined/ (see MENTOR_DATASET_AUDIT.md)
│   ├── kaggle/e005a_audit/        private Kaggle kernel used by E005-A (hashes the dataset next to its mount)
│   ├── e005_segformer.py          E005 SegFormer MiT-B2 segmentation (20 m), tests/train/validate -> results/E005_segformer_b2
│   ├── e006_segformer_test.py     E006 freeze E005 (FROZEN.json) + one-shot official test -> results/E006_segformer_test
│   ├── e008a_candidate_hindcast.py  E008a deterministic hindcast per policy-v1 candidate (frozen recipe; coverage audit) -> results/E008a_candidate_hindcast
│   ├── e008b_prepare_forcing_v2.py  E008b forcing v2: v1 provenance rebuild, domain config, download, build, v1/v2 equivalence, freeze -> data/incident_001/forcing_v2, results/E008b_forcing_expansion
│   ├── e008b_candidate_hindcast_v2.py  E008b 42 deterministic re-runs with forcing v2 (imports E008a unchanged) + v1/v2 comparison -> results/E008b_candidate_hindcast_v2
│   ├── e008c_candidate_ensemble.py  E008c frozen 27-scenario ensemble per candidate, forcing v2 (audit, pilot, run-all, summary) -> results/E008c_candidate_ensemble
│   ├── e008c_baseline_reproduction_check.py  E008c recipe check: historical inputs reproduce results/incident_001 ensemble bit-exactly
│   ├── e008_candidate_policy_v1.py  E008 candidate policy v1: freeze config, apply once to E007 (no grouping, A_min 5000), post-hoc Cerulean map -> results/E008_candidate_policy_v1
│   ├── e007c_validate_clustering.py  E007c grouping policy from E005 val only (G / A_min / orientation sweeps; its G-based freeze/apply stages are unused) -> results/E007c_validation_clustering
│   ├── e007b_component_structure.py  E007b model-only component/cluster structure (no Cerulean) -> results/E007b_component_structure
│   ├── e007_incident001_ml.py     E007 Incident 001 SAFE -> σ0 VV geocoded -> frozen E004/E005 -> mask -> polygons -> results/E007_incident001_ml
│   ├── score_ais_density_v2_1.py  E101 V2.1 = V2 with interpolated AIS positions -> results/incident_001_v2_1
│   └── source_type_v1.py          E102 post-score source-type layer -> results/incident_001_source_type_v1
├── configs/                       ais_v2_1.json, source_type_v1.json (E101/E102 thresholds); e008_candidate_policy_v1.json + .sha256 (FROZEN, never edit)
├── logs/                          E001_*.log, E002_*.log, E003_*.log, E101_*.log, E102_*.log
├── data/
│   ├── sar/extracted/             Zenodo dataset, see SAR_DATA_AUDIT.md §1 (never modified)
│   ├── sar/cache/v1/              E002 cache: vv20/lab20 [3020,1024²], vv40/lab40 [3020,512²], meta.parquet (12 GB)
│   ├── ais/processed/             NOAA AIS 2023-01-01..03 parquet (23.8 M rows)
│   └── incident_001/
│       ├── ais/                   ais_roi.parquet (59,077 rows, 162 MMSI), candidate tracks, closest approach, summary json
│       ├── currents/              ocean_surface_velocity.nc (raw CMEMS SMOC), ocean_opendrift.nc (uo+utide, vsdx/vsdy)
│       ├── wind/                  era5_wind_10m(_151h).nc + download script
│       ├── forcing_v2/            FROZEN forcing v2 (expanded extent, same recipe): raw/, ocean_opendrift.nc, era5_wind_10m_151h.nc, FROZEN.json
│       ├── metadata/              incident.json, slick_3775938.geojson, cerulean_candidates.geojson (*contains the two SLICKS 3775938 & 3750297, not candidates*)
│       ├── sentinel1/             SAFE (IPF 003.52, ascending, VV+VH), asf_scene_metadata.json
│       └── waves/                 empty
└── results/
    ├── incident_001/              FROZEN physics + attribution outputs (verified bit-exact reproducible) — never write here
    ├── incident_001_v2_1/         E101 V2.1 scores + V2 vs V2.1 comparison
    ├── incident_001_source_type_v1/  E102 track behaviour + typed copies of V2 / V2.1 tables
    ├── sar_audit/                 SAR audit artefacts (index, groups, splits, evidence)
    ├── sar_cache_v1/              E002 cache checks
    ├── E003_feature_baseline/     E003 features, metrics
    ├── E004_resnet18_cls/         E004 checkpoint, curves, val + one-shot test metrics
    ├── E005_segformer_b2/         FROZEN E005 checkpoint (FROZEN.json hashes), tests, curves, val sweep/metrics/figures
    ├── E006_segformer_test/       E006 one-shot test: metrics (3 views), per-tile, component recall, figures, u8 probs (git-ignored)
    ├── E008_candidate_policy_v1/  E008 policy v1 applied once: 42 candidate components (CSV/GeoJSON), 7 excluded, report; post-hoc Cerulean mapping (separate file)
    ├── E007c_validation_clustering/ E007c val component tables, intra/inter gaps, G + A_min sweeps, report, curves
    ├── E007b_component_structure/ E007b model-only component table, clusters at predeclared gaps, figure; post-hoc ROI map (separate file)
    └── E007_incident001_ml/       E007 Incident 001 ML: reports, windows, predicted components (GeoJSON/CSV), post-hoc Cerulean comparison; rasters/ git-ignored
```
`data/segformer_combined/` holds the E005-A audit outputs: the manifest, the Drone-RGB "M4D" subset and the kernel outputs. The Trujillo copy there was not stored; it is identical to `data/sar/extracted/`.

## Status

Done and verified (2026-09-26):
- Physics chain reproduces **bit-exactly** (re-ran deterministic hindcast + V2 scoring into scratch: Δlon = Δlat = 0, Δscore = 0, identical ranks).
- OpenDrift readers really deliver wind, current and Stokes drift (no silent zero fallback).
- SAR band order resolved: **band 1 = VH, band 2 = VV** (SAR_DATA_AUDIT.md §4).
- SAR dataset fully indexed; leakage quantified; split files written. **Primary dev split: `acq`**
  (`splits_v1_acq.parquet`); `strict` kept for a later sensitivity check. Every final model is reported on
  (a) official test minus the 3 nodata tiles, (b) acquisition-clean test, (c) place-clean test — never only the aggregate.
- E001 env, E002 cache, E003 feature baseline done; E101 (AIS V2.1) and E102 (source types) done. E004 (CNN) done.
- E005-A approved and frozen. E005 SegFormer (Trujillo only) FROZEN (`results/E005_segformer_b2/FROZEN.json`, epoch 9, thr 0.60). E006 one-shot test accepted as the frozen baseline test result: dominant failure on broad oil masks, strongly associated with a label-coverage distribution shift (region, annotation style, intensity shift not disentangled).
- Reproducibility checkpoint (E005 freeze + E006 test): git commit `971c9490c5a36ecbaafa570f3a766a82c8410903`
  on `main` of the private repo github.com/RishiiGamer2201/SIH26143.
- E007 (2026-09-28) FROZEN (`results/E007_incident001_ml/FROZEN.json`; `python scripts/e007_incident001_ml.py verify`). Approved as an INTEGRATION
  milestone: successful real-scene pipeline integration with partial/over-segmented Incident 001 recovery, NOT evidence of robust scene-wide
  autonomous segmentation. Frozen E005 on our calibrated + GCP/TPS-geocoded VV (EPSG:4326, 17.8 × 19.9 m at the ROI). Post hoc vs Cerulean:
  IoU 0.340, reference covered 80.5%, prediction inside reference 37.1%, ROI union 2.70 vs 1.243 km², centroid offset 938 m, strands ~2–3× wider.
  Only 0.061% of valid scene pixels exceed 0.60, but they form 49 components totalling ~25.9 km² incl. large unverified detections (9.7, 4.9, 2.1 km²).
  Absolute geolocation error of the GCP/TPS product has not been measured (the Cerulean area match validates the AREA method only).
  **E004 must not gate E005** (the reference-slick windows are argmax lookalike, p(oil) 0.12–0.34).
- Reproducibility checkpoint (E007 freeze): git commit `4b9e1783aa5612c91c9db5d625030af512a63969` on `main` (rasters git-ignored; their sha256 are in FROZEN.json).
- E007b: the Incident-area components form one isolated (18.6 km), orientation-coherent cluster at link gap ≥ 2 km, but it is not the largest
  model-only cluster; the northern linear features (unverified) are bigger.
- E007c (validation only): NO defensible single global link gap G (r_area >= 0.95 needs G >= 250 m, where wrong-GT merges are 17.8%; <= 5% only at 100 m).
  A_min rule gives 5,000 m². No candidate policy frozen, nothing applied to E007. G = 5 km via AIS σ was rejected (different concept).
  Analysis commit `cf51f5bd21f22b3e41d9af676e0c9598f9aea360` (rules predeclared in `c56fdde636858ce2770d15f55ce710f06ff124dc`).
- E008 candidate policy v1 FROZEN (2026-09-28, reviewer Option 1): `configs/e008_candidate_policy_v1.json`, sha256 `3fb833df…f5bb27`.
  No grouping (explicit mode, not G = 0); drop components < 5,000 m²; each remaining E007 component = one candidate `E007-C{id}`.
  Applied once: 42 candidates, 7 excluded (19,068 m²). Post hoc only: 8 candidates intersect Cerulean (C30, C33–C38, C40); no winner.
  Cerulean geometry / ROI never selects, merges or ranks candidates. Physics (hindcast / ensemble / scoring) per candidate NOT started.
- Reproducibility checkpoint (E008 policy v1 freeze + application): git commit `e3aa9a63d2cebd043262a4fa1ea54db133a0aba1` on `main`.
- E008a (2026-09-28): deterministic 48 h hindcast per candidate (`scripts/e008a_candidate_hindcast.py`, `results/E008a_candidate_hindcast/`).
  Frozen recipe reproduced exactly (0 config diffs). 40/42 physics-valid; C12 (1,595/2,000) and C14 (2,000/2,000) deactivated at the
  ERA5 wind east edge (-88.5°). Outside current/wind = deactivation (`missing_data`), not silent zero. Ensemble / AIS NOT run.
  Reproducibility checkpoint (E008a): git commit `b4c7e871a3b9d313289224e9a3524c1ab50d0bc5`.
- E008b (2026-09-28): forcing v2 = same products/recipe, domain W -92.75 E -86.25 S 24.5 N 30.25 (E008a trajectory envelope + 2°,
  0.25° snap). v1 provenance reproduced (ocean byte-identical). v1/v2 common domain 100% bit-exact. FROZEN
  (`data/incident_001/forcing_v2/FROZEN.json`). 42/42 re-runs valid incl. C12/C14; 30/40 v1-valid runs bit-exact, 10 differ by 1 float32
  ulp (max 0.76 m). Reproducibility checkpoint (E008b): git commit `262ddaed53fb363191c88630312dfe9c1af93bb4`.
  Forcing v2 is the frozen forcing baseline for all later candidate physics; any change = forcing_v3.
- E008c (2026-09-28): frozen 27-scenario ensemble for all 42 candidates with forcing v2 (`scripts/e008c_candidate_ensemble.py`,
  `results/E008c_candidate_ensemble/`). Recipe audit 0 diffs; E008c functions reproduce the historical ensemble bit-exactly.
  1134/1134 scenarios valid, 0 deactivations/exits, min margin 210.5 km. Hourly positions kept for later age bins.
  AIS / release-age scoring NOT run; no candidate ranking or fusion.

## How to state the attribution result

Required wording for the 0–6 h bin:
> For the 0–6 h release hypothesis, the six Cerulean-associated candidates are consistently compatible with the
> observed slick at detection time. Because their peak compatibility occurs at age 0–1 h, this result is driven
> largely by proximity to the observed slick rather than by long-horizon hindcast reconstruction.

- Do **not** describe 0–6 h rank invariance across the 27 scenarios as evidence that the hindcast robustly
  identifies those six sources (at age ≤ 1 h all scenarios still ≈ the seed polygon).
- Sensitivity to backward drift is discussed with the older bins (6–12, 12–24, 24–36, 36–48 h) only.
- Rankings are shown with the E102 `motion_state` column (AIS behaviour) and a separate `source_type`
  (vessel / fixed_infrastructure / dark_vessel / unknown). **Stationary motion is not evidence of fixed infrastructure**
  (platform, DP drillship and anchored ship all look stationary); `source_type` stays `unknown` until vessel metadata
  + an authoritative offshore-infrastructure database / spatial match is joined. Of the six 0–6 h candidates, 3 have motion_state `stationary`
  (ARGOS PLATFORM, OCEAN BLACKHORNET, OCEAN BLACKLION), C-CONSTRUCTOR `loitering` (≤ 2.5 km for 48 h), HARVEY SUPPORTER
  `mixed` (parked ~60% of the window, then moved), SHELIA BORDELON `moving`.
- V2 is the frozen baseline; V2.1 (interpolated AIS) is reported beside it, not instead of it.

## Planned physics sensitivity (not run; frozen baseline unchanged)

Possible windage/Stokes double counting (OpenOil wind_drift_factor already ~contains wave drift, plus explicit SMOC Stokes).
Future experiment, new script + new output dir: (1) explicit Stokes + lower/no windage, (2) windage without explicit Stokes,
(3) current frozen baseline. Compare release-region and per-bin ranks; do not pick the variant by Cerulean agreement.

Open — see `EXPERIMENT_LOG.md` for the ordered plan. The remaining roadmap from the handover
(classifier, segmentation, incident inference, our polygon into physics, infrastructure branch,
dark vessels, AIS anomaly features, forward forecast, dashboard, more incidents) is unchanged.

## Frozen physics parameters (do not change without a logged methodological reason)

AIS tolerance ±20 min · σ = 5 km · 48 h backward · dt 15 min · output 1 h · age bins 0–6/6–12/12–24/24–36/36–48 h ·
windage 0.02/0.03/0.04 · current unc 0/0.05/0.10 m/s · wind unc 0/0.5/1.0 m/s · water current = uo+utide, Stokes = vsdx/vsdy (never utotal + Stokes).

## Rules that matter

- Cerulean labels: post-hoc validation only; never a feature, never a tuning target.
- Never train on or tune against the official test set; 3 test tiles are all-nodata (exclude, report).
- Pixel counts are image-space diagnostics only. The cache is an EPSG:4326 degree grid ("20 m" holds in y only; x ≈ 20·cos(lat) m),
  so **never** compute area as pixel_count × 400 m². Real slick area: georeference the mask via the geotransform, vectorize,
  then compute geodesically or in a suitable projected CRS.
- Never feed raw GRD DN to models. Never modify original TIFFs. Never load the full SAR set into RAM.
- Separate source types: vessels, fixed infrastructure, dark vessels (E102 layer; post-score, never changes a score, no names hard-coded).
  `motion_state` (from AIS) and `source_type` (from metadata + infrastructure DB) are separate; never derive one from the other.
- Never edit frozen root physics scripts in place (hard links). Modified method = new versioned script + new output dir.
