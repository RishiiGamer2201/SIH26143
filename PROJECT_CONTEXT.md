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
│   ├── score_ais_density_v2_1.py  E101 V2.1 = V2 with interpolated AIS positions -> results/incident_001_v2_1
│   └── source_type_v1.py          E102 post-score source-type layer -> results/incident_001_source_type_v1
├── configs/                       ais_v2_1.json, source_type_v1.json (all thresholds for E101/E102)
├── logs/                          E001_*.log, E002_*.log, E003_*.log, E101_*.log, E102_*.log
├── data/
│   ├── sar/extracted/             Zenodo dataset, see SAR_DATA_AUDIT.md §1 (never modified)
│   ├── sar/cache/v1/              E002 cache: vv20/lab20 [3020,1024²], vv40/lab40 [3020,512²], meta.parquet (12 GB)
│   ├── ais/processed/             NOAA AIS 2023-01-01..03 parquet (23.8 M rows)
│   └── incident_001/
│       ├── ais/                   ais_roi.parquet (59,077 rows, 162 MMSI), candidate tracks, closest approach, summary json
│       ├── currents/              ocean_surface_velocity.nc (raw CMEMS SMOC), ocean_opendrift.nc (uo+utide, vsdx/vsdy)
│       ├── wind/                  era5_wind_10m(_151h).nc + download script
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
    └── E006_segformer_test/       E006 one-shot test: metrics (3 views), per-tile, component recall, figures, u8 probs (git-ignored)
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
