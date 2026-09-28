# SIH26143: Satellite oil-spill detection and AIS-based vessel attribution

> **Smart India Hackathon 2026 · Problem Statement 26143 · National Technical Research Organisation (NTRO)**
> *Leveraging satellite imagery to determine oil spills at sea, along with AIS data correlations to identify the vessel responsible for the spill.*

> **Status (merge of `pipeline-v1` into `main`):** the oilwatch pipeline, portal and their results (OW track) are **exploratory and not yet reviewed**. They deviate from frozen main decisions (seed grouping/ranking, AIS attribution before the E009 gate, geolocation claim, source_type rule); see PROJECT_CONTEXT.md "OW track". The gated E-chain (E001–E008c) in EXPERIMENT_LOG.md is the reviewed baseline.

This repository contains a complete, reproducible **forensic attribution pipeline** and an **analyst web console**:

```
Sentinel-1 SAR ─► oil segmentation ─► slick events ─► backward drift (where did the oil come from?)
                                                   └► forward test (would this ship's oil end up here?)
AIS (1,700 ships) + SAR ship detections + offshore platforms ─► ranked suspects + evidence flags
                                                   └► 72 h forecast (where does the oil go next?)
Everything ─► web console with a −48 h … +72 h time slider
```

**Ground rule for everyone working here:** rankings are *evidence of spatio-temporal consistency* between ship tracks and modelled oil drift, **never proof of culpability**. The tooling, wording and tests enforce this.

---

## Contents

1. [What the problem statement asks, and where it is solved](#1-what-the-problem-statement-asks-and-where-it-is-solved)
2. [Headline results](#2-headline-results)
3. [Repository layout](#3-repository-layout)
4. [Quick start](#4-quick-start)
5. [Data: what is in git, what to download](#5-data-what-is-in-git-what-to-download)
6. [The `oilwatch` pipeline, stage by stage](#6-the-oilwatch-pipeline-stage-by-stage)
7. [Methods in detail (with sources)](#7-methods-in-detail-with-sources)
8. [Validation: how we know it works (and where it does not)](#8-validation-how-we-know-it-works-and-where-it-does-not)
9. [The analyst portal](#9-the-analyst-portal)
10. [Configuration and adding a new incident](#10-configuration-and-adding-a-new-incident)
11. [Outputs and file formats](#11-outputs-and-file-formats)
12. [Engineering notes and gotchas (read before changing drift code)](#12-engineering-notes-and-gotchas-read-before-changing-drift-code)
13. [Testing](#13-testing)
14. [Contributing](#14-contributing)
15. [Known limitations and roadmap](#15-known-limitations-and-roadmap)
16. [Experiment history](#16-experiment-history)
17. [References](#17-references)

---

## 1. What the problem statement asks, and where it is solved

| PS 26143 asks for | Where it lives | Status |
|---|---|---|
| (a) Detect and characterise the slick, geometric properties | SegFormer-B2 segmentation (E005, frozen) on calibrated Sentinel-1 VV (E007); slick events + geodesic geometry (`oilwatch/seed.py`) | ✅ |
| (a) Age of the slick, if feasible | Interval from several cues: forward-drift release window, slick width diffusion, Okubo diffusion, persistence prior, SAR darkness ("oil colour" in radar) (`oilwatch/age.py`, `oilwatch/sar.py`) | ✅ (interval, not a point) |
| (b) Trace the slick back to origin point and time | 48 h OpenDrift backward ensemble, 27 members (`oilwatch/backward.py`) | ✅ |
| (b) Predict the future flow of the slick | 72 h OpenOil forecast with weathering, beaching, oil-type sensitivity (`oilwatch/forecast.py`) | ✅ |
| (c) Reconstruct AIS traffic around the origin window | Scene-wide NOAA AIS (1,715 vessels), interpolated positions (frozen V2.1 rule) | ✅ |
| (c) Filter irrelevant traffic | Explicit funnel: AIS in window → positioned → reached by the drifted oil → ranked | ✅ |
| (c) Score suspects (proximity, trajectory, behavioural anomalies) | Backward density + forward virtual-discharge test + fusion; AIS gaps, SAR "AIS-silent" ships, identity anomalies, platforms (`oilwatch/fusion.py`, `oilwatch/evidence.py`) | ✅ |
| Dark vessels | SAR CFAR ship detection matched to AIS and platforms (`oilwatch/sar.py`) | ✅ |
| Suitable visual interface | React + deck.gl console with time slider (`portal/`) | ✅ |
| (implicit) How accurate is it? | Twin experiments: 45 blind synthetic spills; calibrated decision rule (`oilwatch/twin.py`, `oilwatch/calibrate.py`) | ✅ |

---

## 2. Headline results

### Incident 001 (Gulf of Mexico, Sentinel-1A, 2023-01-03 00:01:42 UTC; reference: SkyTruth Cerulean slick 3775938)

- **22 slick events** from 49 detected fragments. 12 were attributed (the top slick potential with data coverage).
- **Filtering funnel per event:** 1,682 AIS vessels in the window → 1,591 with positions → 5–40 reached by the drifted oil → ranked.
- **Main reference slick (event 18): lead = SHELIA BORDELON.** Three independent lines agree:
  1. forward and backward drift both point to a release 1–2 h before the image;
  2. its AIS was silent during that window;
  3. the SAR image shows a ship at the slick while AIS was off, 120 m from its dead-reckoned position.

  This is still below the Fractions-Skill-Score "useful" threshold: **a lead, not a verdict**.
- **Only one high-confidence flag** (the twin-calibrated rule, §8.3): event 15, **MILLIE**. It is co-located with the **HOLSTEIN** platform (BOEM), and drift cannot separate the two.
- **SAR geolocation error measured** at 39 m median on 59 fixed targets (previously unknown).
- **Radar darkness:** the slicks look weathered, not fresh (VV damping 2–3.4 dB against a 6.4 dB training median).
- **Forecast 72 h:** no beaching. Surface fraction left after 72 h: light crude 2 %, medium 13 %, heavy 19 %. Surface oil moves north-west with the wind, dispersed oil south with the current.

### Measured accuracy (twin experiments, 45 blind cases, real AIS as distractors)

| | #1 | top 3 | top 5 | median release-time error |
|---|---|---|---|---|
| all spills | **67 %** | **89 %** | **98 %** | **0.5 h** |
| 0–3 h old | 67 % | 92 % | 92 % | 0.25 h |
| 3–12 h | 83 % | 100 % | 100 % | 0.5 h |
| 12–24 h | 83 % | 92 % | 100 % | 0.5 h |
| 24–40 h | 22 % | 67 % | 100 % | 1.4 h |

It is a strong **lead generator**. It is not a judge on its own: with the culprit's AIS hidden, some innocent ship usually still looks plausible. That is why the portal separates *ranked leads* from a rare *high-confidence flag* (≈5 % false-accusation rate on held-out twins, which fires for 14 % of true culprits). Details are in §8.

---

## 3. Repository layout

The repo has **two layers**: the team's frozen research experiments (E000–E007, E101, E102) and the `oilwatch` platform built on top of them. The platform never edits frozen files.

```
SIH26143/
├── README.md                      ← you are here
├── PROJECT_CONTEXT.md             ← project rules, environment, frozen parameters (read this second)
├── EXPERIMENT_LOG.md              ← every experiment: command, outputs, result, decision (newest first)
├── E008_SELECTION_POLICY.md       ← FROZEN E008 candidate policy v1 (no grouping, no ranking); the OW seed rule is oilwatch/OW_SEED_POLICY.md
├── SAR_DATA_AUDIT.md / MENTOR_DATASET_AUDIT.md
├── requirements-oilwatch.txt
│
├── run_hindcast_48h.py … score_ensemble_robustness.py   FROZEN physics/attribution scripts (hard-linked, never edit)
├── scripts/                       FROZEN experiment scripts: SAR audit, cache, E003–E007 ML, E101 V2.1, E102 motion
├── configs/
│   ├── ais_v2_1.json, source_type_v1.json     frozen configs
│   ├── pipeline_v1.json           all oilwatch parameters, each with its literature source
│   └── incidents/incident_001.json incident spec (inputs, T0, reference = post-hoc only)
│
├── oilwatch/                      THE PIPELINE (Python package)
│   ├── __main__.py                CLI: parity | run | twin
│   ├── config.py, manifest.py     incident spec, run folders, per-stage manifests (sha256, params, git, env)
│   ├── frozen.py                  imports the importable frozen modules (V2.1, E102) unchanged
│   ├── parity.py                  gate: reproduces the frozen outputs before anything new is trusted
│   ├── seed.py                    OW-E008: fragments → slick events → eligibility → selection
│   ├── context.py                 wind / current / Stokes at any point (external factors)
│   ├── sar.py                     tile-wise S1 processing: darkness, CFAR ships, AIS/platform matching, geolocation
│   ├── drift.py                   OpenDrift wrappers (frozen re-implementations + v3 engine + element-identity guard)
│   ├── backward.py                backward ensemble, age-dependent density score, funnel
│   ├── forward.py                 virtual-discharge test (window search, FSS)
│   ├── evidence.py                fresh-discharge geometry, AIS behaviour anomalies
│   ├── infrastructure.py          BOEM platforms, source_type rule
│   ├── age.py                     age cues and interval
│   ├── fusion.py                  forward–backward fusion, tiers, flags, v3.1 calibrated decision
│   ├── forecast.py                72 h OpenOil forecast with weathering
│   ├── report.py                  summary.json, report.md, maps, post-hoc reference comparison
│   ├── twin.py, calibrate.py      twin experiments and the calibrated decision rule
│   └── fetch.py                   open-data downloaders (Sentinel-1, NOAA AIS, BOEM), no login
├── portal/
│   ├── build_data.py              run → compact JSON for the web app
│   ├── api/main.py                FastAPI: serves app + data, lists runs, triggers pipeline/export
│   └── web/                       React 19 + Vite + MapLibre + deck.gl console
├── tests/test_oilwatch.py         parity gate, OpenDrift identity regression, unit tests
│
├── data/                          incident inputs (bulk raw data git-ignored, see §5)
├── results/                       FROZEN experiment outputs + E008_oilwatch_v3_scene/, twin_v1/
├── logs/                          one log per experiment/run
└── runs/                          (git-ignored) full pipeline runs: runs/<incident>/<run_id>/<stage>/
```

---

## 4. Quick start

### 4.1 Clone (Windows needs long paths)

```bash
git clone https://github.com/RishiiGamer2201/SIH26143.git
cd SIH26143
git config core.longpaths true          # Windows: Sentinel-1 SAFE paths exceed 260 chars
git checkout pipeline-v1                # until merged into main
```
If the first checkout failed on Windows, run `git restore --source=HEAD :/` after setting `core.longpaths`.

### 4.2 Python environment

```bash
python -m venv --system-site-packages .venv          # Python 3.11+
.venv/Scripts/python -m pip install -r requirements-oilwatch.txt   # Linux: .venv/bin/python
# Windows only, GDAL bindings (needed by the SAR stage):
.venv/Scripts/python -m pip install https://github.com/cgohlke/geospatial-wheels/releases/download/v2025.10.25/gdal-3.11.4-cp311-cp311-win_amd64.whl
```
The team's WSL conda envs (`opendrift`, `sih-ml`, see `PROJECT_CONTEXT.md`) also work.

### 4.3 Get the Sentinel-1 image (only needed for the SAR stage)

```bash
SAFE=data/incident_001/sentinel1/extracted/S1A_IW_GRDH_1SDV_20230103T000142_20230103T000207_046612_059620_257F.SAFE
python -m oilwatch.fetch s1 S1A_IW_GRDH_1SDV_20230103T000142_20230103T000207_046612_059620 --dest "$SAFE/measurement" \
  --safe-name-vv s1a-iw-grd-vv-20230103t000142-20230103t000207-046612-059620-001.tiff \
  --safe-name-vh s1a-iw-grd-vh-20230103t000142-20230103t000207-046612-059620-002.tiff
```
Everything else for Incident 001 is already in git (§5), including the 52 MB scene-wide AIS extract.

### 4.4 Run

```bash
python -m oilwatch parity incident_001            # ~40 s: must print EXACT/ULP for every table
python -m oilwatch parity incident_001 --full     # + re-runs OpenDrift (~4 min), bit-exact
python -m oilwatch run incident_001 --run-id myrun                # full pipeline (~45 min on a laptop)
python -m oilwatch run incident_001 --run-id myrun --stages fuse,report   # re-run a subset
python -m oilwatch twin incident_001 --n 48 --run-id twin_x      # twin experiments (~25 min)
python -m oilwatch.calibrate runs/incident_001/twin_x/twin       # calibrate the decision rule
python -m pytest -q tests/                                        # 8 tests (~1 min)
```

### 4.5 Portal

```bash
python portal/build_data.py incident_001 e008_v3_scene     # needs a local run folder (runs/ is git-ignored)
cd portal/web && npm install && npm run build && cd ../..
python -m uvicorn portal.api.main:app --host 127.0.0.1 --port 8000     # open http://127.0.0.1:8000
```
The exported data for run `e008_v3_scene` is committed under `portal/web/public/data/`, so `npm run dev` works right after cloning.

---

## 5. Data: what is in git, what to download

| Data | In git? | Source / how to get it |
|---|---|---|
| Zenodo Sentinel-1 oil-spill training set (Trujillo-Acatitla et al. 2024) | no (≈100 GB) | Zenodo 8346860 / 8253899 / 13761290; teammates' WSL machine |
| E002 training cache `data/sar/cache/v1/*.npy` | no (12 GB) | `scripts/sar_build_cache.py` |
| Incident 001 SAFE annotation, calibration, noise XML | yes | SAFE product |
| Incident 001 SAFE measurement TIFFs (VV/VH) | no (1 GB) | `python -m oilwatch.fetch s1 …` (Microsoft Planetary Computer, anonymous). **Verified identical to the original:** the labelled calibration check reproduces the frozen audit numbers exactly |
| CMEMS SMOC currents + Stokes, ERA5 wind (Incident 001) | yes | `data/incident_001/currents`, `data/incident_001/wind` (cover 2023-01-01 00:00 to 2023-01-07 06:00) |
| AIS, reference ROI extract (frozen, 162 MMSI) | yes | `data/incident_001/ais/ais_roi.parquet` (parity only) |
| **AIS, scene-wide** (1,715 MMSI, 2.2 M reports, 52 MB) | **yes** | `data/incident_001/ais/ais_scene.parquet`; rebuild: `python -m oilwatch.fetch ais 2023-01-01 2023-01-03 --dest data/ais/raw --bbox=-93,24.5,-87.5,30` |
| BOEM offshore platform structures | yes | `data/infrastructure/boem/`; `python -m oilwatch.fetch boem --dest data/infrastructure/boem` |
| E005 / E004 checkpoints | yes | `results/E005_segformer_b2/best.pt` (hash in FROZEN.json), `results/E004_resnet18_cls/best.pt` |
| E007 detected components (49 polygons) | yes | `results/E007_incident001_ml/incident001_predicted_components.geojson` |
| Cerulean reference slick and vessels | yes | **post-hoc validation only.** Never an input, never a tuning target |

Every downloader writes `fetch_manifest.json` with sha256 hashes next to the files.

---

## 6. The `oilwatch` pipeline, stage by stage

Each stage writes `runs/<incident>/<run_id>/<stage>/` plus a `manifest.json` containing the sha256 of inputs and outputs, all parameters, git state, library versions and runtime. Stages read only files, so any stage can be re-run on its own.

| # | stage | what it does | key outputs |
|---|---|---|---|
| 0 | `parity` | recomputes every frozen Incident 001 table from the frozen inputs and requires EXACT or ULP-equal matches; `--full` also re-runs OpenDrift | `parity_report.json` |
| 1 | `seed` | OW seed policy: merge detections into slick events, measure them, check wind validity and data coverage, select | `candidates.geojson/csv` |
| 2 | `context` | wind / current / Stokes at each event at T0 and over the prior 48 h | `context_t0.csv`, `context_history.parquet` |
| 3 | `sar` | radar darkness per event; scene-wide CFAR ship detection; matching to AIS, AIS-silent vessels and platforms; geolocation error | `darkness.csv`, `ship_detections.csv`, `dark_vessels_near_slicks.csv`, `geolocation.json` |
| 4 | `backward` | 27-member backward drift of all selected events in one ensemble; age-dependent density score per vessel; funnel | `backward_robust.csv`, `funnel.csv`, `cloud_summary.parquet`, `particles.parquet` |
| 5 | `forward` | every funnel vessel discharges virtual oil along its AIS track; drift to T0; best release window + FSS per slick | `forward_best.csv`, `forward_windows.parquet`, `virtual_oil_at_t0.parquet` |
| 6 | `evidence` | fresh-discharge geometry, AIS behaviour (gaps, speed jumps, MMSI validity), motion_state (frozen E102) + source_type (BOEM) | `geometry.csv`, `behaviour.csv`, `motion.csv` |
| 7 | `fuse` | forward–backward fusion, tiers, flags, v3.1 calibrated decision, age cues and interval | `suspects.csv`, `age_cues.csv`, `age_summary.csv` |
| 8 | `forecast` | 72 h OpenOil with weathering + coastline; oil-type sensitivity | `beaching.csv`, `mass_budget.csv`, `probability_grids.csv`, `centroid_track.csv` |
| 9 | `report` | `summary.json`, `report.md`, `map.png`, `forecast.png` | |
| 10 | `posthoc` | reference (Cerulean) comparison, the **only** place the reference is read | `posthoc.json` |

---

## 7. Methods in detail (with sources)

All numbers below are in `configs/pipeline_v1.json`, each with a `_why` field citing its source. They were fixed **before** the corresponding results were seen; any later change is logged as such in `EXPERIMENT_LOG.md`.

### 7.1 Detection (frozen, team experiments E000–E007)
- **SAR audit:** band 1 = VH, band 2 = VV, proven physically. Test-set leakage was quantified and a leakage-safe split built. Models use **VV only**, because train/test VH processing differs by 4.7 dB. See `SAR_DATA_AUDIT.md`.
- **E005 SegFormer-B2** (Trujillo data, 20 m grid): val IoU 0.754. One-shot test IoU 0.335 (full) and 0.493 (place-clean). Good on thin/linear slicks; misses broad dark slicks that are absent from training.
- **E007:** raw SAFE → σ0 calibration → GCP thin-plate-spline geocoding → sliding windows → 49 components. **E004 (classifier) must not gate E005**: it called the real slick a look-alike.

### 7.2 Slick events (OW seed policy, `oilwatch/OW_SEED_POLICY.md`; not the frozen E008 policy)
- **Merge:** fragments with a gap ≤ **1 km** become one event. This equals Cerulean's 500 m closing buffer; KSAT data has ~100 polygons per spill.
- **Wind window** for C-band oil visibility: valid **2.5–10 m/s**, marginal 1.5–2.5 or 10–12 m/s (Topouzelis 2008; Fingas & Brown 2018).
- **`slick_potential`** = area-weighted confidence × wind factor × (1 − e^(−A/0.1 km²)).
- **Eligible if:** area ≥ 23,000 m² (64 px, below which E005 misses 73 % of validation components), inside the forcing domain, and inside AIS coverage. The top 12 are selected. An analyst override is possible and logged.

### 7.3 Physics (v3 ensemble)
- **Model:** OpenDrift **OceanDrift**, transport only. Weathering is irreversible, so it is off in backward runs (Breivik et al. 2012).
- **Forcing:** CMEMS SMOC `uo+utide` with **separate** Stokes drift, so windage is **{1, 2, 3} %**. That gives 0.02 + Stokes ≈ 3.5 % total (Röhrs et al. 2018). The frozen 3–4 % + Stokes likely double-counts wave drift.
- **Horizontal diffusivity:** **{1, 3, 10} m²/s** (Okubo 1971; MEDSLIK-II; OpenDrift). It was 0 in the frozen runs.
- **Current uncertainty** {0, 0.05, 0.10} m/s; wind uncertainty 0.5 m/s. 48 h, dt 15 min, hourly output.
- **27 members = 9 simulations × 3 windages per element.** All events (or vessels) share one simulation per member; see §12 for the identity guard this requires.

### 7.4 Backward score (`backward.py`)
- **density(τ)** = mean over particles of exp(−d² / 2σ(τ)²), with **σ(τ)² = (1 km)² + (0.05 m/s · τ)²**.
  - At T0 a vessel must be within about 1 km of the oil. The kernel widens with age for model error; de Aguiar et al. 2023 report centroid errors of 2–3 km per day.
- **Vessel positions** use the frozen **V2.1** rule (interpolated AIS, E101).
- **`S_point`** = max over τ. **`S_weighted`** applies a persistence prior exp(−τ/12 h) (discharge slicks live ~2–12 h).
- Robust over the 27 members: median score, median rank, and top-1/3/5 fractions.
- **Funnel:** vessel passes if it is within 3σ(τ) of some particle and has ≥ 2 positioned hours.

### 7.5 Forward test (`forward.py`, after Longépé et al. 2015, MPB 101:826)
- Every funnel vessel "discharges" every 15 min along its own AIS track over T0−48 h…T0: 2 particles per windage, jittered 50 m, land-masked. The virtual oil is drifted to T0.
- **Window search:** every contiguous release window of 1/2/3/6/12 h.
  - precision = share of the window's virtual oil within 1 km of the slick;
  - recall = share of slick sample points with virtual oil within 1 km;
  - **F** = 2PR/(P+R), with members pooled (probability of presence).
- **Bitmask trick:** each slick point stores a 48-bit mask of the release hours that reached it, so every window is tested with one AND.
- **Skill:** Fractions Skill Score at 0.5/1/2/5 km. Useful if FSS ≥ 0.5 + f_O/2 (Simecek-Beatty & Lehr 2021).
- The best window is the **implied release time**. It also feeds the age estimate.

### 7.6 Fusion and decisions (`fusion.py`)
- **combined** = √(backward_rel · forward_rel) (the forward–backward averaging of Luo et al. 2024, MPB 207:116808).
- **Pre-registered v3 tiers** (kept visible): *strongly consistent / consistent / weak / not supported*. These turned out to be relative (§8.3).
- **v3.1 decision** (twin-calibrated): **high-confidence flag** only if fwd F ≥ 0.7 ∧ backward ≥ 0.7 ∧ ensemble support ≥ 50 % ∧ it is the best such vessel. Otherwise the vessel is a *lead*.
- **Flags** (never change a score):
  - AIS gap inside the release window;
  - **SAR sees it while its AIS is off**;
  - at the slick tip (fresh-discharge geometry, Cerulean-style head proximity + heading);
  - identity anomaly (non-standard MMSI, > 50 kn jumps);
  - release window at the 48 h boundary.
- `direction_agreement`: forward+backward / backward only / forward only / neither / forward not evaluated.

### 7.7 SAR evidence (`sar.py`)
- Tile-wise processing straight from the measurement TIFF, calibrated with the frozen `s1_grd_calibrate.sigma0`. No full-scene rasters are written (laptop friendly).
- **Darkness** ("oil colour" in radar): VV damping ratio (sea-ring median − slick median), edge contrast and texture.
  - Classes: strong/fresh-looking ≥ 6 dB (training median 6.36 dB); weak/weathered < 3 dB.
  - This is a heuristic. No validated SAR age formula exists.
- **Ships:** two-parameter CFAR (guard 11 px, background 41 px, k = 5σ, ≥ +8 dB over clutter, 4–3,000 px). Matching:
  - to AIS at T0 within 0.5 km (stationary) or 1.5 km (moving; allows the SAR azimuth Doppler shift);
  - to BOEM platforms within 0.5 km;
  - **AIS-silent:** dead-reckoned from a last report 30 min–3 h old, radius capped at 3 km;
  - otherwise it is a dark-vessel candidate.
- **Geolocation error** is measured on stationary matches: 39 m median, 156 m p90.

### 7.8 Source type (`infrastructure.py`)
- `motion_state` (frozen E102) and `source_type` are kept separate: stationary does **not** mean platform.
- **fixed_infrastructure:** stationary AIS within 0.5 km of a BOEM structure. Example: BIG FOOT, 103 m from WR 29 A (Chevron).
- **vessel:** AIS ship type code 20–89, or moving with no structure nearby.
- **unknown:** otherwise, for example DP drillships.
- BOEM coordinates are NAD27 and are converted to WGS84.

### 7.9 Age (`age.py`)
- Reported as an **interval**, from:
  - forward best window of the top supported suspect;
  - width diffusion t = ((W/4)² − (W₀/4)²)/(2K);
  - Okubo σ²[cm²] = 0.0108 t^2.34;
  - persistence 2–12 h;
  - SAR darkness class;
  - Bonn Agreement appearance code, when optical imagery exists.
- **Automatic age-conflict note** when the drift age is older than the appearance allows. That points to an old slick, or a young slick from a non-AIS source.

### 7.10 Forecast (`forecast.py`)
- OpenOil **with** evaporation, emulsification, dispersion and vertical mixing, plus the GSHHG coastline (stranding = beaching). Same transport ensemble, 72 h.
- **Oil type is unknown from SAR:** generic medium crude is central, with light/heavy crude as sensitivity.
- **SST is an assumption** (22 °C, January northern Gulf of Mexico): the forcing has no SST.

---

## 8. Validation: how we know it works (and where it does not)

### 8.1 Parity gate
`python -m oilwatch parity incident_001 --full` recomputes the frozen team outputs:
- The **physics is bit-exact**: the deterministic hindcast, and the 27-scenario ensemble (529,200 particle rows).
- The **scoring is ULP-equal**: every rank, name and bin identical; floats within 2e-13, which is numpy-build rounding (the frozen E102 code, imported unchanged, differs by 2e-16 too).

The pipeline refuses to write into `results/`.

### 8.2 Data verification
The downloaded Sentinel-1 VV reproduces `results/sar_audit/incident001_labelled_pol_check.json` exactly (sea −14.88 dB, contrast 2.67 dB, NaN fraction 0.0335). The scene AIS contains 100 % of the frozen ROI extract.

### 8.3 Twin experiments (`oilwatch/twin.py`, results in `results/twin_v1/`)
1. A real AIS vessel secretly discharges for 1–6 h, ending 0–40 h before T0.
2. Truth drift uses physics **outside** the attribution ensemble: windage 2.5 %, K 5 m²/s, its own random seed.
3. The slick is degraded like our product: +60 m wider, shifted 40 m.
4. Attribution runs blind against 1,715 real ships.
5. "Hidden culprit": the same case with the true ship removed.

**Findings:**
- The ranking skill in §2 holds up.
- The **pre-registered v3 tiers label an innocent ship "consistent" in 98–100 % of hidden-culprit cases**. Relative criteria always find someone in crowded water.
- An absolute rule was calibrated on half the cases and tested once on the other half (a second look after adding a margin feature is disclosed):
  - held-out **false accusation 5 %**;
  - fires for **14 %** of true culprits.
- The whole trade-off curve is in `results/twin_v1/operating_curve.csv` and `twin_skill.png`.
- **Limitation:** truth and model share the same forcing fields, so this is an upper bound with respect to forcing error.

### 8.4 Honesty rules we follow
- The reference is never an input.
- The official test set is never tuned on.
- Every method change becomes a new version, with the old one kept beside it.
- Withdrawn results are marked as withdrawn, not deleted from the log.

---

## 9. The analyst portal

`portal/` holds a React 19 + Vite + MapLibre (CARTO dark) + deck.gl console, plus a FastAPI backend.

- **Left panel:** slick events with size, wind window, radar colour, age and lead suspect. The high-confidence flag is shown in red.
- **Map:**
  - slick polygons;
  - **hindcast particles** (blue) for t < 0 and **forecast particles** (orange; solid = at surface) for t > 0;
  - AIS trips with trails;
  - SAR ship rings at T0 (red = no AIS, amber = AIS-silent);
  - BOEM platforms;
  - the selected suspect's **virtual oil at T0** (violet);
  - an optional Cerulean outline, labelled post-hoc.
- **Timeline** −48 h … +72 h with play. The best release window of the focused suspect is marked.
- **Right panel:** event facts (area, age, darkness, wind, beaching), the funnel bars, and suspect cards with evidence bars and flags.
- **"How accurate is this?":** twin KPIs, skill by age, leads vs. accusations, method.
- **Deep links:** `?event=18&focus=367655260&t=-1.5`, `?method=1`.
- **API (localhost only):** `GET /api/incidents`, `GET /api/incidents/{id}/runs`, `POST /api/incidents/{id}/runs/{run_id}?stages=…`, `POST …/export`, `GET /api/jobs`. See `portal/README.md`.

---

## 10. Configuration and adding a new incident

1. **Get the forcing** for the area and time: CMEMS SMOC (uo, vo, utide, vtide, vsdx, vsdy) and ERA5 10 m wind covering T0−48 h…T0+72 h. Convert to OpenDrift-readable NetCDF like `data/incident_001/currents/ocean_opendrift.nc`.
2. **Get AIS:** `python -m oilwatch.fetch ais <start> <end> --dest data/ais/raw --bbox=W,S,E,N` (US waters, NOAA). For India, use real AIS if available; the PS allows synthetic AIS otherwise.
3. **Get the Sentinel-1 scene:** `python -m oilwatch.fetch s1 <scene id without the 4-char suffix> --dest <SAFE>/measurement`. Run detection (E007 procedure) to produce a components GeoJSON.
4. **Write the spec** `configs/incidents/<id>.json`, copying `incident_001.json`: `t0_utc`, `inputs` (`detected_components`, `ocean_nc`, `wind_nc`, `ais_parquet`, `ais_scene_parquet`, `safe`, `platforms_dir`) and an optional `reference` (post-hoc only). `frozen_baseline` is only needed for parity.
5. **Run:** `python -m oilwatch run <id> --skip-parity --run-id first`. Parity applies only to Incident 001.
6. **Export** to the portal and open with `?incident=<id>`.

---

## 11. Outputs and file formats

### `fuse/suspects.csv`
One row per (event, vessel that passed the funnel). Key columns:

- **Identity:**
  - `cluster_id`, `MMSI`, `VesselName`
  - `suspect_rank`, `tier`, `v31_decision`, `direction_agreement`
- **Scores:**
  - backward: `bwd_point`, `bwd_weighted`, `bwd_rank_median`, `bwd_top3_fraction`, `bwd_peak_age_median`
  - forward: `fwd_F`, `fwd_precision`, `fwd_recall`, `fwd_member_support`, `fwd_window_start_h`, `fwd_window_end_h`
  - skill: `fss_0.5km` … `fss_5km`, `fss_skilful`
  - fused: `combined`
- **Context:**
  - `source_type`, `source_type_basis`, `motion_state`, `nearest_boem_structure`
  - `t0_lon/lat`, `dist_to_slick_end_km`, `geo_fresh_discharge`
  - `ais_gaps_ge_60min`, `ais_gap_intervals`
  - `sar_label`, `sar_dist_to_slick_km`
- **Flags:** `flag_*`

### Other outputs
- **`seed/candidates.geojson`:** every slick event with geometry, wind, potential, eligibility and reason.
- **`sar/ship_detections.csv`:** `row, col, lon, lat, size_px, peak_db, label ∈ {ais_stationary, ais_moving, platform_no_ais, ais_silent_at_t0, dark_vessel_candidate, no_ais_coverage}`, plus match fields.
- **`forecast/mass_budget.csv`:** fractions of oil at surface / evaporated / dispersed per oil type and lead hour.

---

## 12. Engineering notes and gotchas (read before changing drift code)

- **OpenDrift 1.14 returns BACKWARD runs in reversed element order** and exposes no element ID. When several events share one simulation, clouds silently swap between events. This invalidated an early OW-E008 run, which is withdrawn in the log.
  - `drift._element_order` recovers identity from seed positions: identity or reversal must match every element within 2e-5° (OpenDrift stores float32), otherwise it raises.
  - `tests/test_simulate_keeps_element_identity` guards this.
- **OpenDrift silently drops elements seeded on land** (coastal AIS, ports). Forward release points are land-masked with GSHHG *before* seeding.
- **The frozen ensemble uses the global `np.random.seed`** for OpenDrift perturbations and a separate `default_rng` for polygon sampling. Parity needs that exact call order.
- **Windows:** Python `open()` fails on deep SAFE paths (> 260 chars). `sar.short_safe` creates a directory junction under `%LOCALAPPDATA%\oilwatch`. Also set `git config core.longpaths true`.
- **Frozen files:** the root physics scripts are hard-linked into a Kaggle mirror on the WSL machine. Never edit them; write new versioned code instead.
- **Backward scoring cutoff:** vessels beyond 6σ(τ) of a cloud are skipped (their contribution is < 1.5e-8). This made scoring 60× faster, with identical funnel and ranks.
- **Do not use the reference** (Cerulean polygon, bbox or MMSI list) anywhere except `posthoc` and `parity`.

---

## 13. Testing

```bash
python -m pytest -q tests/                          # 8 tests, ~1 min
OILWATCH_SKIP_PARITY=1 python -m pytest -q tests/   # fast unit tests only
```
- **Parity gate:** all frozen tables reproduce.
- **OpenDrift element identity:** backward reversal and forward time series.
- **Maths:** FSS extremes, window enumeration, axis bearing, heading-vs-axis angle, age cues, AIS helpers.

---

## 14. Contributing

**Workflow**
- Branch from `pipeline-v1` (or `main` once merged): `feature/<short-name>`. Open a pull request, and get one review from the owner of the area:
  - **ML:** E-series scripts, detection;
  - **physics / attribution:** `oilwatch/drift|backward|forward|fusion`;
  - **SAR:** `oilwatch/sar`;
  - **portal:** `portal/`.

**Rules (same as `PROJECT_CONTEXT.md`)**
1. **Never edit frozen files** (`results/E00x…`, root physics scripts, `scripts/e00x…`). A changed method is a new versioned module or config section with a new output folder.
2. **Pre-register:** new thresholds go into `configs/pipeline_v1.json` with a `_why` (source) **before** you look at results. If you change one after seeing results, say so in the log.
3. **Log every experiment** in `EXPERIMENT_LOG.md` (newest first): date, command, outputs, result, decision. Withdrawn results stay listed as withdrawn.
4. **Reference = post-hoc only.** Never tune on Cerulean or on the official test set.
5. **Wording:** "consistent with", "lead", "evidence". Never "guilty" or "responsible".
6. **Parity must pass** (`python -m oilwatch parity incident_001`) before merging anything that touches `oilwatch/`.

**PR checklist**
- [ ] tests pass
- [ ] parity passes
- [ ] parameters in config with sources
- [ ] log entry added
- [ ] no writes to `results/` from code
- [ ] large data documented in §5 (GitHub limit 100 MB per file)

**Code style:** match the surrounding code (compact, explicit names, a docstring explaining *why*). Each stage takes (run, incident, params) and writes through `manifest.Stage`.

---

## 15. Known limitations and roadmap

**Limitations:**
- One incident only.
- Detection misses broad dark slicks.
- E007 strands are 2–3× too wide.
- No forcing-error term in the twins.
- The decision rule can rarely confirm guilt, by design.
- Co-located vessels and platforms cannot be separated by drift.
- The 48 h hindcast window is limited by the forcing start.
- SST is assumed in the forecast.

**Roadmap:**
- Drifter calibration of windage and diffusivity (Liu–Weisberg skill).
- A 72 h hindcast window.
- SAR ship fixes to bridge AIS gaps in the backward score.
- More twin cases, with forcing perturbation.
- Indian-waters demo with synthetic AIS (allowed by the PS).
- Incident 002.
- Multi-scale detection for broad slicks.
- Live Sentinel-1 ingestion.
- Portal: run launcher UI and evidence PDF export.

---

## 16. Experiment history

See `EXPERIMENT_LOG.md` for full detail.

- **E000:** SAR audit.
- **E001–E003:** env, cache, baseline.
- **E004:** classifier.
- **E005/E006:** SegFormer and test.
- **E007:** real-scene integration.
- **E101:** AIS V2.1.
- **E102:** motion state.
- **OW-E008:** oilwatch v3 attribution (branch `pipeline-v1`, exploratory, not reviewed).
- **OW-E011:** twin experiments and calibrated v3.1 rule (exploratory, not reviewed).

---

## 17. References

- Breivik Ø. et al. (2012) Advancing marine search and rescue / BAKTRAK backtracking, *Ocean Dynamics*.
- Röhrs J. et al. (2018) The effect of vertical mixing on the horizontal drift of oil spills, *Ocean Science* 14:1581.
- Okubo A. (1971) Oceanic diffusion diagrams, *Deep-Sea Research* 18:789.
- Accarino G. et al. (2025) MEDSLIK-II calibration with FSS, *Ecological Informatics* 103368.
- de Aguiar V. et al. (2023) OpenOil vs. SAR slicks, *Frontiers in Marine Science* 10:1122192.
- Longépé N. et al. (2015) Polluter identification with spaceborne radar imagery, AIS and forward drift modeling, *Marine Pollution Bulletin* 101(2):826.
- Luo et al. (2024) A new ship tracing technology from oil spills based on multi-source data, *Marine Pollution Bulletin* 207:116808.
- Simecek-Beatty D. & Lehr W. (2021) Fractions Skill Score for oil spill model verification, *Marine Pollution Bulletin*.
- Topouzelis K. (2008) Oil spill detection by SAR images, *Sensors* 8:6642.
- Fingas M. & Brown C. (2018) A review of oil spill remote sensing, *Sensors* 18:91.
- Bianchi F., Espeseth M. & Borch N. (2020) Large-scale detection and categorization of oil spills from SAR images (KSAT), arXiv:2006.13575.
- Quigley C., Johansson M. & Jones C. (2023) SAR damping ratio methods, *IEEE JSTARS*.
- Liu Y. & Weisberg R. (2011) Skill score for trajectory models, *JGR* 116:C09013.
- Bonn Agreement (2007) Oil Appearance Code.
- SkyTruth Cerulean methods and `cerulean-cloud` source (slick grouping, vessel scoring).
- Trujillo-Acatitla R. et al. (2024) Marine oil spill detection and segmentation in SAR data, *Marine Pollution Bulletin* 204:116549.

---

*Built for SIH 2026 PS 26143. Rankings are evidence of spatio-temporal consistency, never proof of culpability.*
