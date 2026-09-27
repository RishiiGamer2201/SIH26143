# Experiment Log

Newest first. Every entry: date, what ran, exact command, outputs, result, decision.

---

## 2026-09-28 · E007b — Model-only structure of the E007 components (descriptive; prepares E008 selection)

**Run.** `python scripts/e007b_component_structure.py model_only`, then `... posthoc`. Logs `logs/E007b_{model_only,posthoc}.log`. Outputs in `results/E007b_component_structure/`.
- `model_only` reads only the frozen E007 components GeoJSON (its `in_roi` flag is dropped on load), the x4 VV raster and the mask geotransform. It never opens Cerulean.
- Distances are edge-to-edge in the E007 scene-centred LAEA.
- Single-linkage gaps were **predeclared** at 250 / 500 / 1000 / 2000 / 5000 m and are descriptive only. Orientation is used only where elongation ≥ 2 (predeclared).
- `posthoc` runs afterwards, maps E007's reporting-ROI flag onto the clusters, and writes a separate file. Nothing feeds back.

**Model-only results.**
- 49 components; 23 are elongated (≥ 2). The median nearest-component gap is 286 m; 31 of 49 components have a neighbour within 500 m.

| gap G | 250 m | 500 m | 1 km | 2 km | 5 km |
|---|---|---|---|---|---|
| clusters (singletons) | 38 (29) | 30 (18) | 22 (10) | 13 (5) | 10 (2) |

- At G = 2 km, the largest clusters by area:
  - c0: northern group at −89.12°, 28.18°N; 10 components, 12.4 km², orientation coherence 0.88.
  - c1: 4.9 km² single linear feature, 13.3 km long.
  - c2: group at −90.44°, 27.12°N; 13 components, 2.70 km², extent 7.8 km, **coherence 0.99, isolated by 18.6 km**.
  - c3: 1.7 km² near −88.87°.
  - c4: 1.5 km², blocky, near −90.57°, 27.36°N.
- The northern features (c0, c1) are long, very dark and linear, with bright point targets at their heads (`model_only_clusters.png`). They are **unverified** (no reference); visually they are not obvious noise.

**Post hoc (ROI flag only).**
- At G ≥ 2 km the 13 reporting-ROI components form exactly one cluster, with no outside members. By area it ranks 3rd at 2 km and 2nd at 5 km.
- At G ≤ 1 km they split into 3–8 clusters; the internal fragment gaps are about 1.06–1.13 km.

**Answer.** A coherent, isolated, orientation-consistent model-only cluster exists for the Incident 001 region at G ≥ 2 km. However, model-only features do not single it out: larger coherent clusters exist, and `incident.json` offers no location cue independent of Cerulean. Decision: see `E008_SELECTION_POLICY.md` (Mode A analyst-assisted vs Mode B autonomous multi-candidate); awaiting review. No OpenDrift / AIS run.

---

## 2026-09-28 · E007 — Incident 001 real Sentinel-1: calibrated/geocoded VV → frozen E004 + E005 → mask → polygons (no physics)

**Run.** `conda activate sih-ml; python scripts/e007_incident001_ml.py {preprocess|infer|vector}`. Logs `logs/E007_{preprocess,infer,vector}.log`, outputs `results/E007_incident001_ml/`. Rasters are in `rasters/` (git-ignored).
- Input: `S1A_IW_GRDH_1SDV_20230103T000142_…_257F.SAFE`, VV only, IPF 003.52. The SAFE is read-only.

**Preprocessing** (training chain, SAR_DATA_AUDIT §9.7):
- Calibration: σ0 = (DN² − noiseRangeLut·noiseAzimuthLut) / sigmaNought². LUTs are interpolated bilinearly (along range per vector, then linearly between vector lines).
  - Matches the reference `s1_grd_calibrate.sigma0()` on a 512² scene-centre window to 6e-8 relative.
- Nodata: DN = 0 is invalid (2.8% of the radar frame). σ0 ≤ 0 after noise subtraction (0.16% of valid) stays valid, floored at 1e-13 (−130 dB). This matches the training data's valid ~−128 dB pixels; the −40 dB clip makes the floor moot.
- Geocoding: GDAL warp, thin-plate spline on the 210 annotation GCPs (ellipsoid heights ~0 m, ocean, no DEM), bilinear in linear power, NaN excluded. Absolute geolocation error of the GCP/TPS product has not been measured. Target is EPSG:4326 at the training pixel 8.983152841195208e-5°, with origin and size on multiples of 4 px so the ×2/×4 grids nest.
  - Native: 21,452 × 32,148 px.
  - ×2 (E005): 10,726 × 16,074 px, 1.797e-4°, **17.8 m E-W × 19.9 m N-S at 27.1°N**.
  - ×4 (E004): 5,363 × 8,037 px, 35.6 × 39.8 m.
  - Bounds: W −90.768, E −87.880, S 26.445, N 28.372.
- Downsampling: linear-power block mean (block valid if ≥ 50% valid), dB, clip [−40, 5], (dB + 40)/45, float16, then E005/E004 train norm. No speckle filter (as in training).

**Sanity check before ML.** Reporting ROI = Cerulean bbox ± 0.1° (1,298 × 1,661 px).

| VV dB (valid) | min | p01 | p10 | median | p90 | p99 | max | valid | clip <−40 / >+5 |
|---|---|---|---|---|---|---|---|---|---|
| ROI | −30.5 | −19.5 | −17.2 | **−14.8** | −12.5 | −10.8 | 17.4 | 97.1% | 0 / 5e-5 |
| scene (every 2nd px) | −130 | −25.2 | −22.4 | −19.1 | −15.1 | −12.3 | 30.6 | 70.1% | 5e-6 / 9e-6 |
| train tiles (all, clipped) | −40 | −35.1 | −26.2 | −20.1 | −13.8 | −6.1 | 5 | | |
| train oil tiles (clipped) | −40 | −28.6 | −23.8 | −19.8 | −16.2 | −13.1 | 5 | | |

The scene matches training. The **ROI is bright**: its window medians (−14.2 to −16.1 dB) sit at the 93rd–99th percentile of train oil-tile medians. That is a domain-shift sign (bright sea around the slick). Nothing was changed because of it. Figure: `preprocess_vv.png`.

**Inference** (frozen, hash-verified):
- E005: `best.pt` sha256 `1240a10a…9300b` = FROZEN.json, epoch 9, threshold 0.60, bf16.
- E004: sha256 `a05fb37e…68e6e4`, matches git HEAD, epoch 8.
- Windows: 1024² on the ×2 grid, stride 512 (50% overlap), 20 × 31 grid, 543 windows run (windows with no valid pixel skipped). Scene padded bottom/right with invalid pixels.
- **Probabilities blended** as Σ w·p / Σ w with a separable tent weight (1/512 … 1 per axis); 0.60 is applied once to the blended map.
- E004 (512² ×4 windows, same footprints) is reported only, **not used for gating**.
- Runtime: 40 s inference, 63 s stage total. Peak VRAM 3.08 GB allocated / 4.62 GB reserved.
- Blended probability over the valid scene: median 0.0010, p99 0.0015. Only 0.061% of valid scene pixels (73,497 px) exceed the frozen threshold, but the scene-level false-positive burden remains substantial because those pixels form 49 components totalling ~25.9 km², including several large unverified detections.

**Failure-regime check (the 20 windows overlapping the ROI).**
- Predicted oil fraction per window is 0–1.06%; blended fraction ≤ 0.81%. Window VV medians are −14.2 to −16.1 dB.
- This is the **thin-linear regime** where E005 worked, not the broad-dark regime of E006.
- E004 on the four windows that contain the reference slick ((iy,ix) = (12,2), (12,3), (13,2), (13,3)): p(oil) 0.12–0.34, argmax **lookalike** (p 0.39–0.50). Three ROI windows south/east of the slick ((13,5), (14,4), (14,5); no reference slick inside, 0.3% predicted each) give p(oil) 0.69–0.94. The rest are no_oil.
- **Negative result (keep):** classifier gating would have suppressed the core of this slick. Therefore **E004 MUST NOT gate E005 segmentation in E008.** E004 is not modified or retrained here.

**Vectorisation.** Raw 0.60 mask, 4-connected components (as E005/E006), `gdal.Polygonize`, no filtering.
- Area and perimeter are geodesic on WGS84 (OGR GeodesicArea / GeodesicLength). Axes and orientation come from the minimum rotated rectangle in a scene-centred LAEA, corrected to true north.
- **Validation of the area method:** our geodesic area of the Cerulean polygon is 1,243,162.73 m², identical to Cerulean's reported `area_m2`. Perimeter (28,192.68 m) and Polsby-Popper (0.01965) also match. This validates our AREA calculation, not our raster geolocation accuracy.
- Scene: **49 components**, 25.9 km² total. By area: 9 < 0.01 km², 20 at 0.01–0.1, 14 at 0.1–1, 6 ≥ 1 km².
  - The largest (9.7, 4.9, 2.1, 1.65 km²) lie at 28.15–28.22°N, −89.19 to −88.87°, in the northern part of the scene, > 100 km from the incident area. Next is 1.26 km² at −90.565 / 27.354, north of the ROI. Unverified detections (no reference).
- ROI: **13 components**, 2.70 km² union. The two main strands:
  - #37: 1.128 km², centroid −90.4231 / 27.1261, major axis 2.95 km, width proxy 382 m, orientation 87.7°, elongation 3.4.
  - #40: 0.840 km², centroid −90.4566 / 27.1091, major 3.90 km, width 215 m, 82.2°, elongation 9.2.
  - Plus 11 smaller pieces (0.0004–0.28 km²). All are in `predicted_components.csv` / `incident001_predicted_components.geojson`.
- Orientation of small, blocky components is quantised to the pixel grid (0°/90°/180°); treat orientation as meaningful only for elongated components.

**Post-hoc Cerulean comparison** (reference only; it did not crop, select or alter anything):

| | value |
|---|---|
| predicted area (ROI union) / reference area | 2.70 / 1.24 km² |
| intersection / union | 1.00 / 2.94 km² |
| **IoU** | **0.340** |
| reference covered by prediction | 80.5% |
| prediction inside reference | 37.1% |
| centroid distance (union vs reference) | 938 m (component #37: 827 m) |
| major axis pred / ref (min rotated rect) | 7.76 / 9.43 km (Cerulean `length` 9.54 km) |
| orientation pred / ref | 81.8° / 76.8° (Δ 5.0°) |

- The prediction follows both reference strands (`incident001_ml_result.png`) but is wider. Reference width proxy is 132 m; the main predicted strands are 215–382 m. This explains the IoU despite 80% reference coverage.
- Several small predicted pieces sit between or beside the strands.

**Domain-shift / failure signs.**
1. The ROI sea is brighter than nearly all training oil tiles (93–99th percentile), yet the slick is detected.
2. Predicted strands are 2–3× wider than Cerulean's polygon. This could be label-style differences (Trujillo masks) or the 20 m grid plus probability smoothing; not separable here.
3. E004 calls the slick-core windows lookalike.
4. 36 components outside the ROI are unverified; the largest cluster is in the northern part of the scene.
5. Geocoding is GCP thin-plate spline without DEM. Absolute geolocation error of the GCP/TPS product has not been measured.

**Interpretation.** Successful real-scene pipeline integration with partial/over-segmented Incident 001 recovery (post-hoc IoU 0.340, 80.5% of the reference covered, 37.1% of the prediction inside it, predicted ROI union 2.70 km² vs reference 1.243 km², centroid offset 938 m, strands ~2–3× wider). Not evidence of robust scene-wide autonomous segmentation.

Decision (reviewed 2026-09-28): E007 approved as an INTEGRATION milestone and FROZEN (`results/E007_incident001_ml/FROZEN.json`; `python scripts/e007_incident001_ml.py verify`). `preprocess_report.json` geocoding text corrected (no unmeasured error figure); original in `superseded/`. Nothing was tuned to Cerulean. E008 needs a reference-independent rule for which predicted geometry seeds physics: see `E008_SELECTION_POLICY.md`. No OpenDrift / AIS run.

---

## 2026-09-28 · E006 — One-shot official test of the FROZEN E005 SegFormer (evaluation only)

**Freeze.** `python scripts/e006_segformer_test.py freeze` wrote `results/E005_segformer_b2/FROZEN.json`. It holds SHA-256 of 10 E005 artefacts, the split, the cache meta, `vv20`/`lab20` and `e005_segformer.py`, plus epoch 9, threshold 0.60, the architecture, split identity, val metrics and git state (HEAD `5c16912`).
- `best.pt` sha256 `1240a10a184d90a156ddcacd4f72469cca94d33710b4989d1eed1818679b300b`.
- Before freezing, the val_metrics.json `pixel_area_note` "20 m cache pixels (~0.0004 km2 each)" was corrected. It was wrong: the grid is EPSG:4326 degrees. The original is in `superseded/`; the metrics are unchanged.

**Run.** `conda activate sih-ml; python scripts/e006_segformer_test.py test` (log `logs/E006_test.log`, outputs `results/E006_segformer_test/`). The script:
1. Verifies all hashes.
2. Re-runs val and requires E005's metrics (diff 0.0) and the component table (identical) to reproduce.
3. Only then loads test and scores the raw mask at the frozen 0.60 (0.5 as reference).
4. Re-verifies the hashes.

No training, sweep, selection or post-processing. Runtime 83 s (val re-check 39 s, test 35 s). Peak VRAM 3.02 GB allocated / 4.53 GB reserved.

**Test views** (3 all-nodata no_oil tiles excluded: 00005, 00087, 00099):

| view | tiles (oil / LA / NoOil) | IoU | Dice | P | R | IoU / Dice @0.5 (ref) |
|---|---|---|---|---|---|---|
| full | 447 (150/150/147) | **0.335** | **0.502** | 0.888 | 0.350 | 0.346 / 0.514 |
| acquisition-clean | 385 (88/150/147) | 0.257 | 0.409 | 0.876 | 0.267 | 0.269 / 0.424 |
| place-clean | 212 (37/110/65) | 0.493 | 0.661 | 0.864 | 0.535 | 0.525 / 0.688 |

Val was IoU 0.754 / Dice 0.860.

**Oil tiles @0.60:**

| view | n | IoU / Dice / P / R (pixel) | tile Dice mean / median | tile recall mean | any TP | missed entirely |
|---|---|---|---|---|---|---|
| full | 150 | 0.336 / 0.503 / 0.892 / 0.350 | 0.697 / 0.854 | 0.727 | 142 | 8 |
| acq-clean | 88 | 0.257 / 0.410 / 0.883 / 0.267 | 0.640 / 0.832 | 0.669 | 80 | 8 |
| place-clean | 37 | 0.495 / 0.662 / 0.870 / 0.535 | 0.730 / 0.830 | 0.767 | 37 | 0 |

**Negative tiles @0.60** (acq-clean = full, because every negative test tile is acquisition-clean):

| | any predicted oil | mean FP frac | p95 | max | false comps (per tile) | largest false comp |
|---|---|---|---|---|---|---|
| LookAlike full (150) | 8.0% | 5.7e-5 | 1.8e-4 | 0.0027 | 78 (0.52) | 1,174 px |
| LookAlike place-clean (110) | 8.2% | 5.4e-5 | 1.8e-4 | 0.0027 | 56 (0.51) | 1,174 px |
| NoOil full (147) | 4.8% | 1.2e-4 | 0 | 0.0092 | 23 (0.16) | 3,092 px |
| NoOil place-clean (65) | 3.1% | 8.7e-5 | 0 | 0.0056 | 4 (0.06) | 3,092 px |

Look-alike FP is lower than on val (22% of tiles, mean 1.2e-3).

**GT component recall @0.60, full test** (4-connected, same as E005; all views in `test_component_recall_by_size.csv`):

| size (px) | 1 | 2–4 | 5–16 | 17–64 | 65–256 | 257–1024 | >1024 |
|---|---|---|---|---|---|---|---|
| n | 1,311 | 917 | 1,275 | 1,001 | 584 | 279 | 389 |
| missed | 1,136 | 813 | 1,121 | 812 | 361 | 77 | 49 |
| miss rate | 0.87 | 0.89 | 0.88 | 0.81 | 0.62 | 0.28 | 0.13 |
| share of GT px | 0.01% | 0.02% | 0.07% | 0.21% | 0.48% | 0.92% | 98.3% |

**Main failure mode: large, broad slicks.**
- 10 of 150 oil tiles hold 50% of all test FN, and 21 tiles hold 80%. Each has 25–80% of the tile labelled oil, predicted almost empty.
- All 8 complete misses are such tiles (GT 220k–574k px; 00015, 00004, 00059, 00068, 00040, 00061, 00127, 00012).
- By tile GT-size quartile, median tile Dice is Q1 0.83, Q2 0.89, Q3 0.91, **Q4 0.08**. Q4 pixel recall is 0.16.
- Without the 10 largest-FN tiles, oil-pixel IoU is 0.495 (diagnostic only, not a result).
- The dominant test failure is strongly associated with a label-coverage distribution shift: broad oil masks are largely absent from train/validation. Region, annotation style and scene-intensity shift may also contribute and have not been disentangled. Oil-tile coverage (`pos_frac_20`):

  | split | median | tiles > 20% | tiles > 40% |
  |---|---|---|---|
  | train | 1.6% | 3 of 844 | 0 |
  | val | 2.3% | 0 | 0 |
  | test | 4.8% | 25 of 150 | 8 |

  Val could not expose this failure. The missed tiles are broad, low-contrast dark areas with scene-wide VV medians of −23 to −30 dB (`test_examples.png`).
- Thin and linear slicks are segmented well (tile Dice 0.82–0.92 in the figure).

**Other failure modes.**
- Tiny GT fragments (≤ 64 px) are missed 81–89% of the time. They hold 0.3% of GT px.
- LookAlike FPs are thin streaks and small bright-edged blobs; the largest is 1,174 px.
- NoOil FPs are coastal / land-edge dark features (00140, 00093, 00059).

**Contamination effect.** At 0.60: acq-clean vs full is IoU −0.078 / Dice −0.093. Place-clean vs full is IoU +0.158 / Dice +0.159.
- The direction does **not** measure leakage. The large missed slicks are almost all acquisition-clean but not place-clean, so the view deltas mainly reflect which view contains those tiles, plus a different class mix.
- The full score is **not** an unbiased generalisation estimate, and neither clean view is either (n oil 88 / 37).

Decision: E006 reviewed and **accepted as the frozen baseline test result** (2026-09-28). E005 stays frozen and unchanged (no threshold change, no post-processing, no retraining based on test failures).
The large-slick failure matters for E007: an incident slick may fill a whole tile. Any remedy would be a new, independently specified experiment tuned on dev/val only, never on these test results.

---

## 2026-09-27 · E005 — SegFormer MiT-B2 binary oil segmentation, Trujillo only (train/val; test NOT run)

**Run.** `conda activate sih-ml; cd scripts; python e005_segformer.py {tests|train|validate}`.
- Outputs: `results/E005_segformer_b2/`.
- Logs: `logs/E005_{tests,train,validate}.log`.

**Setup.**
- Model: `smp.Segformer("mit_b2", encoder_weights="imagenet", in_channels=2, classes=1)`, 24.72 M params (encoder 24.19 M). MiT-B2 is the moderate-capacity baseline for the RTX 5070 Ti 16 GB.
- 2-channel adaptation (verified exactly): smp `patch_first_conv` sets `encoder.patch_embed1.proj` [64,2,7,7] to the pretrained RGB weights for channels 0–1, ×3/2.
- Data: `vv20`/`lab20` cache, 1024² tiles, `splits_v1_acq` roles train (1941: oil 844 / no_oil 550 / lookalike 547) and val (477: 211 / 131 / 135). The split is unchanged and asserted identical; test rows are never loaded.
- Input: VV standardised with train valid-pixel stats (μ 0.4433, σ 0.1206 on the cache scale; invalid → 0) + validity. Target: `lab20 == 1`. Invalid pixels are excluded from BCE and Dice.
- Loss: 0.5·BCE + 0.5·soft Dice (batch-global).
  - Train pixel imbalance: oil 1.046% (20.76 M / 1,984 M valid px); raw pos_weight 94.6; **used pos_weight = 5.0** (cap).
- Optimiser: AdamW lr 6e-5, wd 0.01, 1-epoch linear warm-up then poly(1.0). bf16, seed 26143.
- Batch: 12 (from the probe). 2000 crops/epoch (167 steps), max 40 epochs, patience 10.
- Augmentation: h/v flip, rot90, ±1.5 dB gain. **No x-scale.**
- Checkpoint: val oil IoU at a FIXED 0.5 threshold, on full 1024² tiles.

**Pre-training tests** (`pretraining_tests.json`, all passed):
- A. 512² → [2,1,512,512] bf16 and 1024² → [1,1,1024,1024], finite.
- B. Invalid pixels change the loss by exactly 0.0 and get gradient exactly 0.0.
- C. VV / validity / target transforms are identical; invalid pixels stay 0; gain within ±1.48 dB.
- D. 12 fixed oil crops (lr 3e-4, no augmentation) reach **Dice 0.954 at step 50** (IoU 0.913, P 0.953, R 0.955). Dice against a shifted target peaks at 0 px in x and y (±1 px: 0.946–0.948; ±8 px: 0.78–0.80), so the output is aligned.
- E. VRAM probe (512², AdamW, bf16, clean baseline 0.05 GB):

  | batch | peak alloc (GB) | peak reserved (GB) | img/s |
  |---|---|---|---|
  | 2 | 1.85 | 1.89 | 19.5 |
  | 4 | 3.18 | 3.44 | 23.6 |
  | 6 | 4.52 | 4.95 | 25.1 |
  | 8 | 5.84 | 6.39 | 26.4 |
  | 12 | 8.51 | 9.39 | 26.8 |
  | 16 | 11.18 | 12.34 | 27.1 |

  Rule: largest batch with ≤ 12 GB reserved, so batch 12. (A first attempt was invalid: test A's graph was still held and inflated the probe by ~6 GB. Fixed with `no_grad` + del.)

**Sampling.** 38,076 crops over 19 epochs:

| crop type | count | share | mean oil fraction | mean valid fraction |
|---|---|---|---|---|
| oil-positive | 19,109 | 50.19% | 5.19% | 99.3% |
| LookAlike | 9,516 | 24.99% | 0 | 98.6% |
| NoOil | 5,648 | 14.83% | 0 | 97.9% |
| oil-scene background | 3,803 | 9.99% | 0 | 97.8% |

- Oil-scene background crops come from 41,848 oil-free 512 windows in 642/844 oil tiles; the fallback was used 0 times.
- Sampled-crop oil fraction is 2.61%, vs 1.05% of train pixels.

**Training.**
- Early stop after 19 epochs; **best epoch 9**.
- Runtime: 31.5 min train (≈ 76 s train + 23 s full-tile val per epoch); 32.2 min including loading; validation stage 76 s.
- Peak VRAM 8.51 GB allocated / 12.20 GB reserved. Checkpoint 94.4 MB.
- Val IoU@0.5 by epoch: 0.32, 0.63, 0.54, 0.60, 0.56, 0.69, 0.73, 0.71, **0.753**, 0.71, 0.72, 0.68, 0.73, 0.71, 0.66, 0.66, 0.74, 0.73, 0.54. The metric is noisy: the P/R balance swings between epochs (`curves.png`).

**Validation** (val only; full 1024² tiles; the threshold is swept after the checkpoint is frozen and then frozen itself):
- The sweep 0.10–0.90 (`val_threshold_sweep.csv`) is flat: Dice 0.830–0.860. Max Dice at **0.60, now frozen** (`frozen_threshold.json`).

| | IoU | Dice | P | R |
|---|---|---|---|---|
| global pixel @0.5 | 0.7530 | 0.8591 | 0.8496 | 0.8688 |
| **global pixel @0.60** | **0.7539** | **0.8597** | **0.8635** | **0.8559** |
| oil tiles only @0.60 (n 211) | 0.7680 | 0.8688 | 0.8820 | 0.8559 |

- Oil tiles @0.60:
  - tile Dice mean 0.841, median 0.873;
  - 210/211 tiles have some TP, 1 missed entirely (00601), 4 have Dice < 0.5;
  - 363 false components, largest 15,870 px;
  - 2,641 of 4,790 GT components missed, but 2,393 of those are ≤ 64 px. The missed components hold 0.98% of GT oil pixels. Only 3 of 474 components > 1,024 px were missed (`val_gt_component_recall_by_size.csv`).
- Negative tiles @0.60:

  | | any predicted oil | mean FP fraction | p95 FP fraction | max FP fraction | false components (per tile) | largest false component |
  |---|---|---|---|---|---|---|
  | LookAlike (135) | 22.2% of tiles | 0.00123 | 0.00514 | 0.0479 | 346 (2.56) | 25,426 px |
  | NoOil (131) | 5.3% of tiles | 1.1e-5 | 9.5e-7 | 8.1e-4 | 26 (0.20) | 508 px |

  16/135 LookAlike tiles have FP > 0.1%.

**Figure.** `val_examples.png`: rule-based, seeded selection. Columns: VV | GT | probability | prediction | overlay. Rows: good, small, large, worst FN, worst FP, random LookAlike, random NoOil.

**Failure modes.**
1. Thin, faint linear slicks are missed: 00601 (recall 0) and 00600 (0.16).
2. Partial segmentation of a large slick inside a heterogeneous dark scene (00526, recall 0.17).
3. Look-alike false positives on dark streaky natural features (00391, 00370: 25–70 small components).
4. False positives next to a no-data swath edge (00337, 53% valid, FP 4.8%).
5. Tiny GT fragments (≤ 64 px) are mostly missed (73% miss rate; all ≤ 64 px fragments together hold 0.5% of GT pixels).
6. The epoch-to-epoch val metric is unstable, and val was used for both checkpoint and threshold selection, so val numbers are optimistic for test.

**Official test:** not evaluated here; one-shot test in E006 (2026-09-28) with this frozen checkpoint and threshold.

---

## 2026-09-27 · E005-A — Mentor dataset audit (`shuban220409/combinedoilspilldataset`), no training

Full report: `MENTOR_DATASET_AUDIT.md`. Scope: Trujillo + "M4D" only (MADOS deferred, DARTIS unused).
- Kaggle side: private CPU kernel `rishiikumarsingh/sih26143-e005a-mentor-dataset-audit` (`scripts/kaggle/e005a_audit/`, ~9 min).
  It SHA-256 hashed the Trujillo subset on the mount and zipped the M4D folder. No 108 GB download.
- Local side: `conda activate sih-ml; python scripts/e005a_audit.py trujillo && python scripts/e005a_audit.py m4d && python scripts/e005a_audit.py report`
  (about 40 s total). Outputs go to `data/segformer_combined/{manifest.parquet, audit/, m4d_raw/, kernel_out/}`.

Results:
- **Trujillo = exact duplicate** of `data/sar/extracted/`: 6,040/6,040 SHA-256 matches, 0 mismatches, 0 unmatched.
  No second copy stored; the manifest points to local files. 2 exact duplicate image pairs (00007=01339, 00356=00357), each within one acq group.
- **"M4D" is mislabelled.** It is the Oil Spill Drone dataset (De Kerf et al., Sci Data 2024; 1,268 images).
  - Drone RGB photographs of a port at camera resolutions (1920×1080 ×1150, …, 5184×3888). Not Sentinel-1, not M4D.
  - The uploader's 1,189 / 5-class SAR description refers to data that is not in the upload.
- **M4D format and masks.** 1,268 clean pairs, 0 corrupt. Real 3-channel colour (R=G=B on 1.2% of pixels).
  Exactly 4 mask colours: 255,0,124 oil 32.2%; 255,204,51 other (ships/quays/land) 38.0%; 51,221,255 water 25.5%; 0,0,0 gaps 4.4%.
  Oil appears only in the OilSpill folder. The "LookAlikes" folder has no SAR look-alikes.
- **M4D ID collisions and duplicates.**
  - 11 ID collisions (`Oil`/`Oil_2` N=1–11): all different images (5 of them cross splits), so not leakage.
  - 1 exact duplicate image pair.
  - 17 identical masks, all near-uniform whole-frame labels.
  - **312 cross-split near-duplicate pairs** (NCC ≥ 0.97 at 128×72; consecutive video frames). The uploader split leaks.
- **Overlap with Trujillo:** none.

Decision: **approved and frozen 2026-09-27**. Also frozen: the uploader's M4D split is not used, and MADOS stays separate.
- E005 = **Trujillo only**, binary oil mask.
- Input VV + validity at 20 m (the E004 representation), `acq` split unchanged.
- SegFormer MiT-B2 via smp (`in_channels=2` verified to build).
- "M4D" excluded: modality mismatch with the Sentinel-1 deployment target.

---

## 2026-09-26 · E003 — Feature/statistical baseline + shortcut probes (CPU)

`conda activate sih-ml; python scripts/e003_feature_baseline.py` → `results/E003_feature_baseline/{features.parquet, metrics.json, metrics_table.csv}`, log `logs/E003_feature_baseline.log` (42 s, 2.6 GB RSS).
Split `acq`: train 1941 (oil 844 / no_oil 550 / lookalike 547), val 477 (211/131/135). HistGradientBoosting defaults,
balanced weights, seed 26143, **no tuning**. 17 image features from the 40 m cache (VV dB percentiles, mean/std, linear CV,
dark/bright fractions, mean gradient, valid fraction). Test scored only for the two fixed `_test` models; nothing selected on test.

| model | eval | n | acc | macro-F1 | R no_oil | R lookalike | R oil |
|---|---|---|---|---|---|---|---|
| image stats | val (= val place-clean) | 477 | 0.841 | **0.833** | 0.840 | 0.778 | 0.882 |
| image stats | test − 3 nodata | 447 | 0.749 | **0.744** | 0.986 | 0.647 | 0.620 |
| image stats | test acq-clean | 385 | 0.748 | 0.724 | 0.986 | 0.647 | 0.523 |
| image stats | test place-clean | 212 | 0.764 | 0.736 | 0.985 | 0.709 | 0.541 |
| metadata only (lat, lon, px size, h, w, valid) | val | 477 | 0.472 | 0.467 | 0.389 | 0.756 | 0.341 |
| metadata only | test − 3 nodata / acq-clean / place-clean | | 0.374 / 0.366 / 0.326 | 0.356 / 0.358 / 0.322 | | | 0.620 / 0.761 / 0.784 |
| image stats + lat/lon | val | 477 | 0.841 | 0.835 | | | |
| valid_frac only | val | 477 | 0.432 | 0.316 | | | |
| VV p50 only | val | 477 | 0.495 | 0.486 | | | |
| ≤ −40 dB fraction only | val | 477 | 0.373 | 0.314 | | | |
| > −5 dB fraction only (land/ships) | val | 477 | 0.415 | 0.380 | | | |
| latitude only | val | 477 | 0.308 | 0.312 | | | |

Confusion (rows true no_oil/lookalike/oil): val `[[110,17,4],[9,105,21],[10,15,186]]`; test `[[145,1,1],[45,97,8],[20,37,93]]`.

Shortcut reading:
- No single feature or metadata-only model beats chance-ish on val (≤ 0.49 macro-F1); lat/lon add nothing on val
  (0.833 → 0.835). The 0.83 comes from the joint intensity distribution, a legitimate (if shallow) cue.
- **Test is geographically confounded by class** (SAR_DATA_AUDIT §10.2): metadata-only reaches oil recall 0.62–0.78 on test.
  Not learnable from trainval, but test per-class numbers carry a region component.
- Val → test drop (0.833 → 0.744) is mostly oil recall (0.88 → 0.62; acq-clean 0.52): test oil is darker and gets called
  lookalike; test no_oil is brighter and nearly always right. This is the floor E004 must beat on val **and** acq-clean test oil recall.

---

## 2026-09-26 · E002 — VV + validity/label cache

`python scripts/sar_build_cache.py` (12 procs, 2 min 15 s, 356 MB RSS) → `data/sar/cache/v1/` 12 GB:
`vv20.npy` f16 [3020,1024,1024], `lab20.npy` u8, `vv40.npy` f16 [3020,512,512], `lab40.npy` u8, `meta.parquet`, `cache_config.json`.
Row order = `splits_v1_acq.parquet`. valid = finite(VH) & finite(VV) & VH≠0 & VV≠0; linear-power block mean over valid
sub-pixels, block valid if ≥ 50% valid; clip [−40, 5] dB; scale (dB+40)/45; lab 0/1 (mask mean ≥ 0.5 over valid), 255 = invalid;
model validity channel = `lab != 255`. 4 non-2048 lookalike tiles cropped/padded to 2048 (logged). Originals untouched.

`python scripts/sar_check_cache.py` → `results/sar_cache_v1/{cache_checks.json, samples.png}`, log `logs/E002_checks.log`. All asserts pass:
0 non-finite, 0 out of [0,1], labels ⊂ {0,1,255}, VV = 0 on all invalid blocks, validity/positive fractions match meta exactly,
independent raw-TIFF recompute of 20 random 40 m blocks: max error **0.005 dB**. Linear-mean round trip exact except 27 partial-validity + 2 cropped tiles (max 0.12 dB).
Valid fraction vs audit index differs only by the one-zero-pixel rule (residual 0). Mask values {0,1}.
Clip fraction mean 0.10% (20 m) but class-skewed (SAR_DATA_AUDIT §10.4). 2 test tiles fully invalid, third 0.018% valid.
New finding: pixels are square in degrees, so x spacing = 9.93·cos(lat) m (§10.1).

---

## 2026-09-26 · E001 — `sih-ml` environment

`conda create -n sih-ml python=3.12` + pip torch cu128 (log `logs/E001_env.log`), verification `logs/E001_verify.log`:
torch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19, arch list incl. sm_120; RTX 5070 Ti CC 12.0, 15.92 GB; fp16 matmul 51.7 TFLOPS;
smp U-Net(ResNet-34, 2-ch) bf16 fwd+bwd on 4×512² OK, peak 1.27 GB. timm 1.0.30, smp 0.5.0, GDAL 3.13.3, sklearn 1.9.1,
numpy 2.5.3, pandas 3.0.6. Env size 8.7 GB. Driver 610.88.

---

## 2026-09-27 · E004 — ResNet-18 tile classifier (VV + validity, 40 m)

Commands (`conda activate sih-ml`, cwd `~/SIH26143`):
```
python scripts/e004_resnet18_cls.py --xscale on          # train; loads only role train/val rows (asserted)
python scripts/e004_resnet18_cls.py --eval-test          # once, after training_complete.json; refuses a second run
```
Outputs `results/E004_resnet18_cls/`: best.pt (42.7 MB), config.json, history.csv, curves.png, training_complete.json,
val_predictions.parquet, test_metrics.json, test_predictions.parquet. Logs `logs/E004_train.log`, `logs/E004_test.log`.

Setup: timm `resnet18.tv_in1k` (ImageNet), conv1 adapted by timm to 2 channels (VV standardised with train stats μ 0.4450 /
σ 0.1191 in cache units, invalid → 0; validity 0/1). acq split: train 1941 (no_oil 550 / lookalike 547 / oil 844), val 477.
Weighted CE (1.176 / 1.183 / 0.767), AdamW lr 3e-4 wd 1e-2, 1-epoch warm-up + cosine over 30 epochs, batch 32, bf16 autocast,
seed 26143, cuDNN deterministic. Early stopping patience 8 on val macro-F1; best checkpoint = max val macro-F1.
Aug (GPU): east-west scale **on** (p 0.5, x-scale s ~ U(0.5, 1), random x offset, bilinear VV / nearest validity; switch `--xscale off`),
h/v flips p 0.5, rot90 k ~ U{0..3}, global gain U(±1.5 dB) re-clipped to cache range.

Run: 16 epochs (early stop), **best epoch 8**, training 76.5 s (82 s wall incl. loading), 4.7 s/epoch,
peak VRAM allocated 3.98 GB (reserved 13.8 GB = PyTorch caching allocator), RSS 2.9 GB.
Determinism: an identical second run reproduced all 16 epochs' train loss / val loss / val macro-F1 exactly (diff 0.0).

**Validation (selection set)** — macro-F1 **0.907** vs E003 0.833 (bar met); acc 0.916, balanced acc 0.908.
P/R/F1: no_oil 0.906/0.878/0.892 · lookalike 0.839/0.889/0.863 · oil 0.976/0.957/0.967. Confusion `[[115,15,1],[11,120,4],[1,8,202]]`.
Val macro-F1 per epoch fluctuates 0.81–0.91 (epochs 4–16), so the selected 0.907 is optimistically biased by max-selection on 477 tiles.

**Official test, evaluated once after selection** (3 all/near-nodata tiles excluded: no_oil 00005, 00087, 00099):

| test variant | n | macro-F1 (E003) | acc (E003) | oil P / R / F1 | lookalike P / R / F1 | no_oil P / R / F1 | confusion (rows true no_oil/lookalike/oil) |
|---|---|---|---|---|---|---|---|
| full − nodata | 447 | **0.777** (0.744) | 0.781 (0.749) | 0.959 / **0.773** / 0.856 | 0.712 / 0.593 / 0.647 | 0.716 / 0.980 / 0.828 | [[144,2,1],[57,89,4],[0,34,116]] |
| acq-clean | 385 | **0.758** (0.724) | 0.761 (0.748) | 0.923 / **0.682** / 0.784 | 0.748 / 0.593 / 0.662 | 0.716 / 0.980 / 0.828 | [[144,2,1],[57,89,4],[0,28,60]] |
| place-clean | 212 | **0.812** (0.736) | 0.793 (0.764) | 0.914 / **0.865** / 0.889 | 0.934 / 0.645 / 0.763 | 0.644 / 1.000 / 0.783 | [[65,0,0],[36,71,3],[0,5,32]] |

Reading (no change made to E004 from these numbers):
- Beats E003 on every variant; biggest gain is oil recall (test 0.62 → 0.77, acq-clean 0.52 → 0.68, place-clean 0.54 → 0.86); oil precision ≥ 0.91 everywhere.
- Val → test gap 0.907 → 0.777. Dominant test error: lookalike → no_oil (57/150; E003 45/150), then oil → lookalike (34/150).
  Test lookalikes sit at a different latitude band (median 43° vs 29° trainval, SAR_DATA_AUDIT §10.2), so this is a
  domain-shift symptom, not a leakage symptom; per-class test numbers carry a region component.
- Oil is never predicted as no_oil on test (0/150), useful for a screening stage (lookalike confusions go to the analyst).

---

## 2026-09-27 · E102 correction — motion_state separated from source_type

Reviewer correction: stationary AIS behaviour must not be equated with fixed infrastructure. `scripts/source_type_v1.py`
now writes `motion_state` ∈ {stationary, loitering, mixed, moving, undetermined} (same rules/thresholds as below, unchanged)
and a separate `source_type` ∈ {vessel, fixed_infrastructure, dark_vessel, unknown}. With no vessel-metadata +
offshore-infrastructure-database match joined yet, **all 155 AIS tracks are `source_type = unknown`** (`source_type_basis`
states why). The earlier mapping stationary → stationary_fixed_infrastructure / others → moving_vessel is withdrawn.
Output `track_behaviour.csv` replaced by `track_motion.csv`; typed tables re-written. Score columns in all four typed tables
verified **bit-identical** to their sources (inputs now read with `float_precision="round_trip"`; the first version lost last-ULP
bits ≤ 3.6e-15 through the default CSV parser). `results/incident_001/`: 0 files modified since 2026-09-26.
Next step for source_type (not done): join an authoritative platform database (e.g. BOEM platform structures) by spatial match + AIS metadata.

---

## 2026-09-26 · E102 — Source-type layer v1 (post-score, AIS behaviour) — superseded in part by the correction above

`conda activate opendrift; python scripts/source_type_v1.py` (config `configs/source_type_v1.json`) →
`results/incident_001_source_type_v1/{track_behaviour.csv, v2_summary_typed.csv, v2_release_age_typed.csv, v2_1_summary_typed.csv, v2_1_release_age_typed.csv, config_used.json}`,
log `logs/E102_source_type_v1.log` (full per-track table). Score columns are copied unchanged (asserted). No vessel names used.

Window: AIS in [T0 − 48 h, T0]. Metrics per MMSI: spread = distance of each report to the median position;
`position_spread_km` = p90, `max_excursion_km`, `stationarity_score` = share of reports within 0.5 km, `median_sog`, share SOG < 0.5 kn.
Rules (in order): sparse (n < 10 or span < 6 h) → `undetermined` unless it moved > 5 km (`moving`); `stationary` if p90 ≤ 0.25 km,
max ≤ 1 km, slow ≥ 0.9; `loitering` if p90 ≤ 2 km, max ≤ 5 km, slow ≥ 0.8; `mixed` if stationarity ≥ 0.5; else `moving`.
source_type: stationary → `stationary_fixed_infrastructure`; loitering/mixed/moving → `moving_vessel`; `dark_vessel` reserved (SAR-only, no rows yet).
AIS Status (1 at anchor, 5 moored) and VesselType 98/99 are reported as `metadata_hint` only.

Threshold justification (stats of all 155 tracks printed before fixing): p90-spread histogram
[0–0.05: 27, 0.05–0.1: 2, 0.1–0.25: 10, 0.25–0.5: 3, 0.5–1: 5, 1–2: 6, 2–5: 5, 5–10: 5, 10–50: 47, > 50: 45 km].
The 39 tracks ≤ 0.25 km all span ~48 h with SOG ≈ 0 (spars, TLPs, FPSOs, DP drillships): 0.25 km covers DP watch circles and
platform offsets; max ≤ 1 km rejects parked-then-left tracks (e.g. p90 0.21 km but max 46 km → `mixed`). Moving tracks sit at p90 > 5 km.
Loitering band (work-site vessels) is the sparse 0.25–2 km region.

Post-hoc logic fixes after the first run (no threshold changed, no Cerulean input): (1) 53 tracks were `undetermined`, incl.
short transits with 60 km excursion → sparse tracks that clearly moved are now `moving`; (2) one TLP had all-NaN SOG and fell to
`mixed` → SOG share uses valid SOG only; if none, geometry decides.
Borderline, left as is: a jack-up rig with p90 0.286 km → `loitering` (threshold 0.25 not moved to fit a known asset).

Result: stationary 31, loitering 4, mixed 20, moving 89, undetermined 11. 0–6 h top six (V2): SHELIA BORDELON moving;
ARGOS PLATFORM, OCEAN BLACKHORNET, OCEAN BLACKLION stationary; HARVEY SUPPORTER mixed; C-CONSTRUCTOR loitering.
V2 point-release top 10: 4 stationary (BIG FOOT, ARGOS, BLACKHORNET, BLACKLION).

---

## 2026-09-26 · E101 — AIS density score V2.1 (interpolated AIS positions)

`conda activate opendrift; python scripts/score_ais_density_v2_1.py` (config `configs/ais_v2_1.json`) → `results/incident_001_v2_1/`
(`ais_density_time_scores_v2_1.parquet`, `ais_density_scores_v2_1.csv`, `release_age_scores_v2_1.csv`, `compare_v2_vs_v2_1_{summary,release_age,per_bin}.csv`, `v2_1_diagnostics.json`), log `logs/E101_v2_1.log`.
V2 untouched and still the frozen baseline. Only the vessel-position step differs; kernel, σ, bins, aggregation identical.

Rule per hourly model time: exact report → exact; bracketing reports with gap ≤ 30 min and implied speed ≤ 50 kn → linear
lat/lon interpolation (longitude-safe); else nearest report within ±5 min → `edge_nearest`; else unavailable (not scored).
Parameters fixed from the AIS gap distribution **before** any V2.1 score was computed: all gaps median 1.3 min, p99 29.5 min
(→ 30 min bracket), moving-vessel gap p95 5.7 min (→ 5 min edge: ≤ 1.9 km at 12 kn, < σ/2), max observed implied speed 23 kn (→ 50 kn glitch guard).

Coverage: 7644 vessel-times → 19 exact, 1939 interp, 131 edge, 5555 unavailable. Scored vessels 155 → 152 (3 lose all coverage).
Matched hours per vessel: mean −2.0 (max −10). Point-release rank Spearman V2 vs V2.1 = 0.993.

| bin | n V2 / V2.1 | Spearman | top-10 overlap | top-6 same order | max abs Δrank (V2 top 10) |
|---|---|---|---|---|---|
| 0–6 h | 65 / 61 | 0.999 | 9 | no | 3 |
| 6–12 h | 78 / 76 | 0.999 | 10 | no | 1 |
| 12–24 h | 82 / 81 | 0.998 | 10 | yes | 0 |
| 24–36 h | 74 / 64 | 0.958 | 8 | yes | 15 |
| 36–48 h | 112 / 111 | 0.995 | 10 | yes | 0 |

Largest change: **SHELIA BORDELON** (V2 0–6 h rank 1, point rank 2 → V2.1 unavailable in 0–6 h, point rank 18). Cause:
last AIS report 22:42:53 Z, 79 min before T0. V2 matched it to the T0−1 h model time (18.8 min offset within ±20 min);
V2.1 has no report after it (edge tolerance 5 min) and the previous bracket 19:51→22:35 spans 164 min > 30 min, so the 0–2 h
positions are unavailable. At its SOG 0.8 kn the 18.8 min extrapolation is only ≈ 0.5 km, so V2's use was physically benign here;
V2.1's time-only rule is conservative. **Not changed** (changing it after seeing a Cerulean vessel drop would be tuning).
Other 0–6 h top-6 members keep their order (ARGOS 1, BLACKHORNET 2, HARVEY SUPPORTER 3, BLACKLION 4, C-CONSTRUCTOR 5).
Candidate for a future pre-registered V2.2: distance-bounded edge rule (allow nearest report while SOG × Δt ≤ σ/5), decided on the gap/speed distribution, not on rank outcomes.

---

## 2026-09-26 · E000 — Repository + data audit (no training)

### Commands run (all read-only on originals; `conda activate opendrift`, cwd `~/SIH26143`)
```
python scripts/sar_channel_evidence.py 40      # 160 sampled tiles, ~10 s
python scripts/s1_grd_calibrate.py             # labelled VV/VH check on Incident 001 GRD, ~20 s
python scripts/sar_build_index.py              # all 3020 tiles, ~2 min, 12 procs
python scripts/sar_overlap_content.py          # 13,417 overlapping pairs, ~3 min
python scripts/sar_make_splits.py acq ; python scripts/sar_make_splits.py strict
# reproducibility: run_hindcast_48h.py + score_ais_density_v2.py copied with OUTDIR -> scratchpad, diffed vs results/
```
Outputs: `results/sar_audit/*` (50 MB). `results/incident_001/` untouched (mtimes unchanged).

### Results — SAR: see `SAR_DATA_AUDIT.md`
- band 1 = VH, band 2 = VV (4 independent physical tests incl. labelled real scene).
- VH processed differently train vs test (same acquisition, VV identical, VH Δ 4.7 dB) → VV-only baseline.
- 62 test oil tiles are pixel-duplicates of trainval tiles; 236/450 test tiles share a place with trainval.
- 3 test no_oil tiles all-nodata; 4 lookalike tiles not 2048²; nodata = 0.0; values to −128 dB.

### Results — physics / attribution audit

Reproducibility: deterministic hindcast and V2 scoring reproduce **bit-exactly**. 48 h mean
endpoint (−90.42331, 26.89010) matches the handover. Ensemble: 27 × 400 × 49 = 529,200 rows, as stated.
Readers checked at (−90.43, 27.0, 2023-01-02 12 UTC): wind (−4.4, 7.9) m/s, current (−0.01, 0.12), Stokes (−0.05, 0.09) — no silent fallback.

No code bug found that changes results. Issues, most important first:

1. **The 0–6 h "all six in ranks 1–6 in all 27 scenarios" result is weaker evidence than it looks.**
   Every one of the six peaks at age 0 or 1 h (`release_age_scores.csv`: peak_age_hours 0–1), i.e. the
   vessel is within 1.7–7.5 km of the observed slick at detection. At t ≤ 1 h all 27 scenarios are
   still ≈ the seed polygon, so rank invariance across scenarios is expected, not a robustness result.
   Cerulean's vessel association (to our understanding; check their docs) also weighs AIS proximity
   around detection time, so agreement in this bin is partly circular.
   Correct framing: "consistent with proximity at detection"; the hindcast-dependent evidence is in the older bins.
2. **Fixed / station-keeping assets are ranked as moving vessels** (violates the source-type rule).
   Position std before T0: ARGOS PLATFORM 0.00 km (VesselType 98, SOG 0), OCEAN BLACKHORNET 0.00 km and
   OCEAN BLACKLION 0.03–0.08 km (drillships), C-CONSTRUCTOR 0.5–0.8 km, BP MAD DOG PLATFORM 0, BIG FOOT 0.
   Only SHELIA BORDELON (≈5 km, SOG ~0.8 kn) and HARVEY SUPPORTER (≈30 km) actually move.
   → 4 of the 0–6 h top six belong in the infrastructure/stationary branch. Needs a stationarity rule before any ranking is shown.
3. **AIS position is taken from the nearest report (±20 min), not interpolated.** At 12 kn a vessel moves
   up to 7.4 km in 20 min, larger than σ = 5 km. Proposed fix: linear interpolation to model time with a
   max-gap guard. This changes a frozen method → needs your approval; would be logged as V2.1, V2 kept.
4. **Possible windage/Stokes double counting.** OpenOil applies wind_drift_factor 0.03 (a rule of thumb
   that empirically already contains wave-induced drift) *plus* explicit SMOC Stokes drift. The ensemble's
   0.02 member partly covers this. Not changed (frozen); flag for the physics write-up / sensitivity note.
5. `score_ais_hindcast_v1.py` is a byte-identical duplicate of `score_ais_hindcast.py`.
6. All scripts hard-code `ROOT = /home/admin_wsl/SIH26143` and write into `results/` → will not run on
   Kaggle (read-only input) and re-running overwrites frozen outputs. Proposal: `ROOT`/`OUTDIR` from env var, default unchanged.
7. `data/incident_001/metadata/cerulean_candidates.geojson` actually contains the two slick polygons (3775938, 3750297).
8. Incident 002 (slick 3750297, centroid ≈ −88.92, 29.40, near the Mississippi delta) lies **outside** the
   current forcing ROI (N ≤ 29.14) and AIS ROI (N ≤ 28.14), ~290 km from Incident 001. Needs its own forcing/AIS and a real land mask (fallback `land_binary_mask = 0` is wrong near the coast).
9. Minor: V1 keeps the last duplicate AIS timestamp, V2 the first. `data/incident_001/waves/` is empty.

Cerulean MMSIs are hard-coded in 3 scoring scripts but only used as a report flag column — verified not used in any score.

### Decisions
- Band order fixed: VH = band 1, VV = band 2.
- ML baselines use **VV only** + validity mask. VH only as a later, separately reported ablation.
- Split policy `acq` recommended (1941 train / 477 val, 152 excluded); report test 3 ways (full, acquisition-clean, place-clean).
- Physics stays frozen. Items 2 and 3 above need your go-ahead.

---

## Planned next experiments (not run — awaiting approval)

E001–E003, E101, E102 done (entries above). E000 item 2 is superseded by E102 (C-CONSTRUCTOR is `loitering`, not stationary).

| ID | What | Cost |
|---|---|---|
| E004 | DONE 2026-09-27 (entry above): val macro-F1 0.907, test 0.777 / 0.758 / 0.812 | — |
| E005 | DONE 2026-09-27 (entry above): SegFormer MiT-B2, Trujillo only. Val IoU@0.5 0.753; frozen threshold 0.60 gives IoU 0.754 / Dice 0.860. Test not run | — |
| E006 | Official test evaluation (3 variants), once, after model selection on val | DONE 2026-09-28 (segmentation; test IoU 0.335 full / 0.257 acq-clean / 0.493 place-clean) |
| E007 | Incident 001 GRD → σ0 VV → geocode → tile → classify → segment → polygon → geometry | ~10 min |
| E008 | Hindcast + attribution from **our** polygon (new output dir, frozen parameters) | ~15 min |
| E009 | Physics sensitivity, windage/Stokes double counting (new script + new output dir; frozen baseline untouched): (1) explicit Stokes + lower/no windage, (2) windage without explicit Stokes, (3) frozen baseline. Compare release region + per-bin ranks (older bins); never choose by Cerulean agreement | ~3 × 15 min CPU |
| E010 | (optional) pre-registered V2.2 edge rule: distance-bounded (SOG × Δt ≤ σ/5) instead of 5 min | < 1 min |
