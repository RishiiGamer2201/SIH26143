# Experiment Log

Newest first. Every entry: date, what ran, exact command, outputs, result, decision.

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

**Official test: NOT evaluated.** E006 awaits review.

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
| E006 | Official test evaluation (3 variants), once, after model selection on val | minutes |
| E007 | Incident 001 GRD → σ0 VV → geocode → tile → classify → segment → polygon → geometry | ~10 min |
| E008 | Hindcast + attribution from **our** polygon (new output dir, frozen parameters) | ~15 min |
| E009 | Physics sensitivity, windage/Stokes double counting (new script + new output dir; frozen baseline untouched): (1) explicit Stokes + lower/no windage, (2) windage without explicit Stokes, (3) frozen baseline. Compare release region + per-bin ranks (older bins); never choose by Cerulean agreement | ~3 × 15 min CPU |
| E010 | (optional) pre-registered V2.2 edge rule: distance-bounded (SOG × Δt ≤ σ/5) instead of 5 min | < 1 min |
