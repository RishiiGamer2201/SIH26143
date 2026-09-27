# SAR Data Audit — Zenodo Sentinel-1 Oil-Spill Dataset

Audit date: 2026-09-26. All checks are read-only on the original TIFFs. Derived artefacts live in
`results/sar_audit/` (≈50 MB). Reproduce with the scripts named in each section.

Dataset: Trujillo-Acatitla et al., *Marine oil spill detection and segmentation in SAR data with two
steps Deep Learning framework*, Mar. Pollut. Bull. 204 (2024), doi:10.1016/j.marpolbul.2024.116549.
Zenodo Part I [8346860](https://zenodo.org/records/8346860), Part II [8253899](https://zenodo.org/records/8253899),
Part III (test) [13761290](https://zenodo.org/records/13761290). Zenodo text: "Sentinel-1 SAR Sigma0, in
decibels, two polarizations (VV, VH)"; "masks are not georeferenced". **No band order is documented.**

---

## 1. Directory structure (`data/sar/extracted/`)

| split | class | images | masks | mask naming |
|---|---|---|---|---|
| trainval | oil | `Oil/` 1200 | `Mask_oil/` 1200 | same filename |
| trainval | lookalike | `Lookalike/` 685 | `Mask_lookalike/` 685 | same filename |
| trainval | no_oil | `No_oil/` 685 | `Mask_no_oil/` 685 | same filename |
| test | oil | `Images/Oil/` 150 | `Mask/Oil/` 150 | `<id>_segmentation.tif` |
| test | lookalike | `Images/Lookalike/` 150 | `Mask/Lookalike/` 150 | `<id>_segmentation.tif` |
| test | no_oil | `Images/No oil/` (space!) 150 | `Mask/No oil/` 150 | `<id>_segmentation.tif` |

- Image ID sets equal mask ID sets in every folder (verified with `diff`).
- Train oil IDs span `00000–01339` with 140 gaps (1200 files). Not an error, just non-contiguous.
- 5 stray `*.tif.aux.xml` files (GDAL PAM histograms written by an earlier `gdalinfo -stats`). Harmless; ignore.
- Directory `data/sar/extracted` is hard-linked with `~/kaggle_upload/sih26143-complete-project/data/sar`.

## 2. File format (`scripts/sar_build_index.py` → `sar_index.parquet`)

- Images: GeoTIFF, 2 bands float32, LZW, 512×512 tiles, EPSG:4326, pixel 8.9831528e-5° (~10 m), no rotation, **no nodata tag**, no band descriptions.
- **Not all images are 2048×2048**: 4 trainval lookalike tiles are 2148×2048 (`00140`), 2048×2049 (`00166`), 2047×2048 (`00288`), 2248×2048 (`00390`). Their masks have matching shapes. Loader must not assume 2048.
- Masks: uint8, values exactly `{0,1}` for oil, `{0}` for lookalike/no_oil (all 3020 checked). Test masks were written by SCIFIO/Fiji. Masks carry no georeferencing (4 trainval oil masks — 00055/58/59/85 — do; irrelevant). Pairing is by filename + identical pixel grid.
- No oil image has an empty mask; no non-oil image has a positive mask.

## 3. Nodata / invalid pixels

- NaN: 0. Inf: 0 (all 3020 images, both channels).
- **Nodata is encoded as exact 0.0** (image borders / outside swath). 0 dB is never a physical ocean value here.
- Train only: 2.12 M pixels where exactly **one** channel is 0 → treat as invalid too.
- Images with any invalid pixel: 464/2570 trainval, 88/450 test. Valid fraction min 0.21 (trainval).
- **3 official-test no_oil tiles are (almost) entirely nodata**: `No oil/00005` (0.000), `00087` (0.000), `00099` (0.0002). Exclude from test metrics and report the exclusion.
- 36 oil masks mark 21,032 pixels on invalid (zero) image pixels → exclude invalid pixels from loss and metrics.
- Extreme values: minima down to −128 dB (log of near-zero noise-subtracted power), maxima to +31 dB (ships/platforms). Clipping needed.

Definition used everywhere: `valid = isfinite(c0) & isfinite(c1) & (c0 != 0) & (c1 != 0)`.

## 4. Channel order — VERIFIED: `channel_0 = VH`, `channel_1 = VV` (high confidence)

Evidence is physical, from four independent tests (`scripts/sar_channel_evidence.py`, `scripts/s1_grd_calibrate.py`):

| Test | Physics expectation | channel_0 | channel_1 | Consistent with |
|---|---|---|---|---|
| Oil damping, median(sea) − median(oil) inside mask, 80 oil images | Bragg damping is strong in VV; VH sits near noise floor so damping is weak/absent | 0.18 dB (CNR 0.13) | **6.36 dB (CNR 4.0)**; ch1 > ch0 in **80/80** images | ch1 = VV |
| Clean-sea level | VV ≈ −15…−22 dB; VH ≈ −25…−35 dB (≈ NESZ) | −33 dB | −20 dB; ch0 < ch1 in **160/160** | ch0 = VH |
| Bright-target excess over sea clutter | Cross-pol has higher target-to-clutter ratio | 16.8 dB | 12.9 dB; ch0 > ch1 in 75% | ch0 = VH |
| **Labelled reference**: Incident 001 GRD, VV/VH identified by SAFE filename + annotation `<polarisation>`, calibrated σ0 around Cerulean slick 3775938 | — | VH: sea −28.7 dB, damping **−0.2 dB** | VV: sea −14.9 dB, damping **+2.7 dB** (CNR 1.1) | same signature as dataset |

Caveat considered: masks may have been drawn on one channel, which could bias the damping test.
The level test and target test do not use masks and agree. Remaining uncertainty is low; no
test points the other way.

**Decision**: code may now name bands `VH = band 1 (channel_0)`, `VV = band 2 (channel_1)`, with a
reference to this section. Keep the constant in one place (config), not scattered.

## 5. CRITICAL: channel_0 (VH) is processed differently in train vs official test

`scripts/sar_overlap_content.py` compares tiles whose footprints overlap. For **the same acquisition**
(VV pixel-identical, corr ≈ 1.000, median |Δ| = 0.00 dB):

| pair | n | VH median \|Δ\| |
|---|---|---|
| test ↔ test | 168 | 0.00 dB |
| trainval ↔ trainval | 1643 | 0.00 dB |
| **trainval ↔ test** | 178 | **4.72 dB** (min 1.6, max 8.2) |

Same pixels, same date, same VV — different VH. Distribution stats agree: VH std 3.2 dB (train) vs
0.95 dB (test); VH p01 −43 dB (train) vs −32 dB (test). In the labelled Incident 001 crop, raw VH
minus thermal-noise-removed VH = **+4.49 dB** median. Most likely explanation: **train VH is
thermal-noise-removed, test VH is not** (hypothesis, consistent but not proven). VV is unaffected
(VV ≫ NESZ).

Consequence: any model using VH will see a train/test distribution shift that does not exist in
reality. **Baseline models use VV only.** VH may be tested later as an ablation with explicit
handling (e.g. per-image standardisation), reported separately.

## 6. Leakage — the official test set is contaminated

Footprint overlaps (EPSG:4326 bounding boxes) + pixel comparison of the shared window at 1/8 res:

- Dataset is global: largest block is the Gulf of Mexico (~870 trainval tiles, 31 test), then E. Med./Red Sea/Persian Gulf/Black Sea/North Sea/SE Asia/W. Africa. Test is over-weighted in regions like 40N 40E (85 test vs 14 trainval).
- 15,863 overlapping tile pairs. Tiles are overlapping crops of the same scenes.
- **1,989 pairs share an acquisition** (VV corr > 0.9, |Δ| < 0.5 dB). 1,654 acquisition groups.
- **62 official-test oil images share identical pixels with trainval oil images.**
- **236 of 450 test images overlap a trainval footprint** (same place; different date unless in the 62).
- 5 exact-footprint duplicate pairs inside trainval oil, 19 inside test oil.
- Label (in)consistency where the same slick appears in two tiles: mask IoU 0.91 (trainval↔trainval), **0.78 (trainval↔test)**, 0.86 (test↔test) → annotation noise ceiling; test masks look more generous (test oil pixel fraction median 4.3% vs 1.8% trainval).
- Same-acquisition overlaps with different class labels: 77 trainval↔trainval, 5 trainval↔test, 5 test↔test pairs (mostly oil tile next to a no_oil/lookalike tile with no oil in the overlap; only 2 pairs have oil in the overlap labelled non-oil in the other tile).

Random image-level train/val splits would put crops of the same scene on both sides — inflated
validation scores. The published ~99% accuracy / 96% IoU should be read in that light.

## 7. Leakage-safe split (`scripts/sar_make_splits.py`)

Grouping unit `geo_group` = connected components of tiles whose footprints overlap (628 groups, max 379 tiles).
StratifiedGroupKFold(5, seed 26143) on class, grouped by `geo_group`; fold 0 = val.

| policy | excluded trainval | train | val | test report |
|---|---|---|---|---|
| **`acq` (recommended)** | 152 tiles sharing an acquisition with test (145 oil) | 1941 | 477 | full official test (447 usable) **+** place-clean subset |
| `strict` | 711 tiles sharing any footprint component with test (432 oil) | 1501 | 358 | full official test is place-clean |

Always report three test numbers: (1) official full test minus the 3 nodata tiles (comparability),
(2) acquisition-clean (147 no_oil/150 lookalike/88 oil usable), (3) place-clean (65/110/37).
Files: `results/sar_audit/splits_v1_{acq,strict}.parquet` (columns `role`, `fold`, `geo_group`,
`scene_group`, flags). Asserts in the script guarantee train/val disjoint groups and no shared
acquisition with test.

## 8. Pixel statistics (valid pixels, per-image medians of the stat)

| split / class | VH p01 | VH p50 | VH p99 | VV p01 | VV p50 | VV p99 |
|---|---|---|---|---|---|---|
| trainval oil | −43.0 | −33.6 | −27.3 | −25.8 | −20.1 | −16.4 |
| trainval lookalike | −41.2 | −32.8 | −26.0 | −31.1 | −22.9 | −16.2 |
| trainval no_oil | −40.1 | −31.7 | −25.8 | −22.4 | −18.4 | −13.8 |
| test oil | −32.0 | −29.7 | −26.8 | −28.2 | −21.4 | −17.5 |
| test lookalike | −31.5 | −29.1 | −26.0 | −29.3 | −22.9 | −17.3 |
| test no_oil | −29.5 | −27.2 | −25.1 | −19.9 | −16.3 | −12.7 |

Shortcut risk: VV scene level differs by class (no_oil brightest, lookalike darkest ≈ low wind).
A classifier can partly separate classes by wind regime alone. The feature-only baseline (below)
measures how much.

## 9. Recommended preprocessing (ML)

1. Read band 2 (VV, dB). Validity mask as in §3.
2. Clip VV to [−40, +5] dB (below the dark tail of low-wind areas, whose per-image p01 reaches −31 dB; only the −128 dB noise-subtraction artefacts and bright ships saturate). Scale `(x + 40) / 45` → [0, 1]. Verify clipped-pixel fraction < 0.1% when caching. Invalid → 0 and pass the validity mask as a 2nd input channel (the model must not learn from border shape without knowing it is border).
3. Downsample by block-averaging in **linear power** (10^(dB/10)), then back to dB — reduces speckle without biasing the mean. Classifier: ×4 (40 m, 512²). Segmentation: ×2 (20 m, 1024²), metrics also at native 10 m by upsampling predictions.
   **Correction (E002):** the grid is square in *degrees* (8.93e-5°): 9.93 m in y everywhere but 9.93·cos(lat) m in x
   (4.8–10 m; median 8.7 m). "20 m / 40 m" is exact in y only; x is 10–20 / 19–40 m. Not resampled (see §10).
4. Masks: downsample with mean ≥ 0.5; invalid pixels → ignore index.
5. Cache once to `data/sar/cache/` as per-image `.npy` float16 (VV 20 m ≈ 2 MB/img, ≈6 GB total; masks ≈3 GB). Originals are never modified. 15 GB WSL RAM → never hold the full set.
6. Augment: flips, 90° rotations, random crops, ±1.5 dB global gain (calibration uncertainty), no photometric tricks that break dB semantics.
7. Same chain for the real incident GRD: σ0 VV (calibration LUT; thermal-noise removal optional for VV) → geocode to EPSG:4326 at 8.9831528e-5° with the annotation GCP grid (ocean, no terrain — ellipsoid geocoding acceptable; document ~tens-of-metres error) → dB → steps 2–3 → 2048² tiles with overlap → predict → mosaic.

## 10. Findings from E002 (cache) and E003 (feature baseline), 2026-09-26

1. **Anisotropic pixels.** EPSG:4326 with square-degree pixels: x spacing = 9.93·cos(lat) m. Tiles span lat ≈ 0–60°,
   so the same ×2/×4 factor gives different ground scales in x. Kept as integer block factors (no interpolation);
   `meta.parquet` carries `lat_mid`, `px_x_m`, `px_y_m`. Option for E004+: random x-scale augmentation (0.5–1.0).
2. **Test classes are geographically separated; trainval classes are not.** Median tile latitude: test oil 25.8°,
   no_oil 35.7°, lookalike 43.0° vs trainval 27.8–29.1° for all classes. A metadata-only model (lat/lon/pixel size/shape)
   is at chance on val (macro-F1 0.47, lat alone 0.31) but reaches oil recall 0.62–0.78 on test. Trainval gives no
   latitude signal to learn, so this cannot leak via training, but test-set per-class numbers partly reflect region
   (sea state, wind regime), not only slick appearance. Keep the 3-way test report; also report per-region if possible.
3. **Test/trainval intensity shift per class (VV p50, 20 m):** no_oil −16.3 vs −18.4 dB; oil −21.4 vs −20.0 dB;
   lookalike −22.8 vs −22.8 dB. Test no_oil is brighter (easier), test oil darker (closer to lookalike).
4. **Dark-tail clipping is class-skewed.** Pixels ≤ −40 dB at 20 m: trainval lookalike mean 0.34% (p99 6.5%, max 13%),
   no_oil 0.09%, oil 0.02%; test ≤ 0.01% for all classes. 79 tiles clip > 1% (59 lookalike, 14 no_oil, 6 oil). These are near-noise-floor
   low-wind scenes (p50 ≈ −36 dB). The floor stays −40 dB (S1 IW VV NESZ ≈ −22…−26 dB, so values below −40 dB carry no signal);
   single-feature probe on this fraction is at chance (val macro-F1 0.31), so not a shortcut by itself.
5. Lookalike tiles often contain land (bright, valid). Land is not masked in the dataset; a land/sea mask is a
   candidate input for E005+. Single "bright fraction" probe: val macro-F1 0.38 (weak).
6. Partial-validity tile edges keep a thin dark rim (blocks ≥ 50% valid include dark edge pixels). Minor; the
   validity channel marks the border. 3rd nodata test tile (no_oil/00099) has 0.018% valid pixels — treat as nodata.
7. Some trainval TIFFs raise a benign libtiff warning (`Photometric ... ExtraSamples`); bands read correctly
   (independent recompute of 20 random 40 m blocks from raw TIFFs matches the cache within 0.005 dB).
8. The `acq` split's val is already place-disjoint from train (no val geo_group appears in train), so
   "val place-clean" = "val all" for this split.
