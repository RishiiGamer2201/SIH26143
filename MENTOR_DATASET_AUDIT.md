# E005-A — Mentor dataset audit (`shuban220409/combinedoilspilldataset`)

**Status: APPROVED and FROZEN (2026-09-27).** Frozen conclusions:
1. The Trujillo subset is 6040/6040 SHA-256 identical to our local data. Use the local files only.
2. The folder labelled "M4D" is the Oil Spill Drone RGB dataset, not Sentinel-1 SAR. It is excluded from E005.
3. The uploader's M4D train/val/test split is not used.
4. MADOS stays separate (Sentinel-2 multispectral).

Date 2026-09-27. Audit only, no training. Scope (per instruction): the Trujillo-Acatitla Sentinel-1 subset and the "M4D" subset.
MADOS (Sentinel-2) is deferred and audited separately. DARTIS (bounding boxes only) is not used for segmentation.

Reproduce: `conda activate sih-ml; python scripts/e005a_audit.py {trujillo|m4d|report}`. The Kaggle side is the private kernel
`rishiikumarsingh/sih26143-e005a-mentor-dataset-audit` (`scripts/kaggle/e005a_audit/`, CPU, runs next to the mounted dataset).
Nothing was downloaded except the kernel outputs (SHA-256 CSV, 906 kB; M4D zip, 1.14 GB). The 107.9 GB dataset was **not** downloaded.

Outputs:
- `data/segformer_combined/manifest.parquet`: one row per image-mask pair; 3,020 Trujillo + 1,268 "M4D"; the two sources are **not mixed**.
- `data/segformer_combined/audit/`:
  - `trujillo_identity.json`, `trujillo_sha256_compare.csv`
  - `m4d_{images,masks,mask_colours,pairs}.parquet`
  - `m4d_summary.json`, `m4d_id_collisions.csv`, `m4d_identical_masks.csv`
  - `m4d_visual_audit.png`
- `data/segformer_combined/m4d_raw/dataset/`: extracted M4D subset (1.1 GB). `kernel_out/` holds the raw kernel outputs.

## 1. Dataset layout (verified on the Kaggle mount, `kernel_out/input_tree.txt`)

69,661 files, 107.9 GB compressed.

| Folder | Content | Files |
|---|---|---|
| `Dataset/Semantic Segmentation/Sentinel-1_SAR_Oil_spill_image_dataset` | Trujillo-Acatitla S1 | 6,040 TIFF (3,020 pairs) |
| `Dataset/Semantic Segmentation/dataset` | labelled "M4D" by the uploader | 1,268 JPG + 1,268 PNG |
| `Dataset/Semantic Segmentation/MADOS` | Sentinel-2 | deferred |
| `Dataset/Bounding Box/DARTIS` | VOC boxes | not used |

## 2. Trujillo subset: exact duplicate of our local data

| Check | Result |
|---|---|
| Files compared (Kaggle vs `data/sar/extracted/`) | 6,040 / 6,040 |
| **SHA-256 exact matches** | **6,040** |
| Mismatches | 0 |
| Unmatched Kaggle / unmatched local | 0 / 0 |

Path normalisation:
- Kaggle `OilSpill/Tiff files/archive2_00000.tif` → local `Oil/00000.tif`.
- Kaggle `.../Masks/archive_*.tif` → local `Mask_<cls>/*`.
- Kaggle `Test/<Cls>/{Tiff files,Masks}` → local `Images|Mask/<cls>`.

The mapping is one-to-one (asserted). The join is by mapped path, not by hash, because 1,668 local files are byte-identical all-zero masks.

**Decision:** we will not store a second copy. The E005 manifest points to our existing local files. Everything in `SAR_DATA_AUDIT.md` still applies unchanged:
- 2048² float32, 2 bands (band 1 = VH, band 2 = VV), dB.
- EPSG:4326, 8.93e-5° pixels (so x spacing = 9.93·cos(lat) m).
- Binary masks {0,1}; masks are not georeferenced.
- 3 no-data test tiles.
- Test–trainval acquisition/place overlap; class-confounded test geography.

Side finding: there are 2 exact duplicate image pairs (00007 = 01339 and 00356 = 00357). Each pair is inside one acquisition group, so the `acq` split already keeps them on the same side.

## 3. "M4D" subset: it is not M4D and not SAR

### 3.1 Provenance

| Source | What it says |
|---|---|
| Packaged `metadata_and_naming_conventions.txt` | "M4D Group Dataset … Sensor Modality: SAR (Sentinel-1) … RGB Semantic Masks"; palette `[255,0,124]` oil, `[255,204,51]` others/lookalikes, `[51,221,255]` water/no-oil, `[0,0,0]` background |
| Kaggle description | "M4D … 5 distinct classes (Oil, Sea, Ship, Land, Look-alike) … ~10 m … **1,189** image patches" |
| **Pixels (all 1,268 files)** | True-colour camera photographs: 1920×1080 (1150), 768×432 (78), 5184×2920 (24), 3840×2160 (11), 5184×3888 (3), 4056×3040 (2). Drone views of a port (quays, cranes, containers, tugs, tankers in locks, booms, sheen) |
| Publication matched to the files | *Oil Spill Drone: a dataset of drone-captured, segmented RGB images for oil spill detection in port environments*, De Kerf et al., Scientific Data 2024 (arXiv 2402.18202). It reports **1,268 images**, classes oil / water / other ("other" = ships, quays, buildings, air) |

Conclusions:
- The folder holds the Oil Spill Drone dataset, mislabelled "M4D" by the uploader.
- The 1,268 file count matches that publication exactly.
- **1,189 vs 1,268 explained:** 1,189 comes from the uploader's description of a *different* dataset (a 5-class Sentinel-1 "M4D" set). Those files are not in this upload.
- No 5-class SAR data exists anywhere in this folder.
- Recorded in the manifest as `source_dataset = m4d_uploader_drone_rgb`.

### 3.2 Image format (all 1,268 decoded, 0 corrupt)

- JPEG, RGB, uint8. No alpha, no EXIF, no ICC, no georeferencing: `crs` and `resolution` are null (oblique drone views have no fixed GSD).
- Six distinct sizes, all camera/video formats (16:9, plus 5 files at 4:3).
- **The 3 channels are real colour, not replicated grayscale:**
  - R = G = B for only 1.2% of pixels (mean; max 15%).
  - Mean |R−G| is 13 DN.
  - Channel correlation (mean): RG 0.94, RB 0.89, GB 0.96.

### 3.3 Mask palette (every pixel of every mask enumerated)

Masks are PNG, RGB, same size as their image. There are **exactly 4 colours**, with no anti-aliased or unexpected values (asserted).

| RGB | Uploader name | Visual meaning (checked in figure) | Pixel share | Masks containing it |
|---|---|---|---|---|
| 255,0,124 | Oil Spill | oil / sheen on water | 32.2% | 998 |
| 255,204,51 | Others / LookAlikes | ships, quays, land, cranes, sky: **not look-alike slicks** | 38.0% | 1,117 |
| 51,221,255 | Water / NoOilSpill | clean water | 25.5% | 940 |
| 0,0,0 | Background | thin unlabelled gaps between polygons and at the frame border (~0.6–7%) | 4.4% | 1,268 |

Masks per class folder:

| Folder | Masks | With oil | With others | With water | Oil fraction (mean / median / min–max) |
|---|---|---|---|---|---|
| OilSpill | 998 | 998 | 888 | 691 | 0.419 / 0.377 / 0.001–0.994 |
| LookAlikes | 229 | 0 | 229 | 208 | 0 |
| NoOilSpill | 41 | 0 | 0 | 41 | 0 |

- Folder class is fully consistent with mask content. Oil appears if and only if the folder is OilSpill.
- The "LookAlikes" folder is really "port scene with structures and no oil". It contains **no SAR look-alikes**.
- 81 masks are ≥ 95% oil and 284 are ≥ 50% oil. Compare Trujillo oil tiles: mean 3.75%, median 1.9% oil.

### 3.4 Pairing

1,268 images ↔ 1,268 masks. Pairing is by stem within each class folder:
- Every image has exactly one mask.
- No missing files, no duplicate stems, no ambiguous pairs.
- Every image-mask pair has identical dimensions.
- 0 corrupt files (PIL `verify` + full decode).

### 3.5 Filenames, ID collisions, 11 `Oil_2` files

Names follow `{train|val|test}_{Oil|Oil_2} (N)`:
- `Oil`: N = 1…1257, every number present once.
- `Oil_2`: N = 1…11.
- Total: 1,257 + 11 = 1,268.

The prefixes are the uploader's split:

| Split | Images | OilSpill | LookAlikes | NoOilSpill |
|---|---|---|---|---|
| train | 811 | 633 | 154 | 24 |
| val | 203 | 163 | 35 | 5 |
| test | 254 | 202 | 40 | 12 |

The 11 numeric IDs that appear twice are exactly the `Oil (N)` / `Oil_2 (N)` pairs, N = 1–11. Each pair was compared on file hash, decoded-pixel hash, dimensions, 128×72 grey NCC and mask agreement:
- Result: **all 11 are different images sharing an ID.**
- `Oil_2` files are 3840×2160 and `Oil` files are 1920×1080.
- NCC ranges from −0.26 to 0.54; no hash is equal and no mask is equal.
- Five of the pairs sit across split prefixes: IDs 2, 3, 4, 9, 10.
- These collisions are **not leakage**. Details in `m4d_id_collisions.csv`.

### 3.6 Duplicates and leakage in the uploader split

- **Exact image duplicates:** 1 pair, `train_Oil (266)` = `train_Oil (267)` (same split).
- **Exact mask duplicates:** 17 files in 7 groups, all on different images.
  - Every one of them is a near-uniform whole-frame mask: 99.3% oil or 99.3% water, with the same border.
  - These are template frame labels, not copy errors. Details in `m4d_identical_masks.csv`.
- **Near duplicates.** Method: candidates had 16×16 NCC ≥ 0.9, then were confirmed at 128×72 grey NCC ≥ 0.97.
  - 649 near-duplicate pairs in 60 groups (largest 34), covering 359 images.
  - These are consecutive video frames: for 81% of images the best match is within ±3 IDs.
  - **312 of the pairs cross train/val/test.**
  - 44 groups span splits, holding 319 images.
  - Weakest confirmed cross-split pairs still have NCC 0.970 and mask agreement 0.87–0.99; e.g. `test_Oil (1188)` ↔ `val_Oil (1183)` is the same scene a few frames apart (see figure).
- **Confirmed leakage:** the uploader's train/val/test split leaks near-duplicate frames and must not be used.
- **Source-scene IDs:** there are none in filenames or metadata. The only proxy is the near-duplicate groups (`source_scene = m4d_nd<k>`, 969 groups, 909 singletons).
  - This is a lower bound on scene sharing: frames from one flight that differ by more than NCC 0.97 are not joined.
  - A scene-safe split would need contiguous-ID blocks plus visual clustering.
  - This dataset's own paper should be checked for its flight/sequence structure before anyone relies on it.

### 3.7 Overlap with Trujillo

- SHA-256: 0 shared files.
- Pixel similarity (16×16 NCC of M4D grey vs Trujillo VV 40 m thumbnails): max 0.95, and 56 M4D images score ≥ 0.9.
- These are low-frequency gradient matches between unrelated modalities (camera photos of a port vs 20 km SAR tiles). Hashes, sizes and content all rule out shared imagery.
- **No overlap.**

### 3.8 Visual audit (`audit/m4d_visual_audit.png`)

The figure shows image | mask | overlay for 5 seeded-random samples per folder class, not hand-picked, plus flagged cases:
- an ID collision pair;
- identical-mask pairs;
- the weakest confirmed cross-split near-duplicate;
- the maximum oil fraction;
- the minimum non-zero oil fraction.

Observations:
- Polygons are coarse but follow quays, hulls and water boundaries.
- Oil polygons cover large sheen areas, often most of the frame. Several are whole-frame labels that cannot be checked at thumbnail scale.
- The minimum-oil case (`train_Oil_2 (5)`, 0.1%) is a tiny patch between pontoons.

## 4. SegFormer compatibility and recommendation

- **Modality.** Our deployment target (Incident 001) is Sentinel-1 IW GRD SAR, and so is Trujillo. The "M4D" folder is oblique drone RGB with no radiometry, geometry or scale in common with SAR:
  - speckle vs sun glint;
  - dark slick vs coloured sheen;
  - 10 m vs centimetre GSD;
  - no georeferencing, so no polygon → OpenDrift path.
- Mixing, or pretraining on drone RGB, adds a large domain shift and little relevant signal. ImageNet pretraining already provides the generic low-level features. Its masks also have a different semantics: "others" includes ships and quays, "LookAlikes" contains no look-alikes.
- **Recommended E005 dataset: A — Trujillo only**, using our local files and the existing `acq` split.
  - "M4D" (Oil Spill Drone) is excluded from E005.
  - It could serve a separate drone/optical use case later, with its own scene-safe split.
- **Label formulation: binary** (oil = 1, else = 0; invalid pixels ignored).
  - Trujillo masks are binary, so no class is lost.
  - Oil probability goes directly to threshold → mask → polygon → OpenDrift, exactly as planned.
  - Multi-class is not possible on Trujillo, and "M4D" multi-class classes do not transfer to SAR.
- **Input representation: 2 channels, VV + validity**, from the existing 20 m cache (`vv20`/`lab20`, 1024²), standardised with train statistics, invalid → 0.
  - This is the same representation as E004.
  - VH is excluded because of the known VH test shift (`SAR_DATA_AUDIT.md`).
  - 3-channel pseudo-RGB (VV, VV, validity) is not needed: `smp.Segformer("mit_b2", in_channels=2)` builds and runs (24.7 M params), with the patch-embed weights adapted from ImageNet RGB.
  - Why 20 m and not 40 m: the median Trujillo oil fraction of 1.9% means small slicks, and 40 m would erase them.
- **Split:** `results/sar_audit/splits_v1_acq.parquet`, unchanged.
  - train 1,941 tiles (oil 844, no_oil 550, lookalike 547); val 477 (211 / 131 / 135).
  - Official test 450 (+ 152 excluded for trainval overlap) is evaluated once, in the 3 E004 variants.
  - Negatives (no_oil + lookalike, all-zero masks) stay in train. Look-alike tiles are the hard negatives the model must learn.
- **Variant:** SegFormer **MiT-B2** (smp 0.5.0, ImageNet encoder weights; no `transformers` needed). Chosen as the moderate-capacity baseline for the available RTX 5070 Ti (16 GB).
  - Proposed training: 512² random crops of 1024² tiles, oil-positive crop sampling (~50%), BCE(pos_weight) + Dice, ignore invalid, bf16, AdamW, seeded.
  - Select the checkpoint on val oil-IoU / Dice only.
  - Full-tile sliding-window inference.
- **Resources:**
  - Disk: 0 new data (the cache exists); ≈ 100 MB per checkpoint.
  - VRAM: ≈ 8–11 GB at batch 8, 512², bf16 on the RTX 5070 Ti (16 GB). This is an estimate, to be measured in a 1-epoch dry run.
  - Runtime: ≈ 1–2 min per epoch, so ≤ 1.5 h for 40 epochs with early stopping. Also an estimate.
