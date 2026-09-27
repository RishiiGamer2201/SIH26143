"""E002: non-destructive VV + validity/label cache for the SAR oil-spill dataset. Read-only on the TIFFs.

Per tile (row order = results/sar_audit/splits_v1_acq.parquet):
  valid  = finite(VH) & finite(VV) & VH != 0 & VV != 0        (band 1 = VH, band 2 = VV; SAR_DATA_AUDIT.md)
  tile cropped / padded (invalid) to 2048 x 2048 (4 tiles are not 2048^2; logged in meta)
  VV block-averaged in LINEAR POWER over valid sub-pixels; a block is valid if >= MIN_VALID of it is valid
  dB clipped to [-40, +5], scaled (dB + 40) / 45 -> [0, 1], float16; invalid blocks -> 0
  lab: 1 if mean(mask > 0) over valid sub-pixels >= 0.5 else 0; invalid blocks -> 255 (ignore)
       (raw mask of the tile's own class; oil-vs-background target is decided at training time)
Outputs (data/sar/cache/v1/): vv20.npy [N,1024,1024] f16, lab20.npy u8, vv40.npy [N,512,512] f16, lab40.npy u8,
  meta.parquet (per-tile stats + round-trip checks). Validity channel for models = (lab != 255).

Usage: python scripts/sar_build_cache.py [--workers 12] [--limit N]
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()
gdal.PushErrorHandler("CPLQuietErrorHandler")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/sar/cache/v1"
SPLITS = ROOT / "results/sar_audit/splits_v1_acq.parquet"
TILE, LO, HI, MIN_VALID = 2048, -40.0, 5.0, 0.5
FACTORS = {"20": 2, "40": 4}  # native 8.93e-5 deg: 9.93 m in y, 9.93*cos(lat) m in x (anisotropic, not resampled)


def fit(a, fill):
    """Crop / pad last two dims to TILE x TILE."""
    a = a[..., :TILE, :TILE]
    pad = [(0, 0)] * (a.ndim - 2) + [(0, TILE - a.shape[-2]), (0, TILE - a.shape[-1])]
    return np.pad(a, pad, constant_values=fill)


def downsample(lin, valid, pos, f):
    """lin/valid/pos: [TILE, TILE]. Returns scaled VV f16, label u8, stats."""
    n = TILE // f
    blk = lambda a: a.reshape(n, f, n, f).sum(axis=(1, 3))
    cnt = blk(valid.astype(np.float32))
    ok = cnt >= MIN_VALID * f * f
    with np.errstate(all="ignore"):
        p = blk(np.where(valid, lin, 0.0)) / cnt
        db = 10 * np.log10(p)
        posf = blk((pos & valid).astype(np.float32)) / cnt
    db_ok = db[ok]
    stats = dict(valid_frac=float(ok.mean()),
                 clip_lo_frac=float((db_ok < LO).mean()) if db_ok.size else np.nan,
                 clip_hi_frac=float((db_ok > HI).mean()) if db_ok.size else np.nan,
                 vv_p50_db=float(np.median(db_ok)) if db_ok.size else np.nan,
                 vv_meanlin_db=float(10 * np.log10(p[ok].mean())) if db_ok.size else np.nan,
                 pos_frac=float((posf[ok] >= 0.5).mean()) if db_ok.size else np.nan)
    vv = np.where(ok, (np.clip(db, LO, HI) - LO) / (HI - LO), 0.0).astype(np.float16)
    lab = np.where(ok, (posf >= 0.5).astype(np.uint8), np.uint8(255)).astype(np.uint8)
    return vv, lab, stats


def work(args):
    i, img, mask = args
    ds = gdal.Open(str(ROOT / img))
    vh, vv = (ds.GetRasterBand(b).ReadAsArray().astype(np.float32) for b in (1, 2))
    h, w = vv.shape
    gt = ds.GetGeoTransform()  # EPSG:4326, square-degree pixels -> x spacing shrinks with cos(lat)
    lat = gt[3] + gt[5] * h / 2
    px_x_m, px_y_m = abs(gt[1]) * 111320 * np.cos(np.radians(lat)), abs(gt[5]) * 110574
    mds = gdal.Open(str(ROOT / mask))  # keep ref: band read segfaults/raises if dataset is GC'd
    m = mds.GetRasterBand(1).ReadAsArray()
    valid = np.isfinite(vh) & np.isfinite(vv) & (vh != 0) & (vv != 0)
    with np.errstate(all="ignore"):
        lin = np.where(valid, 10 ** (vv / 10), 0.0).astype(np.float64)
    r = dict(row=i, h=h, w=w, lat_mid=lat, px_x_m=px_x_m, px_y_m=px_y_m, cropped_px=int(h * w - min(h, TILE) * min(w, TILE)),
             padded_px=int(TILE * TILE - min(h, TILE) * min(w, TILE)),
             valid_frac_native=float(valid.mean()),
             vv_p50_db_native=float(np.median(vv[valid])) if valid.any() else np.nan,
             vv_meanlin_db_native=float(10 * np.log10(lin[valid].mean())) if valid.any() else np.nan,
             pos_frac_native=float(((m > 0) & valid).sum() / max(valid.sum(), 1)),
             mask_values=",".join(map(str, np.unique(m))))
    lin, valid, pos = fit(lin, 0.0), fit(valid, False), fit(m > 0, False)
    for res, f in FACTORS.items():
        vvs, lab, st = downsample(lin, valid, pos, f)
        np.load(OUT / f"vv{res}.npy", mmap_mode="r+")[i] = vvs
        np.load(OUT / f"lab{res}.npy", mmap_mode="r+")[i] = lab
        r.update({f"{k}_{res}": v for k, v in st.items()})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    sp = pd.read_parquet(SPLITS).reset_index(drop=True)
    if a.limit:
        sp = sp.iloc[:a.limit]
    N = len(sp)
    OUT.mkdir(parents=True, exist_ok=True)
    for res, f in FACTORS.items():
        n = TILE // f
        np.lib.format.open_memmap(OUT / f"vv{res}.npy", "w+", np.float16, (N, n, n))
        np.lib.format.open_memmap(OUT / f"lab{res}.npy", "w+", np.uint8, (N, n, n))
    with Pool(a.workers) as pool:
        rows = pool.map(work, list(zip(range(N), sp["img"], sp["mask"])), chunksize=4)
    meta = pd.concat([sp[["split", "cls", "id", "img", "mask", "role", "fold"]], pd.DataFrame(rows).set_index("row")], axis=1)
    meta.to_parquet(OUT / "meta.parquet", index=False)
    json.dump(dict(tile=TILE, clip_db=[LO, HI], scale="(dB+40)/45", min_valid_block=MIN_VALID, factors=FACTORS,
                   valid_def="finite(VH)&finite(VV)&VH!=0&VV!=0", band_order="1=VH,2=VV", split_file=str(SPLITS.relative_to(ROOT)),
                   n=N), open(OUT / "cache_config.json", "w"), indent=1)
    print("cached", N, "tiles ->", OUT)


if __name__ == "__main__":
    main()
