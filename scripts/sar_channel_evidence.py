"""Phase B: physics-based evidence for the band order of the Zenodo S1 oil-spill dataset.

Read-only. Samples images per class and measures, per channel:
  - backscatter level over clean sea (median dB)
  - oil damping contrast inside the mask (median outside - median inside) and its
    contrast-to-noise ratio (VV is expected to show the stronger, cleaner damping)
  - bright point-target excess over local sea clutter (cross-pol expected higher TCR)

Output: results/sar_audit/channel_evidence.parquet + printed summary.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()
gdal.PushErrorHandler("CPLQuietErrorHandler")  # TIFF ExtraSamples warning is benign

ROOT = Path(__file__).resolve().parents[1]
SAR = ROOT / "data/sar/extracted"
OUT = ROOT / "results/sar_audit"
N_PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = 26143


def read_img(p):
    ds = gdal.Open(str(p))
    return np.stack([ds.GetRasterBand(b).ReadAsArray() for b in (1, 2)]).astype(np.float32)


def read_mask(p):
    ds = gdal.Open(str(p))  # keep ref: band dies with its dataset
    return ds.GetRasterBand(1).ReadAsArray()


def analyse(item):
    cls, img_p, mask_p = item
    x = read_img(img_p)
    valid = np.isfinite(x).all(0) & (x != 0).all(0)
    rec = {"cls": cls, "file": str(img_p.relative_to(SAR)), "valid_frac": float(valid.mean())}
    m = read_mask(mask_p) if mask_p else None
    oil = (m > 0) & valid if m is not None else np.zeros_like(valid)
    sea = valid & ~oil
    rec["oil_frac"] = float(oil.mean())
    for c in (0, 1):
        v = x[c]
        s = v[sea]
        med = float(np.median(s)) if s.size else np.nan
        mad = float(np.median(np.abs(s - med)) * 1.4826) if s.size else np.nan
        rec[f"c{c}_sea_med"] = med
        rec[f"c{c}_sea_mad"] = mad
        rec[f"c{c}_p01"], rec[f"c{c}_p99"] = (np.percentile(s, [1, 99]).tolist() if s.size else [np.nan] * 2)
        if oil.sum() > 500:
            o = float(np.median(v[oil]))
            rec[f"c{c}_oil_contrast_db"] = med - o
            rec[f"c{c}_oil_cnr"] = (med - o) / mad if mad > 0 else np.nan
    # Bright targets: pixels >= 12 dB above sea median in EITHER channel (channel-neutral selection)
    tgt = sea & ((x[0] - rec["c0_sea_med"] >= 12) | (x[1] - rec["c1_sea_med"] >= 12))
    rec["n_target_px"] = int(tgt.sum())
    if tgt.sum() >= 20:
        for c in (0, 1):
            rec[f"c{c}_target_excess_db"] = float(np.median(x[c][tgt] - rec[f"c{c}_sea_med"]))
    rec["c0_minus_c1_sea"] = rec["c0_sea_med"] - rec["c1_sea_med"]
    return rec


def items():
    rng = np.random.default_rng(SEED)
    spec = [
        ("oil", SAR / "Oil", lambda f: SAR / "Mask_oil" / f.name),
        ("lookalike", SAR / "Lookalike", None),
        ("no_oil", SAR / "No_oil", None),
        ("test_oil", SAR / "Images/Oil", lambda f: SAR / "Mask/Oil" / f"{f.stem}_segmentation.tif"),
    ]
    out = []
    for cls, d, mk in spec:
        files = sorted(d.glob("*.tif"))
        pick = rng.choice(len(files), size=min(N_PER_CLASS, len(files)), replace=False)
        out += [(cls, files[i], mk(files[i]) if mk else None) for i in sorted(pick)]
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(8) as pool:
        df = pd.DataFrame(pool.map(analyse, items()))
    df.to_parquet(OUT / "channel_evidence.parquet", index=False)

    pd.set_option("display.width", 200)
    print(df.groupby("cls")[["valid_frac", "c0_sea_med", "c1_sea_med", "c0_minus_c1_sea", "c0_sea_mad", "c1_sea_mad"]].median().round(2))
    o = df.dropna(subset=["c0_oil_contrast_db", "c1_oil_contrast_db"])
    print(f"\nOIL DAMPING (n={len(o)} oil images with >500 oil px)")
    print(o[["c0_oil_contrast_db", "c1_oil_contrast_db", "c0_oil_cnr", "c1_oil_cnr"]].describe().round(2))
    print("frac images contrast c0 > c1:", round(float((o.c0_oil_contrast_db > o.c1_oil_contrast_db).mean()), 3))
    print("frac images CNR c0 > c1     :", round(float((o.c0_oil_cnr > o.c1_oil_cnr).mean()), 3))
    t = df.dropna(subset=["c0_target_excess_db", "c1_target_excess_db"])
    print(f"\nBRIGHT TARGETS (n={len(t)} images with >=20 target px)")
    print(t[["c0_target_excess_db", "c1_target_excess_db"]].describe().round(2))
    print("frac images target excess c1 > c0:", round(float((t.c1_target_excess_db > t.c0_target_excess_db).mean()), 3))
    print("frac images sea level c0 > c1   :", round(float((df.c0_sea_med > df.c1_sea_med).mean()), 3))
