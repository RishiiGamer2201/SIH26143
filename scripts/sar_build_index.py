"""Phase C: build a cached per-image index of the SAR oil-spill dataset. Read-only on the TIFFs.

Outputs (results/sar_audit/):
  sar_index.parquet  one row per image: split, class, paths, georeferencing, footprint,
                     nodata/NaN/inf/zero counts, per-channel robust stats, mask stats
  thumbs.npy         float16 [N, 2, 64, 64] block-mean thumbnails (nodata -> NaN), row-aligned
                     with sar_index.parquet; used for duplicate / overlap detection

Channels are kept as channel_0 / channel_1 (see SAR_DATA_AUDIT.md for the VV/VH evidence).
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()
gdal.PushErrorHandler("CPLQuietErrorHandler")

ROOT = Path(__file__).resolve().parents[1]
SAR = ROOT / "data/sar/extracted"
OUT = ROOT / "results/sar_audit"
THUMB = 64

SPLITS = [
    ("trainval", "oil", SAR / "Oil", lambda f: SAR / "Mask_oil" / f.name),
    ("trainval", "lookalike", SAR / "Lookalike", lambda f: SAR / "Mask_lookalike" / f.name),
    ("trainval", "no_oil", SAR / "No_oil", lambda f: SAR / "Mask_no_oil" / f.name),
    ("test", "oil", SAR / "Images/Oil", lambda f: SAR / "Mask/Oil" / f"{f.stem}_segmentation.tif"),
    ("test", "lookalike", SAR / "Images/Lookalike", lambda f: SAR / "Mask/Lookalike" / f"{f.stem}_segmentation.tif"),
    ("test", "no_oil", SAR / "Images/No oil", lambda f: SAR / "Mask/No oil" / f"{f.stem}_segmentation.tif"),
]


def scan(item):
    split, cls, img_p, mask_p = item
    ds = gdal.Open(str(img_p))
    x = np.stack([ds.GetRasterBand(b).ReadAsArray() for b in range(1, ds.RasterCount + 1)])
    gt = ds.GetGeoTransform()
    r = {
        "split": split, "cls": cls, "id": img_p.stem,
        "img": str(img_p.relative_to(ROOT)), "mask": str(mask_p.relative_to(ROOT)),
        "bands": ds.RasterCount, "dtype": x.dtype.name, "h": ds.RasterYSize, "w": ds.RasterXSize,
        "epsg": ds.GetSpatialRef().GetAuthorityCode(None) if ds.GetSpatialRef() else None,
        "nodata_tag": ds.GetRasterBand(1).GetNoDataValue(),
        "gt0": gt[0], "gt1": gt[1], "gt2": gt[2], "gt3": gt[3], "gt4": gt[4], "gt5": gt[5],
    }
    r["lon_min"], r["lat_max"] = gt[0], gt[3]
    r["lon_max"], r["lat_min"] = gt[0] + gt[1] * r["w"], gt[3] + gt[5] * r["h"]
    finite = np.isfinite(x)
    zero = x == 0
    valid = finite.all(0) & ~zero.all(0)
    r["valid_frac"] = float(valid.mean())
    for c in range(x.shape[0]):
        v = x[c]
        r[f"c{c}_nan"] = int(np.isnan(v).sum())
        r[f"c{c}_inf"] = int(np.isinf(v).sum())
        r[f"c{c}_zero"] = int(zero[c].sum())
        s = v[valid]
        if s.size:
            q = np.percentile(s, [0.1, 1, 5, 50, 95, 99, 99.9])
            for k, val in zip(["p001", "p01", "p05", "p50", "p95", "p99", "p999"], q):
                r[f"c{c}_{k}"] = float(val)
            r[f"c{c}_min"], r[f"c{c}_max"] = float(s.min()), float(s.max())
            r[f"c{c}_mean"], r[f"c{c}_std"] = float(s.mean()), float(s.std())
    r["both_zero_px"] = int(zero.all(0).sum())
    r["one_zero_px"] = int((zero.any(0) & ~zero.all(0)).sum())

    mds = gdal.Open(str(mask_p))
    m = mds.GetRasterBand(1).ReadAsArray()
    r["mask_shape_ok"] = m.shape == (r["h"], r["w"])
    r["mask_georef"] = mds.GetGeoTransform(can_return_null=True) is not None
    r["mask_values"] = ",".join(map(str, np.unique(m)))
    r["mask_pos_frac"] = float((m > 0).mean())
    r["mask_pos_on_invalid"] = int(((m > 0) & ~valid).sum())

    bh, bw = r["h"] // THUMB, r["w"] // THUMB  # non-2048 images: trailing rows/cols dropped from thumb only
    t = np.where(valid, x, np.nan)[:, :bh * THUMB, :bw * THUMB].reshape(x.shape[0], THUMB, bh, THUMB, bw)
    with np.errstate(all="ignore"):
        thumb = np.nanmean(t, axis=(2, 4)).astype(np.float16)
    return r, thumb


def items():
    out = []
    for split, cls, d, mk in SPLITS:
        out += [(split, cls, f, mk(f)) for f in sorted(d.glob("*.tif"))]
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    its = items()
    print("images:", len(its))
    with Pool(12) as pool:
        res = pool.map(scan, its, chunksize=4)
    df = pd.DataFrame([r for r, _ in res])
    np.save(OUT / "thumbs.npy", np.stack([t for _, t in res]))
    df.to_parquet(OUT / "sar_index.parquet", index=False)
    print(df.groupby(["split", "cls"]).size())
    print("done ->", OUT)
