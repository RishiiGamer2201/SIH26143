"""Phase C leakage: for spatially overlapping tile pairs, is it the SAME acquisition (duplicate
pixels) or the same place on a different date? Compares the shared window at 1/8 resolution.

Input : results/sar_audit/{sar_index,footprint_overlaps}.parquet (from sar_build_index.py + notebook step)
Output: results/sar_audit/overlap_content.parquet
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()
gdal.PushErrorHandler("CPLQuietErrorHandler")

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results/sar_audit"
DEC = 8
idx = pd.read_parquet(A / "sar_index.parquet")


def window(k, lon0, lat1, lon1, lat0):
    r = idx.iloc[k]
    c0 = int(round((lon0 - r.gt0) / r.gt1)); c1 = int(round((lon1 - r.gt0) / r.gt1))
    r0 = int(round((lat1 - r.gt3) / r.gt5)); r1 = int(round((lat0 - r.gt3) / r.gt5))
    c0, r0 = max(c0, 0), max(r0, 0); c1, r1 = min(c1, r.w), min(r1, r.h)
    ds = gdal.Open(str(ROOT / r.img))
    w, h = c1 - c0, r1 - r0
    return np.stack([ds.GetRasterBand(b).ReadAsArray(c0, r0, w, h, buf_xsize=max(w // DEC, 1), buf_ysize=max(h // DEC, 1),
                                                     resample_alg=gdal.GRIORA_Average) for b in (1, 2)])


def compare(p):
    i, j = map(int, p)
    a, b = idx.iloc[i], idx.iloc[j]
    lon0, lon1 = max(a.lon_min, b.lon_min), min(a.lon_max, b.lon_max)
    lat0, lat1 = max(a.lat_min, b.lat_min), min(a.lat_max, b.lat_max)
    x, y = window(i, lon0, lat1, lon1, lat0), window(j, lon0, lat1, lon1, lat0)
    h, w = min(x.shape[1], y.shape[1]), min(x.shape[2], y.shape[2])
    x, y = x[:, :h, :w], y[:, :h, :w]
    ok = (x != 0).all(0) & (y != 0).all(0)
    out = {"i": i, "j": j, "n_px": int(ok.sum())}
    if ok.sum() >= 50:
        for c in (0, 1):
            out[f"corr_c{c}"] = float(np.corrcoef(x[c][ok], y[c][ok])[0, 1])
            out[f"mad_c{c}"] = float(np.median(np.abs(x[c][ok] - y[c][ok])))
    return out


if __name__ == "__main__":
    pairs = pd.read_parquet(A / "footprint_overlaps.parquet")
    pairs = pairs[pairs.frac >= 0.05]
    with Pool(12) as pool:
        res = pd.DataFrame(pool.map(compare, list(zip(pairs.i, pairs.j)), chunksize=8))
    res = pairs.merge(res, on=["i", "j"])
    res.to_parquet(A / "overlap_content.parquet", index=False)
    print(len(res), "pairs compared")
