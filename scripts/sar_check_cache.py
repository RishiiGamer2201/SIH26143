"""E002 integrity checks for data/sar/cache/v1 -> results/sar_cache_v1/{cache_checks.json, samples.png}."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "data/sar/cache/v1"
OUT = ROOT / "results/sar_cache_v1"
rng = np.random.default_rng(26143)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(C / "meta.parquet")
    sp = pd.read_parquet(ROOT / "results/sar_audit/splits_v1_acq.parquet")
    idx = pd.read_parquet(ROOT / "results/sar_audit/sar_index.parquet")
    chk = {}
    chk["n"] = len(meta)
    assert len(meta) == len(sp) == 3020
    assert (meta[["split", "cls", "id"]].values == sp[["split", "cls", "id"]].values).all(), "row order != split file"

    arrs = {k: np.load(C / f"{k}.npy", mmap_mode="r") for k in ("vv20", "lab20", "vv40", "lab40")}
    for k, a in arrs.items():
        chk[f"{k}_shape"], chk[f"{k}_dtype"] = list(a.shape), a.dtype.name
    assert arrs["vv20"].shape == (3020, 1024, 1024) and arrs["vv40"].shape == (3020, 512, 512)

    for res in ("20", "40"):
        vv, lab = arrs[f"vv{res}"], arrs[f"lab{res}"]
        nonfinite = bad_range = bad_lab = vv_on_invalid = 0
        vfrac, pfrac = [], []
        for s in range(0, len(vv), 64):
            v = vv[s:s + 64].astype(np.float32)
            l = np.asarray(lab[s:s + 64])
            nonfinite += int((~np.isfinite(v)).sum())
            bad_range += int(((v < 0) | (v > 1)).sum())
            bad_lab += int((~np.isin(l, (0, 1, 255))).sum())
            vv_on_invalid += int((v[l == 255] != 0).sum())
            ok = l != 255
            vfrac += list(ok.mean(axis=(1, 2)))
            pfrac += list(np.where(ok.any(axis=(1, 2)), (l == 1).sum(axis=(1, 2)) / np.maximum(ok.sum(axis=(1, 2)), 1), np.nan))
        chk[f"{res}_nonfinite"], chk[f"{res}_out_of_range"] = nonfinite, bad_range
        chk[f"{res}_bad_label_values"], chk[f"{res}_nonzero_vv_on_invalid"] = bad_lab, vv_on_invalid
        chk[f"{res}_max_abs_validfrac_diff_vs_meta"] = float(np.nanmax(np.abs(np.array(vfrac) - meta[f"valid_frac_{res}"])))
        chk[f"{res}_max_abs_posfrac_diff_vs_meta"] = float(np.nanmax(np.abs(np.array(pfrac) - meta[f"pos_frac_{res}"])))
        assert nonfinite == bad_range == bad_lab == vv_on_invalid == 0
        d = (meta[f"vv_meanlin_db_{res}"] - meta.vv_meanlin_db_native).abs()
        chk[f"{res}_meanlin_db_roundtrip_max_abs"] = float(d.max())
        chk[f"{res}_p50_db_shift_median"] = float((meta[f"vv_p50_db_{res}"] - meta.vv_p50_db_native).median())
        chk[f"{res}_clip_lo_frac_mean"] = float(meta[f"clip_lo_frac_{res}"].mean())
        chk[f"{res}_clip_hi_frac_mean"] = float(meta[f"clip_hi_frac_{res}"].mean())
        chk[f"{res}_tiles_clip_gt_0p1pct"] = int(((meta[f"clip_lo_frac_{res}"] + meta[f"clip_hi_frac_{res}"]) > 1e-3).sum())
        chk[f"{res}_posfrac_vs_native_max_abs"] = float((meta[f"pos_frac_{res}"] - meta.pos_frac_native).abs().max())

    # validity definition vs audit index (index: finite both & not both-zero; cache: finite both & neither zero)
    m = meta.merge(idx[["split", "cls", "id", "valid_frac", "c1_p50"]], on=["split", "cls", "id"])
    chk["validfrac_native_vs_index_max_abs"] = float((m.valid_frac_native - m.valid_frac).abs().max())
    chk["vv_p50_native_vs_index_c1_p50_max_abs"] = float((m.vv_p50_db_native - m.c1_p50).abs().max())
    chk["all_invalid_tiles"] = meta.loc[meta.valid_frac_20 == 0, ["split", "cls", "id"]].to_dict("records")
    chk["cropped_or_padded_tiles"] = meta.loc[(meta.cropped_px > 0) | (meta.padded_px > 0),
                                              ["split", "cls", "id", "h", "w", "cropped_px", "padded_px"]].to_dict("records")
    chk["mask_values_seen"] = sorted(set(",".join(meta.mask_values).split(",")))

    # independent recompute of random 40 m blocks from the raw TIFF
    errs = []
    for i in rng.choice(np.flatnonzero(meta.valid_frac_40 > 0.5), 20, replace=False):
        ds = gdal.Open(str(ROOT / meta.img[i]))
        vh, vv = ds.GetRasterBand(1).ReadAsArray(), ds.GetRasterBand(2).ReadAsArray()
        lab = arrs["lab40"][i]
        r, c = [int(x) for x in rng.choice(np.argwhere(lab != 255))]
        bvv, bvh = vv[r * 4:r * 4 + 4, c * 4:c * 4 + 4], vh[r * 4:r * 4 + 4, c * 4:c * 4 + 4]
        ok = np.isfinite(bvv) & np.isfinite(bvh) & (bvv != 0) & (bvh != 0)
        db = np.clip(10 * np.log10(np.mean(10 ** (bvv[ok] / 10))), -40, 5)
        errs.append(abs(float(arrs["vv40"][i, r, c]) * 45 - 40 - db))
    chk["independent_block_recompute_max_abs_db"] = float(max(errs))
    assert max(errs) < 0.05, errs  # float16 quantisation of [0,1] ~ 0.0005 * 45 dB

    # statistical view: VV p50 (20 m) by split x class
    chk["vv_p50_db_20_by_split_class"] = meta.groupby(["split", "cls"]).vv_p50_db_20.median().round(2).unstack().to_dict()
    chk["px_x_m_by_split_class"] = meta.groupby(["split", "cls"]).px_x_m.median().round(2).unstack().to_dict()
    json.dump(chk, open(OUT / "cache_checks.json", "w"), indent=1, default=str)

    # samples: 2 train + 1 test per class, VV 20 m with label contour
    picks = []
    for cls in ("oil", "lookalike", "no_oil"):
        picks += list(meta[(meta.role == "train") & (meta.cls == cls) & (meta.valid_frac_20 > 0.9)].sample(2, random_state=0).index)
        picks += list(meta[(meta.role == "test") & (meta.cls == cls) & (meta.valid_frac_20 > 0.9)].sample(1, random_state=0).index)
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for a, i in zip(ax.ravel(), picks):
        v, l = arrs["vv20"][i].astype(np.float32) * 45 - 40, arrs["lab20"][i]
        a.imshow(np.where(l == 255, np.nan, v), cmap="gray", vmin=-30, vmax=-5)
        if (l == 1).any():
            a.contour(l == 1, levels=[0.5], colors="r", linewidths=0.6)
        a.set_title(f"{meta.split[i]}/{meta.cls[i]}/{meta.id[i]} role={meta.role[i]}\nvalid {meta.valid_frac_20[i]:.2f} pos {meta.pos_frac_20[i]:.3f}", fontsize=9)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "samples.png", dpi=80)
    print(json.dumps(chk, indent=1, default=str))


if __name__ == "__main__":
    main()
