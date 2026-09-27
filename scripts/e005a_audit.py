"""E005-A: mentor-dataset audit (shuban220409/combinedoilspilldataset). No training.

  python scripts/e005a_audit.py trujillo   # Kaggle SHA-256 (from audit kernel) vs our local files
  python scripts/e005a_audit.py m4d        # scan every M4D image + mask -> tables in AUD

Inputs : data/segformer_combined/kernel_out/{trujillo_sha256.csv, m4d.zip}
         data/segformer_combined/local_trujillo_sha256.csv
Outputs: data/segformer_combined/audit/
"""
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
SC = ROOT / "data/segformer_combined"
AUD = SC / "audit"
M4D = SC / "m4d_raw/dataset"
CLS_TRAINVAL = {"LookAlikes": "Lookalike", "NoOilSpill": "No_oil", "OilSpill": "Oil"}
CLS_TEST = {"LookAlikes": "Lookalike", "NoOilSpill": "No oil", "OilSpill": "Oil"}
NAME_RE = re.compile(r"^(train|val|test)_(Oil|Oil_2) \((\d+)\)$")


def kaggle_to_local(rel):
    """Kaggle 'Sentinel-1_SAR_Oil_spill_image_dataset/[Test/]Cls/Kind/[archive[2]_]name' -> local rel path."""
    parts = rel.replace("\\", "/").split("/")[1:]
    test = parts[0] == "Test"
    cls, kind, name = parts[1:] if test else parts
    name = re.sub(r"^archive2?_", "", name)
    if test:
        return f"{'Images' if kind == 'Tiff files' else 'Mask'}/{CLS_TEST[cls]}/{name}"
    lc = CLS_TRAINVAL[cls]
    return f"{lc}/{name}" if kind == "Tiff files" else f"Mask_{lc.lower()}/{name}"


def trujillo():
    k = pd.read_csv(SC / "kernel_out/trujillo_sha256.csv")
    loc = pd.read_csv(SC / "local_trujillo_sha256.csv")
    k["local_rel"] = k.rel_path.map(kaggle_to_local)
    assert not k.local_rel.duplicated().any(), "mapping not one-to-one"
    j = k.merge(loc, left_on="local_rel", right_on="rel_path", how="outer",
                suffixes=("_kaggle", "_local"), indicator=True)
    both = j[j._merge == "both"]
    res = dict(kaggle_files=len(k), local_files=len(loc), compared=len(both),
               sha256_match=int((both.sha256_kaggle == both.sha256_local).sum()),
               size_match=int((both.size_kaggle == both.size_local).sum()),
               sha256_mismatch=int((both.sha256_kaggle != both.sha256_local).sum()),
               unmatched_kaggle=int((j._merge == "left_only").sum()),
               unmatched_local=int((j._merge == "right_only").sum()))
    AUD.mkdir(parents=True, exist_ok=True)
    j.drop(columns="_merge").to_csv(AUD / "trujillo_sha256_compare.csv", index=False)
    (AUD / "trujillo_identity.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    bad = j[(j._merge != "both") | (j.sha256_kaggle != j.sha256_local)]
    if len(bad):
        print(bad.head(20).to_string())


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def thumb(gray, n=16):
    """n x n area-mean thumbnail, zero-mean unit-norm (for NCC near-duplicate search)."""
    t = np.asarray(Image.fromarray(gray).resize((n, n), Image.BOX), np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def dhash(gray):
    a = np.asarray(Image.fromarray(gray).resize((9, 8), Image.BOX), np.int16)
    return int("".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).ravel()), 2)


def scan_image(p):
    r = dict(path=str(p.relative_to(SC)), bytes=p.stat().st_size, sha256=sha256(p))
    try:
        with Image.open(p) as im:
            im.verify()
        im = Image.open(p)
        r.update(format=im.format, mode=im.mode, width=im.width, height=im.height,
                 exif=bool(im.info.get("exif")), icc=bool(im.info.get("icc_profile")),
                 jfif_density=str(im.info.get("dpi", "")))
        a = np.asarray(im)
    except Exception as e:  # corrupt file: record, keep going
        return dict(r, error=repr(e)), None
    ch = 1 if a.ndim == 2 else a.shape[2]
    r.update(channels=ch, dtype=str(a.dtype), min=int(a.min()), max=int(a.max()),
             pixel_sha256=hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest())
    if ch >= 3:
        c = a[..., :3].astype(np.int16)
        r.update(max_abs_rg=int(np.abs(c[..., 0] - c[..., 1]).max()),
                 max_abs_rb=int(np.abs(c[..., 0] - c[..., 2]).max()),
                 frac_r_eq_g_eq_b=float(((c[..., 0] == c[..., 1]) & (c[..., 1] == c[..., 2])).mean()),
                 mean_abs_rg=float(np.abs(c[..., 0] - c[..., 1]).mean()))
        s = c[::4, ::4].reshape(-1, 3).astype(np.float32).T
        cc = np.corrcoef(s)
        r.update(corr_rg=float(cc[0, 1]), corr_rb=float(cc[0, 2]), corr_gb=float(cc[1, 2]))
        r["alpha"] = ch == 4
    gray = np.asarray(im.convert("L"))
    r.update(gray_mean=float(gray.mean()), gray_std=float(gray.std()), dhash=dhash(gray))
    return r, thumb(gray)


def scan_mask(p):
    r = dict(path=str(p.relative_to(SC)), bytes=p.stat().st_size, sha256=sha256(p))
    try:
        with Image.open(p) as im:
            im.verify()
        im = Image.open(p)
        r.update(format=im.format, mode=im.mode, width=im.width, height=im.height)
        raw = np.asarray(im)
        a = np.asarray(im.convert("RGB")).reshape(-1, 3).astype(np.uint32)
    except Exception as e:
        return dict(r, error=repr(e)), {}
    r.update(raw_shape=str(raw.shape), raw_dtype=str(raw.dtype),
             pixel_sha256=hashlib.sha256(raw.tobytes() + str(raw.shape).encode()).hexdigest())
    if raw.ndim == 3 and raw.shape[2] == 4:
        r["alpha_values"] = ",".join(map(str, np.unique(raw[..., 3])))
    code, cnt = np.unique((a[:, 0] << 16) | (a[:, 1] << 8) | a[:, 2], return_counts=True)
    r["n_colours"] = len(code)
    return r, {f"{c >> 16},{(c >> 8) & 255},{c & 255}": int(n) for c, n in zip(code, cnt)}


def m4d():
    AUD.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(M4D.iterdir()):
        for kind, sub in (("image", "Tiff files"), ("mask", "Masks")):
            for p in sorted((d / sub).iterdir()):
                m = NAME_RE.match(p.stem)
                rows.append(dict(cls=d.name, kind=kind, p=p, stem=p.stem, ext=p.suffix,
                                 split_original=m and m[1], variant=m and m[2], num=m and int(m[3])))
    f = pd.DataFrame(rows)
    img, msk = f[f.kind == "image"], f[f.kind == "mask"]
    with ProcessPoolExecutor(8) as ex:
        ri = list(ex.map(scan_image, img.p, chunksize=8))
        rm = list(ex.map(scan_mask, msk.p, chunksize=8))
    ti = pd.concat([img.drop(columns="p").reset_index(drop=True), pd.DataFrame([r for r, _ in ri])], axis=1)
    tm = pd.concat([msk.drop(columns="p").reset_index(drop=True), pd.DataFrame([r for r, _ in rm])], axis=1)
    np.save(AUD / "m4d_thumbs16.npy", np.stack([t if t is not None else np.zeros(256, np.float32) for _, t in ri]))
    col = pd.DataFrame([dict(mask=r["path"], rgb=k, count=n) for r, c in rm for k, n in c.items()])
    ti.to_parquet(AUD / "m4d_images.parquet")
    tm.to_parquet(AUD / "m4d_masks.parquet")
    col.to_parquet(AUD / "m4d_mask_colours.parquet")
    print("images", len(ti), "masks", len(tm), "colour rows", len(col))


PALETTE = {(0, 0, 0): 0, (255, 0, 124): 1, (255, 204, 51): 2, (51, 221, 255): 3}  # verified from pixels
CLASS_NAMES = ["background", "oil", "others", "water"]  # names from uploader doc; meaning checked visually
SMALL = (128, 72)
ND_NCC = 0.97  # near-duplicate: 128x72 grey NCC >= this (confirmed at 128x72, candidates from 16x16)


def small_pair(args):
    """128x72 zero-mean unit-norm grey image + nearest-resized class-index mask."""
    ip, mp = args
    im = Image.open(ip)
    im.draft("RGB", (SMALL[0] * 2, SMALL[1] * 2))
    g = np.asarray(im.convert("L").resize(SMALL, Image.BOX), np.float32).ravel()
    g = (g - g.mean()) / (np.linalg.norm(g - g.mean()) + 1e-6)
    a = np.asarray(Image.open(mp).convert("RGB")).astype(np.uint32)
    code = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    idx = np.full(code.shape, 255, np.uint8)
    for (r, gg, b), k in PALETTE.items():
        idx[code == (r << 16 | gg << 8 | b)] = k
    return g, np.asarray(Image.fromarray(idx).resize(SMALL, Image.NEAREST)).ravel()


def components(n, edges):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    e = np.asarray(edges, int).reshape(-1, 2)
    return connected_components(csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)[1]


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ti = pd.read_parquet(AUD / "m4d_images.parquet")
    tm = pd.read_parquet(AUD / "m4d_masks.parquet").set_index(["cls", "stem"]).loc[list(zip(ti.cls, ti.stem))].reset_index()
    assert (tm.width.values == ti.width.values).all() and (tm.height.values == ti.height.values).all()
    col = pd.read_parquet(AUD / "m4d_mask_colours.parquet")
    assert set(col.rgb) == {",".join(map(str, k)) for k in PALETTE}, "unexpected mask colour"
    T16 = np.load(AUD / "m4d_thumbs16.npy")
    n, sp = len(ti), ti.split_original.values
    with ProcessPoolExecutor(8) as ex:
        sm = list(ex.map(small_pair, zip(SC / ti.path, SC / tm.path), chunksize=8))
    G = np.stack([g for g, _ in sm])
    M = np.stack([m for _, m in sm])
    assert (M != 255).all()

    # class fractions per mask
    frac = np.stack([(M == k).mean(1) for k in range(4)], 1)  # from 128x72 masks (display only)
    cnt = col.assign(k=col.rgb.map(lambda s: CLASS_NAMES[PALETTE[tuple(map(int, s.split(",")))]]))
    cnt = cnt.pivot_table(index="mask", columns="k", values="count", aggfunc="sum", fill_value=0)
    cnt = cnt.div(cnt.sum(axis=1), axis=0).loc[tm.path]
    ti["oil_frac"] = cnt.oil.values
    for k in CLASS_NAMES:
        ti[f"frac_{k}"] = cnt[k].values

    # pairs: candidates from 16x16 NCC >= 0.9, confirmed at 128x72
    S16 = T16 @ T16.T
    i, j = np.where(np.triu(S16 >= 0.9, 1))
    pr = pd.DataFrame(dict(i=i, j=j, ncc16=S16[i, j], ncc128=np.einsum("ij,ij->i", G[i], G[j]),
                           mask_agree=(M[i] == M[j]).mean(1)))
    pr["cross_split"] = sp[pr.i] != sp[pr.j]
    pr["near_dup"] = pr.ncc128 >= ND_NCC
    pr["a"], pr["b"] = ti.stem.values[pr.i], ti.stem.values[pr.j]
    pr.to_parquet(AUD / "m4d_pairs.parquet")
    nd = pr[pr.near_dup]
    ti["nd_group"] = components(n, nd[["i", "j"]].values)
    ti["pixel_dup_group"] = ti.groupby("pixel_sha256").ngroup()

    # identical masks: are the images near-dups?
    tm["img_i"] = range(n)
    idm = []
    for h, g in tm[tm.sha256.duplicated(keep=False)].groupby("sha256"):
        k = g.img_i.values
        for b in k[1:]:
            idm.append(dict(mask_sha=h[:12], a=ti.stem[k[0]], b=ti.stem[b], split_a=sp[k[0]], split_b=sp[b],
                            img_ncc128=float(G[k[0]] @ G[b]), img_pixel_equal=ti.pixel_sha256[k[0]] == ti.pixel_sha256[b]))
    idm = pd.DataFrame(idm)

    # overlap with Trujillo (VV cache, 40 m): 16x16 thumbnails, invalid -> tile mean
    vv = np.load(ROOT / "data/sar/cache/v1/vv40.npy", mmap_mode="r")
    lab = np.load(ROOT / "data/sar/cache/v1/lab40.npy", mmap_mode="r")
    TT = np.empty((len(vv), 256), np.float32)
    for s in range(0, len(vv), 256):
        v = vv[s:s + 256].astype(np.float32)
        ok = lab[s:s + 256] != 255
        v = np.where(ok, v, (v * ok).sum((1, 2), keepdims=True) / np.maximum(ok.sum((1, 2), keepdims=True), 1))
        t = v.reshape(len(v), 16, 32, 16, 32).mean((2, 4)).reshape(len(v), -1)
        t -= t.mean(1, keepdims=True)
        TT[s:s + 256] = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-6)
    X = T16 @ TT.T
    trj = dict(max_ncc16=float(X.max()), n_m4d_with_ncc16_ge_0_9=int((X.max(1) >= 0.9).sum()),
               sha_overlap=int(len(set(ti.sha256) & set(pd.read_csv(SC / "local_trujillo_sha256.csv").sha256))))

    # ID collisions (same numeric ID, several files)
    coll = []
    for num, g in ti.groupby("num"):
        k = g.index.values
        for b in k[1:]:
            coll.append(dict(num=num, a=ti.stem[k[0]], b=ti.stem[b], split_a=sp[k[0]], split_b=sp[b],
                             cls_a=ti.cls[k[0]], cls_b=ti.cls[b], size_a=f"{ti.width[k[0]]}x{ti.height[k[0]]}",
                             size_b=f"{ti.width[b]}x{ti.height[b]}", img_sha_equal=ti.sha256[k[0]] == ti.sha256[b],
                             img_pixel_equal=ti.pixel_sha256[k[0]] == ti.pixel_sha256[b],
                             img_ncc128=float(G[k[0]] @ G[b]), mask_sha_equal=tm.sha256[k[0]] == tm.sha256[b],
                             mask_agree128=float((M[k[0]] == M[b]).mean())))
    coll = pd.DataFrame(coll)
    coll["verdict"] = np.where(coll.img_pixel_equal, "exact duplicate",
                               np.where(coll.img_ncc128 >= ND_NCC, "near duplicate", "different images sharing an ID"))

    gs = ti.groupby("nd_group").split_original.nunique()
    summ = dict(
        pairs=n, trujillo=trj, near_dup_threshold_ncc128=ND_NCC,
        near_dup_pairs=int(len(nd)), near_dup_pairs_cross_split=int(nd.cross_split.sum()),
        images_in_near_dup_groups=int(ti.nd_group.map(ti.nd_group.value_counts()).gt(1).sum()),
        nd_groups=int(ti.nd_group.nunique()), nd_groups_multi=int((ti.nd_group.value_counts() > 1).sum()),
        largest_nd_group=int(ti.nd_group.value_counts().max()),
        nd_groups_spanning_splits=int((gs > 1).sum()),
        images_in_split_spanning_groups=int(ti.nd_group.isin(gs[gs > 1].index).sum()),
        exact_image_dup_files=int(ti.sha256.duplicated(keep=False).sum()),
        pixel_image_dup_files=int(ti.pixel_sha256.duplicated(keep=False).sum()),
        exact_mask_dup_files=int(tm.sha256.duplicated(keep=False).sum()),
        identical_mask_pairs_image_not_near_dup=int((idm.img_ncc128 < ND_NCC).sum()),
        id_collisions=coll.verdict.value_counts().to_dict(),
        id_collisions_cross_split=int((coll.split_a != coll.split_b).sum()),
        class_masks_containing={c: {k: int((ti[ti.cls == c][f"frac_{k}"] > 0).sum()) for k in CLASS_NAMES}
                                for c in sorted(ti.cls.unique())},
        oil_frac_by_folder=ti.groupby("cls").oil_frac.describe().round(4).to_dict("index"),
        oil_pixel_share_total=float((ti.oil_frac * ti.width * ti.height).sum() / (ti.width * ti.height).sum()),
    )
    (AUD / "m4d_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    coll.to_csv(AUD / "m4d_id_collisions.csv", index=False)
    idm.to_csv(AUD / "m4d_identical_masks.csv", index=False)
    print(json.dumps(summ, indent=1, default=str))
    print(coll.to_string())
    print(idm.to_string())
    print(nd[nd.cross_split].sort_values("ncc128").head(8).to_string())

    # figure: 5 random per folder class (seeded, not curated) + flagged cases
    rng = np.random.default_rng(26143)
    rows = [(f"{c}: random", k) for c in sorted(ti.cls.unique())
            for k in rng.choice(ti.index[ti.cls == c], 5, replace=False)]
    flag = [("ID collision", ti.index[ti.stem == coll.a[0]][0]), ("ID collision", ti.index[ti.stem == coll.b[0]][0])]
    for r in idm.head(2).itertuples():
        flag += [("identical mask", ti.index[ti.stem == r.a][0]), ("identical mask", ti.index[ti.stem == r.b][0])]
    x = nd[nd.cross_split].sort_values("ncc128").iloc[0]
    flag += [("cross-split near-dup", x.i), ("cross-split near-dup", x.j)]
    flag += [("max oil frac", ti.oil_frac.idxmax()), ("min non-zero oil frac", ti.oil_frac[ti.oil_frac > 0].idxmin())]
    rows += flag
    nr = -(-len(rows) // 2)
    fig, ax = plt.subplots(nr, 6, figsize=(18, 1.75 * nr))
    for h in ax.ravel():
        h.axis("off")
    for q, (tag, k) in enumerate(rows):
        im = Image.open(SC / ti.path[k])
        im.draft("RGB", (480, 270))
        im = np.asarray(im.convert("RGB").resize((480, 270)))
        mk = np.asarray(Image.open(SC / tm.path[k]).convert("RGB").resize((480, 270), Image.NEAREST))
        ov = (0.55 * im + 0.45 * mk).astype(np.uint8)
        for c, (a, t) in enumerate(((im, f"{tag}\n{ti.stem[k]} [{ti.cls[k]}]"), (mk, f"mask oil={ti.oil_frac[k]:.3f}"), (ov, "overlay"))):
            h = ax[q // 2, (q % 2) * 3 + c]
            h.imshow(a)
            h.set_title(t, fontsize=7)
            h.axis("off")
    fig.suptitle("M4D (mentor dataset 'dataset' folder): image | mask | overlay. Mask colours: magenta 255,0,124 'oil'; "
                 "yellow 255,204,51 'others'; cyan 51,221,255 'water'; black 'background'", fontsize=9)
    fig.tight_layout()
    fig.savefig(AUD / "m4d_visual_audit.png", dpi=80)

    # manifest: one row per image-mask pair, both subsets, NOT mixed for training
    sg = pd.read_parquet(ROOT / "results/sar_audit/splits_v1_acq.parquet")
    lh = pd.read_csv(SC / "local_trujillo_sha256.csv").set_index("rel_path").sha256
    rel = lambda p: p.removeprefix("data/sar/extracted/")
    tru = pd.DataFrame(dict(
        path=sg.img, mask_path=sg["mask"], split_original=sg.split, **{"class": sg.cls},
        source_dataset="trujillo_s1", source_scene="tru_acq" + sg.scene_group.astype(str),
        width=sg.w, height=sg.h, channels=2, dtype="float32", min=np.nan, max=np.nan, crs="EPSG:4326",
        resolution="8.93e-5 deg (~9.9 m N-S, 9.9*cos(lat) m E-W)", has_mask=True,
        mask_positive_fraction=sg.mask_pos_frac, hash=sg.img.map(rel).map(lh)))
    tru["duplicate_group"] = "tru_px" + tru.groupby("hash").ngroup().astype(str)
    m = pd.DataFrame(dict(
        path="data/segformer_combined/" + ti.path, mask_path="data/segformer_combined/" + tm.path,
        split_original=ti.split_original, **{"class": ti.cls}, source_dataset="m4d_uploader_drone_rgb",
        source_scene="m4d_nd" + ti.nd_group.astype(str), width=ti.width, height=ti.height, channels=ti.channels,
        dtype=ti.dtype, min=ti["min"].astype(float), max=ti["max"].astype(float), crs=None, resolution=None,
        has_mask=True, mask_positive_fraction=ti.oil_frac, hash=ti.sha256,
        duplicate_group="m4d_px" + ti.pixel_dup_group.astype(str)))
    man = pd.concat([tru, m], ignore_index=True)
    man.to_parquet(SC / "manifest.parquet")
    print("manifest", man.shape, man.groupby("source_dataset").size().to_dict())


if __name__ == "__main__":
    {"trujillo": trujillo, "m4d": m4d, "report": report}[sys.argv[1]]()
