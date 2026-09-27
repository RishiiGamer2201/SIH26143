"""E006: one-shot official-test evaluation of the FROZEN E005 SegFormer MiT-B2. EVALUATION ONLY.

No training, no threshold sweep, no checkpoint selection, no post-processing: raw mask = sigmoid(logit) >= 0.60
(frozen on val in E005); 0.5 is reported only as a secondary reference (E005 checkpoint selection used 0.5).

  python scripts/e006_segformer_test.py freeze   # hash E005 artefacts -> results/E005_segformer_b2/FROZEN.json (once)
  python scripts/e006_segformer_test.py test     # verify hashes, re-check val, then evaluate test -> results/E006_segformer_test

Test views (SAR audit definitions, as in E004):
  full        official test minus nodata tiles (valid_frac_40 <= 0.05: no_oil 00005, 00087, 00099)
  acq_clean   full & test_shares_acq_with_trainval == False
  place_clean full & test_shares_place_with_trainval == False
Connected components: scipy.ndimage.label default (4-connectivity), exactly as E005 validation. A GT component is missed
when no predicted valid pixel falls inside it. Pixel counts are image-space diagnostics only (see PIXEL_AREA_NOTE).
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e005_segformer import C, PIXEL_AREA_NOTE, ROOT, build_model, counts, infer, prf, seed_all  # noqa: E402

E005 = ROOT / "results/E005_segformer_b2"
OUT = ROOT / "results/E006_segformer_test"
FROZEN_FILES = ["best.pt", "config.json", "frozen_threshold.json", "training_complete.json", "val_metrics.json",
                "pretraining_tests.json", "history.csv", "val_threshold_sweep.csv", "val_tiles.parquet",
                "val_gt_component_recall_by_size.csv"]
DATA_FILES = ["results/sar_audit/splits_v1_acq.parquet", "data/sar/cache/v1/meta.parquet", "data/sar/cache/v1/cache_config.json",
              "data/sar/cache/v1/vv20.npy", "data/sar/cache/v1/lab20.npy", "scripts/e005_segformer.py"]
THR, THR_REF = 0.60, 0.5
MIN_VALID = 0.05  # E004 test_min_valid_frac on valid_frac_40
BINS = [(1, 1, "1"), (2, 4, "2-4"), (5, 16, "5-16"), (17, 64, "17-64"), (65, 256, "65-256"), (257, 1024, "257-1024"),
        (1025, 10**12, ">1024")]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    r = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def git_state(paths):
    st = dict(head=git("rev-parse", "HEAD"), branch=git("rev-parse", "--abbrev-ref", "HEAD"), remote=git("remote", "get-url", "origin"),
              files={})
    for p in paths:
        rel = str(p.relative_to(ROOT))
        head_blob = git("rev-parse", f"HEAD:{rel}")
        st["files"][rel] = dict(tracked=head_blob is not None, matches_head=head_blob is not None and head_blob == git("hash-object", rel))
    return st


# ---------------------------------------------------------------------------------------------------- freeze
def freeze():
    dst = E005 / "FROZEN.json"
    assert not dst.exists(), f"{dst} exists; E005 is already frozen"
    ck = torch.load(E005 / "best.pt", map_location="cpu", weights_only=False)
    cfg, done = ck["cfg"], json.load(open(E005 / "training_complete.json"))
    thr, vm = json.load(open(E005 / "frozen_threshold.json")), json.load(open(E005 / "val_metrics.json"))
    assert ck["epoch"] == done["best_epoch"] == thr["checkpoint_epoch"] == 9 and thr["threshold"] == THR
    files = [E005 / f for f in FROZEN_FILES]
    data = [ROOT / f for f in DATA_FILES]
    fz = dict(
        frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        status="E005 approved and FROZEN. Never retrain, never change the threshold, never overwrite these files (E006+ write elsewhere).",
        best_epoch=ck["epoch"], threshold=THR, threshold_rule=thr["rule"], secondary_reference_threshold=THR_REF,
        model=dict(library="segmentation_models_pytorch 0.5.0", arch="Segformer", encoder=cfg["encoder"], encoder_init=cfg["encoder_weights"],
                   in_channels=cfg["in_channels"], classes=1, params_m=sum(v.numel() for v in ck["state_dict"].values()) / 1e6,
                   input="ch0 = (vv20 - norm_mu) / norm_sd on valid px, 0 on invalid; ch1 = validity (lab20 != 255); full 1024x1024 tiles",
                   norm_mu=cfg["norm_mu"], norm_sd=cfg["norm_sd"], output="1 logit/px, sigmoid, oil if prob >= threshold; no post-processing",
                   inference_dtype="bf16 autocast", checkpoint_val_iou_at_0_5=ck["val_iou"]),
        dataset=dict(source="Trujillo Sentinel-1 (data/sar/extracted), E002 cache v1 20 m (EPSG:4326 degree grid)",
                     split="results/sar_audit/splits_v1_acq.parquet", n_train=cfg["n_train"], n_val=cfg["n_val"],
                     train_tiles=cfg["train_tiles"], val_tiles=cfg["val_tiles"], test_used=False),
        validation=dict(at_frozen_0_60=vm["at_selected"]["global_pixel"], at_0_5=vm["at_0_5"]["global_pixel"],
                        oil_tiles_at_0_60=vm["at_selected"]["oil_tiles"]["pixel"]),
        pixel_area_note=PIXEL_AREA_NOTE,
        sha256={str(p.relative_to(ROOT)): dict(sha256=sha256(p), bytes=p.stat().st_size) for p in files + data},
        git=git_state(files + data),
        notes=["val_metrics.json pixel_area_note corrected on 2026-09-28 before freezing (metrics unchanged); "
               "original kept as superseded/val_metrics.2026-09-27_pixel_area_note.json.",
               "scripts/e005_segformer.py hash is after the same pixel-area-note correction (PIXEL_AREA_NOTE constant); logic unchanged."])
    json.dump(fz, open(dst, "w"), indent=1)
    print(json.dumps({k: v for k, v in fz.items() if k not in ("sha256", "git")}, indent=1))
    print(json.dumps(fz["git"], indent=1))


def verify_frozen():
    fz = json.load(open(E005 / "FROZEN.json"))
    bad = {k: v["sha256"] for k, v in fz["sha256"].items() if sha256(ROOT / k) != v["sha256"]}
    assert not bad, f"frozen artefacts changed: {list(bad)}"
    return fz


# ---------------------------------------------------------------------------------------------------- per-tile stats
def tile_stats(pj, vj, yj, t):
    """Same per-tile quantities as E005 validate(); also returns GT component (size, hit) pairs."""
    pr = (pj >= t) & vj
    tp, fp, fn = counts(pr, yj, vj).tolist()
    prn, gt = pr.cpu().numpy(), yj.cpu().numpy()
    lbl, nc = ndimage.label(prn)
    sizes = np.bincount(lbl.ravel())[1:]
    hit = np.zeros(nc + 1, bool)
    hit[np.unique(lbl[gt & (lbl > 0)])] = True
    false_sizes = sizes[~hit[1:]]
    glbl, ng = ndimage.label(gt)
    gsz = np.bincount(glbl.ravel(), minlength=ng + 1)[1:]
    ghit = np.zeros(ng + 1, bool)
    ghit[np.unique(glbl[prn & (glbl > 0)])] = True
    valid_px, gt_px = int(vj.sum()), int((yj & vj).sum())
    r = dict(pred_px=int(pr.sum()), tp=tp, fp=fp, fn=fn, fp_frac=fp / max(valid_px, 1),
             dice=2 * tp / max(2 * tp + fp + fn, 1) if gt_px else np.nan, recall=tp / max(tp + fn, 1) if gt_px else np.nan,
             n_comp=nc, n_false_comp=int(len(false_sizes)), largest_false_comp_px=int(false_sizes.max()) if len(false_sizes) else 0,
             gt_comp=ng, missed_gt_comp=int(ng - ghit[1:].sum()))
    return r, np.stack([gsz, ghit[1:]], 1)


def run(model, cfg, rows, meta, vv, lab, keep_prob):
    tiles, comps, probs = [], [], []
    for i, p, valid, y in infer(model, vv, lab, cfg["norm_mu"], cfg["norm_sd"], cfg["eval_batch"]):
        for j in range(len(p)):
            n = i + j
            r = dict(row=int(rows[n]), id=meta.id[rows[n]], cls=meta.cls[rows[n]], valid_px=int(valid[j].sum()), gt_px=int((y[j] & valid[j]).sum()))
            for tag, t in (("t05", THR_REF), ("tsel", THR)):
                s, gc = tile_stats(p[j], valid[j], y[j], t)
                r.update({f"{tag}_{k}": v for k, v in s.items()})
                if tag == "tsel":
                    comps += [(int(rows[n]), int(a), bool(b)) for a, b in gc]
            tiles.append(r)
            if keep_prob:
                probs.append((p[j] * 255).round().byte().cpu().numpy())
    return pd.DataFrame(tiles), pd.DataFrame(comps, columns=["row", "size_px", "hit"]), probs


def group(T, tag):
    """Same fields as E005 val_metrics.json."""
    k = [f"{tag}_tp", f"{tag}_fp", f"{tag}_fn"]
    res = dict(n_tiles=len(T), tiles_by_class=T.cls.value_counts().to_dict(), global_pixel=prf(*T[k].sum().tolist()))
    o = T[T.cls == "oil"]
    res["oil_tiles"] = dict(n=len(o), pixel=prf(*o[k].sum().tolist()), tile_dice_mean=float(o[f"{tag}_dice"].mean()),
                            tile_dice_median=float(o[f"{tag}_dice"].median()), tile_recall_mean=float(o[f"{tag}_recall"].mean()),
                            tiles_detected_any_tp=int((o[f"{tag}_tp"] > 0).sum()), tiles_missed_entirely=int((o[f"{tag}_tp"] == 0).sum()),
                            false_components_total=int(o[f"{tag}_n_false_comp"].sum()),
                            largest_false_component_px=int(o[f"{tag}_largest_false_comp_px"].max()),
                            gt_components=int(o[f"{tag}_gt_comp"].sum()), missed_gt_components=int(o[f"{tag}_missed_gt_comp"].sum()))
    for c_ in ("lookalike", "no_oil"):
        n = T[T.cls == c_]
        res[f"{c_}_tiles"] = dict(n=len(n), frac_tiles_any_pred=float((n[f"{tag}_pred_px"] > 0).mean()),
                                  mean_fp_frac=float(n[f"{tag}_fp_frac"].mean()), p95_fp_frac=float(n[f"{tag}_fp_frac"].quantile(0.95)),
                                  max_fp_frac=float(n[f"{tag}_fp_frac"].max()), false_components_total=int(n[f"{tag}_n_false_comp"].sum()),
                                  false_components_per_tile=float(n[f"{tag}_n_false_comp"].mean()),
                                  largest_false_component_px=int(n[f"{tag}_largest_false_comp_px"].max()))
    return res


def by_size(G):
    tot = G.size_px.sum()
    out = []
    for lo, hi, name in BINS:
        g = G[(G.size_px >= lo) & (G.size_px <= hi)]
        out.append(dict(size=name, n=len(g), missed=int((~g.hit).sum()), gt_px=int(g.size_px.sum()),
                        miss_rate=float((~g.hit).mean()) if len(g) else np.nan, share_of_gt_px=float(g.size_px.sum() / tot)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------------- test
def test():
    t_all = time.time()
    assert not (OUT / "test_metrics.json").exists(), f"{OUT} already holds the one-shot test result"
    OUT.mkdir(parents=True, exist_ok=True)
    fz = verify_frozen()
    ck = torch.load(E005 / "best.pt", map_location="cuda", weights_only=False)
    cfg = ck["cfg"]
    assert ck["epoch"] == fz["best_epoch"] == 9 and json.load(open(E005 / "frozen_threshold.json"))["threshold"] == THR
    seed_all(cfg["seed"])
    model = build_model(cfg, None)
    model.load_state_dict(ck["state_dict"])
    torch.cuda.reset_peak_memory_stats()
    meta = pd.read_parquet(C / "meta.parquet")
    sp = pd.read_parquet(ROOT / "results/sar_audit/splits_v1_acq.parquet")
    assert (meta[["split", "cls", "id", "role"]].values == sp[["split", "cls", "id", "role"]].values).all()

    # 1. val re-check (no test data yet): frozen model + component definition must reproduce E005 exactly
    t0 = time.time()
    vrows = meta.index[meta.role == "val"].to_numpy()
    vv_ = np.ascontiguousarray(np.load(C / "vv20.npy", mmap_mode="r")[vrows])
    lab_ = np.ascontiguousarray(np.load(C / "lab20.npy", mmap_mode="r")[vrows])
    TV, GV, _ = run(model, cfg, vrows, meta, vv_, lab_, False)
    del vv_, lab_
    ref = json.load(open(E005 / "val_metrics.json"))
    gv = group(TV, "tsel")
    val_diff = {k: abs(gv["global_pixel"][k] - ref["at_selected"]["global_pixel"][k]) for k in gv["global_pixel"]}
    assert max(val_diff.values()) < 1e-6, f"frozen model does not reproduce E005 val metrics: {val_diff}"
    vs = by_size(GV)
    vref = pd.read_csv(E005 / "val_gt_component_recall_by_size.csv")
    assert (vs[["n", "missed", "gt_px"]].values == vref[["n", "missed", "gt_px"]].values).all(), "component definition differs from E005"
    val_check_s = time.time() - t0

    # 2. official test, loaded only here
    t0 = time.time()
    tmask = (meta.role == "test").to_numpy()
    excluded = meta[tmask & (meta.valid_frac_40 <= MIN_VALID).to_numpy()]
    assert sorted(excluded.id.tolist()) == ["00005", "00087", "00099"] and set(excluded.cls) == {"no_oil"}
    rows = meta.index[tmask & (meta.valid_frac_40 > MIN_VALID).to_numpy()].to_numpy()
    vv = np.ascontiguousarray(np.load(C / "vv20.npy", mmap_mode="r")[rows])
    lab = np.ascontiguousarray(np.load(C / "lab20.npy", mmap_mode="r")[rows])
    T, G, probs = run(model, cfg, rows, meta, vv, lab, True)
    infer_s = time.time() - t0
    T["acq_clean"] = ~sp.test_shares_acq_with_trainval[T.row].to_numpy()
    T["place_clean"] = ~sp.test_shares_place_with_trainval[T.row].to_numpy()
    G = G.merge(T[["row", "acq_clean", "place_clean"]], on="row")
    views = {"full": T.index == T.index, "acq_clean": T.acq_clean.to_numpy(), "place_clean": T.place_clean.to_numpy()}

    met = dict(checkpoint="results/E005_segformer_b2/best.pt", checkpoint_sha256=fz["sha256"]["results/E005_segformer_b2/best.pt"]["sha256"],
               checkpoint_epoch=ck["epoch"], threshold=THR, reference_threshold=THR_REF, post_processing="none (raw mask)",
               excluded_nodata=[f"{c}/{i}" for c, i in zip(excluded.cls, excluded.id)], pixel_area_note=PIXEL_AREA_NOTE,
               val_recheck=dict(max_abs_diff_vs_E005=max(val_diff.values()), component_table_identical=True), views={})
    comp = []
    for v, m in views.items():
        met["views"][v] = dict(at_frozen_0_60=group(T[m], "tsel"), at_0_5_reference=group(T[m], "t05"))
        comp.append(by_size(G[G[v]] if v != "full" else G).assign(view=v))
    pix = lambda v, sec: met["views"][v]["at_frozen_0_60"]["global_pixel" if sec == "global" else "oil_tiles"]
    pix2 = lambda v, sec: pix(v, sec) if sec == "global" else pix(v, sec)["pixel"]
    met["contamination_effect"] = {"note": "view minus full, at the frozen 0.60; views also differ in class mix",
                                   **{v: {f"{sec}_{k}_delta": pix2(v, sec)[k] - pix2("full", sec)[k]
                                          for sec in ("global", "oil_tiles") for k in ("iou", "dice", "precision", "recall")}
                                      for v in ("acq_clean", "place_clean")}}
    met["runtime_s"] = dict(val_recheck=val_check_s, test_inference_and_tile_stats=infer_s, total=time.time() - t_all)
    met["peak_vram_gb"] = dict(allocated=torch.cuda.max_memory_allocated() / 2**30, reserved=torch.cuda.max_memory_reserved() / 2**30)

    T.to_parquet(OUT / "test_tiles.parquet", index=False)
    comp = pd.concat(comp)[["view", "size", "n", "missed", "miss_rate", "gt_px", "share_of_gt_px"]]
    comp.to_csv(OUT / "test_component_recall_by_size.csv", index=False)
    np.save(OUT / "test_prob_u8.npy", np.stack(probs))  # sigmoid * 255, uint8, row order = test_tiles.parquet
    json.dump(met, open(OUT / "test_metrics.json", "w"), indent=1)
    json.dump(dict(script="scripts/e006_segformer_test.py", frozen_manifest="results/E005_segformer_b2/FROZEN.json",
                   checkpoint=met["checkpoint"], checkpoint_sha256=met["checkpoint_sha256"], threshold=THR, reference_threshold=THR_REF,
                   eval_batch=cfg["eval_batch"], amp="bf16 autocast", tiles="full 1024x1024 vv20/lab20, no cropping, no TTA",
                   norm_mu=cfg["norm_mu"], norm_sd=cfg["norm_sd"], post_processing="none",
                   exclusion=f"role == test and valid_frac_40 <= {MIN_VALID} (E004 rule)",
                   views={"full": "official test minus nodata", "acq_clean": "full & ~test_shares_acq_with_trainval",
                          "place_clean": "full & ~test_shares_place_with_trainval"},
                   components="scipy.ndimage.label default 4-connectivity; GT component missed if no predicted valid pixel inside",
                   size_bins=[b[2] for b in BINS], prob_file="test_prob_u8.npy (uint8 = round(prob*255))",
                   figure_rules="seeded 26143: good = Dice>=0.75 random 3; small = GT px <= q25 random 3; large = GT px >= q75 random 3; "
                                "complete misses = TP==0, largest GT first, 3; LookAlike / NoOil = top-3 FP fraction"),
              open(OUT / "evaluation_config.json", "w"), indent=1)
    figures(T, probs, vv, lab)
    verify_frozen()  # E005 untouched
    print(json.dumps({k: v for k, v in met.items() if k != "views"}, indent=1))
    for v in views:
        print(v, json.dumps(met["views"][v]["at_frozen_0_60"], indent=1))
        print(v, "@0.5", json.dumps(met["views"][v]["at_0_5_reference"]["global_pixel"]))
    print(comp.to_string(index=False))


def figures(T, probs, vv, lab):
    rng = np.random.default_rng(26143)
    o = T[T.cls == "oil"]
    q = o.gt_px.quantile([0.25, 0.75])
    pick = lambda df, k: list(rng.choice(df.index, min(k, len(df)), replace=False)) if len(df) else []
    miss = o[o.tsel_tp == 0]
    sel = [("successful detection (tile Dice >= 0.75, random)", pick(o[o.tsel_dice >= 0.75], 3)),
           ("small slick (GT px <= q25, random)", pick(o[o.gt_px <= q[0.25]], 3)),
           ("large slick (GT px >= q75, random)", pick(o[o.gt_px >= q[0.75]], 3)),
           ("complete miss (TP = 0, largest GT)", list(miss.nlargest(3, "gt_px").index)) if len(miss) else
           ("no complete miss: lowest oil recall", list(o.nsmallest(3, "tsel_recall").index)),
           ("strongest LookAlike FP (max FP frac)", list(T[T.cls == "lookalike"].nlargest(3, "tsel_fp_frac").index)),
           ("strongest NoOil FP (max FP frac)", list(T[T.cls == "no_oil"].nlargest(3, "tsel_fp_frac").index))]
    rows = [(tag, i) for tag, ix in sel for i in ix]
    fig, ax = plt.subplots(len(rows), 5, figsize=(15, 3.1 * len(rows)))
    for r, (tag, i) in enumerate(rows):
        valid, y = lab[i] != 255, lab[i] == 1
        prob = probs[i] / 255.0
        pred = (prob >= THR) & valid
        ov = np.stack([vv[i].astype(np.float32)] * 3, -1)
        ov[pred & y], ov[pred & ~y], ov[~pred & y], ov[~valid] = (0, 1, 0), (1, 0, 0), (0, 0.4, 1), (1, 1, 1)
        tr = T.loc[i]
        clean = f"acq-clean={tr.acq_clean} place-clean={tr.place_clean}"
        panels = [(np.where(valid, vv[i], np.nan), "gray", f"{tag}\n{tr.cls} {tr.id}  VV  ({clean})"),
                  (np.where(valid, y, np.nan), "gray", f"GT oil px={tr.gt_px}"),
                  (np.where(valid, prob, np.nan), "magma", "probability"),
                  (np.where(valid, pred, np.nan), "gray", f"pred @ {THR:.2f}  Dice={tr.tsel_dice:.2f}" if tr.gt_px
                   else f"pred @ {THR:.2f}  FP frac={tr.tsel_fp_frac:.4f}"),
                  (ov, None, "overlay: TP green, FP red, FN blue, invalid white")]
        for c, (a, cm, title) in enumerate(panels):
            ax[r, c].imshow(a, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
            ax[r, c].set_title(title, fontsize=7)
            ax[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "test_examples.png", dpi=90)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["freeze", "test"])
    {"freeze": freeze, "test": test}[ap.parse_args().stage]()
