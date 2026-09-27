"""E004: ResNet-18 3-class tile classifier (no_oil / lookalike / oil) on the 40 m VV cache.

Input: 2 channels = VV (cache vv40, standardised with train stats, invalid -> 0) + validity mask (lab40 != 255).
Split: acq (results/sar_audit/splits_v1_acq.parquet via cache meta), train = role train, val = role val.
ImageNet init (timm resnet18.tv_in1k; timm adapts conv1 to 2 channels), class-weighted CE, AdamW, bf16 autocast,
cosine LR with 1-epoch warm-up, max 30 epochs, early stopping + best checkpoint on val macro-F1 only.
Augmentation (GPU): optional east-west scale (--xscale on|off, applied first, in the original orientation),
horizontal/vertical flips, 90 deg rotations, +/-1.5 dB global gain (re-clipped to the cache range).

Two phases, test is structurally separated from every training decision:
  python scripts/e004_resnet18_cls.py [--xscale on|off] [--out DIR]     train; loads only train/val rows
  python scripts/e004_resnet18_cls.py --eval-test [--out DIR]           once, after training_complete.json exists;
                                                                       refuses to overwrite test_metrics.json
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from e003_feature_baseline import CLASSES, score

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "data/sar/cache/v1"
DB_RANGE = 45.0  # cache scale: (dB + 40) / 45

CFG = dict(model="resnet18.tv_in1k", in_chans=2, seed=26143, epochs=30, patience=8, batch=32, eval_batch=64,
           lr=3e-4, weight_decay=1e-2, warmup_epochs=1, amp="bf16", gain_db=1.5, p_flip=0.5,
           xscale=True, xscale_p=0.5, xscale_min=0.5,  # x spacing ratio across the dataset: 4.8/9.9 .. 1 (SAR_DATA_AUDIT 10.1)
           test_min_valid_frac=0.05)


def seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.use_deterministic_algorithms(True, warn_only=True)  # adaptive-avg-pool backward has no deterministic CUDA kernel


def load(rows):
    rows = np.sort(np.asarray(rows))
    vv = torch.from_numpy(np.load(C / "vv40.npy", mmap_mode="r")[rows]).cuda()
    valid = torch.from_numpy(np.load(C / "lab40.npy", mmap_mode="r")[rows] != 255).cuda()
    return rows, vv, valid


def make_input(vv, valid, mu, sd):
    x = torch.where(valid, (vv.float() - mu) / sd, 0.0)
    return torch.stack([x, valid.float()], 1)


def augment(vv, valid, cfg, g):
    vv, B = vv.float(), vv.shape[0]
    u = lambda: torch.rand(B, device="cuda", generator=g)
    if cfg["xscale"]:
        s = torch.where(u() < cfg["xscale_p"], cfg["xscale_min"] + (1 - cfg["xscale_min"]) * u(), torch.ones(B, device="cuda"))
        tx = (2 * u() - 1) * (1 - s)
        theta = torch.zeros(B, 2, 3, device="cuda")
        theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1] = s, tx, 1.0
        grid = F.affine_grid(theta, (B, 1, *vv.shape[1:]), align_corners=False)
        # ponytail: bilinear in scaled dB (not linear power); only upsampling here, error small vs speckle
        vv = F.grid_sample(vv[:, None], grid, mode="bilinear", padding_mode="border", align_corners=False)[:, 0]
        valid = F.grid_sample(valid[:, None].float(), grid, mode="nearest", align_corners=False)[:, 0] > 0.5
    for dim in (-1, -2):
        f = u() < cfg["p_flip"]
        vv[f], valid[f] = vv[f].flip(dim), valid[f].flip(dim)
    k = torch.randint(0, 4, (B,), device="cuda", generator=g)
    for r in (1, 2, 3):
        m = k == r
        vv[m], valid[m] = torch.rot90(vv[m], r, (1, 2)), torch.rot90(valid[m], r, (1, 2))
    gain = (2 * u() - 1) * cfg["gain_db"] / DB_RANGE
    vv = torch.where(valid, (vv + gain[:, None, None]).clamp(0, 1), 0.0)
    return vv, valid


@torch.no_grad()
def predict(model, vv, valid, mu, sd, bs):
    model.eval()
    out = []
    for i in range(0, len(vv), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(make_input(vv[i:i + bs], valid[i:i + bs], mu, sd)).float())
    return torch.cat(out)


def build_model(cfg, pretrained):
    return timm.create_model(cfg["model"], pretrained=pretrained, in_chans=cfg["in_chans"], num_classes=len(CLASSES)).cuda()


def train(cfg, out):
    assert not (out / "best.pt").exists(), f"{out} already has a run; pass a new --out"
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    meta = pd.read_parquet(C / "meta.parquet")
    tr_rows = meta.index[(meta.role == "train") & (meta.valid_frac_40 > 0)].to_numpy()
    va_rows = meta.index[(meta.role == "val") & (meta.valid_frac_40 > 0)].to_numpy()
    assert not set(meta.role[np.r_[tr_rows, va_rows]]) & {"test", "excluded_test_overlap"}
    tr_rows, tr_vv, tr_valid = load(tr_rows)
    va_rows, va_vv, va_valid = load(va_rows)
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    ytr = torch.tensor(meta.cls[tr_rows].map(cls_idx).to_numpy(), device="cuda")
    yva = torch.tensor(meta.cls[va_rows].map(cls_idx).to_numpy(), device="cuda")
    v = tr_vv[tr_valid].float()
    mu, sd = v.mean().item(), v.std().item()
    del v
    counts = torch.bincount(ytr, minlength=len(CLASSES)).float()
    w = counts.sum() / (len(CLASSES) * counts)
    cfg.update(norm_mu=mu, norm_sd=sd, class_weights=w.tolist(), n_train=len(tr_rows), n_val=len(va_rows),
               train_counts=dict(zip(CLASSES, counts.int().tolist())))
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    print(json.dumps(cfg))

    model = build_model(cfg, pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_ep = (len(tr_rows) + cfg["batch"] - 1) // cfg["batch"]
    total, warm = cfg["epochs"] * steps_ep, cfg["warmup_epochs"] * steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1))))
    g = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    gcpu = torch.Generator().manual_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    hist, best, since, t0 = [], -1.0, 0, time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(tr_rows), generator=gcpu).cuda()
        tl, te = 0.0, time.time()
        for i in range(0, len(perm), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            vv, valid = augment(tr_vv[b], tr_valid[b], cfg, g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(make_input(vv, valid, mu, sd)).float(), ytr[b], weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(b)
        logits = predict(model, va_vv, va_valid, mu, sd, cfg["eval_batch"])
        vl = F.cross_entropy(logits, yva, weight=w).item()
        vl_unw = F.cross_entropy(logits, yva).item()
        pred = logits.argmax(1).cpu().numpy()
        mf1 = f1_score(yva.cpu().numpy(), pred, average="macro")
        hist.append(dict(epoch=ep, train_loss=tl / len(tr_rows), val_loss=vl, val_loss_unweighted=vl_unw, val_macro_f1=mf1,
                         val_acc=float((pred == yva.cpu().numpy()).mean()), lr=sched.get_last_lr()[0], epoch_s=time.time() - te))
        if mf1 > best:
            best, since = mf1, 0
            torch.save(dict(state_dict=model.state_dict(), epoch=ep, val_macro_f1=mf1, cfg=cfg), out / "best.pt")
        else:
            since += 1
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in hist[-1].items()), "*" if since == 0 else "", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if since >= cfg["patience"]:
            print(f"early stop: no val macro-F1 gain for {cfg['patience']} epochs")
            break
    runtime = time.time() - t0
    peak_alloc, peak_res = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()

    ck = torch.load(out / "best.pt", map_location="cuda")
    model.load_state_dict(ck["state_dict"])
    logits = predict(model, va_vv, va_valid, mu, sd, cfg["eval_batch"])
    p = logits.softmax(1).cpu().numpy()
    yv = np.array(CLASSES)[yva.cpu().numpy()]
    pv = np.array(CLASSES)[p.argmax(1)]
    val = score(yv, pv)
    assert abs(val["macro_f1"] - round(ck["val_macro_f1"], 4)) < 1e-3, "reloaded checkpoint does not reproduce val macro-F1"
    pd.DataFrame(dict(row=va_rows, id=meta.id[va_rows].values, true=yv, pred=pv, **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)})
                 ).to_parquet(out / "val_predictions.parquet", index=False)
    done = dict(best_epoch=ck["epoch"], best_val_macro_f1=ck["val_macro_f1"], epochs_run=len(hist), runtime_s=runtime,
                peak_vram_allocated_gb=peak_alloc / 2**30, peak_vram_reserved_gb=peak_res / 2**30,
                checkpoint_mb=(out / "best.pt").stat().st_size / 2**20, val=val)
    json.dump(done, open(out / "training_complete.json", "w"), indent=1)
    plot(pd.DataFrame(hist), ck["epoch"], out)
    print(json.dumps({k: v for k, v in done.items() if k != "val"}, indent=1))
    print("val @best:", json.dumps(val))


def plot(h, best_ep, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="train (weighted CE, augmented)")
    ax[0].plot(h.epoch, h.val_loss, "o-", label="val (weighted CE)")
    ax[1].plot(h.epoch, h.val_macro_f1, "o-", c="C2", label="val macro-F1")
    ax[1].axhline(0.833, ls="--", c="grey", label="E003 val macro-F1 0.833")
    for a in ax:
        a.axvline(best_ep, ls=":", c="r", label=f"best epoch {best_ep}")
        a.set_xlabel("epoch")
        a.legend()
    ax[0].set_ylabel("loss")
    ax[1].set_ylabel("macro-F1")
    plt.tight_layout()
    plt.savefig(out / "curves.png", dpi=90)


def eval_test(out):
    assert (out / "training_complete.json").exists(), "training not complete; test evaluation refused"
    assert not (out / "test_metrics.json").exists(), "test already evaluated once for this run; refusing to re-run"
    ck = torch.load(out / "best.pt", map_location="cuda")
    cfg = ck["cfg"]
    seed_all(cfg["seed"])
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    meta = pd.read_parquet(C / "meta.parquet")
    sp = pd.read_parquet(ROOT / "results/sar_audit/splits_v1_acq.parquet")
    assert (meta[["split", "cls", "id"]].values == sp[["split", "cls", "id"]].values).all()
    te = meta.index[(meta.role == "test") & (meta.valid_frac_40 > cfg["test_min_valid_frac"])].to_numpy()
    rows, vv, valid = load(te)
    p = predict(model, vv, valid, cfg["norm_mu"], cfg["norm_sd"], cfg["eval_batch"]).softmax(1).cpu().numpy()
    pred = np.array(CLASSES)[p.argmax(1)]
    df = pd.DataFrame(dict(row=rows, id=meta.id[rows].values, true=meta.cls[rows].values, pred=pred,
                           acq_clean=~sp.test_shares_acq_with_trainval[rows].values,
                           place_clean=~sp.test_shares_place_with_trainval[rows].values,
                           **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}))
    df.to_parquet(out / "test_predictions.parquet", index=False)
    res = {"checkpoint_epoch": ck["epoch"], "excluded_nodata": meta.id[(meta.role == "test") & (meta.valid_frac_40 <= cfg["test_min_valid_frac"])].tolist(),
           "test_full_minus_nodata": score(df.true, df.pred),
           "test_acq_clean": score(df.true[df.acq_clean], df.pred[df.acq_clean]),
           "test_place_clean": score(df.true[df.place_clean], df.pred[df.place_clean])}
    json.dump(res, open(out / "test_metrics.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/E004_resnet18_cls"))
    ap.add_argument("--xscale", choices=["on", "off"], default="on")
    ap.add_argument("--eval-test", action="store_true")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if a.eval_test:
        eval_test(out)
    else:
        train(dict(CFG, xscale=a.xscale == "on"), out)


if __name__ == "__main__":
    main()
