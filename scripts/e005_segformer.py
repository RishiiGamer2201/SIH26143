"""E005: SegFormer MiT-B2 binary oil segmentation, Trujillo Sentinel-1 only (20 m VV cache). TRAIN / VAL ONLY.

Input 2 ch: VV (vv20, standardised with train valid-pixel stats, invalid -> 0) + validity (lab20 != 255).
Target: oil = (lab20 == 1); invalid pixels (lab20 == 255) are excluded from BCE and Dice.
Split: results/sar_audit/splits_v1_acq.parquet via cache meta (asserted identical); only role train / val rows are
ever loaded. Official test is not touched by this script.
Model: smp.Segformer("mit_b2", encoder_weights="imagenet", in_channels=2, classes=1): the moderate-capacity baseline
for the RTX 5070 Ti 16 GB. smp adapts the RGB patch-embed conv to 2 inputs (see tests: weight_adaptation).
Crops: 512 px from 1024 px tiles by an explicit sampler (oil-positive / LookAlike / NoOil / oil-scene background).
Loss: 0.5 * BCE(pos_weight = min(raw train neg/pos, 5)) + 0.5 * soft Dice (batch-global), both over valid pixels only.
Aug (GPU): h/v flip, rot90, +/-1.5 dB VV gain. No x-scale. AdamW 6e-5, 1-epoch linear warm-up then poly(1.0), bf16.
Checkpoint: val oil IoU at a FIXED 0.5 threshold, full 1024 px tiles. Threshold is swept on val only after training.

  python scripts/e005_segformer.py tests       # A-E pre-training tests -> OUT/pretraining_tests.json (sets batch)
  python scripts/e005_segformer.py train       # needs passing tests; best.pt on val IoU@0.5
  python scripts/e005_segformer.py validate    # val-only: threshold sweep, frozen threshold, metrics, figures
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
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "data/sar/cache/v1"
DB_RANGE = 45.0  # cache scale: (dB + 40) / 45
KINDS = ("oil_pos", "lookalike", "no_oil", "oil_bg")
PIXEL_AREA_NOTE = ("pixel counts are image-space diagnostics only, not areas. The cache grid is EPSG:4326 (degrees): "
                   "N-S and E-W ground sizes differ and E-W size varies with cos(latitude). Real slick area: georeference "
                   "the prediction, vectorize, then compute geodesically or in a suitable projected CRS; never pixel_count * 400 m2.")

CFG = dict(encoder="mit_b2", encoder_weights="imagenet", in_channels=2, seed=26143, epochs=40, patience=10,
           crop=512, samples_per_epoch=2000, eval_batch=4, lr=6e-5, weight_decay=1e-2, warmup_epochs=1, poly_power=1.0,
           amp="bf16", gain_db=1.5, p_flip=0.5, mix=dict(oil_pos=0.50, lookalike=0.25, no_oil=0.15, oil_bg=0.10),
           bg_block=32, pos_weight_cap=5.0, bce_w=0.5, dice_w=0.5, dice_smooth=1.0, train_threshold=0.5,
           sweep=[round(t, 2) for t in np.arange(0.10, 0.901, 0.05)],
           probe_batches=[2, 4, 6, 8, 12, 16], probe_max_reserved_gb=12.0, overfit_n=12, overfit_steps=600,
           overfit_lr=3e-4, overfit_target_dice=0.95)


def seed_all(s):
    torch.manual_seed(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.use_deterministic_algorithms(True, warn_only=True)  # bilinear upsample backward has no deterministic CUDA kernel


def load(role):
    assert role in ("train", "val"), "E005 never loads test rows"
    meta = pd.read_parquet(C / "meta.parquet")
    sp = pd.read_parquet(ROOT / "results/sar_audit/splits_v1_acq.parquet")
    assert (meta[["split", "cls", "id", "role"]].values == sp[["split", "cls", "id", "role"]].values).all()
    rows = meta.index[meta.role == role].to_numpy()
    vv = np.ascontiguousarray(np.load(C / "vv20.npy", mmap_mode="r")[rows])
    lab = np.ascontiguousarray(np.load(C / "lab20.npy", mmap_mode="r")[rows])
    return rows, meta.cls[rows].to_numpy(), meta.id[rows].to_numpy(), vv, lab


def train_stats(vv, lab):
    s = s2 = n = pos = neg = 0.0
    for i in range(0, len(vv), 64):
        v, m = vv[i:i + 64].astype(np.float64), lab[i:i + 64]
        ok = m != 255
        s, s2, n = s + v[ok].sum(), s2 + (v[ok] ** 2).sum(), n + ok.sum()
        pos, neg = pos + (m == 1).sum(), neg + (m == 0).sum()
    mu = s / n
    return dict(norm_mu=float(mu), norm_sd=float(np.sqrt(s2 / n - mu ** 2)), train_valid_px=int(n),
                train_oil_px=int(pos), train_bg_px=int(neg), train_oil_frac=float(pos / n), raw_pos_weight=float(neg / pos))


class CropSampler:
    """Explicit crop sampler over train tiles. Returns (tile, oy, ox, kind) specs."""

    def __init__(self, cls, lab, cfg, seed):
        self.c, self.rng, self.mix = cfg["crop"], np.random.default_rng(seed), cfg["mix"]
        self.idx = {k: np.flatnonzero(cls == c) for k, c in (("oil", "oil"), ("lookalike", "lookalike"), ("no_oil", "no_oil"))}
        self.pos = [np.flatnonzero(lab[t].ravel() == 1).astype(np.int32) for t in self.idx["oil"]]
        assert all(len(p) for p in self.pos), "oil tile with no positive pixel"
        b, g, H = cfg["bg_block"], self.c // cfg["bg_block"], lab.shape[1] // cfg["bg_block"]
        bg = []
        for t in self.idx["oil"]:  # crop origins on a b-px grid whose 512 px window holds no oil pixel
            blk = (lab[t] == 1).reshape(H, b, H, b).any((1, 3)).astype(np.int32)
            cs = np.pad(blk.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
            win = cs[g:, g:] - cs[:-g, g:] - cs[g:, :-g] + cs[:-g, :-g]
            bg += [(t, oy * b, ox * b) for oy, ox in zip(*np.nonzero(win == 0))]
        self.bg = np.array(bg, dtype=np.int64).reshape(-1, 3)
        self.bg_tiles = len(np.unique(self.bg[:, 0])) if len(self.bg) else 0
        self.W = lab.shape[1]

    def _rand(self, t):
        o = self.rng.integers(0, self.W - self.c + 1, 2)
        return int(t), int(o[0]), int(o[1])

    def sample(self, n):
        kinds = self.rng.choice(len(KINDS), n, p=[self.mix[k] for k in KINDS])
        out = []
        for k in kinds:
            kind = KINDS[k]
            if kind == "oil_pos":
                j = self.rng.integers(len(self.idx["oil"]))
                y, x = divmod(int(self.rng.choice(self.pos[j])), self.W)
                lo = lambda p: max(0, p - self.c + 1)
                hi = lambda p: min(p, self.W - self.c)
                out.append((int(self.idx["oil"][j]), int(self.rng.integers(lo(y), hi(y) + 1)),
                            int(self.rng.integers(lo(x), hi(x) + 1)), kind))
            elif kind == "oil_bg" and len(self.bg):
                out.append((*map(int, self.bg[self.rng.integers(len(self.bg))]), kind))
            elif kind == "oil_bg":  # no oil-free window exists: fall back to a random oil-scene crop, logged
                out.append((*self._rand(self.rng.choice(self.idx["oil"])), "oil_bg_fallback"))
            else:
                out.append((*self._rand(self.rng.choice(self.idx[kind])), kind))
        return out


def gather(vv, lab, spec, c):
    v = torch.from_numpy(np.stack([vv[t, y:y + c, x:x + c] for t, y, x, _ in spec])).cuda(non_blocking=True)
    m = torch.from_numpy(np.stack([lab[t, y:y + c, x:x + c] for t, y, x, _ in spec])).cuda(non_blocking=True)
    return v.float(), m != 255, m == 1


def augment(vv, valid, y, cfg, g):
    """Identical geometry for VV / validity / target (stacked); gain only on valid VV pixels."""
    B = vv.shape[0]
    u = lambda: torch.rand(B, device=vv.device, generator=g)
    s = torch.stack([vv, valid.float(), y.float()], 1)
    for dim in (-1, -2):
        f = u() < cfg["p_flip"]
        s[f] = s[f].flip(dim)
    k = torch.randint(0, 4, (B,), device=vv.device, generator=g)
    for r in (1, 2, 3):
        m = k == r
        s[m] = torch.rot90(s[m], r, (2, 3))
    vv, valid, y = s[:, 0], s[:, 1] > 0.5, s[:, 2] > 0.5
    gain = (2 * u() - 1) * cfg["gain_db"] / DB_RANGE
    return torch.where(valid, (vv + gain[:, None, None]).clamp(0, 1), 0.0), valid, y


def make_input(vv, valid, mu, sd):
    return torch.stack([torch.where(valid, (vv.float() - mu) / sd, 0.0), valid.float()], 1)


def seg_loss(logits, y, valid, pos_weight, cfg):
    """0.5 BCE + 0.5 soft Dice over valid pixels only (torch.where: invalid pixels give exactly 0 loss and 0 grad)."""
    l = logits.float()
    pw = torch.tensor(pos_weight, device=l.device)
    bce = torch.where(valid, F.binary_cross_entropy_with_logits(l, y.float(), pos_weight=pw, reduction="none"), 0.0)
    bce = bce.sum() / valid.sum().clamp_min(1)
    p = torch.where(valid, torch.sigmoid(l), 0.0)
    t = (y & valid).float()
    dice = 1 - (2 * (p * t).sum() + cfg["dice_smooth"]) / (p.sum() + t.sum() + cfg["dice_smooth"])
    return cfg["bce_w"] * bce + cfg["dice_w"] * dice, bce, dice


def build_model(cfg, weights):
    return smp.Segformer(cfg["encoder"], encoder_weights=weights, in_channels=cfg["in_channels"], classes=1).cuda()


def counts(pred, y, valid):
    return torch.stack([(pred & y & valid).sum(), (pred & ~y & valid).sum(), (~pred & y & valid).sum()]).double()


def prf(tp, fp, fn):
    return dict(iou=tp / max(tp + fp + fn, 1), dice=2 * tp / max(2 * tp + fp + fn, 1),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))


@torch.no_grad()
def infer(model, vv, lab, mu, sd, bs):
    """Full 1024 px tiles (deployment-like). Yields (start, prob [b,H,W] float32 GPU, valid, y)."""
    model.eval()
    for i in range(0, len(vv), bs):
        v = torch.from_numpy(vv[i:i + bs]).cuda().float()
        m = torch.from_numpy(lab[i:i + bs]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(make_input(v, m != 255, mu, sd))[:, 0]
        assert logit.shape == v.shape
        yield i, torch.sigmoid(logit.float()), m != 255, m == 1


def val_epoch(model, vv, lab, cfg, pw):
    c = torch.zeros(3, dtype=torch.float64, device="cuda")
    bce_s = nv = 0.0
    for _, p, valid, y in infer(model, vv, lab, cfg["norm_mu"], cfg["norm_sd"], cfg["eval_batch"]):
        c += counts(p >= cfg["train_threshold"], y, valid)
        l = torch.logit(p.clamp(1e-6, 1 - 1e-6))
        bce_s += torch.where(valid, F.binary_cross_entropy_with_logits(l, y.float(), pos_weight=torch.tensor(pw, device="cuda"),
                                                                       reduction="none"), 0.0).sum().item()
        nv += valid.sum().item()
    return dict(prf(*c.tolist()), bce=bce_s / nv)


# ---------------------------------------------------------------------------------------------------- tests A-E
def tests(cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    res = {}
    # A. forward smoke test + weight adaptation
    model = build_model(cfg, cfg["encoder_weights"])
    ref = smp.Segformer(cfg["encoder"], encoder_weights=cfg["encoder_weights"], in_channels=3, classes=1)
    name, w2 = next((n, p) for n, p in model.named_parameters() if p.ndim == 4 and p.shape[1] == 2)
    w3 = dict(ref.named_parameters())[name]
    adapt_ok = torch.allclose(w2.detach().cpu(), w3.detach()[:, :2] * 1.5)
    x = torch.randn(2, 2, 512, 512, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):  # no graph kept: the VRAM probe below must start clean
        o = model(x)
        o1024 = model(torch.randn(1, 2, 1024, 1024, device="cuda"))
    res["A_smoke"] = dict(out_512=list(o.shape), out_dtype=str(o.dtype), out_1024=list(o1024.shape), finite=bool(torch.isfinite(o).all()),
                          params_m=sum(p.numel() for p in model.parameters()) / 1e6,
                          encoder_params_m=sum(p.numel() for p in model.encoder.parameters()) / 1e6,
                          first_conv=name, first_conv_shape=list(w2.shape),
                          weight_adaptation="2-ch weight == pretrained RGB weight[:, :2] * 3/2 (smp patch_first_conv: "
                                            "channel i copies RGB channel i % 3, rescaled by 3/2)",
                          weight_adaptation_verified=bool(adapt_ok))
    assert list(o.shape) == [2, 1, 512, 512] and list(o1024.shape) == [1, 1, 1024, 1024] and adapt_ok
    del ref, model, o, o1024, x, w2, w3

    # B. loss mask: invalid pixels contribute exactly zero loss and zero gradient
    g = torch.Generator(device="cuda").manual_seed(1)
    l = torch.randn(4, 64, 64, device="cuda", generator=g)
    y = torch.rand(4, 64, 64, device="cuda", generator=g) < 0.2
    valid = torch.rand(4, 64, 64, device="cuda", generator=g) < 0.7
    base = seg_loss(l, y, valid, 5.0, cfg)[0]
    l2 = torch.where(valid, l, 1e4 * torch.sign(torch.randn_like(l)))  # extreme logits at invalid pixels
    y2 = torch.where(valid, y, ~y)                                        # flipped targets at invalid pixels
    l2.requires_grad_(True)
    tot, bce, dice = seg_loss(l2, y2, valid, 5.0, cfg)
    tot.backward()
    res["B_loss_mask"] = dict(loss_equal=bool(torch.equal(tot.detach(), base)), max_abs_diff=float((tot.detach() - base).abs()),
                              max_abs_grad_invalid=float(l2.grad[~valid].abs().max()), grad_valid_nonzero=bool(l2.grad[valid].abs().sum() > 0))
    assert res["B_loss_mask"]["loss_equal"] and res["B_loss_mask"]["max_abs_grad_invalid"] == 0.0

    # C. augmentation alignment: validity/target are fixed functions of VV before; must still be after (gain 0)
    r = torch.rand(64, 96, 96, device="cuda", generator=g)
    va, ya = r > 0.3, r > 0.6
    v0 = torch.where(va, r, 0.0)
    vA, vaA, yA = augment(v0.clone(), va.clone(), ya.clone(), dict(cfg, gain_db=0.0), g)
    c_geom = bool(torch.equal(vaA, vA > 0.3) and torch.equal(yA, vA > 0.6))
    vG, vaG, _ = augment(v0.clone(), va.clone(), ya.clone(), cfg, torch.Generator(device="cuda").manual_seed(2))
    vN, _, _ = augment(v0.clone(), va.clone(), ya.clone(), dict(cfg, gain_db=0.0), torch.Generator(device="cuda").manual_seed(2))
    d = (vG - vN)[vaG] * DB_RANGE
    res["C_augment"] = dict(identical_geometry=c_geom, n_transformed_not_identity=int((vA != v0).flatten(1).any(1).sum()),
                            invalid_stays_zero=bool((vG[~vaG] == 0).all()), gain_db_range=[float(d.min()), float(d.max())])
    assert c_geom and res["C_augment"]["invalid_stays_zero"] and d.abs().max() <= cfg["gain_db"] + 1e-3

    # E. VRAM / throughput probe (before D so the overfit batch fits)
    del l, l2, tot, bce, dice, base
    torch.cuda.empty_cache()
    res["E_baseline_alloc_gb"] = torch.cuda.memory_allocated() / 2**30
    probe = []
    for bs in cfg["probe_batches"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            m = build_model(cfg, None)
            opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"])
            xb = torch.randn(bs, 2, 512, 512, device="cuda")
            yb, vb = torch.rand(bs, 512, 512, device="cuda") < 0.05, torch.ones(bs, 512, 512, dtype=torch.bool, device="cuda")
            for it in range(13):
                if it == 3:
                    torch.cuda.synchronize()
                    t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = seg_loss(m(xb)[:, 0], yb, vb, 5.0, cfg)[0]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            probe.append(dict(batch=bs, ok=True, img_per_s=10 * bs / (time.time() - t0),
                              peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30,
                              peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30))
        except torch.OutOfMemoryError:
            probe.append(dict(batch=bs, ok=False))
            break
        finally:
            m = opt = xb = None
        print(probe[-1], flush=True)
    fit = [p for p in probe if p["ok"] and p["peak_reserved_gb"] <= cfg["probe_max_reserved_gb"]]
    res["E_vram_probe"] = dict(results=probe, rule=f"largest batch with peak reserved <= {cfg['probe_max_reserved_gb']} GB",
                               chosen_batch=max(p["batch"] for p in fit))
    torch.cuda.empty_cache()

    # D. tiny-set overfit on fixed oil-positive crops (no augmentation)
    rows, cls, ids, vv, lab = load("train")
    st = train_stats(vv[:64], lab[:64])  # normalisation for this test only
    sampler = CropSampler(cls, lab, dict(cfg, mix=dict(oil_pos=1.0, lookalike=0.0, no_oil=0.0, oil_bg=0.0)), cfg["seed"])
    spec = sampler.sample(cfg["overfit_n"])
    v, valid, y = gather(vv, lab, spec, cfg["crop"])
    x = make_input(v, valid, st["norm_mu"], st["norm_sd"])
    seed_all(cfg["seed"])
    m = build_model(cfg, cfg["encoder_weights"])
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["overfit_lr"], weight_decay=0.0)
    bs = min(res["E_vram_probe"]["chosen_batch"], cfg["overfit_n"])
    curve, reached = [], None
    for step in range(1, cfg["overfit_steps"] + 1):
        m.train()
        for i in range(0, len(x), bs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = seg_loss(m(x[i:i + bs])[:, 0], y[i:i + bs], valid[i:i + bs], 5.0, cfg)[0]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if step % 25 == 0:
            m.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(torch.cat([m(x[i:i + bs])[:, 0].float() for i in range(0, len(x), bs)]))
            met = prf(*counts(p >= 0.5, y, valid).tolist())
            curve.append(dict(step=step, loss=loss.item(), **met))
            print(curve[-1], flush=True)
            if met["dice"] >= cfg["overfit_target_dice"]:
                reached = step
                break
    # alignment: Dice of prediction against the target shifted by d px peaks at d = 0
    pred = p >= 0.5
    shift = {f"{ax}{d:+d}": prf(*counts(pred, torch.roll(y, d, dims=2 if ax == "x" else 1), valid).tolist())["dice"]
             for ax in ("x", "y") for d in (-8, -4, -2, -1, 0, 1, 2, 4, 8)}
    best_shift = max(shift, key=shift.get)
    res["D_overfit"] = dict(n_crops=len(spec), oil_frac=float((y & valid).sum() / valid.sum()), lr=cfg["overfit_lr"],
                            target_dice=cfg["overfit_target_dice"], reached_at_step=reached, final=curve[-1],
                            curve=curve, alignment_dice_by_shift=shift, best_shift=best_shift)
    assert reached is not None, "tiny-set overfit failed: STOP and debug"
    assert best_shift in ("x+0", "y+0"), "prediction misaligned with target"
    res["all_passed"] = True
    json.dump(res, open(out / "pretraining_tests.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "D_overfit"}, indent=1))
    print(json.dumps({k: v for k, v in res["D_overfit"].items() if k != "curve"}, indent=1))


# ---------------------------------------------------------------------------------------------------- training
def train(cfg, out):
    t_all = time.time()
    tests_res = json.load(open(out / "pretraining_tests.json"))
    assert tests_res.get("all_passed"), "pre-training tests have not passed"
    assert not (out / "best.pt").exists(), f"{out} already has a run"
    cfg["batch"] = tests_res["E_vram_probe"]["chosen_batch"]
    seed_all(cfg["seed"])
    tr_rows, tr_cls, _, tr_vv, tr_lab = load("train")
    va_rows, va_cls, _, va_vv, va_lab = load("val")
    cfg.update(train_stats(tr_vv, tr_lab), n_train=len(tr_rows), n_val=len(va_rows),
               train_tiles=pd.Series(tr_cls).value_counts().to_dict(), val_tiles=pd.Series(va_cls).value_counts().to_dict())
    cfg["pos_weight"] = min(cfg["raw_pos_weight"], cfg["pos_weight_cap"])
    sampler = CropSampler(tr_cls, tr_lab, cfg, cfg["seed"])
    cfg.update(oil_bg_candidate_windows=len(sampler.bg), oil_bg_candidate_tiles=sampler.bg_tiles)
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    print(json.dumps(cfg), flush=True)

    model = build_model(cfg, cfg["encoder_weights"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_ep = -(-cfg["samples_per_epoch"] // cfg["batch"])
    total, warm = cfg["epochs"] * steps_ep, cfg["warmup_epochs"] * steps_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, 1 - (s - warm) / max(total - warm, 1)) ** cfg["poly_power"])
    g = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    hist, samp, best, since, t0 = [], [], -1.0, 0, time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        acc = np.zeros(4)
        te = time.time()
        for _ in range(steps_ep):
            spec = sampler.sample(cfg["batch"])
            v, valid, y = gather(tr_vv, tr_lab, spec, cfg["crop"])
            for (_, _, _, kind), vf, pf in zip(spec, valid.float().mean((1, 2)).tolist(), (y & valid).float().mean((1, 2)).tolist()):
                samp.append((ep, kind, vf, pf))
            v, valid, y = augment(v, valid, y, cfg, g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(make_input(v, valid, cfg["norm_mu"], cfg["norm_sd"]))[:, 0]
            tot, bce, dice = seg_loss(logits, y, valid, cfg["pos_weight"], cfg)
            opt.zero_grad(set_to_none=True)
            tot.backward()
            opt.step()
            sched.step()
            acc += [tot.item(), bce.item(), dice.item(), 1]
        tr_s = time.time() - te
        vm = val_epoch(model, va_vv, va_lab, cfg, cfg["pos_weight"])
        hist.append(dict(epoch=ep, train_loss=acc[0] / acc[3], train_bce=acc[1] / acc[3], train_dice_loss=acc[2] / acc[3],
                         val_bce=vm["bce"], val_iou=vm["iou"], val_dice=vm["dice"], val_precision=vm["precision"],
                         val_recall=vm["recall"], lr=sched.get_last_lr()[0], train_s=tr_s, epoch_s=time.time() - te))
        if vm["iou"] > best:
            best, since = vm["iou"], 0
            torch.save(dict(state_dict=model.state_dict(), epoch=ep, val_iou=vm["iou"], cfg=cfg), out / "best.pt")
        else:
            since += 1
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in hist[-1].items()), "*" if since == 0 else "", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if since >= cfg["patience"]:
            print(f"early stop: no val IoU gain for {cfg['patience']} epochs", flush=True)
            break
    runtime = time.time() - t0
    s = pd.DataFrame(samp, columns=["epoch", "kind", "valid_frac", "oil_frac"])
    s.to_parquet(out / "sampling_log.parquet", index=False)
    sampling = s.groupby("kind").agg(n=("kind", "size"), mean_valid_frac=("valid_frac", "mean"), mean_oil_frac=("oil_frac", "mean"),
                                     crops_with_oil=("oil_frac", lambda x: int((x > 0).sum())))
    sampling["share"] = sampling.n / sampling.n.sum()
    done = dict(best_epoch=torch.load(out / "best.pt", map_location="cpu", weights_only=False)["epoch"], best_val_iou_at_0_5=best,
                epochs_run=len(hist), steps_per_epoch=steps_ep, train_runtime_s=runtime, total_runtime_s=time.time() - t_all,
                peak_vram_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                peak_vram_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
                checkpoint_mb=(out / "best.pt").stat().st_size / 2**20,
                sampled_crop_oil_frac=float(s.oil_frac.mean()),
                sampling=sampling.reset_index().to_dict("records"))
    json.dump(done, open(out / "training_complete.json", "w"), indent=1)
    plot_curves(pd.DataFrame(hist), done["best_epoch"], out)
    print(json.dumps(done, indent=1))


def plot_curves(h, best_ep, out):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="train total (0.5 BCE + 0.5 Dice, sampled crops)")
    ax[0].plot(h.epoch, h.train_bce, ".-", label="train BCE")
    ax[0].plot(h.epoch, h.train_dice_loss, ".-", label="train Dice loss")
    ax[1].plot(h.epoch, h.val_bce, "o-", c="C3", label="val BCE (full tiles)")
    for k in ("val_iou", "val_dice", "val_precision", "val_recall"):
        ax[2].plot(h.epoch, h[k], "o-", label=f"{k} @0.5")
    for a in ax:
        a.axvline(best_ep, ls=":", c="r", label=f"best epoch {best_ep}")
        a.set_xlabel("epoch")
        a.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "curves.png", dpi=90)


# ---------------------------------------------------------------------------------------------------- validation
def validate(out):
    assert (out / "training_complete.json").exists(), "training not complete"
    ck = torch.load(out / "best.pt", map_location="cuda", weights_only=False)
    cfg = ck["cfg"]
    seed_all(cfg["seed"])
    model = build_model(cfg, None)
    model.load_state_dict(ck["state_dict"])
    rows, cls, ids, vv, lab = load("val")
    thr = torch.tensor(cfg["sweep"], device="cuda")
    sw = torch.zeros(len(thr), 3, dtype=torch.float64, device="cuda")
    c05 = torch.zeros(3, dtype=torch.float64, device="cuda")
    for _, p, valid, y in infer(model, vv, lab, cfg["norm_mu"], cfg["norm_sd"], cfg["eval_batch"]):
        c05 += counts(p >= 0.5, y, valid)
        for k in range(len(thr)):
            sw[k] += counts(p >= thr[k], y, valid)
    at05 = prf(*c05.tolist())
    assert abs(at05["iou"] - ck["val_iou"]) < 1e-6, "reloaded checkpoint does not reproduce val IoU@0.5"
    sweep = pd.DataFrame([dict(threshold=float(t), **prf(*c.tolist())) for t, c in zip(cfg["sweep"], sw)])
    sweep.to_csv(out / "val_threshold_sweep.csv", index=False)
    t_sel = float(sweep.threshold[sweep.dice.idxmax()])
    json.dump(dict(threshold=t_sel, rule="max global val Dice over sweep 0.10..0.90 step 0.05; frozen; test never used",
                   checkpoint_epoch=ck["epoch"]), open(out / "frozen_threshold.json", "w"), indent=1)

    # per-tile pass at 0.5 and at the frozen threshold
    tiles, keep = [], {}
    for i, p, valid, y in infer(model, vv, lab, cfg["norm_mu"], cfg["norm_sd"], cfg["eval_batch"]):
        for j in range(len(p)):
            n = i + j
            vj, yj = valid[j], y[j]
            r = dict(row=int(rows[n]), id=ids[n], cls=cls[n], valid_px=int(vj.sum()), gt_px=int((yj & vj).sum()))
            for tag, t in (("t05", 0.5), ("tsel", t_sel)):
                pr = (p[j] >= t) & vj
                tp, fp, fn = counts(pr, yj, vj).tolist()
                lbl, nc = ndimage.label(pr.cpu().numpy())
                gt = yj.cpu().numpy()
                sizes = np.bincount(lbl.ravel())[1:]
                hit = np.zeros(nc + 1, bool)
                hit[np.unique(lbl[gt & (lbl > 0)])] = True
                false_sizes = sizes[~hit[1:]]
                glbl, ng = ndimage.label(gt)
                missed = ng - len(np.unique(glbl[pr.cpu().numpy() & (glbl > 0)]))
                r.update({f"{tag}_pred_px": int(pr.sum()), f"{tag}_tp": tp, f"{tag}_fp": fp, f"{tag}_fn": fn,
                          f"{tag}_fp_frac": fp / max(r["valid_px"], 1), f"{tag}_dice": 2 * tp / max(2 * tp + fp + fn, 1) if r["gt_px"] else np.nan,
                          f"{tag}_recall": tp / max(tp + fn, 1) if r["gt_px"] else np.nan, f"{tag}_n_comp": nc,
                          f"{tag}_n_false_comp": int(len(false_sizes)), f"{tag}_largest_false_comp_px": int(false_sizes.max()) if len(false_sizes) else 0,
                          f"{tag}_gt_comp": ng, f"{tag}_missed_gt_comp": int(missed)})
            tiles.append(r)
            keep[n] = (p[j] * 255).round().byte().cpu().numpy()  # u8 prob for figures
    T = pd.DataFrame(tiles)
    T.to_parquet(out / "val_tiles.parquet", index=False)

    def group(tag):
        res = {}
        c = T[[f"{tag}_tp", f"{tag}_fp", f"{tag}_fn"]].sum()
        res["global_pixel"] = prf(*c.tolist())
        o = T[T.cls == "oil"]
        res["oil_tiles"] = dict(n=len(o), pixel=prf(*o[[f"{tag}_tp", f"{tag}_fp", f"{tag}_fn"]].sum().tolist()),
                                tile_dice_mean=float(o[f"{tag}_dice"].mean()), tile_dice_median=float(o[f"{tag}_dice"].median()),
                                tile_recall_mean=float(o[f"{tag}_recall"].mean()),
                                tiles_detected_any_tp=int((o[f"{tag}_tp"] > 0).sum()),
                                tiles_missed_entirely=int((o[f"{tag}_tp"] == 0).sum()),
                                false_components_total=int(o[f"{tag}_n_false_comp"].sum()),
                                largest_false_component_px=int(o[f"{tag}_largest_false_comp_px"].max()),
                                gt_components=int(o[f"{tag}_gt_comp"].sum()), missed_gt_components=int(o[f"{tag}_missed_gt_comp"].sum()))
        for c_ in ("lookalike", "no_oil"):
            n = T[T.cls == c_]
            res[f"{c_}_tiles"] = dict(n=len(n), frac_tiles_any_pred=float((n[f"{tag}_pred_px"] > 0).mean()),
                                      mean_fp_frac=float(n[f"{tag}_fp_frac"].mean()), p95_fp_frac=float(n[f"{tag}_fp_frac"].quantile(0.95)),
                                      max_fp_frac=float(n[f"{tag}_fp_frac"].max()),
                                      false_components_total=int(n[f"{tag}_n_false_comp"].sum()),
                                      false_components_per_tile=float(n[f"{tag}_n_false_comp"].mean()),
                                      largest_false_component_px=int(n[f"{tag}_largest_false_comp_px"].max()))
        return res

    met = dict(checkpoint_epoch=ck["epoch"], threshold_selected=t_sel, pixel_area_note=PIXEL_AREA_NOTE,
               at_0_5=group("t05"), at_selected=group("tsel"))
    json.dump(met, open(out / "val_metrics.json", "w"), indent=1)
    figures(T, keep, vv, lab, t_sel, out)
    print(sweep.to_string(index=False))
    print(json.dumps(met, indent=1))


def figures(T, keep, vv, lab, t, out):
    """Fixed, rule-based selection (seeded random within each rule; no hand picking)."""
    rng = np.random.default_rng(26143)
    o = T[T.cls == "oil"].copy()
    q = o.gt_px.quantile([0.25, 0.75])
    pick = lambda df, k: list(rng.choice(df.index, min(k, len(df)), replace=False)) if len(df) else []
    sel = [("good detection (tile Dice >= 0.75, random)", pick(o[o.tsel_dice >= 0.75], 3)),
           ("small slick (GT px <= q25, random)", pick(o[o.gt_px <= q[0.25]], 3)),
           ("large slick (GT px >= q75, random)", pick(o[o.gt_px >= q[0.75]], 3)),
           ("worst false negative (lowest oil recall)", list(o.nsmallest(3, "tsel_recall").index)),
           ("worst false positive (negative tile, max FP frac)", list(T[T.cls != "oil"].nlargest(3, "tsel_fp_frac").index)),
           ("LookAlike (random)", pick(T[T.cls == "lookalike"], 3)),
           ("NoOil (random)", pick(T[T.cls == "no_oil"], 3))]
    rows = [(tag, i) for tag, ix in sel for i in ix]
    fig, ax = plt.subplots(len(rows), 5, figsize=(15, 3.1 * len(rows)))
    for r, (tag, i) in enumerate(rows):
        valid, y = lab[i] != 255, lab[i] == 1
        prob = keep[i] / 255.0
        pred = (prob >= t) & valid
        ov = np.stack([vv[i].astype(np.float32)] * 3, -1)
        ov[pred & y] = (0, 1, 0)
        ov[pred & ~y] = (1, 0, 0)
        ov[~pred & y] = (0, 0.4, 1)
        ov[~valid] = (1, 1, 1)
        tr = T.loc[i]
        panels = [(np.where(valid, vv[i], np.nan), "gray", f"{tag}\n{tr.cls} {tr.id}  VV"),
                  (np.where(valid, y, np.nan), "gray", f"GT oil px={tr.gt_px}"),
                  (np.where(valid, prob, np.nan), "magma", "probability"),
                  (np.where(valid, pred, np.nan), "gray", f"pred @ {t:.2f}  Dice={tr.tsel_dice:.2f}" if tr.gt_px else f"pred @ {t:.2f}  FP frac={tr.tsel_fp_frac:.4f}"),
                  (ov, None, "overlay: TP green, FP red, FN blue, invalid white")]
        for c, (a, cm, title) in enumerate(panels):
            ax[r, c].imshow(a, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
            ax[r, c].set_title(title, fontsize=7)
            ax[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(out / "val_examples.png", dpi=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tests", "train", "validate"])
    ap.add_argument("--out", default=str(ROOT / "results/E005_segformer_b2"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    {"tests": lambda: tests(dict(CFG), out), "train": lambda: train(dict(CFG), out), "validate": lambda: validate(out)}[a.stage]()


if __name__ == "__main__":
    main()
