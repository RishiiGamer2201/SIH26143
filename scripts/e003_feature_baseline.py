"""E003: feature/statistical baseline + shortcut probes on the acq split. CPU only, no tuning.

Features from data/sar/cache/v1 at 40 m (VV dB over valid blocks) + tile metadata.
Train on role == train, report on role == val (all / place-clean) as primary.
Official test (minus all-nodata tiles) scored 3 ways (full / acq-clean / place-clean) ONLY for the
fixed main model and the metadata-only shortcut probe; nothing is selected on test.
HistGradientBoosting defaults, balanced sample weights, seed 26143.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "data/sar/cache/v1"
OUT = ROOT / "results/E003_feature_baseline"
CLASSES = ["no_oil", "lookalike", "oil"]
SEED = 26143

IMG = ["vv_p01", "vv_p05", "vv_p25", "vv_p50", "vv_p75", "vv_p95", "vv_p99", "vv_mean", "vv_std", "vv_cv_lin",
       "frac_lt_m25", "frac_lt_m30", "frac_le_m40", "frac_gt_m10", "frac_gt_m5", "grad_mean", "valid_frac"]
META = ["lat_mid", "lon_mid", "px_x_m", "h", "w", "valid_frac"]
MODELS = {  # name -> features; *_test models also scored on test
    "image_stats_test": IMG,
    "metadata_only_test": META,
    "image_stats+latlon": IMG + ["lat_mid", "lon_mid"],
    "valid_frac_only": ["valid_frac"],
    "vv_p50_only": ["vv_p50"],
    "dark_clip_frac_only": ["frac_le_m40"],
    "bright_frac_only(land/ships)": ["frac_gt_m5"],
    "lat_only": ["lat_mid"],
}


def features():
    meta = pd.read_parquet(C / "meta.parquet")
    vv, lab = np.load(C / "vv40.npy", mmap_mode="r"), np.load(C / "lab40.npy", mmap_mode="r")
    rows = []
    for i in range(len(meta)):
        ok = lab[i] != 255
        db = vv[i].astype(np.float32) * 45 - 40
        r = {"valid_frac": float(ok.mean())}
        if ok.sum() > 100:
            d = db[ok]
            r.update(zip(["vv_p01", "vv_p05", "vv_p25", "vv_p50", "vv_p75", "vv_p95", "vv_p99"],
                         np.percentile(d, [1, 5, 25, 50, 75, 95, 99])))
            lin = 10 ** (d / 10)
            r.update(vv_mean=d.mean(), vv_std=d.std(), vv_cv_lin=lin.std() / lin.mean(),
                     frac_lt_m25=(d < -25).mean(), frac_lt_m30=(d < -30).mean(), frac_le_m40=(d <= -39.99).mean(),
                     frac_gt_m10=(d > -10).mean(), frac_gt_m5=(d > -5).mean())
            gx = np.abs(np.diff(db, axis=1))[ok[:, 1:] & ok[:, :-1]]
            gy = np.abs(np.diff(db, axis=0))[ok[1:] & ok[:-1]]
            r["grad_mean"] = float(np.concatenate([gx, gy]).mean())
        rows.append(r)
    f = pd.concat([meta, pd.DataFrame(rows)], axis=1)
    idx = pd.read_parquet(ROOT / "results/sar_audit/sar_index.parquet")[["split", "cls", "id", "lon_min", "lon_max"]]
    f = f.merge(idx, on=["split", "cls", "id"], how="left", validate="one_to_one")
    f["lon_mid"] = (f.lon_min + f.lon_max) / 2
    sp = pd.read_parquet(ROOT / "results/sar_audit/splits_v1_acq.parquet")
    f = f.merge(sp[["split", "cls", "id", "geo_group", "test_shares_acq_with_trainval", "test_shares_place_with_trainval"]],
                on=["split", "cls", "id"], how="left", validate="one_to_one")
    return f


def score(y, p):
    pr, rc, f1, n = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
    return dict(n=int(len(y)), accuracy=round(accuracy_score(y, p), 4), balanced_accuracy=round(balanced_accuracy_score(y, p), 4),
                macro_f1=round(f1_score(y, p, labels=CLASSES, average="macro"), 4),
                per_class={c: dict(precision=round(a, 4), recall=round(b, 4), f1=round(d, 4), support=int(e))
                           for c, a, b, d, e in zip(CLASSES, pr, rc, f1, n)},
                confusion_rows_true_cols_pred=confusion_matrix(y, p, labels=CLASSES).tolist())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = features()
    f.to_parquet(OUT / "features.parquet", index=False)
    tr, va, te = f[f.role == "train"], f[f.role == "val"], f[(f.role == "test") & (f.valid_frac > 0.05)]
    train_places = set(tr.geo_group)
    evals = {"val_all": va, "val_place_clean": va[~va.geo_group.isin(train_places)],
             "test_full_minus_nodata": te, "test_acq_clean": te[~te.test_shares_acq_with_trainval],
             "test_place_clean": te[~te.test_shares_place_with_trainval]}
    res = {"n_train": len(tr), "train_class_counts": tr.cls.value_counts().to_dict(),
           "eval_sizes": {k: v.cls.value_counts().to_dict() for k, v in evals.items()}, "models": {}}
    for name, cols in MODELS.items():
        clf = HistGradientBoostingClassifier(random_state=SEED)
        clf.fit(tr[cols], tr.cls, sample_weight=compute_sample_weight("balanced", tr.cls))
        res["models"][name] = {"features": cols}
        for k, d in evals.items():
            if k.startswith("test") and not name.endswith("_test"):
                continue
            res["models"][name][k] = score(d.cls, clf.predict(d[cols]))
    json.dump(res, open(OUT / "metrics.json", "w"), indent=1)

    print("train", res["train_class_counts"]); print("eval sizes", json.dumps(res["eval_sizes"]))
    rows = []
    for name, r in res["models"].items():
        for k in evals:
            if k in r:
                rows.append(dict(model=name, eval=k, n=r[k]["n"], acc=r[k]["accuracy"], bal_acc=r[k]["balanced_accuracy"],
                                 macro_f1=r[k]["macro_f1"], **{f"R_{c}": r[k]["per_class"][c]["recall"] for c in CLASSES}))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "metrics_table.csv", index=False)
    print(t.to_string(index=False))
    for k in ("val_all", "test_full_minus_nodata"):
        print(f"\nimage_stats confusion {k} (rows true {CLASSES}):", res["models"]["image_stats_test"][k]["confusion_rows_true_cols_pred"])
    print("\nfeature medians by split x class:")
    print(f.groupby(["split", "cls"])[["vv_p50", "frac_le_m40", "frac_gt_m5", "valid_frac", "lat_mid", "px_x_m", "grad_mean"]]
          .median().round(4).to_string())


if __name__ == "__main__":
    main()
