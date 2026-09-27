"""Phase C: leakage-safe split proposal for the SAR dataset (no pixels are read).

Grouping unit = geo_group: connected components of tiles whose footprints overlap at all
(covers same-acquisition overlapping crops AND same-place/different-date revisits).

- trainval tiles overlapping the official test are flagged `test_contaminated` and excluded from
  training. POLICY (argv[1]):
    acq    (default) exclude trainval tiles sharing an ACQUISITION (identical pixels) with test (152)
    strict exclude trainval tiles sharing any PLACE (footprint overlap component) with test (711)
  Under `acq`, 236 test tiles still share a place (other date) with train: always also report the
  `test_shares_place_with_trainval == False` subset.
- remaining trainval -> StratifiedGroupKFold(5) on class, grouped by geo_group; fold 0 = val.
- also flags test tiles that share an acquisition with trainval (for a "strict" test report).

Output: results/sar_audit/splits_v1_<policy>.parquet (+ printed summary)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "results/sar_audit"
SEED = 26143
POLICY = sys.argv[1] if len(sys.argv) > 1 else "acq"
assert POLICY in ("acq", "strict")

idx = pd.read_parquet(A / "sar_index.parquet")
grp = pd.read_parquet(A / "groups.parquet")
assert (idx.id.values == grp.id.values).all() and (idx.split.values == grp.split.values).all()
d = idx[["split", "cls", "id", "img", "mask", "h", "w", "valid_frac", "mask_pos_frac"]].copy()
d[["geo_group", "scene_group"]] = grp[["geo_group", "scene_group"]].values

test_geo = set(d.loc[d.split == "test", "geo_group"])
test_scene = set(d.loc[d.split == "test", "scene_group"])
tv_scene = set(d.loc[d.split == "trainval", "scene_group"])
d["test_contaminated"] = (d.split == "trainval") & (d.geo_group.isin(test_geo) if POLICY == "strict" else d.scene_group.isin(test_scene))
d["test_shares_acq_with_trainval"] = (d.split == "test") & d.scene_group.isin(tv_scene)
d["test_shares_place_with_trainval"] = (d.split == "test") & d.geo_group.isin(set(d.loc[d.split == "trainval", "geo_group"]))
d["usable"] = d.valid_frac >= 0.05  # 3 official-test no_oil tiles are ~all nodata

d["fold"] = -1
pool = d[(d.split == "trainval") & ~d.test_contaminated]
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for k, (_, te) in enumerate(sgkf.split(pool, pool.cls, pool.geo_group)):
    d.loc[pool.index[te], "fold"] = k
d["role"] = np.select(
    [d.split == "test", d.test_contaminated, d.fold == 0, d.fold > 0],
    ["test", "excluded_test_overlap", "val", "train"], default="unassigned")

d.to_parquet(A / f"splits_v1_{POLICY}.parquet", index=False)
print(pd.crosstab(d.role, d.cls, margins=True))
print("\nfold x class (trainval pool):\n", pd.crosstab(d[d.fold >= 0].fold, d[d.fold >= 0].cls))
t = d[d.split == "test"]
print("\ntest sharing acquisition w/ trainval:", int(t.test_shares_acq_with_trainval.sum()),
      "| sharing place:", int(t.test_shares_place_with_trainval.sum()), "| unusable (nodata):", int((~t.usable).sum()))
tr, va = set(d[d.role == "train"].geo_group), set(d[d.role == "val"].geo_group)
assert not tr & va, "train/val group leakage"
assert not set(d[d.role.isin(["train", "val"])].scene_group) & test_scene, "test acquisition leakage"
if POLICY == "strict":
    assert not (tr | va) & test_geo, "test place leakage"
print(f"assert OK ({POLICY}): train/val disjoint geo_groups; no shared acquisition with test")
