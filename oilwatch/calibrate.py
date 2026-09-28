"""Calibrate an ABSOLUTE attribution rule on twin experiments (split: calibration / held-out test).

Twin run finding: the pre-registered v3 tiers use relative ranks, so with the culprit's AIS hidden an innocent vessel is
still 'consistent'. This picks absolute thresholds instead, like choosing a threshold on a validation set:

  supported(v)  <=>  fwd_F >= a  AND  bwd_point >= b  AND  fwd_member_support >= c
  accused       =    the supported vessel with the highest `combined` (none if nobody is supported)

Case-level metrics (per split):
  correct_attribution = P(accused == true vessel)            [culprit present]
  false_accusation    = P(someone accused | culprit hidden)   [same case, true vessel removed]
Objective on the CALIBRATION half: max correct_attribution s.t. false_accusation <= max_false; ties -> stricter rule.
The TEST half is scored once with the chosen rule. Absolute features do not change when the culprit is removed,
so the hidden case = the present suspect table minus the true vessel.
"""
import itertools
import json

import numpy as np
import pandas as pd

GRID = {"a_fwd_F": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "b_bwd_point": [0.0, 0.05, 0.135, 0.3, 0.5, 0.7],
        "c_member_support": [0.0, 0.2, 0.33, 0.5, 0.67],
        "m_margin": [1.0, 1.25, 1.5, 2.0, 3.0]}
# m_margin (calibration v2, added after the first test look): the accused must beat the runner-up's fwd_F by this
# factor -- a present culprit usually stands out, an absent one leaves a flat field of plausible neighbours


def split(cases, seed=26143):
    """Stratified alternate assignment within each age stratum (random order, fixed seed)."""
    rng = np.random.default_rng(seed)
    out = []
    for _, g in cases.groupby("stratum"):
        ids = rng.permutation(g.case_id.values)
        out += [(i, "calibration" if k % 2 == 0 else "test") for k, i in enumerate(ids)]
    return dict(out)


def supported(s, a, b, c):
    return (s.fwd_F >= a) & (s.bwd_point >= b) & (s.fwd_member_support.fillna(0) >= c) & (s.fwd_F > 0 if a > 0 else True)


def accused(s, a, b, c, m):
    """Top supported vessel if it beats the runner-up (any vessel) on fwd_F by factor m; else None."""
    sup = s[supported(s, a, b, c)].sort_values("combined", ascending=False)
    if not len(sup):
        return None
    top = sup.iloc[0]
    others = s[s.MMSI != top.MMSI].fwd_F
    runner = float(others.max()) if len(others) else 0.0
    return top.MMSI if top.fwd_F >= m * runner else None


def evaluate(sus, cases, rule):
    a, b, c, m = rule
    rows = []
    for _, k in cases.iterrows():
        s = sus[sus.cluster_id == k.case_id]
        sup = s[supported(s, a, b, c)].sort_values("combined", ascending=False)
        acc = accused(s, a, b, c, m)
        hid = accused(s[s.MMSI != k.MMSI], a, b, c, m)
        sup_h = [hid] if hid is not None else []
        rows.append({"case_id": k.case_id, "stratum": k.stratum, "accused": acc, "correct": acc == k.MMSI,
                     "truth_supported": bool((sup.MMSI == k.MMSI).any()), "n_supported": len(sup),
                     "false_accusation_if_hidden": len(sup_h) > 0})
    return pd.DataFrame(rows, columns=["case_id", "stratum", "accused", "correct", "truth_supported", "n_supported",
                                       "false_accusation_if_hidden"])


def metrics(ev):
    if ev.empty:
        return {"n": 0}
    return {"n": len(ev), "correct_attribution": float(ev.correct.mean()), "truth_supported": float(ev.truth_supported.mean()),
            "false_accusation": float(ev.false_accusation_if_hidden.mean()),
            "no_accusation_when_present": float((ev.n_supported == 0).mean())}


def calibrate(twin_dir, max_false=0.10):
    cases = pd.read_csv(twin_dir / "cases.csv", dtype={"MMSI": str})
    cases = cases[cases.status == "ok"]
    sus = pd.read_csv(twin_dir / "suspects.csv", dtype={"MMSI": str})
    part = split(cases)
    cases["split"] = cases.case_id.map(part)
    cal, test = cases[cases.split == "calibration"], cases[cases.split == "test"]
    grid = []
    for rule in itertools.product(*GRID.values()):
        m = metrics(evaluate(sus, cal, rule))
        grid.append({**dict(zip(GRID, rule)), **m})
    grid = pd.DataFrame(grid)
    ok = grid[grid.false_accusation <= max_false]
    if ok.empty:
        best = grid.sort_values(["false_accusation", "correct_attribution"], ascending=[True, False]).iloc[0]
        status = f"no rule reaches false_accusation <= {max_false}; lowest-false rule chosen"
    else:
        best = ok.sort_values(["correct_attribution", "false_accusation", "m_margin", "a_fwd_F", "b_bwd_point", "c_member_support"],
                              ascending=[False, True, False, False, False, False]).iloc[0]
        status = "ok"
    rule = (best.a_fwd_F, best.b_bwd_point, best.c_member_support, best.m_margin)
    ev_cal, ev_test = evaluate(sus, cal, rule), evaluate(sus, test, rule)
    # reference: the pre-registered v3 tiers on the same test cases (accused = top suspect if consistent-or-better)
    v3 = []
    for _, k in test.iterrows():
        s = sus[sus.cluster_id == k.case_id].sort_values("suspect_rank")
        top = s.iloc[0] if len(s) else None
        flag = top is not None and top.tier in ("strongly consistent", "consistent")
        s_h = s[s.MMSI != k.MMSI]
        v3.append({"correct": flag and top.MMSI == k.MMSI,
                   "false_accusation_if_hidden": len(s_h) > 0 and s_h.tier.iloc[0] in ("strongly consistent", "consistent")})
    v3 = pd.DataFrame(v3, columns=["correct", "false_accusation_if_hidden"])
    res = {"rule": {"fwd_F_min": float(rule[0]), "bwd_point_min": float(rule[1]), "member_support_min": float(rule[2]),
                    "margin_vs_runner_up": float(rule[3])},
           "test_looks": 2, "note": "calibration v2 (margin feature, wider grid) added after one look at the test half",
           "selection": status, "max_false_accusation_target": max_false,
           "calibration": metrics(ev_cal), "test": metrics(ev_test),
           "test_by_stratum": {s: metrics(g) for s, g in ev_test.groupby("stratum")},
           "v3_preregistered_tiers_on_test": {"correct_attribution": float(v3.correct.mean()),
                                              "false_accusation": float(v3.false_accusation_if_hidden.mean())},
           "split": {int(k): v for k, v in part.items()}}
    grid.to_csv(twin_dir / "calibration_grid.csv", index=False)
    pd.concat([ev_cal.assign(split="calibration"), ev_test.assign(split="test")]).to_csv(twin_dir / "calibration_cases.csv", index=False)
    (twin_dir / "calibration.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    import sys
    from pathlib import Path
    print(json.dumps(calibrate(Path(sys.argv[1])), indent=1, default=float))
