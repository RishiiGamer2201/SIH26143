"""E008 candidate policy v1 (reviewer-approved 2026-09-28): freeze, apply ONCE to the frozen E007 components, then
Cerulean comparison POST HOC only. No physics here.

  python scripts/e008_candidate_policy_v1.py freeze    # configs/e008_candidate_policy_v1.json + .sha256 (never overwritten; changes = v2)
  python scripts/e008_candidate_policy_v1.py apply     # 49 E007 components -> area filter -> one component = one candidate
  python scripts/e008_candidate_policy_v1.py posthoc   # AFTER apply: all candidates intersecting Cerulean (no winner)

Policy: autonomous multi-candidate. Frozen E005 4-connected components from E007 -> drop if area_m2 < 5,000 m2 ->
every retained component is exactly one candidate. grouping mode "none": NO pairwise-distance clustering is performed
(G = 0 single linkage is NOT equivalent: diagonally touching 4-connected components have polygon distance 0).
Candidate identity = original E007 component id ("E007-C{component_id}"), ordered by component id (no ranking).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e007_incident001_ml as e007  # noqa: E402
import e007b_component_structure as e007b  # noqa: E402
from e006_segformer_test import git, sha256  # noqa: E402

ROOT = e007.ROOT
CFG = ROOT / "configs/e008_candidate_policy_v1.json"
SHA = ROOT / "configs/e008_candidate_policy_v1.json.sha256"
E007C = ROOT / "results/E007c_validation_clustering"
E007_FROZEN = e007.OUT / "FROZEN.json"
OUT = ROOT / "results/E008_candidate_policy_v1"
PREDECLARATION_COMMIT = "c56fdde"   # E007c rules committed before analysis
RESULT_COMMIT = "cf51f5b"           # E007c analysis completed; no G satisfied the rule
DECISION_DATE = "2026-09-28"
CAND_COLS = ["candidate_id", "component_id", "area_m2", "perimeter_m", "centroid_lon", "centroid_lat", "major_axis_m", "minor_axis_m",
             "width_proxy_m", "orientation_deg_from_north", "elongation", "compactness_polsby_popper", "prob_mean", "prob_max",
             "n_px_image_space", "bbox_west", "bbox_south", "bbox_east", "bbox_north"]


def verify_policy():
    h = SHA.read_text().split()[0]
    assert sha256(CFG) == h, "policy JSON changed since freeze (changes require v2)"
    return json.load(open(CFG)), h


def freeze():
    assert not CFG.exists() and not SHA.exists(), f"{CFG} exists; v1 is frozen (changes require v2)"
    rep = json.load(open(E007C / "validation_clustering_report.json"))
    ext = json.load(open(E007C / "extract_report.json"))
    # Known (recorded, not fixed: v1 is hashed): default fast float parsing puts 14 descriptive floats within <=1.1e-16 (1 ulp)
    # of the CSV text; the CSVs (sha256 recorded below) are authoritative. Kept as-is so this code reproduces the frozen JSON.
    S = pd.read_csv(E007C / "gap_sweep.csv")
    S = S[S.variant == "none"].set_index("G_m")
    A = pd.read_csv(E007C / "area_threshold_sweep.csv").set_index("A_min_m2")
    a_min = rep["A_min_rule"]["selected_A_min_m2"]
    assert a_min == 5000 and rep["G_rule"]["selected_G_m"] is None and rep["orientation_guard"]["adopted_theta_deg"] is None
    e005_fz = json.load(open(ROOT / "results/E005_segformer_b2/FROZEN.json"))
    full = lambda c: git("rev-parse", c)
    sweep = {str(g): dict(r_area=S.r_area[g], wrong_merge_rate=S.wrong_rate[g], wrong_merge_rate_ci95=[S.wrong_rate_ci_lo[g], S.wrong_rate_ci_hi[g]],
                          r_area_ci95=[S.r_area_ci_lo[g], S.r_area_ci_hi[g]]) for g in [100, 250, 1000, 5000]}
    cfg = dict(
        version="v1", status="frozen", mode="autonomous_multi_candidate",
        reviewer_decision=dict(date=DECISION_DATE, decision="Option 1: no grouping + A_min 5,000 m2 + no orientation guard (E008_SELECTION_POLICY.md)"),
        segmentation_source=dict(experiment="E007", frozen_manifest=str(E007_FROZEN.relative_to(ROOT)), frozen_manifest_sha256=sha256(E007_FROZEN),
                                 components="results/E007_incident001_ml/incident001_predicted_components.geojson",
                                 model="E005 SegFormer MiT-B2 (frozen)", e005_checkpoint_sha256=e005_fz["sha256"]["results/E005_segformer_b2/best.pt"]["sha256"],
                                 threshold=0.60, component_connectivity=4, area="geodesic WGS84 area_m2 from E007 (never pixel_count * 400)"),
        filter=dict(minimum_area_m2=a_min, comparison="area_m2 >= minimum_area_m2", excluded_reason=f"area < {a_min} m2"),
        grouping=dict(mode="none", link_gap_m=None, orientation_guard=None,
                      rule="after the area filter, each original E007 4-connected component is exactly one candidate; no pairwise-distance clustering",
                      reason="No tested global gap satisfied the predeclared validation constraints.",
                      note="Not implemented as G = 0 single linkage: diagonally touching 4-connected components have polygon distance 0. "
                           "A fragmented physical slick may produce several candidates; that is intentional."),
        candidate_identity=dict(id="E007-C{component_id}", order="original component_id (no ranking)"),
        validation_basis=dict(
            source="E005 validation split only", n_validation_tiles=ext["val_tiles"], val_iou_reproduced=ext["val_iou_reproduced"],
            n_validation_pred_components=ext["n_pred_components"], n_fragmented_gt=rep["counts"]["fragmented_gt"],
            predeclared_fragment_recovery_target=rep["predeclared"]["R_AREA_MIN"], predeclared_wrong_merge_max=rep["predeclared"]["WRONG_MAX"],
            gap_grid_m=rep["predeclared"]["G_GRID"], global_gap_rule_satisfied=False, gap_sweep=sweep,
            no_grouping_r_area=S.r_area[0], no_grouping_r_area_ci95=[S.r_area_ci_lo[0], S.r_area_ci_hi[0]],
            no_grouping_meaning="fraction of fragmented-slick overlap area held by the dominant individual component; the rest remains as separate candidates, not deleted",
            area_rule=rep["A_min_rule"]["rule"], area_filter_tp_area_retained=A.tp_area_retained[a_min],
            area_filter_fp_components_removed_fraction=A.fp_reduction[a_min], area_filter_fp_components_removed=int(A.fp_components_removed[a_min]),
            area_filter_tp_components_removed=int(A.tp_components_removed[a_min]),
            orientation_guard_adopted=False, gt_redefined_after_results=False, recovery_requirement_relaxed=False,
            limitation="wrong merges are counted against GT connected components; Trujillo masks are fragmented, so the rate is an upper bound",
            report=str((E007C / "validation_clustering_report.json").relative_to(ROOT)),
            report_sha256=sha256(E007C / "validation_clustering_report.json"),
            gap_sweep_sha256=sha256(E007C / "gap_sweep.csv"), area_sweep_sha256=sha256(E007C / "area_threshold_sweep.csv")),
        provenance=dict(e007c_predeclaration_commit=full(PREDECLARATION_COMMIT), e007c_result_commit=full(RESULT_COMMIT),
                        e007c_script="scripts/e007c_validate_clustering.py", policy_script=str(Path(__file__).resolve().relative_to(ROOT)),
                        git_head_at_freeze=git("rev-parse", "HEAD")),
        rationale=("Predeclared rule (fragment area recovery >= 0.95 AND wrong-GT merge <= 0.05) was met by no tested global gap "
                   "(100 m: 0.908 / 0.045; 250 m: 0.950 / 0.178; 1000 m: 0.989 / 0.492; 5000 m: 1.000 / 0.717), so no grouping is "
                   "applied. The A_min rule (predefined before the validation result was inspected) selected 5,000 m2."),
        forbidden_selection_inputs=["Cerulean polygon", "Cerulean centroid", "Cerulean bbox", "Cerulean ROI / E007 in_roi flag",
                                    "Incident 001 agreement", "E007 component gaps", "AIS candidates", "AIS scoring sigma", "E004 classifier gating"],
    )
    CFG.parent.mkdir(exist_ok=True)
    json.dump(cfg, open(CFG, "w"), indent=1, default=float)
    h = sha256(CFG)
    SHA.write_text(f"{h}  {CFG.name}\n")
    print(json.dumps(cfg, indent=1, default=float))
    print("policy sha256", h)


def apply():
    pol, h = verify_policy()
    assert pol["grouping"]["mode"] == "none" and pol["grouping"]["link_gap_m"] is None
    assert not (OUT / "application_report.json").exists(), "policy v1 already applied once"
    e007.verify()
    assert sha256(E007_FROZEN) == pol["segmentation_source"]["frozen_manifest_sha256"], "E007 FROZEN.json differs from the policy's"
    fc = json.load(open(ROOT / pol["segmentation_source"]["components"]))
    rec = [dict({k: v for k, v in f["properties"].items() if k != "in_roi"}, geometry=f["geometry"]) for f in fc["features"]]  # post-hoc flag dropped
    C = pd.DataFrame(rec).sort_values("component_id").reset_index(drop=True)
    assert len(C) == 49 and C.component_id.is_unique
    amin = pol["filter"]["minimum_area_m2"]
    keep = C.area_m2 >= amin
    K = C[keep].copy()
    K.insert(0, "candidate_id", [f"E007-C{i}" for i in K.component_id])  # one component = one candidate; no grouping
    X = C[~keep][["component_id", "area_m2"]].assign(reason=pol["filter"]["excluded_reason"])
    OUT.mkdir(parents=True, exist_ok=True)
    K[CAND_COLS].to_csv(OUT / "candidate_components.csv", index=False)
    X.to_csv(OUT / "excluded_components.csv", index=False)
    json.dump(dict(type="FeatureCollection", name="e008_candidate_policy_v1",
                   crs=dict(type="name", properties=dict(name="urn:ogc:def:crs:OGC:1.3:CRS84")),
                   description="E008 candidate policy v1: frozen E007 components with area >= 5000 m2; one component = one candidate; no grouping.",
                   features=[dict(type="Feature", geometry=r["geometry"], properties={c: r[c] for c in CAND_COLS}) for _, r in K.iterrows()]),
              open(OUT / "candidate_components.geojson", "w"), default=float)
    tot = float(C.area_m2.sum())
    rep = dict(policy="configs/e008_candidate_policy_v1.json", policy_sha256=h, e007_frozen_sha256=sha256(E007_FROZEN),
               grouping="none (each retained component is exactly one candidate; no pairwise-distance clustering performed)",
               original_component_count=len(C), retained_candidate_count=int(keep.sum()), excluded_count=int((~keep).sum()),
               retained_area_m2=float(K.area_m2.sum()), excluded_area_m2=float(X.area_m2.sum()), total_predicted_area_m2=tot,
               retained_area_fraction=float(K.area_m2.sum()) / tot, excluded_area_fraction=float(X.area_m2.sum()) / tot,
               candidate_ids=K.candidate_id.tolist(), excluded_component_ids=X.component_id.astype(int).tolist(),
               note="No candidate is designated as the Incident 001 slick. Candidate order = component id, not a ranking.",
               output_sha256={f: sha256(OUT / f) for f in ["candidate_components.csv", "candidate_components.geojson", "excluded_components.csv"]},
               git_head=git("rev-parse", "HEAD"))
    json.dump(rep, open(OUT / "application_report.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in rep.items() if k != "candidate_ids"}, indent=1, default=float))
    print(K[CAND_COLS[:12]].round(4).to_string(index=False))


def posthoc():
    """POST HOC only, after the candidate files are permanently written. Reports ALL intersecting candidates; selects nothing."""
    pol, h = verify_policy()
    app = json.load(open(OUT / "application_report.json"))
    for f, v in app["output_sha256"].items():
        assert sha256(OUT / f) == v, f"{f} changed after application"
    dst = OUT / "posthoc_cerulean_candidate_mapping.json"
    assert not dst.exists(), f"{dst} exists"
    C, fwd, back = e007b.load()  # same scene-centred LAEA as E007 vector(); in_roi dropped
    K = C[C.component_id.isin([int(c.split("C")[-1]) for c in app["candidate_ids"]])].sort_values("component_id")
    cer = e007.cerulean()
    cm = fwd(cer)
    cc = cm.centroid
    rows = []
    for r in K.itertuples():
        g = r.g_m
        inter = g.intersection(cm).area
        if inter <= 0:
            continue
        rows.append(dict(candidate_id=f"E007-C{r.component_id}", component_id=int(r.component_id), intersection_area_m2=inter,
                         candidate_area_m2=float(r.area_m2), candidate_area_m2_laea=g.area, reference_area_m2_laea=cm.area,
                         frac_candidate_inside_reference=inter / g.area, frac_reference_covered=inter / cm.area,
                         iou=inter / g.union(cm).area, centroid_distance_m=g.centroid.distance(cc)))
    covered = shapely.union_all([K.g_m[K.component_id == r["component_id"]].iloc[0] for r in rows]).intersection(cm).area if rows else 0.0
    out = dict(note="POST HOC only. Written after the candidate list was frozen. Does not alter the policy, the candidate list, its order "
                    "or downstream eligibility. No candidate is selected or declared the Incident 001 slick; rows are ordered by component id.",
               policy_sha256=h, candidate_files_sha256=app["output_sha256"], reference="data/incident_001/metadata/slick_3775938.geojson",
               areas="intersection / fractions / IoU in the E007 scene-centred LAEA (equal-area); candidate_area_m2 geodesic from E007",
               n_candidates=app["retained_candidate_count"], n_candidates_intersecting_reference=len(rows),
               intersecting_candidate_ids=[r["candidate_id"] for r in rows],
               reference_covered_by_all_intersecting_candidates=covered / cm.area, per_candidate=rows)
    json.dump(out, open(dst, "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "per_candidate"}, indent=1, default=float))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["freeze", "apply", "posthoc"])
    {"freeze": freeze, "apply": apply, "posthoc": posthoc}[ap.parse_args().stage]()
