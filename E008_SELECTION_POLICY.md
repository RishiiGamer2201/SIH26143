# E008 source-geometry selection policy — DRAFT (updated 2026-09-28 after E007c)

> **DRAFT. No candidate policy is frozen.** E007c (validation only) found **no defensible single global link gap G** under
> the predeclared rule. `configs/e008_candidate_policy_v1.json` was therefore NOT written, and nothing was applied to the
> E007 components. A reviewer decision is needed (see "Open decision").

E008 seeds the backward hindcast with **our** predicted slick geometry, taken from the frozen E007 components
(`results/E007_incident001_ml/incident001_predicted_components.geojson`: 49 raw components, 4-connected, threshold 0.60).
This document defines which geometry becomes "the observed slick". Nothing here is implemented yet.

## Forbidden selectors (post-hoc evaluation only)

The following must never choose, crop, merge or rank the E008 seed geometry:
- the Cerulean polygon, its intersection or IoU with any component, its centroid, or its bbox;
- the E007 reporting ROI (Cerulean bbox ± 0.1°) and the `in_roi` flag;
- "the 13 components inside the ROI", the component closest to Cerulean, or the best-IoU component;
- `incident.json` geometry fields (area, length). Every field in that file is Cerulean-derived. Only
  `sentinel1_scene_id` and `slick_timestamp` are neutral.

**Consequence.** Incident 001 has no location cue independent of Cerulean. The scene was chosen because Cerulean
flagged a slick in it. A rule that "finds Incident 001" must therefore either use human judgement (Mode A) or
treat every candidate alike (Mode B).

E004 must not gate or filter E005 components. The reference-slick windows are argmax lookalike with p(oil) 0.12–0.34
(E007 negative result).

## Mode A: analyst-assisted case study

- **What the analyst sees:** calibrated VV (`incident001_vv_db.tif`), the E005 probability map, the thresholded
  components and optionally the model-only cluster figure (`results/E007b_component_structure/model_only_clusters.png`).
  No Cerulean overlay, ROI box, `incident.json` geometry or E007 comparison figure.
- **What the analyst does:** selects the component IDs forming the apparent slick cluster for the incident, and may
  merge them into one seed.
- **What is recorded** (in the E008 config, before any physics run):
  - the component IDs;
  - a free-text rationale;
  - the cue used (for example "largest coherent linear dark feature with a bright point target");
  - the analyst's identity and date;
  - whether the analyst had previously seen Cerulean or E007 post-hoc figures.
- **Blinding caveat.** Everyone involved so far, including Claude and the project owner, has seen the Cerulean
  overlay in E007. A selection made by any of us is **non-blind** and must be labelled as such. A blind selection
  needs an analyst who has not seen E007 post-hoc material.
- **Labelling.** Every downstream hindcast and attribution output is labelled **analyst-assisted, not fully
  autonomous**. The analyst's choice is a methodological input and must be reported as one.
- **Ambiguity.** The scene has at least three visually slick-like groups: the northern linear dark features around
  −89.1°, 28.2°N (with bright point targets at their heads), the group near −90.44°, 27.12°N, and a smaller linear
  feature near −90.22°, 27.39°N. With no independent cue, the choice among them is itself analyst judgement.

## Mode B: autonomous multi-candidate processing

- **Candidate definition.** A candidate is a spatially coherent group of predicted components: single-linkage
  clusters where the edge-to-edge gap is ≤ G, with the candidate's area ≥ A_min. The raw component geometry is kept;
  there is no smoothing, buffering or morphology.
- **Per-candidate processing.** Each candidate is hindcast and scored independently with the frozen physics
  (deterministic run, then the 27-scenario ensemble) and frozen V2 / V2.1 scoring. Each candidate gets its own output
  directory and its own ranked source list. There is no cross-candidate ranking, and "which candidate is the
  incident" is not answered by the pipeline.
- **Threshold justification.** G and A_min must come from the E005 validation split only (E007c). They must not come
  from Cerulean, Incident 001 agreement, E007 gaps or the AIS σ.
  - **Rejected (review 2026-09-28):** G = 5 km "because the AIS scoring kernel σ = 5 km". σ controls vessel-to-particle
    compatibility. It says nothing about whether two predicted mask fragments belong to one physical slick.

## Validation-derived policy (E007c; E005 validation split only)

Source: `scripts/e007c_validate_clustering.py` and `results/E007c_validation_clustering/`. The rules were predeclared
and committed (`c56fdde`) before any sweep result was computed.

**Data**
- Frozen E005 at 0.60 on 477 val tiles; val IoU reproduced exactly (0.7539).
- 2,297 predicted components: 1,279 tp, 283 ambiguous (overlap ≥ 2 GT), 735 fp.
- 4,790 GT components.
- 200 GT components are fragmented (overlapped by ≥ 2 predicted components).

**Intra-slick fragment gaps.** The merge height is the smallest G that reconnects all fragments of one GT component.
- n = 200: median 204 m, p75 416 m, p90 667 m, p95 1,244 m, p99 3,073 m, max 4,373 m.
- For comparison, the nearest other-GT predicted component from a TP component has median 425 m and p25 199 m.
  These distributions overlap heavily.

**G sweep** (single linkage, edge-to-edge; no A_min)

| G | 0 | 100 | 250 | 500 | 1000 | 2000 | 5000 m |
|---|---|---|---|---|---|---|---|
| r_area (fragment signal reconnected) | 0.889 | 0.908 | **0.950** | 0.978 | 0.989 | 0.991 | 1.000 |
| wrong-GT merge rate | 0 | **0.045** | 0.178 | 0.364 | 0.492 | 0.647 | 0.717 |
| TP+FP mixed rate | 0 | 0.016 | 0.042 | 0.080 | 0.124 | 0.223 | 0.371 |
| purity (area-weighted) | 1.000 | 0.991 | 0.954 | 0.925 | 0.904 | 0.873 | 0.850 |

- **Rule outcome:** no G in the grid satisfies r_area ≥ 0.95 AND wrong_rate ≤ 0.05.
  - Wrong-merge is ≤ 5% only at 100 m (95% CI 3.5–5.6%, so not robust even there), where r_area = 0.908.
  - r_area reaches 0.95 at 250 m, where the wrong-merge rate is 17.8%.
  - **No defensible single global G exists on this validation data.**
- **Limitation:** "wrong merge" counts distinct GT *connected components*. Trujillo masks are heavily fragmented
  (~23 GT components per oil tile), so two GT components may be one physical slick. The wrong-merge rate is therefore
  an upper bound for physical-slick mis-merges. The 8-connected GT sensitivity check changes it by < 1 pp, so this is
  not a diagonal-connectivity artefact. The limitation was not "fixed" by redefining GT after seeing results.

**A_min** (predeclared rule met): **A_min = 5,000 m²**.
- It removes 24.8% of fp components (182 of 735) and keeps 99.99% of TP predicted area.
- It also removes 119 small TP-associated components.
- Component precision is 0.34–0.51 in the bins below 10,000 m² and 0.885 at ≥ 100,000 m².
- This value is not frozen, because the policy bundle has no G.

**Orientation guard** (exploratory): not adopted.
- θ = 20° reduces wrong merges by ~12–19% relative at each G, at a cost of ≤ 1 pp r_area.
- It does not make any G meet both constraints.

## Descriptive Incident 001 observations (E007b; NOT used for any threshold)

- Nothing in this section was used to set G, A_min or the orientation rule.
- The Incident-area components are separated by gaps of about 1.06–1.13 km. They merge into one isolated
  (18.6 km), orientation-coherent (0.99) cluster only at G ≥ 2 km.
- Validation says that G ≥ 2 km merges distinct GT components in ≥ 65% of TP clusters.
- The scene also contains larger coherent candidates, the unverified northern linear features.

## Open decision (reviewer)

1. **No grouping.** Every predicted component with area ≥ 5,000 m² (the validation-derived A_min) is its own candidate.
   - Wrong merges are 0 by construction.
   - Validation cost: about 11% of fragmented-slick TP area (r_area 0.889 at G = 0) ends up in smaller sibling
     candidates.
   - Autonomous, but gives more physics runs.
2. **G = 100 m.** This is the only value within the wrong-merge limit (point estimate; the CI exceeds 5%). It needs an
   explicit, reviewer-approved relaxation of the 0.95 recovery target to about 0.91.
3. **A new predeclared validation definition of "distinct slick"** (E007d), addressing GT fragmentation. This carries a
   goal-seeking risk; it is acceptable only if the definition is fixed before any re-analysis.
4. **Mode A** (analyst-assisted, labelled non-blind).

## Allowed after selection (post-hoc)

- Comparing each selected or candidate geometry with the Cerulean polygon (IoU, coverage, centroid offset).
- Reporting which Mode B candidate contains the Cerulean slick.
- None of this may feed back into selection, thresholds or physics parameters.

## Recommendation (for review)

Option 1: Mode B with **no grouping** and A_min = 5,000 m², frozen as `e008_candidate_policy_v1`. It is the only
option fully supported by the validation evidence under the predeclared constraints. Each candidate is hindcast
separately. Incident 001 is identified only post hoc, as the candidate or candidates that intersect the Cerulean slick.

Mode A remains available as a labelled, non-blind case-study view.
