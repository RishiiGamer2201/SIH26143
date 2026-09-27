# E008 source-geometry selection policy — DRAFT (2026-09-28)

> **DRAFT until E007c completes.** G and A_min are NOT approved. They will be derived from the E005 validation split only
> (E007c) and frozen in `configs/e008_candidate_policy_v1.json`.

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
- **Threshold justification.** G and A_min must be fixed and justified before any E008 run, without Cerulean and
  without Incident 001 agreement. Admissible justifications:
  - **Rejected (review 2026-09-28):** G = 5 km "because the AIS scoring kernel σ = 5 km". σ controls vessel-to-particle
    compatibility. It says nothing about whether two predicted mask fragments belong to one physical slick.
  - **G (data):** the distribution of gaps between predicted fragments that belong to the same ground-truth object
    in the Trujillo **validation** split (E005 val predictions). This needs a small new analysis, and the official
    test set must not be used.
  - **A_min (data):** from E005 validation component-level precision by size bin. That means the smallest area above
    which predicted components are mostly true positives on val. Alternatively, A_min = 0 (keep every component).
- **Disclosure: the thresholds are not blind to Incident 001 structure.** E007b already showed the following:
  - The Incident-area fragments are separated by gaps of about 1.06–1.13 km.
  - They merge into one isolated cluster only at G ≥ 2 km.
  - G = 5 km gives 10 candidates.

  Any G chosen now was chosen with that knowledge. It is not Cerulean-tuned, but it is incident-aware. The
  justification must stand on the physical or validation argument alone, and the reasoning must be recorded.
- **Cost.** At G = 5 km with A_min = 0 there are 10 candidates. That is 10 deterministic runs and 10 × 27 ensemble
  scenarios, against 1 + 27 for Incident 001 alone.

## Allowed after selection (post-hoc)

- Comparing each selected or candidate geometry with the Cerulean polygon (IoU, coverage, centroid offset).
- Reporting which Mode B candidate contains the Cerulean slick.
- None of this may feed back into selection, thresholds or physics parameters.

## Recommendation (for review)

Mode B, with G and A_min derived from the E005 validation split only (E007c), not from the AIS σ and not from Incident 001.
Incident 001 is then reported, post hoc, as "the candidate that contains the Cerulean slick".

Mode A remains useful as a labelled, non-blind case-study view, but it cannot be blinded with the current team.
