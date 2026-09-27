# E008 source-geometry selection policy — FROZEN v1 (2026-09-28)

> **Status: FROZEN v1.** Reviewer decision 2026-09-28: Option 1 was approved (no grouping; A_min = 5,000 m²; no
> orientation guard).
> - Config: `configs/e008_candidate_policy_v1.json`, sha256 `3fb833df7268647641f8b9f1696527c10b067607e3fc9ce11363a4cce0f5bb27`
>   (`.sha256` file alongside).
> - The config JSON is never edited. Any change becomes `e008_candidate_policy_v2`.
> - Applied once, by `python scripts/e008_candidate_policy_v1.py apply`, to `results/E008_candidate_policy_v1/`.

E008 seeds backward hindcasts with **our** predicted slick geometry, using the frozen E007 components
(`results/E007_incident001_ml/incident001_predicted_components.geojson`: 49 components, 4-connected, frozen E005 at
threshold 0.60, geodesic areas).

## Primary pipeline (autonomous multi-candidate)

1. Start from the frozen E007 4-connected components. `FROZEN.json` is hash-verified before use.
2. **Remove** a component if `area_m2 < 5,000 m²`.
3. Otherwise, the component is **one independent candidate**, with `candidate_id = "E007-C{component_id}"`.
4. **There is no image-space grouping of any kind** (`grouping.mode = "none"`).
   - This is an explicit mode. It is not single-linkage with G = 0, because diagonally touching 4-connected
     components have polygon distance 0 and would merge.
5. Candidates keep their original component IDs and are ordered by component ID. There is no ranking by area,
   probability or anything else.

**A fragmented physical slick may produce several candidates. This is intentional.**
- On validation, about 11% of fragmented-slick overlap area lies outside the dominant component
  (r_area 0.889 at no grouping).
- That signal is not deleted. It remains as separate candidates.

**Later candidate fusion** (for example, combining candidates whose hindcasts converge) needs an **independent,
predeclared downstream rule**. It cannot be added retrospectively to v1.

**Per-candidate processing.** Each candidate is hindcast and scored independently with the frozen physics and frozen
scoring. Each candidate gets its own output directory. There is no cross-candidate ranking. "Which candidate is the
incident" is not answered by the pipeline.

## Validation basis (E007c; E005 validation split only)

The rules were predeclared in commit `c56fdde636858ce2770d15f55ce710f06ff124dc`. Results are in commit
`cf51f5bd21f22b3e41d9af676e0c9598f9aea360` and `results/E007c_validation_clustering/`.

**Data.** 477 val tiles; 2,297 predicted components; 200 fragmented GT components.

**Grouping.** The predeclared requirement was r_area ≥ 0.95 **and** wrong-GT merge ≤ 0.05. No tested global gap met it:

| G | r_area | wrong-merge rate |
|---|---|---|
| 100 m | 0.908 | 0.045 (95% CI 0.035–0.056) |
| 250 m | 0.950 | 0.178 |
| 1000 m | 0.989 | 0.492 |
| 5000 m | 1.000 | 0.717 |

- No-grouping baseline: r_area 0.889 (CI 0.827–0.938).
- The ground truth was not redefined, and the recovery requirement was not relaxed after the results were seen.
- Wrong merges are counted against GT connected components. Trujillo GT is fragmented, so this rate is an upper
  bound.

**Area filter.** The rule was predefined: the smallest A_min with fp reduction ≥ 0.20 and TP area retained ≥ 0.99.
- A_min = 5,000 m² removes 24.8% of validation fp components (182 of 735) and retains 99.99% of TP area
  (0.999900796270267).
- It also removes 119 small TP-associated components. This was accepted because the rule was predefined.

**Orientation guard.** Not adopted. It did not make any G meet both constraints.

**Rejected.** G = 5 km "because the AIS scoring kernel σ = 5 km". σ is vessel-to-particle compatibility. It says
nothing about whether two mask fragments are one slick.

**Recorded numerical note (v1 JSON).** 14 descriptive validation floats in the v1 JSON were read with pandas'
default float parser. They differ from the E007c CSV text by ≤ 1.1e-16 (≤ 1 ulp); for example, fp fraction
0.2476190476190476 against 182/735 = 0.24761904761904763.
- The operative values (A_min 5000, grouping none) and all counts are exact.
- The E007c CSVs, whose sha256 are in the JSON, are authoritative.
- Not corrected, because v1 is hashed.

## Application result (once, 2026-09-28)

- 49 E007 components → **42 candidates** retained, **7 excluded** (components 11, 13, 15, 20, 22, 32, 42; all
  < 5,000 m²).
- Retained area: 25,858,333 m² (99.926% of the 25,877,401 m² predicted scene area).
- Excluded area: 19,068 m².
- Files:
  - `candidate_components.csv`
  - `candidate_components.geojson`
  - `excluded_components.csv`
  - `application_report.json` (holds the output sha256)
- No candidate is "the Incident 001 slick".

## Forbidden selectors (never choose, crop, merge, rank or filter candidates)

- The Cerulean polygon, its centroid, bbox, intersection, IoU or coverage.
- The E007 reporting ROI and the `in_roi` flag. The flag is dropped on load.
- "The 13 components inside the ROI", the closest component, or the best-IoU component.
- `incident.json` geometry fields. They are all Cerulean-derived.
- E007 component gaps.
- AIS candidates and the AIS σ.
- E004 gating. E004 must not filter E005 components.

## Post hoc only (after candidate files were written and hashed)

**File.** `posthoc_cerulean_candidate_mapping.json` lists **every** candidate intersecting the Cerulean reference, in
component-ID order, with no winner. It includes intersection area, candidate area, fraction of candidate inside the
reference, fraction of the reference covered, IoU and centroid distance.
- 8 of 42 candidates intersect: E007-C30, C33, C34, C35, C36, C37, C38, C40.
- Together they cover 80.5% of the reference.
- None of the 7 excluded components intersects the reference.

This mapping never feeds back into the policy, the candidate list, its order, eligibility, thresholds or physics
parameters.

## Optional: Mode A — analyst-assisted / non-blind demonstration mode (NOT the primary pipeline)

- **What the analyst sees:** calibrated VV, the E005 probability map and the candidate IDs. They do not see the
  Cerulean overlay, the ROI, `incident.json` geometry or the E007 comparison figure.
- **What the analyst does:** picks candidate IDs as a case-study seed.
- **What is recorded before any run:** the IDs, the rationale, the cue used, the analyst's identity and date, and
  prior exposure to Cerulean / E007 post-hoc material.
- **Blinding.** Everyone involved so far has seen the Cerulean overlay, so a selection by any of us is **non-blind**
  and must be labelled that way.
- **Labelling.** Every output from this mode is labelled "analyst-assisted, not fully autonomous".
- **Ambiguity.** The scene has several visually slick-like groups: the northern linear features near −89.1°, 28.2°N;
  the group near −90.44°, 27.12°N; and a feature near −90.22°, 27.39°N.

## Descriptive Incident 001 observations (E007b; not used for any threshold)

- The Incident-area components are separated by gaps of about 1.06–1.13 km.
- They form one isolated, coherent cluster only at link gaps ≥ 2 km. Validation shows such gaps merge distinct GT
  components in ≥ 65% of TP clusters.
- The scene also contains larger coherent unverified candidates to the north.
