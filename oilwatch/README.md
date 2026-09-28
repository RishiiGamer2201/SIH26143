# oilwatch: automated oil-spill attribution pipeline (SIH 2026 · PS 26143)

This package wraps the frozen experiments (E000–E102) into one reproducible pipeline and adds the v3 attribution method (E008).
Frozen files are never edited or written; every run lives in `runs/<incident>/<run_id>/<stage>/` and each stage has its own `manifest.json`.

## Run

```bash
python -m venv --system-site-packages .venv && .venv/Scripts/python -m pip install opendrift netCDF4   # Windows
.venv/Scripts/python -m oilwatch parity incident_001 [--full]   # reproduce frozen outputs (the gate)
.venv/Scripts/python -m oilwatch run incident_001 [--run-id X] [--stages seed,backward,...] [--skip-parity]
.venv/Scripts/python -m pytest -q tests/
```

- A new incident needs a new `configs/incidents/<id>.json`.
- All method parameters, with their literature sources, are in `configs/pipeline_v1.json`.

## Stages

| stage | what | key outputs |
|---|---|---|
| parity | Recomputes the frozen V2, release-age, ensemble robustness, V2.1 and E102 tables. With `--full` it also re-runs OpenDrift. Must match the frozen files. | `parity_report.json` |
| seed | E008 policy: merges E007 fragments into slick events, scores wind validity and potential, checks eligibility, selects events (`oilwatch/OW_SEED_POLICY.md`; not the frozen E008 policy) | `candidates.geojson/csv` |
| context | Wind, current and Stokes at each event at T0, plus 48 h history (external factors) | `context_t0.csv`, `context_history.parquet` |
| backward | v3 ensemble: all events 48 h back; age-dependent kernel; filtering funnel | `backward_robust.csv`, `funnel.csv`, `cloud_summary.parquet` |
| forward | Every funnel vessel "discharges" along its AIS track; virtual oil drifted to T0 and compared with each slick | `forward_best.csv`, `forward_windows.parquet` |
| evidence | Fresh-discharge geometry, AIS behaviour anomalies, motion_state (frozen E102 rules) | `geometry.csv`, `behaviour.csv`, `motion.csv` |
| fuse | Forward–backward fusion, tiers, flags, age cues and interval | `suspects.csv`, `age_summary.csv`, `age_cues.csv` |
| report | Summary JSON, Markdown report, map | `summary.json`, `report.md`, `map.png` |
| posthoc | Reference (Cerulean) comparison, read only here and after everything else is written | `posthoc.json` |

## Method (v3) and sources

- **Physics**:
  - OpenDrift OceanDrift, transport only; weathering is irreversible (Breivik et al. 2012).
  - Forcing: SMOC `uo+utide` with separate Stokes drift, so windage is {1, 2, 3}% (0.02 + Stokes ≈ 3.5% total; Röhrs et al. 2018).
  - Horizontal diffusivity {1, 3, 10} m²/s (Okubo 1971; Accarino et al. 2025).
  - Current uncertainty {0, 0.05, 0.10} m/s.
  - 27 members in total, run as 9 simulations with windage per element.
- **Backward score**:
  - density(τ) = mean over particles of exp(−d²/2σ(τ)²), with σ(τ)² = 1 km² + (0.05 m/s · τ)².
  - AIS positions follow the frozen V2.1 interpolation rule.
  - S_weighted uses the persistence prior exp(−τ/12 h).
  - Absolute floor: peak density ≥ exp(−2).
- **Forward score** (Longépé et al. 2015):
  - Virtual discharge every 15 min along each track.
  - The window search covers every 1/2/3/6/12 h release window: precision/recall/F within 1 km, with the 27 members pooled.
  - Skill is the Fractions Skill Score at 0.5–5 km; "useful" when FSS ≥ 0.5 + f_O/2 (Simecek-Beatty & Lehr 2021).
- **Fusion**:
  - combined = √(backward_rel · forward_rel) (Luo et al. 2024).
  - Tiers are pre-registered with absolute criteria.
  - Flags (fresh-discharge geometry, AIS gap in the release window, identity anomalies, 48 h boundary) never change the score.
- **Age**:
  - Interval built from the forward best window, width diffusion, Okubo, and the 2–12 h persistence prior.
  - Automatic conflict note when the drift-based age exceeds what the slick's appearance allows.
  - SAR darkness (damping ratio, edge contrast) needs the σ0 raster, which is git-ignored here.
  - The optical Bonn code table is included for use when optical imagery exists.

Rankings are evidence of spatio-temporal consistency, never proof of culpability.
