"""E008c recipe check (not part of the candidate pipeline): run the E008c ensemble functions with the HISTORICAL inputs of
run_hindcast_ensemble.py (reference polygon, forcing v1) in memory and compare with the frozen
results/incident_001/hindcast_ensemble_{particles.parquet,scenarios.csv}. Nothing is written except the JSON report.
The reference polygon is used only to reproduce the historical baseline; it never selects, merges or ranks candidates.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e008c_candidate_ensemble as E  # noqa: E402
from opendrift.readers import reader_netCDF_CF_generic  # noqa: E402

ROOT = E.ROOT
V1_OCEAN = ROOT / "data/incident_001/currents/ocean_opendrift.nc"
V1_WIND = ROOT / "data/incident_001/wind/era5_wind_10m_151h.nc"
REF = ROOT / "data/incident_001/metadata/slick_3775938.geojson"
HIST = ROOT / "results/incident_001"

before = E.A.frozen_state()
geom = E.candidate_geometry(shape(json.loads(REF.read_text())["features"][0]["geometry"]))
ocean_reader = reader_netCDF_CF_generic.Reader(str(V1_OCEAN))
wind_reader = reader_netCDF_CF_generic.Reader(str(V1_WIND))
frames, recs = [], []
for scenario_id, (windage, current_unc, wind_unc) in enumerate(E.SCENARIOS, start=1):
    o, result, seed, _, _ = E.run_scenario(ocean_reader, wind_reader, geom, scenario_id, windage, current_unc, wind_unc)
    df, ntraj, ntime = E.scenario_frame(result, scenario_id, windage, current_unc, wind_unc, "reference")
    frames.append(df.drop(columns=["candidate_id", "particle_id", "age_hours"]))
    recs.append(dict(scenario_id=scenario_id, seed=seed, windage=windage, current_uncertainty=current_unc,
                     wind_uncertainty=wind_unc, particles=ntraj, times=ntime, valid_rows=len(df)))
new = pd.concat(frames, ignore_index=True)
old = pd.read_parquet(HIST / "hindcast_ensemble_particles.parquet")
old_s = pd.read_csv(HIST / "hindcast_ensemble_scenarios.csv")
new_s = pd.DataFrame(recs)
diff = (new[["lon", "lat"]] - old[["lon", "lat"]]).abs().max() if len(new) == len(old) else None
rep = dict(particles_identical=bool(new.equals(old)), dtypes_identical=bool((new.dtypes == old.dtypes).all()),
           rows=[len(old), len(new)], max_abs_diff_lon_lat=None if diff is None else diff.to_dict(),
           scenarios_identical=bool(new_s.equals(old_s)), frozen_unchanged=before == E.A.frozen_state(),
           e008c_script_sha256=E.A.sha256(E.__file__), check_script_sha256=E.A.sha256(__file__))
E.OUT.mkdir(parents=True, exist_ok=True)
E.A.dump(rep, E.OUT / "baseline_reproduction_check.json")
print(json.dumps(rep, indent=1, default=str))
