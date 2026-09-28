"""pytest -q tests/  (parity test ~45 s; set OILWATCH_SKIP_PARITY=1 to skip it)."""
import os

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from oilwatch import age, forward, geo
from oilwatch.evidence import gap_overlaps, valid_mmsi


def test_angle_axis():
    assert geo.angle_diff_axis(90, 90) == 0
    assert geo.angle_diff_axis(270, 90) == 0          # heading opposite along the same axis
    assert geo.angle_diff_axis(0, 90) == 90
    assert geo.angle_diff_axis(135, 90) == 45


def test_axis_endpoints_bearing():
    rect = box(0, 0, 10000, 200)                        # 10 km east-west strip (metric)
    p1, p2, length, bearing, short = geo.axis_endpoints(rect)
    assert abs(length - 10000) < 1 and abs(bearing - 90) < 1e-6 and abs(short - 200) < 1e-6


def test_fss_perfect_and_disjoint():
    obs = np.zeros((50, 50), bool); obs[20:30, 20:30] = True
    assert forward.fss(obs, obs, 5) == pytest.approx(1.0)
    far = np.zeros_like(obs); far[0:5, 0:5] = True
    assert forward.fss(obs, far, 3) < 0.05


def test_windows_cover_48h():
    w = forward.windows([1, 12])
    assert (0, 1) in w and (47, 1) in w and (36, 12) in w and (37, 12) not in w


def test_age_cues_monotone():
    assert age.width_age_h(400, 50, 1) > age.width_age_h(400, 50, 10) > 0
    assert age.width_age_h(40, 50, 1) == 0.0
    assert 3 < age.okubo_age_h(377) < 7                 # ~4.8 h for the E007 north strand width


def test_behaviour_helpers():
    assert valid_mmsi("367655260") and not valid_mmsi("11") and not valid_mmsi("123456789")
    assert gap_overlaps("10.0-6.0h", 5, 8) and not gap_overlaps("10.0-6.0h", 0, 5)


@pytest.mark.skipif(os.environ.get("OILWATCH_SKIP_PARITY") == "1", reason="parity skipped")
def test_scoring_parity(tmp_path):
    from oilwatch.config import Incident, Run, load_params
    from oilwatch.parity import run_scoring_parity
    inc, P = Incident.load("incident_001"), load_params()
    res = run_scoring_parity(Run(inc, P, run_id="pytest", root=tmp_path), inc, P)
    assert all(r["ulp_equal"] for r in res)


@pytest.mark.skipif(os.environ.get("OILWATCH_SKIP_PARITY") == "1", reason="needs forcing files (slow)")
def test_simulate_keeps_element_identity():
    """Regression: OpenDrift 1.14 returns backward runs in reversed element order; simulate() must undo it."""
    import pandas as pd
    from oilwatch import drift
    from oilwatch.config import Incident, load_params
    inc, P = Incident.load("incident_001"), load_params()["physics_v3"]
    rdr = drift.readers(inc.input("ocean_nc"), inc.input("wind_nc"))
    el = pd.DataFrame({"lon": [-90.42, -89.10, -90.15], "lat": [27.126, 28.20, 27.42], "time": [inc.t0] * 3, "windage": [.02] * 3})
    sim = dict(sim_id=1, horizontal_diffusivity=1.0, current_uncertainty=0.0)
    lon, lat, t = drift.simulate(el, 1, True, rdr, sim, P, seed=1)       # backward: seed time is the LAST column
    assert np.allclose(lon[:, -1], el.lon) and np.allclose(lat[:, -1], el.lat)
    el2 = el.assign(time=[inc.t0 - pd.Timedelta(hours=h) for h in (3, 2, 1)])
    lon, lat, t = drift.simulate(el2, 3, False, rdr, sim, P, seed=1)     # forward time series
    for i, h in enumerate((3, 2, 1)):
        k = list(t).index(inc.t0 - pd.Timedelta(hours=h))
        assert np.isclose(lon[i, k], el2.lon[i]) and np.isclose(lat[i, k], el2.lat[i])
