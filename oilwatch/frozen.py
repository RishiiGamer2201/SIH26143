"""Import the frozen, importable experiment modules in scripts/ (read-only reuse, never edited).

The root physics scripts run at import time, so they are re-implemented in drift.py / scoring.py instead and
checked against their frozen outputs by the parity stage.
"""
import importlib.util
import sys

from .config import ROOT

_SCRIPTS = ROOT / "scripts"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))  # source_type_v1 imports score_ais_density_v2_1 by bare name
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


v2_1 = _load("score_ais_density_v2_1")          # E101: vessel_positions, age_bin_table
source_type = _load("source_type_v1")           # E102: track_metrics, classify, _selftest
