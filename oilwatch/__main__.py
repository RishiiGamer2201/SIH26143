"""CLI.

  python -m oilwatch parity incident_001 [--full]      reproduce frozen outputs (gate for everything else)
  python -m oilwatch run incident_001 [--run-id ID]     full v3 chain: seed -> context -> backward -> forward -> fuse -> report
"""
import argparse
import sys

from .config import Incident, Run, load_params


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oilwatch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("parity", "run", "twin"):
        p = sub.add_parser(name)
        p.add_argument("incident")
        p.add_argument("--params", default=None)
        p.add_argument("--run-id", default=None)
    sub.choices["parity"].add_argument("--full", action="store_true", help="also re-run OpenDrift (~5 min)")
    sub.choices["run"].add_argument("--skip-parity", action="store_true")
    sub.choices["twin"].add_argument("--n", type=int, default=24, help="number of synthetic culprit cases")
    sub.choices["run"].add_argument("--stages", default=None, help="comma list to run a subset (default: all)")
    a = ap.parse_args(argv)

    inc, P = Incident.load(a.incident), load_params(a.params)
    run = Run(inc, P, **({"run_id": a.run_id} if a.run_id else {}))
    print(f"run: {run.root}")
    if a.cmd == "parity":
        from .parity import run_scoring_parity, run_drift_parity
        run_scoring_parity(run, inc, P)
        if a.full:
            run_drift_parity(run, inc, P)
        return 0
    if a.cmd == "twin":
        from .twin import run as run_twin
        run_twin(run, inc, P, a.n)
        return 0
    from .pipeline import run_all
    run_all(run, inc, P, skip_parity=a.skip_parity, stages=a.stages.split(",") if a.stages else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
