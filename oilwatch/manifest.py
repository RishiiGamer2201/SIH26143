"""Per-stage manifest: inputs/outputs with sha256, parameters, code version. Mirrors the FROZEN.json habit."""
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT


def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_state():
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


class Stage:
    """Context manager: `with Stage(run, "hindcast", params) as st: ...; st.output(path)`."""

    def __init__(self, run, name, params=None, inputs=()):
        self.run, self.name, self.params = run, name, params or {}
        self.dir = run.stage_dir(name)
        self.inputs = [Path(p) for p in inputs]
        self.outputs, self.notes = [], {}

    def output(self, p):
        self.outputs.append(Path(p))
        return Path(p)

    def __enter__(self):
        self.t = time.time()
        print(f"[{self.name}] start -> {rel(self.dir)}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        import numpy, pandas
        m = {"stage": self.name, "incident": self.run.incident.id, "run_id": self.run.run_id,
             "status": "failed" if exc_type else "ok", "error": repr(exc) if exc else None,
             "finished_utc": datetime.now(timezone.utc).isoformat(), "seconds": round(time.time() - self.t, 2),
             "params": self.params, "notes": self.notes,
             "inputs": {rel(p): sha256(p) for p in self.inputs if Path(p).is_file()},
             "outputs": {rel(p): sha256(p) for p in self.outputs if Path(p).is_file()},
             "git": git_state(), "env": {"python": platform.python_version(), "numpy": numpy.__version__,
                                         "pandas": pandas.__version__, "platform": platform.platform()}}
        try:
            import opendrift
            m["env"]["opendrift"] = opendrift.__version__
        except ImportError:
            pass
        (self.dir / "manifest.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
        print(f"[{self.name}] {m['status']} in {m['seconds']} s", flush=True)
        return False
