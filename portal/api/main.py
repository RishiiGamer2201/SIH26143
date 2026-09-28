"""OilWatch portal API (FastAPI). Serves the built web app and exported data; lists runs; triggers pipeline runs.

  cd SIH26143 && .venv/Scripts/python -m uvicorn portal.api.main:app --host 127.0.0.1 --port 8000

Local analyst tool: binds to localhost; incident and run ids are validated against the repo (no arbitrary paths/commands).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "portal/web"
RUN_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
app = FastAPI(title="OilWatch portal API", version="0.1.0")
JOBS: dict[str, dict] = {}


def incidents():
    return sorted(p.stem for p in (ROOT / "configs/incidents").glob("*.json"))


def check(incident, run_id=None):
    if incident not in incidents():
        raise HTTPException(404, f"unknown incident {incident!r}")
    if run_id is not None and not RUN_ID.match(run_id):
        raise HTTPException(400, "run id must match [A-Za-z0-9_-]{1,64}")


@app.get("/api/incidents")
def list_incidents():
    return incidents()


@app.get("/api/incidents/{incident}/runs")
def list_runs(incident: str):
    check(incident)
    out = []
    for d in sorted((ROOT / "runs" / incident).glob("*/")):
        stages = {}
        for m in d.glob("*/manifest.json"):
            j = json.loads(m.read_text())
            stages[j["stage"]] = {"status": j["status"], "seconds": j["seconds"], "finished_utc": j["finished_utc"]}
        out.append({"run_id": d.name, "stages": stages})
    return out


def _job(key, cmd):
    JOBS[key] = {"status": "running", "cmd": cmd[2:]}
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    JOBS[key].update(status="ok" if p.returncode == 0 else "failed", returncode=p.returncode, log_tail=(p.stdout + p.stderr)[-4000:])


@app.post("/api/incidents/{incident}/runs/{run_id}")
def start_run(incident: str, run_id: str, bg: BackgroundTasks, stages: str | None = None):
    """Run the oilwatch pipeline (optionally a comma-separated stage subset) in the background."""
    check(incident, run_id)
    if stages and not re.match(r"^[a-z_,]+$", stages):
        raise HTTPException(400, "bad stages")
    key = f"{incident}/{run_id}"
    if JOBS.get(key, {}).get("status") == "running":
        raise HTTPException(409, "already running")
    cmd = [sys.executable, "-m", "oilwatch", "run", incident, "--run-id", run_id] + (["--stages", stages] if stages else [])
    bg.add_task(_job, key, cmd)
    return {"job": key, "status": "started"}


@app.post("/api/incidents/{incident}/runs/{run_id}/export")
def export_run(incident: str, run_id: str, bg: BackgroundTasks):
    """Re-export a finished run into the portal's JSON data."""
    check(incident, run_id)
    if not (ROOT / "runs" / incident / run_id / "fuse/suspects.csv").exists():
        raise HTTPException(409, "run has no fuse output yet")
    key = f"{incident}/{run_id}/export"
    bg.add_task(_job, key, [sys.executable, str(ROOT / "portal/build_data.py"), incident, run_id])
    return {"job": key, "status": "started"}


@app.get("/api/jobs")
def jobs():
    return JOBS


# exported data + built web app (npm run build -> portal/web/dist)
app.mount("/data", StaticFiles(directory=WEB / "public/data"), name="data")
if (WEB / "dist").exists():
    app.mount("/assets", StaticFiles(directory=WEB / "dist/assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(WEB / "dist/index.html")
