"""Incident spec + pipeline parameters + run context (no hard-coded machine paths)."""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("OILWATCH_ROOT", Path(__file__).resolve().parents[1]))
FROZEN_DIRS = ("results",)  # never written by the pipeline


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


@dataclass
class Incident:
    spec: dict
    path: Path

    @classmethod
    def load(cls, name_or_path):
        p = Path(name_or_path)
        if not p.suffix:
            p = ROOT / "configs/incidents" / f"{name_or_path}.json"
        return cls(load_json(p), p)

    @property
    def id(self):
        return self.spec["incident_id"]

    @property
    def t0(self):
        return pd.Timestamp(self.spec["t0_utc"])

    def input(self, key):
        return ROOT / self.spec["inputs"][key]

    def frozen(self, key):
        return ROOT / self.spec["frozen_baseline"][key]

    def ais_attribution(self):
        """AIS for the new stages: scene-wide extract if present, else the frozen ROI extract (parity always uses the ROI)."""
        k = self.spec["inputs"].get("ais_scene_parquet")
        return ROOT / k if k and (ROOT / k).exists() else self.input("ais_parquet")

    @property
    def reference_mmsi(self):
        return set(self.spec.get("reference", {}).get("vessel_mmsi", []))


@dataclass
class Run:
    """One pipeline run: runs/<incident>/<run_id>/. Every stage gets its own sub-folder."""
    incident: Incident
    params: dict
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    root: Path = None

    def __post_init__(self):
        self.root = self.root or ROOT / "runs" / self.incident.id / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, name):
        d = (self.root / name).resolve()
        for fz in FROZEN_DIRS:
            assert not str(d).startswith(str((ROOT / fz).resolve())), f"refusing to write into frozen {fz}/"
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_params(path=None):
    return load_json(path or ROOT / "configs/pipeline_v1.json")
