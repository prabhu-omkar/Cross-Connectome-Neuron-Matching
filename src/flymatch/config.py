"""Config loading and path resolution. All paths in configs are relative to the repo root."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_paths() -> dict:
    p = REPO / "configs" / "paths.yaml"
    if not p.exists():
        p = REPO / "configs" / "paths.example.yaml"
    raw = _load_yaml(p)
    return {k: (REPO / v).resolve() for k, v in raw.items()}


def load_data_config(path: str | Path | None = None) -> dict:
    cfg = _load_yaml(Path(path) if path else REPO / "configs" / "data.yaml")
    cfg["out_dir"] = (REPO / cfg.get("out_dir", "data")).resolve()
    return cfg


class Layout:
    """Where every artefact of the data pipeline lives (plan.md Section 5.3)."""

    def __init__(self, cfg: dict):
        self.root = Path(cfg["out_dir"])
        self.interim = self.root / "interim"
        self.processed = self.root / "processed"
        self.splits = REPO / "data" / "splits"          # split files are shared across data variants
        self.results = REPO / "results"
        self.maps = REPO / "maps"
        for d in (self.interim, self.processed, self.splits, self.results):
            d.mkdir(parents=True, exist_ok=True)

    neurons_raw = property(lambda s: s.interim / "neurons_raw.parquet")
    edges_raw = property(lambda s: s.interim / "edges_raw.parquet")
    neurons = property(lambda s: s.interim / "neurons.parquet")
    edges_np = property(lambda s: s.interim / "edges_np.parquet")
    edges = property(lambda s: s.interim / "edges.parquet")
    gt_types = property(lambda s: s.interim / "gt_types.parquet")
    gt_pairs = property(lambda s: s.interim / "gt_pairs_1to1.parquet")
    families = property(lambda s: s.interim / "families.parquet")
    feature_spec = property(lambda s: s.processed / "feature_spec.json")
    split_file = property(lambda s: s.splits / "splits_v1.json")

    def features(self, ds: str) -> Path:
        return self.processed / f"features_{ds}.npy"

    def log(self, stage: str) -> Path:
        return self.interim / f"log_{stage}.json"


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=_json_default)


def _json_default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
