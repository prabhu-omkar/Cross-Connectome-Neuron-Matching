"""Re-create a trained run's embeddings from its folder (config.json + model.pt).

Embeddings are not stored in run folders (~80 MB each); a full-graph forward pass without
gradients takes seconds to a minute on the laptop CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Layout
from .models.encoder import Encoder
from .train import DEFAULTS, merge


def run_config(run_dir: Path) -> dict:
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    if "train_config" not in cfg:
        raise ValueError(f"{run_dir} is not a trained-model run (no train_config)")
    return merge(DEFAULTS, cfg["train_config"])


@torch.no_grad()
def embed_run(layout: Layout, run_dir: Path, device: str = "cpu", mirror: bool = False) -> dict:
    """mirror=True: every node's features are left<->right mirrored (feature_spec mirror_perm: _L/_R
    neuropil columns and the side one-hot swapped) before the forward pass; the graph is unchanged.
    Used by control C3 (scripts/c3_mirror.py)."""
    run_dir = Path(run_dir)
    cfg = run_config(run_dir)
    mc = cfg["model"]
    spec = json.loads(layout.feature_spec.read_text())
    keep = np.ones(len(spec["columns"]), bool)
    for b in cfg["features"]["drop_blocks"]:
        a, z = spec["blocks"][b]
        keep[a:z] = False
    model = Encoder(int(keep.sum()), mc["width"], mc["out_dim"], mc["layers"], mc["dropout"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device, weights_only=True))
    model.to(device)
    graphs = None
    if mc["layers"] > 0:
        from .graph import load_graphs
        graphs = load_graphs(layout, undirected=mc["direction"] == "undirected",
                             permute_seed=20261026 if cfg["controls"]["edge_permute"] else None)
    perm = np.asarray(spec["mirror_perm"]) if mirror else np.arange(len(spec["columns"]))
    Z = {}
    for ds in ("mcns", "banc"):
        x = torch.as_tensor(np.ascontiguousarray(np.load(layout.features(ds))[:, perm][:, keep]), device=device)
        A = tuple(a.to(device) for a in graphs[ds]) if graphs else (None, None)
        Z[ds] = model.embed(x, *A).float().cpu().numpy()
    return Z
