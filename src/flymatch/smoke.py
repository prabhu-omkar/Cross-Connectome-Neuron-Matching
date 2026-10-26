"""Week-6 check: one full-graph forward + backward of the GNN encoder on both connectomes,
with timing and peak memory. `python -m flymatch.smoke [--layers 2] [--width 256]`."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .config import Layout, load_data_config, write_json
from .graph import load_graphs
from .models.encoder import Encoder


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--width", type=int, default=256)
    a = ap.parse_args(argv)
    layout = Layout(load_data_config())
    t0 = time.time()
    G = load_graphs(layout)
    t_graph = time.time() - t0
    X = {ds: torch.as_tensor(np.load(layout.features(ds))) for ds in ("mcns", "banc")}
    enc = Encoder(X["mcns"].shape[1], d=a.width, out=128, layers=a.layers)
    t1 = time.time()
    z = {ds: enc(X[ds], *G[ds]) for ds in X}
    loss = sum(v.pow(2).sum() for v in z.values()) * 0 + sum(v[:100].sum() for v in z.values())
    t_fwd = time.time() - t1
    loss.backward()
    t_all = time.time() - t1
    try:
        import psutil
        rss = psutil.Process().memory_info().peak_wset / 2**30
    except Exception:
        rss = float("nan")
    out = {"layers": a.layers, "width": a.width, "nnz": {ds: int(G[ds][0]._nnz()) for ds in G},
           "graph_build_s": round(t_graph, 1), "forward_s": round(t_fwd, 1), "forward_backward_s": round(t_all, 1),
           "peak_rss_gib": round(rss, 2), "grads_ok": all(p.grad is not None for p in enc.parameters())}
    print(json.dumps(out, indent=1))
    write_json(out, layout.results / f"gnn_smoke_cpu_L{a.layers}_w{a.width}.json")


if __name__ == "__main__":
    main()
