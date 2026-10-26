"""Sparse graph objects for message passing (plan.md 7.3.1).

A_in[j, i]  = w_in(i -> j)   row j averages over the neuron's INPUTS
A_out[i, j] = w_out(i -> j)  row i averages over the neuron's OUTPUTS
Never build per-edge message tensors: ~6 M edges x 256 dims would OOM. Use torch.sparse.mm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def sparse_adj(rows, cols, vals, n: int) -> torch.Tensor:
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n)).coalesce()


def build(edges: pd.DataFrame, n: int, undirected: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    pre = torch.from_numpy(edges.pre_uid.to_numpy(np.int64).copy())
    post = torch.from_numpy(edges.post_uid.to_numpy(np.int64).copy())
    w_in = torch.from_numpy(edges.w_in.to_numpy(np.float32).copy())
    w_out = torch.from_numpy(edges.w_out.to_numpy(np.float32).copy())
    A_in = sparse_adj(post, pre, w_in, n)
    A_out = sparse_adj(pre, post, w_out, n)
    if undirected:                       # ablation A6: one matrix, rows renormalised
        A = (A_in + A_out).coalesce()
        deg = torch.zeros(n).index_add_(0, A.indices()[0], A.values())
        A = torch.sparse_coo_tensor(A.indices(), A.values() / deg[A.indices()[0]].clamp_min(1e-12), (n, n)).coalesce()
        return A, A
    return A_in, A_out


def permute_edges(edges: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Control C4: degree-preserving permutation of the post column; recompute fractions."""
    from .harmonise import add_fractions
    e = edges[["pre_uid", "post_uid", "syn"]].copy()
    e["post_uid"] = e.post_uid.to_numpy()[rng.permutation(len(e))]
    e = e[e.pre_uid != e.post_uid]
    e = e.groupby(["pre_uid", "post_uid"], as_index=False).syn.sum()
    n = int(max(edges.pre_uid.max(), edges.post_uid.max())) + 1
    return add_fractions(e, n)


def load_graphs(layout, undirected: bool = False) -> dict:
    neurons = pd.read_parquet(layout.neurons, columns=["dataset"])
    edges = pd.read_parquet(layout.edges)
    out = {}
    for ds in ("mcns", "banc"):
        n = int((neurons.dataset == ds).sum())
        out[ds] = build(edges[edges.dataset == ds], n, undirected)
    return out
