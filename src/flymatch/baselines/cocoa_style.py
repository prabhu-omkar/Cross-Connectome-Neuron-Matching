"""B3 — cocoa-style label-projected connectivity (plan.md 7.2.3), re-implemented because cocoa
does not support BANC.

Each neuron becomes the fraction of its input synapses from / output synapses to each SEED type.
Seed types are training-split types only (the shared coordinate system of both datasets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .scorers import EmbeddingScorer


def label_projected(edges: pd.DataFrame, n_nodes: int, seed_type_of_uid: np.ndarray, n_types: int) -> sp.csr_matrix:
    """edges: pre_uid, post_uid, syn (one dataset). seed_type_of_uid: [n_nodes] index or -1.
    Divides by the neuron's TOTAL synapses: untyped partners are missing mass, which is
    exactly the degradation the seed ablation measures."""
    W = sp.csr_matrix((edges.syn.to_numpy(np.float64), (edges.pre_uid.to_numpy(), edges.post_uid.to_numpy())),
                      shape=(n_nodes, n_nodes))
    lab = seed_type_of_uid >= 0
    Y = sp.csr_matrix((np.ones(lab.sum()), (np.where(lab)[0], seed_type_of_uid[lab])), shape=(n_nodes, n_types))
    out_to_type = W @ Y
    in_from_type = W.T @ Y
    out_tot = np.asarray(W.sum(axis=1)).ravel()
    in_tot = np.asarray(W.sum(axis=0)).ravel()
    scale = lambda A, tot: sp.diags(1 / np.maximum(tot, 1)) @ A
    return sp.hstack([scale(in_from_type, in_tot), scale(out_to_type, out_tot)]).tocsr().astype(np.float32)


def build_scorer(neurons: pd.DataFrame, edges: pd.DataFrame, seed_types: list[str]) -> EmbeddingScorer:
    tidx = {t: i for i, t in enumerate(seed_types)}
    Z = {}
    for ds in ("mcns", "banc"):
        n = neurons[neurons.dataset == ds].sort_values("uid")
        st = n.type_h.map(tidx).fillna(-1).astype(int).to_numpy()
        e = edges[edges.dataset == ds]
        Z[ds] = label_projected(e, len(n), st, len(seed_types))
    return EmbeddingScorer(Z)
