"""Exact blocked cosine retrieval (numpy; the torch version for GPUs comes in Phase 2).

On a laptop a 512 x 158k float32 score block is ~320 MB.
"""
from __future__ import annotations

import numpy as np


def l2_normalise(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(n, 1e-12, None)


def ranks_of_true(Q: np.ndarray, G: np.ndarray, true_idx: np.ndarray, allowed=None,
                  block: int = 512, k: int = 10):
    """1-based rank of G[true_idx[i]] for query Q[i] among all gallery rows (ties count
    against the query: rank = 1 + #strictly-better). `allowed(rows) -> bool[b, ng]` applies
    hard filters (disallowed gallery rows are pushed to -inf; the true partner is always kept).
    Returns (ranks, topk_idx)."""
    Q, G = l2_normalise(Q), l2_normalise(G)
    k = min(k, len(G) - 1)
    ranks = np.empty(len(Q), np.int64)
    top = np.empty((len(Q), k), np.int64)
    for s in range(0, len(Q), block):
        S = Q[s:s + block] @ G.T
        rows = np.arange(s, min(s + block, len(Q)))
        if allowed is not None:
            m = allowed(rows)
            m[np.arange(len(rows)), true_idx[rows]] = True
            S[~m] = -np.inf
        st = S[np.arange(len(rows)), true_idx[rows]]
        ranks[rows] = 1 + (S > st[:, None]).sum(1)
        part = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
        top[rows] = np.take_along_axis(part, order, 1)
    return ranks, top


def recall_metrics(ranks: np.ndarray, ks=(1, 5, 10, 50, 100)) -> dict:
    r = np.asarray(ranks)
    out = {f"R@{k}": float((r <= k).mean()) for k in ks}
    out["MRR"] = float((1.0 / r).mean())
    out["median_rank"] = float(np.median(r))
    out["n"] = int(len(r))
    return out
