"""Scorers: objects with ``score(qd, q_uids, gd, g_uids) -> float32 [nq, ng]`` (see evaluate.py)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..evaluate import MISSING


def l2_rows(X):
    if sp.issparse(X):
        n = np.sqrt(np.asarray(X.multiply(X).sum(1))).ravel()
        return sp.diags(1 / np.clip(n, 1e-12, None)) @ X
    X = np.asarray(X, dtype=np.float32)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


class EmbeddingScorer:
    """Cosine similarity of per-dataset embeddings / feature vectors (dense or sparse)."""

    def __init__(self, Z: dict):
        self.Z = {ds: l2_rows(z) for ds, z in Z.items()}
        self._gT = {}

    def score(self, qd, q, gd, g):
        key = (gd, len(g), int(g[0]), int(g[-1]))
        if key not in self._gT:
            G = self.Z[gd][g]
            self._gT = {key: G.T.tocsr() if sp.issparse(G) else np.ascontiguousarray(G.T)}
        S = self.Z[qd][q] @ self._gT[key]
        return (S.toarray() if sp.issparse(S) else S).astype(np.float32)


class RandomScorer:
    """B0: uniform random scores (meaningful only together with the hard filters)."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def score(self, qd, q, gd, g):
        return self.rng.random((len(q), len(g)), dtype=np.float32)


class PairTableScorer:
    """Scores from a sparse MaleCNS x BANC table (e.g. NBLAST). Absent entries -> MISSING.
    Stored values are shifted by +OFFSET so that a stored 0 differs from 'absent'."""

    OFFSET = 10.0

    def __init__(self, M_shifted: sp.csr_matrix):
        self.M = {"mcns": M_shifted.tocsr(), "banc": M_shifted.T.tocsr()}

    def score(self, qd, q, gd, g):
        S = self.M[qd][q][:, g].toarray().astype(np.float32)
        miss = S == 0
        S -= self.OFFSET
        S[miss] = MISSING
        return S


class ZSumScorer:
    """Combine scorers by summing per-query z-scores (over each row's real scores).
    MISSING entries of a component take that row's minimum z of the component."""

    def __init__(self, scorers: list, weights=None):
        self.scorers = scorers
        self.w = weights or [1.0] * len(scorers)

    def score(self, qd, q, gd, g):
        total = np.zeros((len(q), len(g)), np.float32)
        for s, w in zip(self.scorers, self.w):
            S = s.score(qd, q, gd, g)
            real = S > MISSING / 2
            cnt = real.sum(1, keepdims=True)
            Sm = np.where(real, S, 0)
            mu = Sm.sum(1, keepdims=True) / np.maximum(cnt, 1)
            var = (np.where(real, S - mu, 0) ** 2).sum(1, keepdims=True) / np.maximum(cnt, 1)
            Z = (S - mu) / np.sqrt(var + 1e-12)
            zmin = np.where(real, Z, np.inf).min(1, keepdims=True)
            zmin = np.where(np.isfinite(zmin), zmin, 0)
            total += w * np.where(real, Z, zmin)
        return total
