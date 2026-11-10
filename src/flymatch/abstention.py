"""Abstention head: "does this neuron have a counterpart at all?" (plan.md 7.3.6; week 10).

A small calibrated classifier on top of FROZEN embeddings, not part of the GNN.

Per-query features (filtered retrieval, hard filters on): top-1 score s1, s2, margin s1 - s2,
s1 - mean(top-10), unfiltered-minus-filtered s1, mutual-nearest-neighbour flag, log rank of the
query in its top-1's own neighbour list, within-dataset size quantile, super-class one-hot.

Training data = partner-removal simulation on VALIDATION families: every query appears twice,
once with the full gallery (label 0, "has match") and once with every gallery neuron of its type
removed (label 1, "no match"). Validation families are split into two halves: logistic
regression + isotonic calibration are fitted on half A and checked on half B (ECE < 0.05 is the
Phase-3 target). Biological check: BANC neurons the BANC team flagged `female-specific` should be
flagged far more often than BANC neurons of shared types.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .evaluate import EvalContext

RANK_CAP = 1000


def _norm(Z: dict) -> dict:
    return {ds: torch.nn.functional.normalize(torch.as_tensor(z, dtype=torch.float32), dim=1) for ds, z in Z.items()}


def size_quantile(side) -> np.ndarray:
    n = side.neurons
    tot = (n.in_syn + n.out_syn).to_numpy(np.float64)
    q = np.zeros(len(n))
    inc = n.include.to_numpy()
    q[inc] = pd.Series(tot[inc]).rank(pct=True).to_numpy()
    return q


def query_features(Zt: dict, ctx: EvalContext, qd: str, gd: str, q_uids: np.ndarray,
                   remove_same_type: bool = False, block: int = 1024) -> pd.DataFrame:
    """Features for queries q_uids (dataset qd) against all included neurons of gd.
    remove_same_type: drop every gallery neuron of the query's type (simulated 'no partner')."""
    A, B = ctx.sides[qd], ctx.sides[gd]
    g = B.gallery
    Gt = Zt[gd][torch.as_tensor(g)]
    qg = A.gallery
    Qt_all = Zt[qd][torch.as_tensor(qg)]
    qpos = np.full(len(A.neurons), -1)
    qpos[qg] = np.arange(len(qg))
    sizeq = size_quantile(A)
    n_sc = int(max(A.sc.max(), B.sc.max())) + 1
    rows = []
    for s in range(0, len(q_uids), block):
        q = q_uids[s:s + block]
        S = Zt[qd][torch.as_tensor(q)] @ Gt.T
        if remove_same_type:
            same = (A.type_code[q][:, None] == B.type_code[g][None, :]) & (A.type_code[q][:, None] >= 0)
            S = S.masked_fill(torch.as_tensor(same), float("-inf"))
        s1u = S.max(1).values
        Sf = S.masked_fill(~torch.as_tensor(ctx.allowed(qd, q, gd, g)), float("-inf"))
        top, idx = Sf.topk(min(10, len(g)), dim=1)
        top = torch.where(torch.isfinite(top), top, torch.full_like(top, -1.0))
        g1 = g[idx[:, 0].numpy()]
        # reverse direction: where does the query rank in its top-1's own (filtered) list?
        R = Zt[gd][torch.as_tensor(g1)] @ Qt_all.T
        R = R.masked_fill(~torch.as_tensor(ctx.allowed(gd, g1, qd, qg)), float("-inf"))
        own = (Zt[gd][torch.as_tensor(g1)] * Zt[qd][torch.as_tensor(q)]).sum(1)
        rrank = 1 + (R > own[:, None]).sum(1).numpy()
        rrank = np.minimum(rrank, RANK_CAP)
        t = top.numpy()
        f = pd.DataFrame({
            "uid": q, "top1_uid": g1,
            "s1": t[:, 0], "s2": t[:, 1], "margin": t[:, 0] - t[:, 1], "s1_minus_mean10": t[:, 0] - t.mean(1),
            "unf_minus_f": s1u.clamp_min(-1).numpy() - t[:, 0], "mutual": (rrank == 1).astype(float),
            "log_rev_rank": np.log(rrank), "size_q": sizeq[q],
        })
        oh = np.zeros((len(q), n_sc))
        oh[np.arange(len(q)), A.sc[q]] = 1
        for i in range(n_sc):
            f[f"sc{i}"] = oh[:, i]
        rows.append(f)
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "direction", f"{qd}->{gd}")
    return out


FEATURES = ["s1", "s2", "margin", "s1_minus_mean10", "unf_minus_f", "mutual", "log_rev_rank", "size_q"]


def feature_cols(df: pd.DataFrame) -> list[str]:
    return FEATURES + sorted(c for c in df.columns if c.startswith("sc") and c[2:].isdigit())


def simulation_set(Zt: dict, ctx: EvalContext, per_type: int = 10, seed: int = 0) -> pd.DataFrame:
    """Partner-removal simulation on ctx's split (validation). Label y = 1 means 'no match'."""
    rng = np.random.default_rng(seed)
    frames = []
    for qd, gd in (("mcns", "banc"), ("banc", "mcns")):
        q = ctx.queries[qd]
        q = q.iloc[rng.permutation(len(q))]
        q = q[(q.groupby("type_h").cumcount() < per_type).to_numpy()]
        uids = q.uid.to_numpy()
        for y, remove in ((0, False), (1, True)):
            f = query_features(Zt, ctx, qd, gd, uids, remove_same_type=remove)
            f["y"] = y
            f["family"] = q.family.to_numpy()
            f["type_h"] = q.type_h.to_numpy()
            frames.append(f)
    return pd.concat(frames, ignore_index=True)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


class AbstentionHead:
    """Standardise -> logistic regression -> isotonic calibration. predict() returns p_none."""

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0):
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        oof = cross_val_predict(self.lr, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                method="predict_proba")[:, 1]
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, y)
        self.lr.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.iso.predict(self.lr.predict_proba(X)[:, 1])


def halves(families, seed: int = 20261026) -> tuple[set, set]:
    f = np.array(sorted(set(families)))
    p = np.random.default_rng(seed).permutation(len(f))
    return set(f[p[: len(f) // 2]]), set(f[p[len(f) // 2:]])


def fit_and_check(sim: pd.DataFrame, seed: int = 0) -> tuple[AbstentionHead, dict, pd.DataFrame]:
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    cols = feature_cols(sim)
    a, b = halves(sim.family)
    A, B = sim[sim.family.isin(a)], sim[sim.family.isin(b)]
    head = AbstentionHead().fit(A[cols].to_numpy(), A.y.to_numpy(), seed)
    head.columns = cols
    p = head.predict(B[cols].to_numpy())
    y = B.y.to_numpy()
    prec, rec, _ = precision_recall_curve(y, p)
    m = {"abstain/AUROC": float(roc_auc_score(y, p)), "abstain/AUPRC": float(average_precision_score(y, p)),
         "abstain/ECE": ece(p, y), "abstain/precision@recall50": float(prec[rec >= 0.5].max()),
         "abstain/n_fit": int(len(A)), "abstain/n_check": int(len(B))}
    out = B.assign(p_none=p)
    return head, m, out


def biological_check(head: AbstentionHead, Zt: dict, ctx: EvalContext, base: pd.DataFrame) -> dict:
    """Female-specific BANC neurons (no male counterpart by annotation) vs BANC queries of shared
    types with their partners present (`base`: rows of the check half, direction banc->mcns, y = 0)."""
    from scipy.stats import fisher_exact
    n = ctx.sides["banc"].neurons
    fs = n.uid[n.include & (n.dimorphism.astype(str) == "female-specific")].to_numpy()
    if len(fs) == 0:
        return {}
    f = query_features(Zt, ctx, "banc", "mcns", fs)
    p = head.predict(f[head.columns].to_numpy())
    pb = base.p_none.to_numpy()
    a, b = int((p > 0.5).sum()), int((p <= 0.5).sum())
    c, d = int((pb > 0.5).sum()), int((pb <= 0.5).sum())
    _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
    rate, base_rate = a / max(a + b, 1), c / max(c + d, 1)
    return {"abstain/bio/n_female_specific": int(len(fs)), "abstain/bio/flagged_rate": rate,
            "abstain/bio/base_rate": base_rate, "abstain/bio/enrichment": rate / max(base_rate, 1e-9),
            "abstain/bio/fisher_p": float(pval)}
