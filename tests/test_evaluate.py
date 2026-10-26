"""Harness tests on synthetic data where the answer is known (plan.md 7.2.2)."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flymatch.baselines.scorers import EmbeddingScorer, PairTableScorer, RandomScorer, ZSumScorer
from flymatch.evaluate import MISSING, EvalContext, bootstrap_ci, evaluate, paired_bootstrap, ranks_avg_ties, topk_hits


def synthetic(n_types=60, per=3, seed=0, dim=32):
    """Each type has `per` neurons per dataset (left side); 1:1 pairs for single-neuron types."""
    rng = np.random.default_rng(seed)
    rows = []
    for ds in ("mcns", "banc"):
        uid = 0
        for t in range(n_types):
            k = 1 if t % 3 == 0 else per
            for _ in range(k):
                rows.append(dict(dataset=ds, uid=uid, type_h=f"T{t}", side="left", include=True,
                                 superclass="sc" + str(t % 2), nt="ACH", clean=True))
                uid += 1
        for _ in range(20):        # untyped extras in the gallery
            rows.append(dict(dataset=ds, uid=uid, type_h=None, side="left", include=True,
                             superclass="sc0", nt="ACH", clean=False))
            uid += 1
    n = pd.DataFrame(rows)
    types = pd.DataFrame({"type_h": [f"T{t}" for t in range(n_types)], "family": [f"F{t}" for t in range(n_types)],
                          "superclass": ["sc" + str(t % 2) for t in range(n_types)], "size_bucket": "2"})
    m, b = n[n.dataset == "mcns"], n[n.dataset == "banc"]
    singles = [f"T{t}" for t in range(0, n_types, 3)]
    pairs = pd.DataFrame({"type_h": singles,
                          "mcns_uid": m.set_index("type_h").loc[singles, "uid"].to_numpy(),
                          "banc_uid": b.set_index("type_h").loc[singles, "uid"].to_numpy()})
    pairs["family"] = pairs.type_h.str.replace("T", "F")
    pairs["mcns_clean"] = pairs["banc_clean"] = True
    ctx = EvalContext(n, pairs, types, set(types.family), "val")
    ctx.splits_sha256 = "x"
    # embeddings: one random centre per type, + small noise; untyped random
    C = rng.normal(size=(n_types, dim))
    Z = {}
    for ds, g in n.groupby("dataset"):
        g = g.sort_values("uid")
        Z[ds] = np.stack([C[int(t[1:])] if isinstance(t, str) else rng.normal(size=dim) for t in g.type_h])
        Z[ds] = Z[ds] + 0.01 * rng.normal(size=Z[ds].shape)
    return n, ctx, Z


def test_rank_ties_and_topk():
    S = np.array([[0.5, 0.9, 0.5, 0.1]], np.float32)
    assert ranks_avg_ties(S, np.array([0]))[0] == 2.5        # one better, one tie
    h1, h5 = topk_hits(S, np.array([7]), np.array([1, 7, 2, 3]), k=2)
    assert h1[0] and h5[0]
    h1, h5 = topk_hits(np.full((1, 4), MISSING), np.array([7]), np.array([7, 7, 7, 7]), 2)
    assert not h1[0] and not h5[0]                           # MISSING never counts as a hit


def test_identical_embeddings_perfect():
    _, ctx, Z = synthetic()
    m, pq = evaluate(EmbeddingScorer(Z), ctx, block=17)
    assert m["T1/unfiltered/mean/R@1"] == 1.0
    assert m["T2/unfiltered/mean/acc@1_macro"] == 1.0
    assert m["T1/filtered/mean/R@1"] == 1.0
    assert set(pq.direction) == {"mcns->banc", "banc->mcns"}


def test_random_is_chance():
    _, ctx, _ = synthetic(n_types=150)
    m, _ = evaluate(RandomScorer(0), ctx)
    ng = len(ctx.sides["banc"].gallery)
    assert m["T1/unfiltered/mean/R@1"] < 0.05
    assert abs(m["T1/unfiltered/mean/median_rank"] / ng - 0.5) < 0.15


def test_filters_respect_superclass():
    _, ctx, _ = synthetic()
    m, _ = evaluate(RandomScorer(1), ctx)
    # filtering halves the gallery (two super classes) -> filtered median rank smaller
    assert m["T1/filtered/mean/median_rank"] < m["T1/unfiltered/mean/median_rank"]


def test_pair_table_missing_and_zsum():
    n, ctx, Z = synthetic()
    nm, nb = (n.dataset == "mcns").sum(), (n.dataset == "banc").sum()
    pairs_m = ctx.queries["mcns"].query("partner >= 0")
    M = sp.csr_matrix((np.full(len(pairs_m), 1.0 + PairTableScorer.OFFSET),
                       (pairs_m.uid.to_numpy(), pairs_m.partner.to_numpy())), shape=(nm, nb))
    m, _ = evaluate(PairTableScorer(M), ctx, filters=False)
    assert m["T1/unfiltered/mean/R@1"] == 1.0
    S = PairTableScorer(M).score("mcns", np.array([0]), "banc", np.arange(nb))
    assert (S == MISSING).sum() == nb - 1
    m2, _ = evaluate(ZSumScorer([EmbeddingScorer(Z), PairTableScorer(M)]), ctx, filters=False)
    assert m2["T1/unfiltered/mean/R@1"] == 1.0


def test_bootstrap():
    _, ctx, Z = synthetic()
    _, pq = evaluate(EmbeddingScorer(Z), ctx)
    ci = bootstrap_ci(pq, n_boot=50)
    assert ci["T1/mean/R@1_ci"] == [1.0, 1.0]
    _, pq_r = evaluate(RandomScorer(0), ctx)
    d = paired_bootstrap(pq, pq_r, n_boot=50)
    assert d["diff"] > 0.5 and d["ci"][0] > 0
