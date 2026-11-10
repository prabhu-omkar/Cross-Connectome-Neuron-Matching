"""Week 9-10 units: hard-negative sampling, assignment with a 'none' option, calibration pieces."""
import numpy as np
import pandas as pd
import pytest

from flymatch.abstention import AbstentionHead, ece, halves
from flymatch.assign import assign, components, pair_scores
from flymatch.hardneg import sample


def test_hard_negative_sampling_uses_other_dataset_shortlist():
    # mcns anchors 0,1 have BANC shortlists; banc anchor 0 has an MCNS shortlist
    sl = {"mcns": (np.array([0, 1, -1]), np.array([[5, 6, -1], [7, -1, -1]])),
          "banc": (np.array([0, -1, -1, -1, -1, -1, -1, -1]), np.array([[2, 2, 2]]))}
    rng = np.random.default_rng(0)
    add = sample(sl, {"mcns": np.array([0, 1, 2]), "banc": np.array([0, 3])}, per_anchor=20, max_total=100, rng=rng)
    assert set(add["banc"]) <= {5, 6, 7} and set(add["banc"]) == {5, 6, 7}      # padding (-1) never returned
    assert set(add["mcns"]) == {2}
    capped = sample(sl, {"mcns": np.array([0, 1]), "banc": np.array([0])}, per_anchor=20, max_total=2, rng=rng)
    assert len(capped["banc"]) == 2


def _cands(rows):
    return pd.DataFrame(rows, columns=["mcns_uid", "banc_uid", "sim"]).assign(rank_m=1, rank_b=1)


def test_assignment_is_one_to_one_and_respects_tau():
    # two MaleCNS neurons both prefer BANC 0; the global assignment resolves the conflict
    c = _cands([(0, 0, 0.9), (1, 0, 0.8), (1, 1, 0.7), (2, 2, 0.1)])
    comp, size = components(c, 3, 3)
    m = assign(c, comp, size, tau=0.3)
    assert dict(zip(m.mcns_uid, m.banc_uid)) == {0: 0, 1: 1}     # neuron 2 abstains (0.1 < tau)
    assert m.banc_uid.is_unique
    m_all = assign(c, comp, size, tau=0.0)
    assert dict(zip(m_all.mcns_uid, m_all.banc_uid)) == {0: 0, 1: 1, 2: 2}


def test_large_components_use_mutual_best():
    c = _cands([(0, 0, 0.9), (1, 0, 0.8), (1, 1, 0.7)])
    comp, size = components(c, 2, 2)
    m = assign(c, comp, size, tau=0.0, max_component=2, large="mutual")
    assert set(m.how) == {"mutual_best"} and dict(zip(m.mcns_uid, m.banc_uid)) == {0: 0}
    s = assign(c, comp, size, tau=0.0, max_component=2, large="sparse")       # exact: same answer as Hungarian
    assert set(s.how) == {"sparse_lap"} and dict(zip(s.mcns_uid, s.banc_uid)) == {0: 0, 1: 1}


def test_sparse_match_equals_dense_hungarian():
    from scipy.optimize import linear_sum_assignment
    from flymatch.assign import NO_EDGE, sparse_match
    rng = np.random.default_rng(3)
    for _ in range(5):
        n, tau = 30, 0.6
        e = pd.DataFrame({"m": rng.integers(0, n, 150), "b": rng.integers(0, n, 150)}).drop_duplicates()
        sim = rng.uniform(0, 1, len(e))
        mu, bu = sparse_match(e.m.to_numpy(), e.b.to_numpy(), sim, tau)
        C = np.full((n, 2 * n), NO_EDGE)
        C[e.m.to_numpy(), e.b.to_numpy()] = np.where(sim > tau, -sim, NO_EDGE)
        C[np.arange(n), n + np.arange(n)] = -tau
        r, col = linear_sum_assignment(C)
        dense_total = -C[r, col][col < n].sum()
        S = dict(zip(zip(e.m, e.b), sim))
        assert sum(S[(a, b)] for a, b in zip(mu, bu)) - tau * len(mu) == pytest.approx(dense_total - tau * (col < n).sum())


def test_pair_scores():
    m = pd.DataFrame({"mcns_uid": [0, 1], "banc_uid": [0, 5]})
    p = pd.DataFrame({"mcns_uid": [0, 1, 2], "banc_uid": [0, 1, 2]})
    s = pair_scores(m, p)
    assert s["precision"] == 0.5 and s["recall"] == pytest.approx(1 / 3)


def test_ece_and_head_calibrated_on_separable_data():
    assert ece(np.array([0.0, 1.0]), np.array([0, 1])) == 0
    assert ece(np.full(10, 0.9), np.zeros(10)) == pytest.approx(0.9)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    X = (y[:, None] * 1.5 + rng.normal(size=(2000, 3))).astype(np.float32)
    head = AbstentionHead().fit(X[:1000], y[:1000])
    p = head.predict(X[1000:])
    assert ((p > 0.5) == y[1000:]).mean() > 0.75
    assert ece(p, y[1000:]) < 0.08


def test_halves_disjoint_and_complete():
    fams = [f"f{i}" for i in range(11)]
    a, b = halves(fams)
    assert not (a & b) and (a | b) == set(fams)
