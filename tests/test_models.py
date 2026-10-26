"""Encoder / graph / loss smoke tests on tiny synthetic graphs."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from flymatch.graph import build, permute_edges  # noqa: E402
from flymatch.harmonise import add_fractions  # noqa: E402
from flymatch.models.encoder import Encoder  # noqa: E402
from flymatch.models.losses import coral, multi_pos_infonce  # noqa: E402
from flymatch.baselines.cocoa_style import label_projected  # noqa: E402
from flymatch.seeds import seed_families  # noqa: E402


def tiny_edges(n=30, m=120, seed=0):
    rng = np.random.default_rng(seed)
    e = pd.DataFrame({"pre_uid": rng.integers(0, n, m), "post_uid": rng.integers(0, n, m), "syn": rng.integers(5, 50, m)})
    e = e[e.pre_uid != e.post_uid].groupby(["pre_uid", "post_uid"], as_index=False).syn.sum()
    return add_fractions(e, n)


def test_adjacency_rows_average():
    e = tiny_edges()
    A_in, A_out = build(e, 30)
    s = torch.sparse.sum(A_in, 1).to_dense()
    assert torch.allclose(s[s > 0], torch.ones_like(s[s > 0]), atol=1e-5)
    Au, _ = build(e, 30, undirected=True)
    su = torch.sparse.sum(Au, 1).to_dense()
    assert torch.allclose(su[su > 0], torch.ones_like(su[su > 0]), atol=1e-5)


def test_encoder_forward_backward_sparse():
    e = tiny_edges()
    A_in, A_out = build(e, 30)
    x = torch.randn(30, 12)
    for layers in (0, 1, 3):
        enc = Encoder(12, d=16, out=8, layers=layers)
        z = enc(x, A_in, A_out)
        assert z.shape == (30, 8) and torch.allclose(z.norm(dim=1), torch.ones(30), atol=1e-5)
        z.sum().backward()
        assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)


def test_message_passing_uses_graph():
    e = tiny_edges()
    A_in, A_out = build(e, 30)
    x = torch.randn(30, 12)
    enc = Encoder(12, d=16, out=8, layers=2).eval()
    rng = np.random.default_rng(1)
    B_in, B_out = build(permute_edges(e, rng), 30)
    assert not torch.allclose(enc(x, A_in, A_out), enc(x, B_in, B_out))
    mlp = Encoder(12, d=16, out=8, layers=0).eval()
    assert torch.allclose(mlp(x, A_in, A_out), mlp(x, B_in, B_out))


def test_permute_preserves_out_degree_mass():
    e = tiny_edges(n=200, m=3000)
    p = permute_edges(e, np.random.default_rng(0))
    assert abs(p.syn.sum() - e.syn.sum()) / e.syn.sum() < 0.05


def test_infonce_prefers_aligned():
    torch.manual_seed(0)
    za = torch.nn.functional.normalize(torch.randn(8, 4), dim=1)
    pos = torch.eye(8, dtype=torch.bool)
    lt = torch.tensor(np.log(0.1))
    aligned = multi_pos_infonce(za, za, pos, lt)
    shuffled = multi_pos_infonce(za, za[torch.randperm(8)], pos, lt)
    assert aligned < shuffled
    valid = ~torch.eye(8, dtype=torch.bool)       # mask everything but positives -> loss 0
    assert multi_pos_infonce(za, za, pos, lt, valid=~valid) == pytest.approx(0, abs=1e-6)
    assert coral(za, za) == 0


def test_label_projected():
    e = pd.DataFrame({"pre_uid": [0, 1, 2], "post_uid": [2, 2, 3], "syn": [10, 30, 5]})
    st = np.array([0, 1, -1, -1])
    X = label_projected(e, 4, st, 2).toarray()
    # neuron 2 receives 10 from type 0 and 30 from type 1 -> in-fractions .25 / .75
    assert np.allclose(X[2, :2], [0.25, 0.75])
    # neuron 0 sends all output to neuron 2 (untyped) -> zero out-projection
    assert X[0, 2:].sum() == 0


def test_seed_families_nested():
    gt = pd.DataFrame({"type_h": [f"t{i}" for i in range(200)], "family": [f"f{i}" for i in range(200)],
                       "superclass": ["a", "b"] * 100})
    fams = list(gt.family)
    prev = set()
    for f in (0.05, 0.1, 0.25, 0.5, 1.0):
        cur = set(seed_families(gt, fams, f, draw=1))
        assert prev <= cur
        prev = cur
    assert len(prev) == 200
