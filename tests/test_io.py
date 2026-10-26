"""Unit tests that need no data."""
import gzip

import numpy as np
import pandas as pd
import pytest

from flymatch.ingest import UnmappedValueError, apply_map
from flymatch.io import ids_to_int64, label_columns, parse_mcns_labels, read_neurons, NEURON_RENAME
from flymatch.groundtruth import UnionFind, stem
from flymatch.retrieval import ranks_of_true, recall_metrics


def test_18_digit_ids_survive():
    s = pd.Series(["720575941076956631", "720575940381905254"])
    out = ids_to_int64(s)
    assert out.dtype == np.int64
    assert out.iloc[0] == 720575941076956631
    assert float(s.iloc[0]) != 720575941076956631          # the trap we are avoiding


def test_ids_reject_missing():
    with pytest.raises(ValueError):
        ids_to_int64(pd.Series(["1", None]))


def test_read_neurons_keeps_ids_exact(tmp_path):
    cols = list(NEURON_RENAME) + ["Top in/out region"]
    df = pd.DataFrame([{c: "" for c in cols}, {c: "" for c in cols}])
    df["Root ID"] = ["720575941076956631", "720575940381905254"]
    df["Predicted NT confidence"] = ["0.5", ""]
    p = tmp_path / "n.csv.gz"
    with gzip.open(p, "wt") as f:
        df.to_csv(f, index=False)
    n = read_neurons(p, "banc")
    assert n.neuron_id.tolist() == [720575941076956631, 720575940381905254]
    assert np.isnan(n.nt_conf.iloc[1])


def test_parse_labels():
    s = "hemibrainType: Giant Fiber,instance: DNp01(GF),mancBodyid: 10000,statusLabel: Roughly traced,synonyms: K: GF"
    d = parse_mcns_labels(s)
    assert d == {"statusLabel": "Roughly traced", "hemibrainType": "Giant Fiber", "mancBodyid": "10000"}
    cols = label_columns(pd.Series([s, None, "flywireType: Tm21,supertype: 12"]))
    assert cols.status.iloc[0] == "Roughly traced"
    assert cols.flywire_type.iloc[2] == "Tm21" and cols.supertype.iloc[2] == "12"
    assert cols.isna().iloc[1].all()


def test_unmapped_value_fails_loudly():
    with pytest.raises(UnmappedValueError):
        apply_map(pd.Series(["a", "b"]), {"a": "x"}, "thing", "mcns")
    assert apply_map(pd.Series(["a", None]), {"a": "x", "<missing>": "u"}, "t", "d").tolist() == ["x", "u"]


def test_stem_and_unionfind():
    assert stem("LC10a") == "LC10" and stem("T4a") == "T4" and stem("Dm3a") == "Dm3"
    uf = UnionFind()
    uf.union("a", "b"); uf.union("c", "d"); uf.union("b", "d")
    assert uf.find("a") == uf.find("c") and uf.find("e") == "e"


def test_retrieval_identity_and_random():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(500, 16)).astype(np.float32)
    r, top = ranks_of_true(G[:100], G, np.arange(100), block=37)
    assert (r == 1).all() and (top[:, 0] == np.arange(100)).all()
    r2, _ = ranks_of_true(rng.normal(size=(200, 16)), G, rng.integers(0, 500, 200))
    assert 0.3 < np.median(r2) / 500 < 0.7
    m = recall_metrics(np.array([1, 2, 20]))
    assert m["R@1"] == pytest.approx(1 / 3) and m["R@10"] == pytest.approx(2 / 3)


def test_retrieval_filter_keeps_true_partner():
    G = np.eye(4, dtype=np.float32)
    r, _ = ranks_of_true(G[[0]], G, np.array([0]), allowed=lambda rows: np.zeros((len(rows), 4), bool))
    assert r[0] == 1
