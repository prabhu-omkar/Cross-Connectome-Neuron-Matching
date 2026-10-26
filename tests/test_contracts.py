"""Data contracts (plan.md 5.3). Skipped until the data has been built."""
import json

import numpy as np
import pandas as pd
import pytest

from flymatch.config import load_paths
from flymatch.io import read_map

NEURON_COLS = ["dataset", "neuron_id", "uid", "side", "superclass_raw", "superclass", "class_raw",
               "subclass_raw", "hemilineage", "nt", "nt_conf", "type_primary", "type_alt",
               "flywire_type", "manc_type", "hemibrain_type", "supertype", "status", "type_h", "include"]


@pytest.fixture(scope="module")
def neurons(built):
    return pd.read_parquet(built.neurons)


def test_neuron_columns_and_types(neurons):
    assert set(NEURON_COLS) <= set(neurons.columns)
    assert neurons.neuron_id.dtype == np.int64
    assert neurons.nt_conf.dtype == np.float32
    assert set(neurons.dataset.astype(str).unique()) == {"mcns", "banc"}
    assert set(neurons.side.astype(str).unique()) <= {"left", "right", "center", "unknown"}


def test_ids_unique_and_uid_dense(neurons):
    for ds, g in neurons.groupby("dataset", observed=True):
        assert g.neuron_id.is_unique
        assert (np.sort(g.uid.to_numpy()) == np.arange(len(g))).all()
        assert (g.sort_values("uid").neuron_id.diff().dropna() > 0).all()   # uid sorted by neuron_id


def test_ids_match_raw_text(neurons):
    """No ID changed in the int64 round-trip: compare against the raw CSV read as text."""
    paths = load_paths()
    for ds, key in (("mcns", "mcns_dir"), ("banc", "banc_dir")):
        raw = pd.read_csv(paths[key] / "neurons.csv.gz", dtype=str, usecols=["Root ID"])["Root ID"]
        ours = neurons.loc[neurons.dataset == ds, "neuron_id"].astype(str)
        assert set(raw) == set(ours)


def test_edges(built, neurons):
    e = pd.read_parquet(built.edges)
    n = neurons.groupby("dataset", observed=True).size()
    log = json.loads(built.log("harmonise").read_text())
    for ds, g in e.groupby("dataset", observed=True):
        assert g.pre_uid.min() >= 0 and g.pre_uid.max() < n[ds] and g.post_uid.max() < n[ds]
        assert (g.pre_uid != g.post_uid).all()
        assert g.syn.min() >= log["threshold"][ds]
        assert not g.duplicated(["pre_uid", "post_uid"]).any()
        # fractions sum to 1 per neuron
        s_in = g.groupby("post_uid").w_in.sum()
        assert np.allclose(s_in, 1, atol=1e-4)


def test_edge_row_accounting(built):
    log = json.loads(built.log("ingest").read_text())
    for ds in ("mcns", "banc"):
        assert log[f"{ds}_edge_rows"] == (log[f"{ds}_edge_rows_raw"] - log[f"{ds}_edge_rows_dropped_unknown_endpoint"]
                                          - log[f"{ds}_edge_rows_dropped_self_loop"])


def test_edges_np_consistent_with_pairs(built):
    enp = pd.read_parquet(built.edges_np)
    e = pd.read_parquet(built.edges)
    for ds in ("mcns", "banc"):
        a = enp[enp.dataset == ds].syn.sum()
        b = e[e.dataset == ds].syn.sum()
        assert a == b


def test_maps_cover_canonical_both_sides(built):
    m = read_map(built.maps / "neuropil_map.csv")
    inc = m[m.include.astype(int) == 1]
    assert (inc.groupby("canonical").dataset.nunique() == 2).all()


def test_features(built, neurons):
    spec = json.loads(built.feature_spec.read_text())
    for ds in ("mcns", "banc"):
        X = np.load(built.features(ds), mmap_mode="r")
        assert X.dtype == np.float32
        assert X.shape == ((neurons.dataset == ds).sum(), len(spec["columns"]))
        assert np.isfinite(X).all()
    p = spec["mirror_perm"]
    assert sorted(p) == list(range(len(p)))
    cols = spec["columns"]
    assert cols[p[cols.index("in:AL_L")]] == "in:AL_R"


def test_gt_pairs(built, neurons):
    p = pd.read_parquet(built.gt_pairs)
    assert p.mcns_uid.is_unique and p.banc_uid.is_unique
    m = neurons[neurons.dataset == "mcns"].set_index("uid")
    b = neurons[neurons.dataset == "banc"].set_index("uid")
    assert m.loc[p.mcns_uid, "include"].all() and b.loc[p.banc_uid, "include"].all()
    assert (m.loc[p.mcns_uid, "type_h"].to_numpy() == p.type_h.to_numpy()).all()
    assert (b.loc[p.banc_uid, "type_h"].to_numpy() == p.type_h.to_numpy()).all()
    assert (m.loc[p.mcns_uid, "side"].astype(str).to_numpy() == b.loc[p.banc_uid, "side"].astype(str).to_numpy()).all()
    assert set(p.source.dropna()) <= {"primary", "flywire", "manc"}


def test_gt_types(built):
    g = pd.read_parquet(built.gt_types)
    assert g.type_h.is_unique
    assert (g.n_mcns > 0).all() and (g.n_banc > 0).all()
    assert set(["type_h", "family", "superclass", "n_mcns", "n_banc", "size_bucket"]) <= set(g.columns)
