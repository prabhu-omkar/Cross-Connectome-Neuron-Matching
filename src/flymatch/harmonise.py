"""Phase 1.2 — harmonise connectivity (plan.md 7.1.3-7.1.4).

1. Equal pair thresholds: keep neuron pairs whose total synapses (over all neuropils) are
   >= threshold[dataset]; keep the neuropil rows of surviving pairs.
2. Fractional weights: w_in = syn / input_total(post), w_out = syn / output_total(pre).
3. Per-neuron totals and the inclusion rules.

Writes edges_np.parquet, edges.parquet, neurons.parquet.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import DATASETS
from .config import Layout, load_data_config, write_json
from .ingest import load_maps


def pair_totals(e: pd.DataFrame) -> pd.DataFrame:
    """Sum neuropil-split rows to neuron pairs."""
    return e.groupby(["pre_uid", "post_uid"], sort=True, observed=True).syn.sum().reset_index()


def threshold_rows(e: pd.DataFrame, t: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (neuropil rows of surviving pairs, surviving pair table)."""
    pairs = pair_totals(e)
    keep_pairs = pairs[pairs.syn >= t]
    n = int(max(e.pre_uid.max(), e.post_uid.max())) + 1
    key = e.pre_uid.to_numpy(np.int64) * n + e.post_uid.to_numpy(np.int64)
    kkey = keep_pairs.pre_uid.to_numpy(np.int64) * n + keep_pairs.post_uid.to_numpy(np.int64)
    return e[np.isin(key, kkey)], keep_pairs


def add_fractions(pairs: pd.DataFrame, n_nodes: int) -> pd.DataFrame:
    pre, post, syn = pairs.pre_uid.to_numpy(), pairs.post_uid.to_numpy(), pairs.syn.to_numpy(np.float64)
    out_tot = np.bincount(pre, weights=syn, minlength=n_nodes)
    in_tot = np.bincount(post, weights=syn, minlength=n_nodes)
    pairs = pairs.copy()
    pairs["w_in"] = (syn / in_tot[post]).astype(np.float32)
    pairs["w_out"] = (syn / out_tot[pre]).astype(np.float32)
    return pairs


def node_stats(pairs: pd.DataFrame, n_nodes: int) -> pd.DataFrame:
    pre, post, syn = pairs.pre_uid.to_numpy(), pairs.post_uid.to_numpy(), pairs.syn.to_numpy(np.int64)
    return pd.DataFrame({
        "in_deg": np.bincount(post, minlength=n_nodes).astype(np.int32),
        "out_deg": np.bincount(pre, minlength=n_nodes).astype(np.int32),
        "in_syn": np.bincount(post, weights=syn, minlength=n_nodes).astype(np.int64),
        "out_syn": np.bincount(pre, weights=syn, minlength=n_nodes).astype(np.int64),
    })


def inclusion(n: pd.DataFrame, cfg: dict, sc_include: dict) -> pd.Series:
    """plan.md 7.1.3. Excluded neurons stay in the graph; they leave GT and evaluation only."""
    ok_sc = n.superclass.astype(str).map(sc_include).fillna(False).astype(bool)
    not_frag = ~n.type_primary.fillna("").str.contains("fragment", case=False)
    ok_side = n.side.astype(str).isin(cfg["include_sides"])
    ok_syn = (n.in_syn + n.out_syn) >= cfg["min_total_synapses"]
    return ok_sc & not_frag & ok_side & ok_syn


def run(cfg: dict | None = None):
    t0 = time.time()
    cfg = cfg or load_data_config()
    layout = Layout(cfg)
    maps = load_maps(layout)
    neurons = pd.read_parquet(layout.neurons_raw)
    edges_raw = pd.read_parquet(layout.edges_raw)
    npm = pd.concat(maps["neuropil"].values())
    np_include = dict(zip(npm.canonical, npm.include.astype(int).astype(bool)))
    log: dict = {"threshold": cfg["threshold"], "min_total_synapses": cfg["min_total_synapses"]}
    out_np, out_pairs, out_neurons = [], [], []
    for ds in DATASETS:
        n = neurons[neurons.dataset == ds].reset_index(drop=True)
        e = edges_raw[edges_raw.dataset == ds].drop(columns="dataset")
        N = len(n)
        t = int(cfg["threshold"][ds])
        rows, pairs = threshold_rows(e, t)
        all_pairs = pair_totals(e)
        log[ds] = {
            "pairs_before": len(all_pairs), "pairs_after": len(pairs),
            "pairs_removed": len(all_pairs) - len(pairs),
            "synapses_before": int(all_pairs.syn.sum()), "synapses_after": int(pairs.syn.sum()),
            "synapses_removed": int(all_pairs.syn.sum() - pairs.syn.sum()),
        }
        enp = (rows.groupby(["pre_uid", "post_uid", "neuropil"], observed=True, sort=True)
               .syn.sum().reset_index())
        enp["syn"] = enp.syn.astype(np.int32)
        enp.insert(0, "dataset", ds)
        inc = enp.neuropil.astype(str).map(np_include)
        log[ds]["frac_synapses_in_canonical_neuropils"] = float(enp.syn[inc].sum() / enp.syn.sum())
        pairs = add_fractions(pairs, N)
        pairs["syn"] = pairs.syn.astype(np.int32)
        pairs.insert(0, "dataset", ds)
        n = pd.concat([n, node_stats(pairs, N)], axis=1)
        n["include"] = inclusion(n, cfg, maps["superclass_include"])
        log[ds]["included"] = int(n.include.sum())
        log[ds]["included_by_superclass"] = n.groupby("superclass", observed=True).include.agg(["sum", "size"]).astype(int).to_dict("index")
        log[ds]["median_degree_all"] = float(np.median(n.in_deg + n.out_deg))
        out_np.append(enp)
        out_pairs.append(pairs)
        out_neurons.append(n)
        print(f"[harmonise] {ds}: pairs {log[ds]['pairs_before']:,} -> {log[ds]['pairs_after']:,} "
              f"(-{log[ds]['synapses_removed']:,} synapses); "
              f"{log[ds]['frac_synapses_in_canonical_neuropils']:.1%} of synapses in canonical neuropils; "
              f"included {log[ds]['included']:,}/{N:,}")
    enp = pd.concat(out_np, ignore_index=True)
    for c in ["dataset", "neuropil"]:
        enp[c] = enp[c].astype(str).astype("category")
    pairs = pd.concat(out_pairs, ignore_index=True)
    pairs["dataset"] = pairs.dataset.astype("category")
    nn = pd.concat(out_neurons, ignore_index=True)
    for c in ["dataset", "side", "superclass", "nt"]:
        nn[c] = nn[c].astype(str).astype("category")
    enp.to_parquet(layout.edges_np, index=False)
    pairs.to_parquet(layout.edges, index=False)
    nn.to_parquet(layout.neurons, index=False)
    log["seconds"] = round(time.time() - t0, 1)
    write_json(log, layout.log("harmonise"))
    return log


if __name__ == "__main__":
    run()
