"""Control C1 / hypothesis H4 (plan 7.4.2): train BANC <-> MaleCNS on what the hemibrain also has, then match
hemibrain neurons zero-shot.

    python -m flymatch.run c1-build                 # data variant data/c1/ (needs run hemibrain-prepare first)
    python -m flymatch.run train --exp configs/experiments/gnn_c1.yaml --seeds 0,1,2   # uses data_c1 (yaml key)
    python scripts/c1_eval.py                       # zero-shot hemibrain <-> BANC / MaleCNS vs B1

The variant (configs/data_c1.yaml, out_dir data/c1) keeps neurons, ground truth, families and the frozen split
of the main data, and changes only what the model SEES, for all three datasets alike:
  graph     synapses inside the neuropils that lie in the hemibrain volume (results/hemibrain_neuropil_
            completeness.csv, keep == True); a pair's weight = its synapses there; pairs that passed the
            >= 5 threshold overall are kept even if fewer remain (hemibrain pairs are treated the same way)
  features  np_in / np_out on those neuropils (sqrt fractions, as in the main data); size_rank recomputed on
            the restricted graph (within-dataset quantiles); side one-hot.
            Dropped: region summaries (span OL / VNC / nerves), neurotransmitter, super class, hemilineage -
            hemibrain v1.2.1 has none of them (and super class would leak through the type link).
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from .config import Layout, write_json
from .features import SIDES, mirror_name, neuropil_profiles, one_hot, rank_normalise
from .harmonise import add_fractions, node_stats

DATASETS = ("mcns", "banc")


def kept_neuropils(main: Layout) -> list[str]:
    comp = pd.read_csv(main.results / "hemibrain_neuropil_completeness.csv")
    return sorted(comp.neuropil[comp.keep])


def _restricted(enp: pd.DataFrame, n: int, nps: list[str], side: pd.Series):
    """Restricted graph + feature matrix for one dataset (uids 0..n-1)."""
    e = enp[enp.neuropil.isin(nps)]
    pairs = e.groupby(["pre_uid", "post_uid"], as_index=False).syn.sum()
    edges = add_fractions(pairs, n)
    p_in, p_out, raw_in, raw_out = neuropil_profiles(e, n, nps)
    st = node_stats(pairs, n)
    size = np.stack([rank_normalise(v) for v in [st.in_deg.to_numpy(), st.out_deg.to_numpy(), st.in_syn.to_numpy(),
                                                 st.out_syn.to_numpy(), (raw_in > 0).sum(1), (raw_out > 0).sum(1)]], 1)
    X = np.concatenate([p_in, p_out, size, one_hot(side, SIDES)], axis=1).astype(np.float32)
    assert np.isfinite(X).all()
    return edges, X, int((p_in.sum(1) + p_out.sum(1) == 0).sum())


def build(main: Layout, c1: Layout, log=print) -> dict:
    nps = kept_neuropils(main)
    K = len(nps)
    columns = [f"in:{p}" for p in nps] + [f"out:{p}" for p in nps] + \
              ["in_deg", "out_deg", "in_syn", "out_syn", "n_np_in", "n_np_out"] + [f"side:{s}" for s in SIDES]
    blocks = {"np_in": [0, K], "np_out": [K, 2 * K], "size_rank": [2 * K, 2 * K + 6],
              "side": [2 * K + 6, 2 * K + 6 + len(SIDES)]}
    pos = {c: i for i, c in enumerate(columns)}
    mirrored = lambda c: (f"{c.split(':')[0]}:{mirror_name(c.split(':', 1)[1])}" if c.startswith(("in:", "out:"))
                          else {"side:left": "side:right", "side:right": "side:left"}.get(c, c))
    mirror_perm = [pos.get(mirrored(c), i) for i, c in enumerate(columns)]

    for f in (main.gt_types, main.families):        # identical type ground truth and families
        shutil.copy2(f, c1.interim / f.name)
    neurons = pd.read_parquet(main.neurons)
    enp_all = pd.read_parquet(main.edges_np, columns=["dataset", "pre_uid", "post_uid", "neuropil", "syn"])
    edges, stats, has_syn = [], {}, {}
    for ds in DATASETS:
        n = neurons[neurons.dataset == ds].sort_values("uid").reset_index(drop=True)
        e, X, empty = _restricted(enp_all[enp_all.dataset == ds], len(n), nps, n.side)
        edges.append(e.assign(dataset=ds))
        np.save(c1.features(ds), X)
        has_syn[ds] = X[:, :2 * K].any(1)
        stats[ds] = {"n": len(n), "f": X.shape[1], "edges": len(e), "no_synapse_in_kept_neuropils": empty}
        log(f"[c1] {ds}: X {X.shape}, {len(e):,} edges, {empty:,} neurons without synapses in the kept neuropils")
    del enp_all
    pd.concat(edges, ignore_index=True).to_parquet(c1.edges, index=False)
    # A neuron with no synapse in the kept neuropils has (near-)empty features: not trained on, not evaluated.
    hs = np.zeros(len(neurons), bool)
    for ds in DATASETS:
        m = (neurons.dataset == ds).to_numpy()
        hs[m] = has_syn[ds][neurons.uid.to_numpy()[m]]
    nv = neurons.assign(include_main=neurons.include, include=neurons.include & hs)
    nv.to_parquet(c1.neurons, index=False)
    gp = pd.read_parquet(main.gt_pairs)
    ok = gp.mcns_uid.map(lambda u: has_syn["mcns"][u]) & gp.banc_uid.map(lambda u: has_syn["banc"][u])
    gp[ok].to_parquet(c1.gt_pairs, index=False)
    stats["included"] = {ds: int(nv.include[nv.dataset == ds].sum()) for ds in DATASETS}
    stats["gt_pairs_1to1"] = {"main": int(len(gp)), "c1": int(ok.sum())}
    log(f"[c1] included (synapses in kept neuropils): {stats['included']}; 1:1 pairs {int(ok.sum()):,} of {len(gp):,}")

    hb_dir = main.processed / "hemibrain"
    hb = pd.read_parquet(hb_dir / "neurons.parquet")
    e, X, empty = _restricted(pd.read_parquet(hb_dir / "edges_np.parquet"), len(hb), nps, hb.side)
    out = c1.processed / "hemibrain"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "features.npy", X)
    e.to_parquet(out / "edges.parquet", index=False)
    for f in ("neurons.parquet", "gt_pairs_mcns.parquet", "gt_pairs_banc.parquet"):
        shutil.copy2(hb_dir / f, out / f)
    stats["hemibrain"] = {"n": len(hb), "f": X.shape[1], "edges": len(e), "no_synapse_in_kept_neuropils": empty}
    log(f"[c1] hemibrain: X {X.shape}, {len(e):,} edges, {empty:,} neurons without synapses in the kept neuropils")

    spec = {"columns": columns, "blocks": blocks, "neuropils": nps, "sides": SIDES, "mirror_perm": mirror_perm,
            "stats": stats, "variant": "C1 hemibrain-restricted (plan 7.4.2)",
            "notes": {"np_in/np_out": "sqrt fraction of synapses per kept neuropil",
                      "size_rank": "within-dataset quantile on the restricted graph",
                      "dropped": "region_*, nt, superclass, hemilineage (absent in hemibrain v1.2.1)"}}
    write_json(spec, c1.feature_spec)
    return spec


def hemibrain_graph(c1: Layout):
    """Sparse (A_in, A_out) for the hemibrain, built exactly as graph.load_graphs builds the others."""
    from .graph import build as build_adj
    d = c1.processed / "hemibrain"
    n = len(pd.read_parquet(d / "neurons.parquet", columns=["uid"]))
    return build_adj(pd.read_parquet(d / "edges.parquet"), n), n


def embed_hemibrain(c1: Layout, run_dir, device: str = "cpu") -> np.ndarray:
    """Embed every hemibrain neuron with a model trained on the C1 variant (zero-shot: hemibrain never seen)."""
    import json
    from pathlib import Path

    import torch

    from .inference import run_config
    from .models.encoder import Encoder
    cfg = run_config(Path(run_dir))
    mc = cfg["model"]
    spec = json.loads(c1.feature_spec.read_text())
    keep = np.ones(len(spec["columns"]), bool)
    for b in cfg["features"]["drop_blocks"]:
        a, z = spec["blocks"][b]
        keep[a:z] = False
    model = Encoder(int(keep.sum()), mc["width"], mc["out_dim"], mc["layers"], mc["dropout"])
    model.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=True))
    model.eval()
    x = torch.as_tensor(np.load(c1.processed / "hemibrain" / "features.npy")[:, keep])
    A = hemibrain_graph(c1)[0] if mc["layers"] > 0 else (None, None)
    with torch.no_grad():
        return model.embed(x, *A).float().numpy()
