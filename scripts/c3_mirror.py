"""Control C3 (evaluation form): what does the domain gap cost?

    python scripts/c3_mirror.py [--runs B4 gnn_L2] [--seeds 0,1,2]

Same trained encoder, two retrieval problems scored by ONE evaluator on the validation families:
  cross   MaleCNS <-> BANC, the 1:1 validation pairs (the real task).
  within  inside one dataset, left neuron <-> right neuron of the same type (types with exactly one
          included neuron per side). The gallery side is embedded from MIRRORED features (every node's
          _L/_R neuropil columns and side one-hot swapped; the graph is unchanged), so a right neuron
          looks like a left one. Same animal, same reconstruction, same annotation - no domain gap.
within - cross = what the domain gap costs this encoder. B1 (no learning) is scored the same way.

This is cheaper than the plan's C3 (an encoder TRAINED on bilateral pairs, GPU). The encoders here were
trained cross-dataset only; mirror symmetry was never a training target, so `within` is a lower bound
on the zero-gap ceiling. Ranks: 1 + #(gallery scores > partner) + 0.5 * #ties (as in the harness).
Gallery: all included neurons of the other dataset (cross) / of the opposite side (within).
Filtered: gallery restricted to the query's super class (cross also: same side).
Output: results/c3_mirror.csv (one row per method/seed/setting).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flymatch.config import Layout, load_data_config  # noqa: E402
from flymatch.inference import embed_run  # noqa: E402
from flymatch.splits import load_splits  # noqa: E402

RUNS = REPO / "runs"


def l2(z):
    z = np.asarray(z, np.float32)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def ranks(Zq, q, Zg, gallery, partner, gal_sc=None, q_sc=None, block=512):
    """Rank of partner[i] for query q[i] among `gallery` (uids). Optional super-class filter."""
    Zg_g = Zg[gallery]
    pos = {u: i for i, u in enumerate(gallery)}
    out_u, out_f = np.empty(len(q)), np.empty(len(q))
    for a in range(0, len(q), block):
        qq = q[a:a + block]
        S = Zq[qq] @ Zg_g.T
        p_idx = np.array([pos[u] for u in partner[a:a + block]])
        s_true = S[np.arange(len(qq)), p_idx][:, None]
        gt, eq = S > s_true, S == s_true
        out_u[a:a + len(qq)] = 1 + gt.sum(1) + 0.5 * (eq.sum(1) - 1)
        if gal_sc is not None:
            m = gal_sc[None, :] == q_sc[a:a + block][:, None]
            out_f[a:a + len(qq)] = 1 + (gt & m).sum(1) + 0.5 * ((eq & m).sum(1) - 1)
    return out_u, out_f


def summarise(r):
    return {"R@1": float((r <= 1).mean()), "R@10": float((r <= 10).mean()), "MRR": float((1 / r).mean())}


def evaluate_pair(name, seed, Z, Zmir, n, pairs_cross, within_pairs, include):
    rows = []
    Z = {ds: l2(z) for ds, z in Z.items()}
    Zmir = {ds: l2(z) for ds, z in Zmir.items()}
    sc = {ds: n[ds].superclass.astype(str).to_numpy() for ds in n}
    side = {ds: n[ds].side.astype(str).to_numpy() for ds in n}
    # cross-dataset, both directions
    res_u, res_f = [], []
    for qd, gd, qc, gc in (("mcns", "banc", "mcns_uid", "banc_uid"), ("banc", "mcns", "banc_uid", "mcns_uid")):
        gal = np.union1d(np.where(include[gd])[0], pairs_cross[gc].to_numpy())
        q, p = pairs_cross[qc].to_numpy(), pairs_cross[gc].to_numpy()
        key_g = np.char.add(sc[gd][gal].astype(str), side[gd][gal].astype(str))
        key_q = np.char.add(sc[qd][q].astype(str), side[qd][q].astype(str))
        u, f = ranks(Z[qd], q, Z[gd], gal, p, key_g, key_q)
        res_u.append(u); res_f.append(f)
    u, f = np.concatenate(res_u), np.concatenate(res_f)
    rows.append({"method": name, "seed": seed, "setting": "cross", "dataset": "both", "n_queries": len(u),
                 **summarise(u), **{f"{k}_filtered": v for k, v in summarise(f).items()}})
    # within-dataset left <-> right; gallery embedded from mirrored features
    for ds, wp in within_pairs.items():
        res_u, res_f = [], []
        for qs, gs in (("left", "right"), ("right", "left")):
            gal = np.union1d(np.where(include[ds] & (side[ds] == gs))[0], wp[gs].to_numpy())
            q, p = wp[qs].to_numpy(), wp[gs].to_numpy()
            u, f = ranks(Z[ds], q, Zmir[ds], gal, p, sc[ds][gal], sc[ds][q])
            res_u.append(u); res_f.append(f)
        u, f = np.concatenate(res_u), np.concatenate(res_f)
        rows.append({"method": name, "seed": seed, "setting": "within", "dataset": ds, "n_queries": len(u),
                     **summarise(u), **{f"{k}_filtered": v for k, v in summarise(f).items()}})
    return rows


def latest(method, seed):
    found = [d for d in RUNS.glob(f"*_{method}_seed{seed}_*") if (d / "model.pt").exists() and (d / "metrics.json").exists()]
    return max(found, key=lambda d: (d / "metrics.json").stat().st_mtime) if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["B4", "gnn_L2"])
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    layout = Layout(load_data_config())
    splits = load_splits(layout)
    val = set(splits["val"])
    neurons = pd.read_parquet(layout.neurons)
    n = {ds: neurons[neurons.dataset == ds].sort_values("uid").reset_index(drop=True) for ds in ("mcns", "banc")}
    include = {ds: n[ds].include.to_numpy() for ds in n}
    pairs = pd.read_parquet(layout.gt_pairs)
    pairs_cross = pairs[pairs.family.isin(val)]
    within_pairs = {}
    for ds in n:
        d = n[ds]
        d = d[d.include & d.family.isin(val) & d.side.isin(["left", "right"]) & d.type_h.notna()]
        c = d.groupby(["type_h", "side"], observed=True).uid.agg(["count", "first"]).unstack("side")
        ok = (c[("count", "left")] == 1) & (c[("count", "right")] == 1)
        within_pairs[ds] = pd.DataFrame({"left": c.loc[ok, ("first", "left")].astype(int).to_numpy(),
                                         "right": c.loc[ok, ("first", "right")].astype(int).to_numpy()})
        print(f"[c3] {ds}: {len(within_pairs[ds]):,} validation left/right 1:1 pairs")
    print(f"[c3] cross: {len(pairs_cross):,} validation 1:1 pairs")

    spec = json.loads(layout.feature_spec.read_text())
    perm = np.asarray(spec["mirror_perm"])
    X = {ds: np.load(layout.features(ds)) for ds in n}
    rows = []
    # B1: neuropil in/out profile cosine, no learning
    blk = np.r_[slice(*spec["blocks"]["np_in"]), slice(*spec["blocks"]["np_out"])]
    rows += evaluate_pair("B1", 0, {ds: X[ds][:, blk] for ds in n}, {ds: X[ds][:, perm][:, blk] for ds in n},
                          n, pairs_cross, within_pairs, include)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    del X
    # trained encoders: embed with normal and with mirrored node features
    for method in a.runs:
        for s in [int(x) for x in a.seeds.split(",")]:
            run = latest(method, s)
            if run is None:
                print(f"[c3] {method} seed {s}: no run"); continue
            Z = embed_run(layout, run)
            Zmir = embed_run(layout, run, mirror=True)
            new = evaluate_pair(method, s, Z, Zmir, n, pairs_cross, within_pairs, include)
            rows += new
            print(pd.DataFrame(new).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            del Z, Zmir
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "c3_mirror.csv", index=False)
    g = df.groupby(["method", "setting", "dataset"], sort=False)[["R@1", "R@1_filtered", "MRR"]].agg(["mean", "std"])
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
