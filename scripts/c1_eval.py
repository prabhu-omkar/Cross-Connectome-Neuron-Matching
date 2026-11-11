"""Control C1 / H4: zero-shot hemibrain matching with models trained on BANC <-> MaleCNS (data_c1 variant).

    python scripts/c1_eval.py [--runs mlp_c1 gnn_c1] [--seeds 0,1,2] [--split val]

Pairs: data_c1/processed/hemibrain/gt_pairs_{mcns,banc}.parquet, restricted to the split's families and to
neurons with synapses in the kept neuropils on both sides. Gallery: every neuron of the other dataset on the
query's side with synapses in the kept neuropils (MaleCNS/BANC: also `include`). Both directions.
Rank = 1 + #(scores > partner) + 0.5 * #ties.
B1 = cosine of the np_in/np_out block of the C1 features (no learning; identical to scripts/hemibrain_check.py).
H4 (plan): a model that learned homology, not imaging, keeps its gain over B1 on hemibrain <-> BANC
(same sex as BANC, imaged like MaleCNS).
Output: results/c1_eval.csv (per method/seed/pair) and a printed summary.
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

from flymatch.c1 import embed_hemibrain  # noqa: E402
from flymatch.config import Layout, load_data_config  # noqa: E402
from flymatch.inference import embed_run  # noqa: E402
from flymatch.splits import load_splits  # noqa: E402

RUNS = REPO / "runs"


def l2(z):
    z = np.asarray(z, np.float32)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def ranks(Zq, q, Zg, gal, partner, block=1024):
    pos = {u: i for i, u in enumerate(gal)}
    G = Zg[gal]
    out = np.empty(len(q))
    for a in range(0, len(q), block):
        S = Zq[q[a:a + block]] @ G.T
        t = S[np.arange(S.shape[0]), [pos[p] for p in partner[a:a + block]]][:, None]
        out[a:a + block] = 1 + (S > t).sum(1) + 0.5 * ((S == t).sum(1) - 1)
    return out


def score(name, seed, Z, ctx):
    """Z: {'mcns','banc','hemibrain'} -> embeddings (any dim). Returns rows per pair type."""
    Z = {k: l2(v) for k, v in Z.items()}
    rows = []
    for ds in ("mcns", "banc"):
        p = ctx["pairs"][ds]
        rs = []
        for s in ("left", "right"):
            ps = p[p.side == s]
            if ps.empty:
                continue
            g_ds = np.union1d(np.where(ctx["ok"][ds] & (ctx["side"][ds] == s))[0], ps[f"{ds}_uid"])
            g_hb = np.union1d(np.where(ctx["ok"]["hemibrain"] & (ctx["side"]["hemibrain"] == s))[0], ps.hemibrain_uid)
            rs.append(ranks(Z["hemibrain"], ps.hemibrain_uid.to_numpy(), Z[ds], g_ds, ps[f"{ds}_uid"].to_numpy()))
            rs.append(ranks(Z[ds], ps[f"{ds}_uid"].to_numpy(), Z["hemibrain"], g_hb, ps.hemibrain_uid.to_numpy()))
        r = np.concatenate(rs)
        rows.append({"method": name, "seed": seed, "pair": f"hemibrain-{ds}", "n_queries": len(r),
                     "R@1": float((r <= 1).mean()), "R@10": float((r <= 10).mean()), "MRR": float((1 / r).mean()),
                     "median_rank": float(np.median(r))})
    return rows


def latest(method, seed):
    found = [d for d in RUNS.glob(f"*_{method}_seed{seed}_*") if (d / "model.pt").exists() and (d / "metrics.json").exists()]
    return max(found, key=lambda d: (d / "metrics.json").stat().st_mtime) if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["mlp_c1", "gnn_c1"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    a = ap.parse_args()
    if a.split == "test":
        print("[c1] WARNING: the test split is touched once, at the end (plan 9)")
    c1 = Layout(load_data_config(REPO / "configs" / "data_c1.yaml"))
    spec = json.loads(c1.feature_spec.read_text())
    npb = np.r_[slice(*spec["blocks"]["np_in"]), slice(*spec["blocks"]["np_out"])]
    X = {ds: np.load(c1.features(ds)) for ds in ("mcns", "banc")}
    X["hemibrain"] = np.load(c1.processed / "hemibrain" / "features.npy")
    neurons = pd.read_parquet(c1.neurons, columns=["dataset", "uid", "side", "include"])
    hb = pd.read_parquet(c1.processed / "hemibrain" / "neurons.parquet", columns=["uid", "side"])
    ctx = {"side": {"hemibrain": hb.side.astype(str).to_numpy()}, "ok": {"hemibrain": X["hemibrain"][:, npb].any(1)},
           "pairs": {}}
    for ds in ("mcns", "banc"):
        n = neurons[neurons.dataset == ds].sort_values("uid")
        ctx["side"][ds] = n.side.astype(str).to_numpy()
        ctx["ok"][ds] = n.include.to_numpy() & X[ds][:, npb].any(1)
    fams = set(load_splits(c1)[a.split])
    for ds in ("mcns", "banc"):
        p = pd.read_parquet(c1.processed / "hemibrain" / f"gt_pairs_{ds}.parquet")
        p = p[p.family.isin(fams) & ctx["ok"]["hemibrain"][p.hemibrain_uid] & ctx["ok"][ds][p[f"{ds}_uid"]]]
        ctx["pairs"][ds] = p
        print(f"[c1] {a.split} pairs hemibrain-{ds}: {len(p):,}")

    rows = score("B1", 0, {k: v[:, npb] for k, v in X.items()}, ctx)
    del X
    for method in a.runs:
        for s in [int(x) for x in a.seeds.split(",")]:
            run = latest(method, s)
            if run is None:
                print(f"[c1] {method} seed {s}: no finished run")
                continue
            Z = embed_run(c1, run)
            Z["hemibrain"] = embed_hemibrain(c1, run)
            rows += score(method, s, Z, ctx)
            print(f"[c1] {method} seed {s}: done ({run.name})")
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / f"c1_eval{'' if a.split == 'val' else '_' + a.split}.csv", index=False)
    g = df.groupby(["pair", "method"], sort=False)[["R@1", "MRR"]].agg(["mean", "std", "count"])
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
