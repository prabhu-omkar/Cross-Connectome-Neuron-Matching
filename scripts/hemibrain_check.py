"""Validate the hemibrain type links and give the C1 / H4 B1 baseline.

    python scripts/hemibrain_check.py

B1 = cosine of sqrt in/out neuropil profiles, restricted to the neuropils inside the hemibrain volume
(results/hemibrain_neuropil_completeness.csv, keep == True) and renormalised, for all three datasets.
For every 1:1 pair (data/processed/hemibrain/gt_pairs_{mcns,banc}.parquet): rank of the partner among all
included neurons of the other dataset on the same side (ties averaged), both directions.
Split by link source (`label` = MaleCNS hemibrain_type community label, `name` = identical type name):
if name links were wrong, their R@k would be far below the label links'.
Output: results/hemibrain_link_check.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flymatch.config import Layout, load_data_config  # noqa: E402


def profiles(enp: pd.DataFrame, n: int, nps: list[str]) -> np.ndarray:
    col = enp.neuropil.map({p: i for i, p in enumerate(nps)})
    e = enp[col.notna()]
    col = col[col.notna()].astype(int).to_numpy()
    K = len(nps)
    out = np.bincount(e.pre_uid.to_numpy() * K + col, weights=e.syn.to_numpy(float), minlength=n * K).reshape(n, K)
    inn = np.bincount(e.post_uid.to_numpy() * K + col, weights=e.syn.to_numpy(float), minlength=n * K).reshape(n, K)
    norm = lambda a: np.sqrt(a / np.clip(a.sum(1, keepdims=True), 1, None))
    X = np.hstack([norm(inn), norm(out)]).astype(np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def rank_of(Xq, q, Xg, gal, partner):
    S = Xq[q] @ Xg[gal].T
    pos = {u: i for i, u in enumerate(gal)}
    s_true = S[np.arange(len(q)), [pos[p] for p in partner]][:, None]
    return 1 + (S > s_true).sum(1) + 0.5 * ((S == s_true).sum(1) - 1)


def main():
    L = Layout(load_data_config())
    d = L.processed / "hemibrain"
    comp = pd.read_csv(L.results / "hemibrain_neuropil_completeness.csv")
    nps = sorted(comp.neuropil[comp.keep])
    hb = pd.read_parquet(d / "neurons.parquet")
    X = {"hemibrain": profiles(pd.read_parquet(d / "edges_np.parquet"), len(hb), nps)}
    side = {"hemibrain": hb.side.to_numpy()}
    neurons = pd.read_parquet(L.neurons, columns=["dataset", "uid", "side", "include"])
    enp = pd.read_parquet(L.edges_np, columns=["dataset", "pre_uid", "post_uid", "neuropil", "syn"])
    inc = {}
    for ds in ("mcns", "banc"):
        n = neurons[neurons.dataset == ds].sort_values("uid")
        X[ds] = profiles(enp[enp.dataset == ds], len(n), nps)
        side[ds], inc[ds] = n.side.astype(str).to_numpy(), n.include.to_numpy()
    del enp
    has_np = X["hemibrain"].any(1)
    rows = []
    for ds in ("mcns", "banc"):
        p = pd.read_parquet(d / f"gt_pairs_{ds}.parquet").merge(
            hb[["uid", "link_source"]].rename(columns={"uid": "hemibrain_uid"}), on="hemibrain_uid")
        p = p[has_np[p.hemibrain_uid]]          # hemibrain neuron has synapses in the kept neuropils
        for s in ("left", "right"):
            ps = p[p.side == s]
            if ps.empty:
                continue
            g_ds = np.where(inc[ds] & (side[ds] == s))[0]
            g_hb = np.where((side["hemibrain"] == s) & has_np)[0]
            r1 = rank_of(X["hemibrain"], ps.hemibrain_uid.to_numpy(), X[ds], np.union1d(g_ds, ps[f"{ds}_uid"]),
                         ps[f"{ds}_uid"].to_numpy())
            r2 = rank_of(X[ds], ps[f"{ds}_uid"].to_numpy(), X["hemibrain"], np.union1d(g_hb, ps.hemibrain_uid),
                         ps.hemibrain_uid.to_numpy())
            for direction, r in ((f"hemibrain->{ds}", r1), (f"{ds}->hemibrain", r2)):
                rows.append(pd.DataFrame({"pair": f"hemibrain-{ds}", "direction": direction, "side": s,
                                          "source": ps.link_source.to_numpy(), "split": ps.split.to_numpy(), "rank": r}))
    df = pd.concat(rows)
    summ = df.groupby(["pair", "source"]).agg(n=("rank", "size"), R1=("rank", lambda r: (r <= 1).mean()),
                                              R10=("rank", lambda r: (r <= 10).mean()),
                                              MRR=("rank", lambda r: (1 / r).mean()),
                                              median_rank=("rank", "median")).reset_index()
    by_split = df.groupby(["pair", "split"]).agg(n=("rank", "size"), R1=("rank", lambda r: (r <= 1).mean()),
                                                 MRR=("rank", lambda r: (1 / r).mean())).reset_index()
    summ.to_csv(L.results / "hemibrain_link_check.csv", index=False)
    print(f"{len(nps)} neuropils inside the hemibrain volume")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(by_split.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
