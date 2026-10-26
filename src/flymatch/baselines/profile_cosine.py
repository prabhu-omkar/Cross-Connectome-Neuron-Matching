"""B1 — cosine of sqrt neuropil in/out profiles (zero learning). plan.md 7.2.3.

Week-4 use: reproduce the 21 Oct feasibility number (R@1 29.3%) on the harmonised
features. Evaluated on train + val 1:1 pairs only — test is not touched before Phase 4.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Layout, load_data_config, write_json
from ..retrieval import ranks_of_true, recall_metrics
from ..splits import family_to_split, load_splits

VARIANTS = {"B1": ["np_in", "np_out"], "B1+region": ["np_in", "np_out", "region_in", "region_out"]}


def block_features(X: np.ndarray, spec: dict, blocks: list[str]) -> np.ndarray:
    return np.concatenate([X[:, slice(*spec["blocks"][b])] for b in blocks], axis=1)


def make_filter(qn: pd.DataFrame, gn: pd.DataFrame):
    """Hard filters: same harmonised super class; same side or either 'center'."""
    qs, gs = qn.superclass.astype(str).to_numpy(), gn.superclass.astype(str).to_numpy()
    qd, gd = qn.side.astype(str).to_numpy(), gn.side.astype(str).to_numpy()

    def allowed(rows):
        a = qs[rows][:, None] == gs[None, :]
        side = (qd[rows][:, None] == gd[None, :]) | (qd[rows][:, None] == "center") | (gd[None, :] == "center")
        return a & side
    return allowed


def run(cfg: dict | None = None, splits_to_eval=("train", "val")):
    cfg = cfg or load_data_config()
    layout = Layout(cfg)
    spec = json.loads(layout.feature_spec.read_text())
    splits = load_splits(layout)
    f2s = family_to_split(splits)
    neurons = pd.read_parquet(layout.neurons)
    pairs = pd.read_parquet(layout.gt_pairs)
    pairs["split"] = pairs.family.map(f2s)
    pairs = pairs[pairs.split.isin(splits_to_eval)].reset_index(drop=True)
    X = {ds: np.load(layout.features(ds)) for ds in ("mcns", "banc")}
    N = {ds: neurons[neurons.dataset == ds].sort_values("uid").reset_index(drop=True) for ds in ("mcns", "banc")}
    rng = np.random.default_rng(0)
    rows = []
    for vname, blocks in VARIANTS.items():
        F = {ds: block_features(X[ds], spec, blocks) for ds in X}
        for qd, gd in (("mcns", "banc"), ("banc", "mcns")):
            for gallery in ("all", "included"):
                gmask = np.ones(len(N[gd]), bool) if gallery == "all" else N[gd].include.to_numpy()
                gidx = np.where(gmask)[0]
                pos = np.full(len(N[gd]), -1)
                pos[gidx] = np.arange(len(gidx))
                for filt in (False, True):
                    q = pairs[f"{qd}_uid"].to_numpy()
                    t = pos[pairs[f"{gd}_uid"].to_numpy()]
                    allowed = make_filter(N[qd].iloc[q].reset_index(drop=True), N[gd].iloc[gidx]) if filt else None
                    r, _ = ranks_of_true(F[qd][q], F[gd][gidx], t, allowed=allowed)
                    for split in splits_to_eval:
                        m = (pairs.split == split).to_numpy()
                        rows.append({"variant": vname, "direction": f"{qd}->{gd}", "gallery": gallery,
                                     "gallery_size": len(gidx), "filtered": filt, "split": split,
                                     **recall_metrics(r[m])})
                    if vname == "B1" and gallery == "all" and not filt:
                        rs, _ = ranks_of_true(F[qd][q], F[gd][gidx], rng.permutation(t))
                        rows.append({"variant": "B1-shuffled", "direction": f"{qd}->{gd}", "gallery": gallery,
                                     "gallery_size": len(gidx), "filtered": False, "split": "+".join(splits_to_eval),
                                     **recall_metrics(rs)})
    res = pd.DataFrame(rows)
    res.to_csv(layout.results / "b1_profile_cosine.csv", index=False)
    key = res[(res.variant == "B1") & (res.direction == "mcns->banc") & (res.gallery == "all") & ~res.filtered]
    write_json({"splits_sha256": splits["sha256"], "headline": key.to_dict("records")},
               layout.results / "b1_profile_cosine.json")
    show = res[res.split.isin(["val", "train+val"]) | (res.split == "train")]
    cols = ["variant", "direction", "gallery", "filtered", "split", "R@1", "R@10", "R@100", "MRR", "median_rank", "n"]
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return res


if __name__ == "__main__":
    run()
