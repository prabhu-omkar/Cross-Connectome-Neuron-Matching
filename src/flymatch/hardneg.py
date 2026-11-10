"""Hard negatives from the B1 shortlist (plan.md 7.3.3 step 3; week 9).

For every training anchor (an included neuron of a SEED type) we keep its top-`k` B1 neighbours
in the other dataset that are (a) neurons of seed types — so they come from training families
only and we know they are negatives — (b) of a DIFFERENT type, and (c) allowed by the hard
filters (same super class; same side or either 'center'). k = k_neg = 50 from week 5
(B1 filtered Recall@50 on training pairs = 86%). During training, `per_anchor` of them are
sampled per anchor per step.

Val/test labels are never touched: candidates are seed-type neurons only.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

B1_BLOCKS = ("np_in", "np_out")


def _b1(layout) -> dict:
    spec = json.loads(layout.feature_spec.read_text())
    out = {}
    for ds in ("mcns", "banc"):
        X = np.load(layout.features(ds), mmap_mode="r")
        F = np.concatenate([np.asarray(X[:, slice(*spec["blocks"][b])]) for b in B1_BLOCKS], axis=1)
        F = F / np.clip(np.linalg.norm(F, axis=1, keepdims=True), 1e-12, None)
        out[ds] = F.astype(np.float32)
    return out


def build_shortlist(layout, groups: dict, type_code: dict, k: int = 50, device="cpu", block: int = 2048) -> dict:
    """groups[ds]: {(type, side): uids} of seed-type included neurons (TrainData.groups).
    Returns {ds: (row_of [N_ds] int64 (-1 = not an anchor), table [n_anchor, k] int64 uids of the
    OTHER dataset, -1 padded)}."""
    neurons = pd.read_parquet(layout.neurons, columns=["dataset", "uid", "superclass", "side"])
    sc_codes = {s: i for i, s in enumerate(sorted(neurons.superclass.astype(str).unique()))}
    side_codes = {"left": 0, "right": 1, "center": 2, "unknown": 3}
    meta = {}
    for ds in ("mcns", "banc"):
        n = neurons[neurons.dataset == ds].sort_values("uid")
        meta[ds] = (torch.as_tensor(n.superclass.astype(str).map(sc_codes).to_numpy()),
                    torch.as_tensor(n.side.astype(str).map(side_codes).to_numpy()),
                    torch.as_tensor(type_code[ds]))
    F = _b1(layout)
    dev = torch.device(device)
    out = {}
    for ds, od in (("mcns", "banc"), ("banc", "mcns")):
        anchors = np.sort(np.concatenate(list(groups[ds].values())))
        cands = np.sort(np.concatenate(list(groups[od].values())))
        A = torch.as_tensor(F[ds][anchors], device=dev)
        C = torch.as_tensor(F[od][cands], device=dev)
        a_sc, a_side, a_t = (m[torch.as_tensor(anchors)].to(dev) for m in meta[ds])
        c_sc, c_side, c_t = (m[torch.as_tensor(cands)].to(dev) for m in meta[od])
        kk = min(k, len(cands))
        table = np.full((len(anchors), k), -1, np.int64)
        cands_t = torch.as_tensor(cands, device=dev)
        for s in range(0, len(anchors), block):
            e = slice(s, min(s + block, len(anchors)))
            S = A[e] @ C.T
            ok = (a_sc[e, None] == c_sc[None, :]) & (a_t[e, None] != c_t[None, :])
            ok &= (a_side[e, None] == c_side[None, :]) | (a_side[e, None] == 2) | (c_side[None, :] == 2)
            S = S.masked_fill(~ok, float("-inf"))
            val, idx = S.topk(kk, dim=1)
            uid = torch.where(torch.isfinite(val), cands_t[idx], torch.full_like(idx, -1))
            table[e, :kk] = uid.cpu().numpy()
        row_of = np.full(len(type_code[ds]), -1, np.int64)
        row_of[anchors] = np.arange(len(anchors))
        out[ds] = (row_of, table)
    return out


def sample(shortlist: dict, batch: dict, per_anchor: int, max_total: int, rng) -> dict:
    """Hard negatives to ADD to each dataset's batch: drawn from the shortlists of the OTHER
    dataset's anchors. Returns {ds: uids}."""
    add = {}
    for ds, od in (("mcns", "banc"), ("banc", "mcns")):
        row_of, table = shortlist[od]
        rows = row_of[batch[od]]
        rows = rows[rows >= 0]
        if len(rows) == 0 or per_anchor <= 0:
            add[ds] = np.empty(0, np.int64)
            continue
        cols = rng.integers(0, table.shape[1], (len(rows), per_anchor))
        picks = table[rows[:, None], cols].ravel()
        picks = np.unique(picks[picks >= 0])
        if len(picks) > max_total:
            picks = rng.choice(picks, max_total, replace=False)
        add[ds] = picks
    return add
