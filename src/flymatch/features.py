"""Phase 1.2-1.3 — node features (plan.md 7.1.5).

Writes processed/features_{mcns,banc}.npy (float32 [N, F], row i = uid i) with identical
column order, plus feature_spec.json describing each block and the L<->R mirror permutation.
The seed-dependent label-projected block is NOT built here (it depends on the split and
seed fraction; see baselines/cocoa_style.py in Phase 2).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import DATASETS
from .config import Layout, load_data_config, write_json
from .io import read_map

REGIONS = ["OL", "CB", "GNG", "VNC", "nerve", "unassigned"]
NTS = ["ACH", "GLUT", "GABA", "DA", "HIST", "SER", "OCT", "TYR", "unknown"]
SIDES = ["left", "right", "center"]


def canonical_neuropils(layout: Layout) -> tuple[list[str], dict]:
    m = read_map(layout.maps / "neuropil_map.csv")
    inc = m[m.include.astype(int) == 1]
    per_ds = inc.groupby("canonical").dataset.nunique()
    if (per_ds < 2).any():
        raise ValueError(f"canonical neuropils present in one dataset only: {list(per_ds[per_ds < 2].index)}")
    names = sorted(inc.canonical.unique())
    region = dict(zip(m.canonical, m.region))
    return names, region


def neuropil_profiles(enp: pd.DataFrame, n_nodes: int, neuropils: list[str]):
    """Fraction of a neuron's in/out synapses (within canonical neuropils) per neuropil, then sqrt."""
    col = enp.neuropil.astype(str).map({n: i for i, n in enumerate(neuropils)}).fillna(-1).astype(int).to_numpy()
    keep = col >= 0
    pre, post = enp.pre_uid.to_numpy()[keep], enp.post_uid.to_numpy()[keep]
    syn, col = enp.syn.to_numpy()[keep].astype(np.float64), col[keep]
    K = len(neuropils)
    out = np.bincount(pre * K + col, weights=syn, minlength=n_nodes * K).reshape(n_nodes, K)
    inn = np.bincount(post * K + col, weights=syn, minlength=n_nodes * K).reshape(n_nodes, K)
    norm = lambda a: a / np.clip(a.sum(1, keepdims=True), 1, None)
    return np.sqrt(norm(inn)).astype(np.float32), np.sqrt(norm(out)).astype(np.float32), inn, out


def region_summary(enp: pd.DataFrame, n_nodes: int, region_of: dict):
    col = enp.neuropil.astype(str).map(region_of).map({r: i for i, r in enumerate(REGIONS)}).to_numpy()
    if np.isnan(col.astype(float)).any():
        raise ValueError("neuropil without region")
    col = col.astype(int)
    R = len(REGIONS)
    syn = enp.syn.to_numpy().astype(np.float64)
    out = np.bincount(enp.pre_uid.to_numpy() * R + col, weights=syn, minlength=n_nodes * R).reshape(n_nodes, R)
    inn = np.bincount(enp.post_uid.to_numpy() * R + col, weights=syn, minlength=n_nodes * R).reshape(n_nodes, R)
    norm = lambda a: a / np.clip(a.sum(1, keepdims=True), 1, None)
    return norm(inn).astype(np.float32), norm(out).astype(np.float32)


def rank_normalise(x: np.ndarray) -> np.ndarray:
    """Within-dataset quantile in [0, 1] (average rank for ties)."""
    r = pd.Series(x).rank(method="average").to_numpy()
    return ((r - 1) / max(len(x) - 1, 1)).astype(np.float32)


def one_hot(values: pd.Series, vocab: list[str]) -> np.ndarray:
    idx = values.astype(str).map({v: i for i, v in enumerate(vocab)}).fillna(-1).astype(int).to_numpy()
    out = np.zeros((len(values), len(vocab)), np.float32)
    ok = idx >= 0
    out[np.where(ok)[0], idx[ok]] = 1
    return out


def mirror_name(name: str) -> str:
    if re.search(r"_L$", name):
        return name[:-2] + "_R"
    if re.search(r"_R$", name):
        return name[:-2] + "_L"
    return name


def build(layout: Layout):
    neurons = pd.read_parquet(layout.neurons)
    enp_all = pd.read_parquet(layout.edges_np)
    neuropils, region_of = canonical_neuropils(layout)
    superclasses = sorted(neurons.superclass.astype(str).unique())
    superclasses = [s for s in superclasses if s != "excluded"]
    hl = {ds: set(neurons.loc[neurons.dataset == ds, "hemilineage"].dropna()) for ds in DATASETS}
    hemilineages = sorted(hl["mcns"] & hl["banc"])

    blocks = [
        ("np_in", [f"in:{n}" for n in neuropils]),
        ("np_out", [f"out:{n}" for n in neuropils]),
        ("region_in", [f"rin:{r}" for r in REGIONS]),
        ("region_out", [f"rout:{r}" for r in REGIONS]),
        ("size_rank", ["in_deg", "out_deg", "in_syn", "out_syn", "n_np_in", "n_np_out"]),
        ("nt", [f"nt:{x}" for x in NTS] + ["nt_conf"]),
        ("side", [f"side:{s}" for s in SIDES]),
        ("superclass", [f"sc:{s}" for s in superclasses]),
        ("hemilineage", [f"hl:{h}" for h in hemilineages] + ["hl:none"]),
    ]
    columns = [c for _, cols in blocks for c in cols]
    spec_blocks, start = {}, 0
    for name, cols in blocks:
        spec_blocks[name] = [start, start + len(cols)]
        start += len(cols)

    # Mirror permutation: column j of the mirrored matrix = column perm[j] of the original.
    pos = {c: i for i, c in enumerate(columns)}

    def mirrored(c):
        if c.startswith(("in:", "out:")):
            p, n = c.split(":", 1)
            return f"{p}:{mirror_name(n)}"
        return {"side:left": "side:right", "side:right": "side:left"}.get(c, c)

    mirror_perm = [pos.get(mirrored(c), i) for i, c in enumerate(columns)]

    stats = {}
    for ds in DATASETS:
        n = neurons[neurons.dataset == ds].sort_values("uid").reset_index(drop=True)
        assert (n.uid.to_numpy() == np.arange(len(n))).all()
        N = len(n)
        enp = enp_all[enp_all.dataset == ds]
        p_in, p_out, raw_in, raw_out = neuropil_profiles(enp, N, neuropils)
        r_in, r_out = region_summary(enp, N, region_of)
        size = np.stack([rank_normalise(v) for v in [
            n.in_deg.to_numpy(), n.out_deg.to_numpy(), n.in_syn.to_numpy(), n.out_syn.to_numpy(),
            (raw_in > 0).sum(1), (raw_out > 0).sum(1)]], axis=1)
        nt = np.concatenate([one_hot(n.nt, NTS), np.nan_to_num(n.nt_conf.to_numpy(np.float32))[:, None]], axis=1)
        side = one_hot(n.side, SIDES)
        sc = one_hot(n.superclass, superclasses)
        hlv = n.hemilineage.where(n.hemilineage.isin(hemilineages), "none")
        hlm = one_hot(hlv, hemilineages + ["none"])
        X = np.concatenate([p_in, p_out, r_in, r_out, size, nt, side, sc, hlm], axis=1).astype(np.float32)
        assert X.shape == (N, len(columns)), (X.shape, len(columns))
        if not np.isfinite(X).all():
            raise ValueError(f"{ds}: non-finite features")
        np.save(layout.features(ds), X)
        stats[ds] = {"n": N, "f": X.shape[1],
                     "no_canonical_input": int((p_in.sum(1) == 0).sum()),
                     "no_canonical_output": int((p_out.sum(1) == 0).sum()),
                     "hemilineage_shared_frac": float((hlv != "none").mean())}
        print(f"[features] {ds}: X {X.shape}, {stats[ds]['no_canonical_input']:,} without canonical inputs, "
              f"{stats[ds]['no_canonical_output']:,} without canonical outputs")

    spec = {"columns": columns, "blocks": spec_blocks, "neuropils": neuropils, "regions": REGIONS,
            "nts": NTS, "sides": SIDES, "superclasses": superclasses, "hemilineages": hemilineages,
            "mirror_perm": mirror_perm, "stats": stats,
            "notes": {"np_in/np_out": "sqrt of fraction of synapses in each canonical neuropil",
                      "region_*": "fraction of all synapses per region (incl. nerve/unassigned)",
                      "size_rank": "within-dataset quantile",
                      "label_proj": "seed-dependent; built per seed fraction in Phase 2"}}
    write_json(spec, layout.feature_spec)
    return spec


def run(cfg: dict | None = None):
    return build(Layout(cfg or load_data_config()))


if __name__ == "__main__":
    run()
