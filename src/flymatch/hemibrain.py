"""Hemibrain v1.2.1 for control C1 / hypothesis H4 (plan 7.4.2): does the model transfer to a dataset of
the same sex as BANC but imaged like MaleCNS?

    python -m flymatch.run hemibrain-download     # neuPrint -> data/external/hemibrain_v1.2.1/ (needs the token)
    python -m flymatch.run hemibrain-prepare      # completeness + ground truth -> results/, data/processed/

Download (raw, unmodified neuPrint answers):
  neurons.parquet      every Traced neuron: bodyId, instance, type, status, cropped, pre, post, cellBodyFiber,
                       somaLocation (no roiInfo - it is large and the ROI-split connections carry the same counts)
  connections_roi.parquet  Traced -> Traced pairs with >= 5 synapses in total (the same threshold as the
                       harmonised MaleCNS/BANC graphs), split by PRIMARY ROI: bodyId_pre, bodyId_post, roi, weight
  manifest.json        dataset, neuprint-python version, date, row counts
The token is read from NEUPRINT_APPLICATION_CREDENTIALS (Windows user env var; never stored in the repo).
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Layout, write_json

DATASET = "hemibrain:v1.2.1"
MIN_WEIGHT = 5


def out_dir(layout: Layout) -> Path:
    return layout.root / "external" / "hemibrain_v1.2.1"     # git-ignored, like the other external data


def _client():
    from neuprint import Client
    tok = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not tok and os.name == "nt":       # processes started before `setx` do not see the user variable
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            tok = winreg.QueryValueEx(k, "NEUPRINT_APPLICATION_CREDENTIALS")[0]
    if not tok:
        raise SystemExit("NEUPRINT_APPLICATION_CREDENTIALS is not set (see external/SOURCES.md)")
    return Client("neuprint.janelia.org", dataset=DATASET, token=tok)


def download(layout: Layout, log=print):
    import neuprint
    from neuprint import NeuronCriteria as NC, fetch_adjacencies
    d = out_dir(layout)
    d.mkdir(parents=True, exist_ok=True)
    c = _client()
    q = """MATCH (n:Neuron) WHERE n.status = 'Traced'
           RETURN n.bodyId AS bodyId, n.instance AS instance, n.type AS type, n.status AS status,
                  n.cropped AS cropped, n.pre AS pre, n.post AS post, n.cellBodyFiber AS cellBodyFiber,
                  n.somaLocation AS somaLocation"""
    neurons = c.fetch_custom(q)
    neurons["bodyId"] = neurons.bodyId.astype("int64")
    neurons["somaLocation"] = neurons.somaLocation.map(lambda p: None if p is None else str(p))
    neurons.to_parquet(d / "neurons.parquet", index=False)
    log(f"[hemibrain] {len(neurons):,} traced neurons ({neurons.type.notna().sum():,} typed)")
    traced = NC(status="Traced")
    _, conn = fetch_adjacencies(traced, traced, min_total_weight=MIN_WEIGHT, include_nonprimary=False, client=c)
    conn = conn.astype({"bodyId_pre": "int64", "bodyId_post": "int64", "weight": "int32"})
    conn.to_parquet(d / "connections_roi.parquet", index=False)
    pairs = conn.groupby(["bodyId_pre", "bodyId_post"]).size().shape[0]
    log(f"[hemibrain] {len(conn):,} ROI-split rows, {pairs:,} pairs >= {MIN_WEIGHT} synapses, "
        f"{conn.weight.sum():,} synapses")
    write_json({"dataset": DATASET, "server": "neuprint.janelia.org", "neuprint_python": neuprint.__version__,
                "downloaded": dt.datetime.now().isoformat(timespec="seconds"),
                "neurons_query": "status = 'Traced'", "connections": f"Traced->Traced, total weight >= {MIN_WEIGHT}, "
                "primary ROIs only (rows with roi 'NotPrimary' are synapses outside every primary ROI)",
                "n_neurons": int(len(neurons)), "n_conn_rows": int(len(conn)), "n_pairs": int(pairs),
                "n_synapses": int(conn.weight.sum())}, d / "manifest.json")


def load(layout: Layout):
    d = out_dir(layout)
    return pd.read_parquet(d / "neurons.parquet"), pd.read_parquet(d / "connections_roi.parquet")


def roi_map(layout: Layout) -> dict:
    m = pd.read_csv(Path(layout.maps) / "hemibrain_neuropil_map.csv")
    return dict(zip(m.raw, m.canonical))


def side_of(instance) -> str:
    """Hemibrain instance names end in _R / _L (e.g. MBON01(y5B'2a)_R); otherwise unknown."""
    if isinstance(instance, str):
        if instance.endswith("_R"):
            return "right"
        if instance.endswith("_L"):
            return "left"
    return "unknown"


def link_types(layout: Layout, hb: pd.DataFrame) -> pd.DataFrame:
    """Hemibrain type -> harmonised type (type_h), family and super class, two sources in order:
      1. `label`: MaleCNS community label `hemibrain_type` (plan 7.4.2). Linked only when every labelled
         MaleCNS neuron agrees on ONE type_h.
      2. `name`:  the hemibrain type name equals a MaleCNS type_h (MaleCNS adopted most hemibrain names;
         the label mostly exists where the names differ).
    A type whose label points to a different type_h than its own name is `conflict` and left unlinked.
    Super class = MaleCNS majority for that type_h."""
    n = pd.read_parquet(layout.neurons, columns=["dataset", "hemibrain_type", "type_h", "family", "superclass"])
    mc = n[(n.dataset == "mcns") & n.type_h.notna()]
    by_th = mc.groupby("type_h").agg(family=("family", "first"),
                                     superclass=("superclass", lambda s: s.astype(str).value_counts().idxmax()))
    lab = mc[mc.hemibrain_type.notna()].groupby("hemibrain_type").type_h.agg(["nunique", "first"])
    types = pd.DataFrame({"hemibrain_type": sorted(hb.type.dropna().unique())})
    via_label = types.hemibrain_type.map(lab["first"].where(lab["nunique"] == 1))
    ambiguous = types.hemibrain_type.map(lab["nunique"] > 1).fillna(False).astype(bool)
    via_name = types.hemibrain_type.where(types.hemibrain_type.isin(by_th.index))
    conflict = via_label.notna() & via_name.notna() & (via_label != via_name)
    types["type_h"] = via_label.fillna(via_name)
    types["source"] = np.select([conflict, via_label.notna(), ambiguous, via_name.notna()],
                                ["conflict", "label", "ambiguous", "name"], "unlinked")
    types.loc[types.source.isin(["conflict", "ambiguous", "unlinked"]), "type_h"] = None
    types["status"] = np.where(types.type_h.notna(), "linked", types.source)
    types = types.join(by_th, on="type_h")
    return types


def pairs_1to1(a: pd.DataFrame, b: pd.DataFrame, key="type_h") -> pd.DataFrame:
    """1:1 pairs: same key and side, exactly one included neuron on each side of the pair."""
    ca = a.groupby([key, "side"]).uid.agg(["count", "first"])
    cb = b.groupby([key, "side"]).uid.agg(["count", "first"])
    j = ca.join(cb, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[(j.count_a == 1) & (j.count_b == 1)]
    return j.reset_index().rename(columns={"first_a": "uid_a", "first_b": "uid_b"})[[key, "side", "uid_a", "uid_b"]]


def prepare(layout: Layout, log=print):
    """Harmonised hemibrain tables + neuropil completeness + ground-truth pairs (no features, no training).

    data/processed/hemibrain/neurons.parquet   uid (0..N-1), bodyId (int64), type, side, cropped, link columns
    data/processed/hemibrain/edges_np.parquet  pre_uid, post_uid, neuropil (canonical), syn  (pairs >= 5 in total)
    data/processed/hemibrain/edges.parquet     pre_uid, post_uid, syn  (all synapses of the pair, any ROI)
    results/hemibrain_neuropil_completeness.csv, results/hemibrain_type_link.csv, results/hemibrain_gt_summary.json
    data/processed/hemibrain/gt_pairs_{mcns,banc}.parquet  1:1 pairs with family and split
    """
    from .splits import load_splits
    hb, conn = load(layout)
    d = layout.processed / "hemibrain"
    d.mkdir(parents=True, exist_ok=True)
    hb = hb.sort_values("bodyId").reset_index(drop=True)
    hb["uid"] = np.arange(len(hb))
    hb["side"] = hb.instance.map(side_of)
    link = link_types(layout, hb)
    link.to_csv(layout.results / "hemibrain_type_link.csv", index=False)
    hb = hb.merge(link[["hemibrain_type", "type_h", "family", "superclass", "status", "source"]].rename(
        columns={"hemibrain_type": "type", "status": "link_status", "source": "link_source"}), on="type", how="left")
    hb.to_parquet(d / "neurons.parquet", index=False)

    uid = pd.Series(hb.uid.to_numpy(), index=hb.bodyId.to_numpy())
    rm = roi_map(layout)
    c = conn.assign(pre_uid=conn.bodyId_pre.map(uid), post_uid=conn.bodyId_post.map(uid))
    assert c.pre_uid.notna().all() and c.post_uid.notna().all(), "connection to a non-Traced body"
    c = c.astype({"pre_uid": "int64", "post_uid": "int64"})
    c.groupby(["pre_uid", "post_uid"], as_index=False).weight.sum().rename(columns={"weight": "syn"}) \
        .to_parquet(d / "edges.parquet", index=False)
    enp = c.assign(neuropil=c.roi.map(rm)).dropna(subset=["neuropil"])
    enp = enp.groupby(["pre_uid", "post_uid", "neuropil"], as_index=False).weight.sum().rename(columns={"weight": "syn"})
    enp.to_parquet(d / "edges_np.parquet", index=False)
    log(f"[hemibrain] {len(hb):,} neurons, {c.groupby(['pre_uid', 'post_uid']).ngroups:,} pairs, "
        f"{len(enp):,} neuropil-split rows in canonical neuropils")

    comp = completeness(layout, conn)
    comp.to_csv(layout.results / "hemibrain_neuropil_completeness.csv", index=False)
    log(f"[hemibrain] neuropils inside the volume: {int(comp.keep.sum())} of {len(comp)}: "
        f"{sorted(comp.neuropil[comp.keep])}")

    # ground truth: hemibrain <-> MaleCNS and hemibrain <-> BANC, 1:1 by (type_h, side)
    splits = load_splits(layout)
    split_of = {f: s for s in ("train", "val", "test") for f in splits[s]}
    n = pd.read_parquet(layout.neurons, columns=["dataset", "uid", "type_h", "side", "include", "family"])
    hb_l = hb[hb.type_h.notna() & hb.side.isin(["left", "right"])]
    summary = {"hemibrain_types": int(hb.type.nunique()),
               "type_link": link.source.value_counts().to_dict(),
               "hemibrain_neurons_by_side": hb.side.value_counts().to_dict()}
    for ds in ("mcns", "banc"):
        other = n[(n.dataset == ds) & n.include & n.type_h.notna()]
        p = pairs_1to1(hb_l, other).rename(columns={"uid_a": "hemibrain_uid", "uid_b": f"{ds}_uid"})
        p["family"] = p.type_h.map(dict(zip(n.type_h, n.family)))
        p["split"] = p.family.map(split_of).fillna("none")
        p.to_parquet(d / f"gt_pairs_{ds}.parquet", index=False)
        summary[f"pairs_{ds}"] = p.split.value_counts().to_dict()
        log(f"[hemibrain] 1:1 pairs hemibrain<->{ds}: {len(p):,} ({p.split.value_counts().to_dict()})")
    write_json(summary, layout.results / "hemibrain_gt_summary.json")
    return summary


def completeness(layout: Layout, conn: pd.DataFrame, ref=("EB", "FB", "PB", "NO"),
                 inside_at=0.25, left_vs_right_at=0.7) -> pd.DataFrame:
    """Which canonical neuropils lie inside the hemibrain volume (not cut by its boundary)?

    ratio = hemibrain synapses in the neuropil (Traced -> Traced, pairs >= 5) / MaleCNS synapses in it, divided by
    the median ratio of the central complex (fully inside the volume) -> `relative_completeness`.
    This mixes two things: hemibrain's lower TRACING completeness (only traced partners count; it varies by
    neuropil, ~0.3-1.8 on the right) and spatial TRUNCATION by the volume boundary (optic lobe, GNG, SAD, most of
    the left: < 0.2). Only truncation separates hemibrain by construction (plan 7.4.2), so the rule is:
      right / midline neuropil:  inside if relative_completeness >= inside_at (0.25: the gap between AVLP_R 0.31
                                 and the truncated group <= 0.19 in hemibrain v1.2.1);
      left neuropil:             inside if it reaches >= left_vs_right_at of its OWN right mirror in hemibrain
                                 (medial left neuropils are cut by the boundary: e.g. SMP_L ~0.5 of SMP_R)."""
    rm = roi_map(layout)
    hb = conn.assign(np_=conn.roi.map(rm)).dropna(subset=["np_"]).groupby("np_").weight.sum()
    enp = pd.read_parquet(layout.edges_np, columns=["dataset", "neuropil", "syn"])
    mc = enp[enp.dataset == "mcns"].groupby("neuropil").syn.sum()
    df = pd.DataFrame({"hemibrain_syn": hb, "mcns_syn": mc}).fillna(0)
    df = df[df.hemibrain_syn > 0]
    df["ratio"] = df.hemibrain_syn / df.mcns_syn.replace(0, np.nan)
    scale = df.loc[[r for r in ref if r in df.index], "ratio"].median()
    df["relative_completeness"] = df.ratio / scale
    rc = df.relative_completeness
    left = df.index.str.endswith("_L")
    mirror = pd.Series(df.index.str.replace(r"_L$", "_R", regex=True), index=df.index)
    df["left_vs_right"] = np.where(left, rc.to_numpy() / mirror.map(rc).to_numpy(), np.nan)
    df["keep"] = np.where(left, df.left_vs_right >= left_vs_right_at, rc >= inside_at)
    df["rule"] = np.where(left, f"left: >= {left_vs_right_at} of own right mirror", f"right/midline: >= {inside_at}")
    return df.sort_values("relative_completeness", ascending=False).rename_axis("neuropil").reset_index()
