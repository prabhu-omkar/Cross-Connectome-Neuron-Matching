"""Phase 1.1 — raw Codex exports -> canonical parquet (plan.md 7.1.1-7.1.2).

Writes data/interim/neurons_raw.parquet and edges_raw.parquet. Fails loudly if any raw
super class, NT or neuropil value is missing from the maps in maps/.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import DATASETS
from .config import Layout, load_data_config, load_paths, write_json
from .io import label_columns, read_edges, read_map, read_neurons

RAW_FILES = {"mcns": "mcns_dir", "banc": "banc_dir"}
MISSING = "<missing>"
SIDES = {"left": "left", "right": "right", "center": "center"}


class UnmappedValueError(ValueError):
    pass


def apply_map(values: pd.Series, mapping: dict, what: str, dataset: str) -> pd.Series:
    v = values.fillna(MISSING).astype(str)
    unmapped = sorted(set(v.unique()) - set(mapping))
    if unmapped:
        raise UnmappedValueError(f"{dataset}: {len(unmapped)} {what} value(s) missing from map: {unmapped[:20]}")
    return v.map(mapping)


def load_maps(layout: Layout) -> dict:
    sc = read_map(layout.maps / "superclass_map.csv")
    np_ = read_map(layout.maps / "neuropil_map.csv")
    nt = read_map(layout.maps / "nt_map.csv")
    hl = read_map(layout.maps / "hemilineage_map.csv")
    return {
        "superclass": {ds: dict(zip(g.raw, g.harmonised)) for ds, g in sc.groupby("dataset")},
        "superclass_include": dict(zip(sc.harmonised, sc.include.astype(int).astype(bool))),
        "neuropil": {ds: g.set_index("raw") for ds, g in np_.groupby("dataset")},
        "nt": dict(zip(nt.raw, nt.nt)),
        "hemilineage": {ds: dict(zip(g.raw, g.harmonised)) for ds, g in hl.groupby("dataset")},
    }


def banc_meta(paths: dict) -> pd.DataFrame | None:
    """Optional BANC v888 metadata (Harvard Dataverse): proofread flag, status, dimorphism."""
    f = paths["external_dir"] / "banc_dataverse_v3" / "banc_888_meta.feather"
    if not f.exists():
        return None
    m = pd.read_feather(f, columns=["root_888", "proofread", "status", "sexually_dimorphic"])
    m = m.dropna(subset=["root_888"]).drop_duplicates("root_888")
    m["neuron_id"] = m["root_888"].astype(str)
    return m.drop(columns="root_888")


def ingest_neurons(ds: str, paths: dict, maps: dict, log: dict) -> pd.DataFrame:
    n = read_neurons(paths[RAW_FILES[ds]] / "neurons.csv.gz", ds)
    if n.neuron_id.duplicated().any():
        raise ValueError(f"{ds}: duplicate neuron IDs")
    if ds == "mcns":
        n = pd.concat([n, label_columns(n.labels_raw)], axis=1)
        n["banc_proofread"] = pd.NA
        n["dimorphism"] = pd.NA
    else:
        for c in ["status", "flywire_type", "manc_type", "hemibrain_type", "supertype", "manc_bodyid"]:
            n[c] = pd.NA
        meta = banc_meta(paths)
        if meta is not None:
            j = n[["neuron_id_str"]].merge(meta, left_on="neuron_id_str", right_on="neuron_id", how="left")
            n["status"] = j["status"].to_numpy()
            n["banc_proofread"] = j["proofread"].to_numpy()
            n["dimorphism"] = j["sexually_dimorphic"].to_numpy()
            log[f"{ds}_meta_coverage"] = float(j["proofread"].notna().mean())
        else:
            n["banc_proofread"] = pd.NA
            n["dimorphism"] = pd.NA
            log[f"{ds}_meta_coverage"] = 0.0

    n["side"] = n.side_raw.str.lower().map(SIDES).fillna("unknown")
    n["superclass"] = apply_map(n.superclass_raw, maps["superclass"][ds], "super class", ds)
    n["nt"] = apply_map(n.nt_raw, maps["nt"], "NT", ds)
    hl = maps["hemilineage"].get(ds, {})
    n["hemilineage"] = n.hemilineage_raw.map(lambda x: hl.get(x, x) if isinstance(x, str) else x)

    n = n.sort_values("neuron_id", kind="stable").reset_index(drop=True)
    n["uid"] = np.arange(len(n), dtype=np.int32)
    log[f"{ds}_neurons"] = len(n)
    log[f"{ds}_side_counts"] = n.side.value_counts().to_dict()
    return n


def ingest_edges(ds: str, paths: dict, maps: dict, neurons: pd.DataFrame, log: dict) -> pd.DataFrame:
    e = read_edges(paths[RAW_FILES[ds]] / "connections_princeton.csv.gz", ds)
    n_rows = len(e)
    idx = pd.Index(neurons.neuron_id.to_numpy())
    pre, post = idx.get_indexer(e.pre.to_numpy()), idx.get_indexer(e.post.to_numpy())
    ok = (pre >= 0) & (post >= 0)
    loop = ok & (pre == post)
    log[f"{ds}_edge_rows_raw"] = n_rows
    log[f"{ds}_edge_rows_dropped_unknown_endpoint"] = int((~ok).sum())
    log[f"{ds}_edge_rows_dropped_self_loop"] = int(loop.sum())
    keep = ok & ~loop
    npm = maps["neuropil"][ds]
    raw = e.neuropil_raw.astype(str)
    unmapped = sorted(set(raw.unique()) - set(npm.index))
    if unmapped:
        raise UnmappedValueError(f"{ds}: neuropil value(s) missing from map: {unmapped}")
    out = pd.DataFrame({
        "pre_uid": pre[keep].astype(np.int32),
        "post_uid": post[keep].astype(np.int32),
        "neuropil_raw": e.neuropil_raw[keep].to_numpy(),
        "syn": e.syn[keep].to_numpy(np.int32),
    })
    out["neuropil"] = out.neuropil_raw.astype(str).map(npm.canonical).astype("category")
    log[f"{ds}_edge_rows"] = len(out)
    assert len(out) == n_rows - log[f"{ds}_edge_rows_dropped_unknown_endpoint"] - log[f"{ds}_edge_rows_dropped_self_loop"]
    return out


def run(cfg: dict | None = None):
    t0 = time.time()
    cfg = cfg or load_data_config()
    paths, layout = load_paths(), Layout(cfg)
    maps = load_maps(layout)
    log: dict = {}
    neurons, edges = [], []
    for ds in DATASETS:
        n = ingest_neurons(ds, paths, maps, log)
        e = ingest_edges(ds, paths, maps, n, log)
        e.insert(0, "dataset", ds)
        neurons.append(n)
        edges.append(e)
        print(f"[ingest] {ds}: {len(n):,} neurons, {len(e):,} edge rows "
              f"(dropped {log[f'{ds}_edge_rows_dropped_unknown_endpoint']:,} unknown-endpoint, "
              f"{log[f'{ds}_edge_rows_dropped_self_loop']:,} self-loop)")
    nr = pd.concat(neurons, ignore_index=True)
    for c in ["dataset", "side", "superclass", "nt"]:
        nr[c] = nr[c].astype("category")
    er = pd.concat(edges, ignore_index=True)
    for c in ["dataset", "neuropil_raw", "neuropil"]:
        er[c] = er[c].astype(str).astype("category")
    nr.to_parquet(layout.neurons_raw, index=False)
    er.to_parquet(layout.edges_raw, index=False)
    log["seconds"] = round(time.time() - t0, 1)
    write_json(log, layout.log("ingest"))
    print(f"[ingest] wrote {layout.neurons_raw.name}, {layout.edges_raw.name} in {log['seconds']} s")
    return log


if __name__ == "__main__":
    run()
