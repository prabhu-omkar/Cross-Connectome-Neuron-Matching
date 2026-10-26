"""B2 — NBLAST, from the BANC team's published BANC <-> MaleCNS v0.9 matrix (plan.md 7.1.7, 7.2.3).

The file is a long table: for each BANC neuron, its top-100 MaleCNS v0.9 matches with a
(normalised, mean) NBLAST score. BANC rows are mapped via `root_888` (the v888 root ID = our
export), MaleCNS `match_id` (v0.9 body ID) by exact ID match against v1.0. Duplicate
(BANC, MaleCNS) pairs keep the maximum score. Cached as a sparse matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
import scipy.sparse as sp

from ..config import Layout, load_paths, write_json
from .scorers import PairTableScorer

FILE = "banc_malecns_v0.9_nblast.feather"


def cache_path(layout: Layout):
    return layout.processed / "nblast_mcns_banc.npz"


def build_matrix(layout: Layout) -> dict | None:
    src = load_paths()["external_dir"] / "banc_dataverse_v3" / FILE
    if not src.exists():
        return None
    neurons = pd.read_parquet(layout.neurons, columns=["dataset", "neuron_id", "uid"])
    idx = {ds: pd.Index(neurons.loc[neurons.dataset == ds].sort_values("uid").neuron_id.to_numpy())
           for ds in ("mcns", "banc")}
    r = ipc.open_file(src)
    rows, cols, vals = [], [], []
    n_rows = n_banc_ok = n_mcns_ok = 0
    for i in range(r.num_record_batches):
        b = r.get_batch(i).select(["root_888", "match_id", "score"]).to_pandas()
        n_rows += len(b)
        b = b.dropna(subset=["root_888", "match_id", "score"])
        bi = idx["banc"].get_indexer(pd.to_numeric(b.root_888, errors="coerce").fillna(-1).astype(np.int64))
        mi = idx["mcns"].get_indexer(pd.to_numeric(b.match_id, errors="coerce").fillna(-1).astype(np.int64))
        n_banc_ok += int((bi >= 0).sum())
        n_mcns_ok += int((mi >= 0).sum())
        ok = (bi >= 0) & (mi >= 0)
        rows.append(mi[ok]); cols.append(bi[ok]); vals.append(b.score.to_numpy(np.float32)[ok])
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    df = pd.DataFrame({"r": rows, "c": cols, "v": vals}).groupby(["r", "c"]).v.max().reset_index()
    M = sp.csr_matrix((df.v.to_numpy() + PairTableScorer.OFFSET, (df.r, df.c)),
                      shape=(len(idx["mcns"]), len(idx["banc"])), dtype=np.float32)
    sp.save_npz(cache_path(layout), M)
    stats = {"rows_in_file": n_rows, "rows_banc_mapped": n_banc_ok, "rows_mcns_mapped": n_mcns_ok,
             "pairs_kept": int(M.nnz),
             "banc_neurons_with_scores": int((np.diff(M.tocsc().indptr) > 0).sum()),
             "mcns_neurons_with_scores": int((np.diff(M.indptr) > 0).sum())}
    write_json(stats, layout.processed / "nblast_stats.json")
    return stats


def load_scorer(layout: Layout) -> PairTableScorer | None:
    p = cache_path(layout)
    if not p.exists() and build_matrix(layout) is None:
        return None
    return PairTableScorer(sp.load_npz(p))


def pair_coverage(layout: Layout, pairs: pd.DataFrame) -> dict:
    M = sp.load_npz(cache_path(layout)).tocsr()
    has_pair = np.asarray(M[pairs.mcns_uid.to_numpy(), pairs.banc_uid.to_numpy()]).ravel() > 0
    has_q_m = np.diff(M.indptr)[pairs.mcns_uid.to_numpy()] > 0
    has_q_b = np.diff(M.tocsc().indptr)[pairs.banc_uid.to_numpy()] > 0
    return {"pairs": int(len(pairs)), "true_pair_scored": float(has_pair.mean()),
            "mcns_query_has_any_score": float(has_q_m.mean()), "banc_query_has_any_score": float(has_q_b.mean())}
