"""One-to-one assignment with a "none" option (plan.md 7.3.7; week 10) -> matches.parquet.

Nearest neighbours are greedy (two MaleCNS neurons can claim one BANC neuron), so:
1. candidate graph = top-`k` filtered neighbours in BOTH directions (all included neurons);
2. connected components of that bipartite graph;
3. components with <= `max_component` nodes: Hungarian (`linear_sum_assignment`) on
   cost = -similarity, with one dummy "none" column per row at cost -tau_none, so a neuron stays
   unmatched when no real partner beats tau_none;
4. larger components: the plan said mutual-best pairs only, but with real embeddings ~98% of
   neurons sit in a few giant components (largest ~77k nodes), so mutual-best left most neurons
   unmatched (val recall 0.44). Default `large="sparse"`: an exact min-cost matching over the
   sparse candidate graph of all big components at once (scipy LAPJVsp) with the same "none"
   option, built as the standard dummy-node construction; edges from each side's top-`k_assign`
   (default 5; 5/10/20 give the same F1). ~1.5 min on the laptop. `large="mutual"` keeps the
   plan's rule;
5. tau_none is chosen on VALIDATION 1:1 pairs (max F1) from a fixed grid. Test pairs are never
   scored here.

`python -m flymatch.run match --run runs/<run>` also fits the abstention head (abstention.py) and
writes into the run folder: matches.parquet, match_metrics.json, abstention_check.parquet.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components, min_weight_full_bipartite_matching

from .evaluate import EvalContext

NO_EDGE = 1e6
TAU_GRID = (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9)


def candidates(Zt: dict, ctx: EvalContext, k: int = 20, block: int = 1024) -> pd.DataFrame:
    """Union of filtered top-k lists in both directions: columns mcns_uid, banc_uid, sim,
    rank_m (rank of banc in mcns's list, k+1 if absent), rank_b (vice versa)."""
    parts = []
    for qd, gd in (("mcns", "banc"), ("banc", "mcns")):
        q_all, g = ctx.sides[qd].gallery, ctx.sides[gd].gallery
        Gt = Zt[gd][torch.as_tensor(g)]
        for s in range(0, len(q_all), block):
            q = q_all[s:s + block]
            S = Zt[qd][torch.as_tensor(q)] @ Gt.T
            S = S.masked_fill(~torch.as_tensor(ctx.allowed(qd, q, gd, g)), float("-inf"))
            val, idx = S.topk(min(k, len(g)), dim=1)
            ok = torch.isfinite(val).numpy()
            qq = np.repeat(q, val.shape[1]).reshape(val.shape)[ok]
            gg = g[idx.numpy()][ok]
            rk = np.tile(np.arange(1, val.shape[1] + 1), (len(q), 1))[ok]
            f = pd.DataFrame({f"{qd}_uid": qq, f"{gd}_uid": gg, "sim": val.numpy()[ok],
                              ("rank_m" if qd == "mcns" else "rank_b"): rk})
            parts.append(f)
    c = pd.concat(parts, ignore_index=True)
    c = c.groupby(["mcns_uid", "banc_uid"], as_index=False).agg(sim=("sim", "first"), rank_m=("rank_m", "min"),
                                                                 rank_b=("rank_b", "min"))
    c[["rank_m", "rank_b"]] = c[["rank_m", "rank_b"]].fillna(k + 1).astype(int)
    return c


def components(c: pd.DataFrame, n_m: int, n_b: int) -> tuple[np.ndarray, np.ndarray]:
    """Component label per candidate edge, and component sizes (in nodes)."""
    G = sp.coo_matrix((np.ones(len(c)), (c.mcns_uid.to_numpy(), n_m + c.banc_uid.to_numpy())),
                      shape=(n_m + n_b, n_m + n_b))
    _, lab = connected_components(G, directed=False)
    edge_comp = lab[c.mcns_uid.to_numpy()]
    nodes = np.unique(np.concatenate([c.mcns_uid.to_numpy(), n_m + c.banc_uid.to_numpy()]))
    size = np.bincount(lab[nodes], minlength=lab.max() + 1)
    return edge_comp, size


def sparse_match(m: np.ndarray, b: np.ndarray, sim: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact max-similarity one-to-one matching where leaving a neuron unmatched is worth tau/2
    per side (so a pair is used only when it beats tau). Rows = MaleCNS + one dummy per BANC
    node; cols = BANC + one dummy per MaleCNS node; a dummy-dummy edge mirrors every real edge,
    so a full matching always exists. Costs are shifted to stay positive (explicit zeros would
    vanish from the sparse matrix). Returns matched (mcns_uid, banc_uid)."""
    keep = sim > tau
    m, b, sim = m[keep], b[keep], sim[keep]
    if len(m) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    mu, mi = np.unique(m, return_inverse=True)
    bu, bi = np.unique(b, return_inverse=True)
    nm, nb, E = len(mu), len(bu), len(m)
    d = (3.0 - tau) / 2          # pair: (2 - sim) + 1 ; both unmatched: 2d = 3 - tau  -> match iff sim > tau
    rows = np.concatenate([mi, np.arange(nm), nm + bi, nm + np.arange(nb)])
    cols = np.concatenate([bi, nb + np.arange(nm), nb + mi, np.arange(nb)])
    w = np.concatenate([2.0 - sim, np.full(nm, d), np.ones(E), np.full(nb, d)])
    A = sp.csr_matrix((w, (rows, cols)), shape=(nm + nb, nb + nm))
    r, c = min_weight_full_bipartite_matching(A)
    real = (r < nm) & (c < nb)
    return mu[r[real]], bu[c[real]]


def assign(c: pd.DataFrame, edge_comp: np.ndarray, comp_size: np.ndarray, tau: float,
           max_component: int = 2000, large: str = "sparse", k_assign: int = 5) -> pd.DataFrame:
    """Returns matched pairs (mcns_uid, banc_uid, sim, how)."""
    out = []
    c = c.assign(comp=edge_comp)
    small = comp_size[c.comp.to_numpy()] <= max_component
    for _, e in c[small].groupby("comp"):
        e = e[e.sim > tau]
        if e.empty:
            continue
        mu, mi = np.unique(e.mcns_uid.to_numpy(), return_inverse=True)
        bu, bi = np.unique(e.banc_uid.to_numpy(), return_inverse=True)
        C = np.full((len(mu), len(bu) + len(mu)), NO_EDGE)
        C[mi, bi] = -e.sim.to_numpy()
        C[np.arange(len(mu)), len(bu) + np.arange(len(mu))] = -tau
        r, col = linear_sum_assignment(C)
        real = col < len(bu)
        if real.any():
            m = pd.DataFrame({"mcns_uid": mu[r[real]], "banc_uid": bu[col[real]], "sim": -C[r[real], col[real]]})
            out.append(m.assign(how="hungarian"))
    big = c[~small]
    if len(big) and large == "sparse":
        e = big[(big.rank_m <= k_assign) | (big.rank_b <= k_assign)]
        mu, bu = sparse_match(e.mcns_uid.to_numpy(), e.banc_uid.to_numpy(), e.sim.to_numpy(), tau)
        if len(mu):
            m = pd.DataFrame({"mcns_uid": mu, "banc_uid": bu}).merge(e[["mcns_uid", "banc_uid", "sim"]],
                                                                    on=["mcns_uid", "banc_uid"])
            out.append(m.assign(how="sparse_lap"))
    elif len(big):
        bm = big.loc[big.groupby("mcns_uid").sim.idxmax()]
        bb = big.loc[big.groupby("banc_uid").sim.idxmax()]
        mutual = bm.merge(bb[["mcns_uid", "banc_uid"]], on=["mcns_uid", "banc_uid"])
        mutual = mutual[mutual.sim > tau]
        out.append(mutual[["mcns_uid", "banc_uid", "sim"]].assign(how="mutual_best"))
    if not out:
        return pd.DataFrame(columns=["mcns_uid", "banc_uid", "sim", "how"])
    return pd.concat(out, ignore_index=True)


def pair_scores(matches: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Precision / recall / F1 against 1:1 pairs (MaleCNS side; abstentions = no prediction)."""
    pred = dict(zip(matches.mcns_uid, matches.banc_uid))
    got = pairs.mcns_uid.map(pred)
    n_pred = int(got.notna().sum())
    correct = int((got == pairs.banc_uid).sum())
    p = correct / max(n_pred, 1)
    r = correct / max(len(pairs), 1)
    return {"precision": p, "recall": r, "F1": 2 * p * r / max(p + r, 1e-12), "n_pairs": int(len(pairs)),
            "n_predicted": n_pred, "n_correct": correct}


def run(layout, run_dir, k: int = 20, max_component: int = 2000, large: str = "sparse", device: str = "cpu",
        log=print) -> dict:
    from .abstention import biological_check, feature_cols, fit_and_check, query_features, simulation_set
    from .config import write_json
    from .inference import embed_run
    run_dir = Path(run_dir)
    t0 = time.time()
    Z = embed_run(layout, run_dir, device)
    Zt = {ds: torch.nn.functional.normalize(torch.as_tensor(z), dim=1) for ds, z in Z.items()}
    ctx = EvalContext.from_layout(layout, "val")
    log(f"[match] embeddings ready ({time.time() - t0:.0f}s)")

    # --- abstention head (validation families, half A fit / half B check)
    sim = simulation_set(Zt, ctx)
    head, m_abs, check = fit_and_check(sim)
    base = check[(check.direction == "banc->mcns") & (check.y == 0)]
    m_abs.update(biological_check(head, Zt, ctx, base))
    check.to_parquet(run_dir / "abstention_check.parquet", index=False)
    log("[match] abstention: " + ", ".join(f"{k.split('/')[-1]} {v:.3f}" for k, v in m_abs.items()
                                          if isinstance(v, float)))

    # --- candidate graph + assignment; tau_none on validation 1:1 pairs
    c = candidates(Zt, ctx, k, block=256)
    n_m, n_b = len(ctx.sides["mcns"].neurons), len(ctx.sides["banc"].neurons)
    edge_comp, size = components(c, n_m, n_b)
    big_nodes = int(size[size > max_component].sum())
    log(f"[match] {len(c):,} candidate edges, {int((size > 0).sum()):,} components "
        f"(largest {int(size.max()):,} nodes; {big_nodes:,} nodes in components > {max_component})")
    p = pd.read_parquet(layout.gt_pairs)
    val_pairs = p[p.family.isin(set(ctx.types.family))]
    grid = []
    for tau in TAU_GRID:
        sc = pair_scores(assign(c, edge_comp, size, float(tau), max_component, large), val_pairs)
        grid.append({"tau_none": float(tau), **sc})
        log(f"[match] tau_none {tau:.3f}: P {sc['precision']:.3f} R {sc['recall']:.3f} F1 {sc['F1']:.3f}")
    grid = pd.DataFrame(grid)
    best = grid.loc[grid.F1.idxmax()]
    tau = float(best.tau_none)
    matches = assign(c, edge_comp, size, tau, max_component, large)

    # --- matches.parquet: every included MaleCNS neuron, its BANC partner or null, p_none
    m_side = ctx.sides["mcns"]
    feats = query_features(Zt, ctx, "mcns", "banc", m_side.gallery, block=256)   # small blocks: laptop RAM
    pn = head.predict(feats[feature_cols(feats)].to_numpy())
    out = pd.DataFrame({"mcns_uid": m_side.gallery, "p_none": pn})
    out = out.merge(matches, on="mcns_uid", how="left").merge(c[["mcns_uid", "banc_uid", "rank_m", "rank_b"]],
                                                              on=["mcns_uid", "banc_uid"], how="left")
    out["mcns_id"] = m_side.neurons.neuron_id.to_numpy()[out.mcns_uid]
    b_ids = ctx.sides["banc"].neurons.neuron_id.to_numpy()
    out["banc_id"] = pd.array([b_ids[int(u)] if pd.notna(u) else pd.NA for u in out.banc_uid], dtype="Int64")
    out = out.rename(columns={"sim": "score"})[["mcns_id", "banc_id", "score", "p_none", "rank_m", "rank_b", "how",
                                                "mcns_uid", "banc_uid"]]
    out.to_parquet(run_dir / "matches.parquet", index=False)
    grid.to_csv(run_dir / "assignment_tau_grid.csv", index=False)
    m = {**m_abs, "assign/tau_none": tau, **{f"assign/val/{k}": v for k, v in best.drop("tau_none").items()},
         "assign/k": k, "assign/max_component": max_component, "assign/large": large, "assign/n_matched": int(out.banc_id.notna().sum()),
         "assign/n_mcns": int(len(out)), "assign/nodes_in_big_components": big_nodes}
    write_json({k: (v.item() if hasattr(v, "item") else v) for k, v in m.items()}, run_dir / "match_metrics.json")
    log(f"[match] tau_none {tau:.3f} -> val F1 {best.F1:.3f}; {m['assign/n_matched']:,} of {len(out):,} MaleCNS neurons "
        f"matched; wrote {run_dir / 'matches.parquet'} ({time.time() - t0:.0f}s)")
    return m
