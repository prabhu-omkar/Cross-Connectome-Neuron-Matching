"""Phase 2 — the evaluation harness (plan.md 7.2.2, 8.1-8.3).

Every method — baseline or model — is a *scorer*: ``scorer.score(qd, q_uids, gd, g_uids)``
returns a float32 block [len(q), len(g)] of similarities (higher = better). Entries a method
cannot score (e.g. NBLAST pairs outside the published top-100) are set to ``MISSING``.

Gallery = all INCLUDED neurons of the other dataset. Queries = all included neurons of the
split's shared types (T2); those in 1:1 pairs also get the rank of their partner (T1).
Each block is scored once and evaluated unfiltered and with hard filters (same harmonised
super class; same side or either 'center'; NT equal or either unknown).

Per-query results are returned as a DataFrame so every aggregate (and bootstrap CI over
families) can be recomputed without rescoring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Layout
from .splits import family_to_split, load_splits

MISSING = np.float32(-1e9)        # "no score" — ranks below every real score, above filtered-out
DIRECTIONS = (("mcns", "banc"), ("banc", "mcns"))
KS = (1, 5, 10, 50, 100)


# ---------------------------------------------------------------- pure metric pieces
def ranks_avg_ties(S: np.ndarray, true_col: np.ndarray) -> np.ndarray:
    """1-based rank of S[i, true_col[i]] with ties shared (average rank)."""
    st = S[np.arange(len(S)), true_col][:, None]
    gt = (S > st).sum(1)
    eq = (S == st).sum(1) - 1
    return 1.0 + gt + eq / 2.0


def topk_hits(S: np.ndarray, q_type: np.ndarray, g_type: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """T2: does the top-1 / top-k of each row contain a gallery neuron of the query's type?
    Only real scores count (MISSING / filtered entries never produce a hit)."""
    k = min(k, S.shape[1])
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    sc = np.take_along_axis(S, part, 1)
    order = np.argsort(-sc, axis=1, kind="stable")
    part, sc = np.take_along_axis(part, order, 1), np.take_along_axis(sc, order, 1)
    same = (g_type[part] == q_type[:, None]) & (sc > MISSING / 2)
    return same[:, 0], same.any(1)


# ---------------------------------------------------------------- context
@dataclass
class Side:
    neurons: pd.DataFrame          # all neurons of the dataset, row i = uid i
    gallery: np.ndarray            # uids of included neurons
    sc: np.ndarray                 # int codes, harmonised super class
    side: np.ndarray               # int codes: left 0, right 1, center 2, unknown 3
    nt: np.ndarray                 # int codes; unknown = -1
    type_code: np.ndarray          # int codes over all type_h; untyped = -1


class EvalContext:
    def __init__(self, neurons: pd.DataFrame, pairs: pd.DataFrame, gt_types: pd.DataFrame,
                 families: set, split: str = "val"):
        self.split = split
        types = gt_types[gt_types.family.isin(families)]
        self.types = types.set_index("type_h")
        all_types = sorted(neurons.type_h.dropna().unique())
        tcode = {t: i for i, t in enumerate(all_types)}
        sc_codes = {s: i for i, s in enumerate(sorted(neurons.superclass.astype(str).unique()))}
        side_codes = {"left": 0, "right": 1, "center": 2, "unknown": 3}
        nt_codes = {s: i for i, s in enumerate(sorted(neurons.nt.astype(str).unique())) if s != "unknown"}
        self.sides, self.queries = {}, {}
        p = pairs[pairs.family.isin(families)]
        for ds in ("mcns", "banc"):
            n = neurons[neurons.dataset == ds].sort_values("uid").reset_index(drop=True)
            self.sides[ds] = Side(
                neurons=n,
                gallery=n.uid[n.include].to_numpy(),
                sc=n.superclass.astype(str).map(sc_codes).to_numpy(),
                side=n.side.astype(str).map(side_codes).to_numpy(),
                nt=n.nt.astype(str).map(nt_codes).fillna(-1).astype(int).to_numpy(),
                type_code=n.type_h.map(tcode).fillna(-1).astype(int).to_numpy(),
            )
            q = n[n.include & n.type_h.isin(self.types.index)][["uid", "type_h", "clean"]].copy()
            q["family"] = q.type_h.map(self.types.family)
            q["superclass"] = q.type_h.map(self.types.superclass)
            q["size_bucket"] = q.type_h.map(self.types.size_bucket)
            other = "banc" if ds == "mcns" else "mcns"
            partner = dict(zip(p[f"{ds}_uid"], p[f"{other}_uid"]))
            pclean = dict(zip(p[f"{ds}_uid"], p.mcns_clean & p.banc_clean))
            q["partner"] = q.uid.map(partner).fillna(-1).astype(np.int64)
            q["clean_pair"] = q.uid.map(pclean).fillna(False).astype(bool)
            self.queries[ds] = q.reset_index(drop=True)

    @classmethod
    def from_layout(cls, layout: Layout, split: str = "val"):
        splits = load_splits(layout)
        if split not in ("train", "val", "test"):
            raise ValueError(split)
        fams = set(splits[split])
        ctx = cls(pd.read_parquet(layout.neurons), pd.read_parquet(layout.gt_pairs),
                  pd.read_parquet(layout.gt_types), fams, split)
        ctx.splits_sha256 = splits["sha256"]
        return ctx

    def allowed(self, qd: str, q_uids: np.ndarray, gd: str, g_uids: np.ndarray) -> np.ndarray:
        a, b = self.sides[qd], self.sides[gd]
        same_sc = a.sc[q_uids][:, None] == b.sc[g_uids][None, :]
        qs, gs = a.side[q_uids][:, None], b.side[g_uids][None, :]
        side_ok = (qs == gs) | (qs == 2) | (gs == 2)
        qn, gn = a.nt[q_uids][:, None], b.nt[g_uids][None, :]
        nt_ok = (qn == gn) | (qn < 0) | (gn < 0)
        return same_sc & side_ok & nt_ok


# ---------------------------------------------------------------- evaluation
def evaluate(scorer, ctx: EvalContext, block: int = 512, filters: bool = True,
             directions=DIRECTIONS, t1_only: bool = False) -> tuple[dict, pd.DataFrame]:
    """t1_only: only 1:1-pair queries (fast; used for checkpoint selection during training)."""
    frames = []
    for qd, gd in directions:
        q = ctx.queries[qd]
        if t1_only:
            q = q[q.partner >= 0].reset_index(drop=True)
        G = ctx.sides[gd]
        g = G.gallery
        gpos = np.full(len(G.neurons), -1)
        gpos[g] = np.arange(len(g))
        g_type = G.type_code[g]
        q_uid = q.uid.to_numpy()
        q_type = ctx.sides[qd].type_code[q_uid]
        partner_col = np.where(q.partner.to_numpy() >= 0, gpos[np.maximum(q.partner.to_numpy(), 0)], -1)
        out = {k: np.full(len(q), np.nan) for k in ["rank", "rank_f"]}
        for k in ["t2_hit1", "t2_hit5", "t2_hit1_f", "t2_hit5_f"]:
            out[k] = np.zeros(len(q), bool)
        for s in range(0, len(q), block):
            rows = np.arange(s, min(s + block, len(q)))
            S = np.asarray(scorer.score(qd, q_uid[rows], gd, g), dtype=np.float32)
            assert S.shape == (len(rows), len(g)), S.shape
            variants = [("", S)]
            if filters:
                Sf = S.copy()
                Sf[~ctx.allowed(qd, q_uid[rows], gd, g)] = -np.inf
                variants.append(("_f", Sf))
            pc = partner_col[rows]
            has = pc >= 0
            for suf, M in variants:
                if has.any():
                    Mh = M[has]
                    tc = pc[has]
                    if suf:   # the true partner is never filtered away
                        Mh[np.arange(len(tc)), tc] = S[has][np.arange(len(tc)), tc]
                    out["rank" + suf][rows[has]] = ranks_avg_ties(Mh, tc)
                h1, h5 = topk_hits(M, q_type[rows], g_type, 5)
                out["t2_hit1" + suf][rows] = h1
                out["t2_hit5" + suf][rows] = h5
        f = q.copy()
        f.insert(0, "direction", f"{qd}->{gd}")
        for k, v in out.items():
            f[k] = v
        if not filters:
            f = f.drop(columns=["rank_f", "t2_hit1_f", "t2_hit5_f"])
        frames.append(f)
    per_query = pd.concat(frames, ignore_index=True)
    return summarise(per_query), per_query


def _t1(r: np.ndarray) -> dict:
    if len(r) == 0:
        return {}
    d = {f"R@{k}": float((r <= k).mean()) for k in KS}
    d["MRR"] = float((1.0 / r).mean())
    d["median_rank"] = float(np.median(r))
    d["n"] = int(len(r))
    return d


def summarise(pq: pd.DataFrame) -> dict:
    """Flat metric dict from per-query results. Keys: '<task><filter>/<direction|mean>/<metric>'."""
    m: dict = {}
    fsuf = [""] + (["_f"] if "rank_f" in pq else [])
    dirs = list(pq.direction.unique())
    for suf in fsuf:
        tag = "filtered" if suf else "unfiltered"
        for d in dirs:
            x = pq[(pq.direction == d)]
            for k, v in _t1(x["rank" + suf].dropna().to_numpy()).items():
                m[f"T1/{tag}/{d}/{k}"] = v
            y = x
            m[f"T2/{tag}/{d}/acc@1_micro"] = float(y["t2_hit1" + suf].mean())
            m[f"T2/{tag}/{d}/acc@1_macro"] = float(y.groupby("superclass")["t2_hit1" + suf].mean().mean())
            m[f"T2/{tag}/{d}/acc@5_macro"] = float(y.groupby("superclass")["t2_hit5" + suf].mean().mean())
        for k in ["R@1", "R@5", "R@10", "R@50", "R@100", "MRR", "median_rank"]:
            vals = [m.get(f"T1/{tag}/{d}/{k}") for d in dirs]
            if all(v is not None for v in vals):
                m[f"T1/{tag}/mean/{k}"] = float(np.mean(vals))
        for k in ["acc@1_micro", "acc@1_macro", "acc@5_macro"]:
            m[f"T2/{tag}/mean/{k}"] = float(np.mean([m[f"T2/{tag}/{d}/{k}"] for d in dirs]))
        # clean subset (both partners clean for T1; query clean for T2)
        c = pq[pq.clean_pair]
        rr = c["rank" + suf].dropna().to_numpy()
        if len(rr):
            m[f"T1/{tag}/clean/R@1"] = float((rr <= 1).mean())
            m[f"T1/{tag}/clean/MRR"] = float((1 / rr).mean())
            m[f"T1/{tag}/clean/n"] = int(len(rr))
        cq = pq[pq.clean]
        m[f"T2/{tag}/clean/acc@1_macro"] = float(cq.groupby("superclass")["t2_hit1" + suf].mean().mean())
        # breakdowns (both directions pooled)
        for col in ["superclass", "size_bucket"]:
            for key, g in pq.groupby(col):
                r = g["rank" + suf].dropna().to_numpy()
                if len(r):
                    m[f"T1/{tag}/by_{col}/{key}/R@1"] = float((r <= 1).mean())
                    m[f"T1/{tag}/by_{col}/{key}/n"] = int(len(r))
                m[f"T2/{tag}/by_{col}/{key}/acc@1"] = float(g["t2_hit1" + suf].mean())
    return m


# ---------------------------------------------------------------- statistics
def bootstrap_ci(pq: pd.DataFrame, n_boot: int = 1000, seed: int = 0, suf: str = "") -> dict:
    """95% percentile CIs, resampling FAMILIES (plan.md 8.2), for T1 mean R@1 / MRR
    (pair value = mean over directions) and T2 mean macro acc@1."""
    rng = np.random.default_rng(seed)
    fams = np.array(sorted(pq.family.unique()))
    fidx = {f: i for i, f in enumerate(fams)}
    out = {}
    t1 = pq.dropna(subset=["rank" + suf])
    if len(t1):
        r = t1["rank" + suf].to_numpy()
        fi = t1.family.map(fidx).to_numpy()
        n_dirs = t1.direction.nunique()
        for name, v in [("R@1", (r <= 1).astype(float)), ("MRR", 1 / r)]:
            s = np.bincount(fi, weights=v, minlength=len(fams))
            c = np.bincount(fi, minlength=len(fams)).astype(float)
            boots = []
            for _ in range(n_boot):
                w = np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams))
                boots.append((w * s).sum() / max((w * c).sum(), 1))
            out[f"T1/mean/{name}_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        _ = n_dirs
    sc = pq.superclass.astype(str)
    scs = sorted(sc.unique())
    sci = {s: i for i, s in enumerate(scs)}
    fi = pq.family.map(fidx).to_numpy().astype(int)
    ci = sc.map(sci).to_numpy().astype(int)
    hit = pq["t2_hit1" + suf].to_numpy().astype(float)
    H = np.zeros((len(fams), len(scs)))
    C = np.zeros((len(fams), len(scs)))
    np.add.at(H, (fi, ci), hit)
    np.add.at(C, (fi, ci), 1)
    boots = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams))[:, None]
        hc, cc = (w * H).sum(0), (w * C).sum(0)
        ok = cc > 0
        boots.append((hc[ok] / cc[ok]).mean())
    out["T2/mean/acc@1_macro_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return out


def paired_bootstrap(pq_a: pd.DataFrame, pq_b: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
                     metric: str = "R@1", suf: str = "") -> dict:
    """Difference a - b in T1 mean metric with a family-level paired bootstrap CI."""
    key = ["direction", "uid"]
    j = pq_a.dropna(subset=["rank" + suf]).merge(pq_b[key + ["rank" + suf]], on=key, suffixes=("_a", "_b"))
    ra, rb = j[f"rank{suf}_a"].to_numpy(), j[f"rank{suf}_b"].to_numpy()
    f = (lambda r: (r <= 1).astype(float)) if metric == "R@1" else (lambda r: 1 / r)
    d = f(ra) - f(rb)
    fams = j.family.astype("category").cat.codes.to_numpy()
    nf = fams.max() + 1
    s, c = np.bincount(fams, weights=d, minlength=nf), np.bincount(fams, minlength=nf).astype(float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, nf, nf), minlength=nf)
        boots.append((w * s).sum() / (w * c).sum())
    return {"diff": float(d.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n": int(len(d))}
