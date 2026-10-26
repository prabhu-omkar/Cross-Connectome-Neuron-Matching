"""Phase 1.3 — ground truth (plan.md 7.1.6) and type families (7.1.8 steps 1-2).

MaleCNS type_h, in priority order: Primary Cell Type if it exists as a BANC type, else
flywireType, else mancType. BANC type_h = Primary Cell Type.

Fallback validation: for every (type, fallback source) the MaleCNS neurons that got the type
through that source are compared with the BANC neurons of the type on majority super class,
majority NT and mean neuropil-profile cosine. Fallback types that fail super-class agreement
are dropped; a whole source is dropped if more than `fallback_max_superclass_fail` of its
types fail.

Writes gt_types.parquet, gt_pairs_1to1.parquet, families.parquet, adds type_h / gt_source /
clean / family to neurons.parquet, and results/gt_*.csv validation tables.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import Layout, load_data_config, load_paths, write_json

SOURCE_COL = {"primary": "type_primary", "flywire": "flywire_type", "manc": "manc_type"}
STEM_RE = re.compile(r"^([A-Za-z]+[0-9]*)")


# ---------------------------------------------------------------- type harmonisation
def banc_type_vocab(n_banc: pd.DataFrame) -> set:
    ok = (n_banc.superclass.astype(str) != "excluded") & ~n_banc.type_primary.fillna("").str.contains("fragment")
    return set(n_banc.loc[ok, "type_primary"].dropna())


def assign_mcns_types(n_mcns: pd.DataFrame, banc_types: set, sources: list[str]) -> pd.DataFrame:
    type_h = pd.Series(pd.NA, index=n_mcns.index, dtype="object")
    src = pd.Series(pd.NA, index=n_mcns.index, dtype="object")
    for s in sources:
        cand = n_mcns[SOURCE_COL[s]]
        hit = type_h.isna() & cand.isin(banc_types)
        type_h[hit] = cand[hit]
        src[hit] = s
    return pd.DataFrame({"type_h": type_h, "gt_source": src})


def _majority(s: pd.Series):
    s = s.dropna().astype(str)
    return s.value_counts().index[0] if len(s) else None


def _profiles(layout: Layout, spec: dict, ds: str) -> np.ndarray:
    X = np.load(layout.features(ds), mmap_mode="r")
    a, b = spec["blocks"]["np_in"][0], spec["blocks"]["np_out"][1]
    return np.asarray(X[:, a:b], dtype=np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def validate_sources(n_mcns, n_banc, P_m, P_b) -> pd.DataFrame:
    """One row per (type_h, source): super class / NT agreement and profile cosine."""
    bi = n_banc[n_banc.include & n_banc.type_h.notna()]
    bgrp = {t: g for t, g in bi.groupby("type_h")}
    rows = []
    mi = n_mcns[n_mcns.include & n_mcns.type_h.notna()]
    for (t, s), g in mi.groupby(["type_h", "gt_source"]):
        b = bgrp.get(t)
        if b is None:
            continue
        sc_m, sc_b = _majority(g.superclass), _majority(b.superclass)
        nt_m, nt_b = _majority(g.nt[g.nt.astype(str) != "unknown"]), _majority(b.nt[b.nt.astype(str) != "unknown"])
        rows.append({
            "type_h": t, "source": s, "n_mcns": len(g), "n_banc": len(b),
            "superclass_mcns": sc_m, "superclass_banc": sc_b, "superclass_agree": sc_m == sc_b,
            "nt_mcns": nt_m, "nt_banc": nt_b,
            "nt_agree": (nt_m == nt_b) if (nt_m and nt_b) else np.nan,
            "profile_cosine": _cos(P_m[g.uid.to_numpy()].mean(0), P_b[b.uid.to_numpy()].mean(0)),
        })
    return pd.DataFrame(rows)


def summarise_validation(v: pd.DataFrame) -> pd.DataFrame:
    prim_cos = v.loc[v.source == "primary", "profile_cosine"].dropna()
    lo = prim_cos.quantile(0.05) if len(prim_cos) else np.nan
    out = v.groupby("source").agg(
        types=("type_h", "size"), mcns_neurons=("n_mcns", "sum"),
        superclass_agree=("superclass_agree", "mean"),
        nt_agree=("nt_agree", lambda x: x.dropna().astype(float).mean()),
        profile_cosine_median=("profile_cosine", "median"),
        frac_cosine_below_primary_p5=("profile_cosine", lambda x: float((x.dropna() < lo).mean())),
    ).reset_index()
    out["superclass_fail"] = 1 - out.superclass_agree
    return out


# ---------------------------------------------------------------- families
class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def stem(t: str) -> str:
    m = STEM_RE.match(t)
    return m.group(1) if m else t


def build_families(types: list[str], neurons: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Union-find over (a) shared name stem, (b) names listed together for one neuron
    (primary, alternative types, flywire/manc types, type_h), (c) MaleCNS supertype groups."""
    uf = UnionFind()
    tset = set(types)
    for t in types:
        uf.union(("T", t), ("S", stem(t)))
    links = {"alt": 0, "supertype": 0}
    name_cols = ["type_h", "type_primary", "flywire_type", "manc_type"]
    sub = neurons[neurons[name_cols].notna().any(axis=1) | neurons.type_alt.notna()]
    for row in sub[name_cols + ["type_alt"]].itertuples(index=False):
        names = {x for x in row[:4] if isinstance(x, str)}
        if isinstance(row[4], str):
            names |= {x.strip() for x in row[4].split(",") if x.strip()}
        names = [x for x in names if x in tset]
        for a in names[1:]:
            uf.union(("T", names[0]), ("T", a))
            links["alt"] += 1
    m = neurons[(neurons.dataset == "mcns") & neurons.supertype.notna() & neurons.type_h.notna()]
    for _, g in m.groupby("supertype"):
        ts = [t for t in g.type_h.unique() if t in tset]
        for a in ts[1:]:
            uf.union(("T", ts[0]), ("T", a))
            links["supertype"] += 1
    roots = {t: uf.find(("T", t)) for t in types}
    # name each family after its alphabetically first member type
    first = {}
    for t in sorted(types):
        first.setdefault(roots[t], t)
    fam = pd.DataFrame({"type_h": types, "family": [first[roots[t]] for t in types]})
    return fam, links


# ---------------------------------------------------------------- main
def size_bucket(n: int, buckets) -> str:
    for lo, hi, name in buckets:
        if lo <= n <= hi:
            return name
    raise ValueError(n)


def reviewed_match_check(paths: dict, neurons: pd.DataFrame) -> dict | None:
    """Agreement of our type_h with the BANC team's 2,194 expert-reviewed BANC<->MaleCNS matches."""
    f = paths["external_dir"] / "banc_dataverse_v3" / "banc_malecns_reviewed_matches.csv"
    if not f.exists():
        return None
    r = pd.read_csv(f, dtype=str)
    r = r[r.valid.str.lower().isin(["t", "true"])]
    key = lambda ds: neurons[neurons.dataset == ds].assign(k=lambda d: d.neuron_id.astype(str)).set_index("k")
    b, m = key("banc"), key("mcns")
    r = r[r.query_id.isin(b.index) & r.match_id.isin(m.index)]
    tb = b.loc[r.query_id, "type_h"].to_numpy()
    tm = m.loc[r.match_id, "type_h"].to_numpy()
    src = m.loc[r.match_id, "gt_source"].to_numpy()
    both = pd.notna(tb) & pd.notna(tm)
    agree = both & (tb == tm)
    out = {"reviewed_pairs_in_exports": int(len(r)), "both_typed": int(both.sum()),
           "type_h_agree": int(agree.sum()),
           "agree_rate_when_both_typed": float(agree.sum() / max(both.sum(), 1)), "by_source": {}}
    for s in pd.unique(src[both]):
        k = both & (src == s)
        out["by_source"][str(s)] = {"n": int(k.sum()), "agree_rate": float((agree & k).sum() / k.sum())}
    return out


def run(cfg: dict | None = None):
    cfg = cfg or load_data_config()
    layout, paths = Layout(cfg), load_paths()
    import json
    spec = json.loads(layout.feature_spec.read_text())
    neurons = pd.read_parquet(layout.neurons)
    for c in ["type_h", "gt_source", "family"]:
        if c in neurons:
            neurons = neurons.drop(columns=c)
    nm = neurons[neurons.dataset == "mcns"].copy()
    nb = neurons[neurons.dataset == "banc"].copy()
    log: dict = {}

    banc_types = banc_type_vocab(nb)
    nb["type_h"] = nb.type_primary.where(nb.type_primary.isin(banc_types))
    nb["gt_source"] = np.where(nb.type_h.notna(), "banc_primary", None)
    sources = list(cfg["gt_sources"])
    nm[["type_h", "gt_source"]] = assign_mcns_types(nm, banc_types, sources).to_numpy()

    # ---- fallback validation
    P_m, P_b = _profiles(layout, spec, "mcns"), _profiles(layout, spec, "banc")
    v = validate_sources(nm, nb, P_m, P_b)
    summ = summarise_validation(v)
    v.to_csv(layout.results / "gt_fallback_validation_types.csv", index=False)
    summ.to_csv(layout.results / "gt_fallback_validation_summary.csv", index=False)
    print("[groundtruth] fallback validation:\n" + summ.to_string(index=False))
    dropped_sources = [s for s in summ.source
                       if s != "primary" and float(summ.loc[summ.source == s, "superclass_fail"].iloc[0]) > cfg["fallback_max_superclass_fail"]]
    bad = v[(v.source != "primary") & ~v.superclass_agree][["type_h", "source"]]
    log["fallback_sources_dropped"] = dropped_sources
    log["fallback_types_dropped_superclass_mismatch"] = len(bad)
    if dropped_sources:
        sources = [s for s in sources if s not in dropped_sources]
        nm[["type_h", "gt_source"]] = assign_mcns_types(nm, banc_types, sources).to_numpy()
    if len(bad):
        k = pd.MultiIndex.from_frame(nm[["type_h", "gt_source"]].astype(object))
        drop = k.isin(pd.MultiIndex.from_frame(bad.astype(object)))
        nm.loc[drop, ["type_h", "gt_source"]] = None

    # ---- clean subset flags
    nm["clean"] = nm.status.isin(cfg["mcns_clean_status"])
    if nb.banc_proofread.notna().any():
        flags = "|".join(map(re.escape, cfg["banc_unclean_status_flags"]))
        nb["clean"] = (nb.banc_proofread.astype(str).str.upper() == "TRUE") & ~nb.status.fillna("").str.contains(flags)
    else:
        nb["clean"] = False
    log["clean"] = {"mcns": int(nm.clean.sum()), "banc": int(nb.clean.sum())}

    neurons = pd.concat([nm, nb]).sort_values(["dataset", "uid"]).reset_index(drop=True)

    # ---- type table (types with >= 1 included neuron in both datasets)
    inc = neurons[neurons.include & neurons.type_h.notna()]
    cnt = inc.groupby(["type_h", "dataset"], observed=True).size().unstack(fill_value=0)
    cnt = cnt[(cnt.get("mcns", 0) > 0) & (cnt.get("banc", 0) > 0)]
    types = sorted(cnt.index)
    fam, links = build_families(types, neurons)
    log["family_links"] = links
    gt = fam.set_index("type_h")
    gt["n_mcns"] = cnt.mcns.reindex(gt.index).astype(int)
    gt["n_banc"] = cnt.banc.reindex(gt.index).astype(int)
    gt["superclass"] = inc[inc.type_h.isin(types)].groupby("type_h").superclass.agg(_majority).reindex(gt.index)
    gt["size_bucket"] = [size_bucket(max(a, b), cfg["size_buckets"]) for a, b in zip(gt.n_mcns, gt.n_banc)]
    src = inc[(inc.dataset == "mcns")].groupby("type_h").gt_source.agg(_majority)
    gt["source"] = src.reindex(gt.index)
    gt = gt.reset_index()

    # ---- 1:1 pairs: (type_h, side) with exactly one included neuron in each dataset
    g = inc[inc.type_h.isin(types)]
    per = g.groupby(["type_h", "side", "dataset"], observed=True).uid.agg(["size", "first"]).unstack("dataset")
    one = per[(per[("size", "mcns")] == 1) & (per[("size", "banc")] == 1)]
    pairs = pd.DataFrame({
        "type_h": one.index.get_level_values(0), "side": one.index.get_level_values(1).astype(str),
        "mcns_uid": one[("first", "mcns")].astype(np.int32).to_numpy(),
        "banc_uid": one[("first", "banc")].astype(np.int32).to_numpy(),
    })
    pairs = pairs.merge(fam, on="type_h")
    mi = nm.set_index("uid")
    bi = nb.set_index("uid")
    pairs["source"] = mi.gt_source.reindex(pairs.mcns_uid).to_numpy()
    pairs["mcns_clean"] = mi.clean.reindex(pairs.mcns_uid).to_numpy()
    pairs["banc_clean"] = bi.clean.reindex(pairs.banc_uid).to_numpy()
    pairs = pairs[["type_h", "side", "mcns_uid", "banc_uid", "family", "source", "mcns_clean", "banc_clean"]]

    # pre-inclusion count for comparison with plan.md 2.4 (6,774)
    allt = neurons[neurons.type_h.notna() & neurons.side.astype(str).isin(["left", "right", "center"])]
    pp = allt.groupby(["type_h", "side", "dataset"], observed=True).size().unstack("dataset", fill_value=0)
    log["pairs_1to1_before_inclusion"] = int(((pp.get("mcns", 0) == 1) & (pp.get("banc", 0) == 1)).sum())
    log["pairs_1to1"] = len(pairs)
    log["pairs_1to1_by_source"] = pairs.source.value_counts().to_dict()
    log["pairs_1to1_clean_both"] = int((pairs.mcns_clean & pairs.banc_clean).sum())
    log["gt_types"] = len(gt)
    log["families"] = int(gt.family.nunique())
    fs = gt.groupby("family").size()
    log["largest_families"] = fs.sort_values(ascending=False).head(10).to_dict()
    log["mcns_typed_included"] = int((nm.include & nm.type_h.notna()).sum())
    log["banc_typed_included"] = int((nb.include & nb.type_h.notna()).sum())
    log["no_counterpart_candidates_banc_female_specific"] = int(((nb.dimorphism == "female-specific") & nb.include).sum())
    rc = reviewed_match_check(paths, neurons)
    if rc:
        log["reviewed_match_check"] = rc

    neurons = neurons.merge(fam, on="type_h", how="left")
    for c in ["dataset", "side", "superclass", "nt"]:
        neurons[c] = neurons[c].astype(str).astype("category")
    neurons.to_parquet(layout.neurons, index=False)
    gt.to_parquet(layout.gt_types, index=False)
    pairs.to_parquet(layout.gt_pairs, index=False)
    fam.to_parquet(layout.families, index=False)
    write_json(log, layout.log("groundtruth"))
    print(f"[groundtruth] {len(gt):,} shared types in {log['families']:,} families; "
          f"{len(pairs):,} 1:1 pairs ({log['pairs_1to1_before_inclusion']:,} before inclusion rules); "
          f"by source {log['pairs_1to1_by_source']}")
    if rc:
        print(f"[groundtruth] reviewed-match agreement: {rc['type_h_agree']}/{rc['both_typed']} "
              f"({rc['agree_rate_when_both_typed']:.1%}) of expert-reviewed pairs share type_h")
    return log


if __name__ == "__main__":
    run()
