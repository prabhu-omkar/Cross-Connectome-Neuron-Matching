"""Phase 1.3 — stratified family split, frozen and hashed (plan.md 7.1.8 steps 3-5).

Families are assigned to train/val/test 70/10/20 within strata of (dominant super class x
size bucket), balancing the number of types. The file is written once; re-running with
--freeze refuses to overwrite a different split.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from .config import Layout, load_data_config

SPLITS = ("train", "val", "test")
BUCKET_ORDER = ["1", "2", "3-10", "11-100", ">100"]


def family_table(gt: pd.DataFrame) -> pd.DataFrame:
    def dominant(s):
        return s.value_counts().sort_index().idxmax()

    f = gt.groupby("family").agg(
        n_types=("type_h", "size"),
        superclass=("superclass", dominant),
        size_bucket=("size_bucket", lambda s: max(s, key=BUCKET_ORDER.index)),
    ).reset_index()
    f["stratum"] = f.superclass.astype(str) + "|" + f.size_bucket
    return f


def assign(fam: pd.DataFrame, fractions: dict, seed: int) -> dict:
    """Greedy: within each stratum, place families (largest first, random tie order) into
    the split furthest below its target share of types."""
    rng = np.random.default_rng(seed)
    target = np.array([fractions[s] for s in SPLITS])
    out = {s: [] for s in SPLITS}
    for _, g in fam.sort_values("stratum").groupby("stratum", sort=True):
        g = g.iloc[rng.permutation(len(g))]
        g = g.sort_values("n_types", ascending=False, kind="stable")
        have = np.zeros(3)
        total = g.n_types.sum()
        for fam_name, w in zip(g.family, g.n_types):
            deficit = target * (have.sum() + w) - have
            k = int(np.argmax(deficit))
            out[SPLITS[k]].append(fam_name)
            have[k] += w
        _ = total
    return {s: sorted(v) for s, v in out.items()}


def payload_hash(splits: dict) -> str:
    body = {k: splits[k] for k in ("version", "seed", *SPLITS)}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_splits(layout: Layout, verify: bool = True) -> dict:
    s = json.loads(layout.split_file.read_text())
    if verify and payload_hash(s) != s["sha256"]:
        raise ValueError("splits_v1.json does not match its sha256 — the frozen split was edited")
    return s


def family_to_split(splits: dict) -> dict:
    return {f: s for s in SPLITS for f in splits[s]}


def summary(gt: pd.DataFrame, pairs: pd.DataFrame, splits: dict) -> pd.DataFrame:
    f2s = family_to_split(splits)
    g = gt.assign(split=gt.family.map(f2s))
    p = pairs.assign(split=pairs.family.map(f2s))
    a = g.groupby(["split", "superclass", "size_bucket"]).size().rename("types")
    b = p.merge(gt[["type_h", "superclass", "size_bucket"]], on="type_h") \
         .groupby(["split", "superclass", "size_bucket"]).size().rename("pairs_1to1")
    return pd.concat([a, b], axis=1).fillna(0).astype(int).reset_index()


def run(cfg: dict | None = None, freeze: bool = False):
    cfg = cfg or load_data_config()
    layout = Layout(cfg)
    gt = pd.read_parquet(layout.gt_types)
    pairs = pd.read_parquet(layout.gt_pairs)
    fam = family_table(gt)
    parts = assign(fam, cfg["split_fractions"], cfg["split_seed"])
    splits = {"version": 1, "seed": cfg["split_seed"], **parts}
    splits["sha256"] = payload_hash(splits)

    if layout.split_file.exists():
        old = load_splits(layout)
        if old["sha256"] == splits["sha256"]:
            print(f"[split] existing split matches (sha256 {old['sha256'][:12]}…)")
        elif freeze:
            raise SystemExit(f"[split] {layout.split_file} exists with a DIFFERENT hash "
                             f"({old['sha256'][:12]}… vs {splits['sha256'][:12]}…). The split is frozen; "
                             "refusing to overwrite. Delete it by hand only if you mean to re-freeze.")
        else:
            print(f"[split] WARNING: regenerated split differs from frozen one "
                  f"({old['sha256'][:12]}… vs {splits['sha256'][:12]}…); frozen file kept")
        splits = old
    elif freeze:
        with open(layout.split_file, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=1)
        (layout.splits / "splits_v1.sha256").write_text(splits["sha256"] + "\n")
        print(f"[split] froze {layout.split_file} sha256 {splits['sha256']}")
    else:
        print("[split] dry run (use --freeze to write)")

    s = summary(gt, pairs, splits)
    s.to_csv(layout.results / "split_summary.csv", index=False)
    tot = s.groupby("split")[["types", "pairs_1to1"]].sum()
    fam_n = {k: len(splits[k]) for k in SPLITS}
    print(tot.assign(families=pd.Series(fam_n), types_frac=tot.types / tot.types.sum()).to_string())
    strata = fam.groupby("stratum").size()
    cover = s[s.types > 0].groupby(["superclass", "size_bucket"]).split.nunique()
    missing = cover[cover < 3]
    print(f"[split] {len(strata)} strata; {len(missing)} (superclass, bucket) cells not in all three splits"
          + (f": {list(missing.index)}" if len(missing) else ""))
    return splits


if __name__ == "__main__":
    import sys
    run(freeze="--freeze" in sys.argv)
