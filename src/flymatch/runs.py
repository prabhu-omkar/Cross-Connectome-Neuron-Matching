"""Run folders and aggregation (plan.md 9).

runs/<date>_<name>_<hash>/ : config.json, git_sha.txt, splits_sha256.txt, metrics.json,
ranks.parquet (per-query results). `aggregate` refuses runs made with another split hash.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO, git_sha, write_json

RUNS = REPO / "runs"

HEADLINE = ["T1/unfiltered/mean/R@1", "T1/unfiltered/mean/R@10", "T1/unfiltered/mean/R@100",
            "T1/unfiltered/mean/MRR", "T1/unfiltered/mean/median_rank",
            "T1/filtered/mean/R@1", "T1/filtered/mean/R@10", "T1/filtered/mean/MRR",
            "T2/unfiltered/mean/acc@1_macro", "T2/unfiltered/mean/acc@5_macro", "T2/unfiltered/mean/acc@1_micro",
            "T2/filtered/mean/acc@1_macro",
            "T1/unfiltered/clean/R@1", "T1/unfiltered/clean/n",
            "T1/unfiltered/mcns->banc/R@1", "T1/unfiltered/banc->mcns/R@1"]


def new_run(name: str, config: dict, splits_sha256: str) -> Path:
    h = hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:8]
    d = RUNS / f"{dt.date.today():%Y%m%d}_{name}_{h}"
    d.mkdir(parents=True, exist_ok=True)
    write_json(config, d / "config.json")
    (d / "git_sha.txt").write_text(git_sha() + "\n")
    (d / "splits_sha256.txt").write_text(splits_sha256 + "\n")
    return d


def save(run_dir: Path, metrics: dict, per_query: pd.DataFrame, extra: dict | None = None):
    write_json({**metrics, **(extra or {})}, run_dir / "metrics.json")
    per_query.to_parquet(run_dir / "ranks.parquet", index=False)


def aggregate(splits_sha256: str, out: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(RUNS.glob("*")):
        f = d / "metrics.json"
        if not f.exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        sha = (d / "splits_sha256.txt").read_text().strip()
        if sha != splits_sha256:
            print(f"[aggregate] skipping {d.name}: split hash {sha[:12]}… != frozen {splits_sha256[:12]}…")
            continue
        m = json.loads(f.read_text())
        rows.append({"run": d.name, "method": cfg.get("method"), "split": cfg.get("eval_split"),
                     "seed_fraction": cfg.get("seed_fraction", np.nan), "seed": cfg.get("seed", 0),
                     **{k: m.get(k) for k in HEADLINE},
                     "T1_R@1_ci": m.get("T1/mean/R@1_ci"), "T2_macro_ci": m.get("T2/mean/acc@1_macro_ci")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # latest run per (method, split, seed_fraction, seed)
    df = df.sort_values("run").groupby(["method", "split", "seed_fraction", "seed"], dropna=False).tail(1)
    num = [c for c in HEADLINE if c in df]
    g = df.groupby(["method", "split", "seed_fraction"], dropna=False)
    agg = g[num].mean().add_suffix("")
    sd = g[num].std(ddof=1).add_suffix("_sd")
    res = pd.concat([agg, sd, g.size().rename("n_seeds")], axis=1).reset_index()
    first_ci = g[["T1_R@1_ci", "T2_macro_ci"]].first().reset_index(drop=True)
    res = pd.concat([res, first_ci], axis=1)
    res.to_csv(out, index=False)
    df.to_csv(out.with_name(out.stem + "_all_runs.csv"), index=False)
    return res
