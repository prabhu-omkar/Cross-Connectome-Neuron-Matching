"""Week 8: GNN depth comparison (Kaggle version-2 runs).

    python scripts/week8_compare.py

Paired family-level bootstrap differences in T1 R@1 (unfiltered and filtered), per seed,
same seed vs same seed. Writes results/week8_depth_compare.csv and
results/week8_val_curves.csv (validation MRR every 100 steps, for the patience decision).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flymatch.evaluate import paired_bootstrap  # noqa: E402

RUNS = REPO / "runs"
# Optional cut-off: FLYMATCH_BEFORE="2026-11-10 18:00" reproduces week8_depth_compare.csv (session-2 mlp)
# after session 3 re-ran mlp with the week-9 code.
_b = os.environ.get("FLYMATCH_BEFORE")
BEFORE = datetime.strptime(_b, "%Y-%m-%d %H:%M").timestamp() if _b else None
METHODS = ["mlp", "gnn_L1", "gnn_L2", "gnn_L3"]
PAIRS = [("gnn_L1", "mlp"), ("gnn_L2", "mlp"), ("gnn_L3", "mlp"), ("gnn_L2", "gnn_L1"), ("gnn_L3", "gnn_L2")]


def run_of(method: str, seed: int) -> Path:
    """Most recently finished run of `method` / `seed`, or one finished before `before` (a datetime string
    'YYYY-MM-DD HH:MM', local time) - e.g. to pick the session-2 mlp after session 3 re-ran it."""
    # `method` is a run name ("gnn_L2" -> *_gnn_L2_seed0_*) or a pattern with {seed} ("B3_sf0.05_draw{seed}")
    pat = f"*_{method.format(seed=seed)}_*" if "{seed}" in method else f"*_{method}_seed{seed}_*"
    found = [d for d in RUNS.glob(pat) if (d / "metrics.json").exists()]
    if BEFORE:
        found = [d for d in found if (d / "metrics.json").stat().st_mtime < BEFORE]
    if not found:
        raise FileNotFoundError(f"{method} seed {seed}")
    return max(found, key=lambda d: (d / "metrics.json").stat().st_mtime)


def compare(pairs, out_name, seeds=(0, 1, 2)) -> pd.DataFrame:
    """Paired R@1 differences a - b per seed -> results/<out_name>; prints the per-pair summary."""
    rows = []
    for pair in pairs:
        a, b, ss = (*pair, seeds) if len(pair) == 2 else pair
        for s in ss:
            pa, pb = (pd.read_parquet(run_of(m, s) / "ranks.parquet") for m in (a, b))
            for suf, lab in (("", "unfiltered"), ("_f", "filtered")):
                r = paired_bootstrap(pa, pb, suf=suf, seed=s)
                rows.append({"a": a, "b": b, "seed": s, "metric": f"T1 R@1 {lab}", "diff": r["diff"],
                             "ci_lo": r["ci"][0], "ci_hi": r["ci"][1], "n": r["n"]})
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / out_name, index=False)
    summ = df.groupby(["a", "b", "metric"], sort=False).agg(
        diff_mean=("diff", "mean"), diff_sd=("diff", "std"), ci_lo_min=("ci_lo", "min"), ci_hi_max=("ci_hi", "max"),
        n_seeds=("seed", "size")).reset_index()
    print(summ.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    return df


def main():
    if len(sys.argv) > 2:      # python scripts/week8_compare.py out.csv a:b [a:b@0,1 ...]
        pairs = []
        for p in sys.argv[2:]:
            ab, _, ss = p.partition("@")
            a, b = ab.split(":")
            pairs.append((a, b, tuple(int(x) for x in ss.split(","))) if ss else (a, b))
        compare(pairs, sys.argv[1])
        return
    rows = []
    for a, b in PAIRS:
        for s in (0, 1, 2):
            pa, pb = (pd.read_parquet(run_of(m, s) / "ranks.parquet") for m in (a, b))
            for suf, lab in (("", "unfiltered"), ("_f", "filtered")):
                r = paired_bootstrap(pa, pb, suf=suf, seed=s)
                rows.append({"a": a, "b": b, "seed": s, "metric": f"T1 R@1 {lab}", "diff": r["diff"],
                             "ci_lo": r["ci"][0], "ci_hi": r["ci"][1], "n": r["n"]})
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results" / "week8_depth_compare.csv", index=False)
    summ = df.groupby(["a", "b", "metric"]).agg(diff_mean=("diff", "mean"), diff_sd=("diff", "std"),
                                                ci_lo_min=("ci_lo", "min"), ci_hi_max=("ci_hi", "max")).reset_index()
    print(summ.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    curves = []
    for m in METHODS:
        for s in (0, 1, 2):
            met = json.loads((run_of(m, s) / "metrics.json").read_text())
            for h in met["history"]:
                curves.append({"method": m, "seed": s, **h})
            print(f"{m} seed {s}: best step {met['best_step']}, train {met.get('train_seconds', 0) / 60:.0f} min")
    pd.DataFrame(curves).to_csv(REPO / "results" / "week8_val_curves.csv", index=False)


if __name__ == "__main__":
    main()
