"""Seed-ablation curve (plan.md 7.4.1) — the central result.

    python -m flymatch.run seed-ablation [--methods B3,B4] [--fractions 0.05,0.1,0.25,0.5] [--draws 0,1,2]

For each draw r, nested seed sets 5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100% of the TRAINING families
(seeds.py). Validation never changes.
  B3  cocoa-style label-projected connectivity, seed types = the draw's seed set (deterministic).
  B4  MLP (encoder with 0 GNN layers), trained on the draw's seed set; training seed = draw.
  GNN any run whose experiment config sets `method: GNN` (Kaggle; configs gnn_sa_*.yaml).
At 100% all draws coincide, so the existing B3 run (seed 0) and B4 seeds 0-2 are reused.
At 0% no cross-dataset positives exist: B1 is plotted as the value (plan 7.4.1.3).

Every run lands in runs/ (skip if finished, resume B4 from last.pt), then the table
results/seed_ablation.csv and figure results/figures/F5_seed_ablation.png are rebuilt
from ALL matching runs, so GNN runs brought back from Kaggle are picked up automatically.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .baselines import cocoa_style  # noqa: E402
from .baselines.scorers import EmbeddingScorer  # noqa: E402
from .config import Layout, load_data_config  # noqa: E402
from .evaluate import EvalContext, bootstrap_ci, evaluate  # noqa: E402
from .runs import RUNS, find_or_new, save  # noqa: E402
from .seeds import seed_families, seed_types  # noqa: E402
from .splits import load_splits  # noqa: E402

FRACTIONS = (0.05, 0.10, 0.25, 0.50)
DRAWS = (0, 1, 2)
CURVE_METHODS = ("B3", "B4", "B4_A2", "GNN")
GNN_FULL = "gnn_L2"   # week-8 depth choice; change here if L1 is chosen (its seed_ablation configs must match)
FULL_POINT = {GNN_FULL: "GNN", "mlp_A2": "B4_A2"}   # 100% runs made by `run train` under another method name
METRICS = {"R@1": "T1/unfiltered/mean/R@1", "R@1_filtered": "T1/filtered/mean/R@1",
           "MRR": "T1/unfiltered/mean/MRR", "T2_macro": "T2/unfiltered/mean/acc@1_macro",
           "T2_macro_filtered": "T2/filtered/mean/acc@1_macro"}


def _done(run):
    return json.loads((run / "metrics.json").read_text())


def run_b3(layout, ctx, f, d, gt, splits, neurons, edges, log=print):
    fams = seed_families(gt, splits["train"], f, d)
    st = seed_types(gt, fams)
    cfg = {"eval_split": "val", "method": "B3", "seed": d, "seed_fraction": f, "draw": d,
           "n_seed_families": len(fams), "n_seed_types": len(st)}
    run, status = find_or_new(f"B3_sf{f:g}_draw{d}", cfg, ctx.splits_sha256)
    if status == "done":
        log(f"[seed-ablation] B3 {f:g} draw {d}: done in {run.name}")
        return _done(run)
    sc = cocoa_style.build_scorer(neurons, edges, st)
    m, pq = evaluate(sc, ctx, block=256)
    save(run, m, pq, bootstrap_ci(pq))
    log(f"[seed-ablation] B3 {f:g} draw {d} ({len(st)} seed types): R@1 {m[METRICS['R@1']]:.3f} "
        f"(filtered {m[METRICS['R@1_filtered']]:.3f}) T2 {m[METRICS['T2_macro']]:.3f}")
    return m


def run_b4(layout, ctx, f, d, log=print, device="cpu", variant="B4"):
    """variant B4 = MLP; B4_A2 = MLP without the hemilineage block (a shared cross-dataset
    annotation, so it is label information that no seed fraction removes)."""
    import torch
    from .train import train
    tcfg = {"model": {"layers": 0}, "seed_fraction": f, "draw": d}
    if variant == "B4_A2":
        tcfg["features"] = {"drop_blocks": ["hemilineage"]}
    cfg = {"eval_split": "val", "method": variant, "seed": d, "seed_fraction": f, "draw": d, "train_config": tcfg}
    run, status = find_or_new(f"{variant}_sf{f:g}_draw{d}", cfg, ctx.splits_sha256)
    if status == "done":
        log(f"[seed-ablation] {variant} {f:g} draw {d}: done in {run.name}")
        return _done(run)
    model, Z, info = train(layout, tcfg, seed=d, val_ctx=ctx, log=log, ckpt_dir=run, device=device)
    torch.save(model.state_dict(), run / "model.pt")
    m, pq = evaluate(EmbeddingScorer(Z), ctx)
    save(run, m, pq, {**bootstrap_ci(pq), "best_step": info["best_step"], "history": info["history"]})
    log(f"[seed-ablation] {variant} {f:g} draw {d}: R@1 {m[METRICS['R@1']]:.3f} "
        f"(filtered {m[METRICS['R@1_filtered']]:.3f}) T2 {m[METRICS['T2_macro']]:.3f} best step {info['best_step']}")
    return m


def collect(splits_sha256: str) -> pd.DataFrame:
    """One row per finished run that belongs on the curve (latest run per method/fraction/draw)."""
    rows = []
    for d in sorted(RUNS.glob("*")):
        f = d / "metrics.json"
        if not f.exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        if (d / "splits_sha256.txt").read_text().strip() != splits_sha256 or cfg.get("eval_split") != "val":
            continue
        meth = cfg.get("method")
        meth = FULL_POINT.get(meth, meth)
        if meth == "B1":
            fr, draw = 0.0, 0
        elif meth in CURVE_METHODS and cfg.get("seed_fraction") is not None and not pd.isna(cfg["seed_fraction"]):
            fr = float(cfg["seed_fraction"])
            # at 100% every draw is the same seed set; the B4/GNN repeats are training seeds
            draw = cfg.get("draw", cfg.get("seed", 0)) if fr < 1 else cfg.get("seed", 0)
        else:
            continue
        m = json.loads(f.read_text())
        rows.append({"run": d.name, "method": meth, "seed_fraction": fr, "draw": int(draw),
                     **{k: m.get(v) for k, v in METRICS.items()}})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("run").groupby(["method", "seed_fraction", "draw"]).tail(1)


def table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "seed_fraction"])
    out = pd.concat([g[list(METRICS)].mean(), g[list(METRICS)].std(ddof=1).add_suffix("_sd"),
                     g.size().rename("n_draws")], axis=1).reset_index()
    return out.sort_values(["method", "seed_fraction"])


def figure(tab: pd.DataFrame, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    b1 = tab[tab.method == "B1"]
    style = {"B3": ("cocoa-style (B3)", "s", "--"), "B4": ("MLP (B4)", "o", "-"),
             "B4_A2": ("MLP, no hemilineage (B4 A2)", "o", ":"), "GNN": ("GNN", "D", "-")}
    for a, (col, lab) in zip(ax, [("R@1", "T1 Recall@1 (unfiltered)"), ("T2_macro", "T2 type accuracy@1 (macro)")]):
        for meth, (name, mk, ls) in style.items():
            t = tab[tab.method == meth]
            if t.empty:
                continue
            sd = t[f"{col}_sd"].fillna(0)
            a.errorbar(t.seed_fraction * 100, t[col], yerr=sd, marker=mk, ls=ls, capsize=3, label=name)
        if len(b1):
            a.axhline(float(b1[col].iloc[0]), color="grey", ls=":", label="B1 (no seeds; 0% value)")
        a.set_xscale("log"); a.set_xticks([5, 10, 25, 50, 100]); a.set_xticklabels(["5", "10", "25", "50", "100"])
        a.set_xlabel("seed fraction of training families (%)"); a.set_ylabel(lab); a.grid(alpha=.3)
        a.set_ylim(0, 1)
    ax[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Seed ablation on validation (mean ± sd over 3 nested draws)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(cfg: dict | None = None, methods=("B3", "B4"), fractions=FRACTIONS, draws=DRAWS, device="cpu", log=print):
    cfg = cfg or load_data_config()
    layout = Layout(cfg)
    ctx = EvalContext.from_layout(layout, "val")
    splits = load_splits(layout)
    gt = pd.read_parquet(layout.gt_types)
    if "B3" in methods:
        neurons, edges = pd.read_parquet(layout.neurons), pd.read_parquet(layout.edges)
        for d in draws:
            for f in fractions:
                run_b3(layout, ctx, f, d, gt, splits, neurons, edges, log)
    for variant in ("B4", "B4_A2"):
        if variant in methods:
            for d in draws:
                for f in fractions:
                    run_b4(layout, ctx, f, d, log, device, variant)
    df = collect(ctx.splits_sha256)
    tab = table(df)
    out = layout.results / "seed_ablation.csv"
    tab.to_csv(out, index=False)
    df.to_csv(out.with_name("seed_ablation_all_runs.csv"), index=False)
    figure(tab, layout.results / "figures" / "F5_seed_ablation.png")
    log(tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return tab
