"""Phase 2 driver: B0-B4 on validation through the one harness, plus the shortlist
(hard-negative pool) analysis on training pairs and the Recall@k figure F4.

    python -m flymatch.run baselines [--methods B0,B1,B2,B2+,B3,B4]
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import Layout, load_data_config, write_json  # noqa: E402
from ..evaluate import EvalContext, bootstrap_ci, evaluate  # noqa: E402
from ..runs import aggregate, new_run, save  # noqa: E402
from ..seeds import seed_types  # noqa: E402
from ..splits import load_splits  # noqa: E402
from . import cocoa_style, nblast  # noqa: E402
from .scorers import EmbeddingScorer, RandomScorer, ZSumScorer  # noqa: E402

ALL = ["B0", "B1", "B1+region", "B2", "B2+", "B3", "B4"]
SEEDS = (0, 1, 2)


def _features(layout, blocks):
    spec = json.loads(layout.feature_spec.read_text())
    return {ds: np.concatenate([np.load(layout.features(ds))[:, slice(*spec["blocks"][b])] for b in blocks], 1)
            for ds in ("mcns", "banc")}


def _record(name, cfg, scorer, ctx, store, block=512, extra=None):
    run = new_run(name, cfg, ctx.splits_sha256)
    m, pq = evaluate(scorer, ctx, block=block)
    ci = bootstrap_ci(pq)
    save(run, m, pq, {**ci, **(extra or {})})
    store.setdefault(cfg["method"], pq)
    print(f"[baselines] {name}: T1 R@1 {m['T1/unfiltered/mean/R@1']:.3f} (filtered {m['T1/filtered/mean/R@1']:.3f}) "
          f"MRR {m['T1/unfiltered/mean/MRR']:.3f} | T2 macro acc@1 {m['T2/unfiltered/mean/acc@1_macro']:.3f} "
          f"(filtered {m['T2/filtered/mean/acc@1_macro']:.3f}) -> {run.name}")
    return m, pq, run


def shortlist_analysis(layout: Layout, ks=(10, 25, 50, 100, 200), target=0.85) -> dict:
    """B1 Recall@k on TRAINING 1:1 pairs; k_neg = smallest k with filtered R@k >= target."""
    ctx = EvalContext.from_layout(layout, "train")
    _, pq = evaluate(EmbeddingScorer(_features(layout, ["np_in", "np_out"])), ctx, t1_only=True)
    rows = []
    for k in ks:
        rows.append({"k": k, "R@k_unfiltered": float((pq["rank"] <= k).mean()),
                     "R@k_filtered": float((pq["rank_f"] <= k).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(layout.results / "shortlist_recall_train.csv", index=False)
    ok = df[df["R@k_filtered"] >= target]
    k_neg = int(ok.k.min()) if len(ok) else int(max(ks))
    out = {"k_neg": k_neg, "target": target, "table": df.to_dict("records")}
    write_json(out, layout.results / "shortlist_k_neg.json")
    print(f"[baselines] shortlist (train, B1): " + ", ".join(
        f"R@{r['k']} {r['R@k_unfiltered']:.3f}/{r['R@k_filtered']:.3f}" for r in out["table"]) + f" -> k_neg = {k_neg}")
    return out


def figure_f4(layout: Layout, store: dict):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    ks = np.unique(np.logspace(0, 3, 60).astype(int))
    for name, pq in store.items():
        for a, col in zip(ax, ["rank", "rank_f"]):
            r = pq[col].dropna().to_numpy()
            a.plot(ks, [(r <= k).mean() for k in ks], label=name, lw=1.6)
    for a, t in zip(ax, ["unfiltered", "with hard filters"]):
        a.set_xscale("log"); a.set_xlabel("k"); a.set_title(f"T1 Recall@k on validation ({t}), both directions", fontsize=10)
        a.grid(alpha=.3)
    ax[0].set_ylabel("Recall@k"); ax[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    (layout.results / "figures").mkdir(exist_ok=True)
    fig.savefig(layout.results / "figures" / "F4_recall_at_k_val.png", dpi=130)
    plt.close(fig)


def run(cfg: dict | None = None, methods=None):
    cfg = cfg or load_data_config()
    layout = Layout(cfg)
    methods = methods or ALL
    ctx = EvalContext.from_layout(layout, "val")
    print(f"[baselines] val queries: mcns {len(ctx.queries['mcns']):,} ({(ctx.queries['mcns'].partner >= 0).sum():,} in 1:1 pairs), "
          f"banc {len(ctx.queries['banc']):,}; galleries mcns {len(ctx.sides['mcns'].gallery):,}, banc {len(ctx.sides['banc'].gallery):,}")
    store: dict = {}
    base = {"eval_split": "val", "seed_fraction": np.nan}
    if "B0" in methods:
        for s in SEEDS:
            _record(f"B0_seed{s}", {**base, "method": "B0", "seed": s}, RandomScorer(s), ctx, store)
    b1 = EmbeddingScorer(_features(layout, ["np_in", "np_out"]))
    if "B1" in methods:
        _record("B1", {**base, "method": "B1", "seed": 0, "blocks": ["np_in", "np_out"]}, b1, ctx, store)
    if "B1+region" in methods:
        blocks = ["np_in", "np_out", "region_in", "region_out"]
        _record("B1region", {**base, "method": "B1+region", "seed": 0, "blocks": blocks},
                EmbeddingScorer(_features(layout, blocks)), ctx, store)
    if "B2" in methods or "B2+" in methods:
        nb = nblast.load_scorer(layout)
        if nb is None:
            print("[baselines] B2 skipped: NBLAST file not found in external/")
        else:
            pairs = pd.read_parquet(layout.gt_pairs)
            cov = nblast.pair_coverage(layout, pairs[pairs.family.isin(load_splits(layout)["val"])])
            stats = json.loads((layout.processed / "nblast_stats.json").read_text())
            print(f"[baselines] NBLAST coverage on val pairs: true pair scored {cov['true_pair_scored']:.1%}")
            if "B2" in methods:
                _record("B2", {**base, "method": "B2", "seed": 0}, nb, ctx, store, extra={"coverage": cov, "nblast": stats})
            if "B2+" in methods:
                _record("B2plus", {**base, "method": "B2+", "seed": 0}, ZSumScorer([b1, nb]), ctx, store,
                        extra={"coverage": cov})
    if "B3" in methods:
        splits = load_splits(layout)
        gt = pd.read_parquet(layout.gt_types)
        st = seed_types(gt, splits["train"])
        sc = cocoa_style.build_scorer(pd.read_parquet(layout.neurons), pd.read_parquet(layout.edges), st)
        _record("B3", {**base, "method": "B3", "seed": 0, "seed_fraction": 1.0, "n_seed_types": len(st)},
                sc, ctx, store, block=256)
    if "B4" in methods:
        from ..train import train
        for s in SEEDS:
            tcfg = {"model": {"layers": 0}, "seed_fraction": 1.0, "draw": 0}
            print(f"[baselines] training B4 (MLP) seed {s}")
            model, Z, info = train(layout, tcfg, seed=s, val_ctx=ctx)
            rcfg = {**base, "method": "B4", "seed": s, "seed_fraction": 1.0, "train_config": info["config"]}
            _, _, run = _record(f"B4_seed{s}", rcfg, EmbeddingScorer(Z), ctx, store,
                                extra={"best_step": info["best_step"], "history": info["history"]})
            import torch
            torch.save(model.state_dict(), run / "model.pt")
    shortlist_analysis(layout)
    if store:
        figure_f4(layout, store)
    res = aggregate(ctx.splits_sha256, layout.results / "baselines.csv")
    if len(res):
        show = res[["method", "n_seeds", "T1/unfiltered/mean/R@1", "T1/unfiltered/mean/R@1_sd", "T1/filtered/mean/R@1",
                    "T1/unfiltered/mean/MRR", "T2/unfiltered/mean/acc@1_macro", "T2/filtered/mean/acc@1_macro"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return res


if __name__ == "__main__":
    run()
