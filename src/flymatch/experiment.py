"""Train one model from an experiment config and evaluate it with the full harness.

    python -m flymatch.run train --exp configs/experiments/gnn_L2.yaml --seed 0 [--device cuda]

Checkpoints every `train.checkpoint_every` steps into the run folder; re-running the same
command after a crash (e.g. a Kaggle session dying) resumes from the last checkpoint.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import yaml

from .baselines.scorers import EmbeddingScorer
from .config import Layout
from .evaluate import EvalContext, bootstrap_ci, evaluate
from .runs import find_or_new, save
from .train import DEFAULTS, merge, train


def load_experiment(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        exp = yaml.safe_load(f) or {}
    for key in ("model_config", "train_config"):          # optional includes
        if key in exp:
            with open(Path(path).parent.parent / exp.pop(key), encoding="utf-8") as f:
                exp = merge(yaml.safe_load(f), exp)
    return exp


def run_experiment(layout: Layout, exp_path: str, seed: int, device: str = "auto", log=print) -> dict:
    exp = load_experiment(exp_path)
    name = exp.pop("name", Path(exp_path).stem)
    cfg = merge(DEFAULTS, {k: v for k, v in exp.items() if k in DEFAULTS})
    ctx = EvalContext.from_layout(layout, "val")
    method = exp.get("method", name)
    run_cfg = {"method": method, "eval_split": "val", "seed": seed, "seed_fraction": cfg["seed_fraction"],
               "draw": cfg["draw"], "train_config": cfg}
    run, status = find_or_new(f"{name}_seed{seed}", run_cfg, ctx.splits_sha256)
    if status == "done":
        log(f"[train] {name} seed {seed}: already finished in {run.name} — skipping")
        import json
        return json.loads((run / "metrics.json").read_text())
    log(f"[train] {name} seed {seed} -> {run.name}{' (resuming)' if status == 'resume' else ''}")
    t0 = time.time()
    model, Z, info = train(layout, cfg, seed=seed, val_ctx=ctx, log=log, ckpt_dir=run, device=device)
    torch.save(model.state_dict(), run / "model.pt")
    m, pq = evaluate(EmbeddingScorer(Z), ctx)
    ci = bootstrap_ci(pq)
    save(run, m, pq, {**ci, "best_step": info["best_step"], "history": info["history"],
                      "device": info["device"], "train_seconds": round(time.time() - t0, 1)})
    log(f"[train] {name} seed {seed}: T1 R@1 {m['T1/unfiltered/mean/R@1']:.3f} (filtered {m['T1/filtered/mean/R@1']:.3f}) "
        f"MRR {m['T1/unfiltered/mean/MRR']:.3f} | T2 macro {m['T2/unfiltered/mean/acc@1_macro']:.3f}")
    return m
