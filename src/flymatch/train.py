"""Contrastive training for B4 (layers = 0) and the GNN (layers > 0). plan.md 7.3.2-7.3.3.

One step: embed (full graphs if layers > 0, else only the batch rows) -> sample SEED types
uniformly -> up to `per_type_per_side` neurons per dataset per side -> symmetric multi-positive
InfoNCE against in-batch neurons + random training-family negatives. Positives = same type_h
AND side; same type on the other side is masked (neither positive nor negative).
Validation T1 MRR every `eval_every` steps; best checkpoint kept; early stop on patience.
"""
from __future__ import annotations

import copy
import json
import math
import time

import numpy as np
import pandas as pd
import torch

from .config import Layout
from .evaluate import EvalContext, evaluate
from .models.encoder import Encoder
from .models.losses import coral, multi_pos_infonce
from .seeds import seed_families, seed_types
from .splits import load_splits

DEFAULTS = {
    "model": {"layers": 0, "width": 256, "out_dim": 128, "dropout": 0.1, "direction": "directed"},
    "features": {"drop_blocks": []},
    "train": {"max_steps": 3000, "lr": 1e-3, "weight_decay": 1e-4, "warmup_steps": 100,
              "batch_types": 256, "per_type_per_side": 4, "easy_negatives": 1024,
              "tau_init": 0.07, "eval_every": 100, "patience": 10,
              "align": {"method": "none", "weight": 0.1}},
    "seed_fraction": 1.0, "draw": 0,
}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class TrainData:
    def __init__(self, layout: Layout, cfg: dict):
        spec = json.loads(layout.feature_spec.read_text())
        keep = np.ones(len(spec["columns"]), bool)
        for b in cfg["features"]["drop_blocks"]:
            a, z = spec["blocks"][b]
            keep[a:z] = False
        self.X = {ds: torch.as_tensor(np.load(layout.features(ds))[:, keep]) for ds in ("mcns", "banc")}
        self.f_in = int(keep.sum())
        neurons = pd.read_parquet(layout.neurons)
        gt = pd.read_parquet(layout.gt_types)
        splits = load_splits(layout)
        self.splits_sha256 = splits["sha256"]
        fams = seed_families(gt, splits["train"], cfg["seed_fraction"], cfg["draw"])
        self.seed_types = seed_types(gt, fams)
        train_types = set(seed_types(gt, splits["train"]))
        tcode = {t: i for i, t in enumerate(sorted(neurons.type_h.dropna().unique()))}
        side_code = {"left": 0, "right": 1, "center": 2, "unknown": 3}
        self.type_code, self.side_code, self.groups, self.neg_pool = {}, {}, {}, {}
        for ds in ("mcns", "banc"):
            n = neurons[neurons.dataset == ds].sort_values("uid").reset_index(drop=True)
            self.type_code[ds] = n.type_h.map(tcode).fillna(-1).astype(int).to_numpy()
            self.side_code[ds] = n.side.astype(str).map(side_code).to_numpy()
            s = n[n.include & n.type_h.isin(self.seed_types)]
            self.groups[ds] = {k: g.uid.to_numpy() for k, g in s.groupby(["type_h", "side"], observed=True)}
            # negatives: included, TRAINING-family typed neurons only (never untyped, never val/test)
            self.neg_pool[ds] = n.uid[n.include & n.type_h.isin(train_types)].to_numpy()
        both = set(self.groups["mcns"]) & set(self.groups["banc"])
        self.types = sorted({t for t, _ in both})
        self.sides_of = {}
        for t, s in both:
            self.sides_of.setdefault(t, []).append(s)
        self.graphs = None
        if cfg["model"]["layers"] > 0:
            from .graph import load_graphs
            self.graphs = load_graphs(layout, undirected=cfg["model"]["direction"] == "undirected")

    def sample(self, rng, batch_types, per):
        types = rng.choice(len(self.types), size=min(batch_types, len(self.types)), replace=False)
        idx = {"mcns": [], "banc": []}
        for ti in types:
            t = self.types[ti]
            for s in self.sides_of[t]:
                for ds in ("mcns", "banc"):
                    u = self.groups[ds][(t, s)]
                    idx[ds].append(u if len(u) <= per else rng.choice(u, per, replace=False))
        return {ds: np.concatenate(v) for ds, v in idx.items()}


def _schedule(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def embed_all(model, data: TrainData) -> dict:
    out = {}
    for ds in ("mcns", "banc"):
        A = data.graphs[ds] if data.graphs else (None, None)
        out[ds] = model.embed(data.X[ds], *A).numpy()
    return out


def train(layout: Layout, cfg: dict, seed: int, val_ctx: EvalContext | None = None, log=print):
    cfg = merge(DEFAULTS, cfg)
    tc, mc = cfg["train"], cfg["model"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = TrainData(layout, cfg)
    val_ctx = val_ctx or EvalContext.from_layout(layout, "val")
    model = Encoder(data.f_in, mc["width"], mc["out_dim"], mc["layers"], mc["dropout"])
    log_tau = torch.nn.Parameter(torch.tensor(math.log(tc["tau_init"])))
    opt = torch.optim.AdamW(list(model.parameters()) + [log_tau], lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _schedule(s, tc["warmup_steps"], tc["max_steps"]))
    from .baselines.scorers import EmbeddingScorer
    best, best_state, bad, history = -1.0, None, 0, []
    t0 = time.time()
    for step in range(tc["max_steps"]):
        model.train()
        b = data.sample(rng, tc["batch_types"], tc["per_type_per_side"])
        for ds in ("mcns", "banc"):
            neg = rng.choice(data.neg_pool[ds], tc["easy_negatives"], replace=False)
            b[ds] = np.unique(np.concatenate([b[ds], neg]))
        if data.graphs:
            z = {ds: model(data.X[ds], *data.graphs[ds])[torch.as_tensor(b[ds])] for ds in b}
        else:
            z = {ds: model(data.X[ds][torch.as_tensor(b[ds])]) for ds in b}
        ta, tb = data.type_code["mcns"][b["mcns"]], data.type_code["banc"][b["banc"]]
        sa, sb = data.side_code["mcns"][b["mcns"]], data.side_code["banc"][b["banc"]]
        same_t = torch.as_tensor(ta[:, None] == tb[None, :])
        pos = same_t & torch.as_tensor(sa[:, None] == sb[None, :])
        loss = multi_pos_infonce(z["mcns"], z["banc"], pos, log_tau, valid=~same_t)
        if tc["align"]["method"] == "coral":
            # class-conditional: only neurons of shared seed types in the batch
            ma, mb = pos.any(1), pos.any(0)
            loss = loss + tc["align"]["weight"] * coral(z["mcns"][ma], z["banc"][mb])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (step + 1) % tc["eval_every"] == 0 or step + 1 == tc["max_steps"]:
            Z = embed_all(model, data)
            m, _ = evaluate(EmbeddingScorer(Z), val_ctx, filters=False, t1_only=True)
            mrr = m["T1/unfiltered/mean/MRR"]
            history.append({"step": step + 1, "loss": loss.item(), "tau": log_tau.exp().item(),
                            "val_MRR": mrr, "val_R@1": m["T1/unfiltered/mean/R@1"], "sec": round(time.time() - t0, 1)})
            log(f"  step {step + 1:5d} loss {loss.item():.3f} tau {log_tau.exp().item():.3f} "
                f"val MRR {mrr:.3f} R@1 {m['T1/unfiltered/mean/R@1']:.3f} ({time.time() - t0:.0f}s)")
            if mrr > best:
                best, bad = mrr, 0
                best_state = copy.deepcopy(model.state_dict())
                best_step = step + 1
            else:
                bad += 1
                if bad >= tc["patience"]:
                    log(f"  early stop at step {step + 1}; best step {best_step}")
                    break
    model.load_state_dict(best_state)
    return model, embed_all(model, data), {"history": history, "best_val_MRR": best, "best_step": best_step,
                                           "n_seed_types": len(data.seed_types), "n_train_types_used": len(data.types),
                                           "config": cfg, "seed": seed}
