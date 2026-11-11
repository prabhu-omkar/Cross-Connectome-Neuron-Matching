# Phase 3 — weeks 8–9: the model is chosen (Gate G3)

All numbers: validation split, frozen split `260be072…`, mean ± sd over 3 seeds unless noted. Every number below
comes from a script; the source file is named in each section. The diary (`PROJECT_DIARY.md`, weeks 8–9) has the
full story.

## Final model

**`gnn_L2`**: 2 directed message-passing layers, width 256, output 128, dropout 0.1, multi-positive InfoNCE,
early stopping on validation MRR, patience 5. Hard negatives and CORAL are **not** adopted.

| Method | R@1 | R@1 filtered | MRR | T2 macro |
|---|---|---|---|---|
| B1 neuropil profile cosine | 0.347 | 0.493 | 0.430 | 0.228 |
| B3 cocoa-style, 100 % seeds | 0.396 | **0.830** | 0.597 | 0.574 |
| B4 MLP | 0.616 ± 0.007 | 0.661 | 0.701 | 0.440 |
| **GNN, 2 layers** | **0.777 ± 0.007** | 0.813 | **0.835** | **0.578** |

Source: `results/baselines.csv`.

## Hypotheses so far

Source: `results/week8_depth_compare.csv`, `results/week9_compare.csv`, `results/week9_mlp_variants.csv`.
Paired family bootstrap, per seed.

| Hypothesis | Test | Result | Holds? |
|---|---|---|---|
| H2: message passing helps | GNN − MLP | +16.1 pts R@1, CIs [+0.12, +0.20] | ✅ |
| H2 without the annotation | GNN without hemilineage − MLP with hemilineage | +10.5, [+0.07, +0.15] | ✅ |
| H5: gain comes from real wiring | C4 edge-permuted GNN − MLP | +0.4, [−0.02, +0.03] | ✅ |
| H3: label efficiency | GNN − B3 at 5 % seeds, filtered | +47.5, [+0.39, +0.55] | ✅ |
| H3 | GNN at 5 % − B3 at 50 %, filtered | −1.8, [−0.09, +0.04] (n.s.) | ≈ 10× fewer labels |
| Labels matter | C5 re-paired labels (MLP, fixed protocol) − MLP | −20.9, [−0.26, −0.17] | ✅ |

H2 without the annotation: hemilineage is worth 5.6 points to the GNN and 7.7 to the MLP.

## Depth

| Comparison | Δ R@1 |
|---|---|
| 2 − 1 layers | +2.2 ± 1.3 |
| 3 − 2 layers | −0.4 ± 0.9, at 1.9× the time |

## Tried and dropped

| Variant | MLP (3 seeds) | GNN (seed 0) |
|---|---|---|
| Hard negatives (B1 top-50) | +0.9, CIs include 0, 10× slower steps | 0.0 |
| CORAL | +0.2 | −0.1 |

## Protocol decisions made in these weeks

- **Patience 5 for new runs; week-7 runs used 10.** Replaying the gnn_L2 curves gives the identical best
  checkpoint (`results/patience_sim.csv`).
- **Corrupted-label controls are never early-stopped on validation.** Validation uses the true labels; the first
  `mlp_C5` picked step 100 on every seed. They use `train.select: last` with `train.stop_at` = the real model's
  median best step.
- **The C5 expectation is "≈ feature-only level (B1)", not "≈ chance".** The input features are already
  comparable across datasets.
- **"Latest run" = most recently finished (`metrics.json` time), not the folder name.** Kaggle dates folders in
  UTC, and a folder name ends in a hash.
- **Two Kaggle code versions:** session 2 used the week-7 package, session 3 used `fb47b59`. The MLP sanity score
  is identical (0.616), so the runs are comparable.

## Open

- GNN at 25 % and 50 % seeds (Kaggle session 4).
- `match` (abstention + assignment) on the final GNN. It needs more laptop RAM than a background
  shell allows; run it from a terminal.
- C3 (within-dataset left↔right ceiling); GNN + B3 score ensemble; hemibrain transfer (H4); test set, once.
