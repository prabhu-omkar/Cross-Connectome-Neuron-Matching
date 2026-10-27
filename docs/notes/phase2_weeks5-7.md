# Phase 2 (weeks 5–6) — baselines and evaluation harness

All numbers are on **validation** (669 one-to-one pairs; 12,418 MaleCNS and 7,243 BANC type-level
queries; galleries are all included neurons: 159,632 MaleCNS and 116,553 BANC). They come from
`results/baselines.csv`, which `python -m flymatch.run baselines` builds from `runs/`. Split sha256 `260be072…`.
Test was not touched.

## Gate G2 — numbers to beat

| Method | T1 R@1 | ±sd | 95% CI (families) | T1 R@10 | MRR | T1 R@1 filtered | T2 macro acc@1 | T2 filtered | Clean R@1 |
|---|---|---|---|---|---|---|---|---|---|
| B0 random (3 seeds) | 0.000 | 0.000 | — | 0.000 | 0.000 | 0.001 | 0.001 | 0.035 | 0.000 |
| B1 √ neuropil profiles | 0.347 | — | 0.30–0.40 | 0.584 | 0.430 | 0.493 | 0.228 | 0.326 | 0.276 |
| B1 + region summary | 0.346 | — | 0.30–0.40 | 0.575 | 0.426 | 0.491 | 0.219 | 0.309 | 0.281 |
| B2 NBLAST (published) | 0.289 | — | 0.24–0.34 | 0.310 | 0.298 | 0.296 | 0.358 | 0.352 | 0.338 |
| B2+ = B1 + NBLAST (z-sum) | 0.396 | — | 0.35–0.45 | 0.635 | 0.477 | 0.531 | 0.401 | 0.460 | 0.434 |
| B3 cocoa-style, 100% seeds | 0.396 | — | 0.36–0.43 | 0.855 | 0.597 | **0.830** | **0.574** | **0.610** | 0.412 |
| B4 MLP (3 seeds) | **0.616** | 0.007 | 0.55–0.67 | 0.855 | **0.701** | 0.661 | 0.440 | 0.430 | **0.580** |

T1 metrics average both directions (MaleCNS→BANC and BANC→MaleCNS). "Filtered" means the same
harmonised super class, the same side (or either `center`), and a compatible NT. Figure:
`results/figures/F4_recall_at_k_val.png`.

**What the GNN must beat:** B4 on unfiltered T1 (61.6 ± 0.7% R@1, MRR 0.70), and B3 on filtered T1
(83.0%) and on T2 (57.4%).

## Findings
1. **Ties change B1.** Ranks now share ties (average rank). 8% of validation queries tie their true
   partner exactly, with a median of 973 other neurons: many VNC and GNG neurons have all their synapses in one
   neuropil, so their √-profiles are identical. The Phase 1 code counted a tie as a win (B1 = 40.5% R@1
   MaleCNS→BANC). The fair number is 36.0% (34.7% averaged over directions). The 21 Oct 29.3% probably had
   the same optimistic bias.
2. **B3 cannot see side.** Seed types carry no side, so the left and right copies of a partner get
   identical vectors and tie: R@1 is 39.6% but R@2 is 73%. The side filter removes the tie, giving 83.0%.
   Label-projected connectivity is the strongest baseline once side is known, and the strongest
   at type level (T2). The GNN's label-projected feature block (plan.md 7.1.5) should therefore help, and
   this is exactly what the seed ablation will stress.
3. **The MLP is already much better than every zero-learning baseline** (61.6% vs 34.7% R@1). It peaks
   early (best checkpoint at steps 400–500, early stop around 1,400–1,500) and then validation MRR slowly
   falls, which means it overfits the training types. Its features include annotations (hemilineage, super class),
   so ablation A2 (drop hemilineage) matters before claiming anything.
4. **B4 is weaker than B3 at type level** (T2 macro 44% vs 57%). It is tuned for neuron-level retrieval
   (checkpoints chosen by T1 MRR), and its gain over B1 comes mostly from T1.
5. **The published NBLAST matrix is reliable only for some classes.** It holds the top-100 MaleCNS v0.9
   matches per BANC neuron (16.4 M rows). The true partner is in the list for only 31% of validation
   pairs (31% of all pairs). This is not an ID problem (`root_888` and `query_root_id` give the
   same coverage) and not a left/right flip (the contralateral homolog is scored less often, 10%). By BANC super class,
   the top-1 match has the right type for 60% of descending and 58% of ascending neurons (median top score 0.48), but only 11–17%
   for central-brain, VNC and sensory neurons, whose best matches score negative and are mostly unrelated optic-lobe neurons.
   This looks like a limitation of the published product (perhaps registration in those regions). B2 is
   reported as-is with this caveat. It is still complementary: B1 + NBLAST beats B1 by 5 points R@1 and 17
   points T2.
6. **Hard-negative pool:** B1 shortlist on training pairs, filtered R@k: 72% (k=10), 79% (25), 86% (50),
   92% (100), 97% (200). **k_neg = 50** is the smallest k with ≥ 85% (`shortlist_k_neg.json`).

## Week 6 model code
- `graph.py`: sparse COO `A_in` / `A_out` built from fractional weights, plus an undirected variant (A6) and the
  degree-preserving permutation (C4).
- `models/`: `DirLayer` (uses `torch.sparse.mm`, no per-edge messages), the shared `Encoder` (layers = 0 gives B4),
  multi-positive InfoNCE with a "same type, other side" mask, CORAL, and the gradient-reversal layer.
- `train.py`: type-balanced sampling, training-family-only negatives, learnable τ, warm-up and cosine
  schedule, early stopping on validation MRR, seed fractions (`seeds.py`, nested 5 ⊂ 10 ⊂ 25 ⊂ 50 ⊂ 100%).
- **Full-graph CPU check** (`results/gnn_smoke_cpu_L2_w256.json`): 2 layers, width 256, both graphs
  (7.8 M edges) forward plus backward works, at 5.4 GiB peak RAM and 42 s per step on the laptop. GNN training
  (≈ 3,000 steps) therefore needs the Kaggle GPU; the MLP trains on CPU in about 5 min per seed.

## Next (week 7)
Package `data/processed` + `data/interim` for Kaggle, add checkpoint/resume, and run the first GNN
(layers 1–3) against B4. Before claiming H2, run A2 (B4 without hemilineage).

## Week 7 — training infrastructure (done; GPU runs pending)
- `train.py`: GPU/CPU device choice; checkpoint `last.pt` every 200 steps (atomic write) with model,
  optimiser, scheduler, τ, RNG states, best checkpoint and history; resumes automatically.
- `python -m flymatch.run train --exp configs/experiments/<exp>.yaml --seeds 0,1,2` → run folder with
  `model.pt`, `metrics.json` (full harness on val + family CIs), `ranks.parquet`. Experiments ready:
  `mlp`, `gnn_L1`, `gnn_L2`, `gnn_L3`.
- **Resume tested:** a 1-layer GNN smoke run was stopped on purpose after the step-4 checkpoint
  (`FLYMATCH_SIMULATE_CRASH_AT=4`). Re-running the same command continued from step 4 and finished.
- `python -m flymatch.run package` → `dist/flymatch_kaggle.zip` (84 MB: code + processed data, no raw CSVs).
  Tested from a fresh unzip: the smoke GNN trains end to end. Kaggle steps: `INSTRUCTIONS.md`;
  notebook cell: `kaggle/kaggle_train.py`.
- Remaining for week 7 (user, on Kaggle): run `mlp` (should reproduce B4 ≈ 0.62) and `gnn_L1–L3`,
  3 seeds each; bring back `runs_out.zip` and run `aggregate`.
