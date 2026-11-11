# 📓 Project Diary — Cross-Connectome Neuron Matching

A running log of what was done, what was found, what was decided and why. Newest entries are at the
bottom. Week numbers follow the 16-week plan in [`docs/plan.md`](docs/plan.md).

| Phase | Weeks | Status |
|---|---|---|
| 1 · Data foundation | 1–4 | ✅ done — split frozen (Gate G1) |
| 2 · Baselines + evaluation harness | 5–6 | ✅ done — `baselines.csv` (Gate G2) |
| 3 · The model | 7–10 | 🟡 week 7 code done; GPU runs pending |
| 4 · Experiments + controls | 11–13 | ⬜ |
| 5 · Web tool | 13–15 | ⬜ |
| 6 · Write-up + release | 15–16 | ⬜ |

---

## Before week 1 — feasibility (21 Oct 2026)

**Did**
- Downloaded the FlyWire Codex exports for **MaleCNS v1.0** (166,700 neurons, 9.04 M connection rows) and
  **BANC v888** (158,262 neurons, 3.99 M rows).
- Built a quick zero-learning baseline: cosine of √-normalised neuropil input/output profiles.

**Found**
- Recall@1 29.3%, Recall@10 47.1%, median rank 15; shuffled ≈ 0%. Connectivity alone carries identity
  signal, so the project is feasible.

---

## Planning day (26 Oct 2026, morning)

**Did**
- Profiled both exports in detail and wrote the full 16-week plan (`docs/plan.md`).

**Decided**
- The female dataset is **BANC**, not FlyWire FAFB: it covers brain + nerve cord, the same scope as MaleCNS.
- **Harmonise before anything else.** Re-threshold to ≥ 5 synapses per pair; map neuropil and
  super-class vocabularies; use fractional edge weights and within-dataset rank features.
- Ground truth = harmonised cell types, with FlyWire/MANC names as validated fallbacks.
- Split by **type family**, 70/10/20, frozen and hashed; test is touched once.
- cocoa does not support BANC, so baseline 3 is a re-implemented cocoa-style label projection.
- The central result is the **seed-ablation curve**. Adversarial alignment is optional (CORAL first).

**Found**
- The 21 Oct note was wrong about thresholds: MaleCNS keeps pairs ≥ **5** synapses, BANC ≥ **3**.
- BANC detects 3.8× fewer synapses (23.6 M vs 89.9 M).
- BANC root IDs are 18 digits and silently change if parsed as float.

---

## Week 1 — repository, ingest, vocabulary maps (26 Oct 2026)

**Did**
- Created the `flymatch` repository and a Python 3.11 virtual environment (no conda on the laptop).
- `io.py` / `ingest.py`: every ID is read as text and converted to int64 with a round-trip check.
  Ingest **fails loudly** on any value missing from the maps.
- Hand-wrote `maps/` for neuropils (245 rows), super classes, NT and hemilineages.
- Inclusion rules: harmonised class known, not a fragment, side known, ≥ 20 synapses.
- Downloaded from Harvard Dataverse: the BANC metadata (proofread flags, `sexually_dimorphic`), the expert-reviewed
  BANC↔MaleCNS matches, somas, and the 912 MB NBLAST matrix. One download broke mid-way and was resumed.

**Found**
- 🔎 **BANC puts > 99% of its leg-neuropil synapses under `VNC_T1_ProNm` / `VNC_T2_MesoNm` /
  `VNC_T3_MetaNm`**, not `LegNp_*` (`LegNp_T1_L`: 1,444 synapses vs `VNC_T1_ProNm_L`: 976,400). The plan
  assumed the VNC neuropils were shared by name. They are now mapped, together with the tectulum regions,
  `AMNp` → `Ov` (old name of the ovoid) and `ABDNM` → `ANm`.
- No edge points to a neuron missing from the tables; 56 MaleCNS self-loop rows were dropped.

---

## Week 2 — harmonisation (26 Oct 2026)

**Did**
- `harmonise.py`: both datasets re-thresholded to ≥ 5 synapses per pair; fractional weights `w_in`,
  `w_out`; per-neuron totals.
- Harmonisation report (figure F2) and a grid search over thresholds, both on **training pairs only**.
- Validated every neuropil merge: Pearson r of per-neuron synapse fractions across training pairs, keep if r ≥ 0.5.

**Found**
- Re-thresholding removes **1.51 M of 3.04 M BANC pairs** (5.07 M synapses).
- Surprise: **equal thresholds widen the gap.** The median MaleCNS/BANC degree ratio goes from 1.63 at the export
  thresholds to **3.03 at 5/5**. MaleCNS ≥ 15 with BANC ≥ 5 gives 1.06; kept as ablation A9.
  Fractional weights and rank features carry the comparison.
- 20 of 22 merges pass with r = 0.64–0.94. **`mVAC_T2` fails (r ≈ 0.25) → excluded.**
- 97.0% (MaleCNS) and 99.6% (BANC) of synapses land in the 87 canonical neuropils.

---

## Week 3 — features and ground truth (26 Oct 2026)

**Did**
- `features.py`: 462 features per neuron, the same column order in both datasets, plus an L↔R mirror permutation.
- `groundtruth.py`: harmonised types (primary → FlyWire → MANC), fallback validation, one-to-one pairs,
  clean-subset flags.

**Found**
- **9,570 shared types**; **6,765 one-to-one pairs** (plan estimate 6,774): 5,056 from primary names, 1,567 from
  FlyWire, 142 from MANC.
- The fallbacks are trustworthy: 99.6% super-class agreement for FlyWire and 99.5% for MANC (primary: 99.5%).
  Only 10 fallback types were dropped individually.
- ⚠️ Only **53%** of the BANC team's 2,194 expert-reviewed BANC↔MaleCNS matches share our harmonised type.
  The disagreements are **naming schemes, not wrong labels**: `R7`/`R8` vs `R7y`/`R7p`/`R8y`, `EFFabg*` vs
  `MNad*`, and ORN subtypes. 77% are sensory and 16% motor, so those classes are thin in the ground truth.
  Recorded as a limitation.
- Clean subset: MaleCNS `Reviewed` 54,062 neurons; BANC proofread 122,177. Only 977 pairs are clean on both
  sides, so the confidence intervals there will be wide.
- 804 BANC neurons flagged `female-specific` become the biological "no counterpart" test set.

---

## Week 4 — families, frozen split, B1 (26 Oct 2026)

**Did**
- Families: union-find over shared name stems, co-listed alternative names and MaleCNS supertypes → **3,610 families**.
- Greedy stratified split by super class × size bucket; **frozen** as `splits_v1.json`,
  sha256 `260be072aa62…`. Tests check that families are disjoint and that stem siblings never straddle splits.
- One command regenerates everything: `python -m flymatch.run phase1` (~7 min).

**Found**
- Split: train 6,696 / val 956 / test 1,918 types (69.97 / 9.99 / 20.04%). One-to-one pairs: 4,746 / 669 / 1,350.
- B1 on validation came out higher than on 21 Oct. The neuropil merges explain most of it: +7 points overall
  and +31 points for ascending neurons. The merges *lower* B1 for VNC-intrinsic and motor neurons, whose
  profiles collapse into one big leg neuropil. That is a weighting problem for the model, not a map error.

**Gate G1 ✅ — split frozen on time.**

---

## Week 5 — the evaluation harness, B0–B2 (26 Oct 2026)

**Did**
- `evaluate.py`: one harness for every method.
  - T1 (exact partner) and T2 (cell type), in both directions.
  - Hard filters, the clean subset, and breakdowns by super class and type size.
  - Bootstrap CIs over **families**, and a paired bootstrap for comparing methods.
- Tested on synthetic data: identical embeddings → 100%; random → chance.
- Got a **neuPrint token** for the hemibrain control (stored as a Windows user environment variable, never in the repo).
- B0 random, B1 profiles, B2 NBLAST, B2+ (B1 + NBLAST).

**Found**
- 🔎 **Ties matter.** 8% of validation queries tie their true partner exactly, with a median of 973 other neurons
  (single-neuropil profiles). The first B1 code counted a tie as a win. With average-tie ranks, B1
  (MaleCNS→BANC) is **36.0%, not 40.5%**. The 21 Oct number probably had the same optimistic bias.
- 🔎 **The published NBLAST matrix is weak for most classes.** The true partner is in its top-100 list for only
  **31%** of pairs. It isn't an ID or left/right problem (both checked). It works for ascending/descending neurons
  (≈ 60% top-1 type accuracy) but not for central-brain or VNC neurons. Even so, B1 + NBLAST beats B1.
- Hard-negative pool: B1 filtered Recall@50 on training pairs is 86% → **k_neg = 50**.

---

## Week 6 — B3, B4 and the GNN skeleton (26 Oct 2026)

**Did**
- B3 cocoa-style label-projected connectivity, re-implemented (sparse matrices, seed types from training only).
- B4: the MLP (the encoder with 0 layers) trained with multi-positive InfoNCE and type-balanced sampling, 3 seeds.
- `graph.py` (sparse `A_in` / `A_out`), `models/` (DirLayer, Encoder, InfoNCE, CORAL, gradient reversal),
  `train.py`, nested seed sets.
- Full-graph check: a 2-layer GNN forward + backward over both graphs works on the CPU in **5.4 GiB**, but at
  42 s per step → GNN training goes to Kaggle.

**Found** (validation, T1 R@1 averaged over directions)

| B0 | B1 | B2 | B2+ | B3 | B3 filtered | **B4 MLP** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0.0% | 34.7% | 28.9% | 39.6% | 39.6% | **83.0%** | **61.6 ± 0.7%** |

- The MLP is far ahead of every zero-learning method. It peaks early (around step 400–500) and then slowly
  overfits to the training types.
- B3 is **side-blind**: left and right copies tie (R@1 40%, R@2 73%). The side filter takes it to 83%, and it is
  the best at type level (T2 57%).
- To check: the MLP uses annotations (hemilineage). Ablation A2 must run before any "message passing helps" claim.

**Gate G2 ✅ — `results/baselines.csv` complete.**

---

## Week 7 — training infrastructure for the GPU (27 Oct 2026)

**Did**
- `train.py`: GPU/CPU device selection; checkpoint `last.pt` every 200 steps (written atomically) with model,
  optimiser, scheduler, temperature, RNG states, best weights and history; automatic resume.
- `run train --exp … --seeds …`: config-driven runs, full validation evaluation, finished runs skipped on re-run.
- Experiments: `mlp`, `gnn_L1`, `gnn_L2`, `gnn_L3`.
- **Resume tested on purpose:** a 1-layer GNN smoke run was "crashed" after the step-4 checkpoint. Re-running the
  same command resumed at step 4 and finished.
- `run package` → `flymatch_kaggle.zip` (84 MB). Tested from a fresh unzip: trains end to end.
- Kaggle guide: [`INSTRUCTIONS.md`](INSTRUCTIONS.md).
- Repository published on GitHub with a detailed README, this diary and the Apache-2.0 licence. The raw exports
  stay out of git (too large; download instructions in the README).
- Kaggle package published as a public dataset (`omkarprabhuuuu/flymatch`) with a public one-click notebook
  (`flymatch-train`); the code is also in `kaggle/flymatch_train.ipynb`.

**Next**
- Run `mlp` (it should reproduce ≈ 0.62) and `gnn_L1–L3` × 3 seeds on Kaggle; bring back `runs_out.zip`;
  `run aggregate`.
- Week 8: GNN vs MLP; run A2 (no hemilineage).

---

## Week 8 — GNN depth, and the first half of the seed-ablation curve (10–11 Nov 2026)

**Did**
- Kaggle notebook version 2 (GPU T4, 11.2 h, no failures): `mlp`, `gnn_L1`, `gnn_L2`, `gnn_L3` × 3 seeds.
  Brought back `runs_out.zip`; all 12 runs carry the frozen split hash; `run aggregate`.
- `scripts/week8_compare.py`: paired family-bootstrap differences, same seed vs same seed
  (`results/week8_depth_compare.csv`). `scripts/patience_sim.py` replays the validation curves under smaller
  early-stopping patience (`results/patience_sim.csv`).
- New command `run seed-ablation` (`seed_ablation.py`): cocoa-style (B3) and the MLP (B4) at 5/10/25/50 % of the
  training families × 3 nested draws, on the laptop CPU. Also the MLP without hemilineage (`B4_A2`) on the same grid.
  Output `results/seed_ablation.csv` and figure F5. GNN seed-ablation configs `gnn_sa_{05,10,25,50}` (draw = seed).

**Found** (validation, mean ± sd over 3 seeds)

| Method | R@1 | R@1 filtered | MRR | T2 macro | min / run (T4) |
|---|---|---|---|---|---|
| MLP (Kaggle) | 0.616 ± 0.003 | 0.660 | 0.700 | 0.434 | 2 |
| GNN 1 layer | 0.754 ± 0.006 | 0.791 | 0.818 | 0.544 | 31 |
| **GNN 2 layers** | **0.777 ± 0.007** | **0.813** | **0.835** | **0.578** | 66 |
| GNN 3 layers | 0.773 ± 0.008 | 0.813 | 0.833 | 0.570 | 123 |

- The Kaggle MLP reproduces the laptop B4 (0.616), so the GPU pipeline gives the same numbers.
- GNN 2 layers − MLP = **+16.1 points R@1** (per-seed 95 % CIs all within [+0.12, +0.20]). The gain is far above
  seed noise. H2 is not claimed until `gnn_L2_A2` (no hemilineage) is run.
- 2 − 1 layers = +2.2 ± 1.3 points (one seed's CI touches 0); 3 − 2 layers = −0.4 ± 0.9 points. The third layer
  doubles the time for nothing.
- Still below cocoa-style on filtered R@1 (0.813 vs 0.830); T2 macro is a tie (0.578 vs 0.574).
- Patience 5 would have picked exactly the same best checkpoint in all three 2-layer runs, about 18 min faster
  each. Patience 3 would have lost 0.007 MRR on seed 0.
- **Seed ablation (B3, B4 complete):**

| Seed fraction | B3 R@1 | B3 filtered | B4 R@1 | B4 filtered |
|---|---|---|---|---|
| 5 % | 0.089 ± 0.004 | 0.265 | 0.544 ± 0.016 | 0.598 |
| 10 % | 0.147 ± 0.033 | 0.418 | 0.560 ± 0.007 | 0.611 |
| 25 % | 0.255 ± 0.022 | 0.625 | 0.586 ± 0.007 | 0.634 |
| 50 % | 0.329 ± 0.008 | 0.758 | 0.596 ± 0.005 | 0.641 |
| 100 % | 0.396 | 0.830 | 0.616 ± 0.007 | 0.661 |

  B1 uses no seeds at all and reaches 0.347 / 0.493. Cocoa-style falls below it on filtered R@1 under 25 % seeds.
  The MLP loses only 7 points from 100 % to 5 %.
- The MLP without hemilineage (B4_A2, 3 draws) gets R@1 0.439 ± 0.015 / filtered 0.492 at 5 %,
  0.473 / 0.522 at 10 %, 0.497 / 0.550 at 25 % and 0.513 / 0.561 at 50 % (0.539 / 0.586 at 100 %). That is
  about 8–10 points under the MLP at every fraction, but at 5–10 % seeds still about 3–5× cocoa-style on R@1
  and above it on filtered R@1. So hemilineage, a name shared by both datasets, is not the only reason the
  learned model needs few labels.
- On T2 the curves cross: cocoa-style overtakes the MLP at roughly 35 % seeds (figure F5).

**Decided**
- **Main model = GNN, 2 layers** (L3 ties at 1.9× the time; L1 is the cheaper fallback).
- Patience 5 for all new GNN configs (A2, C4, C5, hn, coral, seed ablation). The week-7 runs used 10;
  `results/patience_sim.csv` shows this does not change the chosen checkpoint.
- The seed-ablation GNN is the plain 2-layer model. It is frozen now, so the central curve does not wait on the
  hard-negative and CORAL experiments; those are reported as separate arms.
- The thesis framing is a crossover, not "the GNN wins everywhere". Expert label projection is best when most
  types are known. The learned model is far better when few are (≤ 10 % of training families).

**Next**
- Kaggle session 3: `gnn_L2_A2`, `gnn_sa_05`, `gnn_sa_10`, `gnn_L2_C4` × 3 seeds, plus `gnn_L2_hn`,
  `gnn_L2_coral` × 1 seed.
- Laptop: `mlp_hn`, `mlp_coral`, `mlp_C5`; `match` on B4 seed 0 and on GNN 2 layers seed 0.
- Later: `gnn_sa_25`, `gnn_sa_50`; the GNN without hemilineage at 5 %, if the A2 gap on the GNN is large.

## Week 9 (laptop part) — MLP variants and control C5 (11 Nov 2026)

**Did**
- `mlp_hn` (hard negatives from the B1 top-50 shortlist), `mlp_coral`, `mlp_C5` × 3 seeds on the CPU.
  Paired differences vs the Kaggle `mlp`: `results/week9_mlp_variants.csv`
  (`python scripts/week8_compare.py week9_mlp_variants.csv mlp_hn:mlp mlp_coral:mlp mlp_C5_last:mlp mlp_C5:mlp`).
- New opt-in training options `train.select: last` (keep the final weights; no validation-based choice) and
  `train.stop_at: N` (end at step N while the LR schedule still spans `max_steps`). Neither is in DEFAULTS, so
  every existing config keeps its hash and finished runs are still recognised.

**Found**
- Hard negatives: +0.9 points R@1 (+1.0 filtered). The mean is above the seed spread, but every single seed's
  CI includes 0. Each step is about 10× slower on the CPU.
- CORAL: +0.2 points, no effect. Domain-probe accuracy 0.58 (hn 0.59), so the domain gap is already small for
  the MLP.
- **C5 protocol bug, found and fixed.** With re-paired labels, validation R@1 falls steadily as training
  goes on (0.47 at step 100 → 0.32 at step 1100). Early stopping uses the TRUE validation labels, so it picked
  step 100 on every seed and reported 0.457. That understates what the control measures.
  `mlp_C5_last` trains to the real MLP's median best step (400) and keeps the final weights: R@1 0.406 ± 0.005,
  i.e. **−20.9 points** vs the real MLP (per-seed CIs within [−0.26, −0.17]).
- C5 does not fall to chance because the input features (neuropil profiles, hemilineage) are already
  comparable across datasets: B1 gets 0.347 without any training. The honest reading is that the features give
  about 0.35–0.40, and the true matches add about 21 points.

**Decided**
- Controls whose labels are corrupted are never early-stopped on validation. `gnn_L2_C5` now uses
  `select: last, stop_at: 800` (gnn_L2's median best step). The old `mlp_C5` runs are kept as the record of
  the bug.
- The plan's expectation for C5 ("collapses to ≈ chance") is replaced by "falls to the feature-only level
  (≈ B1)".
- Unless the Kaggle GNN runs say otherwise, the final model stays the plain 2-layer GNN. Hard negatives and
  CORAL are reported as arms that did not help.

## Week 9 (GPU part) — A2, C4, the GNN at 5 % and 10 % seeds (Kaggle session 3, 10–11 Nov 2026)

**Did**
- Kaggle notebook version 3 on dataset v2 (package from `fb47b59`; "10 data files verified against the manifest").
  7.7 h, 17 runs, none failed: `mlp` (sanity), `gnn_L2_A2`, `gnn_sa_05`, `gnn_sa_10`, `gnn_L2_C4` × 3 seeds,
  plus `gnn_L2_hn` and `gnn_L2_coral` × seed 0. All with patience 5. Weekly GPU use is about 19 of 30 h.
- Paired family bootstrap, same seed vs same seed (draw r = seed r for the seed-ablation runs):
  `results/week9_compare.csv`.

**Found** (validation, mean ± sd over 3 seeds)

| Run | R@1 | R@1 filtered | MRR | T2 macro |
|---|---|---|---|---|
| mlp (week-9 code) | 0.616 ± 0.006 | 0.661 | 0.699 | 0.434 |
| gnn_L2 (session 2) | 0.777 ± 0.007 | 0.813 | 0.835 | 0.578 |
| gnn_L2_A2, no hemilineage | 0.721 ± 0.007 | 0.761 | 0.784 | 0.534 |
| gnn_L2_C4, edge-permuted graphs | 0.620 ± 0.002 | 0.663 | 0.703 | 0.422 |
| GNN, 10 % seeds | 0.715 ± 0.005 | 0.757 | 0.782 | 0.523 |
| GNN, 5 % seeds | 0.693 ± 0.011 | 0.740 | 0.763 | 0.520 |
| gnn_L2_hn (seed 0) | 0.772 | 0.806 | 0.834 | 0.588 |
| gnn_L2_coral (seed 0) | 0.771 | 0.809 | 0.828 | 0.572 |

- **H2 holds without the annotation.** Without hemilineage, the GNN beats the MLP *with* hemilineage by
  +10.5 points R@1 (per-seed CIs [+0.07, +0.15]) and the MLP without it by +18.2 ([+0.14, +0.22]).
  Hemilineage is worth 5.6 points to the GNN, against 7.7 to the MLP.
- **H5 holds (C4).** With degree-preserving edge permutation, the GNN falls exactly to the MLP:
  +0.4 points vs B4 (CI [−0.02, +0.03]) and −15.6 vs the real GNN. The gain comes from who connects to whom,
  not from degree or from having a graph at all.
- **H3, the headline.** At 5 % seeds the GNN beats cocoa-style by +60.4 points R@1 and +47.5 filtered
  (CIs [+0.39, +0.55] filtered). At 10 %: +56.8 / +33.9. It also beats the MLP at the same fraction by
  +15 points.
  - The GNN at 5 % equals cocoa-style at **50 %** on filtered R@1: −1.8 points, CI [−0.09, +0.04], not
    significant. That is about **10× label efficiency**.
  - The GNN keeps 91 % of its full-label filtered accuracy at 5 % seeds; cocoa-style keeps 32 %.
  - Going from 100 % to 5 % costs the GNN 8.3 points R@1.
- **Hard negatives / CORAL on the GNN:** 0.0 and −0.1 points vs gnn_L2 seed 0 (CIs ±0.02). Nothing.
- **Code versions:** session 2 used the week-7 package; session 3 used `fb47b59`. The MLP sanity score is the
  same within noise (0.616 vs 0.616 mean), so the sessions are comparable.
- **Grouping bug found and fixed.** The notebook's own summary grouped by `method` and printed one "GNN"
  row mixing 5 % and 10 %. Folder names are also no good for "latest run": Kaggle dates them in UTC and they
  end in a hash, so session-2 and session-3 `mlp` runs sorted by hash. `aggregate`, `seed_ablation.collect`
  and `week8_compare.run_of` now pick the most recently finished run (`metrics.json` time).
  `FLYMATCH_BEFORE="2026-11-10 18:00"` reproduces the week-8 table exactly.

**Decided**
- **Gate G3: final model = plain `gnn_L2`** (2 layers, directed, patience 5). Hard negatives and CORAL are
  reported as ablations that did not help, on the MLP (3 seeds) and the GNN (1 seed).
- Next Kaggle session: `gnn_sa_25`, `gnn_sa_50` × 3 seeds (about 4–5 h) to finish the curve. The notebook's
  results cell now groups by experiment name.

## Week 10 (start) — final-model matching, GNN + B3 ensemble, control C3 (11 Nov 2026)

**Did**
- `run match` on the final model (`gnn_L2` seed 0), run by the user from their own terminal: abstention head +
  one-to-one assignment. A background shell is killed under memory pressure.
- `run seed-ablation --methods GNN+B3`: GNN + cocoa-style, z-scored per query, **equal weights**. No weight
  tuning, because validation is the evaluation set. Both members use the same seed set (GNN draw r = seed r,
  B3 projected on the same types). Results in `results/seed_ablation.csv` and F5; paired comparisons in
  `results/week9_ensemble_compare.csv`.
- Control C3, evaluation form (`scripts/c3_mirror.py` → `results/c3_mirror.csv`):
  - Inside one dataset, embed the dataset normally and with every node's features left↔right mirrored
    (`embed_run(..., mirror=True)`; the graph is unchanged).
  - Retrieve each left neuron's right twin, and vice versa: 414 MaleCNS and 299 BANC validation 1:1 pairs.
  - Score this and the real cross-dataset task with one evaluator. It reproduces the harness exactly
    (B1 0.347, MLP 0.616, GNN 0.777).

**Found**
- **Matching with the final GNN (validation):**
  - Abstention: AUROC 0.709 (MLP 0.640), ECE 0.024.
  - Known female-specific neurons flagged as "no counterpart": 71 % vs a 35 % base rate (2.0×, Fisher
    p ≈ 1e-60).
  - Assignment at τ_none 0.70: precision **0.867**, recall 0.692, **F1 0.770**.
  - Same settings for the MLP (B4 seed 0, rerun by the user): AUROC 0.640, P 0.723, R 0.567, F1 0.635.
    The GNN gains +0.069 AUROC, +14 points precision, +13 recall, +0.13 F1, and 2.0× vs 1.6× enrichment
    of female-specific flags. `results/match_compare.csv` (`scripts/match_table.py`).
  - 72,503 of 159,632 MaleCNS neurons get a BANC partner (`matches.parquet`).
- **Ensemble:**
  - 100 % seeds: filtered R@1 **0.851 ± 0.000**, the best of any method. vs cocoa-style: +2.1 points,
    per-seed CIs [+0.012, +0.032]. vs GNN alone: +3.8 points filtered, but −18.6 unfiltered, because
    cocoa-style is side-blind and adds left/right ties. T2 macro 0.605, also the best.
  - 5 % / 10 % seeds: 0.561 / 0.602 filtered, 15–18 points *below* the GNN alone. A weak member hurts at
    equal weights.
- **C3:**

| Encoder | Cross | Within MaleCNS | Within BANC |
|---|---|---|---|
| B1 | 0.347 | 0.699 | 0.570 |
| MLP | 0.616 ± 0.007 | 0.837 | 0.744 |
| GNN | 0.777 ± 0.007 | 0.935 | 0.846 |

  - Learning shrinks the cost of the domain gap: 22–35 points for B1 → 13–22 for the MLP → 7–16 for the GNN.
  - Within BANC is always well below within MaleCNS, so BANC's own completeness limits matching.
    The GNN's cross-dataset 0.777 is only 7 points below its within-BANC ceiling.
  - These encoders were never trained for mirror symmetry, so "within" is a lower bound on the zero-gap
    ceiling.

**Decided**
- **Two operating points for the thesis:**
  - Few labels, or open (unfiltered) search → the GNN alone.
  - Most types known, search within super class + side → GNN + cocoa-style.
  The ensemble weights stay equal; not tuned.
- C3 is reported in its evaluation form. A bilaterally *trained* C3 needs GPU time and a new training mode,
  and is cut first if time is short (plan Section 12 cut order).

## Week 10 — hemibrain ingest for C1 / H4 (11 Nov 2026)

**Did**
- `run hemibrain-download`: neuPrint `hemibrain:v1.2.1`, 97,900 Traced neurons; Traced → Traced pairs ≥ 5
  synapses (the same threshold as our graphs), split by primary ROI. That is 955,818 pairs and 13.0 M synapses;
  17 min. Stored in `data/external/hemibrain_v1.2.1/` (git-ignored).
- `maps/hemibrain_neuropil_map.csv`: all 63 primary ROIs → our canonical neuropils (55 of 87 covered).
- `run hemibrain-prepare`: harmonised hemibrain tables, neuropil completeness, type links, 1:1 pairs.
- `scripts/hemibrain_check.py`: link validation and the C1 B1 baseline.

**Found**
- **Completeness: two effects.** Hemibrain counts only traced partners, so its "completeness" relative to
  MaleCNS mixes two things:
  - tracing completeness: 0.3–1.8 of the central-complex level on the right;
  - truncation by the volume boundary: optic lobe, GNG, SAD and most left neuropils, ≤ 0.19.
  The first rule (keep ≥ 0.8) kept only 14 neuropils and dropped clearly-inside ones such as AVLP_R and LH_R.
  The current rule separates the two effects:
  - right / midline neuropils: ≥ 0.25 (the gap);
  - left neuropils: ≥ 0.7 of their own right mirror.
  Result: **32 neuropils** (all right central brain, the central complex, ATL_L).
- **Type links.** Only 350 of 5,620 hemibrain types link through the MaleCNS `hemibrain_type` label. A further
  **2,816 link by identical name**: MaleCNS kept most hemibrain names, and the label mostly marks renamed types.
  139 are ambiguous and 15 conflict; these stay unlinked.
- **Name links validated.** B1 retrieval on the 32 neuropils is as good or better for name links as for label
  links: hemibrain↔MaleCNS R@1 0.495 vs 0.401.
- **1:1 pairs.** 2,073 hemibrain↔MaleCNS (val 183) and 1,773 hemibrain↔BANC (val 160).
- **C1 B1 baseline** (validation families, same-side gallery):

| Pair | Differs in | B1 R@1 |
|---|---|---|
| hemibrain ↔ MaleCNS | sex only, same FIB-SEM imaging | 0.523 |
| hemibrain ↔ BANC | imaging only, both female | 0.247 |

  For raw profiles, reconstruction/imaging costs about twice as much as sex. This is the confound H4 tests.

**Next**
- C1 proper (GPU): train BANC↔MaleCNS on hemibrain-restricted features (the 32 neuropils, no hemilineage /
  NT / super class, which hemibrain lacks), then evaluate zero-shot on hemibrain↔BANC and hemibrain↔MaleCNS
  against this B1.

## Week 10 — Kaggle session 4 (GNN at 25 / 50 % seeds) and C1 set-up (11 Nov 2026)

**Did**
- **Kaggle session 4:** `gnn_sa_25` and `gnn_sa_50` × 3 seeds, plus the mlp sanity runs.
  - The 6 GNN runs were imported. All carry the frozen split hash.
  - The mlp reruns (0.614 / 0.621 / 0.620) were *not* imported. They share the config hash of session 3's runs,
    and seed 2 even has the identical folder name. They confirm the package behaves the same.
- **Ensemble at 25 / 50 %:** the GNN + B3 curve is complete.
- **F5 now has three panels:** unfiltered R@1, filtered R@1 and T2. Cocoa-style's strength is the filtered
  search, so leaving that panel out would flatter the GNN.
- **C1 variant `data/c1`** (`run c1-build`, `configs/data_c1.yaml`, `src/flymatch/c1.py`):
  - the same neurons, ground truth and split as the main data;
  - graph and features restricted to the 32 neuropils inside the hemibrain volume: np_in/np_out, size_rank on
    the restricted graph, side;
  - no region summaries, NT, super class or hemilineage;
  - neurons without synapses in those neuropils are not included: 31k MaleCNS, 24k BANC; 3,478 1:1 pairs,
    of which 325 are in validation.
- **Experiment configs choose their data variant** with a `data_config:` key. `run_cfg` then records `data_dir`,
  so existing run hashes are unchanged.
- **`scripts/c1_eval.py`:** zero-shot hemibrain evaluation; `mlp_c1` × 3 seeds on CPU.
- **Package for session 5** (dataset v3, 115 MB): ships `data/c1`. The notebook list is `mlp`, `gnn_c1`,
  `gnn_L2_C5`.

**Found**
- **GNN seed-ablation curve** (R@1 / filtered): 0.693 / 0.740 at 5 %, 0.715 / 0.757 at 10 %,
  0.736 / 0.775 at 25 %, 0.753 / 0.791 at 50 %, 0.777 / 0.813 at 100 %.
- **GNN − cocoa-style at the same fraction, filtered** (`results/week9_compare_sa.csv`):
  - +47.5 / +33.9 / +14.9 points at 5 / 10 / 25 %, all CIs > 0;
  - +3.4 at 50 % (CI [−0.02, +0.08]);
  - −1.7 at 100 %.
  The GNN beats the MLP by 14–16 points at every fraction.
- **Ensemble, filtered R@1:**
  - 0.561 / 0.602 / 0.713 / 0.800 / 0.851 at 5 / 10 / 25 / 50 / 100 %;
  - it only pays at 100 %: +2.1 vs B3, +3.8 vs GNN (CI [−0.001, +0.069]);
  - it ties the GNN at 50 % and is below it at ≤ 25 %.
- **C1 MLP, zero-shot on hemibrain** (validation pairs; B1 = restricted profiles):

| Pair | Differs in | B1 R@1 / MRR | mlp_c1 R@1 / MRR |
|---|---|---|---|
| hemibrain ↔ MaleCNS | sex | 0.529 / 0.609 | 0.557 ± 0.019 / 0.640 |
| hemibrain ↔ BANC | imaging | 0.250 / 0.317 | 0.298 ± 0.007 / 0.406 |

  The gain over B1 holds on the pair the model never saw and that differs only in imaging. It is larger there in
  MRR (+0.089 vs +0.031). This modest evidence for H4 comes from the MLP; the GNN is session 5.
- On the restricted variant, BANC↔MaleCNS itself drops to R@1 ≈ 0.29 for the MLP (0.616 on full features).
  The 32 neuropils carry much less information.

**Decided**
- The thesis headline is the filtered-panel crossover:
  - the GNN wins clearly up to 25 % seeds;
  - ties at 50 %;
  - cocoa-style (and the ensemble) win only with all labels.
  Unfiltered gains are reported, but not headlined (B3 is side-blind).

---

<!-- Template for new entries
## Week N — title (date)

**Did**
-

**Found**
-

**Decided**
-

**Next**
-
-->
