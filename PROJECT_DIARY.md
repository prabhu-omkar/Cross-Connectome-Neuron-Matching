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
