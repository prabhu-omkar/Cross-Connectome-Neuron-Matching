<div align="center">

# 🪰 Cross-Connectome Neuron Matching

### Finding the same neuron in a female and a male fruit-fly nervous system — with graph neural networks

**BANC** (female brain + nerve cord, 158,262 neurons) &nbsp;⟷&nbsp; **MaleCNS** (male brain + nerve cord, 166,700 neurons)

[![License: Apache 2.0](https://img.shields.io/badge/code-Apache%202.0-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE.md)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-sparse%20GNN-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-54%20passing-brightgreen.svg)](tests/)
[![Progress](https://img.shields.io/badge/progress-week%2010%20of%2016-orange.svg)](PROJECT_DIARY.md)
[![Split](https://img.shields.io/badge/benchmark%20split-frozen%20%C2%B7%20sha256%20260be072-purple.svg)](data/splits/splits_v1.json)

[**Results**](#-results-so-far) · [**Quick start**](#-quick-start) · [**How it works**](#-how-it-works) · [**Data**](#-the-data) · [**Train on Kaggle**](INSTRUCTIONS.md) ([notebook](https://www.kaggle.com/code/omkarprabhuuuu/flymatch-train)) · [**Diary**](PROJECT_DIARY.md) · [**Roadmap**](#%EF%B8%8F-roadmap)

</div>

---

## 🧠 The problem in one paragraph

Two complete wiring diagrams of a fruit-fly central nervous system now exist: one **female** (BANC, serial-section electron microscopy) and one **male** (MaleCNS, FIB-SEM). Each is a directed graph of ~160,000 neurons and millions of synaptic connections. Biologists want to know, for any neuron in one fly, **which neuron is "the same one" in the other**, or whether it has **no counterpart at all** (a sex-specific neuron). Today this relies on hand-curated cell-type names and morphology. This project learns it instead. A **shared-weight graph neural network** embeds neurons from both connectomes into one vector space. Matching then becomes nearest-neighbour search plus an assignment step, with a calibrated *"no counterpart"* answer.

<div align="center">

| What we build | What we will claim |
|---|---|
| A frozen, public **BANC ↔ MaleCNS matching benchmark** with baselines | The first benchmark of its kind for these two datasets |
| A **directed, sparse GNN** trained contrastively across sexes | Accuracy as a function of how many labelled pairs ("seeds") it gets |
| A calibrated **"no counterpart"** output | Abstention evaluated as a first-class task |
| **Controls** that separate biology from imaging artefacts | Evidence the model learned homology, not the microscope |

</div>

---

## 🏆 Results so far

All numbers are on the validation split.
- **T1, exact-partner retrieval:** Recall@1, averaged over both directions. Each of the 669 one-to-one pairs is ranked against **all ~116–160k included neurons** of the other dataset.
- **T2, cell-type accuracy:** whether the top match has the right type, macro-averaged over super classes.
- *Filtered* means the match must have the same super class, the same side and a compatible neurotransmitter.
- 95% confidence intervals are bootstrapped over type families.

The test split has **not** been touched. The validation split was also used for early stopping and model choice, so these numbers are slightly optimistic.

| | Method | Uses labels? | T1 R@1 | T1 R@1 (filtered) | T1 MRR | T2 acc@1 |
|---|---|:-:|:-:|:-:|:-:|:-:|
| B0 | Random | – | 0.0% | 0.1% | 0.000 | 0.1% |
| B1 | √ neuropil in/out profiles (cosine) | – | 34.7% | 49.3% | 0.430 | 22.8% |
| B2 | NBLAST morphology (published matrix) | – | 28.9% | 29.6% | 0.298 | 35.8% |
| B2+ | B1 + NBLAST (z-scored sum) | – | 39.6% | 53.1% | 0.477 | 40.1% |
| B3 | cocoa-style label-projected connectivity | ✔ | 39.6% | 83.0% | 0.597 | 57.4% |
| B4 | MLP, contrastive (= GNN with 0 layers), 3 seeds | ✔ | 61.6 ± 0.7% | 66.1% | 0.701 | 44.0% |
| **GNN** | **Directed message passing, 2 layers, 3 seeds** (final model) | ✔ | **77.7 ± 0.7%** | 81.3% | **0.835** | 57.8% |
| GNN + B3 | Equal-weight z-score ensemble, 3 seeds | ✔ | 59.1% | **85.1%** | 0.716 | **60.5%** |

The ensemble's unfiltered R@1 is low because B3 cannot tell left from right.

<p align="center"><img src="results/figures/F5_seed_ablation.png" width="100%" alt="Seed ablation: accuracy versus the fraction of known matches"></p>

**Headlines**

- 🧬 **Message passing helps, and the gain is real wiring.**
  - The 2-layer GNN beats the MLP by **+16.1 points** R@1 (paired family-bootstrap CIs [+0.12, +0.20] on every seed).
  - Without the shared hemilineage annotation it still wins by **+10.5**.
  - With degree-preserving edge shuffling (control C4) the gain vanishes: +0.4, CI [−0.02, +0.03].
- 📉 **Label efficiency, the central result (F5).** With **5%** of the known matches as training seeds:
  - the GNN keeps **91%** of its full-label filtered accuracy (74.0%); cocoa-style keeps 32% (26.5%);
  - the GNN beats cocoa-style at 5%, 10% and 25% seeds (+47.5, +33.9 and +14.9 points filtered, CIs > 0);
  - at 50% they tie (+3.4, n.s.);
  - only with **all** labels is cocoa-style (83.0%), and the GNN + cocoa-style ensemble (85.1%), better under filters.
- 🎯 **Matching with "no counterpart".** On the final GNN (seed 0), one-to-one assignment has **precision 0.87, recall 0.69, F1 0.77** (MLP: 0.72 / 0.57 / 0.64).
  - The abstention score is calibrated (ECE 0.024).
  - It flags known female-specific BANC neurons at **2.0×** the base rate (Fisher p ≈ 10⁻⁶⁰).
- 🪞 **What the domain gap costs (C3).** The same GNN finds a neuron's left↔right twin *within* one dataset at 93.5% (MaleCNS) and 84.6% (BANC), against 77.7% across datasets.
  - Learning shrinks the gap from 22–35 points (B1) to 7–16 points.
  - BANC's own reconstruction is the main limit.
- 🧪 **Sanity checks behave.**
  - Re-paired training labels (C5) cost −20.9 points; they fall to the feature-only level, not to chance, because the features are directly comparable.
  - Hard negatives and CORAL bring no gain, so they are not adopted.
- 🔗 **Connectivity carries real identity signal.** A zero-learning profile of *where* a neuron sends and receives synapses finds the exact partner among ~116,000 candidates 35% of the time. The shuffled control gets ≈ 0.1%.
- ⚖️ **Honest ranking matters.** 8% of queries tie with hundreds of neurons, because their profiles sit in a single neuropil. Counting ties as wins inflates B1 (MaleCNS → BANC) from 36.0% to 40.5%. Every rank here uses average-tie ranks.
- ↔️ **Label projection is side-blind.** B3 can't tell a partner's left copy from its right copy (40% at R@1, 73% at R@2). With a side filter it jumps to 83%.
- 🔬 **Published NBLAST is reliable only for some classes.** It works for ascending and descending neurons (≈ 60% top-1 type accuracy) but is weak for central-brain and VNC neurons. Its top-100 lists contain the true partner for only 31% of pairs.

<details>
<summary><b>Seed ablation in numbers</b> (T1 R@1 unfiltered / filtered, mean of 3 nested draws; <code>results/seed_ablation.csv</code>)</summary>

| Seeds | Cocoa-style (B3) | MLP (B4) | MLP, no hemilineage | **GNN** | GNN + B3 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 5% | 8.9 / 26.5 | 54.4 / 59.8 | 43.9 / 49.2 | **69.3 / 74.0** | 43.5 / 56.1 |
| 10% | 14.7 / 41.8 | 56.0 / 61.1 | 47.3 / 52.2 | **71.5 / 75.7** | 43.8 / 60.2 |
| 25% | 25.5 / 62.5 | 58.6 / 63.4 | 49.7 / 55.0 | **73.6 / 77.5** | 48.8 / 71.3 |
| 50% | 32.9 / 75.8 | 59.6 / 64.1 | 51.3 / 56.1 | **75.3 / 79.1** | 54.8 / 80.0 |
| 100% | 39.6 / 83.0 | 61.6 / 66.1 | 53.9 / 58.6 | **77.7** / 81.3 | 59.1 / **85.1** |

B1 uses no seeds at all: 34.7 / 49.3.

</details>

<details>
<summary><b>Hemibrain transfer (control C1 / hypothesis H4): in progress</b></summary>

The [hemibrain](https://www.janelia.org/project-team/flyem/hemibrain) is female like BANC but imaged like MaleCNS (FIB-SEM). The test:
- train on BANC ↔ MaleCNS using only what the hemibrain also has: the 32 neuropils inside its volume and the restricted graph, with no neurotransmitter, super class or hemilineage;
- then match hemibrain neurons zero-shot.

Results on validation pairs (`results/c1_eval.csv`):

| Pair | Differs in | B1 R@1 | MLP R@1 | GNN R@1 |
|---|---|:-:|:-:|:-:|
| hemibrain ↔ MaleCNS | sex | 52.9% | 55.7 ± 1.9% | *Kaggle session 5* |
| hemibrain ↔ BANC | imaging | 25.0% | 29.8 ± 0.7% | *Kaggle session 5* |

For raw profiles, imaging costs about twice as much as sex. The learned MLP keeps its gain over B1 on the imaging-only pair, which it never saw: MRR 0.406 vs 0.317.

</details>

Full analysis: [`docs/notes/phase1_weeks1-4.md`](docs/notes/phase1_weeks1-4.md) · [`docs/notes/phase2_weeks5-7.md`](docs/notes/phase2_weeks5-7.md) · [`docs/notes/phase3_weeks8-9.md`](docs/notes/phase3_weeks8-9.md) · [diary](PROJECT_DIARY.md)

---

## ⚡ Quick start

```powershell
git clone https://github.com/prabhu-omkar/Cross-Connectome-Neuron-Matching.git
cd Cross-Connectome-Neuron-Matching

py -3.11 -m venv .venv                       # (Linux/macOS: python3.11 -m venv .venv)
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e .

.venv\Scripts\python.exe -m flymatch.run phase1      # raw CSVs → harmonised tables, features, ground truth, reports (~7 min)
.venv\Scripts\python.exe -m pytest -q                # 54 tests: units, data contracts, leakage
.venv\Scripts\python.exe -m flymatch.run baselines   # B0–B4 on validation → results/baselines.csv (~30 min, CPU)
```

The raw connectome exports are too large for GitHub: download them from [FlyWire Codex](https://codex.flywire.ai) and the [BANC Dataverse](https://doi.org/10.7910/DVN/7WTH1N) into `data/raw/` and `data/external/` (see [The data](#-the-data)). To **train without any local setup**, use the public Kaggle notebook — **[INSTRUCTIONS.md](INSTRUCTIONS.md)**.

<details>
<summary><b>All commands</b></summary>

| Command | What it does |
|---|---|
| `python -m flymatch.run data` | ingest → harmonise → features → ground truth (~1.5 min) |
| `python -m flymatch.run split [--freeze]` | stratified family split; `--freeze` writes `splits_v1.json` **once** (refuses to overwrite) |
| `python -m flymatch.run report` | harmonisation report, neuropil-merge validation, GT composition (training pairs only) |
| `python -m flymatch.run phase1` | all of the above + B1 |
| `python -m flymatch.run baselines [--methods B0,B1,B2,B2+,B3,B4]` | baselines through the evaluation harness → `runs/`, `results/baselines.csv`, F4 |
| `python -m flymatch.run train --exp configs/experiments/gnn_L2.yaml --seeds 0,1,2 [--device cuda]` | train + evaluate; checkpoints every 200 steps, resumes automatically |
| `python -m flymatch.run package` | build `dist/flymatch_kaggle.zip` for Kaggle (`--parts-mb 20` also splits it for a slow upload) |
| `python -m flymatch.run aggregate` | all `runs/` → `results/baselines.csv` (latest finished run per method/seed; rejects runs with a different split hash) |
| `python -m flymatch.run seed-ablation [--methods B3,B4,B4_A2,GNN+B3]` | seed-ablation grid (5–100% × 3 nested draws) → `results/seed_ablation.csv`, figure F5; picks up GNN runs from Kaggle |
| `python -m flymatch.run match --run runs/<run>` | abstention head + one-to-one assignment with a "no counterpart" option → `matches.parquet` (needs ~8 GB RAM) |
| `python -m flymatch.run hemibrain-download` · `hemibrain-prepare` · `c1-build` | hemibrain v1.2.1 from neuPrint (token needed) → completeness, type links, 1:1 pairs → restricted data variant `data/c1` |
| `python scripts/week8_compare.py out.csv a:b …` | paired family-bootstrap differences between runs |
| `python scripts/c3_mirror.py` · `scripts/c1_eval.py` | control C3 (within-dataset left↔right) · zero-shot hemibrain evaluation |
| `python -m flymatch.smoke --layers 2` | full-graph GNN forward/backward memory + timing check |

</details>

---

## 🔬 How it works

```mermaid
flowchart LR
    subgraph RAW["Raw Codex exports"]
        M["MaleCNS v1.0<br/>166,700 neurons<br/>9.0 M edge rows"]
        B["BANC v888<br/>158,262 neurons<br/>4.0 M edge rows"]
    end
    RAW --> I["1 · Ingest<br/>fixed dtypes<br/>exact 18-digit IDs"]
    I --> H["2 · Harmonise<br/>vocab maps · ≥5 syn/pair<br/>fractional weights"]
    H --> F["3 · Features<br/>462 dims / neuron"]
    H --> G["4 · Ground truth<br/>6,765 1:1 pairs<br/>9,570 shared types"]
    G --> S["5 · Frozen split<br/>by type family<br/>70 / 10 / 20"]
    F --> E["Evaluation harness<br/>T1 · T2 · filters · CIs"]
    S --> E
    E --> BL["Baselines B0–B4"]
    E --> GNN["Shared-weight<br/>directed GNN"]
    GNN --> A["Assignment +<br/>'no counterpart'"]
```

### 1 · Harmonisation: removing the microscope before learning

The two datasets differ in ways that are pure artefact, and a model will happily learn them. So before anything else:

| Artefact | MaleCNS | BANC | What we do |
|---|---|---|---|
| Synapse threshold per neuron pair | ≥ 5 | ≥ 3 | Re-threshold **both** to ≥ 5 (removes 1.5 M weak BANC pairs) |
| Synapses detected | 89.9 M | 23.6 M (**3.8× fewer**) | **Fractional** edge weights; **rank-normalised** size features |
| Neuropil names | 134 (nerves, `*_UNASGD`) | 111 (`VNC_*` prefixes, split `IB`) | Hand-written map → **87 canonical neuropils**, every merge validated (Pearson r ≥ 0.5 on training pairs) |
| Super-class names | 27 (`ol_intrinsic`, 3 sensory classes…) | 16 (`optic_lobe_intrinsic`, one `sensory`) | Hand-written map → 12 harmonised classes |
| Root-ID format | small integers | **18-digit** IDs (break as float!) | Read as text → int64 with a round-trip check |

> 💡 **Discovery:** BANC stores **> 99%** of its leg-neuropil synapses under `VNC_T1_ProNm` / `VNC_T2_MesoNm` / `VNC_T3_MetaNm`, not `LegNp_*`. Mapping these, plus the old name `AMNp` → ovoid `Ov`, raised the zero-learning baseline on training pairs by 7 points overall and by 31 points for ascending neurons.

<p align="center"><img src="results/figures/F2_harmonisation.png" width="92%" alt="Harmonisation report"></p>

Equal thresholds actually *widen* the degree gap: the median MaleCNS/BANC degree ratio goes from 1.63 to 3.03. That is why fractional weights and within-dataset rank features are essential. A quantile-matched setting (MaleCNS ≥ 15, BANC ≥ 5, ratio 1.06) is kept as ablation A9.

### 2 · Node features (462 per neuron, identical column order in both datasets)

| Block | Dims | Definition |
|---|:-:|---|
| Neuropil input / output profile | 87 + 87 | √ of the fraction of the neuron's input (output) synapses in each canonical neuropil |
| Region summary | 6 + 6 | Fraction of synapses in optic lobe / central brain / GNG / VNC / nerve / unassigned |
| Size | 6 | In/out degree, in/out synapses, # neuropils — **quantiles within dataset** |
| Neurotransmitter | 9 + 1 | Harmonised predicted NT one-hot + confidence |
| Side · super class | 3 · 12 | One-hots |
| Hemilineage | 242 + 1 | Developmental lineage shared by both datasets (an annotation → ablated in A2) |
| *Label-projected connectivity* | *2 × seeds* | *Fraction of input/output to each seed type — built per seed fraction* |

### 3 · Ground truth, and why it is split by *family*

A MaleCNS neuron's harmonised type is its **Primary Cell Type** if BANC uses that name. Otherwise it is its **FlyWire type**, and failing that its **MANC type**. The fallbacks were validated: **99.6%** super-class agreement, which is better than primary names (99.5%). The result is **9,570 shared types** and **6,765 unambiguous one-to-one pairs** (same type, same side, exactly one neuron in each fly).

Splitting by neuron would leak (`LC10a` in train, `LC10b` in test). So types are grouped into **3,610 families** by union-find over shared name stems, co-listed alternative names and MaleCNS supertypes. The families are split **70 / 10 / 20**, stratified by super class × type size. The split is frozen and hashed: every run records `sha256 260be072…`, and the aggregator rejects anything else.

<p align="center"><img src="results/figures/F3_ground_truth.png" width="92%" alt="Ground-truth composition and split balance"></p>

### 4 · The model

```text
x (462) ─► Linear-GELU-LN (256) ─► [ DirLayer ] × L ─► concat all layers ─► MLP ─► ℓ2-normalised z (128)

DirLayer:  h ← h + Dropout(GELU(LN( W_self·h  +  W_in·(A_in @ h)  +  W_out·(A_out @ h) )))
           A_in[j,i]  = fraction of j's input from i      (who talks to me)
           A_out[i,j] = fraction of i's output going to j  (whom I talk to)
```

- **One encoder with shared weights** embeds both connectomes into one space.
- **Sparse matrix products** (`torch.sparse.mm`) replace per-edge messages. A full 2-layer pass over both graphs (7.8 M edges) fits in **5.4 GiB**.
- **Multi-positive InfoNCE**, symmetric across datasets:
  - positives are neurons of the same type **and** side;
  - the same type on the other side is masked (neither positive nor negative);
  - negatives come only from **training** families.
- **Type-balanced sampling** keeps the 89,000 optic-lobe columnar neurons from drowning everything else.
- **Depth = 2 layers.** It beats 1 layer by +2.2 points; a third layer adds nothing at 1.9× the time.
- **Early stopping** on validation MRR, patience 5. Replaying the curves gives the same checkpoint as patience 10. Hard-negative mining (B1 top-50 shortlist) and class-conditional CORAL were tried and did not help.
- **Matching:**
  - an **abstention head**: logistic regression + isotonic calibration on frozen embeddings. Its features are the top-1 score, the margin to the runner-up, mutual nearest neighbour, and similar. It is trained by simulating partner removal on one half of the validation families and checked on the other half;
  - **one-to-one assignment** with a "no counterpart" option at threshold τ, chosen on validation: exact Hungarian on small components, sparse LAPJV on large ones.

---

## 📦 The data

| Dataset | Source | Licence | In this repo |
|---|---|---|---|
| **MaleCNS v1.0**: neurons + neuropil-split connections | [FlyWire Codex](https://codex.flywire.ai) export | CC BY 4.0 | download → `data/raw/mcns/` (47 MB) |
| **BANC v888**: neurons + neuropil-split connections | [FlyWire Codex](https://codex.flywire.ai) export | CC BY 4.0 | download → `data/raw/banc/` (32 MB) |
| BANC metadata (proofread flag, sexual dimorphism) | [Harvard Dataverse](https://doi.org/10.7910/DVN/7WTH1N) v3 | CC BY 4.0 | download → `data/external/` (58 MB) |
| BANC ↔ MaleCNS expert-reviewed matches · somas | Harvard Dataverse v3 | CC BY 4.0 | download → `data/external/` |
| BANC ↔ MaleCNS v0.9 NBLAST matrix (912 MB) | Harvard Dataverse v3 | CC BY 4.0 | optional, for B2 only: download → `data/external/` |
| Hemibrain v1.2.1: traced neurons + ROI-split connections | [neuPrint](https://neuprint.janelia.org) (free token) | CC BY 4.0 | optional, for C1: `run hemibrain-download` → `data/external/hemibrain_v1.2.1/` |
| Processed tables + features (+ C1 variant), Kaggle package | generated by this code | CC BY 4.0 | [Kaggle dataset](https://www.kaggle.com/datasets/omkarprabhuuuu/flymatch) |
| Frozen benchmark split | this project | CC BY 4.0 | `data/splits/splits_v1.json` |

Exact file IDs, sizes and download dates: [`data/external/SOURCES.md`](data/external/SOURCES.md). Attribution and citations: [`DATA_LICENSE.md`](DATA_LICENSE.md).

<details>
<summary><b>Side-by-side profile of the two connectomes</b></summary>

| Property | MaleCNS v1.0 (male) | BANC v888 (female) |
|---|---|---|
| Imaging | FIB-SEM, 8 × 8 × 8 nm | serial-section TEM, 4 × 4 × 45 nm |
| Neurons in export | 166,700 | 158,262 |
| Connection rows (neuropil-split) | 9,039,608 | 3,990,039 |
| Distinct neuron pairs | 6,242,085 (excluding 56 self-loop rows) | 3,037,361 → 1,528,585 after ≥ 5 |
| Primary cell type filled | 98.7% | 74.9% |
| Super class filled | 100% | 92.7% |
| Included after quality rules | 159,632 | 116,553 |
| Clean subset | 54,062 (`Reviewed`) | 122,177 (proofread, no tracing flags) |

</details>

---

## 🗂️ Repository layout

```text
Cross-Connectome-Neuron-Matching/
├── README.md · INSTRUCTIONS.md (Kaggle) · PROJECT_DIARY.md · LICENSE · NOTICE · DATA_LICENSE.md
├── configs/
│   ├── data.yaml                    thresholds, inclusion rules, GT sources, split seed
│   ├── paths.example.yaml           where the raw data lives (defaults: data/raw, data/external)
│   ├── model/ · train/              GNN / MLP and training defaults
│   ├── data_c1.yaml                 C1 / H4 variant: hemibrain-restricted graph + features (data/c1)
│   └── experiments/                 mlp, gnn_L1–L3, ablations (A2), controls (C4, C5), hn / coral,
│                                    seed ablation (gnn_sa_05…50), C1 (gnn_c1, mlp_c1), smoke tests
├── maps/                            hand-maintained vocabulary maps (ingest fails on any unmapped value)
│   ├── neuropil_map.csv             245 rows: raw → canonical neuropil, region, include, confidence, note
│   ├── hemibrain_neuropil_map.csv   63 hemibrain primary ROIs → canonical neuropils
│   └── superclass_map.csv · nt_map.csv · hemilineage_map.csv
├── data/
│   ├── raw/{mcns,banc}/             the two Codex exports (read-only)
│   ├── external/                    BANC Dataverse files + SOURCES.md
│   ├── splits/splits_v1.json        FROZEN benchmark split + .sha256
│   └── interim/ · processed/        regenerated by `run data` (git-ignored)
├── src/flymatch/
│   ├── io.py · ingest.py            fixed-dtype readers, raw → canonical parquet
│   ├── harmonise.py                 thresholds, fractional weights, inclusion rules
│   ├── features.py                  462-dim node features + L↔R mirror permutation
│   ├── groundtruth.py               harmonised types, fallback validation, 1:1 pairs, families
│   ├── splits.py · seeds.py         frozen family split; nested seed sets for the ablation
│   ├── evaluate.py                  THE harness: T1/T2, filters, clean subset, breakdowns, bootstrap CIs
│   ├── graph.py                     sparse A_in / A_out, undirected variant, degree-preserving permutation
│   ├── models/                      DirLayer, Encoder, InfoNCE / CORAL / gradient reversal
│   ├── train.py · experiment.py     contrastive training with checkpoint/resume; config-driven runs
│   ├── hardneg.py · probe.py        hard negatives from the B1 shortlist; type-balanced domain probe
│   ├── abstention.py · assign.py · inference.py   "no counterpart" head; one-to-one assignment; embeddings from model.pt
│   ├── seed_ablation.py             the seed-ablation grid, table and figure F5
│   ├── hemibrain.py · c1.py         hemibrain ingest + links (C1 / H4); the restricted data variant
│   ├── baselines/                   B0 random · B1 profiles · B2 NBLAST · B3 cocoa-style · B4 MLP
│   ├── report.py · runs.py          Phase-1 reports; run folders + aggregation
│   ├── package.py · smoke.py        Kaggle package; full-graph memory check
│   └── run.py                       one CLI for everything
├── scripts/                         paired comparisons, patience replay, C3 mirror control, C1 evaluation, match table
├── kaggle/flymatch_train.ipynb      the public Kaggle notebook (kaggle_train.py = same code as one cell)
├── results/                         every reported table and figure (written by scripts only)
├── docs/plan.md · docs/notes/       16-week plan; per-phase findings
└── tests/                           54 tests: units, synthetic harness, data contracts, leakage, controls
```

---

## 🧪 Rigour built in

- 🔒 **Frozen, hashed split.** Test families are touched **once**, at the end of week 12.
- 🧱 **Leakage tests:**
  - families are disjoint across splits;
  - sibling types sharing a name stem never sit in different splits;
  - negatives come only from training families;
  - seed types come only from training.
- 📏 **One evaluation harness** for every method, checked on synthetic data (identical embeddings → 100%, random → chance).
- 🎲 **Statistics:** 3 seeds per stochastic method; 95% CIs by bootstrapping **families**, not neurons; paired bootstrap for method comparisons.
- 🧹 **Clean-subset reporting** (MaleCNS `Reviewed` + BANC proofread) for every headline number.
- 🧾 **Every number comes from a script**, never a notebook. Every run folder stores the config, the git SHA and the split hash.
- 🧭 **Controls:**
  - degree-preserving edge shuffling (C4) ✅;
  - label re-pairing (C5) ✅. On the MLP it is never early-stopped on validation, because validation uses the *true* labels, which would rescue it;
  - within-dataset left↔right (C3) ✅, evaluation form;
  - hemibrain transfer (C1): MLP ✅, GNN in progress.
- 🔍 **Model choice is documented** in [`docs/notes/phase3_weeks8-9.md`](docs/notes/phase3_weeks8-9.md): depth, patience, the hard-negative / CORAL decision, and the protocol fixes found along the way.

---

## 🗺️ Roadmap

| Week | Work | Status |
|:-:|---|:-:|
| 1 | Repo, environment, fixed-dtype ingest, vocabulary maps, inclusion rules | ✅ |
| 2 | Threshold harmonisation, fractional weights, harmonisation report | ✅ |
| 3 | Node features, ground truth, fallback validation, clean subset | ✅ |
| 4 | Type families, stratified split — **frozen** (Gate G1) | ✅ |
| 5 | Evaluation harness, B0 / B1 / B2, Recall@k, filters | ✅ |
| 6 | B3 cocoa-style, B4 MLP, sparse graph + encoder (Gate G2) | ✅ |
| 7 | GPU training, checkpoint/resume, Kaggle package · public Kaggle notebook | ✅ |
| 8 | GNN vs MLP; depth (2 layers chosen) | ✅ |
| 9 | Hard negatives, CORAL (both dropped); ablation A2; controls C4, C5 — Gate G3: final model = 2-layer GNN | ✅ |
| 10 | Abstention head, assignment, GNN + B3 ensemble, C3 — Gate G4 | ✅ |
| 11–12 | **Seed-ablation curve** ✅ · hemibrain ingest ✅ · GNN C5 · remaining ablations · **test once** | 🟡 |
| 13 | Hemibrain transfer (C1): MLP ✅, GNN on Kaggle · error analysis | 🟡 |
| 14–15 | Web tool (FastAPI + Streamlit): paste a neuron ID, get its counterpart | ⬜ |
| 16 | Thesis, release, benchmark card, demo | ⬜ |

Day-by-day log: **[PROJECT_DIARY.md](PROJECT_DIARY.md)**.

---

## ⚠️ Known limitations (stated up front)

- **Circularity.** BANC types were partly assigned with NBLAST and connectivity matching to FlyWire, and MaleCNS types were harmonised against FlyWire. So the ground truth reaches across sexes through FlyWire names, and it was built with signals related to ours. Mitigations: the clean subset, manual inspection of errors, and comparing methods relative to each other.
- **Naming ≠ biology.** Only 53% of the BANC team's expert-reviewed BANC ↔ MaleCNS matches share our harmonised type. The disagreements are naming schemes (`R7` vs `R7y`, `EFFabg*` vs `MNad*`), concentrated in sensory and motor neurons. Those classes are therefore under-represented in the one-to-one ground truth.
- **Validation, not test.** Every number so far is on the validation split, which was also used for early stopping and model choice. The frozen test split is evaluated once, at the end.
- **B3 is a re-implementation.** cocoa does not support BANC, so B3 is our own cocoa-style method. It is side-blind, which is why the fair comparison is the **filtered** one. There, cocoa-style still wins with all labels.
- **Optic-lobe columns** (hundreds of near-identical copies) are evaluated at type level (T2), not neuron level.
- **VNC neuropils** were merged coarsely, and `mVAC_T2` was excluded after failing merge validation.

---

## 📚 Citing the data

> Bates, A. S., Phelps, J. S., Kim, M., … Lee, W.-C. A. (2026). *Distributed control circuits across a brain-and-cord connectome.* **Nature** 656, 957–970. Data: Harvard Dataverse, doi:10.7910/DVN/7WTH1N.

> *Sexual dimorphism in the complete Drosophila male CNS connectome.* **Cell** (2026). MaleCNS v1.0, male-cns.janelia.org.

> Matsliah, A., et al. (2026). *Uncovering sex differences in the Drosophila ventral nerve cord through connectome alignment.* bioRxiv, doi:10.64898/2026.06.14.732053.

More in [`DATA_LICENSE.md`](DATA_LICENSE.md) and [`docs/plan.md`](docs/plan.md#14-references-and-data-sources).

---

## 📄 Licence

Code: **Apache License 2.0**; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Data in `data/raw` and `data/external`, and data derived from it: **CC BY 4.0**, belonging to the original authors; see [DATA_LICENSE.md](DATA_LICENSE.md).

<div align="center"><sub>Final-year B.E. project · built by <a href="https://github.com/prabhu-omkar">@prabhu-omkar</a></sub></div>
