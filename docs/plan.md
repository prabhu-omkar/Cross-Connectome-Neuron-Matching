# Cross-Connectome Neuron Matching — Project Plan & Workflow

**Project:** Learning neuron correspondence between the female (BANC) and male (MaleCNS) *Drosophila* central nervous systems
**Data folder:** `C:\projects\FLYbrain`
**Plan version:** 1.0 — 26 Oct 2026
**Schedule:** 16 weeks (matches the Gantt chart in the pre-project deck)

---

## 0. One-page summary

**What we are building.** A model that takes any neuron from one connectome and returns its counterpart in the other — or says, with a calibrated confidence, that it has none. Both connectomes are embedded into one shared vector space by a graph neural network (GNN) with shared weights; matching is nearest-neighbour search plus an assignment step.

**What we will claim (and only this).**

1. A frozen, public benchmark for BANC ↔ MaleCNS neuron matching, with baseline scores. None exists today.
2. The first measurement of how cross-connectome matching accuracy falls as curated correspondence labels ("seeds") are withheld — for our model and for a cocoa-style label-propagation baseline.
3. "No counterpart" as a calibrated, first-class output, evaluated properly.
4. Evidence — via controls — that the model learned homology rather than the imaging pipeline.

**What we will not claim.** That we discovered sex differences (already published), that we beat expert curation, or that the model generalises to other species.

**Where we stand.** Data downloaded; feasibility profiling done; a zero-learning baseline already finds the exact partner at rank 1 for 29.3% of known 1:1 pairs (rank ≤ 10 for 47.1%). There is real signal before any learning.

**The five things that matter most in this plan**

| # | Decision | Why |
|---|---|---|
| 1 | Female dataset is **BANC v888**, not FlyWire FAFB | That is what was downloaded; it covers brain *and* nerve cord, same scope as MaleCNS |
| 2 | Core pipeline uses **only the Codex exports** (connectivity + annotations); morphology is an optional add-on | Removes the riskiest dependency (skeleton registration) from the critical path |
| 3 | **Harmonise the two datasets before anything else** (synapse thresholds, neuropil names, super-class names, scale) | The exports differ in ways that are pure artefact; a model will happily learn them |
| 4 | **Ground truth = harmonised cell-type names**, split by type *family*, frozen and hashed | Prevents leakage; makes every number reproducible |
| 5 | **The seed-ablation curve is the central result**; adversarial alignment is an optional ablation arm | Protects the one novel measurement if time runs short |

---

## 1. Status and what changed since the proposal

### 1.1 Done

- [x] Downloaded Codex exports for MaleCNS v1.0 and BANC v888 (4 files, ~80 MB compressed)
- [x] Feasibility profiling and a zero-learning baseline (21 Oct, saved in the project as `dataset-feasibility-check.md`)
- [x] Vocabulary and ground-truth analysis (26 Oct, this plan — Section 2)
- [x] Research on data sources, tools and prior work (Section 3)

### 1.2 What the proposal assumed vs what is true

| Proposal / deck said | Reality | Action |
|---|---|---|
| Female dataset = FlyWire FAFB (~140k neurons, brain only) | You downloaded **BANC v888** (female brain + nerve cord, 158,262 neurons in the export) | Use BANC. Update deck slide 3. FAFB becomes an optional extra transfer target |
| cocoa is baseline 3 | **cocoa does not support BANC yet** (its README lists BANC as to-do) | Re-implement a cocoa-style baseline ourselves (Section 7.2.3) — it is ~60 lines |
| We compute NBLAST ourselves | The BANC team published **precomputed NBLAST matrices**, including BANC ↔ MaleCNS v0.9 | Download and use for baseline 2; check ID coverage against v1.0 |
| We register skeletons into a common space | The BANC team published **MaleCNS skeletons already in BANC space** (v0.9) plus JRC2018F→BANC registrations | Morphology becomes optional and much cheaper |
| Connections filtered to ≥3 synapses | **BANC export ≥3 per neuron pair; MaleCNS export ≥5 per pair** | Must harmonise (Section 7.1.4). The 21 Oct note said ≥3 for both — that was wrong for MaleCNS |
| 30-week timeline (project guide) | 16 weeks (department format) | Schedule in Section 11; adversarial branch is the first thing to cut |

---

## 2. The data, in detail

All numbers below were measured on the files in `C:\projects\FLYbrain` (26 Oct 2026) unless marked otherwise.

### 2.1 Files

```
C:\projects\FLYbrain\
├── BANC_female_brain_and_nerve_cord\
│   ├── neurons.csv.gz                 2.9 MB   158,262 rows × 21 columns
│   └── connections_princeton.csv.gz  29.2 MB   3,990,039 rows × 5 columns
└── MCNS_male_brain_and_nerve_cord\
    ├── neurons.csv.gz                 2.3 MB   166,700 rows × 21 columns
    └── connections_princeton.csv.gz  45.1 MB   9,039,608 rows × 5 columns
```

`connections_princeton.csv.gz` columns: `pre_root_id, post_root_id, neuropil, syn_count, nt_type`.
Each row is one (pre, post, neuropil) triple, so a single neuron pair is split across several rows when its synapses fall in several neuropils.

`neurons.csv.gz` columns (identical schema in both): `Root ID, Top in/out region, Community labels, Predicted NT type, Predicted NT confidence, Verified NT type, Verified Neuropeptide, Body Part, Function, Flow, Super Class, Class, Sub Class, Hemilineage, Nerve, Soma side, Primary Cell Type, Alternative Cell Type(s), Cable length (nm), Surface area (nm^2), Volume (nm^3)`.

### 2.2 Side-by-side profile

| Property | MaleCNS v1.0 (male) | BANC v888 (female) |
|---|---|---|
| Imaging | FIB-SEM, 8×8×8 nm | serial-section TEM, 4×4×45 nm |
| Neurons in export | 166,700 | 158,262 |
| Connection rows (neuropil-split) | 9,039,608 | 3,990,039 |
| Distinct neuron pairs | 6,242,118 | 3,037,361 |
| **Minimum synapses per pair** | **5** | **3** |
| Mean synapses per row | 9.94 | 5.90 |
| Total synapses (from 21 Oct check) | 89.9 M | 23.6 M (**3.8× fewer**) |
| Median degree (21 Oct check) | 46 | 20 |
| Neuropil labels in connections | 134 | 111 (84 identical names) |
| `nt_type` in connections | empty | empty |
| Primary Cell Type filled | 98.7% (11,723 types) | 74.9% (11,546 types) |
| Predicted NT filled | 90.2% | 94.5% |
| Verified NT filled | 0% | 41.3% |
| Soma side | left 79,102 · right 81,199 · center 392 · missing 6,007 | left 75,760 · right 82,502 |
| Super Class | 100% filled, 27 values | 92.7% filled, 16 values |
| Flow / Body Part / Function | empty | 92.8% / 12.4% / 15.2% |
| Hemilineage | 31.2% (251 values) | 31.9% (259 values) |
| Community labels | `key: value` pairs (see 2.3) | free text, no keys |
| Cable length / surface / volume | **empty** | **empty** |
| Root ID format | small integers (e.g. `10001`) | 18-digit CAVE IDs (e.g. `720575941076956631`) |

Super Class top values:

- MaleCNS: `ol_intrinsic` 89,403 · `cb_intrinsic` 32,164 · `vnc_intrinsic` 13,161 · `visual_projection` 9,201 · `vnc_sensory` 6,370 · `ol_sensory` 6,098 · `cb_sensory` 4,868 · `ascending_neuron` 1,846 · `descending_neuron` 1,314 · `vnc_motor` 708 · (17 more)
- BANC: `optic_lobe_intrinsic` 72,582 · `central_brain_intrinsic` 31,782 · `sensory` 16,557 · `ventral_nerve_cord_intrinsic` 12,851 · *(missing)* 11,499 · `visual_projection` 7,321 · `ascending` 1,849 · `descending` 1,316 · `motor` 805 · `sensory_ascending` 516 · (6 more)

Predicted NT distribution — note the large gaps in DA and HIST, which partly reflect sensory/photoreceptor coverage differences, not biology:

| NT | MaleCNS | BANC |
|---|---|---|
| ACH | 94,947 | 85,006 |
| GLUT | 28,058 | 24,382 |
| GABA | 20,231 | 20,604 |
| DA | 4,443 | 8,072 |
| HIST | 2,126 | 7,694 |
| SER | 465 | 1,612 |
| OCT | 101 | 1,903 |
| TYR | — | 209 |
| missing | 16,329 | 8,780 |

### 2.3 Cross-dataset labels hiding in MaleCNS "Community labels"

MaleCNS rows carry `key: value` labels that link to other datasets. Counts of neurons carrying each key:

| Key | Neurons | Use |
|---|---|---|
| `statusLabel` | 166,689 | Tracing status → defines the "clean" evaluation subset |
| `flywireType` | 47,162 | The FlyWire (FAFB) type name → **extra ground truth** (BANC types are FAFB-style names) |
| `supertype` | 34,096 | Coarser grouping → helps define type families |
| `mancBodyid` | 18,572 | Link to MANC (male VNC) |
| `hemibrainType` | 7,996 | Link to hemibrain → needed for the hemibrain control |
| `instance` | 5,455 | — |
| `mancType` | 4,008 | MANC type name → extra ground truth in the VNC |
| `synonyms` | 3,954 | — |

### 2.4 How much ground truth do we have?

Matching MaleCNS labels against BANC `Primary Cell Type`:

| Join rule | Shared types | MaleCNS neurons in shared types | Unambiguous 1:1 pairs (same type + side, exactly one neuron each) |
|---|---|---|---|
| MaleCNS `Primary Cell Type` = BANC type | 7,534 | 122,129 | **5,096** |
| MaleCNS `flywireType` = BANC type | 2,637 | 43,250 | 1,641 |
| MaleCNS `mancType` = BANC type | 373 | 3,308 | 209 |
| **Combined** (primary → flywireType → mancType) | **9,973** | **158,481** | **6,774** |

The fallback labels add 33% more 1:1 pairs. They must be validated before use (Section 7.1.6) — some fallbacks are near-miss names (`VS` vs `VS1`, `LoVC16` vs `cL21`).

The 21 Oct check reported 5,128 1:1 pairs and found 4,189 MaleCNS-only and 4,012 BANC-only types. Most of those one-sided types are naming differences (`Dm3a/b/c` vs `Dm3p/q/v`, `R7/R8` vs `R8y/R8p`), **not** sex-specific neurons — so they cannot be used as "no counterpart" labels without curation.

### 2.5 Baseline already measured (21 Oct)

Cosine similarity of square-root-normalised neuropil input/output profiles (170 features), 5,128 1:1 pairs, each MaleCNS query ranked against all 153,962 BANC neurons:

| Recall@1 | Recall@10 | Recall@100 | Median rank | Shuffled control |
|---|---|---|---|---|
| 29.3% | 47.1% | 64.4% | 15 | ≈ 0% |

This is baseline 1. The model has to beat it clearly, on the frozen test split, across seeds.

### 2.6 Pitfalls already visible in the data

1. **18-digit IDs.** BANC root IDs exceed float precision. If pandas ever parses them as float (it will if a column has a missing value and no dtype is given) they silently change — `720575941076956631` becomes `720575941076956672`. Always read IDs as `int64` or `str`. **Never open these files in Excel** — Excel rounds 18-digit numbers to 15 significant digits.
2. **Different thresholds.** BANC keeps pairs ≥ 3 synapses, MaleCNS ≥ 5. Untreated, BANC neurons look like they have extra weak partners.
3. **3.8× synapse scale difference.** Synapse detection differs between pipelines. Raw counts and raw degrees are not comparable.
4. **Different neuropil vocabularies.** MaleCNS has nerves (`AbN1_R`, `ADMN_L`, `ProAN_L` …) and `*_UNASGD` buckets; BANC uses `VNC_*` prefixes, splits `IB` into `IB_L/IB_R`, and has `GA_L/R`.
5. **Different super-class vocabularies.** `ol_intrinsic` vs `optic_lobe_intrinsic`, three MaleCNS sensory classes vs one BANC `sensory`.
6. **Coverage differences.** BANC includes many peripheral sensory neurons (retina, lamina, antenna, legs) and `sensory_fragment` entries. MaleCNS has 6,007 neurons with no soma side.
7. **Optic lobe dominance.** ~54% of MaleCNS neurons are optic-lobe intrinsic, mostly columnar types with hundreds of near-identical copies. Neuron-level 1:1 matching is ill-posed there; unstratified metrics will be dominated by them.

---

## 3. Research findings and what they mean for the design

### 3.1 BANC (the female dataset)

- **Paper:** Bates, A. S., Phelps, J. S., Kim, M., … Lee, W.-C. A. (2026). *Distributed control circuits across a brain-and-cord connectome.* Nature 656, 957–970.
- **Imaging:** serial-section EM, 4 nm in-plane (4×4×45 nm).
- **Typing:** 147,846 neurons linked to a cell-type label (93.1% in central brain and VNC). Labels were assigned by matching to earlier datasets (hemibrain, FAFB-FlyWire, FANC, MANC) using NBLAST + manual review, connectivity-based graph alignment, and an annealing algorithm combining morphology and connectivity. Unmatched neurons were attributed to "genetic variation, developmental noise and limitations in data quality."
- **Data:** Harvard Dataverse snapshot (doi:10.7910/DVN/7WTH1N); live data on FlyWire Codex; Python package `banc`; R package `bancr`.
- **Useful published products** (paths from the BANC data-locations page):
  - `banc_888_meta.feather` — fuller metadata than the Codex export. The paper distinguishes "reviewed" from "automated" type assignments; check whether this file has a column for that (needed for the clean evaluation subset)
  - `banc_888_edgelist_simple_v2.feather` (≥ 5 synapses) and `_v3` (≥ 10)
  - `somas_v1.parquet` — soma positions
  - SWC skeletons (`banc_banc_space_swc`), meshes
  - **NBLAST matrices:** `banc_malecns_v0.9_nblast.feather`, `banc_hemibrain_v1.2.1_nblast.feather`, `banc_fafb_783_nblast.feather`, `banc_manc_…`, `banc_fanc_…`, `banc_mirror_nblast.feather`
  - Registrations JRC2018F → BANC and JRC2018VNCF → BANC (elastix)

**Implications.**

- BANC's own type labels were produced partly by NBLAST and connectivity matching to FAFB. MaleCNS types were harmonised against FlyWire. So **BANC ↔ MaleCNS ground truth is two-hop through FAFB names**, and it was built with the same signals our model uses. This is the circularity caveat — report the reviewed/traced subset separately (Section 8.4).
- The precomputed BANC ↔ MaleCNS NBLAST matrix gives baseline 2 almost for free.

### 3.2 MaleCNS (the male dataset)

- **Release:** `male-cns:v1.0`, CC-BY 4.0; downloads at male-cns.janelia.org/download.
- **Imaging:** FIB-SEM, 8 nm isotropic. 166,700 neurons; 99% of nuclei associated with a proofread neuron.
- **Bulk files** (`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`): body annotations (13 MB), body neurotransmitters (42 MB), body stats (780 MB), connectome weights (1.1 GB), synapse points (12.7 GB), synapse partners (6.8 GB).
- **Skeletons:** SWC in native space (8 nm), mirrored, and **JRC2018 unisex template space** (1 µm).
- The BANC team's tutorial bucket also hosts **MaleCNS v0.9 skeletons already transformed into BANC space** (`malecns_banc_space_swc/`) and a `JRC2018U/` folder.

**Implications.** We never need the 12.7 GB or 6.8 GB files. If morphology is added, use pre-transformed skeletons rather than running registrations ourselves.

### 3.3 Prior work on exactly this problem

- **Matsliah, A., et al. (2026).** *Uncovering sex differences in the Drosophila ventral nerve cord through connectome alignment.* bioRxiv, doi:10.64898/2026.06.14.732053. Male/female **VNC** alignment with ACDC (alternating continuous/discrete combinatorial optimisation) plus a connectivity-similarity score (ConSim). Expert review preferred connectivity matching over NBLAST in **62%** of cases vs **7%** for NBLAST (26% tie). Sex-specific neurons get placeholder identities. **Limits:** requires equal-sized graphs (padding), scales to roughly **20,000–25,000 neurons**; whole-CNS alignment "would require a piecewise strategy."
- **Yad & Nabila (2026).** *Reconstruction differences rival individual differences in the fly compass connectome and break trained circuit models.* arXiv:2610.29205. Compares MaleCNS, hemibrain, FlyWire and BANC: changing synapse detection within one fly moved the circuit more than halfway to the smallest between-fly difference; models trained on one reconstruction did not transfer to any other.
- **cocoa** (flyconnectome): co-clusters neurons across datasets using a joint connectivity vector built from already-matched labels. Supports FlyWire, hemibrain, MANC, MaleCNS — **not BANC**.

**Implications.**

- Connectivity-based matching is known to beat morphology (Matsliah). Our contribution is not "use connectivity" — it is **scale (140k–166k nodes in one pass), label efficiency, calibrated abstention, and transfer**.
- Reconstruction differences are large enough to break models (Yad & Nabila). This independently justifies the harmonisation step and the controls. Cite it in the problem statement.
- Matsliah's ACDC is the natural *within-VNC* reference point. If time allows, compare on the VNC subset (stretch goal; not scheduled).

### 3.4 Tools

| Need | Tool | Note |
|---|---|---|
| Tables | pandas ≥ 2.2, pyarrow | Store everything as Parquet |
| Sparse maths | scipy.sparse | Label-projected connectivity, permutations |
| GNN | PyTorch 2.x + PyTorch Geometric ≥ 2.5 | Use sparse adjacency (Section 7.3.1) or you will run out of GPU memory |
| Nearest neighbours | blocked `torch` matmul (exact); FAISS optional | 166k × 158k exact search is seconds on a GPU |
| Assignment | `scipy.optimize.linear_sum_assignment` | Run per connected component |
| Calibration | scikit-learn (`LogisticRegression`, `IsotonicRegression`) | — |
| Morphology (optional) | navis, navis-flybrains | Only if Phase 1.4 is attempted |
| Hemibrain access | neuprint-python (needs free token) or the BANC tutorial bucket `hemibrain_121` | Request the neuPrint token in week 1 |
| Tracking | plain JSON/CSV run logs; Weights & Biases free tier optional | — |
| Web tool | FastAPI + Streamlit (or React) | Streamlit saves ~1 week vs React |

---

## 4. Problem definition

### 4.1 Formal task

Let **M** be the set of MaleCNS neurons and **B** the set of BANC neurons. Each neuron has node features (Section 7.1.5) and sits in its dataset's directed, weighted connectivity graph.

We learn an encoder `g` (shared weights) mapping each neuron to a unit vector `z = g(neuron, graph) ∈ ℝ¹²⁸`. From the embeddings we produce a **partial matching**

```
f : M → B ∪ {∅}        and symmetrically   f' : B → M ∪ {∅}
```

where `∅` means "no counterpart in the other dataset".

### 4.2 Three evaluation tasks

| Task | Question | Ground truth | Main metric |
|---|---|---|---|
| **T1 — neuron retrieval** | For a neuron with a known unique partner, where does the partner rank? | 1:1 pairs (same type + side, one neuron each) in test families | Recall@1/10, MRR |
| **T2 — type transfer** | Does the nearest neighbour in the other dataset have the same cell type? | All neurons of test-family types present in both datasets | Type-accuracy@1, macro-averaged over super classes |
| **T3 — abstention** | Can the model tell when there is no partner? | Partner-removal simulation + curated sex-specific neurons | AUPRC, calibration error |

T1 is strict and only defined where identity is unique. T2 covers the many-copy types (optic lobe columns, Kenyon cells) where neuron-level identity is ill-posed.

### 4.3 The central experiment

Train with only a fraction `s ∈ {100, 50, 25, 10, 5, 0}%` of the training families' cross-dataset correspondences ("seeds"). Plot T1/T2 accuracy against `s` for: our GNN, an MLP (no graph), the cocoa-style baseline, and the seed-free baselines (flat lines). **The gap between our curve and cocoa-style in the 5–25% range is the paper.**

Expected behaviour to state up front: at `s = 0%` the GNN has no cross-dataset supervision, so it should fall back toward the neuropil-profile baseline. That is not a failure — it is the left anchor of the curve. The optional bilateral pre-training (Section 7.3.5) is the one thing that could lift the 0% point, and it must be reported as using within-dataset labels.

### 4.4 Hypotheses (write these into the thesis before running anything)

- **H1.** A shared-weight GNN beats the neuropil-profile baseline (B1) on T1 Recall@1 by more than the across-seed standard deviation.
- **H2.** Message passing helps: the GNN beats the same model with zero layers (the MLP, B4).
- **H3.** As seeds fall from 100% to 10%, the cocoa-style baseline degrades faster than the GNN.
- **H4.** The model transfers to BANC ↔ hemibrain (same sex, different imaging) better than a model that learned imaging signatures would (Control C1).
- **H5.** With degree-preserving edge shuffling, connectivity-driven gains vanish (Control C4).

Each hypothesis can come out false. That is fine — the controls make a negative result interpretable.

---

## 5. Codebase architecture

### 5.1 Repository layout

Create a git repository at `C:\projects\FLYbrain\flymatch`. Keep the raw downloads where they are.

```
C:\projects\FLYbrain\
├── BANC_female_brain_and_nerve_cord\        # raw, read-only, never edited
├── MCNS_male_brain_and_nerve_cord\          # raw, read-only, never edited
├── external\                                 # optional downloads (BANC meta, NBLAST, hemibrain, skeletons)
└── flymatch\                                 # ← git repository
    ├── README.md
    ├── plan.md                               # this file
    ├── environment.yml
    ├── pyproject.toml
    ├── configs\
    │   ├── paths.yaml                        # where raw data lives (machine-specific, git-ignored copy)
    │   ├── data.yaml                         # thresholds, inclusion rules
    │   ├── features.yaml
    │   ├── model\gnn.yaml, mlp.yaml
    │   ├── train\default.yaml
    │   └── experiments\*.yaml                # one file per experiment grid
    ├── maps\                                 # hand-written, version-controlled
    │   ├── neuropil_map.csv
    │   ├── superclass_map.csv
    │   └── nt_map.csv
    ├── src\flymatch\
    │   ├── io.py                             # readers with fixed dtypes
    │   ├── ingest.py                         # raw → canonical parquet
    │   ├── harmonise.py                      # thresholds, vocab maps, scale normalisation
    │   ├── features.py                       # node features
    │   ├── groundtruth.py                    # type harmonisation, 1:1 pairs, families
    │   ├── splits.py                         # family split + freeze + hash
    │   ├── graph.py                          # PyG Data objects, sparse adjacency
    │   ├── retrieval.py                      # blocked exact kNN, ranks
    │   ├── baselines\
    │   │   ├── profile_cosine.py             # B1
    │   │   ├── nblast.py                     # B2
    │   │   ├── cocoa_style.py                # B3
    │   │   └── mlp.py                        # B4
    │   ├── models\
    │   │   ├── layers.py                     # directed GNN layer
    │   │   ├── encoder.py
    │   │   ├── losses.py                     # multi-positive InfoNCE, GRL, CORAL
    │   │   └── abstention.py
    │   ├── train.py
    │   ├── assign.py                         # Hungarian per component + abstention
    │   ├── evaluate.py                       # T1/T2/T3 metrics, bootstrap CIs
    │   ├── controls.py                       # edge permutation, label shuffle, L↔R
    │   └── run.py                            # config-driven entry point
    ├── data\                                 # git-ignored, regenerated by `make data`
    │   ├── interim\                          # canonical parquet tables
    │   ├── processed\                        # features, graphs
    │   └── splits\splits_v1.json             # committed (small) + its hash
    ├── runs\                                 # git-ignored; one folder per run
    ├── results\                              # aggregated tables/figures (committed)
    ├── notebooks\                            # exploration only — never the source of a reported number
    ├── tests\                                # pytest data-contract tests
    └── app\                                  # Phase 5 web tool
```

### 5.2 Rules that keep the project honest

1. **Raw data is read-only.** Everything in `data\` is regenerated from raw by one command.
2. **Every reported number comes from a script**, never from a notebook cell or a hand-copied value.
3. **The split file is frozen** once written: `splits_v1.json` plus its SHA-256 are committed. Every run logs the hash it used; the aggregator refuses runs with a different hash.
4. **Test is touched once**, at the end of Phase 4. All tuning happens on validation.
5. **One config per run**, saved inside the run folder with the git commit hash.
6. **Tokens** (neuPrint, CAVE) live in environment variables, never in the repo.

### 5.3 Canonical data contracts

These tables are the interface between phases. Write `tests\test_contracts.py` to assert them.

**`data/interim/neurons.parquet`** — one row per neuron, both datasets

| Column | Type | Notes |
|---|---|---|
| `dataset` | category | `"mcns"` or `"banc"` |
| `neuron_id` | int64 | Root ID — never float |
| `uid` | int32 | Dense index 0…N−1 *within* dataset (graph node index) |
| `side` | category | `left`, `right`, `center`, `unknown` |
| `superclass_raw` | str | As exported |
| `superclass` | category | Harmonised (Appendix B) |
| `class_raw`, `subclass_raw` | str | Kept for analysis only |
| `hemilineage` | str | Missing allowed |
| `nt` | category | Harmonised NT code |
| `nt_conf` | float32 | Missing → NaN |
| `type_primary` | str | Primary Cell Type |
| `type_alt` | str | Alternative Cell Type(s) — **never used as a feature** |
| `flywire_type`, `manc_type`, `hemibrain_type`, `supertype` | str | Parsed from MaleCNS community labels; empty for BANC |
| `status` | str | MaleCNS `statusLabel`; BANC review status if available |
| `type_h` | str | Harmonised type (Section 7.1.6) |
| `include` | bool | Passes inclusion rules (Section 7.1.3) |

**`data/interim/edges_np.parquet`** — neuropil-split edges after harmonised thresholding

| Column | Type |
|---|---|
| `dataset` | category |
| `pre_uid`, `post_uid` | int32 |
| `neuropil` | category (canonical name) |
| `syn` | int32 |

**`data/interim/edges.parquet`** — pair-level edges (sum over neuropils): `dataset, pre_uid, post_uid, syn, w_in, w_out`
where `w_in = syn / (total input synapses of post)` and `w_out = syn / (total output synapses of pre)`.

**`data/processed/features_{dataset}.npy`** — float32 matrix `[N, F]`, row `i` = neuron with `uid == i`, plus `feature_spec.json` naming each column block.

**`data/interim/gt_pairs_1to1.parquet`** — `type_h, side, mcns_uid, banc_uid, family, source` (`source` ∈ primary / flywire / manc).

**`data/interim/gt_types.parquet`** — one row per harmonised type present in both datasets: `type_h, family, superclass, n_mcns, n_banc, size_bucket`.

**`data/splits/splits_v1.json`** — `{"version": 1, "seed": 20261026, "train": [families], "val": [...], "test": [...], "sha256": "..."}`.

---

## 6. Environment setup (Windows laptop + free GPU)

### 6.1 Laptop (all data work, baselines, analysis)

1. Install **Miniforge** (conda-forge). Avoid mixing pip and conda for the scientific stack.
2. `environment.yml`:

```yaml
name: flymatch
channels: [conda-forge]
dependencies:
  - python=3.11
  - pandas>=2.2
  - pyarrow>=15
  - numpy>=1.26
  - scipy>=1.12
  - scikit-learn>=1.4
  - matplotlib
  - seaborn
  - pyyaml
  - tqdm
  - pytest
  - jupyterlab
  - pip
  - pip:
      - torch            # CPU build is fine on the laptop
      - torch_geometric
      - umap-learn
```

3. `conda env create -f environment.yml` → `conda activate flymatch` → `pip install -e .` from inside `flymatch\`.
4. Set Windows long-path support on, or keep the repo path short (it already is).
5. Check: `python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"`.

The full ingest of both datasets takes roughly 10–20 seconds on a laptop (measured: reading the 9.0 M-row MaleCNS file took ~5 s). RAM needed is under 4 GB.

### 6.2 GPU (Phase 3–4 training)

- **Kaggle Notebooks**: ~30 GPU-hours/week, T4×2 or P100. Better than Colab for the ablation grid.
- **Google Colab free**: T4, sessions disconnect. Use for interactive debugging.
- Upload `data\processed\` (well under 1 GB) as a Kaggle Dataset / Google Drive folder. Do **not** upload raw CSVs — upload the processed Parquet/NumPy.
- Checkpoint every 200 steps to persistent storage. Assume every session dies.

### 6.3 Optional external downloads (start the slow ones in week 1)

| What | From | Needed for | Priority |
|---|---|---|---|
| neuPrint token | neuprint.janelia.org (sign in with Google) | Hemibrain control (C1) | **Week 1** |
| `banc_888_meta.feather` | Harvard Dataverse doi:10.7910/DVN/7WTH1N | Review status, extra columns | High |
| `banc_malecns_v0.9_nblast.feather` | Dataverse (`nblast/`) | Baseline B2 | High |
| `banc_hemibrain_v1.2.1_nblast.feather` | Dataverse | Control C1 with morphology | Medium |
| Hemibrain v1.2.1 neurons + connectivity | neuPrint, or BANC tutorial bucket `compiled_data/hemibrain_121/` | Control C1 | Medium |
| MaleCNS `body-annotations-male-cns-v1.0-minconf-0.5.feather` (13 MB) | male-cns.janelia.org/download | Soma positions, dimorphism fields if present | Medium |
| Skeletons in a common space | BANC tutorial bucket `malecns_banc_space_swc/` + BANC SWCs | Morphology add-on only | Low |

Google Cloud Storage files can be downloaded with `gcloud storage cp gs://<bucket>/<path> .`, or in a browser via `https://storage.googleapis.com/<bucket>/<path>`. Record the exact version and date of every download in `external\SOURCES.md`.

---

## 7. Phase-by-phase workflow

Each phase lists **goal → inputs → steps → outputs → done-when → pitfalls**. Week numbers match the deck's Gantt chart (activity numbers 1.1 … 6.1 in brackets).

```
 raw CSVs ──► 1 ingest ──► 2 harmonise ──► 3 features ──► 4 ground truth ──► 5 splits (frozen)
                                                                              │
              ┌───────────────────────────────────────────────────────────────┘
              ▼
   Phase 2: baselines B0–B4 + evaluation harness ──► Recall@k, numbers to beat
              │
              ▼
   Phase 3: GNN encoder ──► contrastive training ──► abstention ──► assignment
              │
              ▼
   Phase 4: seed ablation · controls C1–C5 · ablations · error analysis  (test touched once)
              │
              ▼
   Phase 5: export match table ──► FastAPI ──► web UI          Phase 6: thesis · paper · release
```

---

### PHASE 1 — Data foundation (Weeks 1–4) [1.1, 1.2, 1.3]

**Goal:** one command turns the four raw files into canonical tables, harmonised features, validated ground truth and a frozen split.

#### 7.1.1 Ingest (Week 1) [1.1]

Read with fixed dtypes; never let pandas guess.

```python
# src/flymatch/io.py
import re
import pandas as pd

NEURON_RENAME = {
    "Root ID": "neuron_id", "Soma side": "side_raw", "Super Class": "superclass_raw",
    "Class": "class_raw", "Sub Class": "subclass_raw", "Hemilineage": "hemilineage",
    "Predicted NT type": "nt_raw", "Predicted NT confidence": "nt_conf",
    "Verified NT type": "nt_verified", "Primary Cell Type": "type_primary",
    "Alternative Cell Type(s)": "type_alt", "Community labels": "labels_raw",
    "Flow": "flow", "Body Part": "body_part",
}

def read_neurons(path, dataset):
    df = pd.read_csv(path, dtype=str)                       # everything as text first
    df = df.rename(columns=NEURON_RENAME)[list(NEURON_RENAME.values())]
    df["neuron_id"] = df["neuron_id"].astype("int64")       # exact for 18-digit BANC IDs
    df["nt_conf"] = pd.to_numeric(df["nt_conf"], errors="coerce").astype("float32")
    df.insert(0, "dataset", dataset)
    return df

def read_edges(path, dataset):
    e = pd.read_csv(
        path, usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"],
        dtype={"pre_root_id": "int64", "post_root_id": "int64",
               "neuropil": "category", "syn_count": "int32"})
    e.columns = ["pre", "post", "neuropil_raw", "syn"]
    e.insert(0, "dataset", dataset)
    return e

LABEL_KEYS = ["statusLabel", "flywireType", "mancType", "hemibrainType", "supertype", "mancBodyid"]

def parse_mcns_labels(s):
    """'hemibrainType: OCG01,statusLabel: Roughly traced' -> dict"""
    out = {}
    if not isinstance(s, str):
        return out
    for k in LABEL_KEYS:
        m = re.search(rf"(?:^|,)\s*{k}:\s*([^,]+)", s)
        if m:
            out[k] = m.group(1).strip()
    return out
```

Then:

1. Build `uid` per dataset (0 … N−1, sorted by `neuron_id`) and map `pre`/`post` to `pre_uid`/`post_uid`.
2. **Drop edges whose endpoints are not in the neuron table** and log how many.
3. Drop self-loops (pre == post); log how many.
4. Write `data/interim/neurons_raw.parquet` and `edges_raw.parquet`.

**Checks (put in `tests/`):** neuron IDs unique per dataset; no ID changed after the int64 round-trip (compare against the string column); edge count equals raw row count minus logged drops.

#### 7.1.2 Vocabulary maps (Week 1) [1.1]

Three hand-written CSVs in `maps/`, version-controlled. The ingest script **fails loudly** if any raw value is missing from a map — that guarantees nothing gets silently dropped when a dataset is updated.

1. **`superclass_map.csv`** — `dataset, raw, harmonised, include`. Full starter table in Appendix B. Key decisions:
   - Merge MaleCNS `ol_sensory`, `cb_sensory`, `vnc_sensory` (+ `_tbc` variants) → `sensory` (BANC has one sensory class).
   - Exclude BANC `glia`, `trachea`, `not_a_neuron`.
   - BANC rows with missing super class (11,499) → `unknown`; keep them in the graph (they carry connectivity) but exclude from ground truth and evaluation.
2. **`neuropil_map.csv`** — `dataset, raw, canonical, region, include`, `region ∈ {OL, CB, GNG, VNC, nerve, unassigned}`. Starter rules in Appendix A:
   - The 84 identical names map to themselves.
   - `IB` (MaleCNS, unsplit) ↔ `IB_L`/`IB_R` (BANC): map **both** to `IB` (lose side, keep comparability).
   - MaleCNS `*_UNASGD` buckets and BANC `Xnerve` → `unassigned`, excluded from profiles but **kept for degree/weight totals**.
   - VNC: the core neuropils are already shared by name (`LegNp_T1–T3_L/R`, `ANm`, `IntTct`, `LTct`, `NTct_UTct_T1_L/R`, `WTct_UTct_T2_L/R`, `HTct_UTct_T3_L/R`). The differences are at the edges: MaleCNS lists nerves and `mVAC_T1–T3`; BANC lists `VNC_*` margin regions, `ABDNM_L/R`, `Tct`. Merge only where the correspondence is clear (`mVAC_T2` ↔ `VNC_T2_mVAC`); send the rest to a per-side `VNC_other` bucket or exclude. Losing resolution is better than inventing correspondences.
   - Nerves (`AbN*`, `ADMN_*`, `PDMN_*`, `CvN_*`, `DProN_*`, `VProN_*`, `PrN_*`, `ProAN_*`, `ProLN_*`, `MesoAN_*`, `MesoLN_*`, `MetaLN_*`, `DMetaN_*`) → region `nerve`, excluded from profiles.
   - Brain names present in only one dataset: proposed merges in Appendix A (`AMMC` + `SAD`, `GA` → `LAL`, `AB` → `FB`, cervical connective in both). `LA_L/R` (lamina, MaleCNS only) is excluded.
   - **Validate every merge with data:** for training 1:1 pairs, the per-neuron synapse fraction in a merged neuropil should correlate across datasets. Keep a merge if Pearson r ≥ 0.5; otherwise exclude that neuropil.
3. **`nt_map.csv`** — `ACH, GLUT, GABA, DA, HIST, SER, OCT, TYR` map to themselves; missing → `unknown`.

**Done when:** `python -m flymatch.ingest` runs with zero unmapped values and prints, per dataset, the fraction of synapses that land in canonical (non-excluded) neuropils. Expect > 90%; if lower, the map is too aggressive.

#### 7.1.3 Inclusion rules (Week 1) [1.1]

A neuron is `include = True` if all hold:

- harmonised super class not in {excluded, unknown}
- `type_primary` does not contain `fragment` (BANC has 388 `sensory_fragment` entries)
- side ∈ {left, right, center} (MaleCNS has 6,007 with no side)
- total synapses (in + out, after thresholding) ≥ **20** — tune once on train data, then freeze

Excluded neurons **stay in the graph** (they are real partners); they are only removed from training anchors, ground truth and evaluation. Log counts per super class.

#### 7.1.4 Harmonise connectivity (Week 2) [1.2]

The single most important step for credibility. The model must not be able to tell the datasets apart from trivial artefacts.

1. **Equal thresholds.** Aggregate neuropil rows to neuron pairs; keep pairs with total ≥ 5 synapses in **both** datasets (MaleCNS is already ≥ 5; this removes BANC's 3–4-synapse pairs). Then keep only the neuropil rows of surviving pairs. Log how many BANC pairs and synapses are removed.
2. **Measure the remaining gap** on the *training* 1:1 pairs only (to avoid leaking test labels into a preprocessing decision): ratio of MaleCNS to BANC in-degree, out-degree, input synapses, output synapses. Report medians.
3. **Optional quantile-matched threshold (decide in week 2).** The BANC export has ~3.8× fewer synapses than MaleCNS (despite keeping weaker pairs), so "5 synapses" does not mean the same thing in both. Variant: choose a BANC threshold `t_B ∈ {3, 4, 5}` and MaleCNS threshold `t_M ∈ {5, 8, 10, 15}` that make the median degree ratio on training pairs closest to 1. Keep the simple `5/5` as the primary setting and run the matched setting as an ablation — do not tune it on validation metrics.
4. **Normalise weights to fractions.** For each edge, `w_in = syn / input_total(post)` and `w_out = syn / output_total(pre)`. Fractions remove most of the global scale difference. These are the GNN edge weights.
5. **Scalar size features are rank-normalised within dataset** (quantile → [0, 1]). Never feed raw synapse counts or raw degrees to the model.

**Deliverable:** a one-page "harmonisation report" figure — before/after histograms of degree and synapse counts for training 1:1 pairs. It goes straight into the thesis.

#### 7.1.5 Node features (Week 2–3) [1.2]

All features are computed per dataset from harmonised tables; every block is listed in `feature_spec.json`.

| Block | Dims (approx.) | Definition | Notes |
|---|---|---|---|
| **Neuropil input profile** | K ≈ 90 | Fraction of the neuron's input synapses in each canonical neuropil, then √ | The baseline that already gives R@1 = 29% |
| **Neuropil output profile** | K ≈ 90 | Same for outputs, then √ | — |
| **Region summary** | 2 × 6 | Fraction of in/out synapses per region (OL, CB, GNG, VNC, nerve, unassigned) | Robust coarse view |
| **Size (rank-normalised)** | 6 | in-degree, out-degree, in-synapses, out-synapses, # neuropils innervated (in/out) | Within-dataset quantiles |
| **NT** | 9 + 1 | One-hot of harmonised NT + confidence | Missing → `unknown` |
| **Side** | 3 | left / right / center | Also used as a hard filter |
| **Super class** | ~12 | One-hot of harmonised super class | Also used as a hard filter |
| **Hemilineage** | ≤ 242 | One-hot of the 242 hemilineages shared by both datasets; `none` otherwise | Strong developmental signal; **ablate it** (it is an annotation) |
| **Label-projected connectivity** *(seed-dependent)* | 2 × S | Fraction of inputs from / outputs to each **seed** type (Section 7.2.3) | Only seed types from the training split; rebuilt per seed fraction |
| **Morphology** *(optional, 7.1.7)* | ~10 | cable length, # branch points, max Strahler order, soma position (x, y, z in common space) | Only if Phase 1.4 succeeds |

```python
# src/flymatch/features.py — neuropil profiles
import numpy as np, pandas as pd

def neuropil_profiles(edges_np, n_nodes, neuropils):
    """edges_np: pre_uid, post_uid, neuropil (canonical), syn — one dataset."""
    col = (edges_np["neuropil"].map({n: i for i, n in enumerate(neuropils)})
           .fillna(-1).astype(int).to_numpy())
    keep = col >= 0                                   # drop excluded neuropils
    e, col = edges_np[keep], col[keep]
    out = np.zeros((n_nodes, len(neuropils)), np.float32)
    inn = np.zeros((n_nodes, len(neuropils)), np.float32)
    np.add.at(out, (e["pre_uid"].to_numpy(), col), e["syn"].to_numpy())
    np.add.at(inn, (e["post_uid"].to_numpy(), col), e["syn"].to_numpy())
    norm = lambda a: a / np.clip(a.sum(1, keepdims=True), 1, None)
    return np.sqrt(norm(inn)), np.sqrt(norm(out))     # Hellinger-style
```

**Mirror-aware features.** Left and right neuropils are separate columns (`AL_L`, `AL_R`). Keep them separate (side carries information), and also store a **mirror permutation** of the columns (`_L ↔ _R`) — it is needed for the bilateral control (C3) and for neurons with `center` side.

**Done when:** `features_mcns.npy` and `features_banc.npy` exist with identical column order, zero NaNs, and the baseline B1 on these features reproduces the 21 Oct result (R@1 ≈ 29%, within a couple of points — differences come from the new thresholds).

#### 7.1.6 Ground truth (Week 3) [1.3]

1. **Harmonised type `type_h`** for each MaleCNS neuron, in priority order: `type_primary` if it exists as a BANC type → else `flywire_type` if it exists in BANC → else `manc_type` if it exists in BANC → else none. BANC: `type_h = type_primary`. Record `source` for every pair.
2. **Validate the fallbacks** before trusting them. For each fallback-derived type, compare MaleCNS vs BANC neurons on: harmonised super class (must match), majority NT (should match), and median neuropil-profile cosine (should be within the distribution of primary-derived pairs). Rule: **if more than 10% of fallback types fail super-class agreement, drop the fallback source entirely.** Report the agreement table in the thesis.
3. **Type table** `gt_types.parquet`: types present in both datasets (after inclusion), with counts and a size bucket: `1`, `2`, `3–10`, `11–100`, `>100` neurons per dataset.
   For reference, raw per-dataset type sizes (before harmonisation): singleton types 312 (MaleCNS) vs 1,404 (BANC); two-neuron types — mostly one per side — 5,353 vs 3,988; 3–10 neurons 4,997 vs 5,028; 11–100 neurons 879 vs 999; more than 100 neurons 182 vs 127. The largest are photoreceptor and optic-lobe columnar types (`R1-R6` 3,377 and `Tm3` 2,054 in MaleCNS; `KCab` 1,810 and `L5` 1,645 in BANC).
4. **1:1 pairs** `gt_pairs_1to1.parquet`: (type_h, side) groups with exactly one included neuron in each dataset. Expect ~6,000–6,800 after inclusion rules.
5. **Clean subset flag:** MaleCNS `statusLabel == "Reviewed"` (54,062 neurons) and — if `banc_888_meta.feather` provides it — BANC type assignments marked as reviewed rather than automated. Every headline metric is also reported on the clean subset.
6. **No-counterpart candidates (for T3, evaluation only):**
   - *Simulation set* — for each query in a held-out type, remove all its true partners from the gallery; the correct answer becomes `∅`. Cheap, clean, unlimited.
   - *Biological set* — MaleCNS neurons of types reported as male-specific in the MaleCNS paper (~262 sex-specific types). Source the list from the paper's supplementary tables, or from a `dimorphism`/`fru`/`dsx` field if the MaleCNS body-annotation feather or the BANC tutorial's `malecns_09` metadata contains one. **Do not** use "type missing from BANC" as a proxy — most of those are naming differences.

#### 7.1.7 Optional morphology add-on (Weeks 2–4, only if ahead of schedule) [1.2]

Skip this entirely if Phase 1 is behind. Nothing downstream depends on it.

1. Download BANC SWCs (BANC space) and the BANC tutorial bucket's `malecns_banc_space_swc/` (MaleCNS **v0.9** in BANC space).
2. Map v0.9 body IDs to v1.0 by exact ID match; report coverage. Bodies that changed between versions are dropped from morphology features only (set features to NaN + a "missing" indicator).
3. With `navis`: per neuron compute cable length, branch points, max Strahler order, soma position (from BANC `somas_v1.parquet`; for MaleCNS from the body-annotation feather if it has a soma location column).
4. NBLAST: use the published `banc_malecns_v0.9_nblast.feather` rather than recomputing. Inspect its format first (full matrix vs top-k long table).

#### 7.1.8 Type families and the frozen split (Week 4) [1.3]

Splitting by neuron leaks: if `LC10a` neurons sit in both train and test, the model memorises the type. Split by **family**, and make families conservative.

1. **Stem rule:** family stem = regex `^([A-Za-z]+[0-9]*)` on `type_h` → `LC10a`, `LC10b` → `LC10`; `Dm3a` → `Dm3`; `T4a…d` → `T4`.
2. **Confusability links:** union-find over (a) types sharing a stem, (b) types listed together in any neuron's `Alternative Cell Type(s)`, (c) MaleCNS `supertype` groups. Connected components = families.
3. **Stratified group split 70 / 10 / 20** (train / val / test) over families, stratified by dominant harmonised super class × size bucket. Use `sklearn.model_selection.StratifiedGroupKFold` or a simple greedy assignment; check that each stratum appears in all three splits.
4. **Freeze:**

```python
import hashlib, json
splits = {"version": 1, "seed": 20261026, "train": sorted(tr), "val": sorted(va), "test": sorted(te)}
payload = json.dumps(splits, sort_keys=True).encode()
splits["sha256"] = hashlib.sha256(payload).hexdigest()
json.dump(splits, open("data/splits/splits_v1.json", "w"), indent=1)
```

5. Commit `splits_v1.json`. From now on every run records `splits.sha256`.

**Leakage rules (enforced in code, tested in `tests/test_leakage.py`):**

- Test/val family labels are never used in a loss, never in label-projected connectivity, never to choose hard negatives.
- Hard negatives are drawn only from **training-family** neurons (we know they are negatives because we know their type; using test-family neurons would leak their labels).
- Untyped neurons are never used as negatives (they might be the true partner).
- The graph itself is transductive: all nodes and edges are visible to message passing. Only *labels* are hidden. State this explicitly in the thesis — it is the standard setting and the honest one.

**Phase 1 done when:** `python -m flymatch.run data` regenerates everything from raw in one go; `pytest` passes; a notebook loads the split and prints counts per split × super class × size bucket; the harmonisation report and GT agreement tables exist.

**Pitfalls.** Spending week 3 perfecting the VNC neuropil map. Coarse is fine. Tag it as a known limitation and move on.

---

### PHASE 2 — Baselines and evaluation harness (Weeks 4–6) [2.1, 2.2]

**Goal:** numbers to beat, and one evaluation function that every later experiment calls.

#### 7.2.1 Retrieval and ranking utilities

Exact search is cheap at this scale. Do not reach for approximate indexes until the web tool.

```python
# src/flymatch/retrieval.py
import torch, torch.nn.functional as F

@torch.no_grad()
def topk_and_rank(Q, G, true_idx=None, k=100, block=1024):
    """Q: [nq, d] queries, G: [ng, d] gallery (other dataset).
    Returns top-k indices/scores and, if true_idx given, the 1-based rank of the true partner."""
    Q, G = F.normalize(Q, dim=1), F.normalize(G, dim=1)
    idx, val, ranks = [], [], []
    for s in range(0, Q.shape[0], block):
        S = Q[s:s + block] @ G.T                        # [b, ng]
        v, i = S.topk(k, dim=1)
        idx.append(i); val.append(v)
        if true_idx is not None:
            t = true_idx[s:s + block]
            st = S.gather(1, t[:, None])                # score of the true partner
            ranks.append(1 + (S > st).sum(1))
    out = (torch.cat(idx), torch.cat(val))
    return out + ((torch.cat(ranks),) if true_idx is not None else ())
```

On a laptop CPU use `block=512` (a 512 × 158k float32 block is ~320 MB). On a GPU, `block=8192`.

**Filtered retrieval.** Also report every metric with hard filters applied: same harmonised super class, same side (or `center`), compatible NT (equal, or either unknown). Filters are a legitimate part of every method, including baselines.

#### 7.2.2 The evaluation harness (`evaluate.py`)

One function, `evaluate(emb_mcns, emb_banc, split="val", subset="all")`, returns a flat dict: T1 Recall@{1,5,10,50,100}, MRR, median rank (both directions, and their mean); T2 type-accuracy@1 and @5, macro-averaged over super classes; per-super-class and per-size-bucket breakdowns; the same on the clean subset. Every method — baseline or model — produces embeddings or a score function and goes through this harness. Write it once, test it on synthetic data where the answer is known (identical embeddings → R@1 = 100%; random → ≈ 1/N).

#### 7.2.3 Baselines

| ID | Method | Seeds used? | Purpose |
|---|---|---|---|
| **B0** | Random within hard filters | No | Sanity floor |
| **B1** | Cosine of √ neuropil in/out profiles (+ region summary) | No | Already ≈ 29% R@1; the number to beat |
| **B2** | NBLAST (published BANC ↔ MaleCNS v0.9 matrix) | No | Morphology-only reference; report coverage |
| **B2+** | B1 and B2 combined (z-scored sum) | No | Is morphology complementary? |
| **B3** | **cocoa-style label-projected connectivity** | **Yes** | The current-practice ceiling at 100% seeds |
| **B4** | MLP on node features (= GNN with 0 layers), trained with the same loss | Yes (training pairs) | Isolates the value of message passing |

**B3 implementation** (cocoa's idea, reimplemented because cocoa does not support BANC):

```python
# src/flymatch/baselines/cocoa_style.py
import numpy as np, scipy.sparse as sp

def label_projected(edges, n_nodes, seed_type_of_uid, seed_types):
    """Fraction of each neuron's inputs from / outputs to each SEED type.
    seed_type_of_uid: array[n_nodes] of type index into seed_types, or -1."""
    T = len(seed_types)
    W = sp.csr_matrix((edges.syn, (edges.pre_uid, edges.post_uid)), shape=(n_nodes, n_nodes))
    lab = seed_type_of_uid >= 0
    Y = sp.csr_matrix((np.ones(lab.sum()), (np.where(lab)[0], seed_type_of_uid[lab])),
                      shape=(n_nodes, T))
    out_to_type = W @ Y                    # [n, T] synapses onto each seed type
    in_from_type = W.T @ Y                 # [n, T] synapses from each seed type
    # Divide by the neuron's TOTAL synapses, not by its seed-typed synapses: untyped
    # partners count as missing mass. At low seed fractions the vectors become sparse —
    # which is exactly the degradation the seed ablation is meant to measure.
    out_tot = np.asarray(W.sum(axis=1)).ravel()
    in_tot = np.asarray(W.sum(axis=0)).ravel()
    scale = lambda A, tot: sp.diags(1 / np.maximum(tot, 1)) @ A
    return sp.hstack([scale(in_from_type, in_tot), scale(out_to_type, out_tot)]).tocsr()
```

Tested on synthetic data. Seed types must be the **same vocabulary in both datasets** (that is the whole point — the types are the shared coordinate system). Compare neurons by cosine of these vectors. At seed fraction `s`, only `s` of the training families' types are in `seed_types`. At 0%, B3 is undefined → plotted as chance.

#### 7.2.4 Candidate generation and Recall@k (Week 5) [2.1]

The GNN does full exact retrieval, so there is no hard shortlist ceiling at inference. The shortlist still matters for **hard-negative mining** and for B1/B2/B3 comparisons.

1. Shortlist = top-k by B1 within hard filters.
2. Report Recall@k for k ∈ {10, 25, 50, 100, 200} on training 1:1 pairs, unfiltered and filtered.
3. Pick `k_neg` (hard-negative pool size) so that filtered Recall@k_neg ≥ 85% — expect k_neg ≈ 100–200 (unfiltered R@100 was 64%; filters should raise it substantially).

**Phase 2 done when:** a script produces `results/baselines.csv` with B0–B4 on validation (3 seeds where stochastic), plus a Recall@k curve figure. The numbers are what the model must beat; write them on a sticky note.

---

### PHASE 3 — The model (Weeks 6–10) [3.1, 3.2, 3.3]

**Goal:** a shared-weight GNN that beats B1 and B4 on validation by more than the across-seed standard deviation.

Build in this order and **check validation after each step**. If a step does not help, record that and move on — a negative ablation is a result.

```
3a  MLP on node features + contrastive loss      (= B4; proves the loss and training loop work)
3b  + directed message passing (1 → 2 → 3 layers)
3c  + hard negatives from the shortlist
3d  + domain alignment (CORAL first; adversarial only if CORAL is stable and time allows)
3e  + abstention head and assignment
```

#### 7.3.1 Graph objects (Week 6) [3.1]

Per dataset, build two sparse matrices from the harmonised pair-level edges:

- `A_in[j, i] = w_in(i→j)` — row *j* averages over the neuron's **inputs**
- `A_out[i, j] = w_out(i→j)` — row *i* averages over the neuron's **outputs**

Direction matters: "receives from X, sends to Y" is the identity signature of a neuron. Undirected aggregation throws half of it away (test this as an ablation).

```python
# src/flymatch/graph.py
import torch

def sparse_adj(rows, cols, vals, n):
    A = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (n, n))
    return A.coalesce()

def build(edges, n):
    pre  = torch.as_tensor(edges.pre_uid.values,  dtype=torch.long)
    post = torch.as_tensor(edges.post_uid.values, dtype=torch.long)
    A_in  = sparse_adj(post, pre,  torch.as_tensor(edges.w_in.values,  dtype=torch.float32), n)
    A_out = sparse_adj(pre,  post, torch.as_tensor(edges.w_out.values, dtype=torch.float32), n)
    return A_in, A_out
```

**Memory warning.** Do not feed `edge_index` into a PyG layer that builds one message per edge: for MaleCNS that is ~6 M edges × 256 dims × 4 bytes ≈ 6 GB *per layer*, and training will run out of memory on a 16 GB GPU. Sparse–dense matrix multiplication (`torch.sparse.mm`) needs only the node activations. Expected footprint for both graphs, 3 layers, width 256: roughly 4–6 GB. If `torch.sparse.mm` backward fails on CSR for your PyTorch version, keep the matrices in COO (as above).

#### 7.3.2 Encoder (Week 6–7) [3.1]

```python
# src/flymatch/models/encoder.py
import torch, torch.nn as nn, torch.nn.functional as F

class DirLayer(nn.Module):
    """h_i ← h_i + GELU(LN(W_s h_i + W_in · mean_inputs(h) + W_out · mean_outputs(h)))"""
    def __init__(self, d, p=0.1):
        super().__init__()
        self.lin_self = nn.Linear(d, d)
        self.lin_in   = nn.Linear(d, d, bias=False)
        self.lin_out  = nn.Linear(d, d, bias=False)
        self.norm, self.drop = nn.LayerNorm(d), nn.Dropout(p)

    def forward(self, h, A_in, A_out):
        m = (self.lin_self(h)
             + self.lin_in(torch.sparse.mm(A_in, h))
             + self.lin_out(torch.sparse.mm(A_out, h)))
        return h + self.drop(F.gelu(self.norm(m)))

class Encoder(nn.Module):
    """layers = 0 gives the MLP baseline B4 with identical everything else."""
    def __init__(self, f_in, d=256, out=128, layers=3, p=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f_in, d), nn.GELU(), nn.LayerNorm(d))
        self.layers = nn.ModuleList([DirLayer(d, p) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(d * (layers + 1), d), nn.GELU(), nn.Linear(d, out))

    def forward(self, x, A_in, A_out):
        h = self.inp(x); hs = [h]
        for layer in self.layers:
            h = layer(h, A_in, A_out); hs.append(h)
        return F.normalize(self.head(torch.cat(hs, dim=1)), dim=1)   # jumping-knowledge concat
```

The **same** `Encoder` instance embeds both datasets. That is what forces a shared space.

Starting hyper-parameters: width 256, output 128, 2 layers (try 1/2/3), dropout 0.1, AdamW lr 1e-3, weight decay 1e-4, 100-step warm-up then cosine decay, max 3,000 steps, evaluate on validation every 100 steps, keep the best checkpoint by validation MRR, stop after 10 evaluations without improvement. Start in fp32; mixed precision is not worth the debugging.

#### 7.3.3 Contrastive training (Weeks 8–9) [3.2]

**Positives.** Many types have several neurons per side, so use a **multi-positive** InfoNCE: for anchor *i* in MaleCNS, the positive set P(i) is every included BANC neuron with the same `type_h` **and** side. Symmetric loss in both directions.

```python
# src/flymatch/models/losses.py
def multi_pos_infonce(za, zb, pos, log_tau):
    """za: [na, d], zb: [nb, d] (unit vectors); pos: bool [na, nb]; rows with no positive are ignored."""
    tau = log_tau.exp().clamp(0.01, 1.0)
    logits = za @ zb.T / tau
    lp_ab = logits - logits.logsumexp(dim=1, keepdim=True)
    lp_ba = logits.T - logits.T.logsumexp(dim=1, keepdim=True)
    def side(lp, P):
        n = P.sum(1)
        return (-(lp * P).sum(1)[n > 0] / n[n > 0]).mean()
    return 0.5 * (side(lp_ab, pos.float()) + side(lp_ba, pos.T.float()))
```

**One training step.**

1. Forward **both full graphs** through the encoder (all ~325k nodes; ~0.3–0.6 s on a T4).
2. Sample a batch of 256 **training** types from the current seed set (Section 7.4.1); for each, up to 4 neurons per dataset per side → anchors and positives.
3. Add **hard negatives**: for each anchor, 8 neurons sampled from its B1 shortlist that are (a) training-family and (b) a different type. Add the same number of random training-family neurons as easy negatives.
4. Loss = InfoNCE (+ alignment term, 7.3.4) (+ bilateral term, 7.3.5).
5. Learnable temperature, initial τ = 0.07.

**Sampling balance.** Optic-lobe columnar types will dominate if you sample neurons uniformly. Sample **types** uniformly, then neurons within a type — this is why the batch is defined over types.

#### 7.3.4 Domain alignment (Weeks 9–10) [3.3]

Run as an ablation arm, not a core dependency.

1. **CORAL first** (stable, no extra network): match the covariance of embeddings between datasets, `L = ‖Cov(Z_M) − Cov(Z_B)‖²_F / (4d²)`, computed on the batch.
2. **Adversarial (DANN), only if time allows:** a discriminator MLP (128 → 64 → 1) predicts the dataset from `z`; a gradient-reversal layer flips its gradient into the encoder. Weight schedule `λ_p = λ_max · (2 / (1 + e^(−10p)) − 1)` with *p* = training progress; try `λ_max ∈ {0.03, 0.1, 0.3}`, chosen on **validation retrieval**, never on discriminator accuracy.

```python
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lam * g, None
```

3. **Class-conditional alignment (important).** Apply the alignment loss **only to neurons of shared training types**, not to all neurons. Otherwise the model is pushed to make sex-specific neurons look like something in the other dataset — erasing exactly the differences the abstention head must detect. This resolves the tension between "remove the microscope" and "find the biology".
4. **Diagnostic — linear probe.** After training, fit a logistic regression to predict dataset from frozen `z`, 5-fold, on a **type-balanced** subsample (equal neurons per shared type from each dataset — otherwise the probe can win on composition, e.g. more photoreceptors in one dataset). Report probe accuracy with and without alignment; target ≈ 50%.

**Fallback if alignment destabilises training:** drop it. Rank-normalised features + fractional edge weights already remove most scale artefacts. Report the result as "alignment not needed / not helpful", which is a legitimate finding.

#### 7.3.5 Optional: bilateral pre-training (only if ahead of schedule)

Re-express neuropil profiles as **ipsilateral / contralateral** relative to the neuron's soma side instead of absolute `_L` / `_R`. Left and right copies of a type then have the same features, so within-dataset left↔right pairs become free extra positives that need **no cross-dataset labels**. Two uses:

- An auxiliary loss during training (could lift the 0%-seed end of the curve — report it as using within-dataset type labels, which it does).
- Control C3 (within-dataset left↔right retrieval = upper bound with no domain gap).

Decide in Phase 2 whether ipsi/contra or absolute L/R profiles work better for B1 on validation; use the winner everywhere.

#### 7.3.6 Abstention head (Week 10) [3.3]

A small calibrated classifier on top of the frozen embeddings — not part of the GNN.

**Per-query features:** top-1 score s₁, s₂, margin s₁ − s₂, s₁ − mean(top-10), mutual-nearest-neighbour flag, rank of the query in its top-1's own neighbour list, filtered-vs-unfiltered s₁ difference, super-class one-hot, degree quantile.

**Training data — partner-removal simulation, on validation families:** each query appears twice — once with its true partners in the gallery (label "has match"), once with all same-type neurons removed (label "no match").

**Model:** logistic regression, then isotonic calibration fitted on one half of validation and checked on the other half. Output `p_none`.

**Evaluation on test:** AUPRC and AUROC on the simulation set; expected calibration error (10 bins) and a reliability diagram; on the biological set (male-specific types), the fraction flagged `p_none > 0.5` versus the base rate among neurons of shared types — enrichment ratio with a Fisher exact test.

#### 7.3.7 Assignment (Week 10) [3.3]

Nearest neighbours are greedy: two MaleCNS neurons can claim the same BANC neuron. Fix with a global assignment.

1. Candidate graph: top-20 neighbours in both directions, within hard filters.
2. Split into connected components (`scipy.sparse.csgraph.connected_components`).
3. Components with ≤ 2,000 nodes: `scipy.optimize.linear_sum_assignment` on cost = −similarity, **augmented with one dummy "none" column per row at cost −τ_none**. A neuron is left unmatched when no real partner beats τ_none. τ_none is chosen on validation.
4. Larger components (optic-lobe columns): mutual-best pairs only; evaluate those types at type level (T2) — neuron-level identity is not well defined there.
5. Output `matches.parquet`: `mcns_id, banc_id | null, score, p_none, rank_in_both_directions`.

**Phase 3 done when:** on validation, the best GNN beats B1 and B4 on T1 Recall@1 and T2 type-accuracy by more than the across-seed standard deviation, the abstention head is calibrated (ECE < 0.05 on the held-out half), and `matches.parquet` is produced by one command.

**If the GNN does not beat B4 by week 9:** stop architecture work. Keep the best of B4/GNN, spend the remaining time on the seed ablation and controls. "Message passing did not help beyond node features, here is why" is publishable; an unfinished ablation grid is not.

---

### PHASE 4 — Experiments and controls (Weeks 10–13) [4.1, 4.2]

**Goal:** the results chapter. Everything here runs from configs; nothing is run by hand twice.

**Before starting:** freeze the model configuration chosen on validation (Gate G4). From here on, no architecture changes.

#### 7.4.1 The seed-ablation curve (Weeks 10–12) [4.1]

1. **Seed sets.** For each draw `r ∈ {0, 1, 2}`, shuffle the *training* families (stratified by super class) and take nested prefixes: 5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100%. Nested sets make the curve smooth and comparable. Validation and test never change.
2. **What "seed fraction s" controls, per method:**
   - B3 (cocoa-style): the vocabulary of seed types used to build the projected vectors.
   - B4 and GNN: which cross-dataset pairs are positives in the loss; which neurons are eligible hard negatives; which types populate the label-projected feature block (if that block is used).
   - B1, B2: unaffected — drawn as flat lines.
3. **At s = 0%** no cross-dataset positives exist, so B4/GNN are undefined; plot B1 as their value. If the bilateral pre-training variant (7.3.5) exists, plot it as a separate curve, labelled as using within-dataset labels.
4. **Run count:** {B3, B4, GNN} × {5, 10, 25, 50, 100}% × 3 draws = 45 runs, of which 30 are trainings. At 10–20 min per training on a Kaggle T4, that is 5–10 GPU-hours.
5. **Output:** `results/seed_ablation.csv` and the figure: x = seed fraction (log scale), y = T1 Recall@1 (panel a) and T2 type-accuracy (panel b), mean ± sd across draws, with the B1/B2 flat lines.

#### 7.4.2 Controls (Weeks 12–13) [4.2]

| ID | Control | How | What it shows | Expected if the model is honest |
|---|---|---|---|---|
| **C1** | **Hemibrain transfer** | Train on BANC ↔ MaleCNS using hemibrain-restricted features, then evaluate zero-shot on BANC ↔ hemibrain (same sex, different imaging) and MaleCNS ↔ hemibrain (different sex, same FIB-SEM imaging). Report gain over B1 on the same restricted features. | Whether the model learned homology or imaging | Gains over B1 hold on BANC ↔ hemibrain |
| **C2** | Alignment ablation | Same model with no alignment / CORAL / adversarial; report retrieval and the type-balanced linear probe | The alignment term does what it claims | Probe accuracy falls toward 50% without retrieval dropping |
| **C3** | Within-dataset left ↔ right | Same encoder trained and evaluated on bilateral pairs inside BANC alone and inside MaleCNS alone | Upper bound with zero domain gap | Clearly higher than cross-dataset; the difference = cost of the domain gap |
| **C4** | Degree-preserving edge permutation | Permute the `post` column of the pair-level edge table (keeps every neuron's in- and out-degree and the weight multiset); keep node features unchanged | That message passing reads real wiring | GNN collapses to ≈ B4 |
| **C5** | Label permutation | Randomly re-pair training positives | Nothing leaks through the pipeline | Learned gains vanish |

**C1 details (the most persuasive experiment — and the most work).**

- Hemibrain covers roughly a third of the central brain, mostly one hemisphere, and its neurons are truncated at the volume boundary. Restrict neuropil profiles for **all three datasets** to the neuropils fully inside the hemibrain volume and renormalise; otherwise truncation alone separates hemibrain from the others.
- Ground truth: BANC types were matched against hemibrain by the BANC team; MaleCNS carries `hemibrainType` for 7,996 neurons.
- Data: neuPrint `hemibrain:v1.2.1` via `neuprint-python` (needs the free token — request it in week 1), or the BANC tutorial bucket's `compiled_data/hemibrain_121/` in the shared schema.
- If time is short, run C1 on B1 vs GNN only, one seed. It is still the single most informative table in the thesis.

**C4 code:**

```python
def permute_edges(edges, rng):
    e = edges.copy()
    e["post_uid"] = e["post_uid"].to_numpy()[rng.permutation(len(e))]
    e = e[e.pre_uid != e.post_uid]                                   # drop created self-loops
    e = e.groupby(["pre_uid", "post_uid"], as_index=False).syn.sum() # merge created duplicates
    return recompute_fractions(e)                                    # w_in, w_out
```

#### 7.4.3 Ablations (Week 12)

Each on validation, 3 seeds, 100% seeds:

| ID | Change | Question |
|---|---|---|
| A1 | Drop neuropil profiles | How much does position alone carry? |
| A2 | Drop hemilineage | Is an annotation doing the work? |
| A3 | Drop NT | — |
| A4 | Drop size features | Are scale artefacts being used? |
| A5 | Drop label-projected block | Is the GNN just re-learning cocoa? |
| A6 | Undirected aggregation (one matrix, A_in + A_out) | Does direction matter? |
| A7 | Layers 0 / 1 / 2 / 3 | How far does useful context reach? |
| A8 | No hard negatives | — |
| A9 | Thresholds 5/5 vs quantile-matched | Was harmonisation right? |
| A10 | Absolute L/R vs ipsi/contra profiles | — |

#### 7.4.4 Final test evaluation (end of Week 12) — once

1. Freeze configs for: B0–B4, the best GNN, and the GNN variants needed for the controls.
2. Run every method on **test** with 3 training seeds. Save raw per-query ranks (not just aggregates) so any metric can be recomputed without rerunning.
3. Write `results/test_main.csv`. Do not change anything after this point except presentation.

#### 7.4.5 Error analysis (Week 13) [4.2]

1. **Stratify failures** (test 1:1 pairs where the GNN's top-1 is wrong) by super class, type size, NT agreement between true partners, degree quantile, MaleCNS status (`Reviewed` vs `Roughly traced`), side.
2. **Sibling vs unrelated errors:** what fraction of wrong top-1s are in the same type family? Sibling errors are the "nearly right" kind; unrelated ones point to data problems.
3. **Inspect the 30 most confident errors by hand** in FlyWire Codex (it hosts both BANC and MaleCNS). For each, write one line: ground-truth label likely wrong / truncated or fragmented neuron / genuine sex difference / naming difference / model error. This also gives an **estimate of ground-truth noise** — report it.
4. **Figure:** 6–8 confidently wrong matches side by side with their true partners, with the one-line cause.

**Phase 4 done when:** every table and figure in the results chapter regenerates from `python -m flymatch.run report`.

---

### PHASE 5 — The web tool (Weeks 13–15) [5.1, 5.2]

**Goal:** a stranger can paste a neuron ID and understand the answer without help. Keep it small — the hard computation is already done.

#### 7.5.1 Export (Week 13–14) [5.1]

`app/data/` contains only precomputed artefacts:

- `neurons.parquet` — id, dataset, type, super class, side, NT
- `neighbours.parquet` — for every neuron: top-10 in the other dataset with score and rank (≈ 3.3 M rows, ~50 MB)
- `matches.parquet` — the assignment output with `p_none`
- `partners.parquet` — top-20 upstream and downstream partners per neuron, with their types (for the comparison view)
- `MODEL_CARD.md` — model version, split hash, test metrics, known limitations

#### 7.5.2 Backend (Week 14) [5.1]

FastAPI, everything in memory, no GPU, no database server:

| Endpoint | Returns |
|---|---|
| `GET /neuron/{dataset}/{id}` | Neuron info, best match, `p_none`, top-10 alternatives |
| `GET /compare/{dataset}/{id}` | Upstream/downstream partner types of the neuron and its match; shared partners neutral, unique ones highlighted |
| `POST /batch` | CSV of IDs in → CSV of matches out |
| `GET /about` | Method summary, benchmark scores, limitations, citations |

Latency target < 1 s — trivially met by dictionary lookups.

#### 7.5.3 Frontend (Week 14–15) [5.2]

**Streamlit** unless someone on the team already knows React well. Screens: search (ID box, dataset selector, three example IDs), match result (match table + confidence + "no counterpart" state), partner comparison, batch upload, about. For 3-D views, **link out** to the neuron's Codex / neuPrint page instead of rendering meshes.

Deploy on Hugging Face Spaces or Render (free tiers). Write the About page honestly: ground truth is partly model-derived, optic-lobe columnar types are matched at type level only, VNC neuropils were merged coarsely.

**Done when:** a classmate who has not seen the project uses it correctly without instructions.

---

### PHASE 6 — Write-up and release (Weeks 15–16, but start in Week 10) [6.1]

**Start writing the methods chapter in week 10 and the results chapter as each experiment finishes.** Everyone underestimates this phase.

1. **Thesis** (outline in Section 12).
2. **Repository release:** code (MIT licence), README with one-command reproduction, `splits_v1.json` + hash, baseline scores, a **benchmark card** (what the data is, how ground truth was built, known noise, intended use). Derived data CC-BY 4.0 with attribution to BANC and MaleCNS.
3. **Paper draft** (4–8 pages) targeting an ML-for-science or computational-neuroscience workshop.
4. **Demo:** rehearsed 5-minute walk-through of the web tool with three prepared neurons — one easy match, one sibling-type confusion, one confident "no counterpart".

---

## 8. Evaluation protocol

### 8.1 Metric definitions

Let Q be test queries with a known unique partner `b*(q)`, and `r(q)` the 1-based rank of `b*(q)` among all included neurons of the other dataset.

- **Recall@k** = (1/|Q|) Σ 𝟙[r(q) ≤ k], for k ∈ {1, 5, 10, 50, 100}
- **MRR** = (1/|Q|) Σ 1/r(q)
- **Median rank** = median of r(q)
- Reported in both directions (MaleCNS → BANC and BANC → MaleCNS) and their mean.
- **Type-accuracy@k** (T2): fraction of test-family neurons whose top-k neighbours in the other dataset include a neuron of the same `type_h`. **Macro-averaged over super classes**, so the optic lobe cannot dominate. Micro-average reported alongside.
- **Assignment precision / recall / F1:** against test 1:1 pairs, counting abstentions as "no prediction".
- **Abstention:** AUROC, AUPRC, expected calibration error (10 equal-width bins), precision at 50% recall.
- **Domain mixing:** type-balanced linear-probe accuracy (5-fold); mean dataset entropy of each neuron's 10 nearest neighbours in the joint space.

### 8.2 Statistics

- **Training seeds:** 3 per configuration; report mean ± sd.
- **Confidence intervals:** bootstrap over test **families** (not neurons — neurons in a family are correlated), 1,000 resamples, 95% percentile intervals.
- **Paired comparisons** (ours vs a baseline): paired bootstrap over families on the difference; report the difference and its CI.
- **Pre-registered comparisons:** only H1–H5 (Section 4.4) are tested formally. Everything else is described as exploratory.

### 8.3 Reporting breakdowns (every main table)

By super class (optic lobe, central brain, VNC, sensory, ascending/descending, motor) · by type size bucket · clean subset (MaleCNS `Reviewed`, plus BANC reviewed if available) vs all.

### 8.4 The circularity caveat — say it before an examiner does

BANC types were assigned using NBLAST and connectivity matching to FAFB; MaleCNS types were harmonised against FlyWire. Our ground truth is therefore partly produced by methods related to the ones we evaluate, and it reaches across sexes through FAFB names. Mitigations: (1) the clean subset; (2) the hand inspection of confident errors, which estimates label noise; (3) relative comparisons between methods are much less affected than absolute accuracy, because all methods face the same noise.

### 8.5 Figures for the thesis

| # | Figure | Source |
|---|---|---|
| F1 | Pipeline diagram | Hand-drawn |
| F2 | Harmonisation before/after (degree, synapse distributions) | Phase 1 |
| F3 | Ground-truth composition (types × size bucket × super class) | Phase 1 |
| F4 | Recall@k curves, all methods | Phase 2–4 |
| F5 | **Seed-ablation curve** | Phase 4 |
| F6 | UMAP of joint embeddings coloured by dataset and by super class, without vs with alignment | Phase 3–4 |
| F7 | Abstention: PR curve + reliability diagram + enrichment on male-specific types | Phase 3–4 |
| F8 | Per-super-class performance | Phase 4 |
| F9 | Hemibrain transfer 2 × 2 (sex × imaging) | Phase 4 |
| F10 | Error gallery | Phase 4 |

---

## 9. Experiment registry

| Block | Configurations | Seeds | Trainings | Approx. GPU-h |
|---|---|---|---|---|
| E1 Baselines B0–B3 | 4 | 3 (stochastic only) | 0 | 0 (CPU) |
| E2 Main: B4 + GNN | 2 | 3 | 6 | 1.5 |
| E3 Seed ablation | 3 methods × 5 fractions | 3 draws | 30 | 5–10 |
| E4 Ablations A1–A10 | ~12 | 3 | 36 | 6–12 |
| E5 Alignment (C2) | 3 | 3 | 9 | 2–3 |
| E6 Controls C1, C3, C4, C5 | ~6 | 1–3 | ~12 | 2–4 |
| E7 Test evaluation | all frozen | 3 | 0 (inference) | < 1 |
| **Total** | | | **~93** | **~17–31** |

Kaggle's ~30 GPU-hours/week covers this in 1–2 weeks of compute. The binding constraint is your time, not compute.

Every run writes `runs/<date>_<name>_<hash>/` containing `config.yaml`, `git_sha.txt`, `splits_sha256.txt`, `metrics.json`, `ranks.parquet` (per-query ranks), and the checkpoint. `python -m flymatch.run aggregate` builds every results table from these folders.

---

## 10. Risks, triggers and fallbacks

| # | Risk | Likelihood | Trigger (when to act) | Fallback |
|---|---|---|---|---|
| R1 | Phase 1 overruns | High | Split not frozen by end of week 4 | Drop morphology; coarsen VNC neuropils; drop fallback ground-truth sources |
| R2 | Fallback ground truth (flywireType/mancType) is noisy | Medium | > 10% of fallback types disagree on super class | Use primary-type pairs only (5,096) |
| R3 | GNN does not beat the MLP | Medium | No gain on validation by end of week 9 | Keep MLP as the main model; report message passing as not helpful; spend time on seed ablation |
| R4 | GNN barely beats B1 | Medium | Gain < 2 points R@1 on validation | Shift emphasis to the seed-ablation curve and controls; they are the contribution either way |
| R5 | Optic lobe dominates metrics | Certain | — | Macro-averaging and stratified reporting (already in the protocol) |
| R6 | Adversarial training unstable | Medium-high | Loss oscillation or validation drop > 2 points | Use CORAL; or no alignment — and report that |
| R7 | GPU memory | Medium | OOM on full-graph forward | Sparse matmul (default); width 128; or `NeighborLoader` mini-batching |
| R8 | Hemibrain data access delays | Medium | No token by week 4 | Use the BANC tutorial bucket's hemibrain tables; or run C1 with B1 vs GNN only |
| R9 | ID mismatch for MaleCNS v0.9 NBLAST/skeletons vs v1.0 | Medium | Coverage < 80% | Report B2 on the covered subset only |
| R10 | Colab/Kaggle session limits | High | — | Checkpoint every 200 steps; resume logic tested in week 7 |
| R11 | Scope creep into the web tool | Medium | Web work starts before Gate G5 | Web tool is last; a static demo is acceptable |
| R12 | Excel corrupts IDs | Medium | Any ID mismatch in a CSV someone opened | Never open ID files in Excel; share Parquet |

---

## 11. Week-by-week schedule (16 weeks)

Fill in the calendar dates once the semester start is fixed. Gates are go/no-go checkpoints — if a gate is missed, apply the fallback in Section 10 immediately rather than hoping to catch up.

| Week | Gantt | Work | Deliverable |
|---|---|---|---|
| 1 | 1.1 | Repo + environment; ingest with fixed dtypes; vocabulary maps; inclusion rules; **request neuPrint token**; start Dataverse downloads | `neurons.parquet`, `edges_raw.parquet`, maps, contract tests |
| 2 | 1.1–1.2 | Threshold harmonisation; fractional weights; harmonisation report; neuropil profiles | Harmonisation figure (F2) |
| 3 | 1.2–1.3 | Remaining features; ground-truth harmonisation and fallback validation; clean subset; (optional) morphology | `features_*.npy`, `gt_*.parquet`, GT agreement table |
| 4 | 1.3–2.1 | Families; stratified split; **freeze + hash**; reproduce B1 | `splits_v1.json` — **Gate G1** |
| 5 | 2.1 | Evaluation harness + synthetic tests; B0, B1, B2; Recall@k; filters | `baselines.csv` (partial), F4 draft |
| 6 | 2.2–3.1 | B3 cocoa-style; B4 MLP; sparse graph objects; encoder skeleton | `baselines.csv` complete — **Gate G2** |
| 7 | 3.1 | Training loop end-to-end; checkpoint/resume tested; layers 1–3 | First GNN validation numbers |
| 8 | 3.1–3.2 | Multi-positive InfoNCE + type-balanced sampling | GNN vs B4 comparison |
| 9 | 3.2–3.3 | Hard negatives; CORAL | **Gate G3**: GNN vs MLP decision |
| 10 | 3.3–4.1 | Abstention head; assignment; freeze model config; **start writing methods** | `matches.parquet` — **Gate G4** |
| 11 | 4.1 | Seed-ablation runs (Kaggle); hemibrain ingest for C1 | `seed_ablation.csv` (partial) |
| 12 | 4.1–4.2 | Finish ablation; A1–A10; C2–C5; **test evaluation once** | `test_main.csv`, F5 |
| 13 | 4.2–5.1 | C1 hemibrain; error analysis + manual inspection; export artefacts | F9, F10 — **Gate G5** |
| 14 | 5.1–5.2 | FastAPI + Streamlit; results chapter | Working app locally |
| 15 | 5.2–6.1 | Deploy; About page; discussion chapter; paper draft | Public app link |
| 16 | 6.1 | Final thesis; repository release; benchmark card; rehearsed demo | Submission |

**If you are behind at a gate, cut in this order:** morphology add-on → adversarial branch (keep CORAL or nothing) → bilateral pre-training → C1 on more than one seed → ablations A6–A10 → web-tool polish. **Never cut:** the frozen split, the seed-ablation curve, C4, the clean-subset reporting.

---

## 12. Thesis outline (and where each result goes)

1. **Introduction** — connectomes, the correspondence problem, why it matters now (two complete CNS connectomes of different sexes), contributions (the four claims in Section 0).
2. **Background and related work** — connectomics datasets; NBLAST; cocoa and label propagation; graph matching (Pedigo 2023; Matsliah 2026 ACDC); contrastive learning; domain adaptation; reconstruction differences (Yad & Nabila 2026).
3. **Data** — the two datasets (Table: Section 2.2); harmonisation (F2); ground truth and its caveats (F3, Section 8.4); the frozen benchmark.
4. **Methods** — features; encoder; loss; alignment; abstention; assignment (F1).
5. **Experimental setup** — splits, metrics, statistics, baselines, controls (Sections 8–9).
6. **Results** — main table; seed ablation (F5); controls (F9 + table); ablations; abstention (F7); per-super-class (F8).
7. **Error analysis and discussion** — error gallery (F10); estimated label noise; limitations; what the domain gap cost (C3 vs main).
8. **The tool** — architecture, screenshots, usage.
9. **Conclusion and future work** — more datasets, cross-species (as future work only).

---

## 13. Corrections needed in the pre-project deck

| Slide | Currently says | Change to |
|---|---|---|
| 3 Introduction | "FlyWire — female · ~140,000 neurons · TEM · 2024" | "BANC — female brain + nerve cord · ~158,000 neurons (v888) · serial-section TEM · 2026" |
| 4 Literature table | No BANC paper | Add Bates et al. (2026), *Nature* — BANC; typed by NBLAST + connectivity + annealing matching |
| 4 Literature table, row [4] | "VNC connectome alignment, 2026, bioRxiv" | "Matsliah et al., 2026, bioRxiv" |
| 8 Methods | "Baseline 3: cocoa with full seeds" | "Baseline 3: cocoa-style label-projected connectivity (re-implemented; cocoa does not yet support BANC)" |
| 11 Gantt, row 1.2 | "Feature cache and skeleton registration" | "Harmonisation and feature cache" (registration is optional — the BANC team already published co-registered skeletons) |
| 13 References, [10] | "[ADD AUTHORS]. (2026). Uncovering sex differences…" | "Matsliah, A., et al. (2026). Uncovering sex differences in the Drosophila ventral nerve cord through connectome alignment. bioRxiv. https://doi.org/10.64898/2026.06.14.732053" |
| 13 References | — | Add: Bates, A. S., Phelps, J. S., Kim, M., … Lee, W.-C. A. (2026). Distributed control circuits across a brain-and-cord connectome. *Nature*, 656, 957–970. |

The hemibrain control on slides 6 and 8 ("female biology, male-style imaging") stays correct: hemibrain is FIB-SEM like MaleCNS, and female like BANC.

---

## 14. References and data sources

**Datasets**

- Bates, A. S., et al. (2026). Distributed control circuits across a brain-and-cord connectome. *Nature* 656, 957–970. https://www.nature.com/articles/s41586-026-10735-w
- BANC project and data index: https://github.com/htem/BANC-project · Harvard Dataverse: https://doi.org/10.7910/DVN/7WTH1N · data locations: https://github.com/htem/BANC-project/blob/main/manuscript/print/banc_data_locations.md
- MaleCNS v1.0 downloads: https://male-cns.janelia.org/download/ · project page: https://www.janelia.org/project-team/flyem/male-cns-connectome
- Sexual dimorphism in the complete *Drosophila* male CNS connectome. *Cell* (2026). https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6
- FlyWire Codex (hosts BANC and MaleCNS): https://codex.flywire.ai
- Fly connectome data tutorial (shared-schema tables, co-registered skeletons): https://github.com/sjcabs/fly_connectome_data_tutorial
- Overview of adult datasets: https://flyconnecto.me/2026/09/04/the-adult-drosophila-connectome-ecosystem/

**Methods and prior work**

- Matsliah, A., et al. (2026). Uncovering sex differences in the *Drosophila* ventral nerve cord through connectome alignment. bioRxiv. https://doi.org/10.64898/2026.06.14.732053
- Yad, M. A. A., & Nabila, N. T. (2026). Reconstruction differences rival individual differences in the fly compass connectome and break trained circuit models. arXiv:2610.29205. https://arxiv.org/abs/2610.29205
- cocoa: https://github.com/flyconnectome/cocoa · coconatfly: https://natverse.org/coconatfly/
- Dorkenwald, S., et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138.
- Schlegel, P., et al. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634, 139–152.
- Pedigo, B. D., et al. (2023). Bisected graph matching improves automated pairing of bilaterally homologous neurons from connectomes. *Network Neuroscience* 7(2), 522–538.
- Costa, M., et al. (2016). NBLAST: rapid, sensitive comparison of neuronal structure. *Neuron* 91(2), 293–311.
- Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. *ICML*.
- Sun, B., & Saenko, K. (2016). Deep CORAL: correlation alignment for deep domain adaptation. *ECCV Workshops*.
- Khosla, P., et al. (2020). Supervised contrastive learning. *NeurIPS*. (multi-positive contrastive loss)
- Ratnayaka, G., et al. (2023). Contrastive learning for supervised graph matching. *UAI*.
- Nagarajan, A., et al. (2025). Graph embeddings for identifying symmetries in connectomes. bioRxiv.

---

## Appendix A — Neuropil map (starter)

**84 names are identical in both datasets and map to themselves:**

- Paired brain (`_L`/`_R`): AL, AME, AOTU, ATL, AVLP, BU, CAN, CRE, EPA, FLA, GOR, ICL, IPS, LAL, LH, LO, LOP, MB_CA, MB_ML, MB_PED, MB_VL, ME, PLP, PVLP, SCL, SIP, SLP, SMP, SPS, VES, WED — 62
- Unpaired brain: EB, FB, NO, PB, GNG, PRW, SAD — 7
- Paired VNC: LegNp_T1, LegNp_T2, LegNp_T3, NTct_UTct_T1, WTct_UTct_T2, HTct_UTct_T3 — 12
- Unpaired VNC: ANm, IntTct, LTct — 3

**Proposed merges — validate each with the r ≥ 0.5 rule (Section 7.1.2) before using it:**

| Canonical | MaleCNS raw | BANC raw | Confidence | Reasoning |
|---|---|---|---|---|
| `IB` | `IB` | `IB_L`, `IB_R` | High | MaleCNS does not split IB by side |
| `SAD_AMMC` | `SAD` | `SAD`, `AMMC_L`, `AMMC_R` | Medium | MaleCNS has no separate AMMC |
| `LAL_L` / `LAL_R` | `LAL_L` / `LAL_R` | `LAL_*` + `GA_*` | Medium | The gall is a sub-region of the LAL in hemibrain-style nomenclature |
| `FB` | `FB`, `AB_L`, `AB_R` | `FB` | Medium | Asymmetric body lies beside the FB; BANC has no AB |
| `CV` | `CV_UNASGD` | `cervical_connective` | Medium | Both are the neck connective |
| `mVAC_T2_L` / `_R` | `mVAC_T2_*` | `VNC_T2_mVAC_*` | Medium | Same structure, different prefix |
| `VNC_other_L` / `_R` | `mVAC_T1_*`, `mVAC_T3_*`, `Ov_*` | `VNC_T1_ProNm_*`, `VNC_T2_MesoNm_*`, `VNC_T3_MetaNm_*`, `VNC_AMNp_*`, `VNC_HTct_*`, `VNC_NTct_*`, `VNC_WTct_*`, `ABDNM_*` | Low | Margins/sub-regions without a clear 1:1; coarse bucket or exclude |

**Excluded from profiles (still counted in degree and synapse totals):**

- MaleCNS nerves: `AbN1_R`, `AbN2–4_L/R`, `AbNT_L/R`, `ADMN_L/R`, `PDMN_L/R`, `CvN_R`, `DProN_L/R`, `VProN_L/R`, `PrN_L`, `ProAN_L/R`, `ProLN_L/R`, `MesoAN_L/R`, `MesoLN_L/R`, `MetaLN_L/R`, `DMetaN_L/R`
- BANC: `Xnerve`, `Tct` (no side)
- MaleCNS unassigned: `UNASGD`, `CB_UNASGD`, `OL_UNASGD_L/R`, `VNC_UNASGD`
- One dataset only: `LA_L/R` (lamina, MaleCNS)

## Appendix B — Super-class map (complete for the current exports)

| Harmonised | MaleCNS raw (count) | BANC raw (count) |
|---|---|---|
| `optic_lobe_intrinsic` | `ol_intrinsic` (89,403) | `optic_lobe_intrinsic` (72,582) |
| `central_brain_intrinsic` | `cb_intrinsic` (32,164) | `central_brain_intrinsic` (31,782) |
| `ventral_nerve_cord_intrinsic` | `vnc_intrinsic` (13,161) | `ventral_nerve_cord_intrinsic` (12,851) |
| `visual_projection` | `visual_projection` (9,201), `visual_projection_tbc` (2) | `visual_projection` (7,321) |
| `visual_centrifugal` | `visual_centrifugal` (563) | `visual_centrifugal` (472) |
| `sensory` | `vnc_sensory` (6,370), `ol_sensory` (6,098), `cb_sensory` (4,868), `vnc_sensory_tbc` (36), `cb_sensory_tbc` (14) | `sensory` (16,557) |
| `sensory_ascending` | `sensory_ascending` (537), `sensory_ascending_tbc` (2) | `sensory_ascending` (516) |
| `sensory_descending` | `sensory_descending` (12) | `sensory_descending` (13) |
| `ascending` | `ascending_neuron` (1,846) | `ascending` (1,849) |
| `descending` | `descending_neuron` (1,314), `descending_neuron_tbc` (2) | `descending` (1,316) |
| `motor` | `vnc_motor` (708), `cb_motor` (107) | `motor` (805) |
| `visceral_endocrine_efferent` *(low confidence — validate)* | `vnc_efferent` (94), `cb_efferent` (4), `cb_endocrine` (72), `vnc_endocrine` (22), `efferent_ascending` (8), `efferent_descending` (4) | `visceral_circulatory` (221), `ascending_visceral_circulatory` (5) |
| `unknown` (in graph, not in GT) | `vnc_tbc` (38), `ENS` (50) | missing (11,499) |
| *excluded* | — | `glia` (458), `trachea` (8), `not_a_neuron` (7) |

Note the counts already show coverage/annotation differences worth a sentence in the thesis: 89,403 vs 72,582 optic-lobe intrinsic neurons; 9,201 vs 7,321 visual projection neurons. Ascending, descending and motor counts agree to within about 1%.

## Appendix C — Example experiment config

```yaml
# configs/experiments/seed_ablation.yaml
name: seed_ablation
data:
  splits: data/splits/splits_v1.json
  splits_sha256: "<paste the frozen hash>"
  threshold: {mcns: 5, banc: 5}
  min_total_synapses: 20
features:
  blocks: [np_in, np_out, region, size_rank, nt, side, superclass, hemilineage, label_proj]
  profile_mode: absolute          # or ipsi_contra (decided in Phase 2)
model:
  name: gnn                       # gnn | mlp (= gnn with layers: 0)
  layers: 2
  width: 256
  out_dim: 128
  dropout: 0.1
  direction: directed             # directed | undirected (ablation A6)
train:
  max_steps: 3000
  lr: 1.0e-3
  weight_decay: 1.0e-4
  warmup_steps: 100
  batch_types: 256
  per_type_per_side: 4
  hard_negatives: 8
  easy_negatives: 8
  tau_init: 0.07
  align: {method: coral, weight: 0.1, class_conditional: true}   # none | coral | dann
  eval_every: 100
  patience: 10
  checkpoint_every: 200
grid:
  method: [cocoa_style, mlp, gnn]
  seed_fraction: [0.05, 0.10, 0.25, 0.50, 1.00]
  draw: [0, 1, 2]
```

## Appendix D — Command cheat-sheet

```bash
# once
conda env create -f environment.yml
conda activate flymatch
pip install -e .

# Phase 1
python -m flymatch.run data                 # ingest → harmonise → features → ground truth
python -m flymatch.run split --freeze       # writes splits_v1.json + sha256 (refuses if it exists)
pytest -q                                   # data contracts + leakage tests

# Phase 2
python -m flymatch.run baselines --split val

# Phase 3–4
python -m flymatch.run train --config configs/experiments/main.yaml --seed 0
python -m flymatch.run grid  --config configs/experiments/seed_ablation.yaml
python -m flymatch.run aggregate            # all runs → results/*.csv
python -m flymatch.run test --frozen results/frozen_configs/   # ONCE, end of week 12
python -m flymatch.run report               # every table and figure

# Phase 5
python -m flymatch.run export-app
uvicorn app.api:app --reload
streamlit run app/ui.py
```

## Appendix E — Glossary

| Term | Meaning here |
|---|---|
| **Connectome** | The full set of neurons and synaptic connections of a nervous system |
| **Cell type** | A group of neurons judged equivalent in shape, connectivity and neurotransmitter; usually a few copies per side |
| **Homologue / counterpart** | The neuron in the other dataset that is "the same neuron" |
| **Seed** | A known cross-dataset correspondence used as supervision or as a shared coordinate |
| **Type family** | A group of confusable types (shared name stem, alternative labels, supertype); the unit of splitting |
| **1:1 pair** | A type + side with exactly one neuron in each dataset — the only place neuron-level identity is unambiguous |
| **Neuropil** | A named brain/VNC region where synapses sit |
| **Hemilineage** | The developmental lineage a neuron comes from — a strong identity cue |
| **NBLAST** | Morphological similarity score between two neuron skeletons |
| **InfoNCE** | Contrastive loss: pull positives together, push negatives apart, relative to a temperature |
| **CORAL** | Domain alignment by matching embedding covariances |
| **DANN / GRL** | Domain-adversarial training through a gradient-reversal layer |
| **Abstention** | Predicting "no counterpart" |
| **ECE** | Expected calibration error: gap between predicted confidence and observed accuracy |
| **Transductive** | All nodes and edges are visible during training; only test labels are hidden |

---

*Numbers in Sections 2 and 7 were measured on the downloaded files on 26 Oct 2026 or taken from the 21 Oct feasibility check, as marked. Re-run the profiling script after any re-download — Codex data is live and changes.*
