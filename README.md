# flymatch

Cross-connectome neuron matching between the female **BANC v888** and male **MaleCNS v1.0**
*Drosophila* connectomes. Final-year B.E. project. The full plan is `../plan.md`; this README
covers how to run the code and what Phase 1 (weeks 1–4) produced.

## Setup (Windows laptop, no conda)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
copy configs\paths.example.yaml configs\paths.yaml   # edit if the raw data is elsewhere
```

`environment.yml` is the conda equivalent. The GNN extras (`torch`, `torch_geometric`) are only
needed from Phase 3: `pip install -e .[gnn]`.

## One command

```powershell
.venv\Scripts\python.exe -m flymatch.run phase1     # data -> split check -> reports -> B1  (~8 min)
.venv\Scripts\python.exe -m pytest -q
```

Phase 2 (weeks 5–6):

```powershell
.venv\Scripts\python.exe -m flymatch.run baselines      # B0–B4 on val -> runs/, results/baselines.csv, F4  (~30 min CPU)
.venv\Scripts\python.exe -m flymatch.smoke --layers 2   # full-graph GNN forward/backward check
```

Individual stages: `data` (ingest → harmonise → features → ground truth, ~1.5 min), `split [--freeze]`,
`report`, `b1`. Everything under `data/` (except `data/splits/`) is regenerated from the four raw
files; nothing in the raw folders is ever written.

## Layout

```
configs/data.yaml        thresholds, inclusion rules, GT sources, split seed
maps/                    hand-maintained vocabulary maps (ingest fails on any unmapped value)
src/flymatch/
  io.py                  fixed-dtype readers; IDs text -> int64 with round-trip check
  ingest.py              raw -> neurons_raw / edges_raw (uids, maps applied)
  harmonise.py           equal pair thresholds, fractional weights, inclusion rules
  features.py            neuropil profiles, region summary, rank-normalised size, NT, side, super class, hemilineage
  groundtruth.py         type_h, fallback validation, 1:1 pairs, clean flags, type families
  splits.py              stratified family split, freeze + SHA-256
  retrieval.py           exact blocked cosine retrieval + recall metrics
  report.py              harmonisation report, neuropil-merge validation, GT composition (training pairs only)
  evaluate.py            THE harness: T1/T2, filters, clean subset, breakdowns, family bootstrap CIs
  seeds.py               nested seed-family sets for the seed ablation
  graph.py               sparse A_in / A_out, undirected variant, edge permutation (C4)
  models/                DirLayer, Encoder (layers=0 = B4), InfoNCE / CORAL / GRL
  train.py               contrastive training loop (B4 now, GNN in Phase 3)
  runs.py                run folders + aggregation (refuses other split hashes)
  baselines/             scorers, B1 profile cosine, B2 NBLAST, B3 cocoa-style, run_all driver
data/splits/splits_v1.json      FROZEN split (committed) + splits_v1.sha256
results/                 tables and figures written by scripts
tests/                   unit tests, data contracts, leakage tests
```

## Phase 1 status (26 Oct 2026)

Done: ingest, maps, inclusion, harmonisation, features (462 columns), ground truth (6,765 1:1 pairs,
9,570 shared types), families, **frozen split** (`sha256 260be072…`), B1 reproduced (val R@1 40.5%).
Findings, deviations from the plan and open items: [`results/PHASE1_NOTES.md`](results/PHASE1_NOTES.md).

## Phase 2 status (26 Oct 2026) — Gate G2 passed

`results/baselines.csv` is complete. Validation T1 R@1 (mean of both directions): B1 34.7%, B2+ 39.6%,
B3 39.6% (83.0% filtered), **B4 MLP 61.6 ± 0.7%**. Details and caveats (tie handling, NBLAST coverage):
[`results/PHASE2_NOTES.md`](results/PHASE2_NOTES.md).
