# External downloads

Every file here was downloaded unmodified (CC BY 4.0, see ../../DATA_LICENSE.md). Record version and date for anything added.

## banc_dataverse_v3/ — BANC Harvard Dataverse, doi:10.7910/DVN/7WTH1N, dataset version 3 (released 2026-07-01)

Downloaded 2026-10-26 via `https://dataverse.harvard.edu/api/access/datafile/<id>`.

| File | Dataverse file id | Bytes | Used for |
|---|---|---|---|
| `banc_888_meta.feather` | 14033740 | 57,550,610 | BANC `proofread` / `status` (clean subset), `sexually_dimorphic` (T3 no-counterpart candidates) |
| `banc_malecns_reviewed_matches.csv` | 13994488 | 208,026 | Independent check of harmonised types (2,194 expert-reviewed BANC↔MaleCNS v0.9 matches) |
| `banc_fafb_reviewed_matches.csv` | 13994485 | 6,581,434 | Not used yet |
| `somas_v1.parquet` | 13916460 | 5,350,459 | Optional morphology (soma positions) |
| `banc_malecns_v0.9_nblast.feather` | 13994483 | 912,369,066 | Baseline B2. Too large for git: attached to the GitHub Release; download it into this folder |

Note: Dataverse serves the two `*_reviewed_matches` files as plain CSV (tabular ingest), although
they are listed as `.csv.gz`; they were renamed to `.csv`.

## neuPrint

Token set 2026-10-26 as the Windows user environment variable `NEUPRINT_APPLICATION_CREDENTIALS`
(never in the repo). Verified with neuprint-python 0.6.4 against `hemibrain:v1.2.1` (MBON01 → 2 neurons).
Datasets visible: hemibrain:v1.2.1, male-cns:v0.9, male-cns:v1.0, manc:v1.0/1.2.1/1.2.3,
optic-lobe:v1.0.1/v1.1. Processes started before `setx` don't see it; read it with
`[Environment]::GetEnvironmentVariable('NEUPRINT_APPLICATION_CREDENTIALS','User')`.

## Still to do

- Optional: MaleCNS `body-annotations-male-cns-v1.0-minconf-0.5.feather` (male-cns.janelia.org/download).
