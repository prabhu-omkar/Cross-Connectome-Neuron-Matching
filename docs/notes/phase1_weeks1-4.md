# Phase 1 (weeks 1–4) — findings and decisions

Every number here comes from a script output in `results/` or `data/interim/log_*.json`
(regenerate with `python -m flymatch.run phase1`). Frozen split sha256:
`260be072aa62135be46e33ceb02e859d39428107a5025bcf7ee38a9843b60499`.

## Ingest (week 1)
- MaleCNS 166,700 neurons, 9,039,552 edge rows (56 self-loop rows dropped); BANC 158,262 neurons,
  3,990,039 rows. No edge endpoints are missing from the neuron tables.
- IDs are read as text and converted to int64; the round-trip is checked against the raw CSV in the tests.
- Ingest fails loudly on any super class, NT or neuropil value missing from `maps/`.

## Neuropil map — differs from the plan's Appendix A
- **BANC puts >99% of its VNC leg-neuropil synapses under `VNC_T{1,2,3}_{Pro,Meso,Meta}Nm_*`**, not
  `LegNp_*` (e.g. BANC `LegNp_T1_L` has 1,444 synapses, `VNC_T1_ProNm_L` 976,400). The plan assumed the core
  VNC neuropils were shared by name. They are now merged:
  `VNC_T*Nm ↔ LegNp_T*`, `VNC_{N,W,H}Tct ↔ {N,W,H}Tct_UTct_T{1,2,3}`, `VNC_AMNp ↔ Ov` (AMNp is the old
  name of the ovoid), `ABDNM_L/R + ANm ↔ ANm`, MaleCNS `mVAC_T1/T3 → LegNp_T1/T3`.
- 89 canonical neuropils at first, all present in both datasets. **Merge validation** on training
  1:1 pairs (Pearson r of per-neuron synapse fractions; rule r ≥ 0.5): 20 of 22 merges passed with
  r between 0.64 and 0.94. `mVAC_T2_L/R` failed (r = 0.27 / 0.22) and is now excluded from profiles,
  which leaves 87. Table: `results/neuropil_merge_validation.csv`.
- 97.0% (MaleCNS) and 99.6% (BANC) of synapses fall in canonical neuropils (target > 90%).

## Harmonisation (week 2) — `harmonisation_thresholds.csv`, `figures/F2_harmonisation.png`
- Re-thresholding BANC to ≥ 5 synapses per pair removes 1,508,776 of 3,037,361 pairs (5.07 M synapses).
- Median MaleCNS/BANC total-degree ratio on training pairs: **1.63 at the export thresholds, 3.03 at 5/5.**
  Equal thresholds *widen* the degree gap, because BANC detects ~4× fewer synapses per connection.
  This is why fractional edge weights and within-dataset rank-normalised size features are essential.
- Quantile-matched setting (grid t_M ∈ {5,8,10,15}, t_B ∈ {3,4,5}): **t_M = 15, t_B = 5 → ratio 1.06.**
  Keep 5/5 as primary; run 15/5 as ablation A9 (`configs/data.yaml` with another `out_dir`).

## Inclusion
- Included: MaleCNS 159,632 of 166,700; BANC 116,553 of 158,262. BANC losses come mainly from the
  11,499 neurons with no super class and from small or fragmentary neurons (< 20 synapses after thresholding).

## Ground truth (week 3) — `gt_fallback_validation_*.csv`
- 9,570 shared types in 3,610 families. **6,765 1:1 pairs** (plan estimate 6,774): 5,056 primary,
  1,567 flywireType, 142 mancType. Inclusion rules remove only 1 pair.
- Fallback validation: super-class agreement is flywire 99.6%, manc 99.5%, primary 99.5%. Neither
  source is dropped. The 10 fallback types that disagree on super class are dropped individually. NT
  agreement is lower for flywire (73%) than for primary (81%), and the profile cosine is similar.
- Clean subset: MaleCNS `Reviewed` 54,062. BANC `proofread` without tracing/LR/alignment/size flags:
  122,177 (from `banc_888_meta.feather`). Only 977 1:1 pairs are clean on both sides, so the clean-subset
  metrics will have wide confidence intervals.
- **Independent check against the BANC team's 2,194 expert-reviewed BANC↔MaleCNS matches:** 1,632
  have a type_h on both sides, and only **53%** share it. The disagreements are almost all *naming* in
  sensory (77%) and motor (16%) neurons: BANC `R7`/`R8` vs MaleCNS `R7y`/`R7p`/`R8y`/`R7_unclear`,
  BANC `EFFabg*` vs MaleCNS `MNad*`, and ORN glomerulus subtypes. Consequences:
  (1) the 1:1 GT barely covers sensory/motor neurons (13 and 91 training pairs); (2) "type missing in
  the other dataset" is a poor proxy for "no counterpart" (as plan.md 2.4 warned); (3) a future type-alias
  table built from these reviewed matches could add positives. Not done yet, to avoid scope creep.
- T3 biological no-counterpart candidates: BANC `sexually_dimorphic == female-specific`, 804 included
  neurons. This replaces the plan's search for a MaleCNS male-specific list.

## Families and split (week 4) — `split_summary.csv`, `figures/F3_ground_truth.png`
- Families: union-find over name stem + names co-listed for a neuron (2,693 links) + MaleCNS supertype
  (3,022 links). Largest families are VNC hemilineage stems (`IN06…` 181 types, `IN19…` 166).
- Split by types: train 6,696 / val 956 / test 1,918 (69.97 / 9.99 / 20.04%). 1:1 pairs: 4,746 / 669 / 1,350.
  Of 42 family strata, 12 small (super class × size) cells are missing from at least one split.
  All are tiny: visceral/efferent, singleton ascending/descending, and similar.

## B1 reproduction — `b1_profile_cosine.csv`, `b1_by_superclass_train.csv`
- MaleCNS→BANC, all-BANC gallery, unfiltered: **R@1 41.7% train / 40.5% val**, R@10 64.8 / 65.8%,
  median rank 3. Shuffled: 0.1%. With filters (same super class and side): 45.2% / 43.3%.
- This is **higher than the 21 Oct number (29.3%)**, so the "reproduce within a couple of points"
  check did not hold. The gain comes from harmonisation: undoing the neuropil merges drops train R@1 to
  34.5%. Ascending neurons go 18→49%, descending 46→64%. The pair set also changed (fallback types added,
  test excluded). B1 uses no labels, so this cannot be leakage. **The number to beat is ≈ 41% R@1 on val.**
- The merges *lower* B1 for VNC-intrinsic (43→33%) and motor (35→22%) neurons. With merges, most VNC
  neurons sit in a single large leg neuropil, so their profiles look alike; the finer tectulum columns
  are more discriminative. This is a weighting problem for the learned model, not a map error.
- Test pairs were not touched.

## Not done / needs you
- **neuPrint token** (hemibrain control C1): needs your Google sign-in at neuprint.janelia.org.
- NBLAST matrix downloaded (`banc_malecns_v0.9_nblast.feather`, long table of ~16 M top-k rows:
  `pt_root_id, match_id, match_cell_type, score, …`); B2 is Phase 2.
- Morphology add-on skipped (optional; nothing depends on it).
