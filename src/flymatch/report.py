"""Phase 1 reports — all measured on TRAINING 1:1 pairs only (never val/test labels).

- Harmonisation report (plan.md 7.1.4): degree / synapse gap before vs after the 5/5
  threshold, quantile-matched threshold search, figure F2.
- Neuropil-merge validation (7.1.2): per-merge Pearson r of synapse fractions across pairs.
- B1 breakdown by super class, and B1 with vs without the neuropil merges.
- Ground-truth composition (F3).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import DATASETS  # noqa: E402
from .config import Layout, load_data_config, write_json  # noqa: E402
from .features import canonical_neuropils, neuropil_profiles  # noqa: E402
from .harmonise import node_stats, pair_totals  # noqa: E402
from .io import read_map  # noqa: E402
from .retrieval import ranks_of_true, recall_metrics  # noqa: E402
from .splits import BUCKET_ORDER, family_to_split, load_splits  # noqa: E402

T_BANC, T_MCNS = (3, 4, 5), (5, 8, 10, 15)


def _train_pairs(layout):
    splits = load_splits(layout)
    pairs = pd.read_parquet(layout.gt_pairs)
    pairs["split"] = pairs.family.map(family_to_split(splits))
    return pairs[pairs.split == "train"].reset_index(drop=True), splits


def harmonisation(layout: Layout, cfg: dict, tr: pd.DataFrame) -> dict:
    er = pd.read_parquet(layout.edges_raw, columns=["dataset", "pre_uid", "post_uid", "syn"])
    neurons = pd.read_parquet(layout.neurons, columns=["dataset", "uid"])
    pt, N = {}, {}
    for ds in DATASETS:
        pt[ds] = pair_totals(er[er.dataset == ds])
        N[ds] = int((neurons.dataset == ds).sum())
    del er

    def stats(ds, t):
        return node_stats(pt[ds][pt[ds].syn >= t], N[ds])

    def ratios(sm, sb):
        m, b = sm.iloc[tr.mcns_uid.to_numpy()], sb.iloc[tr.banc_uid.to_numpy()]
        out = {}
        for c in ["in_deg", "out_deg", "in_syn", "out_syn"]:
            r = m[c].to_numpy() / np.clip(b[c].to_numpy(), 1, None)
            out[f"median_ratio_{c}"] = float(np.median(r))
        tot = (m.in_deg + m.out_deg).to_numpy() / np.clip((b.in_deg + b.out_deg).to_numpy(), 1, None)
        out["median_ratio_total_degree"] = float(np.median(tot))
        return out

    raw = {"mcns": stats("mcns", 1), "banc": stats("banc", 1)}
    rows = [{"setting": "export (MaleCNS>=5, BANC>=3)", "t_mcns": 5, "t_banc": 3, **ratios(raw["mcns"], raw["banc"])}]
    grid = {}
    cache_m = {t: stats("mcns", t) for t in T_MCNS}
    cache_b = {t: stats("banc", t) for t in T_BANC}
    for tm in T_MCNS:
        for tb in T_BANC:
            grid[(tm, tb)] = ratios(cache_m[tm], cache_b[tb])
            rows.append({"setting": f"grid", "t_mcns": tm, "t_banc": tb, **grid[(tm, tb)]})
    tbl = pd.DataFrame(rows)
    primary = (cfg["threshold"]["mcns"], cfg["threshold"]["banc"])
    best = min(grid, key=lambda k: abs(np.log(grid[k]["median_ratio_total_degree"])))
    tbl["primary"] = (tbl.t_mcns == primary[0]) & (tbl.t_banc == primary[1]) & (tbl.setting == "grid")
    tbl["quantile_matched"] = (tbl.t_mcns == best[0]) & (tbl.t_banc == best[1]) & (tbl.setting == "grid")
    tbl.to_csv(layout.results / "harmonisation_thresholds.csv", index=False)

    # Figure F2: training-pair neurons, before (export thresholds) vs after (primary 5/5)
    after = {"mcns": cache_m[primary[0]], "banc": cache_b[primary[1]]}
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
    for row, (label, S) in enumerate([("export thresholds (MaleCNS ≥5, BANC ≥3)", raw), (f"harmonised (both ≥{primary[1]})", after)]):
        m, b = S["mcns"].iloc[tr.mcns_uid.to_numpy()], S["banc"].iloc[tr.banc_uid.to_numpy()]
        for col, (name, fm, fb) in enumerate([
            ("total degree", m.in_deg + m.out_deg, b.in_deg + b.out_deg),
            ("total synapses", m.in_syn + m.out_syn, b.in_syn + b.out_syn),
        ]):
            a = ax[row, col]
            bins = np.linspace(0, np.log10(max(fm.max(), fb.max()) + 1), 50)
            a.hist(np.log10(fm + 1), bins, alpha=.6, label=f"MaleCNS (median {np.median(fm):.0f})", color="#2b6cb0")
            a.hist(np.log10(fb + 1), bins, alpha=.6, label=f"BANC (median {np.median(fb):.0f})", color="#c05621")
            a.set_xlabel(f"log10({name} + 1)"); a.set_ylabel("neurons"); a.legend(fontsize=8)
            a.set_title(f"{name} — {label}", fontsize=9)
        r = (m.in_deg + m.out_deg).to_numpy() / np.clip((b.in_deg + b.out_deg).to_numpy(), 1, None)
        a = ax[row, 2]
        a.hist(np.log2(np.clip(r, 1e-3, None)), 60, color="#4a5568")
        a.axvline(0, color="k", lw=.8); a.set_xlabel("log2(MaleCNS / BANC total degree), per pair")
        a.set_title(f"degree ratio — median {np.median(r):.2f}", fontsize=9)
    fig.suptitle(f"Harmonisation report — {len(tr):,} training 1:1 pairs", fontsize=11)
    fig.tight_layout()
    (layout.results / "figures").mkdir(exist_ok=True)
    fig.savefig(layout.results / "figures" / "F2_harmonisation.png", dpi=130)
    plt.close(fig)
    out = {"primary": {"t_mcns": primary[0], "t_banc": primary[1], **grid[primary]},
           "export": rows[0], "quantile_matched": {"t_mcns": best[0], "t_banc": best[1], **grid[best]}}
    print(f"[report] degree ratio MaleCNS/BANC (median, training pairs): export {rows[0]['median_ratio_total_degree']:.2f}, "
          f"5/5 {grid[primary]['median_ratio_total_degree']:.2f}, matched {best} -> {grid[best]['median_ratio_total_degree']:.2f}")
    return out


def neuropil_merges(layout: Layout, tr: pd.DataFrame) -> pd.DataFrame:
    neuropils, _ = canonical_neuropils(layout)
    m = read_map(layout.maps / "neuropil_map.csv")
    merged = set(m[(m.include.astype(int) == 1) & (m.raw != m.canonical)].canonical)
    enp = pd.read_parquet(layout.edges_np)
    neurons = pd.read_parquet(layout.neurons, columns=["dataset"])
    frac = {}
    for ds in DATASETS:
        n = int((neurons.dataset == ds).sum())
        _, _, rin, rout = neuropil_profiles(enp[enp.dataset == ds], n, neuropils)
        tot = rin + rout
        frac[ds] = tot / np.clip(tot.sum(1, keepdims=True), 1, None)
    fm, fb = frac["mcns"][tr.mcns_uid.to_numpy()], frac["banc"][tr.banc_uid.to_numpy()]
    rows = []
    for j, name in enumerate(neuropils):
        a, b = fm[:, j], fb[:, j]
        nz = (a > 0) | (b > 0)
        r = float(np.corrcoef(a[nz], b[nz])[0, 1]) if nz.sum() > 2 and a[nz].std() > 0 and b[nz].std() > 0 else np.nan
        raws = {ds: sorted(m[(m.dataset == ds) & (m.canonical == name)].raw) for ds in DATASETS}
        rows.append({"canonical": name, "merged": name in merged, "pearson_r": r, "n_pairs_nonzero": int(nz.sum()),
                     "mean_frac_mcns": float(a.mean()), "mean_frac_banc": float(b.mean()),
                     "raw_mcns": " ".join(raws["mcns"]), "raw_banc": " ".join(raws["banc"]),
                     "keep (r>=0.5)": bool(r >= 0.5) if not np.isnan(r) else None})
    df = pd.DataFrame(rows).sort_values(["merged", "pearson_r"], ascending=[False, True])
    df.to_csv(layout.results / "neuropil_merge_validation.csv", index=False)
    mm = df[df.merged]
    print(f"[report] neuropil merges: {int((mm.pearson_r >= 0.5).sum())}/{len(mm)} pass r>=0.5; "
          f"failing: {mm[mm.pearson_r < 0.5][['canonical', 'pearson_r', 'n_pairs_nonzero']].to_dict('records')}")
    return df


def b1_breakdown(layout: Layout, tr: pd.DataFrame) -> pd.DataFrame:
    """B1 (mcns->banc, unfiltered, all-BANC gallery) by super class on training pairs, and
    the same with neuropil merges undone (only identical-name neuropils kept)."""
    spec = json.loads(layout.feature_spec.read_text())
    gt = pd.read_parquet(layout.gt_types)
    X = {ds: np.load(layout.features(ds)) for ds in DATASETS}
    cols = slice(spec["blocks"]["np_in"][0], spec["blocks"]["np_out"][1])
    sc = tr.merge(gt[["type_h", "superclass"]], on="type_h").superclass.to_numpy()
    q, t = tr.mcns_uid.to_numpy(), tr.banc_uid.to_numpy()
    r, _ = ranks_of_true(X["mcns"][q, cols], X["banc"][:, cols], t)
    # "no merges": keep only canonical neuropils whose raw name is identical in both datasets
    m = read_map(layout.maps / "neuropil_map.csv")
    ident = set(m[(m.include.astype(int) == 1) & (m.note == "identical name")].canonical)
    merged_any = set(m[(m.include.astype(int) == 1) & (m.raw != m.canonical)].canonical)
    keep = [i for i, c in enumerate(spec["columns"][cols]) if c.split(":", 1)[1] in ident - merged_any]
    r0, _ = ranks_of_true(X["mcns"][q, cols][:, keep], X["banc"][:, cols][:, keep], t)
    rows = [{"superclass": "ALL", "n": len(r), "R@1": (r <= 1).mean(), "R@10": (r <= 10).mean(),
             "R@1_no_merges": (r0 <= 1).mean(), "R@10_no_merges": (r0 <= 10).mean()}]
    for s in sorted(set(sc)):
        k = sc == s
        rows.append({"superclass": s, "n": int(k.sum()), "R@1": (r[k] <= 1).mean(), "R@10": (r[k] <= 10).mean(),
                     "R@1_no_merges": (r0[k] <= 1).mean(), "R@10_no_merges": (r0[k] <= 10).mean()})
    df = pd.DataFrame(rows)
    df.to_csv(layout.results / "b1_by_superclass_train.csv", index=False)
    print("[report] B1 by super class (training pairs):\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


def gt_composition(layout: Layout, splits: dict):
    gt = pd.read_parquet(layout.gt_types)
    gt["split"] = gt.family.map(family_to_split(splits))
    tab = gt.pivot_table(index="superclass", columns="size_bucket", values="type_h", aggfunc="size", fill_value=0)
    tab = tab[[b for b in BUCKET_ORDER if b in tab.columns]]
    tab.to_csv(layout.results / "gt_types_superclass_x_size.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    tab.plot.barh(stacked=True, ax=ax[0], colormap="viridis")
    ax[0].set_xlabel("shared types"); ax[0].set_title("Ground-truth types by super class × size bucket", fontsize=10)
    sp = gt.pivot_table(index="superclass", columns="split", values="type_h", aggfunc="size", fill_value=0)[["train", "val", "test"]]
    (sp.div(sp.sum(axis=1), axis=0)).plot.barh(stacked=True, ax=ax[1], color=["#2b6cb0", "#d69e2e", "#c53030"])
    ax[1].axvline(.7, color="k", lw=.6, ls="--"); ax[1].axvline(.8, color="k", lw=.6, ls="--")
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax[1].set_xlabel("fraction of types"); ax[1].set_title("Split shares per super class (target 70/10/20)", fontsize=10)
    fig.tight_layout()
    fig.savefig(layout.results / "figures" / "F3_ground_truth.png", dpi=130)
    plt.close(fig)


def run(cfg: dict | None = None):
    cfg = cfg or load_data_config()
    layout = Layout(cfg)
    tr, splits = _train_pairs(layout)
    (layout.results / "figures").mkdir(exist_ok=True)
    out = {"splits_sha256": splits["sha256"], "n_train_pairs": len(tr)}
    out["harmonisation"] = harmonisation(layout, cfg, tr)
    neuropil_merges(layout, tr)
    out["b1_train_by_superclass"] = b1_breakdown(layout, tr).to_dict("records")
    gt_composition(layout, splits)
    write_json(out, layout.results / "phase1_report.json")
    return out


if __name__ == "__main__":
    run()
