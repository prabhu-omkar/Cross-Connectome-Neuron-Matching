"""MLP (B4) vs GNN on abstention + assignment, same `match` settings (k 20, sparse assignment, tau grid).

    python scripts/match_table.py   ->  results/match_compare.csv

Reads results/match_<name>.json (copied from runs/<run>/match_metrics.json; runs/ is not in git).
"""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
RUNS = {"MLP (B4) seed 0": "match_B4_seed0.json", "GNN (gnn_L2) seed 0": "match_gnn_L2_seed0.json"}
KEYS = {"abstention AUROC": "abstain/AUROC", "abstention AUPRC": "abstain/AUPRC", "abstention ECE": "abstain/ECE",
        "female-specific flagged": "abstain/bio/flagged_rate", "flag base rate": "abstain/bio/base_rate",
        "enrichment": "abstain/bio/enrichment", "Fisher p": "abstain/bio/fisher_p",
        "tau_none": "assign/tau_none", "assignment precision": "assign/val/precision",
        "assignment recall": "assign/val/recall", "assignment F1": "assign/val/F1",
        "MaleCNS neurons matched": "assign/n_matched"}
rows = {name: {k: json.loads((R / f).read_text()).get(v) for k, v in KEYS.items()} for name, f in RUNS.items()}
df = pd.DataFrame(rows)
df.to_csv(R / "match_compare.csv")
print(df.to_string())
