import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flymatch.experiment import load_experiment  # noqa: E402
from flymatch.seed_ablation import table  # noqa: E402
from flymatch.train import DEFAULTS, merge  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / "configs" / "experiments"


def test_gnn_sa_configs_resolve():
    for f, frac in [("05", 0.05), ("10", 0.10), ("25", 0.25), ("50", 0.50)]:
        exp = load_experiment(CFG / f"gnn_sa_{f}.yaml")
        assert exp["method"] == "GNN"
        cfg = merge(DEFAULTS, {k: v for k, v in exp.items() if k in DEFAULTS})
        assert cfg["seed_fraction"] == frac and cfg["draw"] == "seed"
        assert cfg["model"]["layers"] == 2   # same depth as GNN_FULL


def test_table_mean_sd_per_fraction():
    df = pd.DataFrame({"method": ["B3"] * 3 + ["B1"], "seed_fraction": [0.1] * 3 + [0.0], "draw": [0, 1, 2, 0],
                       **{m: [0.1, 0.2, 0.3, 0.5] for m in ["R@1", "R@1_filtered", "MRR", "T2_macro", "T2_macro_filtered"]}})
    t = table(df).set_index("method")
    assert t.loc["B3", "n_draws"] == 3 and abs(t.loc["B3", "R@1"] - 0.2) < 1e-9
    assert abs(t.loc["B3", "R@1_sd"] - 0.1) < 1e-9 and t.loc["B1", "n_draws"] == 1
