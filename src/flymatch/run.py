"""Config-driven entry point.

    python -m flymatch.run data              # ingest -> harmonise -> features -> ground truth
    python -m flymatch.run split [--freeze]  # family split; --freeze writes splits_v1.json once
    python -m flymatch.run report            # Phase 1 reports (training pairs only)
    python -m flymatch.run b1                # B1 baseline on train + val
    python -m flymatch.run phase1            # all of the above (split must already be frozen, or pass --freeze)
    python -m flymatch.run baselines [--methods B0,B1,...]   # Phase 2: B0-B4 on val -> runs/, results/baselines.csv
    python -m flymatch.run aggregate         # runs/ -> results/baselines.csv
"""
from __future__ import annotations

import argparse
import time

from .config import load_data_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flymatch.run")
    ap.add_argument("command", choices=["data", "split", "report", "b1", "phase1", "baselines", "aggregate"])
    ap.add_argument("--config", default=None, help="data config (default configs/data.yaml)")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--methods", default=None, help="baselines: comma list of B0,B1,B1+region,B2,B2+,B3,B4")
    a = ap.parse_args(argv)
    cfg = load_data_config(a.config)
    t0 = time.time()
    if a.command in ("data", "phase1"):
        from . import features, groundtruth, harmonise, ingest
        ingest.run(cfg)
        harmonise.run(cfg)
        features.run(cfg)
        groundtruth.run(cfg)
    if a.command in ("split", "phase1"):
        from . import splits
        splits.run(cfg, freeze=a.freeze)
    if a.command in ("report", "phase1"):
        from . import report
        report.run(cfg)
    if a.command in ("b1", "phase1"):
        from .baselines import profile_cosine
        profile_cosine.run(cfg)
    if a.command == "baselines":
        from .baselines import run_all
        run_all.run(cfg, a.methods.split(",") if a.methods else None)
    if a.command == "aggregate":
        from .config import Layout
        from .runs import aggregate
        from .splits import load_splits
        layout = Layout(cfg)
        print(aggregate(load_splits(layout)["sha256"], layout.results / "baselines.csv").to_string())
    print(f"[run] {a.command} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
