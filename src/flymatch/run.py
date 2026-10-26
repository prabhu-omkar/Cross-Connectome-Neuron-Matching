"""Config-driven entry point.

    python -m flymatch.run data              # ingest -> harmonise -> features -> ground truth
    python -m flymatch.run split [--freeze]  # family split; --freeze writes splits_v1.json once
    python -m flymatch.run report            # Phase 1 reports (training pairs only)
    python -m flymatch.run b1                # B1 baseline on train + val
    python -m flymatch.run phase1            # all of the above (split must already be frozen, or pass --freeze)
"""
from __future__ import annotations

import argparse
import time

from .config import load_data_config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flymatch.run")
    ap.add_argument("command", choices=["data", "split", "report", "b1", "phase1"])
    ap.add_argument("--config", default=None, help="data config (default configs/data.yaml)")
    ap.add_argument("--freeze", action="store_true")
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
    print(f"[run] {a.command} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
