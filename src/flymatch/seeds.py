"""Seed sets for the seed-ablation curve (plan.md 7.4.1).

For draw r, training families are shuffled within each super class and nested prefixes are
taken, so 5% ⊂ 10% ⊂ 25% ⊂ 50% ⊂ 100%. Only TRAINING families can ever be seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.00)


def seed_families(gt_types: pd.DataFrame, train_families: list[str], fraction: float, draw: int = 0,
                  base_seed: int = 20261026) -> list[str]:
    if fraction >= 1.0:
        return sorted(train_families)
    fam = gt_types[gt_types.family.isin(train_families)].groupby("family").superclass \
        .agg(lambda s: s.value_counts().sort_index().idxmax())
    rng = np.random.default_rng([base_seed, draw])
    chosen = []
    for _, g in fam.groupby(fam):
        names = sorted(g.index)
        order = rng.permutation(len(names))
        k = int(round(fraction * len(names)))
        chosen += [names[i] for i in order[:k]]
    return sorted(chosen)


def seed_types(gt_types: pd.DataFrame, families: list[str]) -> list[str]:
    return sorted(gt_types.loc[gt_types.family.isin(families), "type_h"])
