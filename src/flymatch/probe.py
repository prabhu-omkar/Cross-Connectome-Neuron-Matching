"""Domain probe (plan.md 7.3.4 step 4; control C2 diagnostic).

Can a linear classifier tell which connectome a neuron's embedding came from? Type-balanced:
for every shared type, the same number of included neurons is taken from each dataset (up to
`per_type`), so the probe cannot win on composition alone. 5-fold CV accuracy; 0.5 = no
dataset signal left in the embedding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def domain_probe(Z: dict, neurons: pd.DataFrame, per_type: int = 5, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    rng = np.random.default_rng(seed)
    inc = neurons[neurons.include & neurons.type_h.notna()]
    by = {ds: inc[inc.dataset == ds].groupby("type_h").uid.apply(np.asarray) for ds in ("mcns", "banc")}
    shared = sorted(set(by["mcns"].index) & set(by["banc"].index))
    X, y = [], []
    for t in shared:
        k = min(per_type, len(by["mcns"][t]), len(by["banc"][t]))
        for lab, ds in enumerate(("mcns", "banc")):
            u = rng.choice(by[ds][t], k, replace=False)
            X.append(np.asarray(Z[ds][u], np.float32))
            y.append(np.full(k, lab))
    X, y = np.concatenate(X), np.concatenate(y)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    acc = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="accuracy")
    return {"probe/domain_acc": float(acc.mean()), "probe/domain_acc_sd": float(acc.std()),
            "probe/n": int(len(y)), "probe/n_types": len(shared)}
