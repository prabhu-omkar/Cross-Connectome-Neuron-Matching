import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flymatch.c1 import _restricted  # noqa: E402


def test_restricted_graph_and_features():
    # 3 neurons; synapses in a kept neuropil (A) and in a dropped one (Z)
    enp = pd.DataFrame({"pre_uid": [0, 0, 1, 2], "post_uid": [1, 1, 2, 0],
                        "neuropil": ["A", "Z", "B", "Z"], "syn": [6, 4, 10, 7]})
    edges, X, empty = _restricted(enp, 3, ["A", "B"], pd.Series(["left", "right", "center"]))
    # only synapses in A/B survive: 0->1 (6 in A), 1->2 (10 in B); 2->0 lives only in Z -> dropped
    assert sorted(zip(edges.pre_uid, edges.post_uid, edges.syn)) == [(0, 1, 6), (1, 2, 10)]
    assert X.shape == (3, 2 + 2 + 6 + 3)
    # neuron 0: outputs only in A -> out profile sqrt(1) at A; no inputs in A/B -> zero in-profile
    assert X[0, 2] == 1.0 and X[0, :2].sum() == 0
    # neuron 1: input 6 in A, output 10 in B
    assert X[1, 0] == 1.0 and X[1, 3] == 1.0
    # side one-hot (left, right, center) in the last 3 columns
    assert (X[:, -3:] == np.eye(3)).all()
    assert empty == 0          # every neuron has at least one synapse in A/B
