import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location("c3_mirror", REPO / "scripts" / "c3_mirror.py")
c3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c3)


def test_ranks_ties_and_filter():
    Zq = np.array([[1.0, 0.0]])
    Zg = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    Zg = Zg / np.linalg.norm(Zg, axis=1, keepdims=True)
    gal = np.array([0, 1, 2, 3])
    # partner 0 ties with gallery 1 -> rank 1.5; filter keeps only class "a" (0, 2, 3) -> rank 1
    u, f = c3.ranks(Zq, np.array([0]), Zg, gal, np.array([0]),
                    gal_sc=np.array(["a", "b", "a", "a"]), q_sc=np.array(["a"]))
    assert u[0] == 1.5 and f[0] == 1.0
    # partner 3 is beaten by 0 and 1 -> rank 3 unfiltered, 2 filtered
    u, f = c3.ranks(Zq, np.array([0]), Zg, gal, np.array([3]),
                    gal_sc=np.array(["a", "b", "a", "a"]), q_sc=np.array(["a"]))
    assert u[0] == 3.0 and f[0] == 2.0


def test_mirror_perm_is_an_involution():
    p = REPO / "data" / "processed" / "feature_spec.json"
    if not p.exists():
        return
    perm = np.asarray(json.loads(p.read_text())["mirror_perm"])
    assert (perm[perm] == np.arange(len(perm))).all()       # mirroring twice = identity
    assert sorted(perm) == list(range(len(perm)))            # a permutation
