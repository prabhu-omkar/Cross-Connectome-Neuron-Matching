import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flymatch.hemibrain import pairs_1to1, side_of  # noqa: E402


def test_side_of_instance_suffix():
    assert side_of("MBON01(y5B'2a)_R") == "right"
    assert side_of("PEN_a(PEN1)_L") == "left"
    assert side_of("EPG") == "unknown" and side_of(None) == "unknown"


def test_pairs_only_unique_on_both_sides():
    a = pd.DataFrame({"type_h": ["X", "X", "Y", "Z", "Z"], "side": ["left", "right", "left", "left", "left"],
                      "uid": [0, 1, 2, 3, 4]})
    b = pd.DataFrame({"type_h": ["X", "X", "Y", "Y", "Z"], "side": ["left", "right", "left", "left", "left"],
                      "uid": [10, 11, 12, 13, 14]})
    p = pairs_1to1(a, b).set_index(["type_h", "side"])
    # X left/right unique on both sides -> 2 pairs; Y has two in b, Z two in a -> excluded
    assert sorted(p.index) == [("X", "left"), ("X", "right")]
    assert p.loc[("X", "right"), "uid_a"] == 1 and p.loc[("X", "right"), "uid_b"] == 11
