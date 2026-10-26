"""Split integrity and leakage rules (plan.md 7.1.8)."""
import json

import pandas as pd
import pytest

from flymatch.splits import SPLITS, load_splits, payload_hash


@pytest.fixture(scope="module")
def splits(built):
    if not built.split_file.exists():
        pytest.skip("split not frozen; run `python -m flymatch.run split --freeze`")
    return load_splits(built)


def test_hash_verifies(built, splits):
    assert payload_hash(splits) == splits["sha256"]
    assert (built.splits / "splits_v1.sha256").read_text().strip() == splits["sha256"]


def test_families_disjoint(splits):
    seen = set()
    for s in SPLITS:
        assert not (seen & set(splits[s]))
        seen |= set(splits[s])


def test_every_gt_family_assigned_once(built, splits):
    gt = pd.read_parquet(built.gt_types)
    assigned = set().union(*(set(splits[s]) for s in SPLITS))
    assert set(gt.family) <= assigned


def test_type_in_one_family(built):
    gt = pd.read_parquet(built.gt_types)
    assert gt.groupby("type_h").family.nunique().max() == 1


def test_no_type_crosses_splits_via_pairs(built, splits):
    f2s = {f: s for s in SPLITS for f in splits[s]}
    p = pd.read_parquet(built.gt_pairs)
    assert p.assign(split=p.family.map(f2s)).groupby("type_h").split.nunique().max() == 1


def test_split_fractions_reasonable(built, splits):
    gt = pd.read_parquet(built.gt_types)
    f2s = {f: s for s in SPLITS for f in splits[s]}
    share = gt.family.map(f2s).value_counts(normalize=True)
    assert abs(share["train"] - 0.7) < 0.03 and abs(share["val"] - 0.1) < 0.02 and abs(share["test"] - 0.2) < 0.03


def test_stem_siblings_share_split(built, splits):
    """Leakage guard: types sharing a name stem (LC10a / LC10b) never sit in different splits."""
    from flymatch.groundtruth import stem
    gt = pd.read_parquet(built.gt_types)
    f2s = {f: s for s in SPLITS for f in splits[s]}
    gt["split"] = gt.family.map(f2s)
    gt["stem"] = gt.type_h.map(stem)
    assert gt.groupby("stem").split.nunique().max() == 1
