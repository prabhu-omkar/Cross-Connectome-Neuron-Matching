"""Readers with fixed dtypes. IDs are read as text and converted to int64 — never float.

BANC root IDs are 18 digits; as float64 `720575941076956631` becomes `720575941076956672`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

NEURON_RENAME = {
    "Root ID": "neuron_id", "Soma side": "side_raw", "Super Class": "superclass_raw",
    "Class": "class_raw", "Sub Class": "subclass_raw", "Hemilineage": "hemilineage_raw",
    "Predicted NT type": "nt_raw", "Predicted NT confidence": "nt_conf",
    "Verified NT type": "nt_verified", "Primary Cell Type": "type_primary",
    "Alternative Cell Type(s)": "type_alt", "Community labels": "labels_raw",
    "Flow": "flow", "Body Part": "body_part",
}

LABEL_KEYS = ["statusLabel", "flywireType", "mancType", "hemibrainType", "supertype", "mancBodyid"]
LABEL_COLUMNS = {"statusLabel": "status", "flywireType": "flywire_type", "mancType": "manc_type",
                 "hemibrainType": "hemibrain_type", "supertype": "supertype", "mancBodyid": "manc_bodyid"}
_LABEL_RE = {k: re.compile(rf"(?:^|,)\s*{k}:\s*([^,]+)") for k in LABEL_KEYS}


def ids_to_int64(s: pd.Series) -> pd.Series:
    """Text IDs -> int64, verifying the conversion is exact."""
    if s.isna().any():
        raise ValueError(f"{s.isna().sum()} missing IDs")
    txt = s.astype(str).str.strip()
    out = pd.Series(np.array([int(x) for x in txt], dtype=np.int64), index=s.index)
    if not (out.astype(str).to_numpy() == txt.to_numpy()).all():
        raise ValueError("ID changed in the int64 round-trip")
    return out


def read_neurons(path: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])   # everything as text first
    missing = set(NEURON_RENAME) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df.rename(columns=NEURON_RENAME)[list(NEURON_RENAME.values())]
    df["neuron_id_str"] = df["neuron_id"].astype(str)
    df["neuron_id"] = ids_to_int64(df["neuron_id"])
    df["nt_conf"] = pd.to_numeric(df["nt_conf"], errors="coerce").astype("float32")
    df.insert(0, "dataset", dataset)
    return df


def read_edges(path: Path, dataset: str) -> pd.DataFrame:
    e = pd.read_csv(
        path, usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"],
        dtype={"pre_root_id": "int64", "post_root_id": "int64",
               "neuropil": "category", "syn_count": "int32"})
    e = e[["pre_root_id", "post_root_id", "neuropil", "syn_count"]]
    e.columns = ["pre", "post", "neuropil_raw", "syn"]
    return e


def parse_mcns_labels(s) -> dict:
    """'hemibrainType: OCG01,statusLabel: Roughly traced' -> {'hemibrainType': 'OCG01', ...}"""
    out = {}
    if not isinstance(s, str):
        return out
    for k, rx in _LABEL_RE.items():
        m = rx.search(s)
        if m:
            out[k] = m.group(1).strip()
    return out


def label_columns(labels: pd.Series) -> pd.DataFrame:
    """Vectorised parse of MaleCNS community labels into one column per key."""
    cols = {}
    for k, rx in _LABEL_RE.items():
        cols[LABEL_COLUMNS[k]] = labels.str.extract(rx, expand=False).str.strip()
    return pd.DataFrame(cols, index=labels.index)


def read_map(path: Path) -> pd.DataFrame:
    """Vocabulary maps: every value is text; '<missing>' stands for an empty raw value."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)
