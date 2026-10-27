"""Build dist/flymatch_kaggle.zip: code + configs + the processed data training needs.

Upload the zip as a Kaggle Dataset (see INSTRUCTIONS.md). Raw CSVs are NOT included.
"""
from __future__ import annotations

import zipfile

from .config import REPO, Layout, load_data_config, sha256_file, write_json

CODE = ["pyproject.toml", "README.md", "INSTRUCTIONS.md", "LICENSE", "src", "configs", "maps", "kaggle"]
SKIP = ("__pycache__", ".pyc", "paths.yaml", ".egg-info")


def build(cfg: dict | None = None):
    layout = Layout(cfg or load_data_config())
    data = [layout.neurons, layout.edges, layout.gt_types, layout.gt_pairs, layout.families,
            layout.feature_spec, layout.features("mcns"), layout.features("banc"),
            layout.split_file, layout.splits / "splits_v1.sha256"]
    missing = [p.name for p in data if not p.exists()]
    if missing:
        raise SystemExit(f"[package] missing {missing}; run `python -m flymatch.run data` first")
    out = REPO / "dist" / "flymatch_kaggle.zip"
    out.parent.mkdir(exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for item in CODE:
            p = REPO / item
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            for f in files:
                rel = f.relative_to(REPO).as_posix()
                if not any(s in rel for s in SKIP):
                    z.write(f, f"flymatch/{rel}")
        for f in data:
            rel = f.relative_to(REPO).as_posix()
            z.write(f, f"flymatch/{rel}")
            manifest[rel] = sha256_file(f)
        z.writestr("flymatch/configs/paths.yaml", "raw_root: ..\nmcns_dir: ..\nbanc_dir: ..\nexternal_dir: ../external\n")
    write_json({"files": manifest}, REPO / "dist" / "manifest.json")
    print(f"[package] wrote {out} ({out.stat().st_size / 2**20:.0f} MB)")
    return out


if __name__ == "__main__":
    build()
