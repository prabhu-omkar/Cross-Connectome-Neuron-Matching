"""Build dist/flymatch_kaggle.zip: code + configs + the processed data training needs.

Upload the zip as a Kaggle Dataset (see INSTRUCTIONS.md). Raw CSVs are NOT included.

    python -m flymatch.run package                  # dist/flymatch_kaggle.zip (+ .sha256, manifest.json)
    python -m flymatch.run package --parts-mb 20    # also dist/kaggle_upload/: the zip in 20 MB parts + .sha256
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .config import REPO, Layout, load_data_config, sha256_file, write_json

CODE = ["pyproject.toml", "README.md", "INSTRUCTIONS.md", "LICENSE", "src", "configs", "maps", "kaggle"]
SKIP = ("__pycache__", ".pyc", "paths.yaml", ".egg-info")


def build(cfg: dict | None = None, parts_mb: int = 0):
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
        z.writestr("flymatch/MANIFEST.json", json.dumps({"files": manifest}, indent=1))   # checked by the notebook
    write_json({"files": manifest}, REPO / "dist" / "manifest.json")
    zip_sha = sha256_file(out)
    (out.parent / (out.name + ".sha256")).write_text(f"{zip_sha}  {out.name}\n")
    print(f"[package] wrote {out} ({out.stat().st_size / 2**20:.0f} MB, sha256 {zip_sha[:12]}…)")
    if parts_mb:
        split(out, parts_mb, zip_sha)
    return out


def split(zip_path: Path, parts_mb: int, zip_sha: str | None = None) -> Path:
    """Cut the zip into numbered parts for a flaky upload. The Kaggle notebook joins and verifies them."""
    dest = zip_path.parent / "kaggle_upload"
    dest.mkdir(exist_ok=True)
    for old in dest.glob(zip_path.name + ".*"):
        old.unlink()
    size, i = parts_mb * 2**20, 0
    with open(zip_path, "rb") as f:
        while chunk := f.read(size):
            i += 1
            (dest / f"{zip_path.name}.part{i:02d}").write_bytes(chunk)
    (dest / f"{zip_path.name}.sha256").write_text(f"{zip_sha or sha256_file(zip_path)}  {zip_path.name}\n")
    print(f"[package] split into {i} parts of <= {parts_mb} MB in {dest}")
    return dest


if __name__ == "__main__":
    build()
