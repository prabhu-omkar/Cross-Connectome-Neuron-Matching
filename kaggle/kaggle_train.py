# FlyMatch — train on Kaggle (script version of kaggle/flymatch_train.ipynb; prefer importing the notebook).
# Paste into ONE notebook cell. GPU on; Internet can stay off.
# Generated from the same source as the notebook — edit both by regenerating, not by hand.

# ── What to train ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = ["mlp", "gnn_L2", "gnn_L1", "gnn_L3"]   # files in configs/experiments/ (most important first)
SEEDS = [0, 1, 2]                                     # every experiment runs once per seed, seed-by-seed
SMOKE_TEST_FIRST = True       # ~1-minute end-to-end check before the long runs
SANITY_MIN_MLP_R1 = 0.55      # stop early if the MLP lands below this (expected ≈ 0.616 on validation)
ALLOW_CPU = False             # training on CPU takes ~40 s per step — leave False


# ── Setup: find the dataset, check it, build a working copy (no internet needed) ─────────────────
import hashlib, importlib.util, json, os, re, shutil, subprocess, sys, time, zipfile
from pathlib import Path

INPUT = Path(os.environ.get("FLYMATCH_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("FLYMATCH_WORK", "/kaggle/working"))
TMP = Path(os.environ.get("FLYMATCH_TMP", "/tmp/flymatch"))
REPO = WORK / "flymatch"
NEEDED = ["data/interim/neurons.parquet", "data/interim/edges.parquet", "data/interim/gt_types.parquet",
          "data/interim/gt_pairs_1to1.parquet", "data/interim/families.parquet",
          "data/processed/feature_spec.json", "data/processed/features_mcns.npy",
          "data/processed/features_banc.npy", "data/splits/splits_v1.json", "data/splits/splits_v1.sha256"]


def sh(cmd, cwd=None, env=None):
    """Run a command, streaming its output into the notebook. Returns the exit code."""
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def find_package():
    """Folder containing flymatch/pyproject.toml — however the dataset was uploaded."""
    # (a) Kaggle unpacked the zip. Ignore copies without data, e.g. an attached earlier notebook output.
    hits = [h for h in sorted(INPUT.glob("**/flymatch/pyproject.toml"))
            if (h.parent / "data/processed/features_mcns.npy").exists()]
    if hits:
        print("[setup] package:", hits[0].parent)
        return hits[0].parent.parent
    # (b) the zip in parts: flymatch_kaggle.zip.part01, .part02, ...   (c) the zip itself, not unpacked
    parts = sorted(INPUT.glob("**/flymatch_kaggle.zip.part*"))
    whole = sorted(INPUT.glob("**/flymatch_kaggle.zip"))
    if not parts and not whole:
        raise SystemExit("Dataset not found. Right-hand panel → Add Input → search 'flymatch' → Add.")
    TMP.mkdir(parents=True, exist_ok=True)
    if parts:
        nums = sorted(int(re.search(r"part(\d+)$", p.name).group(1)) for p in parts)
        gaps = sorted(set(range(1, nums[-1] + 1)) - set(nums))
        if gaps:
            raise SystemExit(f"Missing upload part(s): {gaps}. Add them to the dataset (New Version) and re-run.")
        parts = sorted(parts, key=lambda p: int(re.search(r"part(\d+)$", p.name).group(1)))
        zpath = TMP / "flymatch_kaggle.zip"
        print(f"[setup] joining {len(parts)} parts")
        with open(zpath, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, 1 << 22)
        want = sorted(INPUT.glob("**/flymatch_kaggle.zip.sha256"))
        if want:
            expected, got = want[0].read_text().split()[0], sha256(zpath)
            if got != expected:
                raise SystemExit("The joined zip does not match its checksum: one part is incomplete. "
                                 "Re-upload the parts (dataset → New Version) and re-run.")
            print("[setup] joined zip verified (sha256 OK)")
        else:
            print("[setup] no .sha256 file uploaded — skipping the checksum")
    else:
        zpath = whole[0]
    dest = TMP / "pkg"
    shutil.rmtree(dest, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad:
            raise SystemExit(f"The zip is corrupt (first bad member: {bad}). Re-upload it.")
        z.extractall(dest)
    print("[setup] unpacked to", dest)
    return dest


src = find_package() / "flymatch"
missing = [f for f in NEEDED if not (src / f).exists()]
if missing:
    raise SystemExit(f"The package is incomplete, missing: {missing}. Rebuild it with `run package` and re-upload.")

# integrity check against the build manifest, if one was shipped (inside the zip or uploaded next to it)
manifest = None
for cand in [src / "MANIFEST.json", *sorted(INPUT.glob("**/manifest.json"))]:
    if cand.exists():
        files = json.loads(cand.read_text()).get("files", {})
        if any(k.startswith("data/") for k in files):
            manifest = files
            break
if manifest:
    bad = [rel for rel, h in manifest.items() if (src / rel).exists() and sha256(src / rel) != h]
    if bad:
        raise SystemExit(f"Data files do not match the build manifest: {bad}. Re-upload the package.")
    print(f"[setup] {len(manifest)} data files verified against the manifest")

# working copy: code is copied (small), data files are linked (read-only input, ~700 MB — no copy)
REPO.mkdir(parents=True, exist_ok=True)
for item in src.iterdir():
    if item.name in ("data", "runs"):
        continue
    dst = REPO / item.name
    if item.is_dir():
        shutil.copytree(item, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dst)
for f in (src / "data").rglob("*"):
    if f.is_file():
        dst = REPO / f.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(f)
(REPO / "runs").mkdir(exist_ok=True)

# resume: runs found in any attached earlier output (Add Input → this notebook's previous version)
carried = 0
for cfg_file in sorted(INPUT.glob("**/runs/*/config.json")):
    run = cfg_file.parent
    dst = REPO / "runs" / run.name
    if not dst.exists():
        shutil.copytree(run, dst)
        carried += 1
for z in sorted(INPUT.glob("**/runs_out.zip")):
    with zipfile.ZipFile(z) as zz:
        names = {n.split("/")[1] for n in zz.namelist() if n.startswith("runs/") and n.count("/") >= 2}
        new = [n for n in names if not (REPO / "runs" / n).exists()]
        for n in new:
            for m in zz.namelist():
                if m.startswith(f"runs/{n}/"):
                    zz.extract(m, REPO)
        carried += len(new)
if carried:
    print(f"[setup] carried over {carried} run folder(s) from an earlier output — finished runs will be skipped")

# environment: Kaggle images already have every dependency
need = ["torch", "numpy", "pandas", "pyarrow", "scipy", "sklearn", "yaml", "matplotlib", "tqdm", "psutil"]
absent = [m for m in need if importlib.util.find_spec(m) is None]
if absent:
    print("[setup] installing", absent, "(needs Settings → Internet → On)")
    pip_names = {"sklearn": "scikit-learn", "yaml": "pyyaml"}
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *[pip_names.get(m, m) for m in absent]], check=True)

import torch
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
print(f"[setup] torch {torch.__version__} | GPU: {GPU or 'NONE'}")
if GPU is None and not ALLOW_CPU:
    raise SystemExit("No GPU. Settings → Accelerator → GPU T4 x2 (needs a phone-verified Kaggle account), then run again.")

ENV = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONUNBUFFERED="1")
assert sh([sys.executable, "-c", "import flymatch.run; print('[setup] flymatch imports OK')"], cwd=REPO, env=ENV) == 0
print("[setup] ready:", REPO)


# ── Train ────────────────────────────────────────────────────────────────────────────────────────
# Seed-by-seed order: if time runs out, every experiment has at least one finished seed.
# Re-running this cell (or the notebook) skips finished runs and resumes unfinished ones.

def train(exp, seed):
    t0 = time.time()
    rc = sh([sys.executable, "-u", "-W", "ignore", "-m", "flymatch.run", "train",
             "--exp", f"configs/experiments/{exp}.yaml", "--seeds", str(seed), "--device", "auto"],
            cwd=REPO, env=ENV)
    return rc, (time.time() - t0) / 60


def metrics_of(exp, seed):
    found = sorted((REPO / "runs").glob(f"*_{exp}_seed{seed}_*/metrics.json"))
    return json.loads(found[-1].read_text()) if found else None


if SMOKE_TEST_FIRST:
    print("=" * 30, "smoke test (tiny model, a few steps)", "=" * 30)
    rc, _ = train("smoke_gnn_cpu", 0)
    for d in (REPO / "runs").glob("*_smoke_gnn_cpu_seed0_*"):
        shutil.rmtree(d)                                   # not a result — don't keep it
    if rc:
        raise SystemExit("Smoke test failed (see the error above). Nothing else was run.")

failed = []
for seed in SEEDS:
    for exp in EXPERIMENTS:
        print("=" * 30, f"{exp}  seed {seed}", "=" * 30)
        rc, mins = train(exp, seed)
        if rc:
            failed.append(f"{exp} seed {seed}")
            print(f"!! {exp} seed {seed} failed (exit code {rc}) — continuing with the next run")
            continue
        m = metrics_of(exp, seed)
        if exp == "mlp" and seed == SEEDS[0] and m and m["T1/unfiltered/mean/R@1"] < SANITY_MIN_MLP_R1:
            raise SystemExit(f"Sanity check failed: MLP R@1 = {m['T1/unfiltered/mean/R@1']:.3f} "
                             f"(expected ≈ 0.616). The package is probably stale — rebuild and re-upload it.")
        print(f"   done in {mins:.1f} min")

print("\nAll runs attempted." + (f" FAILED: {failed}" if failed else " None failed."))


# ── Results: summary table + runs_out.zip to download ────────────────────────────────────────────
import pandas as pd

sh([sys.executable, "-m", "flymatch.run", "aggregate"], cwd=REPO, env=ENV)

KEYS = {"R@1": "T1/unfiltered/mean/R@1", "R@1 filtered": "T1/filtered/mean/R@1",
        "MRR": "T1/unfiltered/mean/MRR", "T2 macro": "T2/unfiltered/mean/acc@1_macro"}
rows = []
for f in sorted((REPO / "runs").glob("*/metrics.json")):
    m, c = json.loads(f.read_text()), json.loads((f.parent / "config.json").read_text())
    rows.append({"method": c.get("method"), "seed": c.get("seed"),
                 **{k: m.get(v) for k, v in KEYS.items()},
                 "minutes": round((m.get("train_seconds") or 0) / 60, 1)})
if rows:
    df = pd.DataFrame(rows)
    order = {e: i for i, e in enumerate(EXPERIMENTS)}
    summary = (df.groupby("method")[list(KEYS)].agg(["mean", "std"]).round(3)
                 .sort_index(key=lambda s: s.map(lambda x: order.get(x, 99))))
    print("\nValidation results (mean and std over seeds):\n")
    print(summary.to_string())
    print("\nBars to beat (validation): R@1 0.616 ± 0.007 and MRR 0.701 (B4 MLP on CPU); "
          "R@1 filtered 0.830 and T2 macro 0.574 (B3 cocoa-style).")
    print("The GNN counts as better only if it beats the MLP by more than the across-seed std (plan, H2).")

shutil.make_archive(str(WORK / "runs_out"), "zip", REPO, "runs")
shutil.rmtree(REPO / "data", ignore_errors=True)      # links to the input — keeps the saved output small
print(f"\nDownload {WORK / 'runs_out.zip'} from the Output tab, then on the laptop:")
print(r"  Expand-Archive runs_out.zip -DestinationPath C:\projects\FLYbrain\flymatch -Force")
print(r"  .venv\Scripts\python.exe -m flymatch.run aggregate")
