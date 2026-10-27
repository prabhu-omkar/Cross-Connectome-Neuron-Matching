# Paste this whole file into ONE cell of a Kaggle notebook (GPU on, Internet on), then run it.
# It copies the uploaded dataset into /kaggle/working, installs flymatch, trains, and zips the results.
#
# Edit these two lines for each session:
EXPERIMENTS = ["configs/experiments/mlp.yaml", "configs/experiments/gnn_L1.yaml",
               "configs/experiments/gnn_L2.yaml", "configs/experiments/gnn_L3.yaml"]
SEEDS = "0,1,2"

import glob, os, shutil, subprocess, sys

# 1. find the uploaded package (folder that contains flymatch/pyproject.toml)
hits = glob.glob("/kaggle/input/**/flymatch/pyproject.toml", recursive=True)
assert hits, "Dataset not found: add your flymatch dataset to this notebook (Add Input)."
src = os.path.dirname(hits[0])
work = "/kaggle/working/flymatch"
if not os.path.exists(work):                    # keep an existing copy: it holds checkpoints to resume
    shutil.copytree(src, work)
os.chdir(work)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."], check=True)

import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — turn on GPU in Settings")

# 2. train (re-running this cell after a crash resumes from the last checkpoint)
for exp in EXPERIMENTS:
    subprocess.run([sys.executable, "-W", "ignore", "-m", "flymatch.run", "train",
                    "--exp", exp, "--seeds", SEEDS, "--device", "auto"], check=True)

# 3. summary table + zip of runs/ for download (models are small, so checkpoints are included)
subprocess.run([sys.executable, "-m", "flymatch.run", "aggregate"], check=True)
shutil.make_archive("/kaggle/working/runs_out", "zip", work, "runs")
print("Download /kaggle/working/runs_out.zip and unzip it into C:\\projects\\FLYbrain\\flymatch\\")
