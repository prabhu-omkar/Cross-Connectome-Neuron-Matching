# 🚀 Training on Kaggle — step by step

The GNN does a full-graph forward pass over **both** connectomes every step (325,000 neurons, 7.8 M edges).
On a laptop CPU that takes about 42 s per step; on a Kaggle T4 GPU it takes under a second. This guide
takes you from zero to trained models with results back on your laptop.

> **Time needed:** about 15 minutes of clicking once, then about 2 minutes per training session.
> **Cost:** free. Kaggle gives about 30 GPU-hours per week.

---

## 0 · What you need

- A Kaggle account (kaggle.com, sign in with Google), **phone-verified** — unverified accounts get no GPU or internet.
  Account → Settings → Phone verification.
- The training package **`flymatch_kaggle.zip`** (84 MB). Get it either way:
  - **Download it** from this repository's [Releases page](https://github.com/prabhu-omkar/Cross-Connectome-Neuron-Matching/releases), or
  - **build it yourself** after `python -m flymatch.run data`:
    ```powershell
    .venv\Scripts\python.exe -m flymatch.run package      # → dist\flymatch_kaggle.zip
    ```

The zip holds the code, configs, maps, the frozen split, and the processed tables and features training needs.
It does not hold the raw CSVs.

---

## 1 · Create the Kaggle Dataset (once)

1. kaggle.com → left menu **Datasets** → **+ New Dataset**.
2. Drag in `flymatch_kaggle.zip`. Kaggle unzips it automatically.
   *(Alternative: choose the **Link** tab and paste the zip's URL from the GitHub Release.)*
3. Title: **`flymatch`**. Leave visibility **Private**. Click **Create**.
4. Wait until processing finishes (a minute or two).

Only re-upload if the code or `data/` changes: open the dataset → **New Version** → upload the new zip.

---

## 2 · Create the notebook (once)

1. kaggle.com → **Code** → **+ New Notebook**.
2. Right-hand panel → **Input** → **+ Add Input** → search your **`flymatch`** dataset → **Add**.
3. Right-hand panel → **Settings**:

   | Setting | Value | Why |
   |---|---|---|
   | **Accelerator** | **GPU T4 x2** (or GPU P100) | the code uses one GPU |
   | **Persistence** | **Files only** | checkpoints survive a session restart |
   | **Internet** | **On** | `pip install` of the package |
   | Language | Python | |

4. Delete the starter cell. Open [`kaggle/kaggle_train.py`](kaggle/kaggle_train.py), copy **all** of it, and paste
   it into **one** cell.

---

## 3 · Choose what to train

The top of the cell has two lines to edit:

```python
EXPERIMENTS = ["configs/experiments/mlp.yaml", "configs/experiments/gnn_L1.yaml",
               "configs/experiments/gnn_L2.yaml", "configs/experiments/gnn_L3.yaml"]
SEEDS = "0,1,2"
```

| Experiment | What it is | What it tells you |
|---|---|---|
| `mlp.yaml` | B4: the same model with **0** message-passing layers | Sanity check — must land near **0.616** R@1 (the CPU result). If not, the GPU path is broken. |
| `gnn_L1.yaml` | 1 directed layer | Does seeing direct partners help? |
| `gnn_L2.yaml` | 2 layers (the planned default) | Partners of partners |
| `gnn_L3.yaml` | 3 layers | Is there anything left at 3 hops? |

**Recommended first session:** `mlp.yaml` + `gnn_L2.yaml`, `SEEDS = "0"`. Check the numbers, then run
everything with `SEEDS = "0,1,2"`.

To change hyper-parameters, edit or copy a YAML in `configs/experiments/`, then re-package and re-upload.
Every key in `configs/train/default.yaml` can be overridden there.

---

## 4 · Run

- **Quick / interactive:** click ▶ **Run** on the cell. You'll see lines like

  ```text
  torch 2.x | GPU: Tesla T4
  [train] gnn_L2 seed 0 -> 20261028_gnn_L2_seed0_ab12cd34
    step   100 loss 2.104 tau 0.069 val MRR 0.690 R@1 0.601 (45s)
    ...
    early stop at step 1300; best step 300
  [train] gnn_L2 seed 0: T1 R@1 0.6xx (filtered 0.6xx) MRR 0.7xx | T2 macro 0.4xx
  ```

  Keep the browser tab open; interactive sessions stop after about 20 minutes idle.
- **Long runs (recommended for 3 seeds × 4 experiments):** top right **Save Version** → **Save & Run All
  (Commit)** → **Save**. This runs in the background for up to 12 h, even with your laptop closed.
  Progress: **View Active Events**.

Training checkpoints every **200 steps**, evaluates on validation every **100 steps**, keeps the best
checkpoint, and stops early after 10 evaluations without improvement.

---

## 5 · If the session dies

Run the **same cell again** (with Persistence = Files only). Each unfinished run prints

```text
[train] gnn_L2 seed 0 -> 20261028_gnn_L2_seed0_ab12cd34 (resuming)
  resumed from step 600 (best val MRR 0.712 at step 400)
```

and continues where it stopped. Finished runs are skipped
(`already finished … — skipping`): a run counts as finished once it has a `metrics.json`. A new run starts only
when you change its config.

---

## 6 · Bring the results home

1. When the cell finishes: right-hand panel **Output** (or the notebook's **Output** tab for a committed version)
   → download **`runs_out.zip`**.
2. Unzip it into the repository folder so that `runs\` lands next to `src\`:
   ```powershell
   Expand-Archive runs_out.zip -DestinationPath C:\projects\FLYbrain\flymatch -Force
   ```
3. Rebuild the results table:
   ```powershell
   .venv\Scripts\python.exe -m flymatch.run aggregate
   ```
   `results\baselines.csv` now has rows for `mlp`, `gnn_L1`, `gnn_L2` and `gnn_L3` next to B0–B4,
   with mean ± sd over seeds. Runs made with a different split hash are rejected automatically.

Each run folder contains `config.json`, `git_sha.txt`, `splits_sha256.txt`, `metrics.json` (all metrics +
95% CIs + training history), `ranks.parquet` (per-query results) and `model.pt`.

---

## 7 · How to read the numbers

| Metric | Bar to beat (validation) | Held by |
|---|---|---|
| T1 R@1, unfiltered (`T1/unfiltered/mean/R@1`) | **0.616 ± 0.007** | B4 MLP |
| T1 MRR | **0.701** | B4 MLP |
| T1 R@1, filtered | **0.830** | B3 cocoa-style |
| T2 macro type accuracy | **0.574** | B3 cocoa-style |

The plan's rule (hypothesis H2): the GNN wins only if it beats B4 by **more than the across-seed standard
deviation**. If it doesn't by week 9, keep the MLP as the main model and report that message passing did
not help. That is a legitimate result.

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `Dataset not found` | Add Input → your `flymatch` dataset (step 2.2) |
| `GPU: NONE` | Settings → Accelerator → GPU; phone-verify your account |
| `pip install` fails | Settings → Internet → On |
| CUDA out of memory | Use GPU P100 (16 GB) or set `model: {width: 128}` in the experiment YAML |
| `mlp` gives far below 0.6 | The package is stale: rebuild it with `run package` and upload a new dataset version |
| `aggregate` skips runs | They were made with a different split hash; the frozen split must not change |
| Weekly GPU quota used up | It resets weekly; MLP experiments also run on the laptop CPU (`--device cpu`, ~5 min/seed) |
