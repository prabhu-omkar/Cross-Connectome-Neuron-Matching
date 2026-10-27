# 🚀 Train on Kaggle

Everything is prepared and public on Kaggle. Copy the notebook, switch on the GPU, press one button.
Training runs on Kaggle's free GPU in the background — your laptop can be closed.

> **Your time:** ~2 minutes. **Training time:** ~2–4 hours (12 runs on a T4). **Cost:** free (Kaggle gives ~30 GPU-hours/week).
> **Internet is not needed** inside the notebook — every dependency is already on Kaggle.

---

## The quick way · use the public Kaggle notebook

Data and notebook are already on Kaggle, public:

- **Dataset:** https://www.kaggle.com/datasets/omkarprabhuuuu/flymatch
- **Notebook:** https://www.kaggle.com/code/omkarprabhuuuu/flymatch-train

1. Sign in at kaggle.com and phone-verify the account once (**Settings → Phone verification**) — unverified accounts get no GPU.
2. Open the notebook link → **Copy & Edit**. Your copy already has the `flymatch` dataset attached.
3. Right-hand panel → **Settings → Accelerator → GPU T4 x2** (GPU P100 also works).
4. Top right: **Save Version → Save & Run All (Commit) → Save**.

That's it. The public notebook itself has never been run — training happens only in your copy.
Watch progress under **View Active Events** if you like. The notebook:

1. finds the data and checks every file against its checksum;
2. runs a 1-minute smoke test;
3. trains **MLP, GNN-2, GNN-1, GNN-3** for seed 0, then all four for seed 1, then seed 2 — so if anything is cut short, every model still has a result;
4. stops immediately if the MLP sanity run is far off its known score (≈ 0.616), so a broken setup can't burn hours;
5. prints a results table and writes **`runs_out.zip`**.

---

## The manual way · upload it yourself

Only needed if you rebuilt the package or want a private copy.

1. kaggle.com → **Datasets → + New Dataset** → drag in **`dist\flymatch_kaggle.zip`** (84 MB; Kaggle unpacks it, and the checksum list travels inside it). Title `flymatch` → **Create**.
2. **Code → + New Notebook → File → Import Notebook** → `kaggle\flymatch_train.ipynb`.
3. **Add Input** → your `flymatch` dataset; **Accelerator → GPU T4 x2**; **Save & Run All**.

---

## Bring the results home

1. Open the notebook → the finished version → **Output** tab → download **`runs_out.zip`**.
2. On the laptop:
   ```powershell
   Expand-Archive runs_out.zip -DestinationPath C:\projects\FLYbrain\flymatch -Force
   cd C:\projects\FLYbrain\flymatch
   .venv\Scripts\python.exe -m flymatch.run aggregate
   ```
   `results\baselines.csv` now has `mlp`, `gnn_L1`, `gnn_L2`, `gnn_L3` (mean ± sd over seeds) next to B0–B4.

---

## Reading the numbers

| Metric (validation) | Bar to beat | Held by |
|---|---|---|
| T1 R@1, unfiltered | **0.616 ± 0.007** | B4 MLP |
| T1 MRR | **0.701** | B4 MLP |
| T1 R@1, filtered | **0.830** | B3 cocoa-style |
| T2 macro type accuracy | **0.574** | B3 cocoa-style |

The GNN counts as better only if it beats the MLP by **more than the across-seed standard deviation** (plan, hypothesis H2).
If it doesn't, that is a legitimate result: keep the MLP as the main model and report it.

---

## If something goes wrong

| What you see | Do this |
|---|---|
| `Dataset not found` | Right-hand panel → **Add Input** → search `flymatch` → add the public dataset |
| `No GPU` | **Settings → Accelerator → GPU T4 x2**; phone-verify the account |
| `does not match its checksum` | The upload arrived incomplete. Dataset → **New Version** → re-upload the zip → re-run |
| `Sanity check failed: MLP R@1 …` | The uploaded package is out of date. Rebuild (below) and upload a new dataset version |
| A run prints `failed` | The others continue. The table at the end lists failures — send the error text |
| The commit stopped before finishing (12 h limit) | Right panel **Add Input → Your Work** → add this notebook (its output) → **Save & Run All** again. Finished runs are skipped; unfinished ones resume from their checkpoint |
| CUDA out of memory | Switch the accelerator to **GPU P100**, or set `model: {width: 128}` in the experiment YAML and rebuild |
| Weekly GPU quota used up | It resets weekly. The MLP also trains on the laptop CPU in ~5 min per seed |

---

## Changing what gets trained

The first code cell of the notebook has the lists:

```python
EXPERIMENTS = ["mlp", "gnn_L2", "gnn_L1", "gnn_L3"]   # files in configs/experiments/
SEEDS = [0, 1, 2]
```

Edit them in the Kaggle editor before **Save & Run All**. To change hyper-parameters, edit or add a YAML in
`configs/experiments/` on the laptop, then rebuild and upload a new dataset version.

## Rebuilding the upload (only when code or data change)

```powershell
cd C:\projects\FLYbrain\flymatch
.venv\Scripts\python.exe -m flymatch.run package
```

This writes `dist\flymatch_kaggle.zip` (with its checksum list inside) and `dist\flymatch_kaggle.zip.sha256`. On Kaggle: the `flymatch` dataset → **New Version** → drag in the new zip. (On a flaky connection, `--parts-mb 20` also writes the zip as 20 MB parts in `dist\kaggle_upload\`; the notebook joins them.)

## Where the notebook comes from

`kaggle/flymatch_train.ipynb` and `kaggle/kaggle_train.py` (the same code as one script, for pasting into a single cell) are generated together, so they never drift apart. Each run folder in `runs_out.zip` holds `config.json`, `splits_sha256.txt`, `metrics.json` (all metrics, 95% CIs, training history), `ranks.parquet` and `model.pt`. `aggregate` rejects runs made with a different split hash.
