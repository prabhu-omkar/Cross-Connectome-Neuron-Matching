# Training on Kaggle (GPU)

## Once: make the dataset
1. On the laptop: `.venv\Scripts\python.exe -m flymatch.run package` → `dist\flymatch_kaggle.zip`.
2. kaggle.com → **Datasets → New Dataset** → upload the zip → title `flymatch` → Create (private).
   Re-upload (New Version) only if `data/` or the code changes.

## Each training session
1. kaggle.com → **Code → New Notebook**.
2. Right panel → **Add Input** → your `flymatch` dataset.
3. **Settings → Accelerator → GPU T4 x2** (or P100). **Settings → Persistence → Files only**
   (so checkpoints survive if the session dies). Internet: on.
4. Paste `kaggle/kaggle_train.py` into one cell; edit `EXPERIMENTS` / `SEEDS` at the top; Run.
   For long runs use **Save Version → Save & Run All** (runs in the background, up to 12 h).
5. When done: Output panel → download `runs_out.zip` → unzip into `C:\projects\FLYbrain\flymatch\`
   (it creates `runs\...`). Then on the laptop: `.venv\Scripts\python.exe -m flymatch.run aggregate`.

If a session dies, just run the cell again — each run resumes from its `last.pt` (saved every 200 steps).

## What to run in week 7–8
| Experiment | What it answers | Expected time on a T4 |
|---|---|---|
| `mlp.yaml` | GPU path reproduces B4 (~0.62 R@1)? | minutes |
| `gnn_L1/L2/L3.yaml` | Does message passing beat B4? How many layers? | ~10–30 min per seed |

Bar to beat (validation, T1 R@1 both directions): **B4 0.616 ± 0.007**; filtered: **B3 0.830**.
