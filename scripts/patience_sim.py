"""What would early stopping with a smaller patience have cost? Replays the recorded
validation-MRR curves (results/week8_val_curves.csv, from week8_compare.py).

    python scripts/patience_sim.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
c = pd.read_csv(REPO / "results" / "week8_val_curves.csv")
rows = []
for (m, s), g in c.groupby(["method", "seed"]):
    g = g.sort_values("step")
    full_best = g.val_MRR.max()
    sec_per_eval = g.sec.diff().median()
    for p in (3, 5, 10):
        best, bad, stop, best_r1 = -1, 0, g.step.iloc[-1], None
        for _, r in g.iterrows():
            if r.val_MRR > best:
                best, bad, best_r1 = r.val_MRR, 0, r["val_R@1"]
            else:
                bad += 1
            if bad >= p:
                stop = r.step
                break
        rows.append({"method": m, "seed": s, "patience": p, "stop_step": int(stop), "best_MRR": best,
                     "MRR_lost": full_best - best, "val_R@1_at_best": best_r1,
                     "minutes": stop / 100 * sec_per_eval / 60})
df = pd.DataFrame(rows)
df.to_csv(REPO / "results" / "patience_sim.csv", index=False)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
