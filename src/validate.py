"""Time-split validation: the final method evaluated on past weeks.

For each validation cutoff v, exactly as in predict.py: train RelGNN on seeds whose label week ended before the
last labelled week, select the epoch on that week, predict week v, apply the plug-in decision and compute the
official Score. Probabilities are saved for analysis.

    python src/validate.py --data data --weeks "2026-10-30 16:00" --out validation
"""
import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (set before CUDA starts)

import numpy as np
import pandas as pd
import torch

import decision
import task
import train
from data import TARGETS, AssignmentData
from graph import build_graph

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data", help="output folder of the course preprocess.py (contains db/)")
ap.add_argument("--weeks", default="2026-10-09 16:00,2026-10-16 16:00,2026-10-23 16:00,2026-10-30 16:00",
                help="validation cutoffs (UTC); each predicts the week [cutoff + 1 d, cutoff + 8 d)")
ap.add_argument("--start", default="2026-06-15 16:00",
                help="first training cutoff (UTC) whose user-selection window lies inside the data (from 2026-03-01)")
ap.add_argument("--seeds", type=int, default=2, help="number of RelGNN models averaged")
ap.add_argument("--out", default="validation")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = AssignmentData(a.data)
graph, col_stats = build_graph(data)
cutoffs = task.labelled_cutoffs(data, a.start)
print(f"data end {data.data_end}, last labelled cutoff {cutoffs[-1]}", flush=True)
for v in [pd.Timestamp(w) for w in a.weeks.split(",")]:
    assert v in cutoffs, f"{v} has no complete label week in this release"
    train_c, val_c = task.split_by_time(cutoffs, v)
    tr, va, te = task.task_table(data, train_c), task.task_table(data, val_c), task.task_table(data, [v])
    print(f"week {v}: train {len(tr):,} seeds ({train_c[0]:%m-%d} .. {train_c[-1]:%m-%d}), "
          f"validation {len(va):,} ({val_c[0]:%m-%d} .. {val_c[-1]:%m-%d}), test {len(te):,}", flush=True)
    p = []
    for s in range(a.seeds):
        model, best = train.fit(graph, col_stats, tr, va, device, seed=s, log=lambda m: print(m, flush=True))
        p.append(train.predict_proba(model, train.make_loader(graph, te, False), device))
    y = te[TARGETS].to_numpy()
    res = decision.score(decision.expected_f1_predict(np.mean(p, 0)), y)
    np.savez(os.path.join(a.out, f"week_{v:%m%d}.npz"), p_seeds=np.stack(p), y=y, user_id=te.user_id.to_numpy())
    # the same week in the official CSV format, to be scored with scripts/evaluate.py --pred ... --truth ...
    gh = data.github_id.reindex(te.user_id).to_numpy()
    for kind, m in (("pred", decision.expected_f1_predict(np.mean(p, 0))), ("truth", y)):
        csv = pd.DataFrame(m, columns=TARGETS).astype(int)
        csv.insert(0, "UserID", gh)
        csv.to_csv(os.path.join(a.out, f"week_{v:%m%d}_{kind}.csv"), index=False)
    json.dump(dict(week=str(v), n_train=len(tr), n_val=len(va), n_test=len(te), seeds=a.seeds,
                   score=res["score"], macro_f1=res["macro"], micro_f1=res["micro"],
                   f1=dict(zip(TARGETS, res["per_activity"]))), open(os.path.join(a.out, f"week_{v:%m%d}.json"), "w"), indent=1)
    print(f"week {v}: Score {res['score']:.4f} (macro-F1 {res['macro']:.4f}, micro-F1 {res['micro']:.4f})", flush=True)
