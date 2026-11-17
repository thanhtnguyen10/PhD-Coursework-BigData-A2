"""Final prediction: the 6 activities of every user in test_input.csv for 2026/11/22 00:00 - 11/29 00:00 (Taiwan).

Our data ends at 2026-11-18 16:00 UTC (= 11/19 00:00 Taiwan, the test seed time). The test week starts 3 days
later, 2026-11-21 16:00 UTC (= 11/22 00:00 Taiwan), so training seeds use the same 3-day gap: labels in
[t + 3 d, t + 10 d).
RelGNN is trained on all labelled daily cutoffs (the last labelled week selects the epoch), several models are
averaged, and the plug-in expected-F1 rule gives the 0/1 predictions.

    python src/predict.py --data data --test_input assignment2/submission/test_input.csv --out .
"""
import argparse
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (set before CUDA starts)

import numpy as np
import pandas as pd
import torch

import decision
import task
import train
from data import HORIZON, TARGETS, AssignmentData
from graph import build_graph

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data", help="output folder of the course preprocess.py (contains db/)")
ap.add_argument("--test_input", default="assignment2/submission/test_input.csv")
ap.add_argument("--student_id", default="414540015")
ap.add_argument("--test_time", default="2026-11-18 16:00", help="seed time of the test week (UTC) = end of our data")
ap.add_argument("--test_week", default="2026-11-21 16:00", help="start of the test week (UTC) = 11/22 00:00 Taiwan")
ap.add_argument("--start", default="2026-06-15 16:00",
                help="first training cutoff (UTC) whose user-selection window lies inside the data (from 2026-03-01)")
ap.add_argument("--seeds", type=int, default=2, help="number of RelGNN models averaged")
ap.add_argument("--out", default=".")
a = ap.parse_args()
t0 = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T = pd.Timestamp(a.test_time)
gap = pd.Timestamp(a.test_week) - T
data = AssignmentData(a.data, gap=gap)
if data.data_end < T - pd.Timedelta(hours=1):
    print(f"WARNING: data ends {data.data_end}, before the test seed time {T}: download the hour files up to 2026-11-18-15 (UTC)")
cutoffs = task.labelled_cutoffs(data, a.start)
train_c, val_c = task.split_by_time(cutoffs, cutoffs[-1] + gap + HORIZON, gap)  # all labelled cutoffs are usable
ids = pd.read_csv(a.test_input).UserID.astype(np.int64).to_numpy()
known = np.isin(ids, data.row_id.index)  # all test users are cohort users; this only guards the output format
print(f"data end {data.data_end}; test seed time {T}, gap {gap}; {len(ids)} test users "
      f"({(~known).sum()} unknown); train cutoffs {train_c[0]:%m-%d} .. {train_c[-1]:%m-%d}, "
      f"validation {val_c[0]:%m-%d} .. {val_c[-1]:%m-%d}", flush=True)

graph, col_stats = build_graph(data)
tr, va = task.task_table(data, train_c), task.task_table(data, val_c)
te = task.task_table(data, [T], users=data.row_id.reindex(ids[known]).to_numpy(), with_labels=False)
p = []
for s in range(a.seeds):
    model, best = train.fit(graph, col_stats, tr, va, device, seed=s, log=lambda m: print(m, flush=True))
    p.append(train.predict_proba(model, train.make_loader(graph, te, False), device))
    print(f"model {s + 1}/{a.seeds} done ({time.time() - t0:.0f}s)", flush=True)

pred = np.zeros((len(ids), len(TARGETS)), dtype=int)
pred[known] = decision.expected_f1_predict(np.mean(p, 0))
out = pd.DataFrame(pred, columns=TARGETS)
out.insert(0, "UserID", ids)
path = os.path.join(a.out, f"output_{a.student_id}.csv")
out.to_csv(path, index=False)
print(f"wrote {path}: {len(out)} rows; share of 1s per activity {dict(zip(TARGETS, pred.mean(0).round(3)))}; "
      f"total {time.time() - t0:.0f}s")
