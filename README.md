# BigData 2026 Fall – Assignment 2: Predicting GitHub User Activity

For every user in `test_input.csv`, predict whether they will `open_pr`, `open_issue`, `issue_comment`,
`pr_review`, `pr_review_comment` and `push` during 2026/11/12 00:00 – 11/19 00:00 (Taiwan time).

**Model: RelGNN** (Chen, Kanatsoulis & Leskovec, *RelGNN: Composite Message Passing for Relational Deep
Learning*, ICML 2025), trained directly on the relational database. The code in `src/relgnn/` comes from the
official repository https://github.com/snap-stanford/RelGNN (commit `cffdb8b`, MIT License). Every file there
names its source file and its (minimal) changes.

## Method

1. **Relational entity graph** (`src/graph.py`, built with RelBench's `make_pkey_fkey_graph`):
   - Each of the 8 tables is a node type and each row is a node.
   - Each foreign key gives an edge type plus its reverse.
   - Node features are the table's own columns, encoded with PyTorch Frame.
   - Each node's time is the table's row-time column.
2. **Task table** (`src/task.py`):
   - Seeds are (user, cutoff *t*) for daily cutoffs since 2026-03-01.
   - Users are chosen with the official test-user rule (≥ 5 activity records in the 90 days before *t* − 14 days).
   - Labels are the 6 activities in [*t* + 4 h, *t* + 4 h + 7 days).
3. **RelGNN** (`src/train.py`, `src/relgnn/`):
   - Atomic routes come from the schema: `dim-dim` for a single foreign key, `dim-fact-dim` for each pair of
     foreign keys of a table.
   - Composite message passing with attention.
   - A relative-time encoding of (seed time − row time) on every node.
   - Temporal neighbour sampling: [128, 64] neighbours per edge type, rows before *t* only.
   - Training: binary cross-entropy, Adam with lr 0.005, 10 epochs of 500 steps; the epoch with the best
     macro AUPRC on the last labelled week is kept.
4. **Decision** (`src/decision.py`): two models are averaged. For each activity, the users predicted 1 are the
   top-*m* users, with *m* chosen to maximise the expected (macro-F1 + micro-F1) / 2 under the predicted
   probabilities.

## Files

| File | Content |
|---|---|
| `src/data.py` | Reads the 8 tables; cohort activity log; test-user rule; labels |
| `src/graph.py` | Database → relational entity graph; leakage rules |
| `src/task.py` | (user, cutoff) seeds with labels; time-based train / validation split |
| `src/train.py` | RelGNN model, temporal neighbour loader, training loop, inference |
| `src/decision.py` | Official score; plug-in expected-F1 decision |
| `src/validate.py` | Time-split validation on past weeks |
| `src/predict.py` | Final prediction → `output_<student_id>.csv` |
| `src/relgnn/` | Code from the official RelGNN repository (atomic routes, RelGNN conv / hetero conv / model, RelBench graph builder) |

## Environment

Python 3.11, PyTorch 2.6 (CUDA 12.4), PyTorch Geometric 2.8.1, pyg-lib 0.4.0 (temporal sampling),
PyTorch Frame 0.2.3, pandas, pyarrow, scikit-learn.

```bash
conda create -n bigdata_a2 python=3.11 -y && conda activate bigdata_a2
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric==2.8.1 pandas pyarrow scikit-learn
pip install --no-deps pytorch-frame==0.2.3
pip install --no-deps "pyg-lib==0.4.0+pt26cu124" -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

One GPU with about 12 GB of memory is enough.

## How to run

```bash
# 1. data (to update: delete the folder and clone again; do not git pull)
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2

# 2. train and predict (about 1 hour on one RTX 3090): writes ./output_414540015.csv
python src/predict.py --data assignment2 --student_id 414540015 --out .

# 3. format check
python assignment2/evaluation/evaluate.py --pred output_414540015.csv --input assignment2/evaluation/test_input.csv

# optional: time-split validation on past weeks (one week per GPU is fastest)
python src/validate.py --data assignment2 --weeks "2026-10-14 12:00" --out validation
```

## Data rules followed

* **Only the assignment data is used.**
* **Known-later columns are never used:** `PullRequests.merged_at`, `merged_by_id`, `closed_at`,
  `Issues.closed_at`, `state_reason` and `PRReviews.dismissed_at`. They are not node features, and
  `merged_by_id` is not used as a foreign key.
* **Every row only becomes visible at its row time:** `first_seen_at` for Users, Repositories, PullRequests,
  Issues and PRReviews, and `created_at` for IssueComments, PRReviewComments and Pushes. A seed at time *t*
  only samples rows with row time < *t*. Pushes are shifted by the push lag of the release, so that each
  cutoff sees what the final release would show.
* **Split by time, not at random.** Training seeds have label weeks that end before the validation week, and
  the validation week ends before the predicted week. The final model only uses weeks that have ended in the
  data.
* **The test seed time is 2026-11-11 12:00 UTC** (20:00 Taiwan, the last data update), so the predicted week
  is 11/12 00:00 – 11/19 00:00 Taiwan.
