# BigData 2026 Fall – Assignment 2: Predicting GitHub User Activity

For every user in the course file `submission/test_input.csv` (4,465 users), predict whether the user will
`open_pr`, `open_issue`, `issue_comment`, `pr_review`, `pr_review_comment` and `push` during
**2026/11/22 00:00 – 11/29 00:00 (Taiwan time)**.

**Model: RelGNN** (Chen, Kanatsoulis & Leskovec, *RelGNN: Composite Message Passing for Relational Deep
Learning*, ICML 2025), trained directly on the relational database. The code in `src/relgnn/` comes from the
official repository https://github.com/snap-stanford/RelGNN (commit `cffdb8b`, MIT License). Every file there
names its source file and its (minimal) changes.

## Data Source

All data comes from **GH Archive** (https://www.gharchive.org/), a public archive of GitHub events with one file
per hour. We acknowledge GH Archive as the data source of this project.

- **Period used:** the GH Archive hour files `2026-03-01-0.json.gz` to `2026-11-20-15.json.gz`, i.e. all events
  from 2026-03-01 00:00 to 2026-11-20 16:00 UTC (= 2026-03-01 08:00 to 2026-11-21 00:00 Taiwan time).
- The hour files are turned into the 8 anonymized tables by the course script `scripts/preprocess.py`, run
  unmodified. Only these 8 tables are used.

## How to run

```bash
# 1. course repository: preprocess.py, test_input.csv, evaluate.py
git clone https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2

# 2. GH Archive hour files 2026-03-01-0 ... 2026-11-20-15 (UTC): 6,352 files, roughly 170 GB
python src/download.py --raw gharchive

# 3. the 8 tables (course script, unmodified): writes data/db/*.parquet
python assignment2/scripts/preprocess.py --raw gharchive --out data

# 4. train RelGNN and predict: writes ./output_414540015.csv
python src/predict.py --data data --test_input assignment2/submission/test_input.csv --out .

# 5. format check
python assignment2/scripts/evaluate.py --pred output_414540015.csv --input assignment2/submission/test_input.csv

# optional: time-split validation on a past week; also writes the week's prediction and answer files,
# which the course evaluate.py scores with --pred ... --truth ...
python src/validate.py --data data --weeks "2026-10-30 16:00" --out validation
```

Step 2 can also be done with the course `wget` command (README of the course repository), as long as the same
hour files end up in one folder.

## Reproducibility

- `src/predict.py` trains 2 RelGNN models with the fixed seeds 0 and 1 and runs PyTorch in deterministic mode
  (`torch.use_deterministic_algorithms` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`). On our machine (RTX 3090,
  CUDA 12.4, the versions below), two training runs with the same seed gave identical predictions.
- A different GPU or different library versions can change the last digits of the floating-point results, and
  so a few 0/1 decisions of borderline users.
- The output depends on the hour files processed: use exactly 2026-03-01-0 to 2026-11-20-15 (UTC).

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

## Method

1. **Relational entity graph** (`src/graph.py`, built with RelBench's `make_pkey_fkey_graph`):
   - Each of the 8 tables is a node type and each row is a node.
   - Each foreign key gives an edge type plus its reverse.
   - Node features are the table's own columns, encoded with PyTorch Frame.
   - Each node's time is the table's row-time column.
2. **Task table** (`src/task.py`):
   - Seeds are (user, cutoff *t*) for daily cutoffs at 16:00 UTC (00:00 Taiwan) from 2026-06-15 on (the first
     cutoff whose user-selection window lies inside the data).
   - Users are chosen with the test-user rule: cohort users with ≥ 3 activity records in the 90 days before
     *t* − 16 days 4 hours. At the test seed time this is 2026-11-04 12:00 UTC, and the rule gives exactly the
     4,465 users of `test_input.csv`.
   - Labels are the 6 activities in [*t* + 1 day, *t* + 8 days). For the test, *t* = 2026-11-20 16:00 UTC, so
     the label week is 11/22 00:00 – 11/29 00:00 Taiwan.
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
| `src/download.py` | Downloads the GH Archive hour files of the period used |
| `src/data.py` | Reads the 8 tables; cohort activity log; test-user rule; labels |
| `src/graph.py` | Database → relational entity graph; leakage rules |
| `src/task.py` | (user, cutoff) seeds with labels; time-based train / validation split |
| `src/train.py` | RelGNN model, temporal neighbour loader, training loop, inference |
| `src/decision.py` | Official score; plug-in expected-F1 decision |
| `src/validate.py` | Time-split validation on past weeks |
| `src/predict.py` | Final prediction → `output_<student_id>.csv` |
| `src/relgnn/` | Code from the official RelGNN repository (atomic routes, RelGNN conv / hetero conv / model, RelBench graph builder) |

## Data rules followed

* **Only the 8 tables built by the course `preprocess.py` are used**, from the GH Archive period above.
* **Known-later columns are never used:** `PullRequests.merged_at`, `merged_by_id`, `closed_at`,
  `Issues.closed_at`, `state_reason` and `PRReviews.dismissed_at`. They are not node features, and
  `merged_by_id` is not used as a foreign key.
* **Every row only becomes visible at its row time:** `first_seen_at` for Users, Repositories, PullRequests,
  Issues and PRReviews, and `created_at` for IssueComments, PRReviewComments and Pushes. A seed at time *t*
  only samples rows with row time < *t*.
* **Split by time, not at random.** Training seeds have label weeks that end before the validation week, and
  the validation week ends before the predicted week. The final model only uses weeks that have ended in the
  data.
* Codes, ids, masked text and absolute timestamps are not used as features.
