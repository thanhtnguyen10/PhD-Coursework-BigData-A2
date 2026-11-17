# BigData 2026 Fall – Assignment 2: Predicting GitHub User Activity

For the 4,465 users in the course file `submission/test_input.csv`, predict whether each user will `open_pr`,
`open_issue`, `issue_comment`, `pr_review`, `pr_review_comment` and `push` during
**2026/11/22 00:00 – 11/29 00:00 (Taiwan time)**.

**Model: RelGNN** (Chen, Kanatsoulis & Leskovec, ICML 2025). `src/relgnn/` is code from the official repository
https://github.com/snap-stanford/RelGNN (commit `cffdb8b`, MIT License); each file names its source.

## Data Source

All data comes from **GH Archive** (https://www.gharchive.org/), a public archive of GitHub events.

- Period: hour files `2026-03-01-0` to `2026-11-18-15` (UTC) = 2026-03-01 08:00 to 2026-11-19 00:00 Taiwan time.
- The files are turned into the 8 tables by the course `scripts/preprocess.py` (unmodified). Nothing else is used.

## How to run

```bash
git clone https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2  # course scripts
python src/download.py --raw gharchive                               # 6,304 hour files, more than 100 GB
python assignment2/scripts/preprocess.py --raw gharchive --out data  # the 8 tables
python src/predict.py --data data --test_input assignment2/submission/test_input.csv --out .  # output_414540015.csv
python assignment2/scripts/evaluate.py --pred output_414540015.csv --input assignment2/submission/test_input.csv
```

`predict.py` takes about 2.5 hours on one RTX 3090 (2 models; peak GPU memory about 11 GiB).
Optional: `python src/validate.py --data data --weeks "2026-10-30 16:00" --out validation` checks the method on a
past week (with a 1-day gap).

## Environment

Python 3.11, PyTorch 2.6 (CUDA 12.4), PyTorch Geometric 2.8.1, pyg-lib 0.4.0, PyTorch Frame 0.2.3, pandas,
pyarrow, scikit-learn.

```bash
conda create -n bigdata_a2 python=3.11 -y && conda activate bigdata_a2
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric==2.8.1 pandas pyarrow scikit-learn
pip install --no-deps pytorch-frame==0.2.3
pip install --no-deps "pyg-lib==0.4.0+pt26cu124" -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

## Method

1. **Graph** (`graph.py`): each row is a node and each foreign key an edge type (RelBench `make_pkey_fkey_graph`);
   node time = row time.
2. **Seeds** (`data.py`, `task.py`): (user, daily cutoff *t* at 00:00 Taiwan, from 06/16). Users follow the
   test-user rule: ≥ 3 activity records in the 90 days before the label-week start − 17 d 4 h, which gives exactly
   the 4,465 test users. Labels: the 6 activities in [*t* + 3 d, *t* + 10 d), the same 3-day gap as the test week
   (data end 11/19 → test week 11/22).
3. **RelGNN** (`train.py`): 42 atomic routes, composite message passing, relative-time encoding; temporal sampling of
   [128, 64] neighbours, rows before *t* only. BCE loss, Adam lr 0.005, 10 epochs × 500 steps; the best epoch by
   macro AUPRC on the latest labelled week is kept.
4. **Decision** (`decision.py`): average of 2 models; per activity, the top-*m* users that maximise the expected
   (macro-F1 + micro-F1) / 2.

## Data rules

- Only the 8 tables from the course `preprocess.py`.
- Known-later columns (`merged_at`, `merged_by_id`, `closed_at`, `state_reason`, `dismissed_at`) are never used.
- A seed at time *t* only sees rows with row time < *t*. Training and validation are split by time, not at random.

## Reproducibility

Fixed seeds (0, 1) and deterministic PyTorch: a test run repeated with the same seed gave identical predictions on
our RTX 3090. Another GPU or other library versions may change a few borderline 0/1 decisions. Use exactly the hour
files listed above.
