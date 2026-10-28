"""Seeds of the entity-classification task: one row per (user, cutoff t) with the 6 labels of the following week.

This is the RelBench "task table" (entity = Users, time = seed time, multi-label target). Training seeds use
the official test design: users chosen by the test-user rule at t, labels in [t + 4 h, t + 4 h + 7 d).
"""
import numpy as np
import pandas as pd

from data import GAP, HORIZON, TARGETS


def labelled_cutoffs(data, start):
    """Daily cutoffs at 12:00 UTC (= 20:00 Taiwan, the daily update time) whose label week has ended in the data."""
    return list(pd.date_range(start, data.data_end - GAP - HORIZON, freq="1D"))


def task_table(data, cutoffs, users=None, with_labels=True):
    rows = []
    for t in cutoffs:
        u = data.select_users(t) if users is None else np.asarray(users, dtype=np.int64)
        y = data.labels(t, u) if with_labels else np.zeros((len(u), len(TARGETS)), np.int8)
        rows.append(pd.DataFrame({"user_id": u, "time": t, **{k: y[:, j] for j, k in enumerate(TARGETS)}}))
    return pd.concat(rows, ignore_index=True)


def split_by_time(cutoffs, v):
    """Split for predicting at cutoff v: usable cutoffs are those whose label week has ended by v; the last 7 of
    them form the validation week (epoch selection); training cutoffs have label weeks that end before it."""
    known = [t for t in cutoffs if t + GAP + HORIZON <= v]
    val = [t for t in known if t > known[-1] - pd.Timedelta(days=7)]
    train = [t for t in known if t + GAP + HORIZON <= val[0]]
    return train, val
