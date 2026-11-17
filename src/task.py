"""Task table (RelBench style): one seed per (user, cutoff t), users by the test-user rule, 6 labels."""
import numpy as np
import pandas as pd

from data import GAP, HORIZON, TARGETS


def labelled_cutoffs(data, start):
    """Daily cutoffs from `start` whose label week has ended in the data."""
    return list(pd.date_range(start, data.data_end - data.gap - HORIZON, freq="1D"))


def task_table(data, cutoffs, users=None, with_labels=True):
    rows = []
    for t in cutoffs:
        u = data.select_users(t) if users is None else np.asarray(users, dtype=np.int64)
        y = data.labels(t, u) if with_labels else np.zeros((len(u), len(TARGETS)), np.int8)
        rows.append(pd.DataFrame({"user_id": u, "time": t, **{k: y[:, j] for j, k in enumerate(TARGETS)}}))
    return pd.concat(rows, ignore_index=True)


def split_by_time(cutoffs, v, gap=GAP):
    """Cutoffs whose label week ends by v: the last 7 select the epoch; training cutoffs' label weeks end before them."""
    known = [t for t in cutoffs if t + gap + HORIZON <= v]
    val = [t for t in known if t > known[-1] - pd.Timedelta(days=7)]
    train = [t for t in known if t + gap + HORIZON <= val[0]]
    return train, val
