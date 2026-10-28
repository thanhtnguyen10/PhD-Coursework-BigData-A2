"""Assignment data: the 8 Parquet tables, the cohort activity log, the test-user rule and the 6 labels.

`data_root` is the folder created by
    git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
(it contains data/db/*.parquet and evaluation/).
"""
import os

import numpy as np
import pandas as pd

TARGETS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
GAP = pd.Timedelta(hours=4)           # last data update (20:00 Taiwan) -> start of the predicted week (00:00 Taiwan)
HORIZON = pd.Timedelta(days=7)        # length of the predicted week
SELECT_LAG = pd.Timedelta(days=14)    # test users are selected 14 days before the last data update ...
SELECT_WINDOW = pd.Timedelta(days=90)  # ... among cohort users with at least
SELECT_MIN = 5                        # 5 activity records in the 90 days before


class AssignmentData:
    """Cohort activity log: one row per target-type activity with two times,
    `t` = when the activity happened (labels) and `seen` = the row time (when the row exists in the data)."""

    def __init__(self, data_root):
        self.root = data_root
        users = self.read("Users", ["user_id", "github_user_id", "is_cohort"])
        prs = self.read("PullRequests", ["author_id", "created_at", "first_seen_at"])
        issues = self.read("Issues", ["author_id", "is_pull_request", "created_at", "first_seen_at"])
        icom = self.read("IssueComments", ["author_id", "created_at"])
        reviews = self.read("PRReviews", ["reviewer_id", "submitted_at", "first_seen_at"])
        rcom = self.read("PRReviewComments", ["author_id", "created_at"])
        pushes = self.read("Pushes", ["user_id", "created_at"])
        self.cohort = users.user_id[users.is_cohort].to_numpy()
        self.github_id = users.set_index("user_id")["github_user_id"]        # row id -> GitHub id
        self.row_id = pd.Series(self.github_id.index, index=self.github_id.to_numpy())  # GitHub id -> row id
        parts = [  # type k = index in TARGETS
            pd.DataFrame({"user_id": prs.author_id, "t": prs.created_at, "seen": prs.first_seen_at, "type": 0}),
            pd.DataFrame({"user_id": issues.author_id, "t": issues.created_at, "seen": issues.first_seen_at, "type": 1})[~issues.is_pull_request.to_numpy()],
            pd.DataFrame({"user_id": icom.author_id, "t": icom.created_at, "seen": icom.created_at, "type": 2}),
            pd.DataFrame({"user_id": reviews.reviewer_id, "t": reviews.submitted_at, "seen": reviews.first_seen_at, "type": 3}),
            pd.DataFrame({"user_id": rcom.author_id, "t": rcom.created_at, "seen": rcom.created_at, "type": 4}),
            pd.DataFrame({"user_id": pushes.user_id, "t": pushes.created_at, "seen": pushes.created_at, "type": 5}),
        ]
        ev = pd.concat(parts, ignore_index=True).dropna(subset=["user_id", "t"])
        ev = ev[ev.user_id.isin(set(self.cohort))]
        ev["user_id"] = ev.user_id.astype(np.int64)
        self.events = ev.sort_values("t").reset_index(drop=True)
        # end of the data in this release (the final release ends at 11/11 12:00 UTC = 20:00 Taiwan)
        self.data_end = max(icom.created_at.max(), prs.first_seen_at.max(), issues.first_seen_at.max(),
                            reviews.first_seen_at.max(), pushes.created_at.max())
        # pushes may lag behind the other tables in a release; the same lag is applied at every cutoff (graph.py)
        lag = self.data_end - pushes.created_at.max()
        self.push_lag = lag.ceil("h") if lag > pd.Timedelta(hours=3) else pd.Timedelta(0)

    def read(self, table, columns=None):
        return pd.read_parquet(os.path.join(self.root, "data", "db", f"{table}.parquet"), columns=columns)

    def select_users(self, t):
        """Test-user rule: cohort users with >= 5 activity records in the 90 days before t - 14 d."""
        s = t - SELECT_LAG
        e = self.events
        e = e[(e.seen < s) & (e.t >= s - SELECT_WINDOW) & (e.t < s)]
        n = e.groupby("user_id").size()
        return np.sort(n.index[n >= SELECT_MIN].to_numpy())

    def labels(self, t, users):
        """1 if the user did the activity in [t + 4 h, t + 4 h + 7 d), else 0."""
        e = self.events
        e = e[(e.t >= t + GAP) & (e.t < t + GAP + HORIZON)]
        y = pd.crosstab(e.user_id, e.type).reindex(index=users, columns=range(len(TARGETS)), fill_value=0)
        return (y.to_numpy() > 0).astype(np.int8)
