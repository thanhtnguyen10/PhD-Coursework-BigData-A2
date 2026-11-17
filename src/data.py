"""The 8 tables from the course preprocess.py (<data_root>/db/*.parquet), the test-user rule and the labels."""
import os

import numpy as np
import pandas as pd

TARGETS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
GAP = pd.Timedelta(days=1)            # seed time t -> label-week start (default; predict.py uses 3 days)
HORIZON = pd.Timedelta(days=7)        # label-week length
# test-user rule (gives exactly test_input.csv): >= 3 records in the 90 days before label-week start - 17 d 4 h
SELECT_LEAD = pd.Timedelta(days=17, hours=4)
SELECT_WINDOW = pd.Timedelta(days=90)
SELECT_MIN = 3


class AssignmentData:
    """Cohort activity log: one row per activity, with `t` (activity time, for labels) and `seen` (row time)."""

    def __init__(self, data_root, gap=GAP):
        self.root = data_root
        self.gap = gap  # label week = [t + gap, t + gap + 7 d)
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
        # end of the data, rounded up to the hour
        self.data_end = max(icom.created_at.max(), prs.first_seen_at.max(), issues.first_seen_at.max(),
                            reviews.first_seen_at.max(), pushes.created_at.max()).ceil("h")

    def read(self, table, columns=None):
        return pd.read_parquet(os.path.join(self.root, "db", f"{table}.parquet"), columns=columns)

    def select_users(self, t):
        """Cohort users with >= 3 records in the 90 days before label-week start - 17 d 4 h (rows visible then)."""
        s = t + self.gap - SELECT_LEAD
        e = self.events
        e = e[(e.seen < s) & (e.t >= s - SELECT_WINDOW) & (e.t < s)]
        n = e.groupby("user_id").size()
        return np.sort(n.index[n >= SELECT_MIN].to_numpy())

    def labels(self, t, users):
        """1 if the user did the activity in [t + gap, t + gap + 7 d), else 0."""
        e = self.events
        e = e[(e.t >= t + self.gap) & (e.t < t + self.gap + HORIZON)]
        y = pd.crosstab(e.user_id, e.type).reindex(index=users, columns=range(len(TARGETS)), fill_value=0)
        return (y.to_numpy() > 0).astype(np.int8)
