"""8 tables -> relational entity graph (RelBench make_pkey_fkey_graph, relgnn/rel_graph.py).

Row = node, foreign key = edge type (+ reverse), node time = row time (first_seen_at or created_at).
Known-later columns (merged_at, merged_by_id, closed_at, state_reason, dismissed_at) are not used at all;
ids, codes, masked text and absolute timestamps are not features.
"""
import pandas as pd
from torch_frame import stype

from relgnn.rel_graph import Database, Table, make_pkey_fkey_graph


def _cat(s):
    return s.astype("string").fillna("NA")


def _fk(s):
    return s.astype("float64")  # NaN = no link


def _num(s, fill):
    return s.fillna(fill).astype("float64")


def build_database(data):
    u = data.read("Users", ["user_id", "is_bot", "is_cohort", "first_seen_at"])
    r = data.read("Repositories", ["repo_id", "org_id", "first_seen_at"])
    p = data.read("PullRequests", ["pr_id", "repo_id", "author_id", "draft", "is_cross_repo", "first_seen_at"])
    i = data.read("Issues", ["issue_id", "repo_id", "author_id", "assignee_id", "pull_request_id", "is_pull_request",
                             "num_labels", "issue_type", "locked", "created_via_app", "first_seen_at"])
    c = data.read("IssueComments", ["comment_id", "issue_id", "repo_id", "author_id", "is_pull_request", "via_app", "created_at"])
    v = data.read("PRReviews", ["review_id", "pr_id", "repo_id", "reviewer_id", "state", "first_seen_at"])
    m = data.read("PRReviewComments", ["comment_id", "pr_id", "repo_id", "author_id", "pull_request_review_id",
                                       "in_reply_to_id", "file_extension", "subject_type", "position", "original_position", "created_at"])
    s = data.read("Pushes", ["push_id", "user_id", "repo_id", "ref_type", "is_new_ref", "created_at"])
    tables = {
        "Users": Table(pd.DataFrame({
            "user_id": u.user_id, "is_bot": _cat(u.is_bot), "is_cohort": _cat(u.is_cohort), "time": u.first_seen_at}),
            {}, "user_id", "time"),
        "Repositories": Table(pd.DataFrame({
            "repo_id": r.repo_id, "has_org": _cat(r.org_id.notna()), "time": r.first_seen_at}),
            {}, "repo_id", "time"),
        "PullRequests": Table(pd.DataFrame({
            "pr_id": p.pr_id, "repo_id": _fk(p.repo_id), "author_id": _fk(p.author_id), "draft": _cat(p.draft),
            "is_cross_repo": _cat(p.is_cross_repo), "time": p.first_seen_at}),
            {"repo_id": "Repositories", "author_id": "Users"}, "pr_id", "time"),
        "Issues": Table(pd.DataFrame({
            "issue_id": i.issue_id, "repo_id": _fk(i.repo_id), "author_id": _fk(i.author_id),
            "assignee_id": _fk(i.assignee_id), "pull_request_id": _fk(i.pull_request_id),
            "is_pull_request": _cat(i.is_pull_request), "num_labels": _num(i.num_labels, 0), "issue_type": _cat(i.issue_type),
            "locked": _cat(i.locked), "created_via_app": _cat(i.created_via_app), "time": i.first_seen_at}),
            {"repo_id": "Repositories", "author_id": "Users", "assignee_id": "Users", "pull_request_id": "PullRequests"},
            "issue_id", "time"),
        "IssueComments": Table(pd.DataFrame({
            "comment_id": c.comment_id, "issue_id": _fk(c.issue_id), "repo_id": _fk(c.repo_id), "author_id": _fk(c.author_id),
            "is_pull_request": _cat(c.is_pull_request), "via_app": _cat(c.via_app), "time": c.created_at}),
            {"issue_id": "Issues", "repo_id": "Repositories", "author_id": "Users"}, "comment_id", "time"),
        "PRReviews": Table(pd.DataFrame({
            "review_id": v.review_id, "pr_id": _fk(v.pr_id), "repo_id": _fk(v.repo_id), "reviewer_id": _fk(v.reviewer_id),
            "state": _cat(v.state), "time": v.first_seen_at}),
            {"pr_id": "PullRequests", "repo_id": "Repositories", "reviewer_id": "Users"}, "review_id", "time"),
        "PRReviewComments": Table(pd.DataFrame({
            "comment_id": m.comment_id, "pr_id": _fk(m.pr_id), "repo_id": _fk(m.repo_id), "author_id": _fk(m.author_id),
            "pull_request_review_id": _fk(m.pull_request_review_id), "in_reply_to_id": _fk(m.in_reply_to_id),
            "file_extension": _cat(m.file_extension), "subject_type": _cat(m.subject_type),
            "position": _num(m.position, -1), "original_position": _num(m.original_position, -1), "time": m.created_at}),
            {"pr_id": "PullRequests", "repo_id": "Repositories", "author_id": "Users",
             "pull_request_review_id": "PRReviews", "in_reply_to_id": "PRReviewComments"}, "comment_id", "time"),
        "Pushes": Table(pd.DataFrame({
            "push_id": s.push_id, "user_id": _fk(s.user_id), "repo_id": _fk(s.repo_id),
            "ref_type": _cat(s.ref_type), "is_new_ref": _cat(s.is_new_ref), "time": s.created_at}),
            {"user_id": "Users", "repo_id": "Repositories"}, "push_id", "time"),
    }
    col_to_stype = {}  # features: all columns except keys and row time
    for name, tab in tables.items():
        cols = [col for col in tab.df.columns if col not in (tab.pkey_col, tab.time_col, *tab.fkey_col_to_pkey_table)]
        col_to_stype[name] = {col: stype.numerical if tab.df[col].dtype.kind == "f" else stype.categorical for col in cols}
    return Database(tables), col_to_stype


def build_graph(data):
    """HeteroData graph and PyTorch Frame column statistics."""
    db, col_to_stype = build_database(data)
    return make_pkey_fkey_graph(db, col_to_stype_dict=col_to_stype)
