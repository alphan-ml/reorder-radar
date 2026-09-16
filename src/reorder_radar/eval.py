"""NDCG@10, NDCG@20, recall@10, recall@20 for a ranker's scores against the
binary reorder label, averaged per user over users with at least one
positive (a user whose final order has zero products from their prior
history contributes no signal to ranking quality and is excluded from the
average, but counted separately).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_LIST = (10, 20)


def _dcg(labels_in_rank_order: np.ndarray, k: int) -> float:
    labels = labels_in_rank_order[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2))
    return float(np.sum(labels * discounts))


def _idcg(n_relevant: int, k: int) -> float:
    m = min(n_relevant, k)
    if m == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, m + 2))
    return float(np.sum(discounts))


def per_user_metrics(user_df: pd.DataFrame, k_list=K_LIST) -> dict | None:
    """user_df: rows for one user, columns 'score' and 'label'. Returns None
    if the user has zero positive labels (excluded from ranking averages).
    """
    n_relevant = int(user_df["label"].sum())
    if n_relevant == 0:
        return None
    order = np.argsort(-user_df["score"].to_numpy(dtype="float64"), kind="stable")
    labels_ranked = user_df["label"].to_numpy()[order]
    out = {}
    for k in k_list:
        dcg = _dcg(labels_ranked, k)
        idcg = _idcg(n_relevant, k)
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
        out[f"recall@{k}"] = float(labels_ranked[:k].sum()) / n_relevant
    return out


def evaluate(rows: pd.DataFrame, score_col: str, k_list=K_LIST) -> dict:
    """rows: user_id, label, and `score_col`. Returns averaged metrics plus
    the count of users evaluated (n_relevant > 0) and excluded (n_relevant == 0).
    """
    df = rows[["user_id", "label", score_col]].rename(columns={score_col: "score"})
    per_user = []
    n_excluded = 0
    for _uid, g in df.groupby("user_id", sort=False):
        m = per_user_metrics(g, k_list)
        if m is None:
            n_excluded += 1
        else:
            per_user.append(m)

    if not per_user:
        raise RuntimeError("no evaluated users had a positive label -- cannot compute metrics")

    result = {f"ndcg@{k}": float(np.mean([m[f"ndcg@{k}"] for m in per_user])) for k in k_list}
    result.update({f"recall@{k}": float(np.mean([m[f"recall@{k}"] for m in per_user])) for k in k_list})
    result["n_users_evaluated"] = len(per_user)
    result["n_users_excluded_no_positive"] = n_excluded
    return result
