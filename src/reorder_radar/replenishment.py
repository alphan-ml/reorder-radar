"""Prior-history-only replenishment features for a ranking challenger.

This is NOT imported by the deployed ranker. It may be trained/evaluated on
train/validation users, but requires final holdout evaluation and an updated
Lambda feature pipeline before any release.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXTRA_FEATURES = [
    "mean_product_reorder_gap_orders",
    "std_product_reorder_gap_orders",
    "replenishment_due_ratio",
    "user_department_history_share",
]


def history_features(
    orders: pd.DataFrame,
    op_prior: pd.DataFrame,
    products: pd.DataFrame,
    allowed_users: set,
) -> pd.DataFrame:
    """Only labeled users' historical 'prior' orders, never the final order."""
    prior_orders = orders.loc[
        (orders["eval_set"] == "prior") & orders["user_id"].isin(allowed_users),
        ["order_id", "user_id", "order_number"],
    ]
    history = op_prior[["order_id", "product_id"]].merge(prior_orders, on="order_id", how="inner")
    history = history.sort_values(["user_id", "product_id", "order_number"])
    key = ["user_id", "product_id"]
    history["gap_orders"] = history.groupby(key, sort=False)["order_number"].diff()
    grouped = history.groupby(key, sort=False)["gap_orders"]
    features = grouped.agg(
        mean_product_reorder_gap_orders="mean",
        std_product_reorder_gap_orders="std",
    ).reset_index()

    # Order counts, not calendar days, avoid guessing the final order's date.
    last_product_order = history.groupby(key, sort=False)["order_number"].max()
    last_user_order = prior_orders.groupby("user_id")["order_number"].max()
    features = features.merge(
        last_product_order.rename("last_product_order").reset_index(), on=key
    )
    features["last_user_order"] = features["user_id"].map(last_user_order)
    gap = features["mean_product_reorder_gap_orders"].replace(0, np.nan)
    features["replenishment_due_ratio"] = (
        (features["last_user_order"] - features["last_product_order"]) / gap
    ).clip(lower=0, upper=5)

    # Within-user category mix, computed exclusively from prior order lines.
    history = history.merge(
        products[["product_id", "department_id"]], on="product_id", how="left"
    )
    user_total = history.groupby("user_id").size()
    user_department = history.groupby(["user_id", "department_id"]).size()
    dept = (user_department / user_total.reindex(user_department.index.get_level_values("user_id")).to_numpy())
    dept = dept.rename("user_department_history_share").reset_index()
    features = features.merge(
        products[["product_id", "department_id"]], on="product_id", how="left"
    ).merge(dept, on=["user_id", "department_id"], how="left")
    return features[key + EXTRA_FEATURES].astype({
        feature: "float32" for feature in EXTRA_FEATURES
    })


def add_history_features(candidates: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if extra.duplicated(["user_id", "product_id"]).any():
        raise ValueError("multiple history feature rows for one candidate")
    out = candidates.merge(extra, on=["user_id", "product_id"], how="left", validate="many_to_one")
    # Insufficient prior purchases => no observed replenishment cycle.
    # LightGBM receives NaN, not an invented 0-day cycle.
    return out
