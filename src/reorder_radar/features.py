"""Build the (user, product) candidate feature table for the reorder-ranking
task, and the user-level 80/10/10 split.

Task: for each user's final ("train") order, rank the products that user
bought before by probability of reorder. The candidate universe for a user
is every product that user bought in one of their *prior* orders (eval_set
== "prior"); the label is whether that product appears in the user's final
("train") order.

Leakage discipline: `build_features` takes only `orders` and `op_prior`
(plus `products` for aisle/department lookups) -- it never receives
`op_train`, so the final order's contents cannot influence a single
feature by construction. Labels are attached separately by `attach_labels`.
Every feature besides `days_since_prior_order`-derived timing uses only
rows from `op_prior`, i.e. only orders that are, for every user, strictly
earlier than that user's final order.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd

SEED = 26


def train_final_user_ids(orders: pd.DataFrame) -> pd.Series:
    """User ids whose final (most recent) order is a labeled 'train' order."""
    return orders.loc[orders["eval_set"] == "train", "user_id"].drop_duplicates()


def build_features(orders: pd.DataFrame, op_prior: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (user_id, product_id) candidate with past-only features.

    Restricted to users whose final order is labeled ('train' eval_set) --
    the only users this repo can score.
    """
    valid_users = set(train_final_user_ids(orders).tolist())

    prior_orders = orders.loc[
        (orders["eval_set"] == "prior") & (orders["user_id"].isin(valid_users)),
        ["order_id", "user_id", "order_number", "order_dow", "order_hour_of_day",
         "days_since_prior_order"],
    ].sort_values(["user_id", "order_number"]).reset_index(drop=True)

    # cumulative day offset of each prior order relative to that user's own
    # first prior order (order_number == 1, whose days_since_prior_order is
    # NaN and contributes 0). This is a *relative*, prior-history-only clock
    # -- it never uses the final order's timing.
    gap = prior_orders["days_since_prior_order"].fillna(0.0).astype("float32")
    prior_orders["cum_day"] = gap.groupby(prior_orders["user_id"]).cumsum()

    user_last_day = prior_orders.groupby("user_id")["cum_day"].max()
    user_order_count = prior_orders.groupby("user_id")["order_number"].max()
    user_mean_days_between = prior_orders.groupby("user_id")["days_since_prior_order"].mean()

    op = op_prior.merge(
        prior_orders[["order_id", "user_id", "order_number", "cum_day"]],
        on="order_id", how="inner",
    )
    del prior_orders
    gc.collect()

    basket_size = op.groupby(["user_id", "order_id"]).size()
    user_mean_basket_size = basket_size.groupby("user_id").mean()
    del basket_size

    # per-(user,product) aggregates -- the candidate universe and its features
    grp = op.groupby(["user_id", "product_id"], sort=False)
    cand = grp.size().rename("times_bought").reset_index()
    cand["sum_reordered"] = grp["reordered"].sum().to_numpy()
    cand["reorder_rate"] = (cand["sum_reordered"] / cand["times_bought"]).astype("float32")
    cand["last_order_number"] = grp["order_number"].max().to_numpy()
    cand["last_cum_day"] = grp["cum_day"].max().to_numpy()
    cand["mean_add_to_cart_position"] = grp["add_to_cart_order"].mean().to_numpy().astype("float32")
    cand = cand.drop(columns=["sum_reordered"])

    # global, prior-only aisle/department reorder rates
    op_p = op[["product_id", "reordered"]].merge(
        products[["product_id", "aisle_id", "department_id"]], on="product_id", how="left"
    )
    aisle_reorder_rate = op_p.groupby("aisle_id")["reordered"].mean().rename("aisle_reorder_rate")
    department_reorder_rate = (
        op_p.groupby("department_id")["reordered"].mean().rename("department_reorder_rate")
    )
    del op, op_p
    gc.collect()

    cand = cand.merge(products[["product_id", "aisle_id", "department_id"]], on="product_id", how="left")
    cand = cand.merge(aisle_reorder_rate, on="aisle_id", how="left")
    cand = cand.merge(department_reorder_rate, on="department_id", how="left")

    cand["user_order_count"] = cand["user_id"].map(user_order_count).astype("int16")
    cand["orders_since_last_bought"] = (cand["user_order_count"] - cand["last_order_number"]).astype("int16")
    cand["user_last_day"] = cand["user_id"].map(user_last_day).astype("float32")
    cand["days_since_last_bought"] = (cand["user_last_day"] - cand["last_cum_day"]).astype("float32")
    cand["user_mean_basket_size"] = cand["user_id"].map(user_mean_basket_size).astype("float32")
    cand["user_mean_days_between_orders"] = (
        cand["user_id"].map(user_mean_days_between).astype("float32")
    )

    feature_cols = [
        "user_id", "product_id",
        "times_bought", "reorder_rate", "orders_since_last_bought", "days_since_last_bought",
        "mean_add_to_cart_position", "aisle_reorder_rate", "department_reorder_rate",
        "user_order_count", "user_mean_basket_size", "user_mean_days_between_orders",
    ]
    cand = cand[feature_cols].copy()
    for c in ["times_bought", "user_order_count"]:
        cand[c] = cand[c].astype("int32")
    for c in ["reorder_rate", "orders_since_last_bought", "days_since_last_bought",
              "mean_add_to_cart_position", "aisle_reorder_rate", "department_reorder_rate",
              "user_mean_basket_size", "user_mean_days_between_orders"]:
        cand[c] = cand[c].astype("float32")

    return cand.reset_index(drop=True)


def attach_labels(features: pd.DataFrame, orders: pd.DataFrame, op_train: pd.DataFrame) -> pd.DataFrame:
    """Attach the binary reorder label: 1 iff product_id appears in that
    user's final ('train') order. This is the *only* function in this repo
    that reads op_train alongside user-product candidates.
    """
    final_orders = orders.loc[orders["eval_set"] == "train", ["order_id", "user_id"]]
    final_products = op_train[["order_id", "product_id"]].merge(final_orders, on="order_id", how="inner")
    label_pairs = pd.MultiIndex.from_frame(final_products[["user_id", "product_id"]].drop_duplicates())

    out = features.copy()
    idx = pd.MultiIndex.from_frame(out[["user_id", "product_id"]])
    out["label"] = idx.isin(label_pairs).astype("int8")
    return out


def train_val_test_split(user_ids: np.ndarray, seed: int = SEED) -> tuple[set, set, set]:
    """80/10/10 split of user ids, seeded, disjoint, covering every id."""
    ids = np.array(sorted({int(u) for u in user_ids}))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    ids = ids[perm]
    n = len(ids)
    n_train = round(n * 0.8)
    n_val = round(n * 0.1)
    train_ids = set(ids[:n_train].tolist())
    val_ids = set(ids[n_train:n_train + n_val].tolist())
    test_ids = set(ids[n_train + n_val:].tolist())
    return train_ids, val_ids, test_ids
