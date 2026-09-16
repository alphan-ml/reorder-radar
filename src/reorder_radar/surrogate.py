"""Surrogate-value module.

For each user, reconstruct a relative timeline in days since their first
order (cumulative sum of `days_since_prior_order` across *all* of that
user's orders, prior + their final order -- this is a separate, self-
contained analysis of early-activity vs. long-run activity, not part of
the leakage-sensitive product-ranking task, so using the final order's
timing here is fine; the final order's *contents* are still never used).

Define the 30-day window after a user's first order and the following
12 months (365 days). Activity is measured as order count in each window.
Fit a small model (a two-leaf-deep LightGBM regressor) mapping 30-day
activity features to 12-month activity, and report Spearman correlation
on held-out users against the naive baseline: the raw 30-day order count
itself, used directly as the predictor (no model).

Known caveat: Instacart caps `days_since_prior_order` at 30, so a true gap
longer than 30 days is recorded as exactly 30.0 -- any user whose real
history has a gap over 30 days will have their reconstructed timeline
compressed for that stretch. Documented in data/ATTRIBUTION.md; not
correctable from this data.

Only users whose reconstructed timeline reaches at least 365 days are
in scope (a full 12-month window must actually be observable).
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

WINDOW_DAYS = 30
HORIZON_DAYS = 365


def build_timeline(orders: pd.DataFrame) -> pd.DataFrame:
    """All orders (prior + train + test) per user, chronologically ordered,
    with a cumulative day offset from that user's first order.
    """
    df = orders[["user_id", "order_id", "order_number", "days_since_prior_order"]].copy()
    df = df.sort_values(["user_id", "order_number"]).reset_index(drop=True)
    gap = df["days_since_prior_order"].fillna(0.0).astype("float32")
    df["cum_day"] = gap.groupby(df["user_id"]).cumsum()
    return df


def build_dataset(timeline: pd.DataFrame, op_prior: pd.DataFrame, op_train: pd.DataFrame) -> pd.DataFrame:
    """One row per user with 30-day window features and the 365-day target,
    restricted to users whose timeline spans >= 365 days.
    """
    span = timeline.groupby("user_id")["cum_day"].max()
    in_scope_users = span[span >= HORIZON_DAYS].index

    tl = timeline[timeline["user_id"].isin(in_scope_users)].copy()

    order_basket_size = pd.concat(
        [op_prior[["order_id", "product_id"]], op_train[["order_id", "product_id"]]]
    ).groupby("order_id").size().rename("basket_size")
    tl = tl.merge(order_basket_size, on="order_id", how="left")
    tl["basket_size"] = tl["basket_size"].fillna(0).astype("int32")

    is_30 = (tl["cum_day"] > 0) & (tl["cum_day"] <= WINDOW_DAYS)
    is_365 = (tl["cum_day"] > 0) & (tl["cum_day"] <= HORIZON_DAYS)

    orders_30d = tl[is_30].groupby("user_id").size().rename("orders_30d")
    products_30d = tl[is_30].groupby("user_id")["basket_size"].sum().rename("products_30d")
    mean_basket_30d = tl[is_30].groupby("user_id")["basket_size"].mean().rename("mean_basket_30d")
    orders_365d = tl[is_365].groupby("user_id").size().rename("orders_365d")

    out = pd.DataFrame(index=pd.Index(sorted(in_scope_users), name="user_id"))
    out = out.join(orders_30d).join(products_30d).join(mean_basket_30d).join(orders_365d)
    out = out.fillna(0.0).reset_index()
    out["orders_30d"] = out["orders_30d"].astype("int32")
    out["products_30d"] = out["products_30d"].astype("int32")
    out["mean_basket_30d"] = out["mean_basket_30d"].astype("float32")
    out["orders_365d"] = out["orders_365d"].astype("int32")
    return out


FEATURE_COLS = ["orders_30d", "products_30d", "mean_basket_30d"]
TARGET_COL = "orders_365d"


def fit_and_evaluate(df: pd.DataFrame, train_ids: set, test_ids: set) -> dict:
    import lightgbm as lgb

    train_df = df[df["user_id"].isin(train_ids)]
    test_df = df[df["user_id"].isin(test_ids)]

    model = lgb.LGBMRegressor(
        n_estimators=50, max_depth=3, num_leaves=7, learning_rate=0.1,
        min_child_samples=100, num_threads=4, random_state=26, verbose=-1,
    )
    model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])

    pred = model.predict(test_df[FEATURE_COLS])
    actual = test_df[TARGET_COL].to_numpy()
    baseline_pred = test_df["orders_30d"].to_numpy()

    model_corr, model_p = spearmanr(pred, actual)
    baseline_corr, baseline_p = spearmanr(baseline_pred, actual)

    return {
        "n_users_in_scope": len(df),
        "n_train_users": len(train_df),
        "n_test_users": len(test_df),
        "model_spearman": float(model_corr),
        "model_spearman_p": float(model_p),
        "baseline_30day_count_spearman": float(baseline_corr),
        "baseline_30day_count_spearman_p": float(baseline_p),
        "window_days": WINDOW_DAYS,
        "horizon_days": HORIZON_DAYS,
        "feature_cols": FEATURE_COLS,
    }, model
