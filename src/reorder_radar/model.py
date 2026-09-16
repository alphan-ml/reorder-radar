"""Two rankers for the reorder-ranking task:

(a) `fit_lambdarank` -- a LightGBM lambdarank model, grouped by user_id,
    trained on the 80% split and early-stopped on the 10% validation split.
(b) `baseline_score` -- a buy-it-again-by-frequency baseline: no fit, ranks
    a user's candidate products purely by how many times they were bought
    before (ties broken by reorder_rate), both computed from prior orders
    only.

Both operate on the row table produced by
`features.build_features` + `features.attach_labels`.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "outputs" / "checkpoints" / "lambdarank_model.txt"

FEATURE_COLS = [
    "times_bought", "reorder_rate", "orders_since_last_bought", "days_since_last_bought",
    "mean_add_to_cart_position", "aisle_reorder_rate", "department_reorder_rate",
    "user_order_count", "user_mean_basket_size", "user_mean_days_between_orders",
]

LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10, 20],
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_threads": 4,
    "seed": 26,
    "verbose": -1,
}


def _group_sizes(rows: pd.DataFrame) -> np.ndarray:
    # rows must already be sorted by user_id for lgb.Dataset's group param
    return rows.groupby("user_id", sort=False).size().to_numpy()


def fit_lambdarank(
    rows: pd.DataFrame, train_ids: set, val_ids: set, num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
) -> tuple[lgb.Booster, dict]:
    train_rows = rows[rows["user_id"].isin(train_ids)].sort_values("user_id").reset_index(drop=True)
    val_rows = rows[rows["user_id"].isin(val_ids)].sort_values("user_id").reset_index(drop=True)

    train_set = lgb.Dataset(
        train_rows[FEATURE_COLS], label=train_rows["label"], group=_group_sizes(train_rows),
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        val_rows[FEATURE_COLS], label=val_rows["label"], group=_group_sizes(val_rows),
        reference=train_set, free_raw_data=False,
    )

    evals_result: dict = {}
    booster = lgb.train(
        LGB_PARAMS, train_set, num_boost_round=num_boost_round, valid_sets=[val_set],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.record_evaluation(evals_result),
        ],
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_PATH))

    info = {
        "best_iteration": int(booster.best_iteration),
        "fit_users": len(train_ids),
        "early_stop_users": len(val_ids),
        "fit_rows": len(train_rows),
        "val_ndcg_at_10": float(evals_result["val"]["ndcg@10"][booster.best_iteration - 1]),
        "val_ndcg_at_20": float(evals_result["val"]["ndcg@20"][booster.best_iteration - 1]),
    }
    return booster, info


def load_lambdarank() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODEL_PATH))


def _num_iteration(booster: lgb.Booster) -> int | None:
    bi = getattr(booster, "best_iteration", None)
    return bi if bi and bi > 0 else None


def predict_lambdarank(booster: lgb.Booster, rows: pd.DataFrame) -> np.ndarray:
    return booster.predict(rows[FEATURE_COLS], num_iteration=_num_iteration(booster))


def baseline_score(rows: pd.DataFrame) -> np.ndarray:
    """Buy-it-again-by-frequency: rank by times_bought, tie-broken by
    reorder_rate. Both are computed from prior orders only, no fitting.
    """
    return (rows["times_bought"].to_numpy(dtype="float64") * 1000.0
            + rows["reorder_rate"].to_numpy(dtype="float64"))
