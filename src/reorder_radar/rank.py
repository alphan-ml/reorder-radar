"""Serving module: load the trained lambdarank model and a precomputed
holdout feature parquet, and return a user's top-10 products by predicted
reorder probability. No training or full-dataset load happens here --
this is what a live scoring endpoint would call per request.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import pandas as pd

from reorder_radar.model import FEATURE_COLS, MODEL_PATH, _num_iteration

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_FEATURES_PATH = ROOT / "outputs" / "holdout_features.parquet"


def load_model(model_path: Path = MODEL_PATH) -> lgb.Booster:
    return lgb.Booster(model_file=str(model_path))


def load_holdout_features(path: Path = HOLDOUT_FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def top_n_for_user(
    user_id: int, booster: lgb.Booster, holdout_features: pd.DataFrame, n: int = 10,
) -> list[dict]:
    """Return up to `n` (product_id, score) dicts, highest score first, for
    this user's candidate products. Empty list if the user has no rows in
    `holdout_features` (unknown user or no prior-purchase candidates).
    """
    rows = holdout_features[holdout_features["user_id"] == user_id]
    if rows.empty:
        return []
    scores = booster.predict(rows[FEATURE_COLS], num_iteration=_num_iteration(booster))
    ranked = (
        rows.assign(score=scores)
        .sort_values("score", ascending=False)
        .head(n)[["product_id", "score"]]
    )
    return [
        {"product_id": int(pid), "score": float(s)}
        for pid, s in zip(ranked["product_id"], ranked["score"])
    ]


if __name__ == "__main__":
    import sys

    uid = int(sys.argv[1])
    booster = load_model()
    feats = load_holdout_features()
    for r in top_n_for_user(uid, booster, feats):
        print(r)
