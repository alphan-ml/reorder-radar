import lightgbm as lgb
import pandas as pd
import pytest

from reorder_radar import rank
from reorder_radar.model import FEATURE_COLS


@pytest.fixture
def tiny_booster(tmp_path):
    import numpy as np

    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({c: rng.rand(n) for c in FEATURE_COLS})
    df["user_id"] = np.repeat(np.arange(20), 10)
    df["product_id"] = np.arange(n) + 1000
    label = (df["reorder_rate"] > 0.5).astype(int)
    group = df.groupby("user_id").size().to_numpy()
    train_set = lgb.Dataset(df[FEATURE_COLS], label=label, group=group)
    booster = lgb.train(
        {"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 1, "num_leaves": 7},
        train_set, num_boost_round=5,
    )
    return booster, df


def test_top_n_for_user_returns_sorted_scores(tiny_booster):
    booster, df = tiny_booster
    result = rank.top_n_for_user(0, booster, df, n=5)
    assert len(result) == 5
    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(r["product_id"], int) for r in result)


def test_top_n_for_user_unknown_user_returns_empty(tiny_booster):
    booster, df = tiny_booster
    assert rank.top_n_for_user(999999, booster, df) == []
