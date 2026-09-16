import math

import pandas as pd

from reorder_radar import eval as eval_mod


def test_per_user_metrics_hand_calculated():
    user_df = pd.DataFrame({
        "label": [1, 0, 1, 0],
        "score": [0.9, 0.8, 0.7, 0.1],
    })
    m = eval_mod.per_user_metrics(user_df, k_list=(10,))
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(m["ndcg@10"], dcg / idcg, rel_tol=1e-9)
    assert m["recall@10"] == 1.0


def test_per_user_metrics_none_for_zero_positives():
    user_df = pd.DataFrame({"label": [0, 0, 0], "score": [0.5, 0.4, 0.1]})
    assert eval_mod.per_user_metrics(user_df) is None


def test_evaluate_excludes_zero_positive_users_and_averages():
    rows = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3],
        "label": [1, 0, 0, 0, 0, 1],
        "score": [0.9, 0.1, 0.9, 0.1, 0.1, 0.9],
    })
    result = eval_mod.evaluate(rows, "score", k_list=(10,))
    assert result["n_users_evaluated"] == 2  # user 2 has no positive, excluded
    assert result["n_users_excluded_no_positive"] == 1
    # user1: label at rank1 -> ndcg=1.0; user3: label at rank1 (score 0.9 highest) -> ndcg=1.0
    assert math.isclose(result["ndcg@10"], 1.0, rel_tol=1e-9)
    assert math.isclose(result["recall@10"], 1.0, rel_tol=1e-9)


def test_baseline_score_orders_by_times_bought_then_reorder_rate():
    from reorder_radar import model

    rows = pd.DataFrame({
        "times_bought": [5, 5, 2],
        "reorder_rate": [0.2, 0.8, 0.9],
    })
    scores = model.baseline_score(rows)
    # same times_bought (5) but higher reorder_rate (row1) must score higher than row0
    assert scores[1] > scores[0]
    # higher times_bought always dominates a full-range reorder_rate difference
    assert scores[0] > scores[2]
