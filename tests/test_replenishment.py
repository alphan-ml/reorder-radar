import pandas as pd
import pytest

from reorder_radar.replenishment import (
    EXTRA_FEATURES,
    add_history_features,
    history_features,
)


def examples():
    orders = pd.DataFrame([
        {"order_id": 101, "user_id": 1, "eval_set": "prior", "order_number": 1},
        {"order_id": 102, "user_id": 1, "eval_set": "prior", "order_number": 2},
        {"order_id": 103, "user_id": 1, "eval_set": "prior", "order_number": 3},
        {"order_id": 104, "user_id": 1, "eval_set": "prior", "order_number": 4},
        {"order_id": 105, "user_id": 1, "eval_set": "train", "order_number": 5},
        {"order_id": 201, "user_id": 2, "eval_set": "prior", "order_number": 1},
    ])
    lines = pd.DataFrame([
        {"order_id": 101, "product_id": 10},
        {"order_id": 102, "product_id": 20},
        {"order_id": 103, "product_id": 10},
        {"order_id": 104, "product_id": 10},
        {"order_id": 104, "product_id": 20},
        {"order_id": 201, "product_id": 20},
    ])
    products = pd.DataFrame([
        {"product_id": 10, "department_id": 1},
        {"product_id": 20, "department_id": 2},
    ])
    return orders, lines, products


def test_past_only_gap_and_department_shares():
    orders, lines, products = examples()
    result = history_features(orders, lines, products, {1})
    assert set(result["user_id"]) == {1}
    assert set(EXTRA_FEATURES).issubset(result.columns)
    x = result.set_index("product_id")
    assert x.loc[10, "mean_product_reorder_gap_orders"] == pytest.approx(1.5)
    assert x.loc[10, "user_department_history_share"] == pytest.approx(3 / 5)
    assert x.loc[20, "user_department_history_share"] == pytest.approx(2 / 5)
    assert x.loc[20, "replenishment_due_ratio"] == pytest.approx(0)


def test_future_train_order_cannot_enter_replenishment_features():
    orders, lines, products = examples()
    original = history_features(orders, lines, products, {1})
    # A final labeled order added to the orders table is ignored.
    new_order = pd.DataFrame([
        {"order_id": 106, "user_id": 1, "eval_set": "train", "order_number": 6}
    ])
    amended = pd.concat([orders, new_order], ignore_index=True)
    pd.testing.assert_frame_equal(original, history_features(amended, lines, products, {1}))


def test_add_history_rejects_duplicate_candidate_keys():
    base = pd.DataFrame([{"user_id": 1, "product_id": 10, "label": 1}])
    extra = pd.DataFrame([{"user_id": 1, "product_id": 10}] * 2)
    with pytest.raises(ValueError):
        add_history_features(base, extra)
