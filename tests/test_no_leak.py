"""Proves build_features cannot see the final ('train') order's contents or
timing: build_features's signature does not accept op_train at all, and the
computed features match hand-calculated values derived from prior-only data
-- in particular the final order's own days_since_prior_order (3.0 for
user 1's order 3) never enters days_since_last_bought / user_last_day.
"""
import inspect

import pandas as pd

from reorder_radar import features


def _synthetic():
    orders = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "user_id": [1, 1, 1, 2, 2],
        "eval_set": ["prior", "prior", "train", "prior", "train"],
        "order_number": [1, 2, 3, 1, 2],
        "order_dow": [0, 1, 2, 0, 1],
        "order_hour_of_day": [8, 9, 10, 8, 9],
        "days_since_prior_order": [None, 5.0, 3.0, None, 7.0],
    })
    op_prior = pd.DataFrame({
        "order_id": [1, 2, 2, 4],
        "product_id": [100, 100, 101, 200],
        "add_to_cart_order": [1, 1, 2, 1],
        "reordered": [0, 1, 0, 0],
    })
    op_train = pd.DataFrame({
        "order_id": [3, 3, 5],
        "product_id": [100, 999, 300],
        "add_to_cart_order": [1, 2, 1],
        "reordered": [1, 0, 0],
    })
    products = pd.DataFrame({
        "product_id": [100, 101, 200, 300, 999],
        "product_name": ["a", "b", "c", "d", "e"],
        "aisle_id": [1, 1, 2, 2, 3],
        "department_id": [1, 1, 2, 2, 3],
    })
    return orders, op_prior, op_train, products


def test_build_features_signature_excludes_op_train():
    params = inspect.signature(features.build_features).parameters
    assert "op_train" not in params
    assert set(params) == {"orders", "op_prior", "products"}


def test_features_use_only_prior_data():
    orders, op_prior, _op_train, products = _synthetic()
    feat = features.build_features(orders, op_prior, products)

    # candidate universe is exactly the products bought in prior orders --
    # product 999 (only in the final order) never appears as a candidate.
    assert set(feat["product_id"]) == {100, 101, 200}
    assert 999 not in set(feat["product_id"])

    row = feat[(feat["user_id"] == 1) & (feat["product_id"] == 100)].iloc[0]
    assert row["times_bought"] == 2  # orders 1 and 2 only, order 3 not counted
    assert row["reorder_rate"] == 0.5  # 1 reordered out of 2 prior purchases

    # user 1's prior timeline spans order1 (day 0) -> order2 (day 0+5=5).
    # order 3's own days_since_prior_order (3.0) is the final order's timing
    # and must NOT be added: user_last_day must stay at 5, not 5+3=8.
    assert row["days_since_last_bought"] == 0.0  # last bought at day 5, timeline ends at day 5

    row2 = feat[(feat["user_id"] == 1) & (feat["product_id"] == 101)].iloc[0]
    assert row2["times_bought"] == 1
    assert row2["days_since_last_bought"] == 0.0  # bought at order2 (day5) == end of prior timeline


def test_attach_labels_only_function_reading_op_train():
    orders, op_prior, op_train, products = _synthetic()
    feat = features.build_features(orders, op_prior, products)
    rows = features.attach_labels(feat, orders, op_train)

    labels = dict(zip(zip(rows["user_id"], rows["product_id"]), rows["label"]))
    assert labels[(1, 100)] == 1  # product 100 reappears in user 1's final order
    assert labels[(1, 101)] == 0  # product 101 does not
    assert labels[(2, 200)] == 0  # user 2's final order has a different product (300)

    # perturbing op_train changes only labels, never the features already computed
    op_train_corrupted = op_train.copy()
    op_train_corrupted["product_id"] = 0
    rows2 = features.attach_labels(feat, orders, op_train_corrupted)
    pd.testing.assert_frame_equal(
        rows.drop(columns=["label"]), rows2.drop(columns=["label"])
    )
    assert rows2["label"].sum() == 0


def test_split_is_seeded_and_disjoint():
    ids = list(range(1, 101))
    a1, b1, c1 = features.train_val_test_split(ids, seed=26)
    a2, b2, c2 = features.train_val_test_split(ids, seed=26)
    assert a1 == a2 and b1 == b2 and c1 == c2
    assert a1 & b1 == set() and b1 & c1 == set() and a1 & c1 == set()
    assert a1 | b1 | c1 == set(ids)
    assert len(a1) == 80 and len(b1) == 10 and len(c1) == 10
