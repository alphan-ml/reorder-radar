import pandas as pd

from reorder_radar import surrogate


def _orders_for_user(user_id, n_orders, gap_days):
    """n_orders orders, each `gap_days` apart, order_number starting at 1."""
    rows = []
    for i in range(n_orders):
        rows.append({
            "order_id": user_id * 1000 + i,
            "user_id": user_id,
            "order_number": i + 1,
            "days_since_prior_order": None if i == 0 else float(gap_days),
        })
    return rows


def test_build_timeline_cumulative_days():
    orders = pd.DataFrame(_orders_for_user(1, 4, 10.0))
    tl = surrogate.build_timeline(orders)
    days = tl.sort_values("order_number")["cum_day"].tolist()
    assert days == [0.0, 10.0, 20.0, 30.0]


def test_build_dataset_scope_and_window_counts():
    # user 1: 40 orders, 10 days apart -> spans 390 days, in scope (>=365)
    # user 2: 5 orders, 10 days apart -> spans 40 days, out of scope
    rows = _orders_for_user(1, 40, 10.0) + _orders_for_user(2, 5, 10.0)
    orders = pd.DataFrame(rows)
    timeline = surrogate.build_timeline(orders)

    op_prior = pd.DataFrame({"order_id": [o["order_id"] for o in rows], "product_id": [1] * len(rows)})
    op_train = pd.DataFrame({"order_id": [], "product_id": []}).astype({"order_id": "int64", "product_id": "int64"})

    dataset = surrogate.build_dataset(timeline, op_prior, op_train)
    assert set(dataset["user_id"]) == {1}

    row = dataset[dataset["user_id"] == 1].iloc[0]
    # orders at day 10,20,30 fall within (0, 30] -> 3 orders in the 30-day window
    assert row["orders_30d"] == 3
    # orders at day 10..360 (36 orders) fall within (0, 365]
    assert row["orders_365d"] == 36
