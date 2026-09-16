import pandas as pd

from reorder_radar import clean


def _raw():
    orders = pd.DataFrame({
        "order_id": [1, 2, 2, 3], "user_id": [10, 10, 10, None],
        "eval_set": ["prior", "prior", "prior", "prior"],
        "order_number": [1, 2, 2, 1], "order_dow": [0, 1, 1, 2],
        "order_hour_of_day": [8, 9, 9, 10], "days_since_prior_order": [None, 5.0, 5.0, None],
    })
    op_prior = pd.DataFrame({
        "order_id": [1, 1, 2, None, 3], "product_id": [100, 101, 100, 102, 103],
        "add_to_cart_order": [1, 2, 1, 1, 1], "reordered": [0, 0, 1, 0, 2],
    })
    op_train = op_prior.copy()
    products = pd.DataFrame({
        "product_id": [100, 100, 101], "product_name": ["a", "a", "b"],
        "aisle_id": [1, 1, 2], "department_id": [1, 1, 1],
    })
    aisles = pd.DataFrame({"aisle_id": [1, 2], "aisle": ["x", "y"]})
    departments = pd.DataFrame({"department_id": [1], "department": ["z"]})
    return {
        "orders": orders, "op_prior": op_prior, "op_train": op_train,
        "products": products, "aisles": aisles, "departments": departments,
    }


def test_clean_drops_bad_rows_and_reports_stats():
    raw = _raw()
    cleaned, stats = clean.clean(raw)

    # orders: row with null user_id dropped, duplicate order_id=2 dropped
    assert stats["orders"]["dropped_null_user_id"] == 1
    assert stats["orders"]["dropped_duplicate_order_id"] == 1
    assert len(cleaned["orders"]) == 2

    # op_prior: null order_id row dropped, bad reordered value (2) dropped
    assert stats["op_prior"]["dropped_null_ids"] == 1
    assert stats["op_prior"]["dropped_bad_reordered"] == 1
    assert len(cleaned["op_prior"]) == 3

    # products: duplicate product_id=100 dropped
    assert stats["products"]["dropped_duplicate_product_id"] == 1
    assert len(cleaned["products"]) == 2

    assert len(cleaned["aisles"]) == 2
    assert len(cleaned["departments"]) == 1
