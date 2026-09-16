"""Day-1 data checks for the Instacart 6-file layout.

Reads the raw CSVs, validates the expected columns and row counts, and
writes outputs/data_quality.json. Every number here is measured from the
files on disk -- nothing is typed by hand.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "outputs" / "data_quality.json"

EXPECTED_COLUMNS = {
    "orders.csv": [
        "order_id", "user_id", "eval_set", "order_number",
        "order_dow", "order_hour_of_day", "days_since_prior_order",
    ],
    "order_products__prior.csv": ["order_id", "product_id", "add_to_cart_order", "reordered"],
    "order_products__train.csv": ["order_id", "product_id", "add_to_cart_order", "reordered"],
    "products.csv": ["product_id", "product_name", "aisle_id", "department_id"],
    "aisles.csv": ["aisle_id", "aisle"],
    "departments.csv": ["department_id", "department"],
}


def check() -> dict:
    result: dict = {"files": {}}

    for fname, cols in EXPECTED_COLUMNS.items():
        path = RAW_DIR / fname
        df = pd.read_csv(path, nrows=5)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{fname}: missing expected columns {missing}")
        with open(path, "rb") as fh:
            n_rows = sum(1 for _ in fh) - 1
        result["files"][fname] = {"row_count": n_rows, "columns": list(df.columns)}

    orders = pd.read_csv(
        RAW_DIR / "orders.csv",
        dtype={
            "order_id": "int32", "user_id": "int32", "eval_set": "category",
            "order_number": "int16", "order_dow": "int8", "order_hour_of_day": "int8",
            "days_since_prior_order": "float32",
        },
    )
    eval_set_counts = orders["eval_set"].value_counts().to_dict()
    result["orders_eval_set_counts"] = {str(k): int(v) for k, v in eval_set_counts.items()}
    result["distinct_users"] = int(orders["user_id"].nunique())

    # every user has exactly one 'train' or 'test' order (their most recent);
    # the rest are 'prior'. Users with a 'train' final order are the ones we
    # have ground-truth labels for and are the only ones in scope for the
    # ranking task.
    n_train_users = int((orders["eval_set"] == "train").sum())
    n_test_users = int((orders["eval_set"] == "test").sum())
    result["users_with_train_final_order"] = n_train_users
    result["users_with_test_final_order"] = n_test_users

    days_since_prior_capped_at_30 = int((orders["days_since_prior_order"] == 30.0).sum())
    result["days_since_prior_order_eq_30_count"] = days_since_prior_capped_at_30
    result["days_since_prior_order_note"] = (
        "Instacart caps days_since_prior_order at 30; a true gap longer than "
        "30 days is recorded as exactly 30.0, so this count over-includes "
        "genuine 30-day gaps."
    )

    op_prior_order_ids = pd.read_csv(RAW_DIR / "order_products__prior.csv", usecols=["order_id"])
    prior_order_ids = set(op_prior_order_ids["order_id"].unique())
    orders_prior_ids = set(orders.loc[orders["eval_set"] == "prior", "order_id"])
    result["order_products_prior_orders_match_orders_csv"] = prior_order_ids == orders_prior_ids

    products = pd.read_csv(RAW_DIR / "products.csv", usecols=["product_id"])
    result["distinct_products"] = int(products["product_id"].nunique())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    r = check()
    print(f"distinct_users: {r['distinct_users']:,}")
    print(f"users_with_train_final_order: {r['users_with_train_final_order']:,}")
    print(f"users_with_test_final_order: {r['users_with_test_final_order']:,}")
    print(f"distinct_products: {r['distinct_products']:,}")
    print(f"order_products_prior_orders_match_orders_csv: "
          f"{r['order_products_prior_orders_match_orders_csv']}")
    print(f"\nwrote {OUT_PATH}", file=sys.stderr)
