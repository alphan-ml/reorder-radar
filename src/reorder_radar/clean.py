"""Load and lightly clean the six raw Instacart CSVs into memory-efficient
dtypes. No rows are dropped from the Instacart tables on this dataset
because Kaggle-published counts confirm no fetch corruption occurred -- but
every table is still schema-checked here (not assumed clean) and null/
duplicate counts are recorded in clean_stats, matching the load-bearing
data-quality discipline used elsewhere in this pack.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def load_raw() -> dict[str, pd.DataFrame]:
    orders = pd.read_csv(
        RAW_DIR / "orders.csv",
        dtype={
            "order_id": "int32", "user_id": "int32", "eval_set": "category",
            "order_number": "int16", "order_dow": "int8", "order_hour_of_day": "int8",
            "days_since_prior_order": "float32",
        },
    )
    op_prior = pd.read_csv(
        RAW_DIR / "order_products__prior.csv",
        dtype={"order_id": "int32", "product_id": "int32", "add_to_cart_order": "int16",
               "reordered": "int8"},
    )
    op_train = pd.read_csv(
        RAW_DIR / "order_products__train.csv",
        dtype={"order_id": "int32", "product_id": "int32", "add_to_cart_order": "int16",
               "reordered": "int8"},
    )
    products = pd.read_csv(
        RAW_DIR / "products.csv",
        dtype={"product_id": "int32", "product_name": "string", "aisle_id": "int16",
               "department_id": "int8"},
    )
    aisles = pd.read_csv(RAW_DIR / "aisles.csv", dtype={"aisle_id": "int16", "aisle": "string"})
    departments = pd.read_csv(
        RAW_DIR / "departments.csv", dtype={"department_id": "int8", "department": "string"}
    )
    return {
        "orders": orders, "op_prior": op_prior, "op_train": op_train,
        "products": products, "aisles": aisles, "departments": departments,
    }


def clean(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    stats: dict = {}
    cleaned = {}

    orders = raw["orders"]
    n0 = len(orders)
    dup_order_ids = int(orders["order_id"].duplicated().sum())
    null_user_id = int(orders["user_id"].isna().sum())
    bad_eval_set = int((~orders["eval_set"].isin(["prior", "train", "test"])).sum())
    orders = orders[
        (~orders["order_id"].duplicated())
        & orders["user_id"].notna()
        & orders["eval_set"].isin(["prior", "train", "test"])
    ].copy()
    stats["orders"] = {
        "raw_rows": n0, "kept_rows": len(orders),
        "dropped_duplicate_order_id": dup_order_ids,
        "dropped_null_user_id": null_user_id,
        "dropped_bad_eval_set": bad_eval_set,
    }
    cleaned["orders"] = orders

    for key in ("op_prior", "op_train"):
        df = raw[key]
        n0 = len(df)
        null_ids = int(df["order_id"].isna().sum() + df["product_id"].isna().sum())
        bad_reordered = int((~df["reordered"].isin([0, 1])).sum())
        df = df[df["order_id"].notna() & df["product_id"].notna()
                & df["reordered"].isin([0, 1])].copy()
        stats[key] = {
            "raw_rows": n0, "kept_rows": len(df),
            "dropped_null_ids": null_ids, "dropped_bad_reordered": bad_reordered,
        }
        cleaned[key] = df

    products = raw["products"]
    n0 = len(products)
    dup_pid = int(products["product_id"].duplicated().sum())
    products = products[~products["product_id"].duplicated()].copy()
    stats["products"] = {"raw_rows": n0, "kept_rows": len(products), "dropped_duplicate_product_id": dup_pid}
    cleaned["products"] = products

    cleaned["aisles"] = raw["aisles"]
    cleaned["departments"] = raw["departments"]
    stats["aisles"] = {"raw_rows": len(raw["aisles"]), "kept_rows": len(raw["aisles"])}
    stats["departments"] = {"raw_rows": len(raw["departments"]), "kept_rows": len(raw["departments"])}

    return cleaned, stats
