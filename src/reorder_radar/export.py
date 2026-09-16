"""Write outputs/site_data.json: dataset facts, split sizes, metrics for
both rankers, surrogate-value results, and 10 anonymized example rankings.
Every number here is read back from files earlier pipeline steps wrote --
nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reorder_radar import model as model_mod

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"


def _load_json(name: str) -> dict:
    return json.loads((OUT_DIR / name).read_text())


def _example_rankings(rows: pd.DataFrame, products: pd.DataFrame, test_ids: set, n: int = 10) -> list:
    booster = model_mod.load_lambdarank()
    test_rows = rows[rows["user_id"].isin(test_ids)].copy()
    test_rows["lambdarank_score"] = model_mod.predict_lambdarank(booster, test_rows)
    test_rows["baseline_score"] = model_mod.baseline_score(test_rows)

    name_map = dict(zip(products["product_id"], products["product_name"]))

    rng = np.random.RandomState(26)
    candidate_users = (
        test_rows.groupby("user_id")["label"].sum().loc[lambda s: s > 0].index.to_numpy()
    )
    chosen = rng.choice(candidate_users, size=min(n, len(candidate_users)), replace=False)
    chosen.sort()

    examples = []
    for i, uid in enumerate(chosen):
        g = test_rows[test_rows["user_id"] == uid].sort_values("lambdarank_score", ascending=False)
        top = g.head(10)
        examples.append({
            "anonymized_user": f"holdout_user_{i + 1:02d}",
            "n_candidate_products": len(g),
            "n_actually_reordered": int(g["label"].sum()),
            "top_10_ranked_by_lambdarank": [
                {
                    "product_name": str(name_map.get(pid, f"product_{pid}")),
                    "lambdarank_score": round(float(s), 4),
                    "times_bought_before": int(t),
                    "actually_reordered": bool(lbl),
                }
                for pid, s, t, lbl in zip(
                    top["product_id"], top["lambdarank_score"], top["times_bought"], top["label"]
                )
            ],
        })
    return examples


def export() -> dict:
    data_quality = _load_json("data_quality.json")
    clean_stats = _load_json("clean_stats.json")
    split = _load_json("split_ids.json")
    train_info = _load_json("train_info.json")
    lambdarank_metrics = _load_json("metrics_lambdarank.json")
    baseline_metrics = _load_json("metrics_baseline.json")
    surrogate_results = _load_json("surrogate_results.json")

    rows = pd.read_parquet(OUT_DIR / "candidate_rows.parquet")
    products = pd.read_csv(ROOT / "data" / "raw" / "products.csv", usecols=["product_id", "product_name"])
    test_ids = set(split["test_ids"])

    site_data = {
        "dataset": {
            "name": "Instacart Market Basket Analysis",
            "source": "Kaggle dataset mirror (see data/ATTRIBUTION.md)",
            "raw_row_counts": data_quality["files"],
            "distinct_users": data_quality["distinct_users"],
            "distinct_products": data_quality["distinct_products"],
            "users_with_train_final_order": data_quality["users_with_train_final_order"],
            "users_with_test_final_order": data_quality["users_with_test_final_order"],
        },
        "clean_stats": clean_stats,
        "split": {
            "seed": split["seed"],
            "n_train_users": len(split["train_ids"]),
            "n_val_users": len(split["val_ids"]),
            "n_test_users": len(split["test_ids"]),
        },
        "candidate_rows": {
            "total_rows": len(rows),
            "positive_label_rows": int(rows["label"].sum()),
        },
        "train_info": train_info,
        "metrics": {
            "lambdarank": lambdarank_metrics,
            "baseline_buy_it_again_by_frequency": baseline_metrics,
        },
        "surrogate_value_module": surrogate_results,
        "example_rankings": _example_rankings(rows, products, test_ids, n=10),
    }

    (OUT_DIR / "site_data.json").write_text(json.dumps(site_data, indent=2))
    return site_data


if __name__ == "__main__":
    export()
    print(f"wrote {OUT_DIR / 'site_data.json'}")
