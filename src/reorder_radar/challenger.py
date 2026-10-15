"""Train a replenishment-aware ranking challenger without using the final test users.

Run after the existing features stage with the Instacart source files present:
    python -m reorder_radar.challenger
Only validation metrics are written. No deployment or automatic promotion.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

from reorder_radar import clean, model
from reorder_radar import eval as ranking_eval
from reorder_radar.replenishment import EXTRA_FEATURES, add_history_features, history_features

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"


def train_challenger(rows: pd.DataFrame, train_ids: set, val_ids: set) -> dict:
    if not train_ids or not val_ids or train_ids.intersection(val_ids):
        raise ValueError("training and validation users must be nonempty and disjoint")
    fit_ids, stop_ids = train_test_split(sorted(train_ids), test_size=0.12, random_state=26)
    columns = model.FEATURE_COLS + EXTRA_FEATURES

    def ordered(user_ids):
        return rows.loc[rows["user_id"].isin(user_ids)].sort_values("user_id").reset_index(drop=True)

    fit = ordered(set(fit_ids))
    stop = ordered(set(stop_ids))
    val = ordered(val_ids)
    if min(len(fit), len(stop), len(val)) == 0:
        raise ValueError("one of the challenger evaluation groups has no candidate rows")

    def dataset(frame, reference=None):
        return lgb.Dataset(
            frame[columns],
            label=frame["label"],
            group=frame.groupby("user_id", sort=False).size().to_numpy(),
            reference=reference,
            free_raw_data=False,
        )

    train_data = dataset(fit)
    early_data = dataset(stop, train_data)
    booster = lgb.train(
        model.LGB_PARAMS,
        train_data,
        num_boost_round=500,
        valid_sets=[early_data],
        valid_names=["early_stop"],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=0)],
    )
    val["challenger_score"] = booster.predict(val[columns], num_iteration=booster.best_iteration)
    val["freq_recency_score"] = model.baseline_score_freq_recency(val)
    challenger_metrics = ranking_eval.evaluate(val, "challenger_score")
    baseline_metrics = ranking_eval.evaluate(val, "freq_recency_score")
    return {
        "status": "validation_only; not deployed; test users untouched",
        "fit_users": len(fit_ids),
        "early_stop_users": len(stop_ids),
        "independent_validation_users": len(val_ids),
        "features_added": EXTRA_FEATURES,
        "best_iteration": booster.best_iteration,
        "challenger": challenger_metrics,
        "frequency_recency_baseline": baseline_metrics,
        "ndcg10_validation_relative_lift_vs_baseline": (
            challenger_metrics["ndcg@10"] / baseline_metrics["ndcg@10"] - 1
        ),
        "release_gate": "Select on validation, freeze the model, then evaluate on untouched test users; deployed Lambda must reproduce the added features.",
    }


def main() -> None:
    candidates_path = OUT_DIR / "candidate_rows.parquet"
    split_path = OUT_DIR / "split_ids.json"
    if not candidates_path.exists() or not split_path.exists():
        raise SystemExit("Run reorder-radar features stage with the original Instacart data first")
    split = json.loads(split_path.read_text())
    train, val, test = map(set, (split["train_ids"], split["val_ids"], split["test_ids"]))
    if (train & val) or (train & test) or (val & test):
        raise SystemExit("train, validation and test users are not disjoint")
    eligible = train | val

    rows = pd.read_parquet(candidates_path)
    rows = rows.loc[rows["user_id"].isin(eligible)].copy()
    raw = clean.load_raw()
    clean_tables, _ = clean.clean(raw)
    extra = history_features(
        clean_tables["orders"], clean_tables["op_prior"], clean_tables["products"], eligible,
    )
    rows = add_history_features(rows, extra)
    report = train_challenger(rows, train, val)
    report["holdout_users_excluded"] = len(test)
    report["holdout_users_scored"] = 0
    out = OUT_DIR / "replenishment_challenger_validation.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(out), "validation_lift": report["ndcg10_validation_relative_lift_vs_baseline"]}))


if __name__ == "__main__":
    main()
