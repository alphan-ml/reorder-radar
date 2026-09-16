"""reorder-radar fetch | check | clean | features | train | eval | surrogate | export | all

Each step is resumable: it skips when outputs/checkpoints/<step>.done
exists, unless --force is passed. One line per step, with row counts and
wall time, to stdout and outputs/run.log.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd

from reorder_radar import check as check_mod
from reorder_radar import clean as clean_mod
from reorder_radar import eval as eval_mod
from reorder_radar import features as features_mod
from reorder_radar import fetch as fetch_mod
from reorder_radar import model as model_mod
from reorder_radar import surrogate as surrogate_mod

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "checkpoints"
OUT_DIR = ROOT / "outputs"
LOG_PATH = OUT_DIR / "run.log"


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def _done(step: str) -> bool:
    return (CKPT_DIR / f"{step}.done").exists()


def _mark_done(step: str) -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (CKPT_DIR / f"{step}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")


def step_fetch(force: bool = False) -> None:
    if _done("fetch") and not force:
        _log("fetch: skipped (checkpoint present)")
        return
    t0 = time.time()
    r = fetch_mod.fetch(force=force)
    _log(f"fetch: ref={r['ref']} counts={r['counts']}, done in {time.time() - t0:.1f}s")
    _mark_done("fetch")


def step_check(force: bool = False) -> None:
    if _done("check") and not force:
        _log("check: skipped (checkpoint present)")
        return
    t0 = time.time()
    r = check_mod.check()
    _log(
        f"check: users={r['distinct_users']:,} train_final={r['users_with_train_final_order']:,} "
        f"test_final={r['users_with_test_final_order']:,} products={r['distinct_products']:,}, "
        f"done in {time.time() - t0:.1f}s"
    )
    _mark_done("check")


def step_clean(force: bool = False) -> None:
    if _done("clean") and not force:
        _log("clean: skipped (checkpoint present)")
        return
    t0 = time.time()
    raw = clean_mod.load_raw()
    cleaned, stats = clean_mod.clean(raw)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "clean_stats.json").write_text(json.dumps(stats, indent=2))
    del raw, cleaned
    gc.collect()
    _log(f"clean: {stats}, done in {time.time() - t0:.1f}s")
    _mark_done("clean")


def step_features(force: bool = False) -> None:
    if _done("features") and not force:
        _log("features: skipped (checkpoint present)")
        return
    t0 = time.time()
    raw = clean_mod.load_raw()
    cleaned, _stats = clean_mod.clean(raw)
    del raw
    gc.collect()

    feat = features_mod.build_features(cleaned["orders"], cleaned["op_prior"], cleaned["products"])
    rows = features_mod.attach_labels(feat, cleaned["orders"], cleaned["op_train"])
    del feat
    gc.collect()

    user_ids = rows["user_id"].unique()
    train_ids, val_ids, test_ids = features_mod.train_val_test_split(user_ids)

    rows.to_parquet(OUT_DIR / "candidate_rows.parquet", index=False)
    holdout_features = rows[rows["user_id"].isin(test_ids)].drop(columns=["label"])
    holdout_features.to_parquet(OUT_DIR / "holdout_features.parquet", index=False)

    (OUT_DIR / "split_ids.json").write_text(json.dumps({
        "train_ids": sorted(train_ids), "val_ids": sorted(val_ids), "test_ids": sorted(test_ids),
        "seed": 26,
    }))

    _log(
        f"features: {len(rows):,} candidate rows, {rows['label'].sum():,} positive, "
        f"{len(user_ids):,} users (train={len(train_ids):,} val={len(val_ids):,} "
        f"test={len(test_ids):,}), done in {time.time() - t0:.1f}s"
    )
    _mark_done("features")


def step_train(force: bool = False) -> None:
    if _done("train") and not force:
        _log("train: skipped (checkpoint present)")
        return
    t0 = time.time()
    rows = pd.read_parquet(OUT_DIR / "candidate_rows.parquet")
    split = json.loads((OUT_DIR / "split_ids.json").read_text())
    train_ids, val_ids = set(split["train_ids"]), set(split["val_ids"])

    _booster, info = model_mod.fit_lambdarank(rows, train_ids, val_ids)
    (OUT_DIR / "train_info.json").write_text(json.dumps(info, indent=2))
    _log(f"train: {info}, done in {time.time() - t0:.1f}s")
    _mark_done("train")


def step_eval(force: bool = False) -> None:
    if _done("eval") and not force:
        _log("eval: skipped (checkpoint present)")
        return
    t0 = time.time()
    rows = pd.read_parquet(OUT_DIR / "candidate_rows.parquet")
    split = json.loads((OUT_DIR / "split_ids.json").read_text())
    test_ids = set(split["test_ids"])
    test_rows = rows[rows["user_id"].isin(test_ids)].copy()

    booster = model_mod.load_lambdarank()
    test_rows["lambdarank_score"] = model_mod.predict_lambdarank(booster, test_rows)
    test_rows["baseline_score"] = model_mod.baseline_score(test_rows)

    lambdarank_metrics = eval_mod.evaluate(test_rows, "lambdarank_score")
    baseline_metrics = eval_mod.evaluate(test_rows, "baseline_score")

    (OUT_DIR / "metrics_lambdarank.json").write_text(json.dumps(lambdarank_metrics, indent=2))
    (OUT_DIR / "metrics_baseline.json").write_text(json.dumps(baseline_metrics, indent=2))

    _log(
        f"eval: lambdarank ndcg@10={lambdarank_metrics['ndcg@10']:.4f} "
        f"recall@10={lambdarank_metrics['recall@10']:.4f} | baseline ndcg@10="
        f"{baseline_metrics['ndcg@10']:.4f} recall@10={baseline_metrics['recall@10']:.4f}, "
        f"done in {time.time() - t0:.1f}s"
    )
    _mark_done("eval")


def step_surrogate(force: bool = False) -> None:
    if _done("surrogate") and not force:
        _log("surrogate: skipped (checkpoint present)")
        return
    t0 = time.time()
    raw = clean_mod.load_raw()
    cleaned, _stats = clean_mod.clean(raw)
    del raw
    gc.collect()

    timeline = surrogate_mod.build_timeline(cleaned["orders"])
    dataset = surrogate_mod.build_dataset(timeline, cleaned["op_prior"], cleaned["op_train"])
    del timeline, cleaned
    gc.collect()

    user_ids = dataset["user_id"].unique()
    train_ids, _val_ids, test_ids = features_mod.train_val_test_split(user_ids)
    results, _model = surrogate_mod.fit_and_evaluate(dataset, train_ids, test_ids)

    (OUT_DIR / "surrogate_results.json").write_text(json.dumps(results, indent=2))
    _log(
        f"surrogate: n_in_scope={results['n_users_in_scope']:,} "
        f"model_spearman={results['model_spearman']:.4f} "
        f"baseline_spearman={results['baseline_30day_count_spearman']:.4f}, "
        f"done in {time.time() - t0:.1f}s"
    )
    _mark_done("surrogate")


def step_export(force: bool = False) -> None:
    from reorder_radar import export as export_mod

    if _done("export") and not force:
        _log("export: skipped (checkpoint present)")
        return
    t0 = time.time()
    export_mod.export()
    _log(f"export: done in {time.time() - t0:.1f}s")
    _mark_done("export")


STEPS = {
    "fetch": step_fetch,
    "check": step_check,
    "clean": step_clean,
    "features": step_features,
    "train": step_train,
    "eval": step_eval,
    "surrogate": step_surrogate,
    "export": step_export,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="reorder-radar")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in list(STEPS.keys()) + ["all"]:
        p = sub.add_parser(name)
        p.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.command == "all":
        for step_fn in STEPS.values():
            step_fn(force=args.force)
    else:
        STEPS[args.command](force=args.force)


if __name__ == "__main__":
    main()
