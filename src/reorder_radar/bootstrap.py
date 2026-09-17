"""Paired bootstrap confidence intervals and segment (bucket) tables, built
entirely from the per-user metric rows `eval.per_user_metrics_table` writes
to `outputs/per_user_metrics.csv` -- never the raw candidate/feature data,
so a future interval or segment cut needs only that one small file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_RESAMPLES = 1000
SEED = 26

# (bucket name, inclusive lower bound, exclusive upper bound or None for open-ended)
CANDIDATE_SIZE_BUCKETS = [
    ("under_10", 0, 10),
    ("10_to_29", 10, 30),
    ("30_to_99", 30, 100),
    ("100_plus", 100, None),
]
PRIOR_ORDER_QUARTILE_BUCKETS = ["Q1", "Q2", "Q3", "Q4"]


def _ci95(values: np.ndarray) -> list[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def paired_bootstrap(
    per_user: pd.DataFrame, model_col: str, baseline_col: str,
    n_resamples: int = N_RESAMPLES, seed: int = SEED,
) -> dict:
    """Resample users with replacement -- the same resampled index set is
    used to compute both the model mean and the baseline mean each round, so
    their per-user correlation carries through into the lift estimate (a
    paired design). This is what makes the lift interval narrower than
    resampling the model and baseline scores independently on correlated
    data (see `unpaired_lift_ci95`).
    """
    n = len(per_user)
    if n == 0:
        raise ValueError("no users to bootstrap")
    rng = np.random.RandomState(seed)
    model_vals = per_user[model_col].to_numpy(dtype="float64")
    baseline_vals = per_user[baseline_col].to_numpy(dtype="float64")

    model_means = np.empty(n_resamples)
    baseline_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        model_means[i] = model_vals[idx].mean()
        baseline_means[i] = baseline_vals[idx].mean()
    lift_means = model_means - baseline_means

    return {
        "n_users": n,
        "n_resamples": n_resamples,
        "model_point": float(model_vals.mean()),
        "model_ci95": _ci95(model_means),
        "baseline_point": float(baseline_vals.mean()),
        "baseline_ci95": _ci95(baseline_means),
        "lift_point": float(model_vals.mean() - baseline_vals.mean()),
        "lift_ci95": _ci95(lift_means),
    }


def unpaired_lift_ci95(
    per_user: pd.DataFrame, model_col: str, baseline_col: str,
    n_resamples: int = N_RESAMPLES, seed: int = SEED,
) -> list[float]:
    """The same lift statistic as `paired_bootstrap`, but each resample draws
    an *independent* index set for the model and baseline columns, discarding
    the per-user pairing. Exists so a test can demonstrate the paired
    interval is narrower on correlated data -- the eval pipeline itself never
    calls this.
    """
    n = len(per_user)
    rng = np.random.RandomState(seed)
    model_vals = per_user[model_col].to_numpy(dtype="float64")
    baseline_vals = per_user[baseline_col].to_numpy(dtype="float64")
    lift_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx_m = rng.randint(0, n, size=n)
        idx_b = rng.randint(0, n, size=n)
        lift_means[i] = model_vals[idx_m].mean() - baseline_vals[idx_b].mean()
    return _ci95(lift_means)


def candidate_size_bucket(n_candidates: pd.Series) -> pd.Series:
    """Bucket label per row: under_10 / 10_to_29 / 30_to_99 / 100_plus."""
    labels = pd.Series(index=n_candidates.index, dtype=object)
    for name, lo, hi in CANDIDATE_SIZE_BUCKETS:
        mask = (n_candidates >= lo) if hi is None else ((n_candidates >= lo) & (n_candidates < hi))
        labels[mask] = name
    return labels


def prior_orders_quartile_bucket(n_prior_orders: pd.Series) -> pd.Series:
    """Quartile label per row (Q1 = fewest prior orders, Q4 = most)."""
    return pd.qcut(n_prior_orders, 4, labels=PRIOR_ORDER_QUARTILE_BUCKETS, duplicates="drop").astype(str)


def segment_table(
    per_user: pd.DataFrame, bucket_labels: pd.Series, bucket_order: list[str],
    model_col: str, baseline_col: str, range_col: str | None = None,
    n_resamples: int = N_RESAMPLES, seed: int = SEED,
) -> list[dict]:
    """One row per bucket present in `bucket_order`: n users, model/baseline
    point estimates, lift, and a paired-bootstrap CI on the lift computed
    from only that bucket's users.
    """
    out = []
    for name in bucket_order:
        bucket_df = per_user.loc[bucket_labels == name]
        if bucket_df.empty:
            continue
        result = paired_bootstrap(bucket_df, model_col, baseline_col, n_resamples=n_resamples, seed=seed)
        row = {
            "bucket": name,
            "n_users": result["n_users"],
            "ndcg10_model": result["model_point"],
            "ndcg10_baseline": result["baseline_point"],
            "ndcg10_lift": result["lift_point"],
            "ndcg10_lift_ci95": result["lift_ci95"],
        }
        if range_col is not None:
            row["range"] = f"{int(bucket_df[range_col].min())}-{int(bucket_df[range_col].max())}"
        out.append(row)
    return out
