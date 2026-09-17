import numpy as np
import pandas as pd

from reorder_radar import bootstrap


def _toy_per_user(n=200, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "ndcg10_model": rng.uniform(0.4, 0.9, size=n),
        "ndcg10_baseline_freq": rng.uniform(0.3, 0.8, size=n),
    })


def test_ci_contains_point_estimate():
    per_user = _toy_per_user()
    result = bootstrap.paired_bootstrap(per_user, "ndcg10_model", "ndcg10_baseline_freq", n_resamples=1000)

    lo, hi = result["model_ci95"]
    assert lo <= result["model_point"] <= hi
    lo, hi = result["baseline_ci95"]
    assert lo <= result["baseline_point"] <= hi
    lo, hi = result["lift_ci95"]
    assert lo <= result["lift_point"] <= hi


def test_paired_interval_narrower_than_unpaired_on_correlated_data():
    # Model and baseline scores share a per-user "difficulty" effect (large
    # variance) plus small independent noise -- a realistic case where a
    # user's model and baseline scores rise and fall together.
    rng = np.random.RandomState(1)
    n = 300
    user_effect = rng.normal(0.0, 1.0, size=n)
    model_vals = user_effect + rng.normal(0.0, 0.05, size=n)
    baseline_vals = user_effect + rng.normal(0.0, 0.05, size=n)
    per_user = pd.DataFrame({"ndcg10_model": model_vals, "ndcg10_baseline_freq": baseline_vals})

    paired = bootstrap.paired_bootstrap(per_user, "ndcg10_model", "ndcg10_baseline_freq", n_resamples=1000)
    paired_width = paired["lift_ci95"][1] - paired["lift_ci95"][0]

    unpaired_lo, unpaired_hi = bootstrap.unpaired_lift_ci95(
        per_user, "ndcg10_model", "ndcg10_baseline_freq", n_resamples=1000
    )
    unpaired_width = unpaired_hi - unpaired_lo

    assert paired_width < unpaired_width


def test_candidate_size_bucket_edges():
    n_candidates = pd.Series([5, 9, 10, 29, 30, 99, 100, 500])
    labels = bootstrap.candidate_size_bucket(n_candidates)
    assert list(labels) == [
        "under_10", "under_10", "10_to_29", "10_to_29",
        "30_to_99", "30_to_99", "100_plus", "100_plus",
    ]


def test_prior_orders_quartile_bucket_has_four_groups():
    n_prior_orders = pd.Series(range(1, 101))
    labels = bootstrap.prior_orders_quartile_bucket(n_prior_orders)
    assert set(labels) == {"Q1", "Q2", "Q3", "Q4"}
