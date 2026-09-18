# Reorder Radar

Ranks a returning grocery shopper's past products by probability of
reorder in their next basket. Built and evaluated on the full real
Instacart Market Basket Analysis dataset, 3,421,083 orders from 206,209
users (Kaggle dataset mirror; the original Kaggle competition of the same
name is delisted -- see `data/ATTRIBUTION.md`).

## Task

For each user's final ("train") order, rank the products that user bought
in an earlier order by probability of reappearing. Candidate universe:
every product a user bought in a *prior* order. Label: 1 iff that product
appears in the user's final order. Only the 131,209 users whose final
order is Kaggle's labeled "train" order are in scope (the 75,000 "test"
users have no published labels and are excluded, not silently dropped).

## Architecture

Two rankers, both trained/scored on the same per-(user, product) feature
table (`features.py`):

- **`model.fit_lambdarank`** -- a LightGBM `lambdarank` model, grouped by
  `user_id`, trained on 80% of users and early-stopped on a held-out 10%.
- **`model.baseline_score`** -- a buy-it-again-by-frequency baseline: no
  fitting, ranks a user's candidates by `times_bought` (tie-broken by
  `reorder_rate`), both computed from prior orders only.

Every feature is past-only: `times_bought`, `reorder_rate`,
`orders_since_last_bought`, `days_since_last_bought`,
`mean_add_to_cart_position`, `aisle_reorder_rate`,
`department_reorder_rate`, `user_order_count`, `user_mean_basket_size`,
`user_mean_days_between_orders`. `days_since_last_bought` is computed
from a cumulative day-offset clock built only from the user's *prior*
orders -- the final order's own timing is never added.
`tests/test_no_leak.py` proves this two ways: `build_features`'s
signature structurally excludes `op_train`, and hand-calculated feature
values on a small synthetic dataset confirm the final order's timing
(a 3.0-day gap in the test fixture) is never folded into the clock.

A separate **surrogate-value module** (`surrogate.py`) asks a different
question: does 30 days of early activity predict a user's activity over
the following 12 months? For each user with a reconstructed order
timeline spanning >=365 days, it fits a small LightGBM regressor from
30-day window features (order count, product count, mean basket size) to
365-day order count, and reports Spearman correlation on held-out users
against the naive baseline (the raw 30-day order count itself).

## How to run

```bash
git clone <this repo> && cd reorder-radar
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
python3 -m reorder_radar.cli all    # fetch -> check -> clean -> features -> train -> eval -> surrogate -> export
python3 -m pytest -q
python3 -m ruff check .
```

Each step is resumable: rerunning `reorder-radar all` skips any step whose
`outputs/checkpoints/<step>.done` marker exists; pass `--force` to redo a
step (or all of them). `fetch` needs a logged-in Kaggle CLI
(`~/.kaggle/kaggle.json`); every later step runs from local files only.

Mac-specific note: `lightgbm`'s wheel expects Homebrew's `libomp.dylib`,
which is not installed on this machine. Fixed once, inside this repo's own
`.venv`, with:
```
install_name_tool -change @rpath/libomp.dylib /opt/anaconda3/lib/libomp.dylib \
  .venv/lib/python3.11/site-packages/lightgbm/lib/lib_lightgbm.dylib
```
Anaconda's existing `libomp.dylib` is used as the real library; nothing was
installed system-wide. Not needed again unless the venv is recreated.

## Serving

`rank.top_n_for_user(user_id, booster, holdout_features, n=10)` loads the
trained booster and a precomputed holdout feature parquet
(`outputs/holdout_features.parquet`) and returns a user's top-N candidate
products by predicted score -- no training or full-dataset load. Example,
a real holdout user from this run:

```
$ python3 -m reorder_radar.rank 156122
{'product_id': 13176, 'score': 3.32}
{'product_id': 47329, 'score': 3.12}
{'product_id': 13245, 'score': 2.27}
...
```

## Results (10% test split, seed 26, 13,121 users; 12,231 evaluated, 890
excluded for having zero candidate products that reappear in their final
order)

Two baselines, no fitting for either: **buy-it-again-by-frequency** (rank by
`times_bought`, tie-broken by `reorder_rate`) and **frequency+recency**
(rank by `times_bought`, tie-broken by days since the item was last bought
-- more recent scores higher). Point estimates are 95% confidence intervals
from a paired bootstrap over users (1,000 resamples, seed 26, computed by
`src/reorder_radar/bootstrap.py` from `outputs/per_user_metrics.csv`, so
future re-runs of the interval never need the raw candidate rows again).

| Metric | LightGBM lambdarank | Buy-it-again-by-frequency baseline | Frequency+recency baseline |
|---|---|---|---|
| NDCG@10 | 0.5486 (0.5439-0.5534) | 0.5013 (0.4965-0.5059) | 0.5177 |
| NDCG@20 | 0.6051 | 0.5570 | 0.5756 |
| Recall@10 | 0.5782 (0.5728-0.5839) | 0.5246 (0.5191-0.5304) | 0.5443 |
| Recall@20 | 0.7550 | 0.7016 | 0.7254 |

Lift, lambdarank minus the frequency baseline, same paired bootstrap (the
pairing -- resampling the same users for both sides of the subtraction each
round -- is what keeps this interval tight despite each metric's own CI
being wider): **NDCG@10 lift 0.0447 (95% CI 0.0445-0.0501)**, **Recall@10
lift 0.0535 (95% CI 0.0497-0.0573)**. Zero is nowhere near either interval,
so the lift is not bootstrap noise.

The lambdarank model beats both baselines on every metric, with the
largest gap at NDCG@10 over the plain frequency baseline -- it is doing
more than just re-deriving "buy what you buy most", most visibly on users
with large candidate sets (see `outputs/site_data.json` example rankings,
e.g. `holdout_user_08`: 139 candidates, 11 reordered, ranked almost
entirely correctly by purchase frequency *and* recency together). The
frequency+recency baseline closes part of that gap on its own, which is
exactly what the segment tables below make visible.

### Results by candidate-set size (paired bootstrap CI on the NDCG@10 lift, same 1,000 resamples)

| Candidate-set size | n users | NDCG@10 model | NDCG@10 baseline | Lift | 95% CI |
|---|---|---|---|---|---|
| under 10 | 662 | 0.8301 | 0.8154 | 0.0147 | 0.0037-0.0257 |
| 10 to 29 | 2,798 | 0.6479 | 0.5947 | 0.0531 | 0.0457-0.0596 |
| 30 to 99 | 6,204 | 0.5108 | 0.4642 | 0.0466 | 0.0427-0.0507 |
| 100 and more | 2,567 | 0.4593 | 0.4082 | 0.0511 | 0.0458-0.0567 |

### Results by shopper history (prior-order count quartile)

| Quartile (prior orders) | n users | NDCG@10 model | NDCG@10 baseline | Lift | 95% CI |
|---|---|---|---|---|---|
| Q1 (3-5) | 3,311 | 0.5864 | 0.5473 | 0.0390 | 0.0332-0.0447 |
| Q2 (6-10) | 3,142 | 0.5557 | 0.5139 | 0.0418 | 0.0365-0.0476 |
| Q3 (11-20) | 2,801 | 0.5339 | 0.4839 | 0.0500 | 0.0443-0.0562 |
| Q4 (21-99) | 2,977 | 0.5131 | 0.4533 | 0.0597 | 0.0545-0.0652 |

Users with the smallest candidate sets are near a ranking ceiling for both
model and baseline (few products to reorder from, so both agree most of
the time) -- the lift there is real but small and has the widest CI
relative to its size. Lift grows and stays statistically clear of zero
once a user has 10+ candidates or 6+ prior orders, and is largest for
users with the longest shopper history (Q4), where a frequency count alone
has the most room to be wrong. Full numbers: `outputs/segments.json`.

Val-set (early-stopping) NDCG during training: NDCG@10 0.5751, NDCG@20
0.6298, best iteration 147 of 500, fit on 104,967 users.

## Surrogate-value module results

Only 1,568 of 131,209 users (1.2%) have a reconstructed order-history
timeline spanning a full 365 days -- Instacart's own data collection
window, plus the `days_since_prior_order` 30-day cap (see
`data/ATTRIBUTION.md`), leaves most users with much shorter observed
histories. This is reported as a real, small-n result, not padded out.

| Metric | Small LightGBM model (30d features -> 365d count) | Naive baseline (raw 30d count) |
|---|---|---|
| Spearman correlation (157 held-out users) | 0.7032 (p=1.0e-24) | 0.7145 (p=8.1e-26) |

The naive 30-day count is a slightly *better* predictor of 12-month
activity than the small model built on top of it, on this sample size --
logged as a real, unflattering result rather than tuned away. A plausible
read: with only three simple 30-day features and 1,254 training rows, the
small model has little room to beat a single well-correlated raw count;
a richer feature set or more training data might change this, but that
was not built here (see `CONTEXT.md` open items).

## Design decisions

1. **Kaggle dataset mirror, not the delisted competition.** The official
   `instacart-market-basket-analysis` Kaggle *competition* is delisted.
   `fetch.py` searches Kaggle *datasets* for a mirror with all six files,
   picks the first candidate by vote count, and verifies its row counts
   against the known published Instacart figures before trusting it. The
   chosen mirror (`psparks/instacart-market-basket-analysis`, CC0-1.0)
   matched every published figure exactly. See `data/ATTRIBUTION.md`.
2. **Only "train"-final-order users are in scope.** Kaggle's original
   `test`-eval_set users (75,000 of them) have no published final-order
   labels; including them as unlabeled candidates would silently shrink
   recall's denominator or require fabricating labels. They are counted
   in `data_quality.json` and excluded, not hidden.
3. **`days_since_last_bought` uses a prior-orders-only clock.** The final
   order's own `days_since_prior_order` (its gap from the user's last
   prior order) is a real, pre-order-content signal a live system would
   have, but it was left out of every feature anyway, to keep the
   leakage proof in `test_no_leak.py` simple and airtight rather than
   arguing case-by-case about which final-order fields are "safe".
4. **Buy-it-again-by-frequency baseline, not a second learned model.**
   The task asks for exactly this baseline; it needs no train/val split
   of its own and isolates how much of the lambdarank model's lift is
   pure "frequency + recency" logic already captured by simple counts.
5. **Surrogate-value module scoped to users with a full 365-day
   timeline**, even though this leaves only 1,568 users. Extending the
   window to users with shorter spans (right-censoring their 12-month
   count) would inflate the sample at the cost of a biased target; that
   tradeoff was not taken.

## Data provenance

Dataset: Instacart Market Basket Analysis, Kaggle dataset mirror
`psparks/instacart-market-basket-analysis`, CC0-1.0. Downloaded
2026-09-16. Verified row counts: 3,421,083 orders, 32,434,489
`order_products__prior` rows, 1,384,617 `order_products__train` rows,
49,688 products, 134 aisles, 21 departments, 206,209 distinct users --
every one matches the known published Instacart figures exactly. Full
attribution and the known `days_since_prior_order` 30-day-cap data quirk:
`data/ATTRIBUTION.md`.

Full metrics, split sizes, clean-step stats, surrogate results, and 10
anonymized example rankings: `outputs/site_data.json` -- committed,
produced by `reorder-radar export`, never typed by hand.

## Live serving endpoint (AWS Lambda + API Gateway)

The trained lambdarank model is served live over HTTPS, mirroring the
Buyer Value Radar deployment (Lambda + API Gateway + S3, no idle cost).

- **URL:** `https://fmyrmgb4h1.execute-api.us-east-1.amazonaws.com`
- **Routes:**
  - `GET /health` -> `{"status": "ok"}`
  - `GET /users/sample` -> 10 real holdout `user_id`s you can try
  - `POST /rank` with body `{"user_id": <int>}` -> that user's top-10
    candidate products by predicted reorder score
  - CORS enabled for `https://www.giggitai.com` and `https://giggitai.com`

Example:

```bash
curl -s https://fmyrmgb4h1.execute-api.us-east-1.amazonaws.com/health
# {"status": "ok"}

curl -s https://fmyrmgb4h1.execute-api.us-east-1.amazonaws.com/users/sample
# {"user_ids": [2, 9, 50, 109, 110, 116, 117, 121, 135, 192]}

curl -s -X POST https://fmyrmgb4h1.execute-api.us-east-1.amazonaws.com/rank \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 156122}'
```

Response shape:

```json
{
  "user_id": 156122,
  "products": [
    {"product_id": 13176, "product_name": "Bag of Organic Bananas", "score": 3.318618, "times_bought_before": 51},
    ...
  ]
}
```

An unknown `user_id` (not in the holdout set) returns HTTP 404 with an
`error` message.

### How it's built

`aws-lambda/score.py` is the Lambda handler. It reimplements
`reorder_radar.rank.top_n_for_user` without pandas or pyarrow (to keep
the deployment zip small): the holdout feature table
(`outputs/holdout_features.parquet`, 847,516 rows) is converted at build
time into a plain `numpy` `.npz` (user_id / product_id / float32 feature
matrix, sorted by user_id) plus a JSON `user_id -> [row_start, row_count]`
index, and a `product_id -> product_name` JSON lookup from
`data/raw/products.csv`. These, the trained model file
(`outputs/checkpoints/lambdarank_model.txt`), and `train_info.json`
(for `best_iteration`) are uploaded to S3 and pulled into `/tmp` on
Lambda cold start. See `aws-lambda/build_artifacts.py`.

### Verification (2026-09-16)

- `GET /health` -> 200 `{"status": "ok"}`
- `GET /users/sample` -> 200, 10 valid holdout user_ids
- `POST /rank` for holdout user 156122 -> **identical product order and
  scores** to a local run of `reorder_radar.rank.top_n_for_user(156122, ...)`
  against the real trained model and the real `holdout_features.parquet`
  -- max abs score diff ~5e-7 (an artifact of the endpoint's 6-decimal
  JSON rounding, not a real prediction difference).
- `POST /rank` for an unknown user_id (999999999) -> 404
- `OPTIONS /rank` preflight from `https://www.giggitai.com` -> 200 with
  `access-control-allow-origin: https://www.giggitai.com`

### Infrastructure

- S3 bucket `giggit-reorder-radar-models` (`models/` = artifacts,
  `lambda-code/` = deployment zip)
- IAM role `reorder-lambda-execution-role` (AWSLambdaBasicExecutionRole +
  inline `s3:GetObject` on `models/*`)
- Lambda function `reorder-rank-user` (python3.11, 1024 MB, 30 s timeout)
- API Gateway HTTP API `reorder-rank-api`

## Live Eval canary

`.github/workflows/canary.yml` runs `src/reorder_radar/canary.py` every 6
hours (cron `17 */6 * * *`, plus manual `workflow_dispatch`) against the
**live** `/rank` endpoint above -- not a local model load.

What it does:

1. Calls `POST /rank` once per shopper in the fixed 200-shopper canary set,
   `canary/rows.json` -- real held-out shoppers (from the 12,231 evaluated
   in `outputs/metrics_lambdarank.json`) paired with the products they
   actually reordered next. A test (`tests/test_canary.py`) proves every
   `user_id` in that file is in `outputs/split_ids.json`'s `test_ids`.
2. Scores each response with `eval.per_user_metrics` -- the same NDCG code
   the offline evaluation uses, imported rather than reimplemented -- and
   averages NDCG@10 across the set.
3. Compares the observed NDCG@10 to the recorded value in
   `outputs/metrics_lambdarank.json` (0.5486) within a tolerance of 0.030,
   and also records `recall_at_10`. An endpoint error on a row is counted
   in `errors` and contributes no score; `match` is only `true` if every
   row scored cleanly and the observed metric is within tolerance.
4. Prints one JSON record to stdout, then commits it to this repo's
   `ledger` branch (created as an orphan on first run): appended as one
   line to `ledger/runs.jsonl`, and overwritten as `ledger/latest.json`.
   Both files are public at
   `https://raw.githubusercontent.com/alphan-ml/reorder-radar/ledger/latest.json`
   and `.../ledger/runs.jsonl` -- this is what giggitai.com's Live Eval tab
   reads and replays as a terminal. Ledger commits are authored as
   `Alpha N <45754668+alphan-ml@users.noreply.github.com>`, message
   `canary: <system> <metric> <observed> (recorded <recorded>)`.

How to read `match`: `true` means the live endpoint's ranking quality on
this run is within 0.030 NDCG@10 of the recorded offline metric and every
row scored without error -- the deployed model matches what was measured
at training time. `false` means either an endpoint error occurred or the
live metric drifted outside tolerance; check `errors` and `observed` vs
`recorded` in that run's record to tell which.

Run it by hand:

```bash
python3 -m reorder_radar.canary
```
