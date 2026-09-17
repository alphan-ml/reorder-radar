# CONTEXT.md -- reorder-radar

## How to run

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
python3 -m reorder_radar.cli all
python3 -m pytest -q
python3 -m ruff check .
```

Mac-specific note: `lightgbm`'s wheel expects Homebrew's `libomp.dylib`,
which is not installed on this machine (no sudo/Homebrew). Fixed once,
inside this repo's own `.venv`, with:
```
install_name_tool -change @rpath/libomp.dylib /opt/anaconda3/lib/libomp.dylib \
  .venv/lib/python3.11/site-packages/lightgbm/lib/lib_lightgbm.dylib
```
Anaconda's existing `libomp.dylib` is used as the real library; nothing was
installed system-wide. Not needed again unless the venv is recreated.

Machine constraints this build ran under: 8 GB RAM, 8 cores (LightGBM
capped at 4 threads to share the machine with another training job
running concurrently), ~22 GiB free disk at the start of the build. Every
dtype in `clean.py`/`features.py` is deliberately narrow (int8/int16/
int32/float32/category) to fit the 32.4M-row `order_products__prior`
join inside that budget; intermediate frames are `del`eted and
`gc.collect()`ed between stages rather than kept alive.

## Data facts from the real run (Sep 16, 2026)

- Source: Kaggle dataset mirror `psparks/instacart-market-basket-analysis`
  (CC0-1.0) -- the official Kaggle *competition* of the same name is
  delisted. Chosen because it was the first dataset-search candidate (by
  vote count) whose file listing contained all six required files; its
  row counts matched every known published Instacart figure exactly.
  Full detail: `data/ATTRIBUTION.md`.
- Raw: 3,421,083 orders, 32,434,489 `order_products__prior` rows,
  1,384,617 `order_products__train` rows, 49,688 products, 134 aisles,
  21 departments, 206,209 distinct users. `clean.py` dropped 0 rows from
  every table on this mirror (no nulls, no duplicate ids, no
  out-of-range `reordered` values found) -- counted and logged
  (`outputs/clean_stats.json`), not assumed.
- Task scope: 131,209 users whose final order is a labeled "train" order
  (the only ones with a real label); 75,000 "test"-eval_set users have no
  published label and are excluded, counted separately
  (`outputs/data_quality.json`).
- Candidate table (`features.py` + `attach_labels`): 8,474,661
  (user, product) rows, 828,824 (9.8%) positive. Split 80/10/10 by user,
  seed 26: 104,967 train / 13,121 val / 13,121 test.
- LightGBM lambdarank: best iteration 133 of 500 (early-stopped, patience
  30), fit on 104,967 users / 6,779,743 rows, early-stopped on 13,121
  users. Val NDCG@10 0.5756, NDCG@20 0.6301.
- Test-set results (13,121 users, 12,231 evaluated / 890 excluded for
  zero positive candidates): lambdarank NDCG@10 0.5487, NDCG@20 0.6053,
  recall@10 0.5783, recall@20 0.7553; buy-it-again-by-frequency baseline
  NDCG@10 0.5013, NDCG@20 0.5570, recall@10 0.5246, recall@20 0.7016.
  Lambdarank beats the baseline on every metric.
- Surrogate-value module: 1,568 of 131,209 users (1.2%) have a
  reconstructed order timeline spanning >=365 days; small LightGBM model
  (30-day order/product/basket-size features -> 365-day order count)
  Spearman 0.7032 (n=157 held-out users, p=1.0e-24) vs. naive 30-day-count
  baseline Spearman 0.7145 (p=8.1e-26) -- the naive baseline is slightly
  *better* on this sample, reported as-is.
- Full pipeline wall time (fetch skip + check + clean + features + train
  + eval + surrogate + export, on the machine above): under 100 seconds
  end to end, per `outputs/run.log`.

## Data facts from the eval-hardening run (Sep 17, 2026)

- `outputs/per_user_metrics.csv`: one row per evaluated test-split user
  (12,231 rows) -- `n_candidates`, `n_prior_orders`, and NDCG@10/Recall@10
  for the lambdarank model and both baselines. Lets any future bootstrap or
  segment cut skip re-running the pipeline entirely.
- `src/reorder_radar/bootstrap.py`: paired bootstrap (1,000 resamples, seed
  26) over that CSV, plus the candidate-size and prior-order-quartile
  bucket logic used for the segment tables.
- `outputs/metrics_lambdarank.json` and `outputs/metrics_baseline.json`
  each gained `ndcg10_ci95` / `recall10_ci95`; the lambdarank file also
  gained `lift_vs_baseline_ci95` and `lift_vs_baseline_recall10_ci95`
  (lift = lambdarank minus the frequency baseline, paired bootstrap).
- `outputs/metrics_baseline_freq_recency.json`: point-estimate metrics for
  the new frequency+recency baseline (NDCG@10 0.5177, Recall@10 0.5443 --
  beats the plain frequency baseline, still below lambdarank).
- `outputs/segments.json`: the two bucket tables (candidate-set size:
  under 10 / 10-29 / 30-99 / 100+; prior-order-count quartile) with n
  users, model/baseline NDCG@10, lift, and a per-bucket paired-bootstrap CI
  on the lift.
- Retraining note: no trained model or raw data was committed by the
  original build, so this run re-executed `fetch` through `eval` on the
  same real dataset mirror, unchanged features/split/seed/hyperparameters,
  to get real scores to evaluate. LightGBM's histogram building is not
  perfectly deterministic across machines even at a fixed seed: this run's
  booster stopped at iteration 147 (val NDCG@10 0.5751) versus the
  original build's 133 (0.5756); test-set NDCG@10 0.5486 versus 0.5487.
  Both are noise-level differences from the same unchanged training code,
  not a retune.

## Decisions

| ID | Decision | Rejected alternative |
|---|---|---|
| D1 | Kaggle dataset mirror `psparks/instacart-market-basket-analysis` (first candidate by votes with all 6 files, verified row counts) | The official Kaggle competition dataset -- delisted, not downloadable |
| D2 | Only "train"-final-order users (131,209) are in scope; "test"-final-order users (75,000) excluded | Scoring "test" users with a fabricated or missing label -- would misstate recall's denominator |
| D3 | `days_since_last_bought` built from a prior-orders-only cumulative clock, never adding the final order's own `days_since_prior_order` | Using the final order's timing gap (arguably not "content leakage") -- left out anyway to keep the leakage proof in `test_no_leak.py` unconditional |
| D4 | Buy-it-again-by-frequency baseline computed directly (times_bought, tie-broken by reorder_rate), no fitting | A second learned baseline model -- not what the task asked for, and would have obscured how much lift lambdarank adds over plain frequency |
| D5 | Surrogate-value module scoped to the 1,568 users with a full 365-day observed timeline | Right-censoring shorter-history users into the 12-month target -- would bias the target toward 0 and was rejected rather than silently inflating n |
| D6 | Public identity on this repo is "Alpha N" / `45754668+alphan-ml@users.noreply.github.com`; local commits only, no push | -- (standing instruction for this build) |

## Open items

- Surrogate-value module's small model does not beat the naive 30-day-count
  baseline (Spearman 0.7032 vs 0.7145, n=157). Logged as a real result,
  not tuned away; a richer feature set (e.g. days-to-first-reorder,
  category diversity in the 30-day window) or a larger in-scope
  population (relaxing the 365-day-span requirement, with an explicit
  censoring correction) could close this gap but was not built here.
- No live serving endpoint was built for this repo (unlike
  `buyer-value-radar`'s AWS Lambda deployment) -- `rank.py` is a
  library-level serving module only, tested against the real trained
  model and a real holdout feature parquet on this machine, not deployed
  anywhere. No site file on giggitai.com was touched.

## Task reports

### Task: build the reorder-ranking pipeline (fetch through export) and the surrogate-value module

- STATUS: done, all steps run on the real, full dataset, no sampling.
- BUILT: `fetch.py`, `check.py`, `clean.py`, `features.py`, `model.py`,
  `eval.py`, `surrogate.py`, `export.py`, `rank.py`, `cli.py`;
  `tests/test_clean.py`, `tests/test_no_leak.py`, `tests/test_eval.py`,
  `tests/test_surrogate.py`, `tests/test_rank.py`; `pyproject.toml`,
  `.gitignore`, `.env.example`, `LICENSE`, `.github/workflows/ci.yml`.
- TESTED: `python3 -m pytest -q` -- 13 passed. `python3 -m ruff check .`
  -- all checks passed. Full pipeline (`reorder-radar all`) run once end
  to end on the real data; `rank.top_n_for_user` smoke-tested against
  the real trained model and real `outputs/holdout_features.parquet` for
  a real holdout user (156122) -- returned 10 scored products, correctly
  sorted descending.
- SPEC CHECK: task definition, feature list, models, split, metrics, and
  surrogate-value module built exactly as specified. Identity rules
  followed (grep of the repo and `git log` for the banned identity strings/
  secrets/`.env` came back clean outside `.venv/` third-party source,
  which is not part of this repo).
- OPEN: see Open items above.
- NEXT: nothing further planned for this build; local commit only, no
  push, no site change, no cloud deploy, per standing instructions.
- TIME: 2026-09-16, ~10:44-11:05 ET (data download through pipeline
  completion), plus write-up.

## Decisions (added -- AWS serving endpoint)

| ID | Decision | Rejected alternative |
|---|---|---|
| D10 | AWS (Lambda + API Gateway + S3), mirroring the Buyer Value Radar deployment | SageMaker endpoint (idle cost for a stateless scorer with occasional traffic; Lambda scales to zero) |
| D11 | Holdout feature table converted at build time to a plain numpy `.npz` (sorted by user_id, float32 feature matrix) plus a JSON `user_id -> [start, count]` index, instead of shipping pyarrow/pandas into the Lambda zip | Bundling pyarrow+pandas to read the parquet directly at cold start -- unnecessary dependency weight (would push the zip well past what's needed) for a lookup this simple; the reference implementation (`reorder_radar.rank.top_n_for_user`) is verified byte-order-identical against this reimplementation for a real holdout user |
| D12 | Reused the buyer-value-radar Lambda's numpy 1.26.4 / lightgbm 3.3.5 / scipy 1.13.1 manylinux2014 build (with bundled `libgomp.so.1`) rather than rebuilding wheels from scratch | Building a fresh Lambda-compatible wheel set for this repo -- the existing build is already verified working on python3.11/x86_64 Lambda and lightgbm 3.3.5 loads models trained with lightgbm 4.7.0 without numeric difference (same finding as D9 in buyer-value-radar/CONTEXT.md) |
| D13 | Product name lookup (`data/raw/products.csv` -> `product_id: product_name` JSON, 49,688 products, ~2.1MB) shipped as a plain S3 JSON file, loaded once per cold start | Bundling `products.csv` into the zip and parsing with pandas -- same pyarrow/pandas-avoidance reasoning as D11 |
| D14 | Only `aws-lambda/score.py` and `aws-lambda/build_artifacts.py` are committed to git; `aws-lambda/package/`, `aws-lambda/deploy.zip`, and `aws-lambda/artifacts/` are gitignored | Committing the ~164MB unpacked Lambda package (numpy/scipy/lightgbm binaries) -- matches the buyer-value-radar repo's existing convention (only `score.py` tracked) |

## Decisions (added -- eval hardening: bootstrap, second baseline, segments)

| ID | Decision | Rejected alternative |
|---|---|---|
| D15 | "The baseline" for the bootstrap lift and both segment tables is the original buy-it-again-by-frequency baseline, not the new frequency+recency one | Computing a lift/CI against the frequency+recency baseline instead (or in addition) -- would make the headline lift number a moving target depending on which baseline is "the" comparison; frequency+recency gets its own point-estimate metrics file instead |
| D16 | Frequency+recency baseline's recency term is `1 / (1 + days_since_last_bought)`, added to `times_bought * 1000` | A z-scored or rank-based recency term -- the bounded-(0,1] reciprocal keeps the same "times_bought dominates, second field only breaks ties" shape the existing `baseline_score` already uses, so the two baselines read as variations on one idea rather than two different scoring schemes |
| D17 | Per-user metrics persisted once to `outputs/per_user_metrics.csv`; `bootstrap.py` and any future segment cut read only that file | Re-deriving per-user metrics from `candidate_rows.parquet` inside `bootstrap.py` -- would tie every future interval recompute back to the multi-GB raw candidate table instead of a ~1MB CSV |

## Task reports (added -- eval hardening)

### Task: paired bootstrap CIs, frequency+recency baseline, results by segment

- STATUS: done, run against the real trained model and real test-split data (see the eval-hardening data facts above for the retraining note).
- BUILT: `model.baseline_score_freq_recency`; `eval.per_user_metrics_table`; `src/reorder_radar/bootstrap.py` (`paired_bootstrap`, `unpaired_lift_ci95`, `candidate_size_bucket`, `prior_orders_quartile_bucket`, `segment_table`); `cli.step_eval` updated to write `per_user_metrics.csv`, `metrics_baseline_freq_recency.json`, `segments.json`, and the new CI fields on `metrics_lambdarank.json` / `metrics_baseline.json`; `tests/test_bootstrap.py`; new tests in `tests/test_eval.py`.
- TESTED: `python3 -m pytest -q` -- 19 passed (13 prior + 6 new). `python3 -m ruff check .` -- all checks passed.
- SPEC CHECK: ranker features/split/seed/hyperparameters untouched (`fit_lambdarank` and `FEATURE_COLS` unchanged); no Lambda or site file touched; no data committed.
- NEXT: nothing further planned for this task.
- TIME: 2026-09-17.

## Task reports (added)

### Task: build and deploy the AWS scoring endpoint (Lambda + API Gateway + S3)

- STATUS: done, live, tested end-to-end against the real trained model.
- BUILT: `aws-lambda/score.py` (Lambda handler -- loads the lambdarank
  booster and the npz/index/product-name artifacts from S3 into `/tmp`
  on cold start; reimplements `top_n_for_user` without pandas/pyarrow);
  `aws-lambda/build_artifacts.py` (build-time converter: parquet ->
  npz + index JSON + product-name JSON, run once locally with the repo's
  own venv, which has pandas/pyarrow).
- INFRASTRUCTURE: S3 bucket `giggit-reorder-radar-models`; IAM role
  `reorder-lambda-execution-role`; Lambda function `reorder-rank-user`
  (python3.11, 1024MB, 30s timeout); API Gateway HTTP API
  `reorder-rank-api`, live at
  `https://fmyrmgb4h1.execute-api.us-east-1.amazonaws.com`
  (`GET /health`, `GET /users/sample`, `POST /rank`), CORS enabled for
  `https://www.giggitai.com` and `https://giggitai.com`.
- VERIFIED: `GET /health` -> 200; `GET /users/sample` -> 200 with 10
  real holdout user_ids; `POST /rank` for real holdout user 156122 ->
  product order and scores identical to a local run of
  `reorder_radar.rank.top_n_for_user` against the real trained model and
  real `outputs/holdout_features.parquet` (max abs score diff ~5e-7,
  attributable to the endpoint's 6-decimal JSON rounding, not a real
  difference); unknown user_id -> 404; `OPTIONS /rank` preflight from
  `https://www.giggitai.com` -> 200 with the correct
  `access-control-allow-origin` header.
- SPEC CHECK: no file on giggitai.com was touched -- backend-only, per
  the standing rule that site changes need explicit go-ahead before any
  deploy.
- NEXT: front-end block on the site (out of scope for this task); a
  distinct sibling build (fraud-radar) is being deployed in parallel by
  another session, with distinct AWS resource names by design.
- TIME: 2026-09-16, ~11:15-11:25 ET (see commit and BUILD-STATUS doc for
  exact stamp).
