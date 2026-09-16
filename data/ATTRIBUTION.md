# Data attribution

Dataset: Instacart Market Basket Analysis (Kaggle **dataset** mirror; the original Kaggle **competition** of the same name is delisted, so this build uses the most complete Kaggle dataset mirror of the same six files instead).

Source: Kaggle Datasets, ref `psparks/instacart-market-basket-analysis`
https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis

License: CC0-1.0, as declared by the Kaggle dataset listing.

The underlying data is "The Instacart Online Grocery Shopping Dataset 2017", originally released by Instacart for the 2017 Kaggle competition. Per Instacart's original citation request: "Accessed from https://www.instacart.com/datasets/grocery-shopping-2017 on 2026-09-16".

Known data quirk (Instacart's own documentation): `days_since_prior_order` is capped at 30 -- a true gap longer than 30 days is recorded as exactly 30.0. This affects any reconstructed absolute timeline (used by the surrogate-value module) for users with long gaps between orders; noted here rather than silently ignored.

## Files and verified row counts (data lines, header excluded)

| file | verified rows | known published figure |
|---|---|---|
| orders.csv | 3,421,083 | 3,421,083 |
| order_products__prior.csv | 32,434,489 | 32,434,489 |
| order_products__train.csv | 1,384,617 | 1,384,617 |
| products.csv | 49,688 | 49,688 |
| aisles.csv | 134 | -- |
| departments.csv | 21 | -- |

Distinct `user_id` values in `orders.csv`: 206,209 (known published figure: ~206,209; matches).

Verified on a prior run (this run skipped re-downloading because the files and counts already matched).
