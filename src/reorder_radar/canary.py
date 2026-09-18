"""Live Eval canary: score the deployed `/rank` endpoint against a fixed
200-shopper canary set (`canary/rows.json`, drawn from the held-out split
only) and print one ledger record (JSON, one line) to stdout.

Metric is NDCG@10, computed with `eval.per_user_metrics` -- the same code
the offline evaluation uses -- against each shopper's real reordered-product
set. No fallback numbers: an endpoint error for a row is counted in
`errors` and that row contributes no score; `match` is only true if every
row scored cleanly and the observed metric is within tolerance of the
recorded one.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from reorder_radar import eval as eval_mod

ROOT = Path(__file__).resolve().parents[2]
ROWS_PATH = ROOT / "canary" / "rows.json"
METRICS_PATH = ROOT / "outputs" / "metrics_lambdarank.json"
DEFAULT_ENDPOINT = "https://fmyrmgb4h1.execute-api.us-east-1.amazonaws.com/rank"
TOLERANCE = 0.030
SYSTEM = "reorder-radar"
METRIC_NAME = "ndcg_at_10"


def _git_sha(cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True, cwd=cwd,
    ).stdout.strip()


def _call_rank(
    session: requests.Session, endpoint: str, user_id: int, timeout: float = 15.0,
) -> tuple[list[int] | None, float, str | None]:
    """POST {user_id} to the live endpoint. Returns (ranked product ids in
    the endpoint's own order, or None on error; elapsed ms; error message
    or None)."""
    t0 = time.monotonic()
    try:
        resp = session.post(endpoint, json={"user_id": user_id}, timeout=timeout)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if resp.status_code != 200:
            return None, elapsed_ms, f"HTTP {resp.status_code}"
        payload = resp.json()
        products = payload.get("products")
        if not isinstance(products, list):
            return None, elapsed_ms, "response missing 'products' list"
        return [int(p["product_id"]) for p in products], elapsed_ms, None
    except requests.RequestException as e:
        return None, (time.monotonic() - t0) * 1000.0, str(e)
    except (ValueError, KeyError, TypeError) as e:
        return None, (time.monotonic() - t0) * 1000.0, f"bad response body: {e}"


def _ndcg10_recall10(
    ranked_product_ids: list[int], reordered_ids: set[int],
) -> tuple[float, float] | None:
    """Score one shopper's endpoint ranking against their real reordered
    set, via eval.per_user_metrics. The endpoint returns only the top 10
    candidates, but idcg needs the *total* relevant count, so any relevant
    product not present in the returned top 10 is added back as a
    lowest-priority row so it still counts toward idcg and n_relevant, the
    same way the full candidate table `eval.per_user_metrics` normally
    sees it. Zero-label filler rows pad the returned list up to position
    10 first -- without them, a shopper with fewer than 10 returned
    products would let those lowest-priority rows slide into the dcg's
    own top-10 window and inflate the score.
    """
    n_relevant = len(reordered_ids)
    if n_relevant == 0:
        return None
    top = ranked_product_ids[:10]
    found = sum(1 for pid in top if pid in reordered_ids)
    n_unseen_relevant = max(0, n_relevant - found)
    n_filler = 10 - len(top)

    scores = list(range(len(top), 0, -1))
    labels = [1 if pid in reordered_ids else 0 for pid in top]
    scores += [-1 - i for i in range(n_filler)]
    labels += [0] * n_filler
    scores += [-1 - n_filler - i for i in range(n_unseen_relevant)]
    labels += [1] * n_unseen_relevant

    user_df = pd.DataFrame({"score": scores, "label": labels})
    m = eval_mod.per_user_metrics(user_df, k_list=(10,))
    return m["ndcg@10"], m["recall@10"]


def run(
    rows_path: Path = ROWS_PATH,
    endpoint: str = DEFAULT_ENDPOINT,
    metrics_path: Path = METRICS_PATH,
    tolerance: float = TOLERANCE,
) -> dict:
    rows = json.loads(rows_path.read_text())
    recorded = json.loads(metrics_path.read_text())["ndcg@10"]

    ndcgs: list[float] = []
    recalls: list[float] = []
    latencies_ms: list[float] = []
    errors = 0
    error_examples: list[str] = []

    session = requests.Session()
    t_start = time.monotonic()
    for row in rows:
        user_id = row["user_id"]
        reordered_ids = set(row["reordered_product_ids"])
        ranked, elapsed_ms, err = _call_rank(session, endpoint, user_id)
        latencies_ms.append(elapsed_ms)
        if err is not None:
            errors += 1
            if len(error_examples) < 2:
                error_examples.append(f"user {user_id}: {err}")
            continue
        scored = _ndcg10_recall10(ranked, reordered_ids)
        if scored is None:
            continue
        ndcg, recall = scored
        ndcgs.append(ndcg)
        recalls.append(recall)
    duration_s = time.monotonic() - t_start

    n = len(ndcgs)
    observed = float(np.mean(ndcgs)) if n else 0.0
    recall_at_10 = float(np.mean(recalls)) if n else 0.0
    match = errors == 0 and n > 0 and abs(observed - recorded) <= tolerance

    p50_ms = round(float(np.percentile(latencies_ms, 50))) if latencies_ms else 0
    p95_ms = round(float(np.percentile(latencies_ms, 95))) if latencies_ms else 0

    release = _git_sha()
    lines = [
        f"$ canary {SYSTEM} --n {len(rows)} --endpoint /rank",
        f"release {release} · {len(rows)} holdout rows · {errors} errors",
        (
            f"ndcg@10 {observed:.4f} · recorded {recorded:.4f} · "
            f"tolerance {tolerance:.3f} · {'MATCH' if match else 'MISMATCH'}"
        ),
        f"recall@10 {recall_at_10:.4f}",
        f"p50 {p50_ms} ms · p95 {p95_ms} ms · {duration_s:.1f}s total",
    ]
    if error_examples:
        lines.append("errors: " + "; ".join(error_examples))

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": SYSTEM,
        "kind": "canary",
        "release": release,
        "endpoint": endpoint,
        "n": n,
        "metric": METRIC_NAME,
        "recorded": recorded,
        "observed": round(observed, 6),
        "tolerance": tolerance,
        "match": match,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "errors": errors,
        "duration_s": round(duration_s, 1),
        "extra": {"recall_at_10": round(recall_at_10, 6)},
        "lines": lines[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorder Radar live-eval canary")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--rows", type=Path, default=ROWS_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()
    record = run(
        rows_path=args.rows, endpoint=args.endpoint,
        metrics_path=args.metrics, tolerance=args.tolerance,
    )
    print(json.dumps(record))


if __name__ == "__main__":
    main()
