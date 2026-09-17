"""Reorder Radar -- Lambda ranking handler.

Loads the trained LightGBM lambdarank booster plus a precomputed,
pyarrow-free holdout feature array from S3 (npz + a JSON index mapping
user_id -> row range), and returns a user's top-N candidate products by
predicted reorder score -- the same call as
reorder_radar.rank.top_n_for_user(user_id), reimplemented without pandas
or pyarrow so the Lambda zip stays small.
"""
from __future__ import annotations

import json
import os

import boto3
import lightgbm as lgb
import numpy as np

S3_BUCKET = os.environ.get("MODEL_BUCKET", "giggit-reorder-radar-models")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models")

_booster = None
_best_iteration = None
_user_id = None
_product_id = None
_features = None
_feature_cols = None
_index = None
_products = None


def _load_artifacts() -> None:
    global _booster, _best_iteration, _user_id, _product_id, _features
    global _feature_cols, _index, _products
    if _booster is not None:
        return
    s3 = boto3.client("s3")
    for fname in (
        "lambdarank_model.txt",
        "train_info.json",
        "holdout_features.npz",
        "holdout_index.json",
        "products.json",
    ):
        local_path = f"/tmp/{fname}"
        if not os.path.exists(local_path):
            s3.download_file(S3_BUCKET, f"{MODEL_PREFIX}/{fname}", local_path)

    _booster = lgb.Booster(model_file="/tmp/lambdarank_model.txt")
    with open("/tmp/train_info.json") as f:
        train_info = json.load(f)
    _best_iteration = train_info["best_iteration"]

    npz = np.load("/tmp/holdout_features.npz")
    _user_id = npz["user_id"]
    _product_id = npz["product_id"]
    _features = npz["features"]

    with open("/tmp/holdout_index.json") as f:
        idx_doc = json.load(f)
    _feature_cols = idx_doc["feature_cols"]
    _index = idx_doc["index"]

    with open("/tmp/products.json") as f:
        _products = json.load(f)


def sample_user_ids(n: int = 10) -> list[int]:
    _load_artifacts()
    ids = sorted(int(u) for u in _index)
    return ids[:n]


def top_n_for_user(user_id: int, n: int = 10) -> list[dict] | None:
    _load_artifacts()
    entry = _index.get(str(user_id))
    if entry is None:
        return None
    start, count = entry
    X = _features[start : start + count]
    pids = _product_id[start : start + count]
    scores = _booster.predict(X, num_iteration=_best_iteration)
    order = np.argsort(-scores)[:n]
    times_bought_idx = _feature_cols.index("times_bought")
    out = []
    for i in order:
        pid = int(pids[i])
        out.append(
            {
                "product_id": pid,
                "product_name": _products.get(str(pid), f"Product {pid}"),
                "score": round(float(scores[i]), 6),
                "times_bought_before": int(X[i, times_bought_idx]),
            }
        )
    return out


ALLOWED_ORIGINS = {"https://www.giggitai.com", "https://giggitai.com"}


def _headers(event: dict) -> dict:
    origin = (
        event.get("headers", {}).get("origin")
        or event.get("headers", {}).get("Origin")
        or ""
    )
    allow_origin = origin if origin in ALLOWED_ORIGINS else "https://www.giggitai.com"
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": allow_origin,
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    }


def _resp(event: dict, status: int, body) -> dict:
    return {
        "statusCode": status,
        "headers": _headers(event),
        "body": body if isinstance(body, str) else json.dumps(body),
    }


def handler(event, context):
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "GET"
    )
    path = event.get("rawPath") or event.get("path") or ""

    if method == "OPTIONS":
        return _resp(event, 200, "")

    if path.endswith("/health"):
        return _resp(event, 200, {"status": "ok"})

    if path.endswith("/users/sample"):
        return _resp(event, 200, {"user_ids": sample_user_ids(10)})

    if path.endswith("/rank"):
        if method != "POST":
            return _resp(event, 405, {"error": "use POST /rank"})
        try:
            raw_body = event.get("body") or "{}"
            if event.get("isBase64Encoded"):
                import base64

                raw_body = base64.b64decode(raw_body).decode()
            payload = json.loads(raw_body)
            if "user_id" not in payload:
                return _resp(event, 400, {"error": "missing field: user_id"})
            try:
                user_id = int(payload["user_id"])
            except (TypeError, ValueError):
                return _resp(event, 400, {"error": "user_id must be an integer"})
            products = top_n_for_user(user_id, n=10)
            if products is None:
                return _resp(
                    event, 404, {"error": f"unknown user_id {user_id} (not in holdout set)"}
                )
            return _resp(event, 200, {"user_id": user_id, "products": products})
        except json.JSONDecodeError:
            return _resp(event, 400, {"error": "invalid JSON body"})
        except Exception as e:  # noqa: BLE001
            return _resp(event, 500, {"error": "internal error", "detail": str(e)})

    return _resp(event, 404, {"error": "not found"})
