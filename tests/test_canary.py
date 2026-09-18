import json
import math
from pathlib import Path
from unittest.mock import patch

from reorder_radar import canary as canary_mod

ROOT = Path(__file__).resolve().parents[1]


def test_canary_rows_are_all_in_the_holdout_split():
    rows = json.loads((ROOT / "canary" / "rows.json").read_text())
    split = json.loads((ROOT / "outputs" / "split_ids.json").read_text())
    test_ids = set(split["test_ids"])
    assert len(rows) == 200
    user_ids = [r["user_id"] for r in rows]
    assert len(user_ids) == len(set(user_ids))  # no duplicate shoppers
    assert all(uid in test_ids for uid in user_ids)


def test_canary_rows_have_nonempty_distinct_reordered_sets():
    rows = json.loads((ROOT / "canary" / "rows.json").read_text())
    for row in rows:
        pids = row["reordered_product_ids"]
        assert len(pids) > 0
        assert len(set(pids)) == len(pids)


def test_ndcg10_recall10_hand_calculated():
    # 3 relevant products total; endpoint returns 2 of them ranked 1st and
    # 3rd among 4 results, the 3rd relevant product (99) is never returned.
    # Only 4 results means positions 5-10 are zero-label filler, so the
    # unreturned relevant product (rank-11+) never enters the dcg window.
    ranked = [10, 11, 12, 13]
    reordered = {10, 12, 99}
    ndcg, recall = canary_mod._ndcg10_recall10(ranked, reordered)
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(ndcg, dcg / idcg, rel_tol=1e-9)
    assert math.isclose(recall, 2 / 3, rel_tol=1e-9)


def test_ndcg10_recall10_perfect_rank_is_one():
    ranked = [1, 2, 3]
    reordered = {1, 2}
    ndcg, recall = canary_mod._ndcg10_recall10(ranked, reordered)
    assert math.isclose(ndcg, 1.0, rel_tol=1e-9)
    assert math.isclose(recall, 1.0, rel_tol=1e-9)


def test_ndcg10_recall10_none_for_zero_relevant():
    assert canary_mod._ndcg10_recall10([1, 2, 3], set()) is None


def _write_canary_fixture(tmp_path, rows, recorded_ndcg10):
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps(rows))
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"ndcg@10": recorded_ndcg10}))
    return rows_path, metrics_path


def test_run_reports_match_true_within_tolerance(tmp_path):
    rows = [
        {"user_id": 1, "reordered_product_ids": [10]},
        {"user_id": 2, "reordered_product_ids": [20]},
    ]
    rows_path, metrics_path = _write_canary_fixture(tmp_path, rows, recorded_ndcg10=1.0)

    def fake_call_rank(session, endpoint, user_id, timeout=15.0):
        return [10 if user_id == 1 else 20], 5.0, None

    with patch.object(canary_mod, "_call_rank", side_effect=fake_call_rank):
        record = canary_mod.run(rows_path=rows_path, metrics_path=metrics_path, tolerance=0.03)

    assert record["n"] == 2
    assert record["errors"] == 0
    assert record["observed"] == 1.0
    assert record["recorded"] == 1.0
    assert record["match"] is True
    assert record["extra"]["recall_at_10"] == 1.0
    assert 1 <= len(record["lines"]) <= 8


def test_run_reports_match_false_when_endpoint_errors(tmp_path):
    rows = [{"user_id": 1, "reordered_product_ids": [10]}]
    rows_path, metrics_path = _write_canary_fixture(tmp_path, rows, recorded_ndcg10=1.0)

    def failing_call_rank(session, endpoint, user_id, timeout=15.0):
        return None, 5.0, "HTTP 500"

    with patch.object(canary_mod, "_call_rank", side_effect=failing_call_rank):
        record = canary_mod.run(rows_path=rows_path, metrics_path=metrics_path, tolerance=0.03)

    assert record["errors"] == 1
    assert record["n"] == 0
    assert record["observed"] == 0.0
    assert record["match"] is False


def test_run_reports_match_false_when_outside_tolerance(tmp_path):
    rows = [{"user_id": 1, "reordered_product_ids": [10, 11]}]
    rows_path, metrics_path = _write_canary_fixture(tmp_path, rows, recorded_ndcg10=0.99)

    def fake_call_rank(session, endpoint, user_id, timeout=15.0):
        return [999], 5.0, None  # never returns either relevant product

    with patch.object(canary_mod, "_call_rank", side_effect=fake_call_rank):
        record = canary_mod.run(rows_path=rows_path, metrics_path=metrics_path, tolerance=0.03)

    assert record["errors"] == 0
    assert record["n"] == 1
    assert record["observed"] == 0.0
    assert record["match"] is False
