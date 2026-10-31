import json

import pytest

from bookrec.gate import check
from bookrec.promote import production_entry, require_passed_gate

POP = {"ndcg_at_20": 0.04}
PROD = {
    "version": 1,
    "data_hash": "abc",
    "eval_split": "t1",
    "metrics": {"ndcg_at_20": 0.10, "coverage_at_20": 0.30},
}


def cand(**kw):
    base = {"data_hash": "abc", "eval_split": "t1", "ndcg_at_20": 0.11, "coverage_at_20": 0.31}
    return {**base, **kw}


def names(report, passed):
    return {c["name"] for c in report["checks"] if c["passed"] is passed and not c["skipped"]}


def test_good_candidate_passes():
    assert check(cand(), PROD, POP, 20.0)["passed"]


def test_first_deploy_skips_production_comparisons():
    report = check(cand(), None, POP, 20.0)
    assert report["passed"]
    skipped = {c["name"] for c in report["checks"] if c["skipped"]}
    assert skipped == {"same_data_as_production", "no_ndcg_regression", "no_coverage_regression"}


def test_version_null_counts_as_first_deploy():
    assert check(cand(), {"version": None, "metrics": {}}, POP, 20.0)["passed"]


def test_worse_model_is_blocked():
    report = check(cand(ndcg_at_20=0.08), PROD, POP, 20.0)
    assert not report["passed"]
    assert "no_ndcg_regression" in names(report, False)


def test_one_percent_noise_is_allowed():
    assert check(cand(ndcg_at_20=0.0995), PROD, POP, 20.0)["passed"]


def test_model_not_beating_popularity_is_blocked():
    report = check(cand(ndcg_at_20=0.045), None, POP, 20.0)
    assert names(report, False) == {"beats_popularity"}


def test_coverage_collapse_is_blocked():
    report = check(cand(coverage_at_20=0.10), PROD, POP, 20.0)
    assert names(report, False) == {"no_coverage_regression"}


def test_different_data_is_blocked():
    report = check(cand(data_hash="other"), PROD, POP, 20.0)
    assert "same_data_as_production" in names(report, False)


def test_slow_model_is_blocked_and_unmeasured_latency_is_skipped():
    assert names(check(cand(), PROD, POP, 150.0), False) == {"latency_p95"}
    assert check(cand(), PROD, POP, None)["passed"]


def test_promote_refuses_without_passing_report(tmp_path):
    with pytest.raises(SystemExit):
        require_passed_gate("run1", tmp_path)  # no report at all
    (tmp_path / "run1.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(SystemExit):
        require_passed_gate("run1", tmp_path)
    (tmp_path / "run1.json").write_text(json.dumps({"passed": True}))
    assert require_passed_gate("run1", tmp_path) == tmp_path / "run1.json"


def test_production_entry_keeps_only_the_gate_metrics():
    card = {"run_id": "r", "data_hash": "h", "eval_split": "t1", "git_hash": "g"}
    entry = production_entry(2, "gs://b/models/v2/", card, {"ndcg_at_20": 0.1, "train_time_s": 9})
    assert entry["version"] == 2 and entry["metrics"] == {"ndcg_at_20": 0.1}
