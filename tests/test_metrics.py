import numpy as np
import pytest

from bookrec.eval.metrics import coverage_at_k, evaluate, ndcg_at_k, recall_at_k


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(np.array([3, 1, 2, 9]), {1, 2, 3}, k=3) == pytest.approx(1.0)


def test_ndcg_hand_computed_case():
    # hits at ranks 2 and 3: DCG = 1/log2(3) + 1/log2(4); ideal = 1 + 1/log2(3)
    dcg = 1 / np.log2(3) + 1 / np.log2(4)
    idcg = 1 + 1 / np.log2(3)
    assert ndcg_at_k(np.array([1, 0, 2]), {0, 2}, k=3) == pytest.approx(dcg / idcg)


def test_ndcg_no_hits_and_empty_relevant():
    assert ndcg_at_k(np.array([1, 2, 3]), {9}, k=3) == 0.0
    assert ndcg_at_k(np.array([1, 2, 3]), set(), k=3) == 0.0


def test_ndcg_short_ranked_list_uses_full_ideal():
    # Only 1 item ranked but 3 relevant: the ideal DCG still covers 3 positions.
    idcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert ndcg_at_k(np.array([0]), {0, 1, 2}, k=5) == pytest.approx(1 / idcg)


def test_recall():
    assert recall_at_k(np.array([0, 5, 1]), {0, 1, 2, 3}, k=3) == pytest.approx(0.5)
    assert recall_at_k(np.array([0, 5, 1]), {0, 1, 2, 3}, k=1) == pytest.approx(0.25)


def test_coverage():
    recs = np.array([[0, 1], [1, 2]])
    assert coverage_at_k(recs, n_items=6) == pytest.approx(0.5)


def _fixed_scores(n_users, n_items):
    scores = np.tile(np.arange(n_items, 0, -1, dtype=np.float32), (n_users, 1))
    return lambda users: scores[: len(users)]


def test_evaluate_excludes_seen_items_and_empty_users():
    # Scores rank items 0,1,2,... for everyone.
    # user 0 has seen item 0 -> item 1 becomes rank 1 -> NDCG 1.0
    # user 1 targets the worst item (5) -> NDCG 0 with k=3
    # user 2 has no targets -> skipped
    res = evaluate(_fixed_scores(3, 6), {0: {0}}, {0: {1}, 1: {5}, 2: set()}, n_items=6, k=3)
    assert res["user_ids"].tolist() == [0, 1]
    assert res["per_user_ndcg"].tolist() == [1.0, 0.0]
    assert res["ndcg@3"] == pytest.approx(0.5)
    assert res["recall@3"] == pytest.approx(0.5)


def test_evaluate_seen_item_never_recommended():
    # If masking failed, the seen top-scored item 0 would be a hit for user 0.
    res = evaluate(_fixed_scores(1, 6), {0: {0}}, {0: {0}}, n_items=6, k=3)
    assert res["per_user_ndcg"][0] == 0.0


def test_evaluate_batching_gives_same_result():
    scores = np.random.default_rng(0).random((20, 30)).astype(np.float32)
    target = {u: {u % 30, (u + 7) % 30} for u in range(20)}
    fn = lambda users: scores[users]  # noqa: E731
    a = evaluate(fn, {}, target, n_items=30, k=5, batch_size=3)
    b = evaluate(fn, {}, target, n_items=30, k=5, batch_size=1000)
    assert a["ndcg@5"] == pytest.approx(b["ndcg@5"])
    assert np.allclose(a["per_user_ndcg"], b["per_user_ndcg"])
