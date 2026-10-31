import json
import logging

import numpy as np
import pytest
from app import main
from app.recommender import Recommender
from fastapi.testclient import TestClient

from bookrec.fixtures import make_fixture_model


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return make_fixture_model(tmp_path_factory.mktemp("model"))


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setattr(main, "rec", Recommender(model_dir))
    return TestClient(main.app)


def _history(rec: Recommender, user_id: str) -> set[str]:
    u = rec.user_to_idx[user_id]
    idx = rec.hist_indices[rec.hist_indptr[u] : rec.hist_indptr[u + 1]]
    return {rec.idx_to_book[i] for i in idx}


def test_health_reports_version(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_version": "vtest"}


def test_known_user_returns_k_without_read_books(client):
    body = client.post("/recommend", json={"user_id": "u3", "k": 10}).json()
    ids = [i["book_id"] for i in body["items"]]
    assert body["strategy"] == "lightgcn"
    assert len(ids) == 10 and len(set(ids)) == 10
    assert not set(ids) & _history(main.rec, "u3")
    assert body["model_version"] == "vtest"


def test_known_user_scores_are_descending(client):
    items = client.post("/recommend", json={"user_id": "u1", "k": 15}).json()["items"]
    scores = [i["score"] for i in items]
    assert scores == sorted(scores, reverse=True)


def test_unknown_user_with_recent_books_not_popular(client):
    body = client.post(
        "/recommend", json={"user_id": "nobody", "recent_book_ids": ["b1", "b2"], "k": 5}
    ).json()
    ids = {i["book_id"] for i in body["items"]}
    assert body["strategy"] == "item_avg"
    assert len(ids) == 5
    assert not ids & {"b1", "b2"}  # the books they just read are not recommended back


def test_recent_books_without_user_id_also_work(client):
    body = client.post("/recommend", json={"recent_book_ids": ["b4"], "k": 3}).json()
    assert body["strategy"] == "item_avg"


def test_empty_request_returns_popular(client):
    body = client.post("/recommend", json={"k": 7}).json()
    assert body["strategy"] == "popular"
    assert len(body["items"]) == 7


def test_unknown_user_and_unknown_books_fall_back_to_popular(client):
    body = client.post(
        "/recommend", json={"user_id": "nobody", "recent_book_ids": ["zzz"], "k": 4}
    ).json()
    assert body["strategy"] == "popular"


@pytest.mark.parametrize("k", [0, 51, -1])
def test_k_out_of_range_422(client, k):
    assert client.post("/recommend", json={"k": k}).status_code == 422


def test_default_k_is_10(client):
    assert len(client.post("/recommend", json={}).json()["items"]) == 10


def test_book_lookup(client):
    body = client.get("/books/b5").json()
    assert body["title"] == "Book 5"
    assert body["authors"] == ["a5"]
    assert client.get("/books/missing").status_code == 404


def test_request_log_is_json_with_strategy(client, caplog):
    with caplog.at_level(logging.INFO, logger="bookrec"):
        client.post("/recommend", json={"user_id": "u2", "k": 5})
    record = json.loads(caplog.records[-1].message)
    assert record["strategy"] == "lightgcn"
    assert record["k"] == 5
    assert record["model_version"] == "vtest"
    assert record["latency_ms"] >= 0 and record["request_id"]


def test_503_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "rec", Recommender(tmp_path / "nothing"))
    c = TestClient(main.app)
    assert c.get("/health").json()["status"] == "no_model"
    assert c.post("/recommend", json={}).status_code == 503


def test_faiss_matches_bruteforce(model_dir):
    rec = Recommender(model_dir)
    u = rec.user_to_idx["u7"]
    seen = rec.hist_indices[rec.hist_indptr[u] : rec.hist_indptr[u + 1]]
    scores = rec.item_emb @ rec.user_emb[u]
    scores[seen] = -np.inf
    expected = [rec.idx_to_book[i] for i in np.argsort(-scores)[:10]]
    got = [i["book_id"] for i in rec.recommend("u7", None, 10)[0]]
    assert got == expected
