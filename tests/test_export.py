import json

import faiss
import numpy as np

from bookrec.eval.significance import wilcoxon_paired
from bookrec.export import next_version
from bookrec.fixtures import make_fixture_model, make_smoke_data

FILES = {
    "item_emb.npy",
    "user_emb.npy",
    "faiss.index",
    "id_map.json",
    "popular.json",
    "books_meta.parquet",
    "user_history.npz",
    "metrics.json",
    "model_card.json",
}


def test_export_writes_every_artifact(tmp_path):
    out = make_fixture_model(tmp_path / "v1")
    assert FILES <= {p.name for p in out.iterdir()}
    card = json.loads((out / "model_card.json").read_text())
    assert card["n_users"] == 50 and card["n_items"] == 100 and card["dim"] == 8


def test_faiss_index_matches_item_embeddings(tmp_path):
    out = make_fixture_model(tmp_path / "v1")
    items = np.load(out / "item_emb.npy")
    index = faiss.read_index(str(out / "faiss.index"))
    assert index.ntotal == len(items)
    q = np.load(out / "user_emb.npy")[:1]
    _, ids = index.search(q, 5)
    assert ids[0].tolist() == np.argsort(-(items @ q[0]))[:5].tolist()


def test_history_csr_roundtrip(tmp_path):
    out = make_fixture_model(tmp_path / "v1")
    h = np.load(out / "user_history.npz")
    assert len(h["indptr"]) == 51 and h["indptr"][-1] == len(h["indices"])
    assert (np.diff(h["indptr"]) >= 5).all()


def test_popular_is_sorted_by_count(tmp_path):
    out = make_fixture_model(tmp_path / "v1")
    pop = json.loads((out / "popular.json").read_text())
    assert pop["counts"] == sorted(pop["counts"], reverse=True)


def test_next_version(tmp_path):
    assert next_version(tmp_path) == 1
    (tmp_path / "v1").mkdir()
    (tmp_path / "v4").mkdir()
    (tmp_path / "notes").mkdir()
    assert next_version(tmp_path) == 5


def test_smoke_data_has_three_splits(tmp_path):
    out = make_smoke_data(tmp_path)
    assert {p.stem for p in out.glob("*.parquet")} == {"train", "val", "t1"}


def test_wilcoxon_detects_a_clear_improvement():
    rng = np.random.default_rng(0)
    b = rng.random(300) * 0.1
    res = wilcoxon_paired(b + 0.05, b)
    assert res["p_value"] < 1e-6 and res["mean_diff"] > 0.04


def test_wilcoxon_identical_inputs_are_not_significant():
    x = np.array([0.1, 0.2, 0.3])
    assert wilcoxon_paired(x, x)["p_value"] == 1.0
