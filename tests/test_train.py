"""End-to-end smoke test: tiny processed dataset -> train -> evaluate -> MLflow."""

import numpy as np
import pandas as pd
import pytest

from bookrec import train


@pytest.fixture
def data_dir(tmp_path):
    # User u likes the 10 items of block u // 20, split by order into train/val/t1.
    rng = np.random.default_rng(0)
    rows = []
    for u in range(200):
        block = rng.permutation(np.arange(u // 20 * 10, u // 20 * 10 + 10))
        rows += [(u, int(b), j) for j, b in enumerate(block)]
    df = pd.DataFrame(rows, columns=["user_idx", "book_idx", "order"])
    d = tmp_path / "data"
    d.mkdir()
    for name, part in {
        "train": df[df.order < 6],
        "val": df[(df.order >= 6) & (df.order < 8)],
        "t1": df[df.order >= 8],
    }.items():
        part.drop(columns="order").to_parquet(d / f"{name}.parquet", index=False)
    return d


def _cfg(data_dir, **kw):
    cfg = {
        "model": "lightgcn",
        "seed": 0,
        "dim": 16,
        "n_layers": 2,
        "lr": 0.02,
        "l2": 1e-4,
        "batch_size": 128,
        "max_epochs": 15,
        "eval_every": 5,
        "patience": 3,
        "k": 20,
        "data_dir": str(data_dir),
    }
    cfg.update(kw)
    return cfg


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # mlruns/ artifacts land in tmp
    monkeypatch.setattr(train, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")


def test_lightgcn_beats_popularity(data_dir):
    pop = train.run(_cfg(data_dir, model="popularity"))[0]
    gcn = train.run(_cfg(data_dir))[0]
    assert gcn["ndcg_at_20"] > pop["ndcg_at_20"] * 2
    assert {"ndcg_at_20", "recall_at_20", "coverage_at_20", "train_time_s"} <= set(gcn)


def test_final_mode_evaluates_on_t1(data_dir):
    res = train.run(_cfg(data_dir, final=True, final_epochs=5))[0]
    assert 0.0 <= res["ndcg_at_20"] <= 1.0


def test_all_seeds_makes_one_run_per_seed(data_dir):
    out = train.run(_cfg(data_dir, seeds=[0, 1], max_epochs=5), all_seeds=True)
    assert len(out) == 2
    assert out[0]["run_id"] != out[1]["run_id"]
