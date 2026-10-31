"""Tiny fake data and a fake serving model, for CI and tests.

python -m bookrec.fixtures --smoke-data tests/fixtures   # fake processed parquet
python -m bookrec.fixtures --model service/model         # fake exported model
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bookrec.export import write_artifacts


def make_smoke_data(out_dir="tests/fixtures", seed: int = 0) -> Path:
    """200 users in 10 taste blocks (user likes 10 books of its block), split by order."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(200):
        block = rng.permutation(np.arange(u // 20 * 10, u // 20 * 10 + 10))
        rows += [(u, int(b), j) for j, b in enumerate(block)]
    df = pd.DataFrame(rows, columns=["user_idx", "book_idx", "order"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": df[df.order < 6],
        "val": df[(df.order >= 6) & (df.order < 8)],
        "t1": df[df.order >= 8],
    }
    for name, part in splits.items():
        part.drop(columns="order").to_parquet(out / f"{name}.parquet", index=False)
    return out


def make_fixture_model(out_dir, n_users=50, n_items=100, dim=8, seed=0) -> Path:
    """Random embeddings with the real artifact layout. Book ids are b0.., users u0.."""
    rng = np.random.default_rng(seed)
    id_map = {
        "user": {f"u{i}": i for i in range(n_users)},
        "book": {f"b{i}": i for i in range(n_items)},
    }
    users, items = [], []
    for u in range(n_users):
        seen = rng.choice(n_items, size=int(rng.integers(5, 16)), replace=False)
        users += [u] * len(seen)
        items += seen.tolist()
    meta = pd.DataFrame(
        {
            "book_id": [f"b{i}" for i in range(n_items)],
            "title": [f"Book {i}" for i in range(n_items)],
            "authors": [[f"a{i % 7}"] for i in range(n_items)],
            "average_rating": [3.0 + (i % 20) / 10 for i in range(n_items)],
        }
    )
    return write_artifacts(
        out_dir,
        user_emb=rng.normal(size=(n_users, dim)),
        item_emb=rng.normal(size=(n_items, dim)),
        id_map=id_map,
        books_meta=meta,
        hist_users=np.array(users),
        hist_items=np.array(items),
        metrics={"ndcg_at_20": 0.1, "coverage_at_20": 0.5},
        model_card={"version": "vtest", "run_id": "fixture", "data_hash": "fixture"},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-data")
    ap.add_argument("--model")
    args = ap.parse_args()
    if args.smoke_data:
        print("wrote", make_smoke_data(args.smoke_data))
    if args.model:
        print("wrote", make_fixture_model(args.model))


if __name__ == "__main__":
    main()
