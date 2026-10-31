"""Export serving artifacts for one MLflow run -> artifacts/v<N>/.

The output folder is what the service loads (and what gets baked into the Docker image):
item_emb.npy, user_emb.npy, faiss.index, id_map.json, popular.json, books_meta.parquet,
user_history.npz, metrics.json, model_card.json
"""

import argparse
import json
import re
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
RAW_BOOKS = Path("data/raw/goodreads_books_comics_graphic.json.gz")
ARTIFACTS = Path("artifacts")
SPLITS = ["train", "val", "t1", "t2", "t3"]


def next_version(artifacts_dir: Path = ARTIFACTS) -> int:
    nums = [
        int(m.group(1)) for p in artifacts_dir.glob("v*") if (m := re.fullmatch(r"v(\d+)", p.name))
    ]
    return max(nums, default=0) + 1


def invert(mapping: dict) -> list:
    """{external_id: idx} -> list where list[idx] = external_id."""
    out = [None] * len(mapping)
    for key, idx in mapping.items():
        out[idx] = key
    return out


def load_books_meta(book_ids: set[str]) -> pd.DataFrame:
    """book_id, title, authors (author ids), average_rating for the books we serve."""
    frames = []
    reader = pd.read_json(
        RAW_BOOKS, lines=True, dtype={"book_id": str}, convert_dates=False, chunksize=20_000
    )
    for chunk in reader:
        chunk = chunk[chunk["book_id"].isin(book_ids)]
        if len(chunk):
            frames.append(chunk[["book_id", "title", "authors", "average_rating"]])
    meta = pd.concat(frames, ignore_index=True)
    meta["authors"] = meta["authors"].map(
        lambda xs: [str(a["author_id"]) for a in xs] if isinstance(xs, list) else []
    )
    meta["average_rating"] = pd.to_numeric(meta["average_rating"], errors="coerce")
    return meta.drop_duplicates("book_id")


def write_artifacts(
    out_dir,
    *,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    id_map: dict,
    books_meta: pd.DataFrame,
    hist_users: np.ndarray,
    hist_items: np.ndarray,
    popular_n: int = 200,
    metrics: dict | None = None,
    model_card: dict | None = None,
) -> Path:
    """Write every file the service needs. Shared by the real export and the test fixtures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    user_emb = np.ascontiguousarray(user_emb, dtype=np.float32)
    item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
    n_users, n_items = len(user_emb), len(item_emb)

    np.save(out / "user_emb.npy", user_emb)
    np.save(out / "item_emb.npy", item_emb)

    # Exact inner-product search; the model scores are dot products, so this matches training.
    index = faiss.IndexFlatIP(item_emb.shape[1])
    index.add(item_emb)
    faiss.write_index(index, str(out / "faiss.index"))

    (out / "id_map.json").write_text(json.dumps(id_map))

    # Per-user read history in CSR layout (indptr/indices), used to exclude read books.
    hist_users = np.asarray(hist_users, dtype=np.int64)
    hist_items = np.asarray(hist_items, dtype=np.int64)
    order = np.argsort(hist_users, kind="stable")
    indptr = np.concatenate([[0], np.cumsum(np.bincount(hist_users, minlength=n_users))])
    np.savez(out / "user_history.npz", indptr=indptr, indices=hist_items[order])

    # Fallback list for requests we cannot personalise.
    counts = np.bincount(hist_items, minlength=n_items)
    top = np.argsort(-counts, kind="stable")[:popular_n]
    idx_to_book = invert(id_map["book"])
    (out / "popular.json").write_text(
        json.dumps(
            {"book_ids": [idx_to_book[i] for i in top], "counts": [int(counts[i]) for i in top]}
        )
    )

    books_meta.to_parquet(out / "books_meta.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics or {}, indent=2))
    card = {
        "n_users": n_users,
        "n_items": n_items,
        "dim": int(item_emb.shape[1]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(model_card or {}),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    return out


def export(run_id: str, out_dir: str | Path | None = None, version: int | None = None) -> Path:
    """Export one MLflow run (must be a LightGCN/MF run, which stores embeddings)."""
    import mlflow  # imported here so tests of write_artifacts stay light

    from bookrec.train import TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    run = mlflow.get_run(run_id)
    if run.data.params.get("model") == "popularity":
        raise ValueError("popularity runs have no embeddings; export a lightgcn/mf run")

    # Use the data the run was trained on (smoke runs point data_dir at tests/fixtures).
    data_dir = Path(run.data.params.get("data_dir", PROCESSED))
    data_hash = run.data.tags.get("data_hash")
    card_path = data_dir / "data_card.json"
    if card_path.exists() and json.loads(card_path.read_text())["data_hash"] != data_hash:
        raise ValueError(
            f"run was trained on different data than {data_dir} now holds; "
            "retrain or re-run preprocess"
        )

    if out_dir is None:
        version = next_version()
        out_dir = ARTIFACTS / f"v{version}"
    version_name = f"v{version}" if version is not None else "dev"

    with tempfile.TemporaryDirectory() as tmp:
        user_emb = np.load(
            mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="user_emb.npy", dst_path=tmp
            )
        )
        item_emb = np.load(
            mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="item_emb.npy", dst_path=tmp
            )
        )

    id_map_path = data_dir / "id_map.json"
    if id_map_path.exists():
        id_map = json.loads(id_map_path.read_text())
    else:  # fake data has no id map: external id is just the index as a string
        id_map = {
            "user": {str(i): i for i in range(len(user_emb))},
            "book": {str(i): i for i in range(len(item_emb))},
        }
    # History = every known interaction (all splits) so already-read books are always excluded.
    hist = pd.concat(
        [
            pd.read_parquet(data_dir / f"{s}.parquet", columns=["user_idx", "book_idx"])
            for s in SPLITS
            if (data_dir / f"{s}.parquet").exists()
        ]
    )
    if RAW_BOOKS.exists():
        meta = load_books_meta(set(id_map["book"]))
    else:  # no raw metadata (CI): placeholder titles keep the API shape intact
        ids = list(id_map["book"])
        meta = pd.DataFrame(
            {
                "book_id": ids,
                "title": [f"Book {b}" for b in ids],
                "authors": [[] for _ in ids],
                "average_rating": [float("nan")] * len(ids),
            }
        )

    metrics = {k: float(v) for k, v in run.data.metrics.items() if k != "train_loss"}
    return write_artifacts(
        out_dir,
        user_emb=user_emb,
        item_emb=item_emb,
        id_map=id_map,
        books_meta=meta,
        hist_users=hist["user_idx"].to_numpy(),
        hist_items=hist["book_idx"].to_numpy(),
        metrics=metrics,
        model_card={
            "version": version_name,
            "run_id": run_id,
            "data_hash": data_hash,
            "git_hash": run.data.tags.get("git_hash", "unknown"),
            "eval_split": run.data.tags.get("eval_split"),
            "train_time_s": metrics.get("train_time_s"),
            "params": run.data.params,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", help="output folder (default artifacts/v<next>)")
    ap.add_argument("--version", type=int, help="version number to record when --out is given")
    args = ap.parse_args()
    out = export(args.run_id, args.out, args.version)
    print(f"exported to {out}")


if __name__ == "__main__":
    main()
