"""Loads artifacts once and routes requests to a strategy."""

import json
import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


class Recommender:
    def __init__(self, model_dir: str | None = None):
        self.model_dir = Path(model_dir or os.environ.get("MODEL_DIR", "model"))
        self.loaded = False
        self.model_version = "unloaded"
        if (self.model_dir / "model_card.json").exists():
            self._load()

    def _load(self):
        d = self.model_dir
        card = json.loads((d / "model_card.json").read_text())
        self.model_version = card["version"]

        self.user_emb = np.load(d / "user_emb.npy")
        self.item_emb = np.load(d / "item_emb.npy")
        self.index = faiss.read_index(str(d / "faiss.index"))
        self.n_items = self.item_emb.shape[0]

        id_map = json.loads((d / "id_map.json").read_text())
        self.user_to_idx: dict[str, int] = id_map["user"]
        self.book_to_idx: dict[str, int] = id_map["book"]
        self.idx_to_book: list[str] = [None] * self.n_items
        for book_id, idx in self.book_to_idx.items():
            self.idx_to_book[idx] = book_id

        hist = np.load(d / "user_history.npz")
        self.hist_indptr, self.hist_indices = hist["indptr"], hist["indices"]

        popular = json.loads((d / "popular.json").read_text())
        self.popular_ids: list[str] = popular["book_ids"]
        self.popular_counts: list[int] = popular["counts"]

        meta = pd.read_parquet(d / "books_meta.parquet")
        self.meta = meta.set_index("book_id").to_dict("index")
        self.loaded = True

    def _title(self, book_id: str):
        row = self.meta.get(book_id)
        return row["title"] if row else None

    def _search(self, vec: np.ndarray, seen: np.ndarray, k: int):
        # Fetch enough candidates that removing every already-read book still leaves k.
        n_cand = min(k + len(seen), self.n_items)
        scores, ids = self.index.search(vec.reshape(1, -1).astype(np.float32), n_cand)
        scores, ids = scores[0], ids[0]
        keep = ~np.isin(ids, seen) & (ids >= 0)
        return [
            {
                "book_id": self.idx_to_book[i],
                "title": self._title(self.idx_to_book[i]),
                "score": float(s),
            }
            for i, s in zip(ids[keep][:k], scores[keep][:k], strict=True)
        ]

    def _popular(self, k: int, exclude: set[str]):
        items = []
        for book_id, count in zip(self.popular_ids, self.popular_counts, strict=True):
            if book_id in exclude:
                continue
            items.append({"book_id": book_id, "title": self._title(book_id), "score": float(count)})
            if len(items) == k:
                break
        return items

    def recommend(self, user_id, recent_book_ids, k: int) -> tuple[list[dict], str]:
        """Route: known user -> lightgcn; else recent books -> item_avg; else popular."""
        u = self.user_to_idx.get(user_id) if user_id is not None else None
        if u is not None:
            seen = self.hist_indices[self.hist_indptr[u] : self.hist_indptr[u + 1]]
            return self._search(self.user_emb[u], seen, k), "lightgcn"

        recent = [b for b in (recent_book_ids or []) if b in self.book_to_idx]
        if recent:
            idx = np.array([self.book_to_idx[b] for b in recent])
            # No user vector for a new user: use the mean of the books they just read.
            return self._search(self.item_emb[idx].mean(axis=0), idx, k), "item_avg"

        return self._popular(k, set(recent_book_ids or [])), "popular"

    def book(self, book_id: str) -> dict | None:
        row = self.meta.get(book_id)
        if row is None:
            return None
        return {
            "book_id": book_id,
            "title": row["title"],
            "authors": list(row["authors"]),
            "average_rating": None
            if pd.isna(row["average_rating"])
            else float(row["average_rating"]),
        }
