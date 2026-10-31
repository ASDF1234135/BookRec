"""Popularity baseline."""

import numpy as np


class Popularity:
    def fit(self, train_pos: dict, n_items: int):
        """Count positives per book. train_pos maps user -> iterable of book indices."""
        self.n_items = n_items
        self.counts = np.zeros(n_items, dtype=np.float32)
        for items in train_pos.values():
            # Each (user, book) pair appears once after dedupe, so this is the positive count.
            self.counts[list(items)] += 1
        return self

    def score(self, user_ids) -> np.ndarray:
        """Same popularity row for every user -> (len(user_ids), n_items)."""
        return np.tile(self.counts, (len(user_ids), 1))
