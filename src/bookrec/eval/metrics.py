"""Ranking metrics and the single shared evaluate()."""

import numpy as np


def ndcg_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """TODO(you): binary-relevance NDCG@k for one user."""
    if k <= 0 or len(relevant) == 0 or len(ranked) == 0:
        return 0.0

    ranked = ranked[:k]
    hits = np.array([1 if item in relevant else 0 for item in ranked])

    if hits.sum() == 0:
        return 0.0

    ideal_len = min(len(relevant), k)
    # FIX: size the discounts for the longer of the two, otherwise a ranked list shorter
    # than the ideal list gives a wrong (too small) ideal DCG.
    discounts = np.log2(np.arange(2, max(len(hits), ideal_len) + 2))
    dcg = np.sum(hits / discounts[: len(hits)])

    ideal_hits = np.ones(ideal_len)
    idcg = np.sum(ideal_hits / discounts[:ideal_len])

    if idcg == 0.0:
        return 0.0

    return float(dcg / idcg)


def recall_at_k(ranked: np.ndarray, relevant: set[int], k: int) -> float:
    """TODO(you)"""
    if k <= 0 or len(relevant) == 0 or len(ranked) == 0:
        return 0.0

    ranked = ranked[:k]
    hits = np.array([1 if item in relevant else 0 for item in ranked])

    return float(hits.sum() / len(relevant))


def coverage_at_k(all_recs: np.ndarray, n_items: int) -> float:
    """TODO(you): fraction of catalog appearing in any top-k list."""
    if n_items <= 0 or all_recs.size == 0:
        return 0.0

    unique_recommended_items = np.unique(all_recs)

    coverage = len(unique_recommended_items) / n_items

    return float(coverage)


def evaluate(
    score_fn,
    train_pos: dict,
    target_pos: dict,
    n_items: int,
    k: int = 20,
    batch_size: int = 1024,
) -> dict:
    """Full-ranking evaluation used by EVERY model.

    score_fn(user_ids: np.ndarray) -> (len(user_ids), n_items) scores (numpy or torch).
    train_pos: user -> items to EXCLUDE from the ranking (train for validation;
               pass train+val merged when testing).
    target_pos: user -> held-out positives (val or test).
    Returns the mean of each per-user metric, plus per_user_ndcg aligned with user_ids
    (for the Wilcoxon test).
    """
    # Users with no target positives are skipped, so they never dilute the averages.
    users = np.array(sorted(u for u, items in target_pos.items() if len(items) > 0))
    if len(users) == 0:
        raise ValueError("target_pos has no users with positives")

    # Item popularity (count of exclusion-set interactions), used for avg_pop.
    pop = np.zeros(n_items, dtype=np.float64)
    for items in train_pos.values():
        pop[list(items)] += 1

    top = min(max(k, 10), n_items)  # enough for both @k and @10
    ndcg_k, ndcg_10, recall_k, rec_pop = [], [], [], []
    all_recs = []

    # Score in batches so the full (users x items) matrix never sits in memory at once.
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        scores = score_fn(batch)
        if hasattr(scores, "detach"):  # torch tensor
            scores = scores.detach().cpu().numpy()
        scores = np.array(scores, dtype=np.float32, copy=True)

        # Mask already-seen items so they can never be recommended.
        for row, u in enumerate(batch):
            seen = train_pos.get(u)
            if seen:
                scores[row, list(seen)] = -np.inf

        # Top-`top` per user: argpartition is O(n), then sort only those few.
        part = np.argpartition(-scores, top - 1, axis=1)[:, :top]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        ranked = np.take_along_axis(part, order, axis=1)

        for row, u in enumerate(batch):
            relevant = set(target_pos[u])
            r = ranked[row]
            ndcg_k.append(ndcg_at_k(r, relevant, k))
            ndcg_10.append(ndcg_at_k(r, relevant, 10))
            recall_k.append(recall_at_k(r, relevant, k))
            rec_pop.append(pop[r[:k]].mean())
        all_recs.append(ranked[:, :k])

    per_user_ndcg = np.array(ndcg_k)
    return {
        f"ndcg@{k}": float(per_user_ndcg.mean()),
        "ndcg@10": float(np.mean(ndcg_10)),
        f"recall@{k}": float(np.mean(recall_k)),
        f"coverage@{k}": coverage_at_k(np.vstack(all_recs), n_items),
        "avg_pop": float(np.mean(rec_pop)),
        "per_user_ndcg": per_user_ndcg,
        "user_ids": users,
    }
