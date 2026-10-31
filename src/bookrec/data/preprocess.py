"""Raw Goodreads JSON -> cleaned positives, id maps, temporal splits, data_card.json."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
MIN_DATE = pd.Timestamp("2007-01-01", tz="UTC")
# Goodreads data was scraped in late 2017; check df["date_updated"].max() and adjust if needed.
COLLECTION_CUTOFF = pd.Timestamp("2017-12-31", tz="UTC")


def load_interactions(path: Path):
    """Parse JSON lines, keep last record per (user, book)."""
    # FIX: force ids to str (hex user_ids can be mis-parsed as numbers) and stop pandas
    # from guessing date columns, since we parse them ourselves below.
    df = pd.read_json(
        path,
        lines=True,
        dtype={"user_id": str, "book_id": str},
        convert_dates=False,
    )

    # FIX: errors="coerce" turns empty/odd date strings into NaT instead of crashing;
    # utc=True avoids mixed-timezone problems from the "-0700"/"-0800" offsets.
    for col in ["date_added", "date_updated"]:
        df[col] = pd.to_datetime(df[col], format=DATE_FORMAT, errors="coerce", utc=True)

    # FIX: event time is date_updated (per the plan), so rows without it are unusable.
    df = df.dropna(subset=["date_updated"])

    # FIX (was missing): keep only the latest record per (user, book).
    df = df.sort_values("date_updated").drop_duplicates(["user_id", "book_id"], keep="last")

    # FIX: df.type() does not exist; use dtypes.
    print(df.head())
    print(df.dtypes)
    return df.reset_index(drop=True)


def clean_and_label(df):
    """Filter dates, grade signals (strong / positive / weak / negative), keep positives only."""
    # FIX: compare against a tz-aware Timestamp, not a bare string, and add the upper cutoff.
    df = df[(df["date_updated"] >= MIN_DATE) & (df["date_updated"] <= COLLECTION_CUTOFF)].copy()

    # np.select takes the FIRST matching condition, so order matters.
    conditions = [
        (df["is_read"] == False),  # noqa: E712
        (df["is_read"] == True) & (df["rating"] >= 4),  # noqa: E712
        # FIX: rating == 0 means "read but unrated" -> positive per the plan.
        # Before, these fell through to the "weak" default.
        (df["is_read"] == True) & ((df["rating"] == 3) | (df["rating"] == 0)),  # noqa: E712
        (df["is_read"] == True) & (df["rating"] > 0) & (df["rating"] < 3),  # noqa: E712
    ]
    choices = ["weak", "strong", "positive", "negative"]
    df["label"] = np.select(conditions, choices, default="weak")

    # FIX: print the frame that actually has the "label" column (was `df` -> KeyError).
    print(df["label"].value_counts())

    # FIX (was missing): main experiment uses only strong + positive signals.
    # Weak/negative are labelled above but dropped here; keep them in a separate frame
    # if you want them for later weighting experiments.
    return df[df["label"].isin(["strong", "positive"])].reset_index(drop=True)


def kcore(df, k: int = 10, max_users: int | None = None, seed: int = 0):
    """Iterative k-core on users and books; optional user subsampling (~60k).

    Order per the plan: k-core first, THEN subsample if too many users, then
    re-run k-core because dropping users can push books below k.
    """
    df = _kcore_once(df, k)

    # FIX: subsampling used to run before the k-core and took a bool as `size`.
    # Now it is optional (None = off), seeded for reproducibility, and runs after the core.
    if max_users is not None and df["user_id"].nunique() > max_users:
        rng = np.random.default_rng(seed)
        sampled = rng.choice(df["user_id"].unique(), size=max_users, replace=False)
        df = df[df["user_id"].isin(sampled)]
        df = _kcore_once(df, k)

    return df.reset_index(drop=True)


def _kcore_once(df, k: int):
    # Removing users can drop books below k and vice versa, so loop until nothing changes.
    prev_shape = None
    while prev_shape != df.shape:
        prev_shape = df.shape

        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= k].index)]

        item_counts = df["book_id"].value_counts()
        df = df[df["book_id"].isin(item_counts[item_counts >= k].index)]
    # FIX: the original loop never returned df.
    return df


def temporal_split(df):
    """Quantile split 60/10/10/10/10 -> train, val, T1, T2, T3.

    Returns (splits, dropped) where splits is a dict of DataFrames and dropped holds rows
    whose book is not in train (cold-start analysis).

    Idea: pretend data arrives in time order. Everyone shares the same cut dates, so the
    model trains on the past and is tested on the future.
    """
    # FIX: split on date_updated (event time per the plan), not date_added.
    ts = "date_updated"

    # Work on integer timestamps so quantile/searchsorted behave the same regardless of tz.
    ts_int = df[ts].astype("int64")
    # Cut points at the 60/70/80/90% quantiles of TIME (not of row index). Using time
    # means rows with an identical timestamp can never straddle a boundary.
    cuts = ts_int.quantile([0.6, 0.7, 0.8, 0.9]).to_numpy()

    # bucket 0 = before the 60% cut ... bucket 4 = after the 90% cut.
    bucket = np.searchsorted(cuts, ts_int.to_numpy(), side="right")
    names = ["train", "val", "t1", "t2", "t3"]
    parts = {name: df[bucket == i] for i, name in enumerate(names)}

    train_users = set(parts["train"]["user_id"])
    train_books = set(parts["train"]["book_id"])

    splits = {"train": parts["train"].reset_index(drop=True)}
    dropped = []
    for name in names[1:]:
        part = parts[name]
        # A model cannot score a user/book it never saw in training, so drop them.
        known = part["user_id"].isin(train_users) & part["book_id"].isin(train_books)
        splits[name] = part[known].reset_index(drop=True)
        # Unseen BOOKS are kept aside for the (optional) two-tower cold-start analysis.
        dropped.append(part[~part["book_id"].isin(train_books)])

    return splits, pd.concat(dropped).reset_index(drop=True)


def main():
    """Run the whole pipeline and write parquet splits, id_map.json and data_card.json."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_interactions(RAW_DIR / "goodreads_interactions_comics_graphic.json.gz")
    pos = clean_and_label(raw)
    core = kcore(pos, k=10, max_users=60_000)
    splits, dropped = temporal_split(core)

    # Id maps come from TRAIN only: every user/book in val/test is already in train
    # (temporal_split guarantees it), so the maps cover all splits.
    users = sorted(splits["train"]["user_id"].unique())
    books = sorted(splits["train"]["book_id"].unique())
    user_map = {u: i for i, u in enumerate(users)}
    book_map = {b: i for i, b in enumerate(books)}

    keep_cols = ["user_id", "book_id", "date_updated", "label"]
    split_stats = {}
    for name, part in splits.items():
        out = part[keep_cols].copy()
        out["user_idx"] = out["user_id"].map(user_map).astype("int32")
        out["book_idx"] = out["book_id"].map(book_map).astype("int32")
        out.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        split_stats[name] = {
            "rows": len(out),
            "users": int(out["user_idx"].nunique()),
            "books": int(out["book_idx"].nunique()),
            "start": str(out["date_updated"].min()),
            "end": str(out["date_updated"].max()),
        }

    # Books unseen in train: kept for the optional two-tower cold-start analysis.
    # These are not in book_map, so they keep their original ids only.
    dropped[["user_id", "book_id", "date_updated"]].to_parquet(
        OUT_DIR / "cold_start_dropped.parquet", index=False
    )

    with open(OUT_DIR / "id_map.json", "w") as f:
        json.dump({"user": user_map, "book": book_map}, f)

    # Hash of the processed train file; training logs it so results trace back to the data.
    data_hash = hashlib.sha256((OUT_DIR / "train.parquet").read_bytes()).hexdigest()[:16]

    card = {
        "data_hash": data_hash,
        "pipeline": {
            "raw_rows_after_load": len(raw),
            "positives_after_date_filter_and_label": len(pos),
            "rows_after_kcore": len(core),
            "kcore_k": 10,
            "max_users": 60_000,
            "dropped_cold_start_rows": len(dropped),
        },
        "splits": split_stats,
        "n_users": len(user_map),
        "n_books": len(book_map),
    }
    with open(OUT_DIR / "data_card.json", "w") as f:
        json.dump(card, f, indent=2)

    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
