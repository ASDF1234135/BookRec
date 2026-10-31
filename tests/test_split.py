import numpy as np
import pandas as pd

from bookrec.data.preprocess import clean_and_label, kcore, temporal_split


def _interactions(n=1500, n_users=40, n_books=25, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2012-01-01", tz="UTC")
    df = pd.DataFrame(
        {
            "user_id": rng.integers(0, n_users, n).astype(str),
            "book_id": rng.integers(0, n_books, n).astype(str),
            "date_updated": base + pd.to_timedelta(rng.integers(0, 10**8, n), unit="s"),
        }
    )
    return df.drop_duplicates(["user_id", "book_id"]).reset_index(drop=True)


def test_time_order_train_le_val_le_test():
    splits, _ = temporal_split(_interactions())
    order = ["train", "val", "t1", "t2", "t3"]
    for earlier, later in zip(order, order[1:], strict=False):
        assert splits[earlier]["date_updated"].max() <= splits[later]["date_updated"].min()


def test_no_user_book_pair_in_two_splits():
    splits, _ = temporal_split(_interactions())
    seen = set()
    for part in splits.values():
        pairs = set(zip(part["user_id"], part["book_id"], strict=True))
        assert not (pairs & seen)
        seen |= pairs


def test_eval_splits_only_contain_users_and_books_from_train():
    splits, _ = temporal_split(_interactions())
    users, books = set(splits["train"]["user_id"]), set(splits["train"]["book_id"])
    for name in ("val", "t1", "t2", "t3"):
        assert set(splits[name]["user_id"]) <= users
        assert set(splits[name]["book_id"]) <= books


def test_split_sizes_are_roughly_60_10_10_10_10():
    df = _interactions()
    splits, dropped = temporal_split(df)
    assert abs(len(splits["train"]) / len(df) - 0.6) < 0.02


def test_dropped_rows_are_books_unseen_in_train():
    splits, dropped = temporal_split(_interactions(n=300, n_users=40, n_books=200))
    train_books = set(splits["train"]["book_id"])
    assert len(dropped) > 0
    assert not (set(dropped["book_id"]) & train_books)


def test_clean_and_label_grades_signals_and_filters_dates():
    ok = pd.Timestamp("2015-01-01", tz="UTC")
    df = pd.DataFrame(
        {
            "is_read": [True, True, True, True, False, True, True],
            "rating": [5, 0, 3, 2, 0, 5, 5],
            "date_updated": [
                ok,
                ok,
                ok,
                ok,
                ok,
                pd.Timestamp("2005-01-01", tz="UTC"),  # before 2007
                pd.Timestamp("2030-01-01", tz="UTC"),  # after collection cutoff
            ],
        }
    )
    out = clean_and_label(df)
    # strong (5), positive (unrated read), positive (3); negative / unread / bad dates gone
    assert sorted(out["label"]) == ["positive", "positive", "strong"]
    assert sorted(out["rating"]) == [0, 3, 5]


def test_kcore_removes_iteratively_until_stable():
    rows = [("a", "1"), ("a", "2"), ("b", "1"), ("b", "2"), ("c", "3"), ("d", "1"), ("d", "4")]
    df = pd.DataFrame(rows, columns=["user_id", "book_id"])
    out = kcore(df, k=2)
    # c (1 interaction) goes, then book 3; d keeps only book 1 after book 4 goes, then d goes
    assert set(out["user_id"]) == {"a", "b"}
    assert set(out["book_id"]) == {"1", "2"}
    assert (out["user_id"].value_counts() >= 2).all()
    assert (out["book_id"].value_counts() >= 2).all()


def test_kcore_subsampling_is_seeded_and_capped():
    df = _interactions(n=4000, n_users=100, n_books=30)
    a = kcore(df, k=3, max_users=20, seed=1)
    b = kcore(df, k=3, max_users=20, seed=1)
    assert a["user_id"].nunique() <= 20
    assert a.equals(b)
