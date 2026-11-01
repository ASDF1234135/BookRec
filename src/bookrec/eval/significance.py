"""Paired significance tests on per-user NDCG, and a small run-vs-run comparison CLI.

python -m bookrec.eval.significance --a <run_id> --b <run_id> [--segments]
"""

import argparse
import tempfile

import numpy as np
import pandas as pd
from scipy import stats


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test on paired per-user NDCG@20 (a vs b), plus the effect size."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must be paired (same users, same order)")
    diff = a - b
    if np.all(diff == 0):  # scipy raises when every difference is zero
        p_value, statistic = 1.0, 0.0
    else:
        res = stats.wilcoxon(a, b)
        p_value, statistic = float(res.pvalue), float(res.statistic)
    return {
        "n": int(len(a)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "relative_diff": float(diff.mean() / b.mean()) if b.mean() else float("nan"),
        "p_value": p_value,
        "statistic": statistic,
    }


def load_per_user(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    import mlflow

    from bookrec.train import TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="per_user_ndcg.npz", dst_path=tmp
        )
        # Close the npz before the temp dir is removed; Windows can't delete open files.
        with np.load(path) as data:
            return data["user_ids"], data["ndcg"]


def compare_runs(run_a: str, run_b: str) -> dict:
    """Align two runs on the users both evaluated, then test a vs b."""
    ua, na = load_per_user(run_a)
    ub, nb = load_per_user(run_b)
    common, ia, ib = np.intersect1d(ua, ub, return_indices=True)
    return {**wilcoxon_paired(na[ia], nb[ib]), "users": common, "ndcg_a": na[ia]}


def segment_means(users: np.ndarray, ndcg: np.ndarray, train_path="data/processed/train.parquet"):
    """Mean NDCG by how many training interactions the user has: <20, 20-100, >100."""
    counts = pd.read_parquet(train_path, columns=["user_idx"])["user_idx"].value_counts()
    c = counts.reindex(users).fillna(0).to_numpy()
    groups = {"<20": c < 20, "20-100": (c >= 20) & (c <= 100), ">100": c > 100}
    return {
        name: (int(m.sum()), float(ndcg[m].mean()) if m.any() else float("nan"))
        for name, m in groups.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run id of the model expected to be better")
    ap.add_argument("--b", required=True, help="run id of the baseline")
    ap.add_argument("--segments", action="store_true", help="also report NDCG by user activity")
    args = ap.parse_args()

    res = compare_runs(args.a, args.b)
    print(
        f"n={res['n']}  mean_a={res['mean_a']:.4f}  mean_b={res['mean_b']:.4f}  "
        f"diff={res['mean_diff']:+.4f} ({res['relative_diff']:+.1%})  p={res['p_value']:.3g}"
    )
    if args.segments:
        for label, run in (("a", args.a), ("b", args.b)):
            users, ndcg = load_per_user(run)
            print(f"segments for {label}:", segment_means(users, ndcg))


if __name__ == "__main__":
    main()
