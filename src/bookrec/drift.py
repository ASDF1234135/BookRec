"""Drift simulation: evaluate a FROZEN model on the later time windows T1, T2, T3.

    python -m bookrec.drift --run-id <id>

NDCG@20 falling from T1 to T3 is the cost of not retraining. Cheap drift signals are printed
next to it. The "retrain after every window" control group is not automated here: train
again on data up to each window and compare (see TODO.md).
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from bookrec.eval.metrics import evaluate
from bookrec.train import TRACKING_URI, to_pos

PROCESSED = Path("data/processed")
WINDOWS = ["t1", "t2", "t3"]


def top_books(df: pd.DataFrame, n: int = 20) -> set:
    return set(df["book_idx"].value_counts().head(n).index)


def run_drift(run_id: str, data_dir: Path = PROCESSED, k: int = 20) -> list[dict]:
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
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
    n_items = item_emb.shape[0]

    splits = {s: pd.read_parquet(data_dir / f"{s}.parquet") for s in ["train", "val", *WINDOWS]}
    dropped = pd.read_parquet(data_dir / "cold_start_dropped.parquet")
    baseline_top = top_books(pd.concat([splits["train"], splits["val"]]))

    def score_fn(user_ids):
        return user_emb[user_ids] @ item_emb.T

    rows = []
    for i, window in enumerate(WINDOWS):
        # Everything that happened before this window is "already read" at ranking time.
        history = pd.concat([splits[s] for s in ["train", "val", *WINDOWS[:i]]])
        part = splits[window]
        res = evaluate(score_fn, to_pos(history), to_pos(part), n_items, k)

        start, end = part["date_updated"].min(), part["date_updated"].max()
        in_window = dropped[(dropped["date_updated"] >= start) & (dropped["date_updated"] <= end)]
        rows.append(
            {
                "window": window,
                f"ndcg_at_{k}": res[f"ndcg@{k}"],
                f"recall_at_{k}": res[f"recall@{k}"],
                f"coverage_at_{k}": res[f"coverage@{k}"],
                "rows": len(part),
                # share of that period's interactions on books the model never saw
                "unseen_book_share": len(in_window) / (len(in_window) + len(part)),
                # how many of the top-20 books are still the same as in train+val
                "top20_overlap": len(baseline_top & top_books(part)) / 20,
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    rows = run_drift(args.run_id)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/drift.json").write_text(
        json.dumps({"run_id": args.run_id, "windows": rows}, indent=2)
    )


if __name__ == "__main__":
    main()
