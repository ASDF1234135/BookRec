"""Eval gate. Exits non-zero if the candidate must not be promoted.

python -m bookrec.gate --run-id <id|latest> [--skip-latency]
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PRODUCTION_JSON = Path("deploy/production.json")
REPORT_DIR = Path("gate_reports")

POP_RATIO = 1.2  # candidate NDCG must be >= 1.2 x popularity
NDCG_RATIO = 0.99  # ... and >= 99% of production (1% noise allowed)
COVERAGE_RATIO = 0.90  # ... coverage >= 90% of production
P95_MS = 100.0  # local p95 latency budget


def _check(name: str, passed: bool, detail: str, skipped: bool = False) -> dict:
    return {"name": name, "passed": bool(passed), "skipped": skipped, "detail": detail}


def check(
    candidate: dict,
    production: dict | None,
    popularity: dict,
    latency_p95_ms: float | None,
) -> dict:
    """Apply the 5 promotion conditions.

    candidate:  {data_hash, eval_split, ndcg_at_20, coverage_at_20}
    production: contents of deploy/production.json, or None / version null on first deploy
    popularity: {ndcg_at_20} of the popularity baseline on the same data and split
    """
    first_deploy = production is None or production.get("version") is None
    checks = []

    # 1. Same data and same evaluation split, otherwise the numbers are not comparable.
    if first_deploy:
        checks.append(_check("same_data_as_production", True, "first deploy", skipped=True))
    else:
        same = candidate["data_hash"] == production.get("data_hash") and candidate[
            "eval_split"
        ] == production.get("eval_split")
        checks.append(
            _check(
                "same_data_as_production",
                same,
                f"candidate {candidate['data_hash']}/{candidate['eval_split']} vs "
                f"production {production.get('data_hash')}/{production.get('eval_split')}",
            )
        )

    # 2. Clearly better than popularity.
    need = POP_RATIO * popularity["ndcg_at_20"]
    checks.append(
        _check(
            "beats_popularity",
            candidate["ndcg_at_20"] >= need,
            f"ndcg {candidate['ndcg_at_20']:.4f} vs required {need:.4f} "
            f"({POP_RATIO}x popularity {popularity['ndcg_at_20']:.4f})",
        )
    )

    # 3 and 4. No regression against production.
    if first_deploy:
        checks.append(_check("no_ndcg_regression", True, "first deploy", skipped=True))
        checks.append(_check("no_coverage_regression", True, "first deploy", skipped=True))
    else:
        prod = production["metrics"]
        need = NDCG_RATIO * prod["ndcg_at_20"]
        checks.append(
            _check(
                "no_ndcg_regression",
                candidate["ndcg_at_20"] >= need,
                f"ndcg {candidate['ndcg_at_20']:.4f} vs required {need:.4f} "
                f"({NDCG_RATIO:.0%} of production {prod['ndcg_at_20']:.4f})",
            )
        )
        need = COVERAGE_RATIO * prod["coverage_at_20"]
        checks.append(
            _check(
                "no_coverage_regression",
                candidate["coverage_at_20"] >= need,
                f"coverage {candidate['coverage_at_20']:.4f} vs required {need:.4f} "
                f"({COVERAGE_RATIO:.0%} of production {prod['coverage_at_20']:.4f})",
            )
        )

    # 5. Serving latency.
    if latency_p95_ms is None:
        checks.append(_check("latency_p95", True, "not measured (--skip-latency)", skipped=True))
    else:
        checks.append(
            _check(
                "latency_p95",
                latency_p95_ms < P95_MS,
                f"p95 {latency_p95_ms:.2f} ms vs < {P95_MS} ms",
            )
        )

    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def measure_latency(artifact_dir, n: int = 1000, seed: int = 0) -> float:
    """p95 of n simulated requests (known user / recent books / empty), in milliseconds."""
    sys.path.insert(0, str(Path("service").resolve()))
    from app.recommender import Recommender

    rec = Recommender(artifact_dir)
    rng = np.random.default_rng(seed)
    users, books = list(rec.user_to_idx), list(rec.book_to_idx)

    def request(i):
        kind = i % 3
        if kind == 0:
            return rng.choice(users), None
        if kind == 1:
            return None, list(rng.choice(books, size=3, replace=False))
        return None, None

    for i in range(30):  # warm-up
        rec.recommend(*request(i), 10)
    times = []
    for i in range(n):
        user_id, recent = request(i)
        t = time.perf_counter()
        rec.recommend(user_id, recent, 10)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.percentile(times, 95))


def _mlflow():
    import mlflow

    from bookrec.train import EXPERIMENT, TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow, EXPERIMENT


def resolve_run_id(run_id: str) -> str:
    if run_id != "latest":
        return run_id
    mlflow, experiment = _mlflow()
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string="params.model = 'lightgcn' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit("no finished lightgcn run found")
    return runs.iloc[0]["run_id"]


def find_popularity(data_hash: str, eval_split: str) -> dict:
    mlflow, experiment = _mlflow()
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=(
            "params.model = 'popularity' and attributes.status = 'FINISHED' "
            f"and tags.data_hash = '{data_hash}' and tags.eval_split = '{eval_split}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit(
            f"no popularity run for data {data_hash} on {eval_split}; run "
            "'python -m bookrec.train --config configs/popularity.yaml'"
            + (" --final" if eval_split == "t1" else "")
        )
    return {"ndcg_at_20": float(runs.iloc[0]["metrics.ndcg_at_20"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="MLflow run id, or 'latest'")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--production", default=str(PRODUCTION_JSON))
    args = ap.parse_args()

    mlflow, _ = _mlflow()
    run_id = resolve_run_id(args.run_id)
    run = mlflow.get_run(run_id)
    candidate = {
        "data_hash": run.data.tags["data_hash"],
        "eval_split": run.data.tags["eval_split"],
        "ndcg_at_20": run.data.metrics["ndcg_at_20"],
        "coverage_at_20": run.data.metrics["coverage_at_20"],
    }
    popularity = find_popularity(candidate["data_hash"], candidate["eval_split"])
    prod_path = Path(args.production)
    production = json.loads(prod_path.read_text()) if prod_path.exists() else None

    latency = None
    if not args.skip_latency:
        from bookrec.export import export

        with tempfile.TemporaryDirectory() as tmp:
            latency = measure_latency(export(run_id, Path(tmp) / "gate"))

    report = check(candidate, production, popularity, latency)
    report.update(run_id=run_id, candidate=candidate, popularity=popularity)

    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / f"{run_id}.json").write_text(json.dumps(report, indent=2))
    for c in report["checks"]:
        status = "SKIP" if c["skipped"] else ("PASS" if c["passed"] else "FAIL")
        print(f"[{status}] {c['name']}: {c['detail']}")
    print("GATE PASSED" if report["passed"] else "GATE FAILED")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
