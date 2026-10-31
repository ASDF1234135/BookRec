"""Upload artifacts to GCS and update deploy/production.json.

    python -m bookrec.promote --run-id <id> --bucket <gcs-bucket>    # or BOOKREC_BUCKET env
    python -m bookrec.promote --run-id <id> --no-upload              # local dry run

The upload uses the gcloud CLI ('gcloud storage cp'), so run 'gcloud auth login' first.
After promoting, open a PR with the changed deploy/production.json; merging it triggers deploy.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from bookrec.export import ARTIFACTS, export
from bookrec.gate import PRODUCTION_JSON, REPORT_DIR

KEPT_METRICS = ["ndcg_at_20", "ndcg_at_10", "recall_at_20", "coverage_at_20", "avg_pop"]


def require_passed_gate(run_id: str, report_dir: Path = REPORT_DIR) -> Path:
    """Refuse to promote a run whose gate report is missing or failed."""
    path = report_dir / f"{run_id}.json"
    if not path.exists():
        raise SystemExit(
            f"no gate report for {run_id}; run 'python -m bookrec.gate --run-id {run_id}'"
        )
    report = json.loads(path.read_text())
    if not report.get("passed"):
        raise SystemExit(f"gate failed for {run_id}; not promoting (see {path})")
    return path


def production_entry(version: int, gcs_path: str | None, model_card: dict, metrics: dict) -> dict:
    return {
        "version": version,
        "gcs_path": gcs_path,
        "run_id": model_card["run_id"],
        "data_hash": model_card["data_hash"],
        "eval_split": model_card["eval_split"],
        "git_hash": model_card.get("git_hash"),
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": {k: metrics[k] for k in KEPT_METRICS if k in metrics},
    }


def promote(run_id: str, bucket: str | None, upload: bool = True) -> dict:
    report_path = require_passed_gate(run_id)

    current = json.loads(PRODUCTION_JSON.read_text()) if PRODUCTION_JSON.exists() else {}
    version = (current.get("version") or 0) + 1
    out_dir = ARTIFACTS / f"v{version}"
    export(run_id, out_dir, version)
    shutil.copy(report_path, out_dir / "gate_report.json")  # stored with the model

    gcs_path = None
    if upload:
        if not bucket:
            raise SystemExit("--bucket (or BOOKREC_BUCKET) is required unless --no-upload")
        bucket = bucket.removeprefix("gs://").strip("/")
        gcs_path = f"gs://{bucket}/models/v{version}/"
        # On Windows gcloud is gcloud.cmd, which subprocess only finds via its full path.
        gcloud = shutil.which("gcloud")
        if not gcloud:
            raise SystemExit("gcloud CLI not found on PATH; install it and run 'gcloud auth login'")
        subprocess.run([gcloud, "storage", "cp", "-r", f"{out_dir}/*", gcs_path], check=True)

    card = json.loads((out_dir / "model_card.json").read_text())
    metrics = json.loads((out_dir / "metrics.json").read_text())
    entry = production_entry(version, gcs_path, card, metrics)
    PRODUCTION_JSON.parent.mkdir(exist_ok=True)
    PRODUCTION_JSON.write_text(json.dumps(entry, indent=2) + "\n")
    print(f"promoted run {run_id} as v{version}; updated {PRODUCTION_JSON}")
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--bucket", default=os.environ.get("BOOKREC_BUCKET"))
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    promote(args.run_id, args.bucket, upload=not args.no_upload)


if __name__ == "__main__":
    main()
