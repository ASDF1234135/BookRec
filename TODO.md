# TODO

The pipeline is built, deployed and verified end to end. What remains is mostly filling in
evidence and hardening. See README "Project summary" for the current state.

## Done

### Code (tested locally, ruff clean)
- [x] Repo, Makefile, ruff, pytest
- [x] Preprocess: dedupe, signal grading, 10-core, temporal split, id maps, `data_card.json`
- [x] `evaluate()` and metrics, Popularity, LightGCN/MF, `train.py` with MLflow, `grid.py`
- [x] Wilcoxon + user-activity segments (`bookrec.eval.significance`)
- [x] `export.py`, FastAPI service with 3-way routing, JSON request logging, API tests
- [x] `gate.py` (5 checks), `promote.py`, `gate_reports/`
- [x] `ci.yml` (lint, tests, smoke train + gate, Docker build + container test)
- [x] `deploy.yml` (WIF login, download, build, push, deploy, smoke test)
- [x] `drift.py` (frozen model on T1/T2/T3 + drift signals)
- [x] README with architecture, protocol, trade-offs, limitations, repro steps
- [x] Fix `significance.py` on Windows (close the `.npz` before the temp dir is removed)

### Training and results
- [x] Final LightGCN v1 trained, gated and promoted (NDCG@20 0.0926 on T1)
- [x] Final MF and popularity runs on the same split
- [x] Significance: LightGCN vs MF (+8.8%, p=7.4e-26) and MF vs popularity (+92.9%, p=1.1e-266)
- [x] Activity segments (< 20, 20-100, > 100) for LightGCN vs MF, with interpretation in README

### GCP and GitHub
- [x] GCP project, billing, Cloud Run / Artifact Registry / GCS APIs
- [x] Bucket, Artifact Registry repo `bookrec`, service account, Workload Identity Federation
- [x] GitHub variables `GCP_PROJECT_ID`, `GCS_BUCKET`; secrets `WIF_PROVIDER`, `WIF_SERVICE_ACCOUNT`
- [x] Push to GitHub, `ci` green
- [x] `gate` then `promote --bucket` v1, PR with `deploy/production.json`, merge, `deploy` green
- [x] Live checks: `/health`, all three routes, 422 on bad `k`, 404 on unknown book
- [x] Cloud Run URL and latency in README (p50 32 ms, p95 44 ms; cold start not measured)

### Evidence for the definition of done
- [x] Gate blocks a worse model: `configs/lightgcn_weak.yaml` fails the NDCG and coverage checks
      (`gate_reports/29742990d25f453b98ab217c41416747.json`), documented in README
- [x] Cleanup instructions for all cloud resources (README "Cleanup")

## Next (small, finishes the write-up)
- [ ] Fill the remaining `TBD` cells from MLflow: Recall@20 and Coverage@20 for popularity and MF
- [ ] Commit the `significance.py` fix, `lightgcn_weak.yaml`, `mf.yaml` changes and the new gate report
- [ ] Point the local git remote at the real GitHub repo (it still points at `Movie_Rec`)
- [ ] Set a GCP billing budget alert
- [ ] Run `make docker` locally once (Docker Desktop was not running earlier)
- [ ] Measure Cloud Run cold start (idle 15+ minutes, then time one request)

## Future work (in suggested order)
0. **Confirm later:** run `python -m bookrec.drift --run-id <final lightgcn run>` and record the
   T1 -> T3 result; report seed mean and standard deviation from the `--all-seeds` runs.
1. **Fair control:** run MF with the LightGCN hyperparameters, to separate graph propagation from tuning.
2. **Binding gate:** CI check that `deploy/production.json` points to a passing report in
   `gate_reports/`; branch protection on `main`.
3. **Safer rollout:** `gcloud run deploy --no-traffic`, verify, then shift traffic or canary;
   practise a rollback to the previous revision.
4. **Back up MLflow** (`mlflow.db`, `mlruns/`) to GCS.
5. **Monitoring:** confirm the JSON request log in Cloud Logging, add log-based metrics (p95, share
   of `popular`) and an alert.
6. **Gate for new data:** when the data hash changes, re-evaluate production on the new evaluation
   set and compare against that baseline instead of failing the same-data check.
7. **Automated retraining:** Cloud Scheduler or scheduled workflow: export, preprocess, train, gate,
   open a promotion PR for human approval.
8. **Models:** two-tower / content features for new books, Optuna, book-popularity segment analysis,
   "retrain after each window" control group for `drift.py`.
9. **Serving:** dedicated least-privilege runtime service account; optional store for recent user
   history so requests need only `user_id`; optional authentication.

## Known small issues (accepted, not blocking)
- A French title shows mojibake in Windows PowerShell 5.1 (likely a display decoding issue, not verified)
- `/books/{id}` returns author IDs, not names (author names were not downloaded)
- Structured JSON request log not confirmed in Cloud Run logs

## Cost control
- [ ] Budget alert (see Next)
- [ ] When the project is finished, tear down resources (README "Cleanup")
