# TODO

Everything that can be built and tested without your accounts or your training results is done.
What is left needs **you** (training time, GCP, GitHub).

## Done (code, tested locally: 61 tests, ruff clean)

- [x] Repo, Makefile, ruff, pytest
- [x] Preprocess: dedupe, signal grading, 10-core, temporal split, id maps, `data_card.json`
- [x] `evaluate()` and metrics, Popularity, LightGCN/MF, `train.py` with MLflow, `grid.py`
- [x] Wilcoxon + user-activity segments (`bookrec.eval.significance`)
- [x] `export.py`, FastAPI service with 3-way routing, JSON request logging, API tests
- [x] `gate.py` (5 checks), `promote.py`, `gate_reports/`
- [x] `ci.yml` (lint, tests, smoke train + gate, Docker build + container test)
- [x] `deploy.yml` written (WIF login, download, build, push, deploy, smoke test)
- [x] `drift.py` (frozen model on T1/T2/T3 + drift signals)
- [x] README with architecture, protocol, trade-offs, limitations, repro steps

## Yours to do

### Training and results
- [ ] Finish the grid search (`python -m bookrec.grid ...`), pick the best config, note `best_epoch`
- [ ] Set `final_epochs`, then `python -m bookrec.train --config configs/lightgcn.yaml --final --all-seeds`
- [ ] Same `--final --all-seeds` for `configs/mf.yaml` and `configs/popularity.yaml` (the gate needs a
      popularity run on the same split, and `--final` evaluates on T1)
- [ ] `python -m bookrec.eval.significance --a <lightgcn> --b <mf> --segments`, and MF vs popularity
- [ ] `python -m bookrec.drift --run-id <final lightgcn run>`
- [ ] (optional) retrain after each window to compare against the frozen model in `drift.py`
- [ ] Fill the `TBD` cells in README.md from MLflow

### GCP and GitHub
- [ ] GCP project, billing, Cloud Run / Artifact Registry / GCS APIs (README "Deploying")
- [ ] Bucket, Artifact Registry repo `bookrec`, service account, Workload Identity Federation
- [ ] GitHub variables `GCP_PROJECT_ID`, `GCS_BUCKET`; secrets `WIF_PROVIDER`, `WIF_SERVICE_ACCOUNT`
- [ ] Push the repo to GitHub and check the `ci` workflow goes green (not yet run on GitHub)
- [ ] `gate` then `promote --bucket <bucket>` the final model, open a PR with `deploy/production.json`
- [ ] Merge it, watch `deploy` run, curl the public URL for all three routes
- [ ] Docker Desktop was not running here, so `docker build` was never run locally; try
      `make docker` (or the command in README) once before relying on CI

### Evidence for the definition of done
- [ ] Show the gate blocking a worse model: run `gate` on a deliberately worse run (for example an
      under-trained one) after v1 is promoted, and keep its `gate_reports/<run_id>.json`
- [ ] Add the Cloud Run URL and the measured p95 to the README

## Not built (optional in the plan)
Two-tower model, Optuna, Cloud Scheduler retraining, book-popularity segment analysis.
