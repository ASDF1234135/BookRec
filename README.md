# bookrec

A book recommendation API built on the Goodreads comics & graphic-novel data. A LightGCN model is
trained locally on CPU, gated by an automated evaluation check, packaged as a FastAPI + Docker
service and deployed to Google Cloud Run through GitHub Actions.

> **Status:** the full pipeline is built and working end to end: local training with MLflow,
> evaluation gate, promotion to GCS, CI, and automatic deployment of model v1 to Cloud Run
> (verified against the live URL). Cells marked `TBD` are results not yet collected; see
> [Project summary](#project-summary) and `TODO.md` for what is done and what comes next.

## Project summary

| Area | State |
| --- | --- |
| Data and evaluation | Done: temporal split, shared `evaluate()`, Wilcoxon test, activity segments |
| Models | Done: popularity baseline, MF, LightGCN (CPU) |
| Experiment tracking | Done: MLflow (local SQLite) with git hash and data hash on every run |
| Quality gate | Done: 5 checks; passes v1, **blocks a deliberately under-trained model** (see below) |
| Promotion and versioning | Done: `promote` -> GCS `models/v<N>/` + `deploy/production.json` |
| CI | Done: lint, tests, smoke train + gate on fake data, Docker build and container test |
| CD | Done: Workload Identity Federation (keyless), build, push, deploy, version check |
| Serving | Live on Cloud Run: 3 routing strategies, FAISS, p95 44 ms end to end |
| Drift analysis | Code done (`drift.py`); running it and recording results is future work |
| Monitoring | Designed (JSON request logs); no log-based metric or alert yet |
| Automated retraining | Not built; see "Future work" |

## Architecture

```text
 LOCAL                               GITHUB ACTIONS                      GOOGLE CLOUD
 -----                               --------------                      ------------
 raw JSON -> preprocess              push / PR  -> ci.yml                GCS  models/v<N>/  (artifacts)
 train (MF, LightGCN) -> MLflow        ruff, pytest, smoke train+gate    Artifact Registry  (image)
 gate  (5 checks, exit 1 = blocked)    docker build + container test     Cloud Run          (API)
 promote -> GCS + production.json
        | PR merged to main         -> deploy.yml (WIF login)
        +------------------------->   download model, build, push,
                                       gcloud run deploy, smoke test
```

The model artifacts are baked into the Docker image, so a Cloud Run cold start never reads GCS and
a rollback is just routing traffic back to the previous revision. Every production change is a git
commit of `deploy/production.json`.

### Request routing

| Request | Strategy | How |
| --- | --- | --- |
| `user_id` seen in training | `lightgcn` | user embedding -> FAISS inner-product search |
| unknown user, known `recent_book_ids` | `item_avg` | mean of those books' embeddings -> FAISS |
| neither | `popular` | precomputed popularity list |

Books the user has already read are always removed (the search fetches `k + len(read)` candidates
so `k` results always survive the filter).

## Data

UCSD Goodreads comics & graphic novels (interactions + book metadata).

- Positive = read and rated 3+ or unrated; ratings of 1-2 and unread shelf adds are excluded.
- Iterative 10-core filter, then up to 60,000 users sampled (seeded).
- Result (`data/processed/data_card.json`): 48,597 users, 30,382 books, about 2.0M training
  interactions.
- **Global temporal split** on `date_updated`: 60% train, 10% validation, then three 10% future
  windows T1/T2/T3. Validation and test keep only users and books seen in training; dropped books
  are saved for cold-start analysis. T1 is the official test set.
- Hyperparameters are chosen on validation; the final model is retrained on train + validation.

## Evaluation protocol

One shared `evaluate()` for every model: full ranking over all books, already-read books masked,
metrics computed per user then averaged, users with no held-out positives skipped.

Primary metric NDCG@20; also Recall@20, NDCG@10, Coverage@20 and average recommended popularity.
Significance: Wilcoxon signed-rank on per-user NDCG@20
(`python -m bookrec.eval.significance --a <run> --b <run>`).

## Results

Test set T1, 18,834 evaluated users. NDCG@20 is the per-user mean from
`bookrec.eval.significance`; the LightGCN row is the promoted v1 run (`3c4dd708...`, also in
`deploy/production.json`). Figures are from one final run per model, without seed mean and standard
deviation (planned, see Future work). `TBD` = not yet collected from MLflow.

| Model | NDCG@20 | Recall@20 | Coverage@20 | vs. previous (Wilcoxon p) |
| --- | --- | --- | --- | --- |
| Popularity | 0.0441 | TBD | TBD | - |
| MF (LightGCN, 0 layers) | 0.0851 | TBD | TBD | 1.1e-266 (vs popularity, +92.9%) |
| LightGCN (v1) | 0.0926 | 0.1353 | 0.3052 | 7.4e-26 (vs MF, +8.8%) |

**NDCG@20 by user activity** (number of training interactions), LightGCN vs MF:

| Group | Users | LightGCN | MF | Relative gain |
| --- | --- | --- | --- | --- |
| < 20 | 10,073 | 0.1137 | 0.1059 | +7.4% |
| 20-100 | 6,486 | 0.0737 | 0.0669 | +10.0% |
| > 100 | 2,275 | 0.0534 | 0.0448 | +19.0% |

What this shows:
- Personalisation is the main win: MF alone nearly doubles the popularity baseline.
- Graph propagation adds a smaller but consistent gain (+8.8%) and wins in every activity group.
  With 18,834 users the p-value is tiny, so the effect size matters more than the p-value.
- The relative gain *grows* with user activity, the opposite of the common assumption that graph
  methods mainly help sparse users. A possible reason (not tested) is that heavy users have richer
  neighbourhoods. NDCG falls with activity for both models, probably because heavy users have
  read more of the easy items already.
- MF and LightGCN were tuned separately (learning rate, epochs), so the gap includes tuning, not
  only graph propagation. A fairer control would run MF with the LightGCN settings.

Frozen-model drift on T1 -> T3 (`python -m bookrec.drift`): not run yet; planned, see Future work.

### Gate demonstration: a worse model is blocked

`configs/lightgcn_weak.yaml` is LightGCN trained for only 5 epochs. Run against production v1
(same data hash and split), the gate exits with code 1
(`gate_reports/29742990d25f453b98ab217c41416747.json`):

| Check | Result | Detail |
| --- | --- | --- |
| same data as production | pass | `7d35e339cd8236a2` / t1 |
| beats popularity | pass | NDCG 0.0722 vs required 0.0529 |
| no NDCG regression | **fail** | 0.0722 vs required 0.0917 (99% of production 0.0926) |
| no coverage regression | **fail** | 0.0738 vs required 0.2747 (90% of production 0.3052) |
| latency p95 | pass | 0.70 ms vs < 100 ms |

The weak model is still clearly better than popularity, so a "beats the baseline" test alone would
have let it through; the regression checks against the current production model are what stop it.
`promote` refuses to run without a passing report.

Serving latency: the gate measures p95 over 1,000 simulated requests against the exported model
(v1 measured p95 0.92 ms; the budget is 100 ms). Cloud Run, measured from a local machine
over 50 sequential `POST /recommend` calls (k=10, warm instance, includes network round trip):
p50 32 ms, p95 44 ms, max 68 ms. Server-side `latency_ms` was about 12 ms for a known user
(`lightgcn`), 1.3 ms for `item_avg` and under 0.1 ms for `popular`. Cold start was not measured.

## MLOps

- **Experiment tracking:** MLflow with a SQLite backend. Every run logs its config, metrics,
  per-user NDCG, embeddings, git commit hash and a hash of the data.
- **Eval gate** (`python -m bookrec.gate --run-id <id>`, exit code 1 blocks promotion):
  1. same data hash and split as production,
  2. NDCG@20 at least 1.2x the popularity baseline,
  3. NDCG@20 at least 99% of production,
  4. Coverage@20 at least 90% of production,
  5. local p95 latency under 100 ms.
  The report is written to `gate_reports/<run_id>.json`. `tests/test_gate.py` shows a worse model
  being blocked.
- **Promotion:** `python -m bookrec.promote --run-id <id> --bucket <bucket>` refuses without a
  passing report, exports `artifacts/v<N>`, uploads it to GCS and rewrites `deploy/production.json`.
  Merge that change to `main` to deploy.
- **CI** (`ci.yml`): lint, tests, smoke train + gate on fake data, Docker build and container
  smoke test. **CD** (`deploy.yml`): Workload Identity Federation login, download the promoted
  model, build, push, deploy, post-deploy smoke test.
- **Monitoring:** each request logs one JSON line (`request_id`, `strategy`, `k`, `latency_ms`,
  `model_version`), which Cloud Logging can turn into p95 latency and strategy-mix metrics. A rising
  share of `popular` responses means many users are unknown to the model, which is the retraining signal.

## Design trade-offs

- **LightGCN, CPU only.** The graph propagation is one sparse matrix product, fine on CPU. The
  adjacency is symmetric, so the custom backward pass reuses it instead of transposing every step
  (about 25x faster per step than the default autograd path on this graph).
- **MF and LightGCN share one implementation** (0 layers = MF), which keeps the comparison close.
  Their hyperparameters were tuned separately, so the gap is graph propagation plus tuning.
- **Exact FAISS search** (`IndexFlatIP`): tens of thousands of books need only milliseconds; swap the
  index when the catalogue grows.
- **Model baked into the image** instead of loaded at startup: faster cold starts, trivially
  reproducible rollbacks, at the cost of one image per model version.
- **Training stays local; promotion goes through git.** No training infrastructure to run, and
  every production model has an auditable commit.

## Known limitations

- Book metadata gives author ids only (the authors file was not downloaded), so `/books/{id}`
  returns ids, not names.
- New books (not in training) cannot be recommended; the optional two-tower model that would handle
  them is not built. They are saved in `cold_start_dropped.parquet` for future analysis.
- Evaluation uses a single global time split; per-user chronological splits would be harder.
- Users only get `lightgcn` recommendations if they were in the training set; others get the mean
  of the books they supply, which is weaker.
- The gate requires the same data hash and split as production, so a model retrained on newer
  data cannot pass it as is. Retraining needs a step that re-evaluates the current production model
  on the new evaluation set and uses that as the baseline.
- Experiment history lives only in the local `mlflow.db` / `mlruns/`; there is no remote backup.
- Cloud Run is public with no authentication; `--max-instances 2` bounds the cost.
- Not implemented: Optuna search, scheduled retraining, the "retrain after every window" drift
  control group, book-popularity segment analysis, online A/B testing.

## Reproduce

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# data: put the three UCSD files in data/raw/, then
python -m bookrec.data.preprocess

python -m bookrec.train --config configs/popularity.yaml
python -m bookrec.train --config configs/mf.yaml
python -m bookrec.grid  --config configs/lightgcn.yaml --grid n_layers=2,3 lr=1e-3,5e-3
python -m bookrec.train --config configs/lightgcn.yaml --final --all-seeds   # set final_epochs first
mlflow ui --backend-store-uri sqlite:///mlflow.db

python -m bookrec.gate    --run-id <run_id>
python -m bookrec.promote --run-id <run_id> --no-upload     # local dry run (or --bucket <b>)
python -m bookrec.export  --run-id <run_id>                 # artifacts/v<N>
```

Run the API locally (artifacts from `export`):

```powershell
cd service; $env:MODEL_DIR = "../artifacts/v1"; uvicorn app.main:app --port 8080
```

Docker: copy `artifacts/v1` to `service/model`, then
`docker build --platform linux/amd64 -t bookrec-service -f service/Dockerfile service`.

## API

```bash
curl -s $URL/health
curl -s -X POST $URL/recommend -H 'content-type: application/json' -d '{"user_id": "<goodreads user id>", "k": 5}'
curl -s -X POST $URL/recommend -H 'content-type: application/json' -d '{"recent_book_ids": ["287365", "750692"], "k": 5}'
curl -s -X POST $URL/recommend -H 'content-type: application/json' -d '{"k": 5}'
curl -s $URL/books/287365
```

`URL`: `https://bookrec-wujoiqkj6q-km.a.run.app` (region australia-southeast2). Interactive docs are at `$URL/docs`.

## Deploying (one-time GCP setup)

1. Create a project, enable billing and the Cloud Run, Artifact Registry and Cloud Storage APIs.
2. Create a GCS bucket, an Artifact Registry Docker repository named `bookrec` in
   `australia-southeast2`, and a service account with Cloud Run Admin, Artifact Registry Writer,
   Storage Object Viewer and Service Account User roles.
3. Set up a Workload Identity pool and provider for this GitHub repo and let the service account be
   impersonated from it.
4. In the GitHub repo, add variables `GCP_PROJECT_ID`, `GCS_BUCKET` and secrets `WIF_PROVIDER`,
   `WIF_SERVICE_ACCOUNT`.
5. `gcloud auth login`, then `promote` a gated run with `--bucket`, and merge the resulting
   `deploy/production.json` change to `main`.

## Cleanup (stop all cloud costs)

Cloud Run with `--min-instances 0` costs almost nothing when idle; the recurring costs are image and
model storage. To remove everything:

```bash
gcloud run services delete bookrec --region australia-southeast2 --project bookrec-513223
gcloud artifacts repositories delete bookrec --location australia-southeast2 --project bookrec-513223
gcloud storage rm -r gs://bookrecbucket
# or delete the whole project (removes everything above, including IAM and Workload Identity):
gcloud projects delete bookrec-513223
```

To pause without deleting, delete only the Cloud Run service; `deploy.yml` can recreate it
(`workflow_dispatch`). A rollback is routing traffic back to a previous revision.

## Future work

Ordered roughly by value for the effort:

1. **Close the evidence gaps:** run `drift` and record the T1 -> T3 result, report seed mean and
   standard deviation from the `--all-seeds` runs, fill the remaining `TBD` cells, and run MF with
   the LightGCN hyperparameters as a fairer control.
2. **Make the gate binding:** a CI check that `deploy/production.json` points to a run with a
   passing report in `gate_reports/`, plus branch protection on `main`.
3. **Safer rollout:** deploy with `--no-traffic`, verify, then shift traffic (or canary at 5%);
   rehearse a rollback to the previous Cloud Run revision.
4. **Back up MLflow** (`mlflow.db`, `mlruns/`) to GCS.
5. **Monitoring:** confirm the JSON request log in Cloud Logging, add log-based metrics (p95,
   share of `popular`) and an alert; set a billing budget alert.
6. **Automated retraining:** scheduled export, preprocess, train, gate, then an automatically
   opened promotion PR for human approval. Requires the data-version gate change above.
7. **Models:** two-tower or content features for new books, Optuna search, book-popularity segments.
8. **Serving:** run the service under a dedicated least-privilege identity; optionally look up
   recent user history from a store (Redis/Firestore) instead of requiring `recent_book_ids`.

