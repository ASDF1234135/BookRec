MODEL ?= lightgcn
RUN_ID ?=
IMAGE ?= bookrec-service
# folder with exported artifacts for `make serve` / `make stage` (e.g. artifacts/v1)
MODEL_DIR ?= artifacts/v1

.PHONY: setup data train gate promote export stage serve docker lint test smoke fixtures drift

setup:
	python -m venv .venv
	.venv/Scripts/pip install -e ".[dev]"

data:
	python -m bookrec.data.preprocess

train:
	python -m bookrec.train --config configs/$(MODEL).yaml

gate:
	python -m bookrec.gate --run-id $(RUN_ID)

promote:
	python -m bookrec.promote --run-id $(RUN_ID)

export:
	python -m bookrec.export --run-id $(RUN_ID)

# copy a chosen artifact folder into the Docker build context
stage:
	python -c "import shutil; shutil.rmtree('service/model', ignore_errors=True); shutil.copytree('$(MODEL_DIR)', 'service/model')"

serve:
	cd service && MODEL_DIR=../$(MODEL_DIR) uvicorn app.main:app --reload --port 8080

docker: stage
	docker build --platform linux/amd64 -t $(IMAGE) -f service/Dockerfile service

lint:
	ruff check . && ruff format --check .

test:
	pytest -q

# the same steps CI runs on the fake data
smoke:
	python -m bookrec.train --config configs/smoke_popularity.yaml
	python -m bookrec.train --config configs/smoke.yaml
	python -m bookrec.gate --run-id latest --production none.json

fixtures:
	python -m bookrec.fixtures --smoke-data tests/fixtures

drift:
	python -m bookrec.drift --run-id $(RUN_ID)
