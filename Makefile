PY ?= python

.PHONY: install dev lint fmt test pipeline flow stream-up stream-down produce consume status clean

install:            ## install the package
	$(PY) -m pip install -e .

dev:                ## install with dev, orchestration and kafka extras
	$(PY) -m pip install -e ".[dev,orchestration,kafka]"

lint:               ## static checks
	ruff check src tests
	ruff format --check src tests

fmt:                ## auto-format
	ruff format src tests
	ruff check --fix src tests

test:               ## unit + integration tests (synthetic data, no network)
	$(PY) -m pytest --cov=fraudpipe --cov-report=term-missing

pipeline:           ## full batch run: ingest -> bronze -> silver -> gold -> train -> score
	$(PY) -m fraudpipe run-all

flow:               ## same, orchestrated by Prefect
	$(PY) -m fraudpipe.flows

stream-up:          ## start Redpanda + Console (http://localhost:8080)
	docker compose up -d

stream-down:
	docker compose down -v

produce:            ## replay transactions into Kafka at 120x real time
	$(PY) -m fraudpipe produce --backend kafka --speedup 120

consume:            ## score the Kafka stream in micro-batches
	$(PY) -m fraudpipe consume --backend kafka

status:             ## registry, last run ledger, DQ summary
	$(PY) -m fraudpipe status

clean:              ## remove generated data and artifacts (keeps the downloaded source)
	rm -rf data/bronze data/silver data/gold data/scores data/quarantine data/stream \
	       data/warehouse.duckdb models reports/runs
