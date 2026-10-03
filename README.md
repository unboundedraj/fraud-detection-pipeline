# Fraud Detection Pipeline

[![ci](https://github.com/unboundedraj/fraud-detection-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/unboundedraj/fraud-detection-pipeline/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![stack](https://img.shields.io/badge/stack-DuckDB%20·%20Parquet%20·%20Prefect%20·%20Kafka-informational)

A fraud detection pipeline for the
[ULB credit-card dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 0.17% fraud). It has a batch path and a streaming path: a medallion
lakehouse on Parquet/DuckDB, YAML data contracts, a model registry with a promotion gate,
PSI drift monitoring, and a Kafka micro-batch scorer that reproduces the batch features
exactly.

> Built on top of the original exploratory notebook analysis by
> [@tfkhushij](https://github.com/tfkhushij/Credit-Card-Fraud-Detection)
> (Isolation Forest / LOF / One-Class SVM). Those three models remain in this project as baselines.

```
 creditcard.csv ─▶ landing ─▶ bronze ─▶ silver ─▶ gold ─▶ train ─▶ registry ─▶ batch scores
                  sha256     typed    dedup +    window   5 models   champion /     + PSI drift
                  manifest   Parquet  quarantine features  compared   challenger
                               │         │          │
                               └── contract gates (253 checks per run) ──┘
                                           │
                    silver ─▶ replay producer ─▶ Kafka ─▶ micro-batch scorer ─▶ fraud.alerts
                                                          (same features, DLQ, exactly-once sink)
```

## Results

| | |
|---|---|
| **Full batch run** (raw CSV to scored transactions) | **~32 s** on a laptop |
| Records | 284,807 bronze → **283,726 silver** (1,081 exact duplicates removed and audited) |
| Data quality | **253 contract checks** per run, stored in `ops.dq_results` |
| Best model (test window, never tuned on) | HistGradientBoosting: **PR-AUC 0.757**, ROC-AUC 0.985 |
| Original approach, same window | Isolation Forest: PR-AUC 0.030 |
| At the cost-optimal threshold | 75% of fraud caught with 68 alerts out of 42,559 transactions, **61% lower expected fraud cost** |
| Streaming throughput (file backend) | ~15k events/s scored, p95 batch latency 41 ms per 500 events |
| Batch vs streaming parity | **0.0 max score difference** over 30k events, including a crash and restart |

Full comparison: [`reports/model_leaderboard.md`](reports/model_leaderboard.md).

## Data engineering features

**Ingestion and storage**
- Checksum-verified download with retries and a landing manifest. Re-runs skip work when nothing changed.
- An explicit source schema instead of type inference. Bronze is Hive-partitioned (`event_date/event_hour`) ZSTD Parquet with lineage columns (`_batch_id`, `_source_file`, `_ingested_at`).
- Silver uses deterministic content-hash `transaction_id`s, dedups exact duplicates, and **routes bad rows to quarantine** instead of dropping them.

**Data quality**
- A [declarative contract engine](src/fraudpipe/contracts.py) where YAML compiles to DuckDB SQL. It checks schema, types, nulls, ranges, pattern columns (`V1..V28`), uniqueness, accepted values, duplicate fraction, metric bands and arbitrary SQL assertions, including cross-dataset ones ("gold has the same grain as silver").
- Severity levels: `error` blocks promotion to the next layer, `warn` is recorded.

**Transformations**
- Window features (velocity, rolling mean, rolling z-score) are written as [SQL window functions](src/fraudpipe/sql/gold_features.sql) over `[ts - window, ts - 1s]`. That makes them leakage-free and reproducible online.

**ML engineering**
- Chronological train/validation/test split, PR-AUC ranking, and a **business-cost-optimised alert threshold**.
- A [model registry](src/fraudpipe/registry.py) with atomic index writes, a training-data fingerprint, a reference profile, and a champion/challenger promotion gate.

**Streaming**
- [Pluggable transports](src/fraudpipe/stream/transport.py): an offset-addressed file log for CI, or Kafka/Redpanda via `docker compose`.
- An event-time replay producer, payload validation, a **dead-letter queue** for malformed and late events, and **commit-after-write with idempotent sinks**, which gives effectively-once processing.
- Window state is **rebuilt from the sink on restart**.

**Operations**
- A run ledger (`ops.pipeline_runs`), scoring history and PSI drift tables, all queryable with SQL ([examples](docs/queries.sql)).
- A Prefect flow with retries and a nightly schedule.
- A Dockerfile, CI (ruff, pytest on 3.10–3.12, Docker build), and 27 tests that run on synthetic data with no network.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,orchestration,kafka]"

fraudpipe run-all          # ingest -> bronze -> silver -> gold -> train -> score (downloads ~150 MB once)
fraudpipe status           # registry + run ledger + data-quality summary
```

Run each step separately:

```bash
fraudpipe ingest           # landing + bronze + bronze contract
fraudpipe silver           # clean, dedup, quarantine + silver contract
fraudpipe gold             # features + gold contract + warehouse views
fraudpipe train            # compare 5 models, register, promotion gate
fraudpipe score --date 2013-09-02
```

Orchestrate with Prefect:

```bash
python -m fraudpipe.flows            # one run (temporary Prefect server)
python -m fraudpipe.flows --serve    # nightly deployment at 02:00
```

Streaming over Kafka (Redpanda):

```bash
docker compose up -d                                   # broker + console at http://localhost:8080
fraudpipe produce --backend kafka --speedup 120        # 1 hour of history per 30 s
fraudpipe consume --backend kafka                      # micro-batch scorer -> fraud.alerts
```

No Docker? Use `--backend file`, which works the same way with a local log.

Explore the warehouse:

```python
import duckdb
con = duckdb.connect("data/warehouse.duckdb", read_only=True)
con.sql("SELECT * FROM ops.pipeline_runs ORDER BY started_at DESC LIMIT 10").show()
con.sql("SELECT hour(event_ts) h, avg(is_fraud) FROM lake.silver_transactions GROUP BY h ORDER BY h").show()
```

## Project layout

```
config/pipeline.yaml        all paths, windows, costs, thresholds
contracts/*.yaml            bronze / silver / gold data contracts
src/fraudpipe/
  ingest.py                 landing download + bronze
  sql/*.sql                 silver + gold transformations
  transform.py              layer builds, warehouse views
  contracts.py              contract engine
  features.py               feature list + OnlineFeatureState
  models.py / train.py      model zoo, chronological training, leaderboard
  evaluation.py             PR-AUC, cost model, threshold search
  registry.py               model registry + promotion gate
  score.py / monitoring.py  batch scoring, PSI drift
  stream/                   transports, producer, consumer
  pipeline.py / cli.py      step wiring + CLI
  flows.py                  Prefect flow
tests/                      27 tests on synthetic data
docs/                       architecture, design decisions, example queries
```

## Documentation

- [Architecture and streaming semantics](docs/architecture.md)
- [Design decisions](docs/decisions.md): why DuckDB, why a chronological split, why PR-AUC, and more
- [Operational SQL](docs/queries.sql)

## Dataset

Machine Learning Group, ULB: *Credit Card Fraud Detection* (European cardholders, September
2013). Features `V1`–`V28` are PCA components, and `Time` is seconds since the first
transaction. The pipeline anchors `Time` to `2013-09-01 00:00` so the data can be partitioned
and replayed. The file is downloaded at runtime and never committed.
