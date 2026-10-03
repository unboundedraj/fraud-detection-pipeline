"""Prefect orchestration.

Each pipeline step becomes a Prefect task with retries; the flow wires them in
dependency order and can be deployed on a schedule:

    prefect server start                       # UI on http://127.0.0.1:4200
    python -m fraudpipe.flows                  # one-off run
    python -m fraudpipe.flows --serve          # nightly deployment (cron below)
"""

from __future__ import annotations

import argparse

from prefect import flow, task
from prefect.logging import get_run_logger

from fraudpipe import pipeline
from fraudpipe.config import load_settings
from fraudpipe.observability import new_run_id


@task(retries=3, retry_delay_seconds=[10, 60, 300], name="ingest")
def ingest_task(run_id: str) -> None:
    pipeline.step_ingest(load_settings(), run_id)


@task(name="silver")
def silver_task(run_id: str) -> dict:
    return pipeline.step_silver(load_settings(), run_id)


@task(name="gold")
def gold_task(run_id: str) -> None:
    pipeline.step_gold(load_settings(), run_id)


@task(name="train")
def train_task(run_id: str) -> dict:
    report = pipeline.step_train(load_settings(), run_id)
    return {k: report[k] for k in ("registered_version", "promoted", "best_model")}


@task(name="batch-score")
def score_task(run_id: str) -> dict:
    return pipeline.step_score(load_settings(), run_id)


@flow(name="fraud-pipeline", log_prints=True)
def fraud_pipeline(retrain: bool = True) -> dict:
    logger = get_run_logger()
    run_id = new_run_id()
    logger.info("run_id=%s", run_id)

    ingest = ingest_task.submit(run_id)
    silver = silver_task.submit(run_id, wait_for=[ingest])
    gold = gold_task.submit(run_id, wait_for=[silver])
    upstream = [gold]
    result: dict = {"run_id": run_id, "silver": silver.result()}
    if retrain:
        train = train_task.submit(run_id, wait_for=[gold])
        result["train"] = train.result()
        upstream = [train]
    result["score"] = score_task.submit(run_id, wait_for=upstream).result()
    logger.info("flow result: %s", result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="serve a nightly deployment")
    args = ap.parse_args()
    if args.serve:
        fraud_pipeline.serve(name="nightly", cron="0 2 * * *")
    else:
        fraud_pipeline()
