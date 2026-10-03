"""Structured logging and a lightweight run ledger.

Every pipeline step is wrapped in `track_step`, which records rows in/out, duration
and status into `ops.pipeline_runs` inside the DuckDB warehouse. This gives a queryable
audit trail ("which run produced this model?", "how long does silver take?") without
needing an external metadata service.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import duckdb

_LOGGER_NAME = "fraudpipe"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "ctx", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


def get_logger(name: str = _LOGGER_NAME, json_logs: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        if json_logs:
            handler.setFormatter(_JsonFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
            )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = get_logger()

RUN_LEDGER_DDL = """
CREATE SCHEMA IF NOT EXISTS ops;
CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id       VARCHAR,
    step         VARCHAR,
    status       VARCHAR,
    started_at   TIMESTAMP,
    finished_at  TIMESTAMP,
    duration_s   DOUBLE,
    rows_in      BIGINT,
    rows_out     BIGINT,
    details      JSON
);
"""


@dataclass
class StepResult:
    rows_in: int | None = None
    rows_out: int | None = None
    details: dict = field(default_factory=dict)


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


@contextmanager
def track_step(warehouse: Path, run_id: str, step: str) -> Iterator[StepResult]:
    """Context manager that times a step and appends its outcome to the run ledger."""
    result = StepResult()
    started = datetime.now(timezone.utc).replace(tzinfo=None)
    t0 = time.perf_counter()
    status = "success"
    log.info("step started: %s (run=%s)", step, run_id)
    try:
        yield result
    except Exception as exc:
        status = "failed"
        result.details.setdefault("error", repr(exc))
        raise
    finally:
        duration = time.perf_counter() - t0
        finished = datetime.now(timezone.utc).replace(tzinfo=None)
        warehouse.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(warehouse)) as con:
            con.execute(RUN_LEDGER_DDL)
            con.execute(
                "INSERT INTO ops.pipeline_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    step,
                    status,
                    started,
                    finished,
                    duration,
                    result.rows_in,
                    result.rows_out,
                    json.dumps(result.details, default=str),
                ],
            )
        log.info(
            "step %s: %s in %.2fs (rows_in=%s rows_out=%s)",
            status,
            step,
            duration,
            result.rows_in,
            result.rows_out,
        )
