"""Batch scoring with the champion model.

Writes one Parquet file per scored partition into data/scores/ and appends a summary
row to `ops.scoring_runs`. Partition-level overwrite keeps re-runs idempotent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from fraudpipe.monitoring import drift_report, persist_drift
from fraudpipe.observability import log
from fraudpipe.registry import ModelRegistry

SCORING_DDL = """
CREATE SCHEMA IF NOT EXISTS ops;
CREATE TABLE IF NOT EXISTS ops.scoring_runs (
    run_id        VARCHAR,
    partition     VARCHAR,
    model_version VARCHAR,
    rows_scored   BIGINT,
    alerts        BIGINT,
    scored_at     TIMESTAMP
);
"""


def score_frame(model, meta: dict, frame: pd.DataFrame) -> pd.DataFrame:
    scores = model.score(frame)
    out = frame[["transaction_id", "event_ts", "amount"]].copy()
    out["fraud_score"] = scores
    out["is_alert"] = scores >= meta["threshold"]
    out["model_version"] = meta["version"]
    return out


def batch_score(
    gold: Path,
    scores_dir: Path,
    registry_root: Path,
    warehouse: Path,
    run_id: str,
    partition_date: str | None,
    monitoring_cfg: dict,
) -> dict[str, int]:
    model, meta = ModelRegistry(registry_root).champion()
    where = f"WHERE event_date = DATE '{partition_date}'" if partition_date else ""
    with duckdb.connect() as con:
        frame = con.execute(
            f"SELECT * FROM read_parquet('{gold.as_posix()}/**/*.parquet', hive_partitioning = true) "
            f"{where} ORDER BY event_ts"
        ).df()
    if frame.empty:
        log.warning("nothing to score for partition=%s", partition_date)
        return {"rows_scored": 0, "alerts": 0}

    results = []
    for date, part in frame.groupby(frame["event_date"].astype(str)):
        scored = score_frame(model, meta, part)
        target = scores_dir / f"event_date={date}"
        target.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(target / "scores.parquet", index=False)  # overwrite = idempotent

        report = drift_report(
            part,
            meta["reference_profile"],
            monitoring_cfg["psi_warn"],
            monitoring_cfg["psi_alert"],
            monitoring_cfg.get("seasonal_features", []),
        )
        persist_drift(warehouse, report, f"batch:{date}", meta["version"], len(part))
        results.append((date, len(scored), int(scored["is_alert"].sum())))

    with duckdb.connect(str(warehouse)) as con:
        con.execute(SCORING_DDL)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        con.executemany(
            "INSERT INTO ops.scoring_runs VALUES (?, ?, ?, ?, ?, ?)",
            [[run_id, d, meta["version"], n, a, now] for d, n, a in results],
        )
    total = {"rows_scored": sum(r[1] for r in results), "alerts": sum(r[2] for r in results)}
    log.info("batch scoring done with %s: %s", meta["version"], total)
    return total
