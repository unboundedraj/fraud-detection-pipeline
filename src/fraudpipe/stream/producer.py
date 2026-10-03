"""Replay producer: turns the historical silver table into a live event stream.

Events are emitted in event-time order. With `speedup=60` one hour of history plays
in one minute; `speedup=0` emits as fast as the transport accepts. The label
(`is_fraud`) is deliberately NOT part of the payload - in production it only arrives
days later as a chargeback.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

from fraudpipe.features import PCA_COLUMNS
from fraudpipe.observability import log
from fraudpipe.stream.transport import Producer

PAYLOAD_COLUMNS = ["transaction_id", "event_epoch", *PCA_COLUMNS, "amount"]


def replay(
    silver: Path,
    producer: Producer,
    speedup: float = 0.0,
    limit: int | None = None,
    start_after_epoch: float | None = None,
    chunk: int = 2_000,
) -> int:
    where = f"WHERE epoch(event_ts) > {float(start_after_epoch)}" if start_after_epoch else ""
    lim = f"LIMIT {int(limit)}" if limit else ""
    query = f"""
        SELECT transaction_id, epoch(event_ts) AS event_epoch, {", ".join(PCA_COLUMNS)}, amount
        FROM read_parquet('{silver.as_posix()}/**/*.parquet', hive_partitioning = true)
        {where}
        ORDER BY event_ts, transaction_id
        {lim}
    """
    sent = 0
    wall_start = time.perf_counter()
    first_event: float | None = None
    with duckdb.connect() as con:
        result = con.execute(query)
        # duckdb >= 1.4 renamed fetch_record_batch -> to_arrow_reader
        reader = (
            result.to_arrow_reader(chunk)
            if hasattr(result, "to_arrow_reader")
            else result.fetch_record_batch(chunk)
        )
        for batch in reader:
            rows = batch.to_pylist()
            if speedup > 0:
                first_event = first_event if first_event is not None else rows[0]["event_epoch"]
                due = (rows[-1]["event_epoch"] - first_event) / speedup
                lag = due - (time.perf_counter() - wall_start)
                if lag > 0:
                    time.sleep(lag)
            sent += producer.send((r["transaction_id"], r) for r in rows)
    producer.close()
    elapsed = time.perf_counter() - wall_start
    log.info("replayed %s events in %.1fs (%.0f ev/s)", sent, elapsed, sent / max(elapsed, 1e-9))
    return sent
