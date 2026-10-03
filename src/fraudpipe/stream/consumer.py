"""Micro-batch streaming scorer.

Loop:
  1. poll up to `micro_batch_size` events
  2. validate each payload; malformed or too-late events -> dead-letter sink
  3. compute online features (OnlineFeatureState == gold SQL, see test_feature_parity)
  4. score with the champion model, flag alerts
  5. write the batch to an idempotent Parquet sink (file name = offset range)
  6. publish alerts, then commit the consumer offset

A crash between 5 and 6 replays the batch, which overwrites the same file - no
duplicates (effectively-once). On restart, feature state is rebuilt from the last
hour of the scored sink, so window features stay correct across restarts.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from fraudpipe.features import FEATURE_COLUMNS, PCA_COLUMNS, OnlineFeatureState
from fraudpipe.observability import log
from fraudpipe.registry import ModelRegistry
from fraudpipe.stream.transport import Consumer, Producer, Record

REQUIRED_FIELDS = ("transaction_id", "event_epoch", "amount", *PCA_COLUMNS)


@dataclass
class StreamStats:
    batches: int = 0
    events: int = 0
    alerts: int = 0
    dead_letters: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        lat = sorted(self.latencies_ms) or [0.0]
        return {
            "batches": self.batches,
            "events": self.events,
            "alerts": self.alerts,
            "dead_letters": self.dead_letters,
            "p50_batch_latency_ms": round(lat[len(lat) // 2], 2),
            "p95_batch_latency_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 2),
        }


def validate_payload(value: dict[str, Any]) -> str | None:
    for f in REQUIRED_FIELDS:
        if f not in value or value[f] is None:
            return f"missing_field:{f}"
    for f in REQUIRED_FIELDS[1:]:
        v = value[f]
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            return f"non_numeric:{f}"
    if value["amount"] < 0:
        return "negative_amount"
    return None


class StreamScorer:
    def __init__(
        self,
        consumer: Consumer,
        sink_dir: Path,
        registry_root: Path,
        alerts_producer: Producer | None = None,
        velocity_window_s: int = 60,
        amount_window_s: int = 3600,
        allowed_lateness_s: int = 0,
    ):
        self.consumer = consumer
        self.sink_dir = sink_dir
        self.scored_dir = sink_dir / "scored"
        self.dlq_dir = sink_dir / "dead_letter"
        self.scored_dir.mkdir(parents=True, exist_ok=True)
        self.dlq_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_producer = alerts_producer
        self.allowed_lateness_s = allowed_lateness_s
        self.model, self.meta = ModelRegistry(registry_root).champion()
        self.state = OnlineFeatureState(velocity_window_s, amount_window_s)
        self.stats = StreamStats()
        self._restore_state()

    # ------------------------------------------------------------ recovery
    def _restore_state(self) -> None:
        files = list(self.scored_dir.glob("*.parquet"))
        if not files:
            return
        glob = f"{self.scored_dir.as_posix()}/*.parquet"
        with duckdb.connect() as con:
            hist = con.execute(
                f"""
                WITH s AS (SELECT event_epoch, amount FROM read_parquet('{glob}'))
                SELECT event_epoch, amount FROM s
                WHERE event_epoch >= (SELECT max(event_epoch) FROM s) - {self.state.amount_window_s + 1}
                ORDER BY event_epoch
                """
            ).fetchall()
        for ts, amt in hist:
            self.state.update(ts, amt)
        log.info("restored feature state from sink: %s events up to %s", len(hist), self.state.watermark)

    # ---------------------------------------------------------------- loop
    def process_batch(self, records: list[Record]) -> None:
        t0 = time.perf_counter()
        good_rows, dead = [], []
        for rec in records:
            reason = validate_payload(rec.value)
            ts = rec.value.get("event_epoch")
            wm = self.state.watermark
            if reason is None and wm is not None and ts < wm - self.allowed_lateness_s:
                reason = "late_event"
            if reason:
                dead.append({"offset": rec.offset, "reason": reason, "payload": json.dumps(rec.value)})
                continue
            feats = self.state.compute(ts, rec.value["amount"])
            self.state.update(ts, rec.value["amount"])
            good_rows.append({**{k: rec.value[k] for k in REQUIRED_FIELDS}, **feats})

        first, last = records[0].offset, records[-1].offset
        batch_name = f"offsets_{first:012d}_{last:012d}.parquet"
        if good_rows:
            frame = pd.DataFrame(good_rows)
            frame["fraud_score"] = self.model.score(frame[FEATURE_COLUMNS])
            frame["is_alert"] = frame["fraud_score"] >= self.meta["threshold"]
            frame["model_version"] = self.meta["version"]
            frame["scored_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
            out = frame[
                ["transaction_id", "event_epoch", "amount", *FEATURE_COLUMNS[len(PCA_COLUMNS):],
                 "fraud_score", "is_alert", "model_version", "scored_at"]
            ]
            out.to_parquet(self.scored_dir / batch_name, index=False)  # idempotent by name
            alerts = out[out["is_alert"]]
            if self.alerts_producer is not None and not alerts.empty:
                self.alerts_producer.send(
                    (r["transaction_id"], {k: (v.item() if hasattr(v, "item") else v)
                                           for k, v in r.items() if k != "scored_at"})
                    for r in alerts.to_dict("records")
                )
            self.stats.alerts += len(alerts)
        if dead:
            pd.DataFrame(dead).to_parquet(self.dlq_dir / batch_name, index=False)
            self.stats.dead_letters += len(dead)

        self.consumer.commit(last + 1)  # commit only after sinks are durable
        self.stats.batches += 1
        self.stats.events += len(records)
        self.stats.latencies_ms.append((time.perf_counter() - t0) * 1000)

    def run(self, micro_batch_size: int, max_idle_polls: int = 3, max_batches: int | None = None) -> dict:
        idle = 0
        while max_batches is None or self.stats.batches < max_batches:
            records = self.consumer.poll(micro_batch_size, timeout_s=1.0)
            if not records:
                idle += 1
                if idle >= max_idle_polls:
                    break
                time.sleep(0.2)
                continue
            idle = 0
            self.process_batch(records)
            if self.stats.batches % 50 == 0:
                log.info("stream progress: %s", self.stats.summary())
        self.consumer.close()
        summary = self.stats.summary()
        (self.sink_dir / "last_run_metrics.json").write_text(json.dumps(summary, indent=2))
        log.info("stream finished: %s", summary)
        return summary
