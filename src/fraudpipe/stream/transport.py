"""Pluggable message transports for the streaming path.

* FileLog   - a tiny append-only, offset-addressed log on local disk. Zero
              infrastructure, used by tests/CI and for laptop demos.
* KafkaLog  - the same interface over Kafka / Redpanda (docker-compose.yml).

Both give the consumer the same contract: poll records with monotonically increasing
offsets, then commit an offset *after* results are durably written. Combined with
idempotent sinks this yields effectively-once processing.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Record:
    offset: int
    key: str | None
    value: dict[str, Any]


class Producer(ABC):
    @abstractmethod
    def send(self, records: Iterable[tuple[str | None, dict[str, Any]]]) -> int: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class Consumer(ABC):
    @abstractmethod
    def poll(self, max_records: int, timeout_s: float = 1.0) -> list[Record]: ...

    @abstractmethod
    def commit(self, next_offset: int) -> None: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


# --------------------------------------------------------------------------- file
class FileLog:
    """Single-partition topic stored as `<root>/<topic>/log.jsonl` + committed offsets."""

    def __init__(self, root: Path, topic: str):
        self.dir = root / topic
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "log.jsonl"
        self.offsets_dir = self.dir / "_offsets"
        self.offsets_dir.mkdir(exist_ok=True)

    def producer(self) -> FileProducer:
        return FileProducer(self)

    def consumer(self, group: str) -> FileConsumer:
        return FileConsumer(self, group)


class FileProducer(Producer):
    def __init__(self, log: FileLog):
        self.log = log

    def send(self, records: Iterable[tuple[str | None, dict[str, Any]]]) -> int:
        n = 0
        with open(self.log.log_path, "a", encoding="utf-8") as fh:
            for key, value in records:
                fh.write(json.dumps({"key": key, "value": value}) + "\n")
                n += 1
            fh.flush()
            os.fsync(fh.fileno())
        return n


class FileConsumer(Consumer):
    def __init__(self, log: FileLog, group: str):
        self.log = log
        self.offset_path = log.offsets_dir / f"{group}.json"
        self.position = self._committed()
        self._fh = None
        self._line = 0

    def _committed(self) -> int:
        if self.offset_path.exists():
            return int(json.loads(self.offset_path.read_text())["next_offset"])
        return 0

    def _ensure_open(self) -> None:
        if self._fh is None and self.log.log_path.exists():
            self._fh = open(self.log.log_path, encoding="utf-8")  # noqa: SIM115
            # fast-forward to the committed position
            while self._line < self.position and self._fh.readline():
                self._line += 1

    def poll(self, max_records: int, timeout_s: float = 1.0) -> list[Record]:
        self._ensure_open()
        if self._fh is None:
            return []
        out: list[Record] = []
        while len(out) < max_records:
            pos = self._fh.tell()
            line = self._fh.readline()
            if not line or not line.endswith("\n"):  # EOF or partially written line
                self._fh.seek(pos)
                break
            msg = json.loads(line)
            out.append(Record(self._line, msg["key"], msg["value"]))
            self._line += 1
        return out

    def commit(self, next_offset: int) -> None:
        tmp = self.offset_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"next_offset": next_offset}))
        tmp.replace(self.offset_path)
        self.position = next_offset

    def close(self) -> None:
        if self._fh:
            self._fh.close()


# -------------------------------------------------------------------------- kafka
class KafkaProducer(Producer):
    def __init__(self, bootstrap: str, topic: str):
        from confluent_kafka import Producer as _P

        self.topic = topic
        self._p = _P({"bootstrap.servers": bootstrap, "enable.idempotence": True, "linger.ms": 20})

    def send(self, records: Iterable[tuple[str | None, dict[str, Any]]]) -> int:
        n = 0
        for key, value in records:
            self._p.produce(self.topic, key=key, value=json.dumps(value).encode())
            n += 1
            if n % 10_000 == 0:
                self._p.poll(0)
        self._p.flush()
        return n

    def close(self) -> None:
        self._p.flush()


class KafkaConsumer(Consumer):
    def __init__(self, bootstrap: str, topic: str, group: str):
        from confluent_kafka import Consumer as _C

        self._c = _C(
            {
                "bootstrap.servers": bootstrap,
                "group.id": group,
                "enable.auto.commit": False,  # we commit only after the sink write succeeds
                "auto.offset.reset": "earliest",
            }
        )
        self._c.subscribe([topic])
        self._last = None

    def poll(self, max_records: int, timeout_s: float = 1.0) -> list[Record]:
        msgs = self._c.consume(num_messages=max_records, timeout=timeout_s)
        out = []
        for m in msgs:
            if m.error():
                continue
            self._last = m
            out.append(
                Record(m.offset(), m.key().decode() if m.key() else None, json.loads(m.value()))
            )
        return out

    def commit(self, next_offset: int) -> None:
        if self._last is not None:
            self._c.commit(asynchronous=False)

    def close(self) -> None:
        self._c.close()
