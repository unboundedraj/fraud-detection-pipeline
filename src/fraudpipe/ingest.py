"""Landing + bronze layers.

landing/  : the raw source file exactly as downloaded, plus a manifest (sha256, size, ts)
bronze/   : the same records as typed, Hive-partitioned Parquet with lineage columns
            (event_date=YYYY-MM-DD/event_hour=HH/part-*.parquet)

Bronze is intentionally *not* cleaned - duplicates and bad values are preserved so the
silver layer can be rebuilt (and audited) from it at any time.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import requests

from fraudpipe.observability import log

RAW_FILENAME = "creditcard.csv"
MANIFEST = "_manifest.json"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, landing: Path, expected_sha256: str | None = None,
             retries: int = 4, force: bool = False) -> Path:
    """Download the source file into the landing zone (idempotent, resumable by retry).

    Skips the download when a file with a matching checksum already exists.
    """
    landing.mkdir(parents=True, exist_ok=True)
    target = landing / RAW_FILENAME
    manifest_path = landing / MANIFEST

    if target.exists() and not force:
        known = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        checksum = known.get("sha256") or sha256_of(target)
        if expected_sha256 is None or checksum == expected_sha256:
            log.info("landing file already present (sha256=%s...), skipping download", checksum[:12])
            _write_manifest(manifest_path, url, target, checksum)
            return target

    tmp = target.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            log.info("downloading %s (attempt %d/%d)", url, attempt, retries)
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for block in resp.iter_content(chunk_size=1 << 20):
                        fh.write(block)
            break
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            backoff = 2**attempt
            log.warning("download failed (%s); retrying in %ss", exc, backoff)
            time.sleep(backoff)

    checksum = sha256_of(tmp)
    if expected_sha256 and checksum != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch: expected {expected_sha256}, got {checksum}")
    tmp.replace(target)
    _write_manifest(manifest_path, url, target, checksum)
    return target


def _write_manifest(path: Path, url: str, target: Path, checksum: str) -> None:
    path.write_text(
        json.dumps(
            {
                "source_url": url,
                "file": target.name,
                "bytes": target.stat().st_size,
                "sha256": checksum,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )


def load_bronze(raw_csv: Path, bronze_dir: Path, time_anchor: str, batch_id: str) -> int:
    """Convert the landing CSV into partitioned Parquet with lineage metadata.

    Re-running replaces the whole bronze dataset (full refresh), which keeps the
    operation idempotent.
    """
    if bronze_dir.exists():
        shutil.rmtree(bronze_dir)
    bronze_dir.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect() as con:
        con.execute(
            f"""
            COPY (
                SELECT
                    row_number() OVER ()                         AS source_row_id,
                    *,
                    TIMESTAMP '{time_anchor}' + to_seconds(CAST("Time" AS BIGINT))
                                                                 AS event_ts,
                    CAST(event_ts AS DATE)                       AS event_date,
                    hour(event_ts)                               AS event_hour,
                    '{raw_csv.name}'                             AS _source_file,
                    '{batch_id}'                                 AS _batch_id,
                    now()::TIMESTAMP                             AS _ingested_at
                FROM read_csv_auto(?, header = true)
            ) TO '{bronze_dir.as_posix()}'
            (FORMAT PARQUET, PARTITION_BY (event_date, event_hour), COMPRESSION ZSTD)
            """,
            [str(raw_csv)],
        )
        rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{bronze_dir.as_posix()}/**/*.parquet')"
        ).fetchone()[0]
    log.info("bronze written: %s rows -> %s", rows, bronze_dir)
    return rows
