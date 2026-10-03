"""Silver and gold layer builds (DuckDB SQL over Parquet)."""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

import duckdb

from fraudpipe.features import PCA_COLUMNS
from fraudpipe.observability import log


def _sql(name: str) -> str:
    return resources.files("fraudpipe.sql").joinpath(name).read_text(encoding="utf-8")


def _glob(path: Path) -> str:
    return f"{path.as_posix()}/**/*.parquet"


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def build_silver(bronze: Path, silver: Path, quarantine: Path) -> dict[str, int]:
    """Clean + deduplicate bronze into silver. Returns row accounting for the run ledger."""
    sql = _sql("silver_transactions.sql").format(
        bronze_glob=_glob(bronze),
        v_columns_concat=", ".join(f"V{i}" for i in range(1, 29)),
        v_columns_select=",\n    ".join(f"CAST(V{i} AS DOUBLE) AS v{i}" for i in range(1, 29)),
    )
    _reset(silver)
    _reset(quarantine)
    with duckdb.connect() as con:
        con.execute(sql)
        stats = {
            "bronze_rows": con.execute("SELECT count(*) FROM conformed").fetchone()[0],
            "quarantined_rows": con.execute("SELECT count(*) FROM quarantine").fetchone()[0],
            "silver_rows": con.execute("SELECT count(*) FROM silver").fetchone()[0],
        }
        stats["duplicates_removed"] = (
            stats["bronze_rows"] - stats["quarantined_rows"] - stats["silver_rows"]
        )
        con.execute(
            f"COPY (SELECT * FROM silver ORDER BY event_ts, transaction_id) TO '{silver.as_posix()}' "
            "(FORMAT PARQUET, PARTITION_BY (event_date), COMPRESSION ZSTD)"
        )
        if stats["quarantined_rows"]:
            quarantine.mkdir(parents=True, exist_ok=True)
            con.execute(
                f"COPY quarantine TO '{(quarantine / 'quarantine.parquet').as_posix()}' (FORMAT PARQUET)"
            )
    log.info("silver stats: %s", stats)
    return stats


def build_gold(silver: Path, gold: Path, velocity_s: int, amount_s: int) -> int:
    sql = _sql("gold_features.sql").format(
        silver_glob=_glob(silver),
        velocity_s=int(velocity_s),
        amount_s=int(amount_s),
        v_columns=", ".join(PCA_COLUMNS),
    )
    _reset(gold)
    with duckdb.connect() as con:
        con.execute(
            f"COPY ({sql}) TO '{gold.as_posix()}' "
            "(FORMAT PARQUET, PARTITION_BY (event_date), COMPRESSION ZSTD)"
        )
        rows = con.execute(f"SELECT count(*) FROM read_parquet('{_glob(gold)}')").fetchone()[0]
    log.info("gold features written: %s rows", rows)
    return rows


def publish_views(warehouse: Path, layers: dict[str, Path]) -> None:
    """Expose lake layers as DuckDB views so analysts can query them with plain SQL."""
    warehouse.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(warehouse)) as con:
        for qualified, path in layers.items():
            schema, _ = qualified.split(".")
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            con.execute(
                f"CREATE OR REPLACE VIEW {qualified} AS "
                f"SELECT * FROM read_parquet('{_glob(path.resolve())}', hive_partitioning = true)"
            )
    log.info("published warehouse views: %s", ", ".join(layers))
