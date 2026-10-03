"""A small, declarative data-contract engine on top of DuckDB.

Contracts live in `contracts/*.yaml` and describe what a dataset *must* look like:
schema (names, types, nullability, ranges) plus dataset-level rules (row counts,
uniqueness, accepted values, freshness, arbitrary SQL assertions).

Each rule compiles to a single SQL query, so validation runs where the data lives
(Parquet via DuckDB) and scales to the full dataset without loading it into pandas.
Results are appended to `ops.dq_results` for trend analysis.

Severity:
    error -> the pipeline stops (data is not promoted to the next layer)
    warn  -> recorded and logged, pipeline continues
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import yaml

from fraudpipe.observability import log

DQ_DDL = """
CREATE SCHEMA IF NOT EXISTS ops;
CREATE TABLE IF NOT EXISTS ops.dq_results (
    run_id     VARCHAR,
    contract   VARCHAR,
    check_name VARCHAR,
    severity   VARCHAR,
    passed     BOOLEAN,
    observed   VARCHAR,
    expected   VARCHAR,
    checked_at TIMESTAMP DEFAULT now()
);
"""


class ContractViolation(Exception):
    def __init__(self, contract: str, failures: list[CheckResult]):
        self.failures = failures
        names = ", ".join(f.check for f in failures)
        super().__init__(f"contract '{contract}' failed {len(failures)} blocking check(s): {names}")


@dataclass
class CheckResult:
    check: str
    severity: str
    passed: bool
    observed: Any
    expected: Any


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class Contract:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.name: str = spec["name"]

    @classmethod
    def from_yaml(cls, path: Path) -> Contract:
        with open(path, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    # ------------------------------------------------------------------ schema
    def _expand_columns(self, actual: dict[str, str]) -> list[dict[str, Any]]:
        """Expand `pattern:` column specs (e.g. V1..V28) against the actual schema."""
        expanded: list[dict[str, Any]] = []
        for col in self.spec.get("columns", []):
            if "pattern" in col:
                rx = re.compile(col["pattern"])
                matches = [c for c in actual if rx.fullmatch(c)]
                expanded.append({"__pattern__": col["pattern"], "__matches__": matches, **col})
                expanded.extend({**col, "name": m} for m in matches)
            else:
                expanded.append(col)
        return expanded

    def _schema_checks(self, con: duckdb.DuckDBPyConnection, rel: str) -> list[CheckResult]:
        actual = {
            row[0]: row[1].upper()
            for row in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()
        }
        results: list[CheckResult] = []
        for col in self._expand_columns(actual):
            sev = col.get("severity", "error")
            if "__pattern__" in col:
                if "count" in col:
                    results.append(
                        CheckResult(
                            f"pattern_count:{col['__pattern__']}",
                            sev,
                            len(col["__matches__"]) == col["count"],
                            len(col["__matches__"]),
                            col["count"],
                        )
                    )
                continue

            name = col["name"]
            if name not in actual:
                results.append(CheckResult(f"exists:{name}", sev, False, "missing", "present"))
                continue
            if "type" in col:
                results.append(
                    CheckResult(
                        f"type:{name}", sev, actual[name] == col["type"].upper(),
                        actual[name], col["type"].upper(),
                    )
                )
            if col.get("nullable") is False:
                nulls = con.execute(f"SELECT count(*) FROM {rel} WHERE {_q(name)} IS NULL").fetchone()[0]
                results.append(CheckResult(f"not_null:{name}", sev, nulls == 0, nulls, 0))
            if "min" in col or "max" in col:
                lo, hi = con.execute(f"SELECT min({_q(name)}), max({_q(name)}) FROM {rel}").fetchone()
                ok = True
                if "min" in col and lo is not None:
                    ok &= lo >= col["min"]
                if "max" in col and hi is not None:
                    ok &= hi <= col["max"]
                results.append(
                    CheckResult(
                        f"range:{name}", sev, bool(ok), [lo, hi], [col.get("min"), col.get("max")]
                    )
                )

        if self.spec.get("strict_schema"):
            declared = {c["name"] for c in self._expand_columns(actual) if "name" in c}
            extra = sorted(set(actual) - declared)
            results.append(CheckResult("strict_schema", "error", not extra, extra, []))
        return results

    # ----------------------------------------------------------------- dataset
    def _dataset_checks(self, con: duckdb.DuckDBPyConnection, rel: str) -> list[CheckResult]:
        results: list[CheckResult] = []
        for check in self.spec.get("checks", []):
            kind = check["type"]
            sev = check.get("severity", "error")
            label = check.get("name", kind)

            if kind == "row_count":
                n = con.execute(f"SELECT count(*) FROM {rel}").fetchone()[0]
                ok = check.get("min", 0) <= n <= check.get("max", float("inf"))
                results.append(CheckResult(label, sev, ok, n, [check.get("min"), check.get("max")]))

            elif kind == "unique":
                cols = ", ".join(_q(c) for c in check["columns"])
                dupes = con.execute(
                    f"SELECT count(*) FROM (SELECT {cols} FROM {rel} GROUP BY ALL HAVING count(*) > 1)"
                ).fetchone()[0]
                results.append(CheckResult(label, sev, dupes == 0, dupes, 0))

            elif kind == "duplicate_fraction":
                cols = ", ".join(_q(c) for c in check["columns"])
                total, distinct = con.execute(
                    f"SELECT count(*), (SELECT count(*) FROM (SELECT DISTINCT {cols} FROM {rel})) FROM {rel}"
                ).fetchone()
                frac = 0.0 if total == 0 else (total - distinct) / total
                results.append(
                    CheckResult(label, sev, frac <= check["max"], round(frac, 6), check["max"])
                )

            elif kind == "accepted_values":
                col = _q(check["column"])
                observed = sorted(
                    r[0] for r in con.execute(f"SELECT DISTINCT {col} FROM {rel}").fetchall()
                )
                ok = set(observed) <= set(check["values"])
                results.append(CheckResult(label, sev, ok, observed, check["values"]))

            elif kind == "sql":
                # Any boolean SQL aggregate over the dataset, e.g. "avg(Class) < 0.01".
                value = con.execute(f"SELECT {check['expression']} FROM {rel}").fetchone()[0]
                results.append(CheckResult(label, sev, bool(value), value, check["expression"]))

            elif kind == "metric_between":
                value = con.execute(f"SELECT {check['metric']} FROM {rel}").fetchone()[0]
                ok = value is not None and check.get("min", float("-inf")) <= value <= check.get(
                    "max", float("inf")
                )
                results.append(
                    CheckResult(label, sev, ok, value, [check.get("min"), check.get("max")])
                )
            else:
                raise ValueError(f"unknown check type '{kind}' in contract {self.name}")
        return results

    # -------------------------------------------------------------------- run
    def validate(
        self,
        source: str,
        warehouse: Path | None = None,
        run_id: str = "adhoc",
        raise_on_error: bool = True,
    ) -> list[CheckResult]:
        """Validate a Parquet glob / table expression against this contract."""
        rel = (
            f"read_parquet('{source}', hive_partitioning = true)"
            if source.endswith(".parquet")
            else source
        )
        with duckdb.connect() as con:
            results = self._schema_checks(con, rel) + self._dataset_checks(con, rel)

        failed_errors = [r for r in results if not r.passed and r.severity == "error"]
        failed_warns = [r for r in results if not r.passed and r.severity == "warn"]
        log.info(
            "contract %s: %d checks, %d errors, %d warnings",
            self.name,
            len(results),
            len(failed_errors),
            len(failed_warns),
        )
        for r in failed_errors + failed_warns:
            log.warning("  [%s] %s observed=%s expected=%s", r.severity, r.check, r.observed, r.expected)

        if warehouse is not None:
            _persist(warehouse, run_id, self.name, results)
        if raise_on_error and failed_errors:
            raise ContractViolation(self.name, failed_errors)
        return results


def _persist(warehouse: Path, run_id: str, contract: str, results: list[CheckResult]) -> None:
    warehouse.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(warehouse)) as con:
        con.execute(DQ_DDL)
        con.executemany(
            "INSERT INTO ops.dq_results (run_id, contract, check_name, severity, passed, observed, expected)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                [
                    run_id,
                    contract,
                    r.check,
                    r.severity,
                    r.passed,
                    json.dumps(r.observed, default=str),
                    json.dumps(r.expected, default=str),
                ]
                for r in results
            ],
        )


def results_to_dicts(results: list[CheckResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]
