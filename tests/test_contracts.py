from pathlib import Path

import pandas as pd
import pytest

from fraudpipe.contracts import Contract, ContractViolation


@pytest.fixture()
def parquet(tmp_path: Path) -> str:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 3],
            "v1": [0.1, 0.2, 0.3, 0.3],
            "v2": [1.0, 2.0, 3.0, 3.0],
            "amount": [10.0, 20.0, None, 5.0],
            "label": [0, 1, 0, 0],
        }
    )
    path = tmp_path / "data.parquet"
    df.to_parquet(path, index=False)
    return path.as_posix()


def _contract(**spec) -> Contract:
    return Contract({"name": "t", **spec})


def test_schema_checks_detect_missing_column_type_and_nulls(parquet):
    c = _contract(
        columns=[
            {"name": "id", "type": "BIGINT", "nullable": False},
            {"name": "amount", "type": "DOUBLE", "nullable": False},
            {"name": "missing_col"},
            {"name": "v1", "type": "VARCHAR"},
        ]
    )
    results = {r.check: r for r in c.validate(parquet, raise_on_error=False)}
    assert results["not_null:id"].passed
    assert not results["not_null:amount"].passed and results["not_null:amount"].observed == 1
    assert not results["exists:missing_col"].passed
    assert not results["type:v1"].passed


def test_pattern_columns_expand_and_count(parquet):
    c = _contract(columns=[{"pattern": "v\\d+", "type": "DOUBLE", "count": 2}])
    results = c.validate(parquet, raise_on_error=False)
    assert all(r.passed for r in results)
    assert {r.check for r in results} >= {"type:v1", "type:v2", "pattern_count:v\\d+"}


def test_dataset_checks(parquet):
    c = _contract(
        checks=[
            {"type": "unique", "name": "uid", "columns": ["id"]},
            {"type": "row_count", "min": 1, "max": 10},
            {"type": "accepted_values", "column": "label", "values": [0, 1]},
            {
                "type": "metric_between",
                "name": "rate",
                "metric": "avg(label)",
                "min": 0.2,
                "max": 0.3,
            },
            {"type": "duplicate_fraction", "name": "dupes", "columns": ["id"], "max": 0.1},
            {"type": "sql", "name": "positive", "expression": "bool_and(v1 > 0)"},
        ]
    )
    results = {r.check: r for r in c.validate(parquet, raise_on_error=False)}
    assert not results["uid"].passed
    assert results["row_count"].passed
    assert results["accepted_values"].passed
    assert results["rate"].passed
    assert not results["dupes"].passed and results["dupes"].observed == 0.25
    assert results["positive"].passed


def test_error_raises_but_warning_does_not(parquet):
    warn_only = _contract(checks=[{"type": "unique", "columns": ["id"], "severity": "warn"}])
    warn_only.validate(parquet)  # no exception

    blocking = _contract(checks=[{"type": "unique", "columns": ["id"]}])
    with pytest.raises(ContractViolation):
        blocking.validate(parquet)


def test_results_are_persisted(parquet, tmp_path):
    import duckdb

    wh = tmp_path / "wh.duckdb"
    _contract(checks=[{"type": "row_count", "min": 1}]).validate(parquet, warehouse=wh, run_id="r1")
    with duckdb.connect(str(wh)) as con:
        rows = con.execute("SELECT run_id, check_name, passed FROM ops.dq_results").fetchall()
    assert rows == [("r1", "row_count", True)]


def test_unknown_check_type_is_rejected(parquet):
    with pytest.raises(ValueError, match="unknown check type"):
        _contract(checks=[{"type": "nope"}]).validate(parquet)
