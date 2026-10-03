"""Feature drift monitoring with the Population Stability Index (PSI).

PSI compares the distribution of a feature in a scoring batch against the training
reference profile (decile edges captured at training time and stored with the model):

    PSI = sum((actual% - expected%) * ln(actual% / expected%))

Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from fraudpipe.observability import log

DRIFT_DDL = """
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE TABLE IF NOT EXISTS monitoring.feature_drift (
    batch_id      VARCHAR,
    model_version VARCHAR,
    feature       VARCHAR,
    psi           DOUBLE,
    status        VARCHAR,
    batch_rows    BIGINT,
    measured_at   TIMESTAMP
);
"""


def psi(reference: dict[str, list[float]], values: np.ndarray, eps: float = 1e-4) -> float:
    edges = np.asarray(reference["edges"], dtype=float)
    if len(edges) < 2 or len(values) == 0:
        return 0.0
    inner = edges[1:-1]
    n_bins = len(inner) + 1
    expected = np.asarray(reference["expected"], dtype=float)
    actual_counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=n_bins)
    actual = actual_counts / actual_counts.sum()
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(
    batch: pd.DataFrame,
    reference_profile: dict[str, dict],
    warn: float,
    alert: float,
    seasonal: list[str] | tuple[str, ...] = (),
) -> pd.DataFrame:
    rows = []
    for feature, reference in reference_profile.items():
        if feature not in batch:
            continue
        value = psi(reference, batch[feature].to_numpy(dtype=float))
        if feature in seasonal:
            status = "seasonal"
        else:
            status = "alert" if value >= alert else "warn" if value >= warn else "ok"
        rows.append({"feature": feature, "psi": value, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def persist_drift(
    warehouse: Path, report: pd.DataFrame, batch_id: str, model_version: str, batch_rows: int
) -> None:
    if report.empty:
        return
    frame = report.assign(
        batch_id=batch_id,
        model_version=model_version,
        batch_rows=batch_rows,
        measured_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )[["batch_id", "model_version", "feature", "psi", "status", "batch_rows", "measured_at"]]
    with duckdb.connect(str(warehouse)) as con:
        con.execute(DRIFT_DDL)
        con.register("frame", frame)
        con.execute("INSERT INTO monitoring.feature_drift SELECT * FROM frame")
    flagged = report[report.status.isin(["warn", "alert"])]
    if not flagged.empty:
        log.warning(
            "drift in batch %s: %s",
            batch_id,
            ", ".join(f"{r.feature}={r.psi:.3f}({r.status})" for r in flagged.itertuples()),
        )
