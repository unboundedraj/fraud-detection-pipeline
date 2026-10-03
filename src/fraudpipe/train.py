"""Train, compare and register fraud models on the gold feature table."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from fraudpipe.evaluation import choose_threshold, cost_at_threshold, ranking_metrics
from fraudpipe.features import FEATURE_COLUMNS, LABEL_COLUMN
from fraudpipe.models import ScoringModel, candidate_models
from fraudpipe.observability import log
from fraudpipe.registry import ModelRegistry


def load_features(gold: Path) -> pd.DataFrame:
    with duckdb.connect() as con:
        return con.execute(
            f"SELECT * FROM read_parquet('{gold.as_posix()}/**/*.parquet', hive_partitioning = true) "
            "ORDER BY event_ts, transaction_id"
        ).df()


def data_fingerprint(df: pd.DataFrame) -> str:
    """Order-independent fingerprint of the training set (which rows, not which order)."""
    hashed = pd.util.hash_pandas_object(df["transaction_id"], index=False)
    return f"{int(np.bitwise_xor.reduce(hashed.to_numpy())):016x}-{len(df)}"


def chronological_split(
    df: pd.DataFrame, train_frac: float, val_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def _window(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(df),
        "frauds": int(df[LABEL_COLUMN].sum()),
        "start": str(df["event_ts"].min()),
        "end": str(df["event_ts"].max()),
    }


def train_and_register(
    gold: Path,
    registry_root: Path,
    reports_dir: Path,
    training_cfg: dict[str, Any],
    registry_cfg: dict[str, Any],
    run_id: str,
    max_unsupervised_rows: int = 40_000,
    only: list[str] | None = None,
) -> dict[str, Any]:
    df = load_features(gold)
    train, val, test = chronological_split(
        df, training_cfg["train_fraction"], training_cfg["validation_fraction"]
    )
    seed = training_cfg["random_seed"]
    fp_cost = training_cfg["cost_false_positive"]
    fn_floor = training_cfg["cost_false_negative_floor"]
    log.info("split: train=%s val=%s test=%s", _window(train), _window(val), _window(test))

    fraud_rate = float(train[LABEL_COLUMN].mean())
    leaderboard: list[dict[str, Any]] = []
    fitted: dict[str, ScoringModel] = {}

    for model in candidate_models(FEATURE_COLUMNS, fraud_rate, seed):
        if only and model.name not in only:
            continue
        t0 = time.perf_counter()
        model.fit(train, train[LABEL_COLUMN], seed, max_unsupervised_rows)
        fit_s = time.perf_counter() - t0

        val_score = model.score(val)
        threshold = choose_threshold(
            val[LABEL_COLUMN].to_numpy(), val_score, val["amount"].to_numpy(), fp_cost, fn_floor
        )
        test_score = model.score(test)
        row = {
            "model": model.name,
            "kind": model.kind,
            "fit_seconds": round(fit_s, 2),
            "validation": ranking_metrics(val[LABEL_COLUMN].to_numpy(), val_score),
            "test": ranking_metrics(test[LABEL_COLUMN].to_numpy(), test_score),
            "test_at_threshold": cost_at_threshold(
                test[LABEL_COLUMN].to_numpy(),
                test_score,
                test["amount"].to_numpy(),
                threshold,
                fp_cost,
                fn_floor,
            ),
        }
        leaderboard.append(row)
        fitted[model.name] = model
        log.info(
            "%-24s val PR-AUC=%.4f  test PR-AUC=%.4f  recall=%.3f precision=%.3f (%.1fs)",
            model.name,
            row["validation"]["pr_auc"],
            row["test"]["pr_auc"],
            row["test_at_threshold"]["recall"],
            row["test_at_threshold"]["precision"],
            fit_s,
        )

    # Model selection uses validation only; the test window is reported, never tuned on.
    best = max(leaderboard, key=lambda r: r["validation"]["pr_auc"])
    metric = registry_cfg["primary_metric"]
    registry = ModelRegistry(registry_root)
    version = registry.register(
        fitted[best["model"]],
        {
            "run_id": run_id,
            "algorithm": best["model"],
            "kind": best["kind"],
            "feature_columns": FEATURE_COLUMNS,
            "threshold": best["test_at_threshold"]["threshold"],
            "primary_metric": {"name": metric, "value": best["test"][metric], "window": "test"},
            "metrics": {k: best[k] for k in ("validation", "test", "test_at_threshold")},
            "training_data": {
                "fingerprint": data_fingerprint(train),
                "train": _window(train),
                "validation": _window(val),
                "test": _window(test),
            },
            "reference_profile": reference_profile(train),
            "cost_model": {"false_positive": fp_cost, "false_negative_floor": fn_floor},
        },
    )
    promoted = registry.promote_if_better(version, metric, registry_cfg["min_improvement"])

    report = {
        "run_id": run_id,
        "registered_version": version,
        "promoted": promoted,
        "best_model": best["model"],
        "leaderboard": leaderboard,
        "splits": {"train": _window(train), "validation": _window(val), "test": _window(test)},
    }
    write_reports(reports_dir, report)
    return report


def reference_profile(train: pd.DataFrame, bins: int = 10) -> dict[str, dict[str, list[float]]]:
    """Per-feature decile edges + observed bin shares - the drift monitor's baseline.

    Shares are measured rather than assumed uniform because low-cardinality features
    (e.g. hour_of_day) collapse duplicate quantile edges into uneven bins.
    """
    profile = {}
    for col in FEATURE_COLUMNS:
        values = train[col].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        inner = edges[1:-1]
        counts = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(inner) + 1)
        profile[col] = {
            "edges": [float(e) for e in edges],
            "expected": [float(c) for c in counts / counts.sum()],
        }
    return profile


def write_reports(reports_dir: Path, report: dict[str, Any]) -> None:
    runs = reports_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"train_{report['run_id']}.json").write_text(json.dumps(report, indent=2, default=str))
    (reports_dir / "model_leaderboard.md").write_text(
        render_leaderboard(report), encoding="utf-8", newline="\n"
    )


def render_leaderboard(report: dict[str, Any]) -> str:
    s = report["splits"]
    lines = [
        "# Model leaderboard",
        "",
        "_Auto-generated by `fraudpipe train`. Do not edit by hand._",
        "",
        f"Chronological split: train {s['train']['rows']:,} rows "
        f"({s['train']['frauds']} frauds) · validation {s['validation']['rows']:,} "
        f"({s['validation']['frauds']}) · test {s['test']['rows']:,} ({s['test']['frauds']}). "
        f"Test window: {s['test']['start']} to {s['test']['end']}.",
        "",
        "| Model | Type | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Recall @ thr | Precision @ thr "
        "| Alerts | Fraud $ caught | Cost reduction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(report["leaderboard"], key=lambda r: -r["validation"]["pr_auc"]):
        t = r["test_at_threshold"]
        marker = " **(selected)**" if r["model"] == report["best_model"] else ""
        lines.append(
            f"| {r['model']}{marker} | {r['kind']} | {r['validation']['pr_auc']:.3f} "
            f"| {r['test']['pr_auc']:.3f} | {r['test']['roc_auc']:.3f} | {t['recall']:.1%} "
            f"| {t['precision']:.1%} | {t['alerts']} | ${t['fraud_amount_caught']:,.0f} "
            f"| {t['cost_reduction_pct']:.1f}% |"
        )
    lines += [
        "",
        "Thresholds are chosen on the validation window by minimising "
        "`false_positives x review_cost + sum(max(floor, amount) of missed frauds)` "
        "and then applied unchanged to the test window.",
        "",
    ]
    return "\n".join(lines)
