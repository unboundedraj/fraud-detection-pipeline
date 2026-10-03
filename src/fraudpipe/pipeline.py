"""Pipeline steps as plain functions - shared by the CLI and the Prefect flow."""

from __future__ import annotations

from typing import Any

from fraudpipe.config import Settings
from fraudpipe.contracts import Contract
from fraudpipe.ingest import download, load_bronze
from fraudpipe.observability import track_step
from fraudpipe.score import batch_score
from fraudpipe.train import train_and_register
from fraudpipe.transform import build_gold, build_silver, publish_views


def _glob(s: Settings, layer: str) -> str:
    return f"{s.path(layer).as_posix()}/**/*.parquet"


def step_ingest(s: Settings, run_id: str, force: bool = False) -> None:
    wh = s.path("warehouse")
    with track_step(wh, run_id, "ingest.landing") as r:
        raw = download(s["source"]["url"], s.path("landing"), s["source"]["sha256"], force=force)
        r.details["file"] = str(raw)
    with track_step(wh, run_id, "ingest.bronze") as r:
        r.rows_out = load_bronze(raw, s.path("bronze"), s["source"]["time_anchor"], run_id)
    validate(s, run_id, "bronze")


def step_silver(s: Settings, run_id: str) -> dict[str, int]:
    with track_step(s.path("warehouse"), run_id, "transform.silver") as r:
        stats = build_silver(s.path("bronze"), s.path("silver"), s.path("silver").parent.parent / "quarantine")
        r.rows_in, r.rows_out, r.details = stats["bronze_rows"], stats["silver_rows"], stats
    validate(s, run_id, "silver")
    return stats


def step_gold(s: Settings, run_id: str) -> None:
    f = s["features"]
    with track_step(s.path("warehouse"), run_id, "transform.gold") as r:
        r.rows_out = build_gold(
            s.path("silver"), s.path("gold"), f["velocity_window_seconds"], f["amount_window_seconds"]
        )
    validate(s, run_id, "gold", references={"silver_ref": _glob(s, "silver")})
    publish_views(
        s.path("warehouse"),
        {"lake.bronze_transactions": s.path("bronze"),
         "lake.silver_transactions": s.path("silver"),
         "lake.gold_features": s.path("gold")},
    )


def validate(s: Settings, run_id: str, layer: str, references: dict[str, str] | None = None) -> None:
    with track_step(s.path("warehouse"), run_id, f"contract.{layer}") as r:
        results = Contract.from_yaml(s.contract(layer)).validate(
            _glob(s, layer), warehouse=s.path("warehouse"), run_id=run_id, references=references
        )
        r.details = {
            "checks": len(results),
            "failed": [x.check for x in results if not x.passed],
        }


def step_train(s: Settings, run_id: str, only: list[str] | None = None) -> dict[str, Any]:
    with track_step(s.path("warehouse"), run_id, "ml.train") as r:
        report = train_and_register(
            s.path("gold"), s.path("models"), s.path("reports"),
            s["training"], s["registry"], run_id, only=only,
        )
        r.details = {k: report[k] for k in ("registered_version", "promoted", "best_model")}
    return report


def step_score(s: Settings, run_id: str, partition_date: str | None = None) -> dict[str, int]:
    with track_step(s.path("warehouse"), run_id, "ml.batch_score") as r:
        out = batch_score(
            s.path("gold"), s.path("gold").parent.parent / "scores", s.path("models"),
            s.path("warehouse"), run_id, partition_date, s["monitoring"],
        )
        r.rows_out, r.details = out["rows_scored"], out
    return out


def run_all(s: Settings, run_id: str, force_download: bool = False) -> None:
    step_ingest(s, run_id, force=force_download)
    step_silver(s, run_id)
    step_gold(s, run_id)
    step_train(s, run_id)
    step_score(s, run_id)
