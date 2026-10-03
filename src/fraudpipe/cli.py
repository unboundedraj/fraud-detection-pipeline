"""Command line entry point: `fraudpipe <command>` (or `python -m fraudpipe`)."""

from __future__ import annotations

import argparse
import json
import sys

import duckdb

from fraudpipe import pipeline
from fraudpipe.config import load_settings
from fraudpipe.observability import new_run_id


def _stream_transports(s, args, topic: str):
    from fraudpipe.stream.transport import FileLog, KafkaConsumer, KafkaProducer

    cfg = s["stream"]
    if args.backend == "kafka":
        return (
            KafkaProducer(cfg["bootstrap_servers"], topic),
            lambda group: KafkaConsumer(cfg["bootstrap_servers"], topic, group),
        )
    log = FileLog(s.path("stream_root"), topic)
    return log.producer(), log.consumer


def cmd_produce(s, args) -> None:
    from fraudpipe.stream.producer import replay

    producer, _ = _stream_transports(s, args, s["stream"]["topic_transactions"])
    replay(s.path("silver"), producer, speedup=args.speedup, limit=args.limit)


def cmd_consume(s, args) -> None:
    from fraudpipe.stream.consumer import StreamScorer

    _, consumer_factory = _stream_transports(s, args, s["stream"]["topic_transactions"])
    alerts_producer, _ = _stream_transports(s, args, s["stream"]["topic_alerts"])
    f = s["features"]
    scorer = StreamScorer(
        consumer_factory(args.group),
        s.path("stream_root"),
        s.path("models"),
        alerts_producer,
        f["velocity_window_seconds"],
        f["amount_window_seconds"],
    )
    scorer.run(args.batch_size or s["stream"]["micro_batch_size"], max_idle_polls=args.idle_polls)


def cmd_status(s, args) -> None:
    from fraudpipe.registry import ModelRegistry

    print("== model registry ==")
    print(json.dumps(ModelRegistry(s.path("models")).list_versions(), indent=2))
    wh = s.path("warehouse")
    if not wh.exists():
        return
    with duckdb.connect(str(wh), read_only=True) as con:
        print("\n== last pipeline run ==")
        print(
            con.sql(
                """
                SELECT step, status, round(duration_s, 2) AS secs, rows_in, rows_out
                FROM ops.pipeline_runs
                WHERE run_id = (SELECT run_id FROM ops.pipeline_runs ORDER BY started_at DESC LIMIT 1)
                ORDER BY started_at
                """
            )
        )
        print("== data quality (last run) ==")
        print(
            con.sql(
                """
                SELECT contract, count(*) AS checks, count(*) FILTER (WHERE NOT passed) AS failed
                FROM ops.dq_results
                WHERE run_id = (SELECT run_id FROM ops.dq_results ORDER BY checked_at DESC LIMIT 1)
                GROUP BY 1 ORDER BY 1
                """
            )
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fraudpipe", description=__doc__)
    p.add_argument("--config", help="path to pipeline.yaml (default: config/pipeline.yaml)")
    p.add_argument("--run-id", help="override the generated run id")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="download source + build bronze layer")
    ing.add_argument("--force", action="store_true", help="re-download even if checksum matches")
    sub.add_parser("silver", help="clean/dedupe bronze into silver")
    sub.add_parser("gold", help="build gold feature table")
    tr = sub.add_parser("train", help="train, compare and register models")
    tr.add_argument("--only", nargs="*", help="restrict to these model names")
    sc = sub.add_parser("score", help="batch-score gold with the champion model")
    sc.add_argument("--date", help="score a single event_date partition (YYYY-MM-DD)")
    ra = sub.add_parser("run-all", help="ingest -> silver -> gold -> train -> score")
    ra.add_argument("--force", action="store_true")

    for name in ("produce", "consume"):
        sp = sub.add_parser(name, help=f"stream: {name} events")
        sp.add_argument("--backend", choices=["file", "kafka"], default="file")
    prod = sub.choices["produce"]
    prod.add_argument("--speedup", type=float, default=0.0, help="event-time speedup (0 = max)")
    prod.add_argument("--limit", type=int)
    cons = sub.choices["consume"]
    cons.add_argument("--group", default="fraud-scorer")
    cons.add_argument("--batch-size", type=int)
    cons.add_argument("--idle-polls", type=int, default=5, help="stop after N empty polls")

    sub.add_parser("status", help="show registry, last run ledger and DQ summary")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    s = load_settings(args.config)
    run_id = args.run_id or new_run_id()
    match args.command:
        case "ingest":
            pipeline.step_ingest(s, run_id, force=args.force)
        case "silver":
            pipeline.step_silver(s, run_id)
        case "gold":
            pipeline.step_gold(s, run_id)
        case "train":
            pipeline.step_train(s, run_id, only=args.only)
        case "score":
            pipeline.step_score(s, run_id, args.date)
        case "run-all":
            pipeline.run_all(s, run_id, force_download=args.force)
        case "produce":
            cmd_produce(s, args)
        case "consume":
            cmd_consume(s, args)
        case "status":
            cmd_status(s, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
