import json

import duckdb
import pytest

from fraudpipe import pipeline
from fraudpipe.stream.consumer import StreamScorer, validate_payload
from fraudpipe.stream.producer import replay
from fraudpipe.stream.transport import FileLog


@pytest.fixture()
def trained(built_settings):
    pipeline.step_train(built_settings, "t", only=["logistic_regression", "hist_gradient_boosting"])
    pipeline.step_score(built_settings, "t")
    return built_settings


def test_file_log_offsets_and_commit(tmp_path):
    log = FileLog(tmp_path, "topic")
    log.producer().send((str(i), {"i": i}) for i in range(10))
    c = log.consumer("g")
    first = c.poll(4)
    assert [r.offset for r in first] == [0, 1, 2, 3]
    c.commit(4)
    c.close()

    resumed = log.consumer("g").poll(100)
    assert [r.value["i"] for r in resumed] == list(range(4, 10))


def test_payload_validation():
    good = {
        "transaction_id": "x",
        "event_epoch": 1.0,
        "amount": 1.0,
        **{f"v{i}": 0.0 for i in range(1, 29)},
    }
    assert validate_payload(good) is None
    assert validate_payload({**good, "amount": -1}) == "negative_amount"
    assert validate_payload({k: v for k, v in good.items() if k != "v7"}) == "missing_field:v7"
    assert validate_payload({**good, "v3": "abc"}) == "non_numeric:v3"


def test_end_to_end_pipeline_and_ledger(trained):
    s = trained
    with duckdb.connect(str(s.path("warehouse")), read_only=True) as con:
        steps = [
            r[0]
            for r in con.execute(
                "SELECT step FROM ops.pipeline_runs WHERE status = 'success' ORDER BY started_at"
            ).fetchall()
        ]
        views = con.execute("SELECT count(*) FROM lake.gold_features").fetchone()[0]
    assert steps[:3] == ["ingest.landing", "ingest.bronze", "contract.bronze"]
    assert "ml.train" in steps and "ml.batch_score" in steps
    assert views > 3_000
    assert (s.path("reports") / "model_leaderboard.md").exists()


def test_stream_matches_batch_and_survives_restart(trained):
    s = trained
    root = s.path("stream_root")
    tx, alerts = FileLog(root, "tx"), FileLog(root, "alerts")
    sent = replay(s.path("silver"), tx.producer())

    first = StreamScorer(tx.consumer("g"), root, s.path("models"), alerts.producer())
    first.run(micro_batch_size=300, max_batches=5)  # simulate a crash after 5 batches
    second = StreamScorer(tx.consumer("g"), root, s.path("models"), alerts.producer())
    second.run(micro_batch_size=300, max_idle_polls=1)
    assert first.stats.events + second.stats.events == sent

    scores_glob = (s.path("gold").parent.parent / "scores").as_posix() + "/**/*.parquet"
    diff = duckdb.sql(
        f"""
        SELECT count(*) AS n, count(DISTINCT st.transaction_id) AS nd,
               max(abs(st.fraud_score - b.fraud_score)) AS max_diff
        FROM read_parquet('{root.as_posix()}/scored/*.parquet') st
        JOIN read_parquet('{scores_glob}', hive_partitioning=true) b USING (transaction_id)
        """
    ).fetchone()
    assert diff[0] == diff[1] == sent  # exactly once, no gaps, no duplicates
    assert diff[2] < 1e-9


def test_stream_routes_bad_and_late_events_to_dead_letter(trained):
    s = trained
    root = s.path("stream_root")
    tx = FileLog(root, "tx")
    base = {"amount": 5.0, **{f"v{i}": 0.0 for i in range(1, 29)}}
    tx.producer().send(
        [
            ("a", {"transaction_id": "a", "event_epoch": 1000.0, **base}),
            ("b", {"transaction_id": "b", "event_epoch": 2000.0, **base}),
            ("late", {"transaction_id": "late", "event_epoch": 1500.0, **base}),
            ("bad", {"transaction_id": "bad", "event_epoch": 2001.0, **{**base, "amount": -3}}),
        ]
    )
    scorer = StreamScorer(tx.consumer("g"), root, s.path("models"))
    scorer.run(micro_batch_size=10, max_idle_polls=1)
    dlq = duckdb.sql(
        f"SELECT reason FROM '{root.as_posix()}/dead_letter/*.parquet' ORDER BY \"offset\""
    ).fetchall()
    assert dlq == [("late_event",), ("negative_amount",)]
    assert scorer.stats.dead_letters == 2
    metrics = json.loads((root / "last_run_metrics.json").read_text())
    assert metrics["events"] == 4
