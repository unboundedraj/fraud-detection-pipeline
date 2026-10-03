from pathlib import Path

import duckdb
import pytest

from fraudpipe import ingest
from fraudpipe.transform import build_silver
from tests.conftest import N_DUPLICATES, N_ROWS


def _count(glob: str, where: str = "") -> int:
    return duckdb.sql(
        f"SELECT count(*) FROM read_parquet('{glob}', hive_partitioning=true) {where}"
    ).fetchone()[0]


def test_bronze_is_partitioned_with_lineage(settings):
    raw = settings.path("landing") / "creditcard.csv"
    bronze = settings.path("bronze")
    rows = ingest.load_bronze(raw, bronze, "2013-09-01T00:00:00", "batch-1")

    assert rows == N_ROWS + N_DUPLICATES  # bronze keeps duplicates
    assert sorted(p.name for p in bronze.iterdir()) == [
        "event_date=2013-09-01",
        "event_date=2013-09-02",
    ]
    glob = f"{bronze.as_posix()}/**/*.parquet"
    row = duckdb.sql(
        f"SELECT min(event_ts), any_value(_batch_id), any_value(_source_file) "
        f"FROM read_parquet('{glob}', hive_partitioning=true)"
    ).fetchone()
    assert str(row[0]).startswith("2013-09-01 00:0")
    assert row[1:] == ("batch-1", "creditcard.csv")


def test_bronze_rebuild_is_idempotent(settings):
    raw = settings.path("landing") / "creditcard.csv"
    first = ingest.load_bronze(raw, settings.path("bronze"), "2013-09-01T00:00:00", "a")
    second = ingest.load_bronze(raw, settings.path("bronze"), "2013-09-01T00:00:00", "b")
    assert first == second == _count(f"{settings.path('bronze').as_posix()}/**/*.parquet")


def test_silver_dedupes_and_quarantines(settings, tmp_path):
    raw = settings.path("landing") / "creditcard.csv"
    # poison one row so it must be quarantined
    text = raw.read_text().splitlines()
    cols = text[1].split(",")
    cols[-2] = "-10"  # negative Amount
    text[1] = ",".join(cols)
    raw.write_text("\n".join(text) + "\n")

    ingest.load_bronze(raw, settings.path("bronze"), "2013-09-01T00:00:00", "b")
    stats = build_silver(settings.path("bronze"), settings.path("silver"), tmp_path / "q")

    assert stats["quarantined_rows"] == 1
    assert stats["duplicates_removed"] == N_DUPLICATES
    assert stats["silver_rows"] == N_ROWS - 1
    q = duckdb.sql(
        f"SELECT quarantine_reason FROM '{(tmp_path / 'q' / 'quarantine.parquet').as_posix()}'"
    )
    assert q.fetchall() == [("invalid_amount",)]


def test_transaction_ids_are_deterministic(settings, tmp_path):
    raw = settings.path("landing") / "creditcard.csv"
    ids = []
    for run in ("one", "two"):
        ingest.load_bronze(raw, settings.path("bronze"), "2013-09-01T00:00:00", run)
        build_silver(settings.path("bronze"), settings.path("silver"), tmp_path / "q")
        glob = f"{settings.path('silver').as_posix()}/**/*.parquet"
        ids.append(
            duckdb.sql(
                f"SELECT list(transaction_id ORDER BY transaction_id) FROM '{glob}'"
            ).fetchone()[0]
        )
    assert ids[0] == ids[1]


def test_download_skips_when_checksum_matches(settings, monkeypatch):
    landing = settings.path("landing")

    def boom(*a, **k):
        raise AssertionError("network must not be used")

    monkeypatch.setattr(ingest.requests, "get", boom)
    expected = ingest.sha256_of(landing / "creditcard.csv")
    assert ingest.download("http://x", landing, expected) == landing / "creditcard.csv"
    assert (landing / ingest.MANIFEST).exists()


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"tampered"

    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResp())
    with pytest.raises(ValueError, match="checksum mismatch"):
        ingest.download("http://x", tmp_path / "landing", expected_sha256="0" * 64)
    assert not (Path(tmp_path / "landing" / "creditcard.csv")).exists()
