"""Training/serving skew guard: streaming features must equal batch (SQL) features."""

import duckdb
import numpy as np

from fraudpipe.features import ENGINEERED_COLUMNS, OnlineFeatureState


def test_online_features_match_gold_sql(built_settings):
    s = built_settings
    gold = duckdb.sql(
        f"SELECT epoch(event_ts) AS ts, amount, {', '.join(ENGINEERED_COLUMNS)} "
        f"FROM read_parquet('{s.path('gold').as_posix()}/**/*.parquet', hive_partitioning=true) "
        "ORDER BY event_ts, transaction_id"
    ).df()
    # the synthetic feed must contain same-second events for this test to be meaningful
    assert gold["ts"].duplicated().any()

    f = s["features"]
    state = OnlineFeatureState(f["velocity_window_seconds"], f["amount_window_seconds"])
    online = []
    for ts, amount in zip(gold["ts"], gold["amount"], strict=True):
        online.append(state.compute(ts, amount))
        state.update(ts, amount)

    for col in ENGINEERED_COLUMNS:
        np.testing.assert_allclose(
            [o[col] for o in online],
            gold[col].to_numpy(dtype=float),
            rtol=1e-7,
            atol=1e-7,
            err_msg=f"online/offline mismatch for {col}",
        )


def test_state_rejects_out_of_order_events():
    state = OnlineFeatureState()
    state.update(100.0, 1.0)
    try:
        state.update(99.0, 1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-order event")


def test_compaction_preserves_results():
    rng = np.random.default_rng(0)
    ts = np.sort(rng.integers(0, 200_000, 60_000)).astype(float)
    amt = rng.lognormal(3, 1, ts.size)
    compacting = OnlineFeatureState()
    for t, a in zip(ts, amt, strict=True):
        compacting.update(t, a)
    assert len(compacting._ts) < ts.size  # compaction happened
    probe_t, probe_a = ts[-1] + 5, 42.0

    reference = OnlineFeatureState()
    reference._ts = list(ts)
    reference._cum_amount = [0.0, *np.cumsum(amt)]
    reference._cum_amount_sq = [0.0, *np.cumsum(amt**2)]
    a, b = compacting.compute(probe_t, probe_a), reference.compute(probe_t, probe_a)
    for k in a:
        assert np.isclose(a[k], b[k], rtol=1e-6), k
