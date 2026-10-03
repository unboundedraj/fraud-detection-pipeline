-- Silver -> Gold feature table
--
-- Time-window features are defined over events strictly *before* the current second
-- ([ts - window, ts - 1s]). This makes them:
--   * leakage-free (a transaction never sees itself or same-second peers), and
--   * reproducible online - the streaming scorer (fraudpipe.features.OnlineFeatureState)
--     computes exactly the same values incrementally. tests/test_feature_parity.py
--     guards this training/serving consistency.
--
-- Parameters: {silver_glob}, {velocity_s}, {amount_s}

SELECT
    transaction_id,
    event_ts,
    event_date,
    {v_columns},
    amount,
    ln(1 + amount)                                                   AS amount_log,
    hour(event_ts)                                                   AS hour_of_day,
    count(*) OVER w_velocity                                         AS txn_count_prev_window,
    coalesce(avg(amount) OVER w_amount, 0)                           AS amount_avg_prev_window,
    coalesce(
        (amount - avg(amount) OVER w_amount)
            / nullif(stddev_samp(amount) OVER w_amount, 0),
        0
    )                                                                AS amount_zscore_prev_window,
    is_fraud
FROM read_parquet('{silver_glob}', hive_partitioning = true)
WINDOW
    w_velocity AS (ORDER BY event_ts
                   RANGE BETWEEN INTERVAL {velocity_s} SECONDS PRECEDING
                             AND INTERVAL 1 SECOND PRECEDING),
    w_amount   AS (ORDER BY event_ts
                   RANGE BETWEEN INTERVAL {amount_s} SECONDS PRECEDING
                             AND INTERVAL 1 SECOND PRECEDING)
ORDER BY event_ts, transaction_id
