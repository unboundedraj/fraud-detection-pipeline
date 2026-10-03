-- Example operational queries against data/warehouse.duckdb
--   duckdb data/warehouse.duckdb < docs/queries.sql

-- 1. Step durations and row counts for the latest pipeline run
SELECT step, status, round(duration_s, 2) AS seconds, rows_in, rows_out
FROM ops.pipeline_runs
WHERE run_id = (SELECT run_id FROM ops.pipeline_runs ORDER BY started_at DESC LIMIT 1)
ORDER BY started_at;

-- 2. Slowest steps across all runs (p50 / max)
SELECT step,
       count(*)                                   AS runs,
       round(quantile_cont(duration_s, 0.5), 2)   AS p50_s,
       round(max(duration_s), 2)                  AS max_s
FROM ops.pipeline_runs
GROUP BY step
ORDER BY p50_s DESC;

-- 3. Data-quality failures, most recent first
SELECT checked_at, contract, check_name, severity, observed, expected
FROM ops.dq_results
WHERE NOT passed
ORDER BY checked_at DESC
LIMIT 20;

-- 4. Fraud rate and volume by hour of day (silver)
SELECT hour(event_ts)                       AS hour,
       count(*)                             AS txns,
       sum(is_fraud)                        AS frauds,
       round(100.0 * avg(is_fraud), 3)      AS fraud_pct,
       round(sum(amount) FILTER (WHERE is_fraud = 1), 2) AS fraud_amount
FROM lake.silver_transactions
GROUP BY ALL
ORDER BY hour;

-- 5. Features drifting the most in the latest scoring
SELECT batch_id, feature, round(psi, 3) AS psi, status
FROM monitoring.feature_drift
WHERE status IN ('warn', 'alert')
  AND measured_at = (SELECT max(measured_at) FROM monitoring.feature_drift)
ORDER BY psi DESC;

-- 6. Alert volume per partition and model version
SELECT partition, model_version, rows_scored, alerts,
       round(100.0 * alerts / rows_scored, 3) AS alert_rate_pct
FROM ops.scoring_runs
ORDER BY scored_at DESC, partition;
