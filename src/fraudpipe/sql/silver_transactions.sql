-- Bronze -> Silver
--   * conform names to snake_case and stable types
--   * derive a deterministic transaction_id from the record content (idempotent re-runs)
--   * drop exact duplicate records, keeping the first occurrence in source order
--   * route rows that break row-level rules to quarantine instead of silently dropping them
--
-- Parameters (string-substituted by transform.py): {bronze_glob}

CREATE OR REPLACE TEMP VIEW bronze AS
SELECT * FROM read_parquet('{bronze_glob}', hive_partitioning = true);

CREATE OR REPLACE TEMP TABLE conformed AS
SELECT
    md5(concat_ws('|', "Time", {v_columns_concat}, Amount, Class)) AS transaction_id,
    source_row_id,
    event_ts,
    CAST(event_ts AS DATE)           AS event_date,
    CAST("Time" AS DOUBLE)           AS time_offset_s,
    {v_columns_select},
    CAST(Amount AS DOUBLE)           AS amount,
    CAST(Class AS TINYINT)           AS is_fraud,
    _batch_id                        AS source_batch_id,
    _ingested_at                     AS source_ingested_at
FROM bronze;

CREATE OR REPLACE TEMP TABLE row_checks AS
SELECT
    *,
    CASE
        WHEN amount IS NULL OR amount < 0      THEN 'invalid_amount'
        WHEN event_ts IS NULL                  THEN 'missing_event_ts'
        WHEN is_fraud NOT IN (0, 1)            THEN 'invalid_label'
        WHEN isnan(v1) OR isinf(v1)            THEN 'non_finite_feature'
        ELSE NULL
    END AS quarantine_reason
FROM conformed;

CREATE OR REPLACE TEMP TABLE quarantine AS
SELECT * FROM row_checks WHERE quarantine_reason IS NOT NULL;

CREATE OR REPLACE TEMP TABLE silver AS
SELECT * EXCLUDE (quarantine_reason, rn)
FROM (
    SELECT
        *,
        row_number() OVER (PARTITION BY transaction_id ORDER BY source_row_id) AS rn
    FROM row_checks
    WHERE quarantine_reason IS NULL
)
WHERE rn = 1;
