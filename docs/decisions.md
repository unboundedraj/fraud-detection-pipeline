# Design decisions

Short records of why the pipeline is built the way it is.

### 1. DuckDB + Parquet instead of Spark
The dataset is about 150 MB and 285k rows, and a full batch run finishes in about 30 s on a
laptop. Spark would add cluster overhead without any benefit. All transforms are plain SQL
over Hive-partitioned Parquet, so moving to Spark SQL, Databricks or Snowflake would mostly
mean changing the engine, not the logic.

### 2. Medallion layers with bronze kept "dirty"
Bronze keeps the 1,081 exact duplicate rows (0.38%) that the source contains. Cleaning happens
in silver, so silver can always be rebuilt from bronze and the dedup is auditable
(`ops.pipeline_runs.details.duplicates_removed`).

### 3. Content-hash transaction ids
The source has no primary key. `transaction_id = md5(all source columns)` is deterministic,
so re-runs produce the same ids. That keeps downstream joins and the streaming sink
idempotent.

### 4. Data contracts as YAML, compiled to SQL
Every check runs as one DuckDB query against Parquet, which avoids pulling data into pandas
and adds no heavy dependency. Severity separates blocking problems (`error`) from monitored
ones (`warn`, such as duplicates in bronze). Results are stored so quality can be trended.

### 5. Chronological split, not random
Fraud patterns change over time, and a random split leaks future behaviour into training.
The pipeline uses the first 70% of time for training, the next 15% for validation (model
selection and threshold) and the last 15% for test. The test window is reported but never
tuned on. The HGB hyperparameters were picked by comparing a handful of settings on the
validation window only.

### 6. PR-AUC and expected cost, not accuracy
At a 0.17% fraud rate, a model that never predicts fraud is 99.83% accurate. Models are
ranked by PR-AUC. The alert threshold minimises
`FP x $5 review cost + sum(max($25, amount))` over missed frauds.

### 7. The original unsupervised models stay as baselines
Isolation Forest, LOF and One-Class SVM were the original approach. On the same chronological
hold-out they reach PR-AUC 0.02 to 0.09, against 0.76 for gradient boosting. Keeping them in
the leaderboard makes that comparison reproducible on every run.

### 8. Window features exclude the current second
Features look at `[ts - window, ts - 1s]`. A transaction never sees itself or its same-second
peers. That makes the features leakage-free and also order-independent within a second, which
is what lets the streaming path reproduce them exactly.

### 9. Single-partition transaction topic
The dataset has no card or merchant id, so the window features are global and need a total
event-time order, which means one partition. With an entity key, the topic would be
partitioned by that key and the state kept per key.

### 10. File-based registry instead of MLflow
The registry is a JSON index, per-version artifacts and a promotion gate in about 120 lines,
with no server to run. The metadata stored per version (metrics, threshold, training-data
fingerprint, reference profile) maps directly onto MLflow's model registry if that's needed
later.

### 11. Seasonal features are measured but never alert
Validation and test windows cover evening hours, so `hour_of_day` and the rolling-volume
features always "drift" from the 36-hour training window. They're listed as
`seasonal_features` in `config/pipeline.yaml`. Some PCA components, such as `v1`, also drift
by time of day. That's a real property of this data and is left visible.
