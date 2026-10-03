"""Model zoo with a uniform "higher score = more suspicious" interface.

The original analysis compared three *unsupervised* detectors (Isolation Forest,
Local Outlier Factor, One-Class SVM). They are kept here as baselines next to
supervised learners so every run produces an apples-to-apples comparison on the
same chronological hold-out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass
class ScoringModel:
    name: str
    kind: str  # "supervised" | "unsupervised"
    estimator: Any
    feature_columns: list[str]

    def fit(
        self, X: pd.DataFrame, y: pd.Series, seed: int, max_unsupervised_rows: int
    ) -> ScoringModel:
        X = X[self.feature_columns]
        if self.kind == "supervised":
            self.estimator.fit(X, y)
        else:
            # Unsupervised detectors learn "normal" behaviour. Distance-based ones are
            # O(n^2), so we fit them on a reproducible sample.
            sample = (
                X
                if len(X) <= max_unsupervised_rows
                else X.sample(max_unsupervised_rows, random_state=seed)
            )
            self.estimator.fit(sample)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_columns]
        if self.kind == "supervised":
            return self.estimator.predict_proba(X)[:, 1]
        # sklearn convention: lower decision_function = more anomalous
        return -self.estimator.decision_function(X)


def candidate_models(
    feature_columns: list[str], fraud_rate: float, seed: int
) -> list[ScoringModel]:
    contamination = float(min(max(fraud_rate, 1e-4), 0.5))
    return [
        ScoringModel(
            "isolation_forest",
            "unsupervised",
            IsolationForest(
                n_estimators=200, contamination=contamination, random_state=seed, n_jobs=-1
            ),
            feature_columns,
        ),
        ScoringModel(
            "local_outlier_factor",
            "unsupervised",
            make_pipeline(
                StandardScaler(),
                LocalOutlierFactor(n_neighbors=20, contamination=contamination, novelty=True),
            ),
            feature_columns,
        ),
        ScoringModel(
            "one_class_svm",
            "unsupervised",
            make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)),
            feature_columns,
        ),
        ScoringModel(
            "logistic_regression",
            "supervised",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1),
            ),
            feature_columns,
        ),
        ScoringModel(
            "hist_gradient_boosting",
            "supervised",
            HistGradientBoostingClassifier(
                # Tuned on the validation window only (see docs/decisions.md).
                learning_rate=0.03,
                max_iter=800,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                class_weight={0: 1, 1: 5},
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=seed,
            ),
            feature_columns,
        ),
    ]
