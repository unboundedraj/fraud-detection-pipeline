"""Imbalance-aware evaluation and cost-based threshold selection.

Accuracy is meaningless at a 0.17% positive rate (predicting "never fraud" scores
99.83%), so models are ranked by area under the precision-recall curve, and the
alerting threshold is chosen to minimise the expected business cost:

    cost = FP * review_cost  +  sum(max(floor, amount) for each missed fraud)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(y, score)
    def recall_at(p: float) -> float:
        mask = precision >= p
        return float(recall[mask].max()) if mask.any() else 0.0

    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "recall_at_precision_80": recall_at(0.80),
        "recall_at_precision_90": recall_at(0.90),
    }


def cost_at_threshold(
    y: np.ndarray, score: np.ndarray, amount: np.ndarray, threshold: float,
    fp_cost: float, fn_floor: float,
) -> dict[str, float]:
    pred = score >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    missed_loss = float(np.maximum(amount[(pred == 0) & (y == 1)], fn_floor).sum())
    caught_value = float(amount[(pred == 1) & (y == 1)].sum())
    total_cost = fp * fp_cost + missed_loss
    baseline_cost = float(np.maximum(amount[y == 1], fn_floor).sum())  # "no model" cost
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "alerts": tp + fp,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fraud_amount_caught": caught_value,
        "expected_cost": total_cost,
        "cost_without_model": baseline_cost,
        "cost_reduction_pct": 100.0 * (1 - total_cost / baseline_cost) if baseline_cost else 0.0,
    }


def choose_threshold(
    y: np.ndarray, score: np.ndarray, amount: np.ndarray, fp_cost: float, fn_floor: float,
    n_candidates: int = 400,
) -> float:
    """Grid-search the threshold minimising expected cost (on validation data only)."""
    candidates = np.unique(np.quantile(score, np.linspace(0.90, 1.0, n_candidates)))
    best_t, best_cost = float(candidates[-1]), float("inf")
    for t in candidates:
        c = cost_at_threshold(y, score, amount, t, fp_cost, fn_floor)["expected_cost"]
        if c < best_cost:
            best_t, best_cost = float(t), c
    return best_t
