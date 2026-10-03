import numpy as np
import pytest

from fraudpipe.evaluation import choose_threshold, cost_at_threshold, ranking_metrics
from fraudpipe.monitoring import drift_report, psi
from fraudpipe.registry import ModelRegistry


def test_cost_at_threshold_arithmetic():
    y = np.array([1, 1, 0, 0, 0])
    score = np.array([0.9, 0.2, 0.8, 0.1, 0.05])
    amount = np.array([100.0, 10.0, 50.0, 5.0, 5.0])
    out = cost_at_threshold(y, score, amount, threshold=0.5, fp_cost=5, fn_floor=25)
    assert (out["true_positives"], out["false_positives"], out["false_negatives"]) == (1, 1, 1)
    # 1 FP * 5  +  missed fraud max(25, 10)
    assert out["expected_cost"] == 30
    assert out["cost_without_model"] == 100 + 25
    assert out["fraud_amount_caught"] == 100


def test_threshold_minimises_cost():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.02).astype(int)
    score = y * 0.6 + rng.random(5000) * 0.5
    amount = rng.lognormal(4, 1, 5000)
    t = choose_threshold(y, score, amount, fp_cost=5, fn_floor=25)
    best = cost_at_threshold(y, score, amount, t, 5, 25)["expected_cost"]
    for other in (0.3, 0.5, 0.9, 1.05):
        assert best <= cost_at_threshold(y, score, amount, other, 5, 25)["expected_cost"]


def test_ranking_metrics_perfect_model():
    y = np.array([0, 0, 1, 1])
    m = ranking_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["recall_at_precision_90"] == 1.0


def _profile(values):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
    counts = np.bincount(
        np.searchsorted(edges[1:-1], values, side="right"), minlength=len(edges) - 1
    )
    return {"edges": list(edges), "expected": list(counts / counts.sum())}


def test_psi_detects_shift_but_not_noise():
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, 20_000)
    prof = _profile(ref)
    assert psi(prof, rng.normal(0, 1, 5_000)) < 0.02
    assert psi(prof, rng.normal(1.0, 1, 5_000)) > 0.25


def test_drift_report_marks_seasonal_features():
    import pandas as pd

    rng = np.random.default_rng(4)
    prof = {"a": _profile(rng.normal(0, 1, 5000)), "hour": _profile(rng.integers(0, 24, 5000))}
    batch = pd.DataFrame({"a": rng.normal(3, 1, 500), "hour": np.full(500, 23)})
    rep = drift_report(batch, prof, 0.1, 0.25, seasonal=["hour"]).set_index("feature")
    assert rep.loc["a", "status"] == "alert"
    assert rep.loc["hour", "status"] == "seasonal"


def test_registry_promotion_gate(tmp_path):
    reg = ModelRegistry(tmp_path)
    meta = lambda v: {"algorithm": "m", "primary_metric": {"name": "pr_auc", "value": v}}  # noqa: E731
    v1 = reg.register({"w": 1}, meta(0.70))
    assert reg.promote_if_better(v1, "pr_auc")
    v2 = reg.register({"w": 2}, meta(0.65))
    assert not reg.promote_if_better(v2, "pr_auc")
    v3 = reg.register({"w": 3}, meta(0.80))
    assert reg.promote_if_better(v3, "pr_auc")

    model, champ = reg.champion()
    assert model == {"w": 3} and champ["version"] == v3
    stages = {v["version"]: v["stage"] for v in reg.list_versions()}
    assert stages == {v1: "archived", v2: "rejected", v3: "champion"}


def test_registry_without_champion_raises(tmp_path):
    with pytest.raises(LookupError):
        ModelRegistry(tmp_path).champion()
