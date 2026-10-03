"""Shared fixtures: a small synthetic dataset with the exact source schema.

Tests never touch the network or the real 150 MB file. The synthetic generator
injects the same kinds of problems the real feed has (exact duplicates, same-second
events) so cleaning and windowing logic is exercised.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fraudpipe.config import REPO_ROOT, Settings, load_settings

N_ROWS = 4_000
FRAUD_RATE = 0.008
N_DUPLICATES = 25


def make_source_frame(n: int = N_ROWS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 172_000, size=n)).astype(float)
    frame = pd.DataFrame({"Time": t})
    y = (rng.random(n) < FRAUD_RATE).astype(int)
    for i in range(1, 29):
        frame[f"V{i}"] = rng.normal(0, 1, n)
    # make fraud learnable
    frame.loc[y == 1, "V14"] -= 4
    frame.loc[y == 1, "V4"] += 3
    frame.loc[y == 1, "V12"] -= 3
    frame["Amount"] = np.round(rng.lognormal(3, 1.2, n), 2)
    frame["Class"] = y
    dupes = frame.sample(N_DUPLICATES, random_state=seed)
    return pd.concat([frame, dupes]).sort_values("Time", kind="stable").reset_index(drop=True)


@pytest.fixture(scope="session")
def source_frame() -> pd.DataFrame:
    return make_source_frame()


def _relax_contract(spec: dict) -> dict:
    """Scale volume-based expectations down to the synthetic dataset size."""
    for check in spec.get("checks", []):
        if check["type"] == "row_count":
            check["min"], check["max"] = 1_000, 10_000
        if check.get("name") == "positive_class_has_enough_support_to_train":
            check["min"] = 10
    return spec


@pytest.fixture()
def settings(tmp_path: Path, source_frame: pd.DataFrame) -> Settings:
    """Isolated project root in tmp_path: config, relaxed contracts and landing file."""
    base = load_settings()
    raw = yaml.safe_load(yaml.safe_dump(base.raw))
    raw["source"]["sha256"] = None
    raw["registry"]["min_improvement"] = 0.0

    (tmp_path / "contracts").mkdir()
    for layer, rel in raw["contracts"].items():
        spec = yaml.safe_load((REPO_ROOT / rel).read_text())
        (tmp_path / rel).write_text(yaml.safe_dump(_relax_contract(spec)))
        raw["contracts"][layer] = rel

    landing = tmp_path / raw["paths"]["landing"]
    landing.mkdir(parents=True)
    source_frame.to_csv(landing / "creditcard.csv", index=False)
    return Settings(raw=raw, root=tmp_path)


@pytest.fixture()
def built_settings(settings: Settings) -> Settings:
    """Settings whose bronze/silver/gold layers are already built."""
    from fraudpipe import pipeline

    pipeline.step_ingest(settings, "test-run")
    pipeline.step_silver(settings, "test-run")
    pipeline.step_gold(settings, "test-run")
    return settings


@pytest.fixture()
def copy_tree():
    def _copy(src: Path, dst: Path) -> Path:
        shutil.copytree(src, dst)
        return dst

    return _copy
