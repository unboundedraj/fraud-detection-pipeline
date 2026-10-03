"""File-based model registry with lineage and a champion/challenger promotion gate.

models/
  registry.json              <- index of every registered version
  v0001/model.joblib         <- serialized ScoringModel
  v0001/metadata.json        <- metrics, threshold, features, training-data fingerprint

A new version only becomes `champion` (the model the scorers load) if it beats the
current champion on the primary metric measured on the *same* hold-out window.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from fraudpipe.observability import log


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = root
        self.index_path = root / "registry.json"

    # ---------------------------------------------------------------- index io
    def _load(self) -> dict[str, Any]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {"champion": None, "versions": []}

    def _save(self, index: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.index_path)  # atomic swap - readers never see a half-written index

    # ----------------------------------------------------------------- public
    def register(self, model: Any, metadata: dict[str, Any]) -> str:
        index = self._load()
        version = f"v{len(index['versions']) + 1:04d}"
        vdir = self.root / version
        vdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, vdir / "model.joblib")
        meta = {
            **metadata,
            "version": version,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "stage": "candidate",
        }
        (vdir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        index["versions"].append({k: meta[k] for k in ("version", "algorithm", "registered_at", "stage")}
                                 | {"primary_metric": meta["primary_metric"]})
        self._save(index)
        log.info("registered model %s (%s)", version, metadata.get("algorithm"))
        return version

    def promote_if_better(self, version: str, metric: str, min_improvement: float = 0.0) -> bool:
        index = self._load()
        candidate = self.metadata(version)
        cand_score = candidate["primary_metric"]["value"]
        champion = index.get("champion")

        promote = champion is None
        if champion:
            champ_score = self.metadata(champion)["primary_metric"]["value"]
            promote = cand_score > champ_score + min_improvement
            log.info(
                "promotion gate %s: challenger %s=%.4f vs champion %s=%.4f -> %s",
                metric, version, cand_score, champion, champ_score,
                "PROMOTE" if promote else "keep champion",
            )
        if promote:
            for entry in index["versions"]:
                if entry["version"] == champion:
                    entry["stage"] = "archived"
                    self._set_stage(champion, "archived")
                if entry["version"] == version:
                    entry["stage"] = "champion"
            self._set_stage(version, "champion")
            index["champion"] = version
        else:
            for entry in index["versions"]:
                if entry["version"] == version:
                    entry["stage"] = "rejected"
            self._set_stage(version, "rejected")
        self._save(index)
        return promote

    def _set_stage(self, version: str, stage: str) -> None:
        path = self.root / version / "metadata.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["stage"] = stage
        path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def metadata(self, version: str) -> dict[str, Any]:
        return json.loads((self.root / version / "metadata.json").read_text(encoding="utf-8"))

    def champion(self) -> tuple[Any, dict[str, Any]]:
        index = self._load()
        if not index.get("champion"):
            raise LookupError("no champion model registered yet - run `fraudpipe train` first")
        version = index["champion"]
        return joblib.load(self.root / version / "model.joblib"), self.metadata(version)

    def list_versions(self) -> list[dict[str, Any]]:
        return self._load()["versions"]
