"""Load and apply a fitted detector at inference time.

The fitted model is optional. When absent — or present but flagged as trained
on too little data — the ensemble falls back to its hand-tuned tier blend
rather than trusting a model it has no reason to trust.

A stale model is worse than no model, so the schema version is checked on load:
if the feature set has changed since training, the model is refused rather than
fed columns that no longer mean what it learned.
"""

from __future__ import annotations

import functools
import logging

import numpy as np

from ..config import settings
from .dataset import SCHEMA_VERSION
from .train import MODEL_FILE

log = logging.getLogger(__name__)


class FittedDetector:
    def __init__(self, payload: dict) -> None:
        self.estimator = payload["estimator"]
        self.calibrator = payload["calibrator"]
        self.feature_names: list[str] = payload["feature_names"]
        self.metrics: dict = payload.get("metrics", {})

    @property
    def trustworthy(self) -> bool:
        """Whether the training run produced metrics worth acting on."""
        m = self.metrics
        return bool(
            m.get("n_samples", 0) >= 60
            and m.get("fpr_resolvable", False)
            and m.get("auroc_mean", 0.0) >= 0.80
        )

    @property
    def summary(self) -> str:
        m = self.metrics
        return (
            f"AUROC {m.get('auroc_mean', float('nan')):.3f} "
            f"(n={m.get('n_samples', 0)}, "
            f"TPR@{m.get('target_fpr', 0):.0%}FPR={m.get('tpr_at_target_fpr', 0):.2f})"
        )

    def predict(self, feats: dict[str, float]) -> float:
        """Calibrated probability that the text is machine-generated."""
        row = np.full((1, len(self.feature_names)), np.nan, dtype=np.float64)
        for index, name in enumerate(self.feature_names):
            value = feats.get(name)
            if value is not None and np.isfinite(value):
                row[0, index] = value
        raw = float(self.estimator.predict_proba(row)[0, 1])
        return float(self.calibrator.predict([raw])[0])


@functools.lru_cache(maxsize=1)
def load() -> FittedDetector | None:
    path = settings.data_dir / MODEL_FILE
    if not path.exists():
        return None

    try:
        import joblib

        payload = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 — a broken model must not break scoring
        log.warning("Could not load fitted detector: %s", exc)
        return None

    stored_version = payload.get("schema_version")
    if stored_version != SCHEMA_VERSION:
        log.warning(
            "Fitted detector was trained on feature schema v%s but the current "
            "schema is v%s. Ignoring it — retrain with `humanizer calibrate train`.",
            stored_version,
            SCHEMA_VERSION,
        )
        return None

    return FittedDetector(payload)


def reset_cache() -> None:
    load.cache_clear()
