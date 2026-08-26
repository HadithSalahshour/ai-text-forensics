"""Fit and evaluate the calibrated detector.

Two deliberate choices about honesty in the metrics:

  Cross-validated, not a single holdout. With a corpus this size one split is
  mostly luck; the fold-to-fold spread is reported so a lucky split cannot be
  mistaken for a good model.

  FPR resolution is bounded by sample size. You cannot measure a 1% false-
  positive rate with 30 human samples — the finest resolvable rate is 1/n_human.
  Rather than print a confident "TPR@1%FPR" that is really TPR@3.3%FPR, we
  report the tightest rate the data can actually support and say so.

False positives are the failure that matters here. Flagging real human writing
as machine-generated does direct harm; missing some machine text does not.
Everything below is oriented around the low-FPR end of the curve.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field

import numpy as np

from ..config import settings
from . import corpus
from .dataset import SCHEMA_VERSION, Dataset, build

log = logging.getLogger(__name__)

MODEL_FILE = "calibrated_detector.joblib"
REPORT_FILE = "calibration_report.json"


@dataclass(slots=True)
class Metrics:
    auroc_mean: float
    auroc_std: float
    accuracy: float
    tpr_at_target_fpr: float
    target_fpr: float
    fpr_resolvable: bool
    n_samples: int
    n_human: int
    n_machine: int


@dataclass(slots=True)
class TrainingReport:
    metrics: Metrics
    feature_importance: list[tuple[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trustworthy: bool = True

    def to_dict(self) -> dict:
        return {
            "metrics": {
                "auroc_mean": self.metrics.auroc_mean,
                "auroc_std": self.metrics.auroc_std,
                "accuracy": self.metrics.accuracy,
                "tpr_at_target_fpr": self.metrics.tpr_at_target_fpr,
                "target_fpr": self.metrics.target_fpr,
                "fpr_resolvable": self.metrics.fpr_resolvable,
                "n_samples": self.metrics.n_samples,
                "n_human": self.metrics.n_human,
                "n_machine": self.metrics.n_machine,
            },
            "feature_importance": self.feature_importance,
            "warnings": self.warnings,
            "trustworthy": self.trustworthy,
            "schema_version": SCHEMA_VERSION,
        }


def _make_estimator():
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Shallow and heavily regularized: the corpus is small, and an
    # unconstrained booster will memorize it and report a beautiful, useless
    # AUROC.
    return HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=200,
        learning_rate=0.06,
        min_samples_leaf=5,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )


def _tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, scores)
    eligible = tpr[fpr <= target_fpr]
    return float(eligible.max()) if len(eligible) else 0.0


def cross_validate(data: Dataset, n_splits: int = 5) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold predictions plus per-fold AUROC."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    n_splits = max(2, min(n_splits, data.n_human, data.n_machine))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    oof = np.zeros(len(data), dtype=np.float64)
    fold_auc: list[float] = []

    for train_idx, test_idx in splitter.split(data.X, data.y):
        estimator = _make_estimator()
        estimator.fit(data.X[train_idx], data.y[train_idx])
        probabilities = estimator.predict_proba(data.X[test_idx])[:, 1]
        oof[test_idx] = probabilities
        if len(set(data.y[test_idx])) > 1:
            fold_auc.append(float(roc_auc_score(data.y[test_idx], probabilities)))

    return oof, fold_auc


def _permutation_importance(data: Dataset, top: int = 20) -> list[tuple[str, float]]:
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    if len(data) < 20:
        return []

    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.3, stratify=data.y, random_state=0
    )
    estimator = _make_estimator()
    estimator.fit(X_train, y_train)
    result = permutation_importance(
        estimator, X_test, y_test, n_repeats=10, random_state=0, scoring="roc_auc"
    )
    ranked = sorted(
        zip(data.feature_names, result.importances_mean, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return [(name, float(value)) for name, value in ranked[:top]]


def train(*, use_models: bool = True, save: bool = True) -> TrainingReport:
    """Fit on the whole corpus and report cross-validated performance."""
    from sklearn.isotonic import IsotonicRegression

    issues = corpus.audit()
    warnings = [f"[{w.severity}] {w.message}" for w in issues]
    blockers = [w for w in issues if w.severity == "blocker"]

    if blockers:
        # Stop before fitting. A cross-validation split on a class with one or
        # two members raises an opaque sklearn error; refusing here with the
        # actual reason is more useful than a traceback.
        counts = corpus.summary().get("by_label", {})
        return TrainingReport(
            metrics=Metrics(
                auroc_mean=float("nan"),
                auroc_std=0.0,
                accuracy=float("nan"),
                tpr_at_target_fpr=float("nan"),
                target_fpr=float("nan"),
                fpr_resolvable=False,
                n_samples=sum(counts.values()),
                n_human=counts.get("human", 0),
                n_machine=counts.get("machine", 0),
            ),
            warnings=warnings,
            trustworthy=False,
        )

    data = build(use_models=use_models)
    log.info(
        "Dataset: %d samples (%d human, %d machine), %d features",
        len(data),
        data.n_human,
        data.n_machine,
        len(data.feature_names),
    )

    oof, fold_auc = cross_validate(data)
    auroc_mean = float(np.mean(fold_auc)) if fold_auc else float("nan")
    auroc_std = float(np.std(fold_auc)) if len(fold_auc) > 1 else 0.0

    # The tightest false-positive rate this many human samples can resolve.
    finest_fpr = 1.0 / data.n_human if data.n_human else 1.0
    target_fpr = max(0.01, finest_fpr)
    fpr_resolvable = finest_fpr <= 0.01
    tpr = _tpr_at_fpr(data.y, oof, target_fpr)

    accuracy = float(((oof >= 0.5).astype(float) == data.y).mean())

    if not fpr_resolvable:
        warnings.append(
            f"[warning] {data.n_human} human samples can only resolve a false-positive "
            f"rate down to {finest_fpr:.1%}; the reported TPR is at {target_fpr:.1%} FPR, "
            "not 1%. Add human samples to measure the low-FPR regime that matters."
        )
    if len(data) < 60:
        warnings.append(
            f"[warning] {len(data)} samples total. Treat these metrics as a smoke test, "
            "not evidence of accuracy."
        )
    if auroc_mean > 0.99 and len(data) < 100:
        warnings.append(
            "[warning] AUROC above 0.99 on a small corpus usually means the classes "
            "differ in some incidental way (length, topic, formatting), not that "
            "detection is solved. Check the feature importances."
        )

    metrics = Metrics(
        auroc_mean=auroc_mean,
        auroc_std=auroc_std,
        accuracy=accuracy,
        tpr_at_target_fpr=tpr,
        target_fpr=target_fpr,
        fpr_resolvable=fpr_resolvable,
        n_samples=len(data),
        n_human=data.n_human,
        n_machine=data.n_machine,
    )

    report = TrainingReport(
        metrics=metrics,
        feature_importance=_permutation_importance(data),
        warnings=warnings,
        trustworthy=not blockers and len(data) >= 60 and fpr_resolvable,
    )

    if save and not blockers:
        import joblib

        estimator = _make_estimator()
        estimator.fit(data.X, data.y)

        # Isotonic on out-of-fold predictions: fitting the calibrator on
        # in-sample predictions would just learn the model's overconfidence.
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof, data.y)

        settings.data_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": estimator,
                "calibrator": calibrator,
                "feature_names": data.feature_names,
                "schema_version": SCHEMA_VERSION,
                "metrics": report.to_dict()["metrics"],
            },
            settings.data_dir / MODEL_FILE,
        )
        (settings.data_dir / REPORT_FILE).write_text(
            json.dumps(report.to_dict(), indent=2)
        )
        log.info("Saved model to %s", settings.data_dir / MODEL_FILE)

    if fold_auc and not math.isnan(auroc_mean):
        log.info("AUROC %.3f ± %.3f over %d folds", auroc_mean, auroc_std, len(fold_auc))

    return report
