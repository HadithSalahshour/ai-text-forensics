"""Turn the corpus into a feature matrix.

Feature extraction is the slow part (the statistical tier runs two language
models per sample), so results are cached on disk keyed by sample id and
feature-schema version. Bump SCHEMA_VERSION whenever a feature is added,
removed, or its definition changes — otherwise a stale cache silently trains
the model on features that no longer mean what their names say.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np

from .. import segment
from ..config import settings
from ..detect import features as F
from ..detect import statistical
from . import corpus

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2


def _cache_path():
    return settings.data_dir / f"feature_cache_v{SCHEMA_VERSION}.json"


@dataclass(slots=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray  # 1 = machine, 0 = human
    feature_names: list[str]
    sample_ids: list[str]
    registers: list[str]
    generators: list[str]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_machine(self) -> int:
        return int(self.y.sum())

    @property
    def n_human(self) -> int:
        return int(len(self.y) - self.y.sum())


def extract_one(text: str, *, use_models: bool = True) -> dict[str, float]:
    """Full feature vector for a single text — the same path used at inference,
    so training and scoring can never drift apart."""
    doc = segment.build(**{"text": text})
    feats = F.extract(doc)
    if use_models:
        stat = statistical.analyze(doc)
        if stat.available:
            feats.update(stat.as_features())
    # Raw size counters would let the model separate classes on document
    # length, which is an artifact of how the corpus was collected rather
    # than a property of machine text.
    for key in ("word_count", "sentence_count", "paragraph_count"):
        feats.pop(key, None)
    return feats


def build(*, use_models: bool = True, use_cache: bool = True) -> Dataset:
    samples = corpus.load()
    if not samples:
        raise RuntimeError("Corpus is empty. Add samples before building a dataset.")

    cache: dict[str, dict[str, float]] = {}
    cache_file = _cache_path()
    if use_cache and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            log.warning("Feature cache unreadable; rebuilding from scratch.")

    rows: list[dict[str, float]] = []
    dirty = False

    for index, sample in enumerate(samples, 1):
        key = f"{sample.id}:{use_models}"
        if key in cache:
            rows.append(cache[key])
            continue
        log.info("[%d/%d] extracting %s (%s)", index, len(samples), sample.id, sample.label)
        feats = extract_one(sample.text, use_models=use_models)
        cache[key] = feats
        rows.append(feats)
        dirty = True

    if dirty and use_cache:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache))

    # Union of keys, sorted for a stable column order across runs.
    feature_names = sorted({name for row in rows for name in row})

    matrix = np.full((len(rows), len(feature_names)), np.nan, dtype=np.float64)
    for r, row in enumerate(rows):
        for c, name in enumerate(feature_names):
            value = row.get(name)
            if value is not None and np.isfinite(value):
                matrix[r, c] = value

    # HistGradientBoosting handles NaN natively — a missing feature (spaCy or
    # the models unavailable for that sample) stays missing rather than being
    # imputed to a value that means something else.
    y = np.array([1.0 if s.label == "machine" else 0.0 for s in samples])

    return Dataset(
        X=matrix,
        y=y,
        feature_names=feature_names,
        sample_ids=[s.id for s in samples],
        registers=[s.register for s in samples],
        generators=[s.generator or "human" for s in samples],
    )
