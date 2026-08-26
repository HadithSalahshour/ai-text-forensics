"""Calibration harness tests.

These use a temporary corpus directory so they never touch the real one — the
user's own writing lives there and tests must not read, write, or delete it.
"""

import numpy as np
import pytest

from humanizer.calibration import corpus, dataset, fitted
from humanizer.calibration import train as train_module
from humanizer.config import settings

LONG_HUMAN = (
    "Our office went remote in March 2020 and never came back. Not by choice — the "
    "lease ran out in August. Four years in, I have stopped believing most of what "
    "gets written about this. The productivity studies contradict each other. What I "
    "actually noticed was that the meetings got worse before they got better. For "
    "about eight months everyone overcorrected, and we had a daily standup nobody "
    "needed plus a Friday social that three people ever attended. We killed most of "
    "it eventually. The thing nobody warned me about was hiring: on paper a global "
    "talent pool sounds great, but my team now spans nine time zones and there is "
    "exactly one hour where everyone is awake. We write a lot more than we used to. "
    "That turned out to suit us, though it took two years and a couple of bad hires "
    "to work out. Isolation is real but it lands unevenly across the team. "
) * 2

LONG_MACHINE = (
    "In today's fast-paced world, remote work has fundamentally transformed the way "
    "organizations operate. As companies navigate the ever-evolving landscape of "
    "distributed collaboration, it is important to note that the implications extend "
    "far beyond simple logistics. The transition underscores a pivotal shift in "
    "organizational culture. Companies must delve into the intricate dynamics of "
    "virtual team building, communication protocols, and employee engagement. "
    "Moreover, the multifaceted nature of these challenges requires a comprehensive "
    "approach that addresses technological infrastructure, management practices, and "
    "individual wellbeing. Furthermore, organizations can harness a global talent "
    "pool, fostering diversity and innovation within their teams. "
) * 2


@pytest.fixture(autouse=True)
def isolated_corpus(tmp_path, monkeypatch):
    """Point the corpus at a temp directory for the duration of each test."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    fitted.reset_cache()
    yield tmp_path
    fitted.reset_cache()


def _fill(n_human: int, n_machine: int) -> None:
    for i in range(n_human):
        corpus.add(f"{LONG_HUMAN} Sample variation number {i} follows here.", "human")
    for i in range(n_machine):
        corpus.add(
            f"{LONG_MACHINE} Sample variation number {i} follows here.",
            "machine",
            generator="test-model",
        )


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


def test_add_and_load_roundtrip():
    sample = corpus.add(LONG_HUMAN, "human", register="casual", topic="remote work")
    assert sample is not None

    loaded = corpus.load("human")
    assert len(loaded) == 1
    assert loaded[0].register == "casual"
    assert loaded[0].topic == "remote work"
    assert loaded[0].text.strip() == LONG_HUMAN.strip()


def test_short_samples_are_rejected():
    assert corpus.add("Too short to be useful.", "human") is None
    assert corpus.load() == []


def test_exact_duplicates_are_not_stored_twice():
    assert corpus.add(LONG_HUMAN, "human") is not None
    assert corpus.add(LONG_HUMAN, "human") is None
    assert len(corpus.load()) == 1


def test_invalid_label_rejected():
    with pytest.raises(ValueError, match="label must be one of"):
        corpus.add(LONG_HUMAN, "robot")


def test_summary_counts_by_label():
    _fill(3, 5)
    info = corpus.summary()
    assert info["by_label"] == {"human": 3, "machine": 5}
    assert info["total"] == 8
    assert info["total_words"] > 0


# --------------------------------------------------------------------------
# Audit — the guard against confidently wrong models
# --------------------------------------------------------------------------


def test_audit_blocks_on_too_few_samples():
    _fill(2, 2)
    assert any(w.severity == "blocker" for w in corpus.audit())


def test_audit_warns_on_class_imbalance():
    _fill(11, 40)
    messages = " ".join(w.message for w in corpus.audit())
    assert "imbalance" in messages.lower()


def test_audit_warns_on_single_generator():
    _fill(12, 12)
    messages = " ".join(w.message for w in corpus.audit())
    assert "one generator" in messages


def test_audit_clean_corpus_has_no_blockers():
    for i in range(12):
        corpus.add(f"{LONG_HUMAN} variant {i}", "human", register="casual")
    for i in range(12):
        gen = "model-a" if i % 2 else "model-b"
        corpus.add(f"{LONG_MACHINE} variant {i}", "machine", register="casual", generator=gen)
    assert [w for w in corpus.audit() if w.severity == "blocker"] == []


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


def test_dataset_shape_and_labels():
    _fill(4, 6)
    data = dataset.build(use_models=False, use_cache=False)
    assert data.X.shape == (10, len(data.feature_names))
    assert data.n_human == 4
    assert data.n_machine == 6
    assert set(np.unique(data.y)) == {0.0, 1.0}


def test_dataset_excludes_raw_size_counters():
    """Document length is an artifact of how the corpus was gathered, not a
    property of machine text. Leaving it in lets the model cheat."""
    _fill(3, 3)
    data = dataset.build(use_models=False, use_cache=False)
    for banned in ("word_count", "sentence_count", "paragraph_count"):
        assert banned not in data.feature_names


def test_dataset_raises_on_empty_corpus():
    with pytest.raises(RuntimeError, match="empty"):
        dataset.build(use_models=False, use_cache=False)


def test_feature_columns_are_stable_across_builds():
    _fill(3, 3)
    first = dataset.build(use_models=False, use_cache=False)
    second = dataset.build(use_models=False, use_cache=False)
    assert first.feature_names == second.feature_names


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_tpr_at_fpr_perfect_separation():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.05, 0.9, 0.95, 0.85, 0.99])
    assert train_module._tpr_at_fpr(y, scores, 0.01) == 1.0


def test_tpr_at_fpr_no_separation():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    scores = np.array([0.5] * 8)
    assert train_module._tpr_at_fpr(y, scores, 0.01) == 0.0


def test_training_refuses_on_blocker():
    """A corpus too small to learn from must produce a refusal, not a
    traceback and not a confident-looking number."""
    _fill(2, 20)
    report = train_module.train(use_models=False, save=False)
    assert report.trustworthy is False
    assert any(w.startswith("[blocker]") for w in report.warnings)
    assert np.isnan(report.metrics.auroc_mean)


def test_training_reports_coarse_fpr_honestly():
    """With few human samples a 1% FPR is not measurable; the report must say
    so rather than quietly reporting a coarser rate as if it were 1%."""
    _fill(12, 12)
    report = train_module.train(use_models=False, save=False)
    assert report.metrics.fpr_resolvable is False
    assert report.metrics.target_fpr > 0.01
    assert any("resolve a false-positive" in w for w in report.warnings)


def test_training_produces_metrics_on_viable_corpus():
    _fill(15, 15)
    report = train_module.train(use_models=False, save=False)
    assert 0.0 <= report.metrics.auroc_mean <= 1.0
    assert report.metrics.n_samples == 30


# --------------------------------------------------------------------------
# Fitted model loading
# --------------------------------------------------------------------------


def test_no_fitted_model_returns_none():
    assert fitted.load() is None


def test_stale_schema_version_is_refused(tmp_path):
    """A model trained on a different feature set must be refused, not fed
    columns that no longer mean what it learned."""
    import joblib

    joblib.dump(
        {
            "estimator": None,
            "calibrator": None,
            "feature_names": ["a", "b"],
            "schema_version": dataset.SCHEMA_VERSION + 99,
            "metrics": {},
        },
        tmp_path / train_module.MODEL_FILE,
    )
    fitted.reset_cache()
    assert fitted.load() is None


def test_untrustworthy_model_is_not_applied():
    _fill(15, 15)
    train_module.train(use_models=False, save=True)
    fitted.reset_cache()
    model = fitted.load()
    if model is not None:
        # 30 samples cannot resolve a 1% FPR, so this must not be trusted.
        assert model.trustworthy is False


def test_ensemble_ignores_untrustworthy_model():
    from humanizer import ingest, segment
    from humanizer.detect import ensemble

    _fill(15, 15)
    train_module.train(use_models=False, save=True)
    fitted.reset_cache()

    doc = segment.build(**ingest.from_text(LONG_MACHINE))
    result = ensemble.analyze(doc, use_models=False)
    assert result.fitted_used is False
    assert result.heuristic_score == result.score
