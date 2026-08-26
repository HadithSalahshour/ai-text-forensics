"""Tier A tests.

These load two local language models (~2 GB) and are skipped automatically
when the models aren't present, so the Phase 1 suite still runs standalone.
"""

from pathlib import Path

import pytest

from humanizer import ingest, segment
from humanizer.detect import ensemble, statistical
from humanizer.detect.backend import get_models

FIXTURES = Path(__file__).parent / "fixtures"

models_required = pytest.mark.skipif(
    not get_models().available, reason="local models not installed"
)


def _doc(name: str):
    return segment.build(**ingest.from_file(FIXTURES / f"{name}.txt"))


@pytest.fixture(scope="module")
def generated():
    return _doc("generated_sample")


@pytest.fixture(scope="module")
def human():
    return _doc("human_sample")


# --------------------------------------------------------------------------
# Statistic direction
#
# Validated against genuine model output, not a hand-written imitation. Tier A
# measures sampling signature; a human pastiche of LLM style does not have one,
# so it is the wrong control for these assertions.
# --------------------------------------------------------------------------


@models_required
def test_analysis_is_available(generated):
    result = statistical.analyze(generated)
    assert result.available, result.error
    assert result.n_tokens > 50


@models_required
def test_fast_detect_higher_for_machine_text(generated, human):
    machine = statistical.analyze(generated).fast_detect
    person = statistical.analyze(human).fast_detect
    assert machine is not None and person is not None
    assert machine > person, f"machine={machine} should exceed human={person}"
    assert machine > 1.0, "genuine model output should show clear positive curvature"


@models_required
def test_binoculars_lower_for_machine_text(generated, human):
    """Binoculars is inverted by construction: low ratio means machine."""
    machine = statistical.analyze(generated).binoculars
    person = statistical.analyze(human).binoculars
    assert machine < person
    assert machine < 0.9, "published threshold sits near 0.9"


@models_required
def test_perplexity_lower_for_machine_text(generated, human):
    assert statistical.analyze(generated).perplexity < statistical.analyze(human).perplexity


@models_required
def test_log_rank_lower_for_machine_text(generated, human):
    assert statistical.analyze(generated).mean_log_rank < statistical.analyze(human).mean_log_rank


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


@models_required
def test_long_document_is_windowed_not_truncated():
    """Documents past one window must be scored across all of it. Every
    downstream statistic is a ratio of sums, which is what makes the windowed
    result equal the whole-document result."""
    base = (FIXTURES / "human_sample.txt").read_text()
    doc = segment.build(**ingest.from_text("\n\n".join([base] * 6)))
    result = statistical.analyze(doc)
    assert result.available
    assert not result.truncated
    assert result.n_tokens > 1024, "expected multiple windows"
    assert result.perplexity > 1.0


@models_required
@pytest.mark.parametrize("copies", [1, 3, 6])
def test_every_token_is_scored_exactly_once(copies):
    """The windowing invariant.

    Position i predicts token i+1, so exactly T-1 of T tokens are scorable.
    The stride logic must cover all of them with no gap and no double count —
    if it drifts, every ratio-of-sums statistic silently skews.

    Note this is asserted on token bookkeeping rather than on score stability:
    concatenating copies of a text makes the later copies trivially
    predictable, so perplexity legitimately collapses and cannot be used as a
    windowing check.
    """
    base = (FIXTURES / "human_sample.txt").read_text()
    doc = segment.build(**ingest.from_text("\n\n".join([base] * copies)))

    models = get_models()
    expected = len(models._tokenizer(doc.text)["input_ids"]) - 1

    result = statistical.analyze(doc)
    assert result.n_tokens == expected, (
        f"{copies} copies: scored {result.n_tokens} of {expected} scorable tokens"
    )


@models_required
def test_multi_window_document_scores_sanely():
    """Distinct content across several windows — no repetition artifact, so
    the statistics should stay in a plausible range."""
    parts = [
        (FIXTURES / "human_sample.txt").read_text(),
        (FIXTURES / "ai_sample.txt").read_text(),
        (FIXTURES / "generated_sample.txt").read_text(),
    ]
    doc = segment.build(**ingest.from_text("\n\n".join(parts)))
    result = statistical.analyze(doc)
    assert result.n_tokens > 1024, "expected more than one window"
    assert not result.truncated
    assert 2.0 < result.perplexity < 100.0
    assert result.binoculars is not None and 0.5 < result.binoculars < 1.5


@models_required
def test_token_offsets_align_with_token_count(generated):
    statistical.analyze(generated)
    stats = generated.meta["_token_stats"]
    assert len(stats.token_logp) == stats.n_tokens
    assert len(stats.token_offsets) == stats.n_tokens


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------


@models_required
def test_ensemble_separates_machine_from_human(generated, human):
    machine = ensemble.analyze(generated)
    person = ensemble.analyze(human)
    assert machine.score > 65, f"machine text scored {machine.score}"
    assert person.score < 25, f"human text scored {person.score}"
    assert machine.tier_scores["statistical"] > 70
    assert person.tier_scores["statistical"] < 30


@models_required
def test_ensemble_reports_statistical_availability(generated):
    assert ensemble.analyze(generated).statistical_available is True


def test_ensemble_degrades_without_models(human):
    """With Tier A off, the remaining tiers must still produce a usable score
    and the missing weight must redistribute rather than drag the score down."""
    result = ensemble.analyze(human, use_models=False)
    assert result.statistical_available is False
    assert "statistical" not in result.tier_scores
    assert 0 <= result.score <= 100
    assert "Statistical tier disabled" in result.note


def test_ensemble_still_flags_style_without_models():
    """The pastiche case: human-written text imitating LLM style has no
    sampling signature, so the style tiers have to carry it alone."""
    pastiche = _doc("ai_sample")
    assert ensemble.analyze(pastiche, use_models=False).score > 55


def test_verdict_bands_match_score():
    from humanizer.detect.ensemble import Detection

    assert Detection(score=10.0).verdict == "Reads as human"
    assert Detection(score=30.0).verdict == "Mostly human"
    assert Detection(score=50.0).verdict == "Mixed signals"
    assert Detection(score=70.0).verdict == "Likely AI-generated"
    assert Detection(score=90.0).verdict == "Very likely AI-generated"


def test_short_text_flagged_unreliable():
    doc = segment.build(**ingest.from_text("A short note. Nothing more to say here."))
    assert ensemble.analyze(doc, use_models=False).reliable is False
