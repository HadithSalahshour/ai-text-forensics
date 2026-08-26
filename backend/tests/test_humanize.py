"""Humanizer tests.

The fact-preservation tests are the load-bearing ones. A rewriter that quietly
alters a number or invents a citation does real damage to the user's work, so
those properties are asserted directly rather than inferred from the score.
"""

from pathlib import Path

import pytest

from humanizer import ingest, segment
from humanizer.humanize import guards, loop, transforms

FIXTURES = Path(__file__).parent / "fixtures"


def _doc(name: str):
    return segment.build(**ingest.from_file(FIXTURES / f"{name}.txt"))


# --------------------------------------------------------------------------
# Fact guard — no fabrication, no silent loss
# --------------------------------------------------------------------------


def test_rejects_invented_number():
    result = guards.check_facts("Revenue grew last year.", "Revenue grew 47% last year.")
    assert not result.passed
    assert any("invents number" in r for r in result.reasons)


def test_rejects_dropped_number():
    result = guards.check_facts("Revenue grew 47% last year.", "Revenue grew a lot last year.")
    assert not result.passed
    assert any("drops number" in r for r in result.reasons)


def test_accepts_number_spelled_as_word():
    """'5' and 'five' are the same fact and must not read as a change."""
    result = guards.check_facts("We ran 5 trials.", "We ran five trials.")
    assert result.passed, result.reasons


def test_rejects_invented_citation():
    result = guards.check_facts(
        "Prior work disagrees.", "Prior work disagrees (Smith, 2019)."
    )
    assert not result.passed
    assert any("invents citation" in r for r in result.reasons)


def test_rejects_dropped_citation():
    result = guards.check_facts(
        "Prior work disagrees (Smith, 2019).", "Prior work disagrees."
    )
    assert not result.passed
    assert any("drops citation" in r for r in result.reasons)


def test_rejects_invented_entity():
    result = guards.check_facts(
        "The study was published last year.",
        "The Stanford study was published last year.",
    )
    assert not result.passed
    assert any("invents entity" in r for r in result.reasons)


def test_allows_partial_entity_reference():
    """NER boundaries are noisy — shortening a known entity is not invention."""
    result = guards.check_facts(
        "Stanford University released the data.", "Stanford released the data."
    )
    assert result.passed, result.reasons


def test_preserving_rewrite_passes_fact_guard():
    result = guards.check_facts(
        "The team shipped 3 features in 2024.",
        "In 2024 the team shipped 3 features.",
    )
    assert result.passed, result.reasons


# --------------------------------------------------------------------------
# Semantic guard
# --------------------------------------------------------------------------


def test_identical_text_is_rejected_as_unchanged():
    assert not guards.check("The cat sat down.", "The cat sat down.").passed


def test_empty_candidate_rejected():
    assert not guards.check("The cat sat down.", "   ").passed


def test_meaning_drift_is_rejected():
    result = guards.check(
        "Remote work improved our retention numbers substantially.",
        "The weather in Portugal is pleasant during spring.",
    )
    assert not result.passed
    assert any("meaning drift" in r for r in result.reasons)


def test_paraphrase_passes_semantic_guard():
    result = guards.check(
        "Companies must delve into the intricate dynamics of team building.",
        "Companies must examine the complex dynamics of team building.",
    )
    assert result.passed, result.reasons


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------


def test_plausibility_screen_rejects_wreckage():
    assert not transforms.is_plausible("and then , the")
    assert not transforms.is_plausible("lowercase start here.")
    assert not transforms.is_plausible("No ending punctuation")
    assert not transforms.is_plausible("Unbalanced (parens here.")
    assert transforms.is_plausible("The committee met on Tuesday.")


def test_drop_opening_connective():
    out = transforms.drop_opening_connective(
        "Moreover, the results were inconclusive.", "professional", __import__("random").Random(0)
    )
    assert out and not out[0].text.lower().startswith("moreover")
    assert "results were inconclusive" in out[0].text


def test_marker_substitution_replaces_and_keeps_case():
    out = transforms.generate(
        "Delve into the intricate details of the system.", register="professional"
    )
    assert out
    assert any("delve" not in c.text.lower() for c in out)
    assert all(c.text[0].isupper() for c in out)


def test_bulk_substitution_fixes_multiple_markers_at_once():
    """One accepted candidate has to be able to clear a whole sentence — the
    loop applies at most one edit per sentence per iteration."""
    text = (
        "Moreover, we must delve into the intricate and multifaceted realm of "
        "seamless collaboration."
    )
    out = transforms.generate(text, register="professional")
    bulk = [c for c in out if c.transform == "substitute_all"]
    assert bulk, "expected a bulk substitution candidate"
    lowered = bulk[0].text.lower()
    assert sum(w in lowered for w in ("delve", "intricate", "multifaceted", "seamless")) <= 1


def test_split_requires_both_halves_to_stand_alone():
    """A clause with no subject must not become a sentence fragment."""
    out = transforms.split_sentence(
        "The system was rebuilt from scratch, and shipped in March of that year.",
        "professional",
        __import__("random").Random(0),
    )
    for candidate in out:
        for part in candidate.text.split(". "):
            assert len(part.split()) >= 3


def test_generated_candidates_are_all_plausible():
    text = (
        "Furthermore, it is important to note that the system serves as a "
        "testament to good design."
    )
    for candidate in transforms.generate(text, register="professional"):
        assert transforms.is_plausible(candidate.text), candidate.text


# --------------------------------------------------------------------------
# Loop
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["ai_sample", "generated_sample"])
def test_score_improves_on_machine_text(fixture):
    result = loop.humanize(_doc(fixture), strength="aggressive", verify_with_models=False)
    assert result.improvement > 10, f"only improved {result.improvement}"
    assert result.edits


def test_human_text_is_left_alone():
    """Text already below target must not be churned for no reason."""
    result = loop.humanize(_doc("human_sample"), verify_with_models=False)
    assert result.edits == []
    assert result.final_text == result.original_text


def test_meaning_is_preserved_document_wide():
    result = loop.humanize(_doc("ai_sample"), strength="aggressive", verify_with_models=False)
    assert result.document_similarity is not None
    assert result.document_similarity > 0.90


def test_no_facts_are_invented_end_to_end():
    """The property that matters most: run the full loop over text dense with
    numbers, dates, and names, then verify not one was added, dropped, or
    altered."""
    source = (
        "In 2019 the team delivered 14 projects across 3 regions. "
        "Moreover, the intricate coordination with Siemens and the Ministry of "
        "Health required a comprehensive and multifaceted approach. "
        "Furthermore, it is important to note that costs rose by 12% while "
        "headcount fell from 240 to 198 over the same period. "
        "The initiative serves as a testament to the pivotal role of planning, "
        "coordination, and oversight in the ever-evolving landscape of delivery. "
        "Ultimately, leveraging robust frameworks allowed the organization to "
        "navigate these challenges and foster seamless collaboration throughout."
    )
    doc = segment.build(**ingest.from_text(source))
    result = loop.humanize(doc, strength="aggressive", verify_with_models=False)

    before_numbers = guards.extract_numbers(result.original_text)
    after_numbers = guards.extract_numbers(result.final_text)
    assert after_numbers == before_numbers, (
        f"numbers changed: added {after_numbers - before_numbers}, "
        f"lost {before_numbers - after_numbers}"
    )

    for entity in guards.extract_entities(result.final_text):
        assert not guards._entity_is_new(
            entity, guards.extract_entities(result.original_text), result.original_text
        ), f"invented entity: {entity!r}"


def test_search_is_deterministic():
    """Best-first replaced a sequential per-sentence pass precisely because the
    latter made results depend on sentence order."""
    first = loop.humanize(_doc("ai_sample"), strength="moderate", verify_with_models=False)
    second = loop.humanize(_doc("ai_sample"), strength="moderate", verify_with_models=False)
    assert first.final_text == second.final_text
    assert first.score_after == second.score_after


def test_applied_edits_do_not_overlap():
    result = loop.humanize(_doc("ai_sample"), strength="aggressive", verify_with_models=False)
    for edit in result.edits:
        assert edit.original in result.original_text or edit.original.strip()


def test_output_is_valid_segmentable_text():
    result = loop.humanize(
        _doc("generated_sample"), strength="aggressive", verify_with_models=False
    )
    rebuilt = segment.build(result.final_text)
    assert rebuilt.sentences
    for sentence in rebuilt.sentences:
        assert rebuilt.text[sentence.start : sentence.end] == sentence.text


def test_strength_presets_scale_effort():
    light = loop.humanize(_doc("ai_sample"), strength="light", verify_with_models=False)
    aggressive = loop.humanize(
        _doc("ai_sample"), strength="aggressive", verify_with_models=False
    )
    assert aggressive.candidates_considered > light.candidates_considered
    assert aggressive.improvement >= light.improvement


def test_register_detection_ignores_nominalization():
    """Regression: keying academic register off nominalization density
    misclassified nearly all machine text, because heavy nominalization is an
    LLM marker rather than an academic one."""
    heavy = (
        "The implementation of the framework and the development of the process "
        "required the utilization of considerable resources. The integration of "
        "these systems and the adoption of new standards drove the transformation. "
        "The creation of documentation supported the expansion of the programme."
    )
    doc = segment.build(**ingest.from_text(heavy))
    assert loop._register_for(doc, None) != "academic"


def test_explicit_register_override_is_respected():
    doc = _doc("ai_sample")
    assert loop._register_for(doc, "casual") == "casual"
