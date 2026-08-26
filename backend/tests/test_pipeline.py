from pathlib import Path

import pytest

from humanizer import ingest, segment
from humanizer.detect import features as F
from humanizer.detect import rulescore
from humanizer.document import Document

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def ai_doc() -> Document:
    return segment.build(**ingest.from_file(FIXTURES / "ai_sample.txt"))


@pytest.fixture(scope="module")
def human_doc() -> Document:
    return segment.build(**ingest.from_file(FIXTURES / "human_sample.txt"))


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------


def test_markdown_syntax_stripped_prose_kept():
    payload = ingest.from_text(
        "# Heading\n\nSome **bold** and *italic* and `code` and "
        "[a link](http://x.com).\n\n- bullet one\n- bullet two\n",
        source_format="md",
    )
    text = payload["text"]
    assert "#" not in text and "**" not in text and "`" not in text
    assert "http://x.com" not in text
    assert "a link" in text  # link label survives
    assert "bullet one" in text


def test_hard_wrapped_lines_are_rejoined():
    payload = ingest.from_text("The quick brown\nfox jumps over\nthe lazy dog.")
    assert "\n" not in payload["text"]


def test_blank_line_still_separates_paragraphs():
    doc = segment.build(**ingest.from_text("First para here.\n\nSecond para here."))
    assert len(doc.paragraphs) == 2


def test_de_hyphenation_across_line_break():
    payload = ingest.from_text("This is a hyphen-\nated word.")
    assert "hyphenated" in payload["text"]


def test_punctuation_signals_survive_normalization():
    # Curly quotes and em-dashes are detection evidence — normalization must
    # not launder them away.
    payload = ingest.from_text("He said “hello” — then left.")
    assert "“" in payload["text"]
    assert "—" in payload["text"]


def test_unsupported_format_rejected(tmp_path):
    bad = tmp_path / "notes.rtf"
    bad.write_text("x")
    with pytest.raises(ingest.UnsupportedFormatError):
        ingest.from_file(bad)


# --------------------------------------------------------------------------
# Segmentation — the offset contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["ai_sample", "human_sample"])
def test_sentence_offsets_round_trip(fixture):
    """The contract the UI heatmap and the rewrite splicer both depend on."""
    doc = segment.build(**ingest.from_file(FIXTURES / f"{fixture}.txt"))
    assert doc.sentences, "expected at least one sentence"
    for sentence in doc.sentences:
        assert doc.text[sentence.start : sentence.end] == sentence.text


@pytest.mark.parametrize("fixture", ["ai_sample", "human_sample"])
def test_paragraph_offsets_round_trip(fixture):
    doc = segment.build(**ingest.from_file(FIXTURES / f"{fixture}.txt"))
    for para in doc.paragraphs:
        assert doc.text[para.start : para.end] == para.text


def test_sentences_are_ordered_and_non_overlapping(ai_doc):
    sentences = ai_doc.sentences
    for prev, nxt in zip(sentences, sentences[1:], strict=False):
        assert prev.end <= nxt.start
        assert prev.index < nxt.index


def test_abbreviations_do_not_split_sentences():
    doc = segment.build(**ingest.from_text("Dr. Smith met Mr. Jones at 5 p.m. They talked."))
    assert len(doc.sentences) == 2


def test_clock_abbreviation_continuation_is_not_split():
    """The other half of the p.m. fix: a following proper noun means the
    sentence continues, so splitting there would be wrong."""
    doc = segment.build(**ingest.from_text("We met at 5 p.m. Tuesday in the hall."))
    assert len(doc.sentences) == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ("See Fig. 3 for details. The trend is clear.", 2),
        ("They arrived Jan. 5. Nobody was there.", 2),
        ("The U.S. economy grew. Analysts were surprised.", 2),
        ("It was 9 a.m. The room was cold.", 2),
    ],
)
def test_abbreviation_boundary_cases(text, expected):
    assert len(segment.build(**ingest.from_text(text)).sentences) == expected


# --------------------------------------------------------------------------
# Span replacement
# --------------------------------------------------------------------------


def test_replace_spans_applies_all_edits():
    doc = Document(text="alpha beta gamma")
    out = doc.replace_spans([(0, 5, "ALPHA"), (11, 16, "GAMMA")])
    assert out == "ALPHA beta GAMMA"


def test_replace_spans_handles_length_changes():
    doc = Document(text="one two three")
    out = doc.replace_spans([(0, 3, "a"), (4, 7, "muchlonger")])
    assert out == "a muchlonger three"


def test_replace_spans_rejects_overlap():
    doc = Document(text="abcdefghij")
    with pytest.raises(ValueError, match="overlapping"):
        doc.replace_spans([(0, 5, "x"), (3, 8, "y")])


# --------------------------------------------------------------------------
# Feature maths
# --------------------------------------------------------------------------


def test_cv_is_zero_for_constant_input():
    assert F._cv([5.0, 5.0, 5.0]) == 0.0


def test_cv_rises_with_spread():
    tight = F._cv([10.0, 11.0, 10.0, 9.0])
    loose = F._cv([2.0, 25.0, 6.0, 40.0])
    assert loose > tight


def test_mtld_rewards_varied_vocabulary():
    repetitive = ["cat", "dog"] * 60
    varied = [f"w{i}" for i in range(120)]
    assert F._mtld(varied) > F._mtld(repetitive)


def test_mtld_returns_zero_for_short_input():
    assert F._mtld(["a", "b", "c"]) == 0.0


def test_entropy_zero_for_single_symbol():
    from collections import Counter

    assert F._entropy(Counter({"a": 10})) == 0.0


def test_dependency_depth_terminates():
    """Regression: comparing spaCy tokens by identity never reaches the root,
    which pinned every depth at the loop guard."""
    doc = segment.build(**ingest.from_text("The committee that reviewed the proposal met."))
    feats = F.syntactic_features(doc)
    if feats:  # skipped when spaCy is unavailable
        assert 0 < feats["dep_depth_mean"] < 20


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_ai_scores_far_above_human(ai_doc, human_doc):
    ai = rulescore.score_document(ai_doc).score
    human = rulescore.score_document(human_doc).score
    assert ai > 60, f"AI sample scored only {ai}"
    assert human < 35, f"human sample scored {human}"
    assert ai - human > 30


def test_score_is_bounded(ai_doc):
    assert 0 <= rulescore.score_document(ai_doc).score <= 100


def test_short_text_flagged_unreliable():
    doc = segment.build(**ingest.from_text("This is a short sentence. So is this one."))
    assert rulescore.score_document(doc).reliable is False


def test_long_text_not_flagged_unreliable(ai_doc):
    assert rulescore.score_document(ai_doc).reliable is True


def test_signals_carry_evidence_strings(ai_doc):
    for signal in rulescore.score_document(ai_doc).top_signals(5):
        assert signal.evidence
        assert 0.0 <= signal.score <= 1.0


def test_sentence_scores_cover_every_sentence(ai_doc):
    scored = rulescore.score_sentences(ai_doc)
    assert len(scored) == len(ai_doc.sentences)
    assert {i for i, _, _ in scored} == {s.index for s in ai_doc.sentences}


def test_marker_laden_sentence_outscores_plain_one():
    doc = segment.build(
        **ingest.from_text(
            "Moreover, we must delve into the intricate tapestry of this "
            "multifaceted realm.\n\n"
            "The dog ate at 6pm."
        )
    )
    scored = dict((i, s) for i, s, _ in rulescore.score_sentences(doc))
    assert scored[0] > scored[1]


def test_sentence_reasons_are_populated_for_flagged_text():
    doc = segment.build(
        **ingest.from_text("Furthermore, this serves as a testament to the pivotal shift.")
    )
    _, score, reasons = rulescore.score_sentences(doc)[0]
    assert score > 40
    assert reasons
