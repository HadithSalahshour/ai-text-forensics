"""Split text into paragraphs and sentences, preserving character offsets.

Offsets are the contract with the rest of the system: the UI paints the heatmap
with them and the humanizer splices rewrites back with them. Every span must
satisfy `document.text[s.start:s.end] == s.text`.
"""

from __future__ import annotations

import re

from .document import Document, Paragraph, Sentence
from .nlp import fallback_sentence_spans, get_nlp

_PARA_SPLIT = re.compile(r"\n\s*\n")


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _PARA_SPLIT.finditer(text):
        chunk = text[cursor : match.start()]
        if chunk.strip():
            lead = len(chunk) - len(chunk.lstrip())
            trail = len(chunk) - len(chunk.rstrip())
            spans.append((cursor + lead, match.start() - trail))
        cursor = match.end()

    tail = text[cursor:]
    if tail.strip():
        lead = len(tail) - len(tail.lstrip())
        trail = len(tail) - len(tail.rstrip())
        spans.append((cursor + lead, len(text) - trail))
    return spans


# spaCy resolves every abbreviation we tested correctly except clock
# abbreviations, where the trailing period is simultaneously abbreviation-final
# and sentence-final. It under-splits "at 5 p.m. They left."
_TIME_ABBREV = frozenset({"a.m.", "p.m.", "am.", "pm."})

# ...but "at 5 p.m. Tuesday" is a genuine continuation. The POS tag of the
# following token separates the two cleanly: a proper noun or number continues
# the sentence, anything else starts a new one.
_CONTINUATION_POS = frozenset({"PROPN", "NUM"})


def _time_abbreviation_splits(spacy_doc) -> list[int]:
    """Character offsets where a clock abbreviation actually ended a sentence."""
    splits: list[int] = []
    tokens = [t for t in spacy_doc if not t.is_space]
    # strict=False is intended: this is a pairwise walk, so the offset slice is
    # deliberately one shorter.
    for token, following in zip(tokens, tokens[1:], strict=False):
        if token.text.lower() not in _TIME_ABBREV:
            continue
        if not following.text[:1].isupper():
            continue
        if following.pos_ in _CONTINUATION_POS:
            continue
        splits.append(following.idx)
    return splits


def _apply_splits(text: str, spans: list[tuple[int, int]], splits: list[int]):
    if not splits:
        return spans
    out: list[tuple[int, int]] = []
    for start, end in spans:
        cuts = sorted(p for p in splits if start < p < end)
        cursor = start
        for cut in cuts:
            piece = text[cursor:cut].rstrip()
            if piece:
                out.append((cursor, cursor + len(piece)))
            cursor = cut
        if text[cursor:end].strip():
            out.append((cursor, end))
    return out


def _sentence_spans_spacy(nlp, text: str) -> tuple[list[tuple[int, int]], object]:
    doc = nlp(text)
    spans = []
    for sent in doc.sents:
        stripped = sent.text.strip()
        if not stripped:
            continue
        lead = sent.text.index(stripped[0]) if stripped else 0
        start = sent.start_char + lead
        spans.append((start, start + len(stripped)))
    return _apply_splits(text, spans, _time_abbreviation_splits(doc)), doc


def segment(document: Document, *, use_nlp: bool = True) -> Document:
    """Populate `document.paragraphs` (and their sentences) in place.

    `use_nlp=False` swaps spaCy for the regex splitter. The humanizer's inner
    loop re-segments a candidate document hundreds of times per run, and a
    spaCy parse per candidate makes that unusably slow; the regex splitter is
    ~100x faster and accurate enough to rank candidates. Final scoring always
    uses the full path.
    """
    text = document.text
    nlp = get_nlp() if use_nlp else None

    spacy_doc = None
    if nlp is not None:
        # Parse the whole document once and cache it; features.py reuses this
        # rather than paying for a second parse.
        sent_spans, spacy_doc = _sentence_spans_spacy(nlp, text)
        document.meta["_spacy"] = spacy_doc
    else:
        sent_spans = None

    paragraphs: list[Paragraph] = []
    sentence_counter = 0

    for para_index, (p_start, p_end) in enumerate(_paragraph_spans(text)):
        para_text = text[p_start:p_end]
        paragraph = Paragraph(text=para_text, start=p_start, end=p_end, index=para_index)

        if sent_spans is not None:
            local = [(s, e) for s, e in sent_spans if s >= p_start and e <= p_end]
        else:
            local = [
                (p_start + s, p_start + e) for s, e in fallback_sentence_spans(para_text)
            ]

        if not local:
            local = [(p_start, p_end)]

        for s_start, s_end in local:
            paragraph.sentences.append(
                Sentence(
                    text=text[s_start:s_end],
                    start=s_start,
                    end=s_end,
                    para_index=para_index,
                    index=sentence_counter,
                )
            )
            sentence_counter += 1

        paragraphs.append(paragraph)

    document.paragraphs = paragraphs
    return document


def build(
    text: str,
    *,
    source_format: str = "txt",
    source_name: str = "input",
    use_nlp: bool = True,
) -> Document:
    """Convenience: text -> fully segmented Document."""
    return segment(
        Document(text=text, source_format=source_format, source_name=source_name),
        use_nlp=use_nlp,
    )
