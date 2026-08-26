"""spaCy pipeline singleton, with a graceful fallback.

Everything here runs locally. `en_core_web_sm` is ~13 MB and gives us POS tags,
dependency depth, and named entities — the last of which is what the rewrite
fact-guard uses to prove no new entity was invented.

If the model isn't installed yet, we degrade to a regex sentence splitter so
Phase 1 features still work; the syntactic features just return None.
"""

from __future__ import annotations

import functools
import logging
import re

from .config import settings

log = logging.getLogger(__name__)

# Abbreviations that must not end a sentence. Used only by the fallback
# splitter; spaCy handles these itself.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "cf", "al", "fig", "no", "vol", "pp", "ed", "eds", "approx",
    "dept", "univ", "inc", "ltd", "co", "corp", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


@functools.lru_cache(maxsize=1)
def get_nlp():
    """Load the spaCy pipeline once. Returns None if unavailable."""
    try:
        import spacy
    except ImportError:
        log.warning("spaCy not installed — syntactic features disabled.")
        return None

    try:
        return spacy.load(settings.spacy_model)
    except OSError:
        log.warning(
            "spaCy model %r not found. Install it with:\n"
            "    python -m spacy download %s\n"
            "Falling back to regex segmentation; syntactic features disabled.",
            settings.spacy_model,
            settings.spacy_model,
        )
        return None


def has_nlp() -> bool:
    return get_nlp() is not None


def fallback_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Regex sentence splitter used when spaCy is unavailable.

    Returns (start, end) character offsets. Conservative about abbreviations
    so "Dr. Smith" doesn't become two sentences.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENT_BOUNDARY.finditer(text):
        end = match.start()
        candidate = text[start:end]
        # Look at the token before the boundary punctuation.
        tail = candidate.rstrip("\"')]").rstrip()
        last_word = tail.split()[-1].rstrip(".").lower() if tail.split() else ""
        if last_word in _ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
            continue  # initial like "J." or a known abbreviation — keep going
        if candidate.strip():
            spans.append((start, end))
        start = match.end()

    if text[start:].strip():
        spans.append((start, len(text)))
    return spans
