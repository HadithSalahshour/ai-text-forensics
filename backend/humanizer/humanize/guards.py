"""Guards that every candidate rewrite must pass.

Two independent checks, both of which can veto:

  Semantic guard   Embedding cosine between original and rewrite. Catches
                   rewrites that read well but drifted in meaning.

  Fact guard       No invented specifics. This is the one that separates a
                   legitimate style-transfer tool from something that quietly
                   corrupts your work, so it is deliberately strict and
                   deliberately asymmetric:

                     - A number or entity in the rewrite that is NOT in the
                       original is fabrication. Always rejected.
                     - A number from the original that is missing from the
                       rewrite is information loss. Also rejected.
                     - Entities are checked for addition only, not deletion:
                       spaCy's NER is noisy enough that requiring exact
                       round-trip would reject good rewrites for no reason.

Numbers get the stricter treatment because they carry the most meaning per
token and are the easiest for a language model to silently alter.
"""

from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field

from ..config import settings
from ..detect.lexicon import CITATION
from ..nlp import get_nlp

log = logging.getLogger(__name__)

NUMBER_TOKEN = re.compile(r"\d+(?:[.,]\d+)*%?")

# Number words we normalize so "5" and "five" don't read as different facts.
_NUMBER_WORDS: dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "hundred": 100, "thousand": 1000, "million": 1_000_000,
}

# Entity types worth protecting. Ordinals and cardinals are covered by the
# number check; the vaguer types produce too much NER noise to be useful.
_PROTECTED_ENTS = frozenset(
    {"PERSON", "ORG", "GPE", "LOC", "DATE", "TIME", "MONEY", "PERCENT", "WORK_OF_ART", "EVENT"}
)


@dataclass(slots=True)
class GuardResult:
    passed: bool
    similarity: float | None = None
    reasons: list[str] = field(default_factory=list)

    def reject(self, reason: str) -> GuardResult:
        self.passed = False
        self.reasons.append(reason)
        return self


# --------------------------------------------------------------------------
# Fact guard
# --------------------------------------------------------------------------


def extract_numbers(text: str) -> set[float]:
    values: set[float] = set()
    for token in NUMBER_TOKEN.findall(text):
        cleaned = token.rstrip("%").replace(",", "")
        try:
            values.add(float(cleaned))
        except ValueError:
            continue
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            values.add(float(value))
    return values


def extract_entities(text: str) -> set[str]:
    nlp = get_nlp()
    if nlp is None:
        # Fallback: capitalized words that aren't sentence-initial.
        return {m.group(0).lower() for m in re.finditer(r"(?<=[a-z,] )[A-Z][A-Za-z'-]{2,}", text)}
    return {
        ent.text.strip().lower()
        for ent in nlp(text).ents
        if ent.label_ in _PROTECTED_ENTS and len(ent.text.strip()) > 1
    }


def extract_citations(text: str) -> set[str]:
    return {m.group(0) for m in CITATION.finditer(text)}


def _entity_is_new(candidate_ent: str, original_ents: set[str], original_text: str) -> bool:
    """An entity counts as pre-existing if it appears in the original at all —
    as a recognized entity, a substring, or a containing string. NER is noisy
    about boundaries ("Stanford" vs "Stanford University"), and punishing that
    would reject good rewrites."""
    if candidate_ent in original_ents:
        return False
    if candidate_ent in original_text.lower():
        return False
    return not any(candidate_ent in ent or ent in candidate_ent for ent in original_ents)


def check_facts(original: str, candidate: str) -> GuardResult:
    result = GuardResult(passed=True)

    original_numbers = extract_numbers(original)
    candidate_numbers = extract_numbers(candidate)

    invented = candidate_numbers - original_numbers
    if invented:
        result.reject(f"invents number(s): {sorted(invented)}")

    lost = original_numbers - candidate_numbers
    if lost:
        result.reject(f"drops number(s): {sorted(lost)}")

    original_ents = extract_entities(original)
    for ent in extract_entities(candidate):
        if _entity_is_new(ent, original_ents, original):
            result.reject(f"invents entity: {ent!r}")

    original_citations = extract_citations(original)
    for citation in extract_citations(candidate):
        if citation not in original_citations:
            result.reject(f"invents citation: {citation}")
    for citation in original_citations - extract_citations(candidate):
        result.reject(f"drops citation: {citation}")

    return result


# --------------------------------------------------------------------------
# Semantic guard
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.warning(
            "sentence-transformers not installed — falling back to lexical "
            "similarity, which is a weaker meaning guard."
        )
        return None
    try:
        return SentenceTransformer(
            settings.embedding_model, cache_folder=str(settings.model_dir)
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load embedding model (%s); using lexical fallback.", exc)
        return None


_WORD = re.compile(r"[A-Za-z']+")


def _lexical_similarity(a: str, b: str) -> float:
    """Content-word Jaccard. Only used when embeddings are unavailable."""
    tokens_a = {w.lower() for w in _WORD.findall(a)}
    tokens_b = {w.lower() for w in _WORD.findall(b)}
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def similarity(original: str, candidate: str) -> float:
    model = _embedder()
    if model is None:
        return _lexical_similarity(original, candidate)

    import numpy as np

    vectors = model.encode([original, candidate], normalize_embeddings=True)
    return float(np.dot(vectors[0], vectors[1]))


def embeddings_available() -> bool:
    return _embedder() is not None


# --------------------------------------------------------------------------
# Combined
# --------------------------------------------------------------------------


def check(original: str, candidate: str, *, threshold: float | None = None) -> GuardResult:
    """Run both guards. A candidate must pass everything to be eligible."""
    if not candidate.strip():
        return GuardResult(passed=False, reasons=["empty candidate"])
    if candidate.strip() == original.strip():
        return GuardResult(passed=False, reasons=["unchanged"])

    result = check_facts(original, candidate)

    # The lexical fallback scores much lower than cosine for the same meaning,
    # so its bar has to be lower or every rewrite would be rejected.
    if threshold is None:
        threshold = settings.min_sentence_similarity if embeddings_available() else 0.45

    score = similarity(original, candidate)
    result.similarity = score
    if score < threshold:
        result.reject(f"meaning drift (similarity {score:.3f} < {threshold:.2f})")

    return result
