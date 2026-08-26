"""Deterministic rewrite operations.

Scope note: this module implements the transforms that reliably preserve
grammaticality, not every transform that would help the score. Naive
clause-reordering and voice-flipping produce broken English often enough that
they would cost more (in rejected candidates and mangled prose) than the score
points they buy. The riskier operations are left to the Claude pass in Phase 5,
which can be verified by the same guards.

What is here still covers the biggest levers:

  Lexicon    marker-word substitution, stock-phrase removal, connective pruning
  Structure  sentence splitting and merging — which is what actually moves the
             sentence-length variation signal, the single heaviest structural
             feature in the detector

Every transform returns *candidates*, not decisions. The loop scores them and
the guards veto them; nothing here is applied on its own authority.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from ..detect import lexicon
from ..nlp import get_nlp

REGISTERS = ("academic", "professional", "casual", "creative")

# Registers where contractions read naturally.
_CONTRACTION_OK = frozenset({"professional", "casual", "creative"})

_CONTRACTIONS: list[tuple[str, str]] = [
    (r"\bit is\b", "it's"),
    (r"\bthat is\b", "that's"),
    (r"\bthere is\b", "there's"),
    (r"\bwhat is\b", "what's"),
    (r"\bdo not\b", "don't"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bdid not\b", "didn't"),
    (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"),
    (r"\bwas not\b", "wasn't"),
    (r"\bwere not\b", "weren't"),
    (r"\bcannot\b", "can't"),
    (r"\bcan not\b", "can't"),
    (r"\bwill not\b", "won't"),
    (r"\bwould not\b", "wouldn't"),
    (r"\bshould not\b", "shouldn't"),
    (r"\bhas not\b", "hasn't"),
    (r"\bhave not\b", "haven't"),
    (r"\bthey are\b", "they're"),
    (r"\bwe are\b", "we're"),
    (r"\byou are\b", "you're"),
    (r"\bwe will\b", "we'll"),
    (r"\bthey will\b", "they'll"),
    (r"\bwe have\b", "we've"),
    (r"\bthey have\b", "they've"),
]

# Split points, ordered by how safely the right-hand side stands alone.
_SPLIT_PATTERNS = [
    re.compile(r";\s+"),
    re.compile(r",\s+(?=but\s)"),
    re.compile(r",\s+(?=and\s)"),
    re.compile(r",\s+(?=so\s)"),
    re.compile(r",\s+(?=yet\s)"),
    re.compile(r",\s+(?=because\s)"),
    re.compile(r",\s+(?=which\s)"),
    re.compile(r"\s+—\s+"),
]

# A "which" clause becomes a standalone sentence only by promoting the relative
# pronoun to a demonstrative: "..., which matters" -> "This matters."
_RELATIVE_PROMOTION = re.compile(r"^which\s+", re.IGNORECASE)

_LEADING_CONJUNCTION = re.compile(r"^(and|but|so|yet)\s+", re.IGNORECASE)

# "the implementation of X" -> "implementing X"
_NOMINALIZATIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bthe implementation of\b", re.IGNORECASE), "implementing"),
    (re.compile(r"\bthe development of\b", re.IGNORECASE), "developing"),
    (re.compile(r"\bthe application of\b", re.IGNORECASE), "applying"),
    (re.compile(r"\bthe utilization of\b", re.IGNORECASE), "using"),
    (re.compile(r"\bthe creation of\b", re.IGNORECASE), "creating"),
    (re.compile(r"\bthe adoption of\b", re.IGNORECASE), "adopting"),
    (re.compile(r"\bthe integration of\b", re.IGNORECASE), "integrating"),
    (re.compile(r"\bthe expansion of\b", re.IGNORECASE), "expanding"),
    (re.compile(r"\bmake use of\b", re.IGNORECASE), "use"),
    (re.compile(r"\bin order to\b", re.IGNORECASE), "to"),
    (re.compile(r"\bdue to the fact that\b", re.IGNORECASE), "because"),
    (re.compile(r"\bat this point in time\b", re.IGNORECASE), "now"),
    (re.compile(r"\ba large number of\b", re.IGNORECASE), "many"),
    (re.compile(r"\bthe majority of\b", re.IGNORECASE), "most"),
]


@dataclass(slots=True)
class Candidate:
    text: str
    transform: str
    note: str = ""


def _match_case(replacement: str, original: str) -> str:
    """Carry the original's capitalization onto the replacement."""
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    if text:
        text = text[0].upper() + text[1:]
    return text


def is_plausible(text: str) -> bool:
    """Cheap grammaticality screen.

    Not a parser — just enough to catch the obvious wreckage a regex transform
    can produce, so those candidates never reach the (much more expensive)
    guards.
    """
    stripped = text.strip()
    if len(stripped) < 12 or len(stripped.split()) < 3:
        return False
    if not stripped[0].isupper():
        return False
    if stripped[-1] not in ".!?\"'”’":
        return False
    if re.search(r"\s{2,}|\s[,.;:]|,,|\.\.(?!\.)", stripped):
        return False
    if stripped.count("(") != stripped.count(")"):
        return False

    nlp = get_nlp()
    if nlp is not None:
        parsed = nlp(stripped)
        # A finite verb is the minimum bar for a standalone sentence.
        if not any(t.pos_ in {"VERB", "AUX"} for t in parsed):
            return False
    return True


# --------------------------------------------------------------------------
# Lexical transforms
# --------------------------------------------------------------------------


def _options_for(table: dict, key: str, register: str) -> list[str]:
    entry = table[key]
    return entry.get(register) or next(iter(entry.values()))


def _apply_phrase_substitution(text: str, pattern: re.Pattern[str], replacement: str) -> str:
    """Substitute a phrase, carrying any captured group into the replacement.

    Patterns like `the ever-evolving (landscape|world)` capture the noun so the
    replacement can keep it — otherwise the rewrite would drop content.
    """

    def _sub(match: re.Match[str]) -> str:
        out = replacement
        for index, group in enumerate(match.groups(), start=1):
            out = out.replace(f"{{{index}}}", group or "")
        return _match_case(out, match.group(0))

    return pattern.sub(_sub, text, count=1)


def substitute_markers(text: str, register: str, rng: random.Random) -> list[Candidate]:
    """Replace LLM-marker vocabulary with register-appropriate plain wording.

    Emits both single-swap candidates and one "fix everything at once"
    candidate. The greedy loop accepts at most one candidate per sentence per
    iteration, so without the bulk option a sentence carrying four markers
    would need four iterations to clean up — and the loop usually stalls first.
    """
    candidates: list[Candidate] = []

    # Longest keys first so "delve into" wins over "delve".
    single_keys = sorted(lexicon.SUBSTITUTIONS, key=len, reverse=True)
    phrase_keys = sorted(lexicon.PHRASE_SUBSTITUTIONS, key=len, reverse=True)

    for phrase in single_keys:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
        if not pattern.search(text):
            continue
        options = _options_for(lexicon.SUBSTITUTIONS, phrase, register)
        for replacement in rng.sample(options, min(2, len(options))):
            rewritten = pattern.sub(
                lambda m, r=replacement: _match_case(r, m.group(0)), text, count=1
            )
            candidates.append(
                Candidate(_tidy(rewritten), "substitute_marker", f"{phrase} -> {replacement}")
            )

    for raw in phrase_keys:
        pattern = re.compile(raw, re.IGNORECASE)
        if not pattern.search(text):
            continue
        options = _options_for(lexicon.PHRASE_SUBSTITUTIONS, raw, register)
        for replacement in options[:2]:
            rewritten = _apply_phrase_substitution(text, pattern, replacement)
            candidates.append(
                Candidate(_tidy(rewritten), "substitute_phrase", f"{raw[:28]} -> {replacement}")
            )

    # Bulk candidates: every marker and formula in one pass.
    #
    # Two variants with *different* replacement choices, and never a fixed
    # "first option every time". Always taking the most obvious synonym is
    # argmax behavior: it walks the text toward the local maximum of the
    # model's probability surface, which is precisely the signature
    # Fast-DetectGPT measures. Measured directly — a fixed-choice bulk pass
    # cut the lexical tier by half while pushing curvature from -0.41 to 2.22,
    # making the text *more* detectable overall. Varying the choice keeps the
    # lexical win without handing the statistical tier a cleaner signal.
    for variant in range(2):
        chooser = random.Random(rng.random())
        bulk = text
        applied = 0
        for raw in phrase_keys:
            pattern = re.compile(raw, re.IGNORECASE)
            if pattern.search(bulk):
                options = _options_for(lexicon.PHRASE_SUBSTITUTIONS, raw, register)
                bulk = _apply_phrase_substitution(bulk, pattern, chooser.choice(options))
                applied += 1
        for phrase in single_keys:
            pattern = re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            if pattern.search(bulk):
                replacement = chooser.choice(_options_for(lexicon.SUBSTITUTIONS, phrase, register))
                bulk = pattern.sub(lambda m, r=replacement: _match_case(r, m.group(0)), bulk)
                applied += 1

        if applied > 1:
            candidates.append(
                Candidate(_tidy(bulk), "substitute_all", f"{applied} markers, variant {variant}")
            )

    return candidates


def strip_stock_phrase(text: str, register: str, rng: random.Random) -> list[Candidate]:
    """Delete formulaic filler outright. Most of it carries no meaning, which
    is exactly why it reads as machine-written."""
    candidates: list[Candidate] = []

    deletable = [
        r"it is important to note that\s*",
        r"it'?s worth noting that\s*",
        r"it should be noted that\s*",
        r"needless to say,?\s*",
        r"as we all know,?\s*",
        r"when it comes to\s*",
        r"in today'?s (?:fast[- ]paced|ever[- ]changing|modern|digital) world,?\s*",
    ]

    for raw in deletable:
        pattern = re.compile(raw, re.IGNORECASE)
        if not pattern.search(text):
            continue
        stripped = _tidy(pattern.sub("", text, count=1))
        if stripped:
            candidates.append(Candidate(stripped, "strip_stock_phrase", raw[:34]))

    for pattern, replacement in _NOMINALIZATIONS:
        if not pattern.search(text):
            continue
        rewritten = pattern.sub(
            lambda m, r=replacement: _match_case(r, m.group(0)), text, count=1
        )
        candidates.append(Candidate(_tidy(rewritten), "denominalize", replacement))

    # Bulk: strip every deletable formula and denominalize everything at once,
    # for the same reason substitute_all exists.
    bulk = text
    applied = 0
    for raw in deletable:
        pattern = re.compile(raw, re.IGNORECASE)
        if pattern.search(bulk):
            bulk = pattern.sub("", bulk)
            applied += 1
    for pattern, replacement in _NOMINALIZATIONS:
        if pattern.search(bulk):
            bulk = pattern.sub(lambda m, r=replacement: _match_case(r, m.group(0)), bulk)
            applied += 1
    if applied > 1:
        bulk = _tidy(bulk)
        if bulk:
            candidates.append(Candidate(bulk, "strip_all", f"{applied} formulas"))

    return candidates


def drop_opening_connective(text: str, register: str, rng: random.Random) -> list[Candidate]:
    """Remove a mechanical sentence-opening connective.

    The highest value-per-risk transform in the set: deleting "Moreover," is
    almost always safe, and the opening-connective rate is one of the loudest
    lexical signals.
    """
    lowered = text.lstrip().lower()
    for connective in sorted(lexicon.OPENING_CONNECTIVES, key=len, reverse=True):
        if not lowered.startswith(connective):
            continue
        remainder = text.lstrip()[len(connective) :].lstrip(" ,")
        if not remainder:
            return []
        return [Candidate(_tidy(remainder), "drop_connective", connective)]
    return []


def toggle_contractions(text: str, register: str, rng: random.Random) -> list[Candidate]:
    if register not in _CONTRACTION_OK:
        return []
    applicable = [(p, r) for p, r in _CONTRACTIONS if re.search(p, text, re.IGNORECASE)]
    if not applicable:
        return []

    # One contraction, and all of them: a single change is often enough to
    # shift the register without over-informalizing.
    single = rng.choice(applicable)
    one = re.sub(single[0], single[1], text, count=1, flags=re.IGNORECASE)

    everything = text
    for pattern, replacement in applicable:
        everything = re.sub(pattern, replacement, everything, flags=re.IGNORECASE)

    out = [Candidate(_tidy(one), "contract", single[1])]
    if everything != one:
        out.append(Candidate(_tidy(everything), "contract_all", f"{len(applicable)} sites"))
    return out


def break_tricolon(text: str, register: str, rng: random.Random) -> list[Candidate]:
    """Reduce a rule-of-three list to two items.

    Only fires on short, single-word-ish items, where dropping the third is a
    genuine style edit rather than a loss of content.
    """
    pattern = re.compile(r"\b(\w+), (\w+), and (\w+)\b")
    match = pattern.search(text)
    if not match:
        return []
    first, second, third = match.groups()
    if any(len(w) > 14 for w in (first, second, third)):
        return []
    return [
        Candidate(
            _tidy(text[: match.start()] + f"{first} and {second}" + text[match.end() :]),
            "break_tricolon",
            f"dropped {third!r}",
        )
    ]


# --------------------------------------------------------------------------
# Structural transforms
#
# These move sentence-length variation, which carries more weight in the
# detector than any other structural feature.
# --------------------------------------------------------------------------


def _standalone_ok(fragment: str) -> bool:
    """Whether a fragment can stand as its own sentence."""
    nlp = get_nlp()
    if nlp is None:
        return len(fragment.split()) >= 4
    parsed = nlp(fragment)
    has_subject = any(t.dep_ in {"nsubj", "nsubjpass", "expl"} for t in parsed)
    has_verb = any(t.pos_ in {"VERB", "AUX"} for t in parsed)
    return has_subject and has_verb


def split_sentence(text: str, register: str, rng: random.Random) -> list[Candidate]:
    """Split one long sentence into two.

    Both halves must independently parse as sentences, so a clause without its
    own subject is left alone rather than turned into a fragment.
    """
    if len(text.split()) < 16:
        return []

    candidates: list[Candidate] = []
    for pattern in _SPLIT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue

        left = text[: match.start()].rstrip(" ,;")
        right = text[match.end() :].strip()
        if not left or not right:
            continue

        # A leading "and"/"but" is fine to keep occasionally — humans do it —
        # but the clause must stand up without it.
        bare_right = _LEADING_CONJUNCTION.sub("", right)
        if _RELATIVE_PROMOTION.match(bare_right):
            bare_right = _RELATIVE_PROMOTION.sub("This ", bare_right)
            right = bare_right
        if not _standalone_ok(left) or not _standalone_ok(bare_right):
            continue

        if not left.endswith((".", "!", "?")):
            left += "."

        candidates.append(
            Candidate(f"{_tidy(left)} {_tidy(bare_right)}", "split", pattern.pattern)
        )
        # Keeping the conjunction gives a distinctly human opener.
        if bare_right != right and register != "academic":
            candidates.append(
                Candidate(f"{_tidy(left)} {_tidy(right)}", "split_keep_conj", pattern.pattern)
            )
        break

    return candidates


_SUBJECT_PRONOUN = {"Sing": "It", "Plur": "They"}


def split_compound_predicate(
    text: str, register: str, rng: random.Random
) -> list[Candidate]:
    """Split "X does A and does B" into two sentences.

    This is the transform that actually produces *short* sentences. Plain
    clause splitting halves a 24-word sentence into two 12-word ones, which
    does nothing for the short-sentence signal; peeling off a coordinated
    predicate and giving it a pronoun subject yields a genuinely short second
    sentence, the shape machine text almost never has.

    Requires spaCy: the two verbs must actually share a subject, and the
    pronoun has to agree with it in number. Without that check this produces
    confident nonsense.
    """
    nlp = get_nlp()
    if nlp is None or len(text.split()) < 12:
        return []

    parsed = nlp(text.strip())
    candidates: list[Candidate] = []

    for token in parsed:
        # A second verb coordinated onto the main verb.
        if token.dep_ != "conj" or token.pos_ != "VERB":
            continue
        head = token.head
        if head.pos_ != "VERB":
            continue

        subjects = [c for c in head.children if c.dep_ in {"nsubj", "nsubjpass"}]
        if len(subjects) != 1:
            continue
        subject = subjects[0]
        # If the coordinated verb has its own subject it is already a full
        # clause, and plain splitting handles it better.
        if any(c.dep_ in {"nsubj", "nsubjpass"} for c in token.children):
            continue

        # Cut at the conjunction immediately preceding the second verb.
        conjunction = next(
            (c for c in head.children if c.dep_ == "cc" and c.i < token.i), None
        )
        if conjunction is None or conjunction.i <= subject.i:
            continue

        left = parsed[: conjunction.i].text.strip().rstrip(",;")
        right = parsed[token.i :].text.strip()
        if not left or not right:
            continue
        if len(left.split()) < 4 or len(right.split()) < 2:
            continue

        number = subject.morph.get("Number")
        pronoun = _SUBJECT_PRONOUN.get(number[0] if number else "Sing", "It")
        if subject.pos_ == "PRON" and subject.text.lower() in {"i", "we", "you", "they"}:
            pronoun = subject.text.capitalize()

        if not left.endswith((".", "!", "?")):
            left += "."
        second = f"{pronoun} also {right}"
        if not second.endswith((".", "!", "?")):
            second += "."

        candidates.append(
            Candidate(
                f"{_tidy(left)} {_tidy(second)}",
                "split_predicate",
                f"{len(second.split())}-word follow-up",
            )
        )
        break

    return candidates


def merge_sentences(first: str, second: str, register: str) -> list[Candidate]:
    """Join two adjacent sentences.

    Used to widen the length distribution from the other end: merging a pair of
    mid-length sentences creates a long one, which most machine text lacks.
    """
    if not first.rstrip().endswith("."):
        return []
    left = first.rstrip().rstrip(".")
    right = second.strip()
    if not left or not right:
        return []
    if len(left.split()) + len(right.split()) > 45:
        return []

    lowered = right[0].lower() + right[1:]
    out = [Candidate(_tidy(f"{left}, and {lowered}"), "merge", "and")]
    if register != "academic":
        out.append(Candidate(_tidy(f"{left} — {lowered}"), "merge_dash", "em dash"))
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

SENTENCE_TRANSFORMS = (
    drop_opening_connective,
    substitute_markers,
    strip_stock_phrase,
    split_sentence,
    split_compound_predicate,
    toggle_contractions,
    break_tricolon,
)


def generate(
    text: str, *, register: str = "professional", seed: int = 0, limit: int = 12
) -> list[Candidate]:
    """All plausible single-step rewrites of one sentence."""
    rng = random.Random(seed)
    candidates: list[Candidate] = []

    for transform in SENTENCE_TRANSFORMS:
        try:
            candidates.extend(transform(text, register, rng))
        except Exception:  # noqa: BLE001 — one bad transform must not stop the rest
            continue

    seen: set[str] = {text.strip()}
    unique: list[Candidate] = []
    for candidate in candidates:
        key = candidate.text.strip()
        if key in seen or not is_plausible(candidate.text):
            continue
        seen.add(key)
        unique.append(candidate)

    return _diverse_subset(unique, limit)


def _diverse_subset(candidates: list[Candidate], limit: int) -> list[Candidate]:
    """Truncate while keeping one candidate from each transform class.

    A flat "sort by priority, take top N" loses diversity: lexical transforms
    outnumber structural ones, so they crowd out sentence splitting — and
    splitting drives sentence-length variation, the single heaviest structural
    signal. Round-robin across classes keeps every kind of move on the table,
    which matters because the loop's objective is document-level and the best
    move is often structural even when the sentence looks lexically bad.
    """
    buckets: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        buckets.setdefault(_PRIORITY.get(candidate.transform, 5), []).append(candidate)

    selected: list[Candidate] = []
    while len(selected) < limit and any(buckets.values()):
        for priority in sorted(buckets):
            if not buckets[priority]:
                continue
            selected.append(buckets[priority].pop(0))
            if len(selected) >= limit:
                break
    return selected


_PRIORITY: dict[str, int] = {
    "substitute_all": 0,
    "strip_all": 0,
    "strip_stock_phrase": 1,
    "substitute_phrase": 1,
    "drop_connective": 2,
    "split_predicate": 2,
    "split": 3,
    "split_keep_conj": 3,
    "denominalize": 4,
    "substitute_marker": 4,
    "break_tricolon": 5,
    "contract_all": 5,
    "contract": 6,
}


def generate_compound(
    text: str, *, register: str = "professional", seed: int = 0, depth: int = 2
) -> list[Candidate]:
    """Chain transforms so a sentence with several problems can be fixed in one
    accepted step — a marker word *and* a stock phrase *and* a connective."""
    results: list[Candidate] = []
    frontier = [Candidate(text, "identity")]

    for level in range(depth):
        next_frontier: list[Candidate] = []
        for parent in frontier:
            for child in generate(parent.text, register=register, seed=seed + level, limit=6):
                if parent.transform == "identity":
                    combined = child
                else:
                    combined = Candidate(
                        child.text,
                        f"{parent.transform}+{child.transform}",
                        f"{parent.note}; {child.note}",
                    )
                next_frontier.append(combined)
                results.append(combined)
        frontier = next_frontier[:6]

    seen: set[str] = {text.strip()}
    unique: list[Candidate] = []
    for candidate in results:
        if candidate.text.strip() in seen:
            continue
        seen.add(candidate.text.strip())
        unique.append(candidate)
    return unique
