"""Tier B/C/D feature extraction.

Every feature is a named float so the UI can attribute a score to specific
evidence and the Phase 3 gradient-boosting model can learn its own weights.

Naming convention: features ending in `_rate` are normalized per-unit (per
sentence, per 100 words, or per 1000 characters — stated in the docstring of
each group). Features ending in `_cv` are coefficients of variation, which is
where most of the burstiness signal lives.

Syntactic features require spaCy and are omitted from the dict when it isn't
installed. Consumers must tolerate missing keys.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..document import Document
from ..nlp import get_nlp
from . import lexicon

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
CONTRACTION_RE = re.compile(r"\b\w+['’](?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
NOMINALIZATION_RE = re.compile(
    r"\b\w{4,}(?:tion|sion|ment|ness|ity|ance|ence|ing\s+of)\b", re.IGNORECASE
)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"})

# Top ~250 English word forms. Used for a coarse vocabulary-band measure:
# machine text draws from a narrower band — fewer rare words *and* fewer
# colloquialisms — so both tails matter.
COMMON_WORDS = frozenset(
    """the be to of and a in that have i it for not on with he as you do at this
but his by from they we say her she or an will my one all would there their what
so up out if about who get which go me when make can like time no just him know
take people into year your good some could them see other than then now look only
come its over think also back after use two how our work first well way even new
want because any these give day most us is are was were been has had did does
said made find where much before right through back little only man came should
between own under never same another while last might great old year off might
came every own may part place made live must such long own does own why
""".split()
)


# --------------------------------------------------------------------------
# Small statistics helpers (avoiding a scipy dependency)
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _cv(values: list[float]) -> float:
    """Coefficient of variation. The single best burstiness proxy we have
    without a language model: human sentence lengths scatter, machine ones
    cluster."""
    mu = _mean(values)
    return _std(values) / mu if mu > 0 else 0.0


def _skew(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    mu, sd = _mean(values), _std(values)
    if sd == 0:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * sum(((v - mu) / sd) ** 3 for v in values)


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def _mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity.

    Plain type-token ratio falls as text gets longer, which makes it useless
    for comparing documents of different lengths. MTLD counts how many tokens
    it takes for TTR to decay past a threshold, then averages forward and
    backward passes. Higher = more varied vocabulary.
    """

    def one_pass(seq: list[str]) -> float:
        factors, types, count = 0.0, set(), 0
        for token in seq:
            types.add(token)
            count += 1
            if len(types) / count <= threshold:
                factors += 1
                types, count = set(), 0
        if count > 0:
            ttr = len(types) / count
            factors += (1 - ttr) / (1 - threshold) if threshold < 1 else 0
        return len(seq) / factors if factors > 0 else float(len(seq))

    if len(tokens) < 20:
        return 0.0
    return (one_pass(tokens) + one_pass(tokens[::-1])) / 2


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------
# Tier B — structural
# --------------------------------------------------------------------------


def structural_features(doc: Document) -> dict[str, float]:
    """Sentence and paragraph shape. Normalized per sentence unless noted."""
    sentences = doc.sentences
    if not sentences:
        return {}

    lengths = [float(s.word_count) for s in sentences]
    n = len(lengths)

    feats: dict[str, float] = {
        "sent_len_mean": _mean(lengths),
        "sent_len_std": _std(lengths),
        "sent_len_cv": _cv(lengths),
        "sent_len_skew": _skew(lengths),
        "sent_len_min": min(lengths),
        "sent_len_max": max(lengths),
        # Humans reliably produce both very short and very long sentences.
        # Machine text produces neither.
        "short_sentence_rate": sum(1 for x in lengths if x < 8) / n,
        "long_sentence_rate": sum(1 for x in lengths if x > 30) / n,
        "midband_sentence_rate": sum(1 for x in lengths if 13 <= x <= 25) / n,
    }

    paragraphs = [p for p in doc.paragraphs if p.sentences]
    if paragraphs:
        para_words = [float(p.word_count) for p in paragraphs]
        para_sents = [float(len(p.sentences)) for p in paragraphs]
        feats["para_len_mean"] = _mean(para_words)
        feats["para_len_cv"] = _cv(para_words)
        feats["para_sent_count_mean"] = _mean(para_sents)
        feats["para_sent_count_cv"] = _cv(para_sents)

    # Opening-word diversity. AI cycles a small set of sentence openers.
    openers = Counter()
    for sentence in sentences:
        words = WORD_RE.findall(sentence.text)
        if words:
            openers[words[0].lower()] += 1
    if openers:
        feats["opening_repeat_rate"] = openers.most_common(1)[0][1] / n
        feats["opening_entropy"] = _entropy(openers)
        feats["opening_determiner_rate"] = (
            sum(openers[w] for w in ("the", "this", "these", "it", "in", "as")) / n
        )

    # Structural tics
    words_100 = max(doc.word_count, 1) / 100
    feats["tricolon_rate"] = len(lexicon.TRICOLON.findall(doc.text)) / words_100
    feats["negation_parallel_rate"] = (
        len(lexicon.NEGATION_PARALLELISM.findall(doc.text)) / words_100
    )
    feats["from_to_sweep_rate"] = len(lexicon.FROM_TO_SWEEP.findall(doc.text)) / words_100

    return feats


def syntactic_features(doc: Document) -> dict[str, float]:
    """Dependency-tree and POS features. Empty dict when spaCy is unavailable."""
    spacy_doc = doc.meta.get("_spacy")
    if spacy_doc is None:
        nlp = get_nlp()
        if nlp is None:
            return {}
        spacy_doc = nlp(doc.text)
        doc.meta["_spacy"] = spacy_doc

    def depth(token) -> int:
        # Compare by index, not identity: spaCy builds a fresh Token wrapper on
        # every `.head` access, so `token.head is token` is never true even at
        # the root.
        d, guard = 0, 0
        while token.head.i != token.i and guard < 100:
            token = token.head
            d += 1
            guard += 1
        return d

    depths: list[float] = []
    clause_counts: list[float] = []
    pos_sequence: list[str] = []
    passive = 0
    total_tokens = 0

    for sent in spacy_doc.sents:
        tokens = [t for t in sent if not t.is_space]
        if not tokens:
            continue
        depths.append(float(max(depth(t) for t in tokens)))
        clause_counts.append(
            float(sum(1 for t in tokens if t.dep_ in {"advcl", "ccomp", "xcomp", "relcl", "acl"}))
        )
        for token in tokens:
            pos_sequence.append(token.pos_)
            if token.dep_ in {"nsubjpass", "auxpass"}:
                passive += 1
            total_tokens += 1

    if not depths:
        return {}

    trigrams = Counter(
        tuple(pos_sequence[i : i + 3]) for i in range(len(pos_sequence) - 2)
    )

    feats = {
        "dep_depth_mean": _mean(depths),
        "dep_depth_cv": _cv(depths),
        "clause_count_mean": _mean(clause_counts),
        "clause_count_cv": _cv(clause_counts),
        "pos_trigram_entropy": _entropy(trigrams),
        "passive_rate": passive / max(total_tokens, 1),
    }

    ents = [e for e in spacy_doc.ents if e.label_ not in {"CARDINAL", "ORDINAL"}]
    feats["named_entity_rate"] = len(ents) / max(doc.word_count, 1) * 100
    return feats


# --------------------------------------------------------------------------
# Tier C — lexical
# --------------------------------------------------------------------------


def lexical_features(doc: Document) -> dict[str, float]:
    """Vocabulary and discourse markers. Rates are per 100 words unless noted."""
    text = doc.text
    words = [w.lower() for w in WORD_RE.findall(text)]
    if not words:
        return {}

    n_words = len(words)
    per_100 = n_words / 100
    counts = Counter(words)

    # GPT-ism load, weighted by how discriminative each marker is.
    gptism_weight = sum(
        counts[word] * weight
        for word, weight in lexicon.GPTISM_WEIGHTS.items()
        if word in counts
    )
    gptism_unique = sum(1 for word in lexicon.GPTISM_WEIGHTS if word in counts)

    phrase_hits = lexicon.phrase_hits(text)
    phrase_weight = sum(w for _, w, _, _ in phrase_hits)

    # Sentence-initial connectives, per sentence.
    sentences = doc.sentences
    opening_weight = 0.0
    for sentence in sentences:
        head = sentence.text.lstrip().lower()
        for connective, weight in lexicon.OPENING_CONNECTIVES.items():
            if head.startswith(connective) and head[len(connective) : len(connective) + 1] in {
                "",
                ",",
                " ",
            }:
                opening_weight += weight
                break

    hedges = sum(counts[w] for w in lexicon.HEDGES if w in counts)
    boosters = sum(counts[w] for w in lexicon.BOOSTERS if w in counts)

    hapax = sum(1 for c in counts.values() if c == 1)
    common = sum(counts[w] for w in counts if w in COMMON_WORDS)

    feats = {
        "gptism_weighted_rate": gptism_weight / per_100,
        "gptism_unique_count": float(gptism_unique),
        "phrase_weighted_rate": phrase_weight / per_100,
        "phrase_hit_count": float(len(phrase_hits)),
        "opening_connective_rate": opening_weight / max(len(sentences), 1),
        "hedge_rate": hedges / per_100,
        "booster_rate": boosters / per_100,
        # AI hedges uniformly and commits to nothing. A high ratio with a low
        # booster rate is the shape we're looking for.
        "hedge_booster_ratio": hedges / max(boosters, 1),
        "type_token_ratio": len(counts) / n_words,
        "mtld": _mtld(words),
        "hapax_rate": hapax / len(counts),
        "common_word_rate": common / n_words,
        "mean_word_length": _mean([float(len(w)) for w in words]),
        "contraction_rate": len(CONTRACTION_RE.findall(text)) / per_100,
        "nominalization_rate": len(NOMINALIZATION_RE.findall(text)) / per_100,
        "of_genitive_rate": counts.get("of", 0) / per_100,
        "first_person_rate": sum(counts[w] for w in FIRST_PERSON if w in counts) / per_100,
    }
    return feats


# --------------------------------------------------------------------------
# Tier D — mechanical
# --------------------------------------------------------------------------


def mechanical_features(doc: Document) -> dict[str, float]:
    """Punctuation, specificity, redundancy. Rates per 1000 characters unless
    noted."""
    text = doc.text
    per_1k = max(len(text), 1) / 1000
    per_100_words = max(doc.word_count, 1) / 100
    sentences = doc.sentences

    straight_quotes = text.count('"') + text.count("'")
    curly_quotes = sum(text.count(c) for c in "“”‘’")

    feats = {
        "em_dash_rate": (text.count("—") + text.count("–")) / per_1k,
        "semicolon_rate": text.count(";") / per_1k,
        "colon_rate": text.count(":") / per_1k,
        "exclamation_rate": text.count("!") / per_1k,
        "question_rate": text.count("?") / per_1k,
        "parenthetical_rate": text.count("(") / per_1k,
        "ellipsis_rate": (text.count("...") + text.count("…")) / per_1k,
        "curly_quote_ratio": curly_quotes / max(curly_quotes + straight_quotes, 1),
        "double_space_rate": len(re.findall(r"(?<=[.!?]) {2}", text)) / per_1k,
        "comma_per_sentence": text.count(",") / max(len(sentences), 1),
        # Specificity: machine text writes *around* facts. Concrete numbers,
        # dates, and citations are among the strongest human signals we have
        # that need no language model.
        "number_rate": len(NUMBER_RE.findall(text)) / per_100_words,
        "citation_rate": len(lexicon.CITATION.findall(text)) / per_100_words,
        "capitalized_midsentence_rate": len(
            re.findall(r"(?<=[a-z] )[A-Z][a-z]{2,}", text)
        )
        / per_100_words,
    }

    # Intro/conclusion redundancy — AI restates its opening in its closing.
    # Phase 2 swaps this Jaccard proxy for embedding cosine.
    paragraphs = [p for p in doc.paragraphs if p.word_count >= 15]
    if len(paragraphs) >= 2:
        head = {w.lower() for w in WORD_RE.findall(paragraphs[0].text)} - COMMON_WORDS
        tail = {w.lower() for w in WORD_RE.findall(paragraphs[-1].text)} - COMMON_WORDS
        feats["intro_conclusion_overlap"] = _jaccard(head, tail)

    # Information density variance: content words per sentence, scattered in
    # human writing, flat in machine writing.
    if sentences:
        density = []
        for sentence in sentences:
            tokens = [w.lower() for w in WORD_RE.findall(sentence.text)]
            if tokens:
                density.append(sum(1 for t in tokens if t not in COMMON_WORDS) / len(tokens))
        if density:
            feats["content_density_mean"] = _mean(density)
            feats["content_density_cv"] = _cv(density)

    return feats


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def extract(doc: Document, *, fast: bool = False) -> dict[str, float]:
    """Full Tier B/C/D feature vector for a document.

    `fast=True` skips the syntactic block, which is the only part that needs a
    spaCy parse and therefore dominates the cost. Used by the humanizer's inner
    loop, where hundreds of candidate documents are scored per run and relative
    ranking matters more than completeness.
    """
    feats: dict[str, float] = {}
    feats.update(structural_features(doc))
    feats.update(lexical_features(doc))
    feats.update(mechanical_features(doc))
    if not fast:
        feats.update(syntactic_features(doc))
    feats["word_count"] = float(doc.word_count)
    feats["sentence_count"] = float(len(doc.sentences))
    feats["paragraph_count"] = float(len(doc.paragraphs))
    return feats
