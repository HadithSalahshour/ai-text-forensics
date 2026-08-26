"""Rule-based scorer over the Tier B/C/D feature vector.

This is a standalone 0-100 score, useful on its own and cheap enough to run in
the humanizer's inner loop thousands of times. In Phase 2 it becomes one term
in the ensemble alongside the model-based statistical signals; in Phase 3 the
hand-set thresholds below get replaced by weights fit on real data.

Every threshold here is an explicit, documented prior. They are written as
(human_typical, ai_typical) pairs so that when the calibration data disagrees
we can see exactly which prior was wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..document import Document
from . import features as F
from . import lexicon


@dataclass(slots=True)
class Signal:
    """One piece of evidence, with enough context for the UI to explain it."""

    key: str
    label: str
    value: float
    score: float  # 0-1, higher = more machine-like
    weight: float
    tier: str
    evidence: str


@dataclass(slots=True)
class RuleResult:
    score: float  # 0-100
    signals: list[Signal] = field(default_factory=list)
    reliable: bool = True
    note: str = ""

    def top_signals(self, n: int = 8) -> list[Signal]:
        return sorted(
            self.signals, key=lambda s: s.score * s.weight, reverse=True
        )[:n]


def _ramp(value: float, low: float, high: float) -> float:
    """0 at `low`, 1 at `high`, linear between. Works in either direction —
    pass high < low for an inverted ramp."""
    if high == low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


# (key, label, tier, human_end, ai_end, weight)
#
# `human_end` is where a typical human document sits and scores 0.
# `ai_end` is where typical machine text sits and scores 1.
# When ai_end < human_end the ramp is inverted (lower values are the AI tell).
_SIGNAL_SPEC: list[tuple[str, str, str, float, float, float]] = [
    # --- Tier B: structure -------------------------------------------------
    ("sent_len_cv", "Sentence-length variation", "structure", 0.70, 0.28, 3.0),
    ("short_sentence_rate", "Short sentences (<8 words)", "structure", 0.14, 0.01, 2.0),
    ("long_sentence_rate", "Long sentences (>30 words)", "structure", 0.14, 0.02, 1.2),
    ("midband_sentence_rate", "Sentences clustered at 13-25 words", "structure", 0.32, 0.68, 1.8),
    ("para_sent_count_cv", "Paragraph-length variation", "structure", 0.50, 0.14, 1.2),
    ("opening_repeat_rate", "Repeated sentence openers", "structure", 0.10, 0.30, 1.0),
    ("opening_determiner_rate", "Sentences opening with The/This/It", "structure", 0.22, 0.50, 1.0),
    ("tricolon_rate", "Rule-of-three lists", "structure", 0.15, 1.20, 1.5),
    ("negation_parallel_rate", "\"Not just X, but Y\" constructions", "structure", 0.0, 0.45, 1.5),
    ("from_to_sweep_rate", "\"From X to Y\" sweeps", "structure", 0.05, 0.60, 0.8),
    # --- Tier C: lexicon ---------------------------------------------------
    ("gptism_weighted_rate", "LLM-marker vocabulary", "lexicon", 0.10, 2.50, 3.0),
    ("phrase_weighted_rate", "Formulaic stock phrases", "lexicon", 0.02, 0.70, 2.5),
    ("opening_connective_rate", "Moreover/Furthermore openers", "lexicon", 0.04, 0.28, 2.0),
    ("hedge_booster_ratio", "Hedging without commitment", "lexicon", 2.0, 9.0, 1.5),
    ("hapax_rate", "Once-only word rate", "lexicon", 0.62, 0.38, 1.0),
    ("common_word_rate", "Reliance on common vocabulary", "lexicon", 0.44, 0.60, 0.8),
    ("nominalization_rate", "Nominalized verbs", "lexicon", 2.0, 6.0, 1.0),
    ("first_person_rate", "First-person presence", "lexicon", 1.5, 0.05, 0.8),
    ("contraction_rate", "Contractions", "lexicon", 1.5, 0.05, 0.6),
    # --- Tier D: mechanics -------------------------------------------------
    ("number_rate", "Concrete numbers and dates", "mechanics", 1.30, 0.10, 1.5),
    ("intro_conclusion_overlap", "Conclusion restates the intro", "mechanics", 0.08, 0.32, 1.2),
    ("content_density_cv", "Information-density variation", "mechanics", 0.34, 0.14, 1.2),
    ("em_dash_rate", "Em-dash density", "mechanics", 0.8, 4.5, 0.8),
    ("capitalized_midsentence_rate", "Proper nouns mid-sentence", "mechanics", 1.00, 0.10, 1.0),
    # --- Syntax (requires spaCy) ------------------------------------------
    ("dep_depth_cv", "Syntactic-depth variation", "syntax", 0.60, 0.25, 1.5),
    ("clause_count_cv", "Clause-count variation", "syntax", 1.25, 0.80, 0.6),
    ("pos_trigram_entropy", "Grammatical-pattern diversity", "syntax", 7.60, 6.70, 1.2),
]

# Deliberately excluded, with reasons — so nobody re-adds them later:
#
#   passive_rate      Measured *backwards* on real samples. Current-generation
#                     models are tuned toward active voice, so the old
#                     "AI writes passively" heuristic is stale and harmful.
#   exclamation_rate  Reads 0.0 for both human and machine formal prose.
#                     Register noise, no discriminative power.


def _evidence(label: str, value: float, human: float, ai: float, score: float) -> str:
    direction = "below" if ai < human else "above"
    if score < 0.25:
        return f"{label}: {value:.2f} — in the normal human range (~{human:.2f})."
    if score < 0.6:
        return f"{label}: {value:.2f} — drifting {direction} the human norm of ~{human:.2f}."
    return (
        f"{label}: {value:.2f} — well {direction} the human norm (~{human:.2f}), "
        f"close to the machine-text pattern (~{ai:.2f})."
    )


def score_features(feats: dict[str, float], word_count: int) -> RuleResult:
    """Score a precomputed feature vector."""
    signals: list[Signal] = []
    weighted_total = 0.0
    weight_total = 0.0

    for key, label, tier, human_end, ai_end, weight in _SIGNAL_SPEC:
        if key not in feats:
            continue  # feature unavailable (e.g. spaCy not installed)
        value = feats[key]
        score = _ramp(value, human_end, ai_end)
        signals.append(
            Signal(
                key=key,
                label=label,
                value=value,
                score=score,
                weight=weight,
                tier=tier,
                evidence=_evidence(label, value, human_end, ai_end, score),
            )
        )
        weighted_total += score * weight
        weight_total += weight

    raw = weighted_total / weight_total if weight_total else 0.0

    # Mild sharpening: the linear mean is too forgiving of documents that trip
    # many signals at once, which is exactly the case we most want to catch.
    sharpened = raw**0.85
    result = RuleResult(score=round(sharpened * 100, 1), signals=signals)

    if word_count < 150:
        result.reliable = False
        result.note = (
            f"Only {word_count} words. Below ~150 words these statistics are too "
            "noisy for a confident verdict — treat this number as indicative only."
        )
    return result


def score_document(doc: Document) -> RuleResult:
    return score_features(F.extract(doc), doc.word_count)


# --------------------------------------------------------------------------
# Per-sentence scoring for the heatmap
# --------------------------------------------------------------------------


def score_sentences(doc: Document) -> list[tuple[int, float, list[str]]]:
    """Score each sentence 0-100 with human-readable reasons.

    Returns (sentence_index, score, reasons). Sentence-level statistics are
    noisy by nature, so this leans on local, directly observable evidence
    (a marker word is *present* or it isn't) rather than distributional
    features that need a whole document to mean anything.
    """
    sentences = doc.sentences
    if not sentences:
        return []

    doc_lengths = [s.word_count for s in sentences]
    doc_mean = F._mean([float(x) for x in doc_lengths]) or 1.0

    results: list[tuple[int, float, list[str]]] = []

    for sentence in sentences:
        text = sentence.text
        lowered = text.lower()
        words = F.WORD_RE.findall(lowered)
        reasons: list[str] = []
        score = 0.0

        # Marker vocabulary — the clearest local evidence.
        markers = [w for w in words if w in lexicon.GPTISM_WEIGHTS]
        if markers:
            top = sorted(markers, key=lambda w: -lexicon.GPTISM_WEIGHTS[w])[:3]
            marker_load = sum(lexicon.GPTISM_WEIGHTS[w] for w in markers)
            score += min(marker_load * 0.30, 0.45)
            reasons.append("LLM-marker words: " + ", ".join(f"“{w}”" for w in top))

        # Stock phrases.
        hits = lexicon.phrase_hits(text)
        if hits:
            score += min(sum(w for _, w, _, _ in hits) * 0.35, 0.40)
            reasons.append("Stock phrasing: " + ", ".join(f"“{h[0]}”" for h in hits[:2]))

        # Mechanical opener.
        head = lowered.lstrip()
        for connective, weight in lexicon.OPENING_CONNECTIVES.items():
            if head.startswith(connective):
                score += weight * 0.20
                reasons.append(f"Opens with “{connective.capitalize()}”")
                break

        if lexicon.NEGATION_PARALLELISM.search(text):
            score += 0.22
            reasons.append("“Not just X, but Y” construction")

        if lexicon.TRICOLON.search(text):
            score += 0.12
            reasons.append("Rule-of-three list")

        # Length conformity: sitting inside the safe midband, close to the
        # document mean, is itself weak evidence of machine rhythm.
        wc = sentence.word_count
        if 13 <= wc <= 25 and abs(wc - doc_mean) < 0.25 * doc_mean:
            score += 0.12
            reasons.append("Length sits in the machine-typical 13-25 word band")
        elif wc < 8 or wc > 32:
            score -= 0.15
            reasons.append("Unusual length — reads as human rhythm")

        # A concrete number or citation is a genuine human signal.
        if F.NUMBER_RE.search(text) or lexicon.CITATION.search(text):
            score -= 0.12
            reasons.append("Contains a concrete number or citation")

        final = max(0.0, min(1.0, score))
        results.append((sentence.index, round(final * 100, 1), reasons))

    return results
