"""Fusion of all four tiers into one calibrated 0-100 score.

Tiers are normalized independently and then blended, rather than thrown into
one flat weighted mean. Two reasons:

  1. It keeps the stated tier balance (Tier A carries the most weight) stable
     no matter how many individual features each tier happens to contain.
  2. When a tier is unavailable — no local models installed, or spaCy missing —
     its weight redistributes across the others instead of silently dragging
     the score toward zero.

Why both tiers are needed, concretely: Tier A measures *sampling signature* and
catches genuine model output that has been styled to look human. Tiers B-D
measure *style* and catch human-edited or paraphrased machine text whose
sampling signature has been scrubbed. Text that defeats one routinely trips the
other.

THRESHOLD PROVENANCE: the Tier A endpoints below come from a single measured
pair (one genuine Qwen2.5 generation, one human sample) plus published
reference values. They are provisional. Phase 3 refits them on a real corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..document import Document
from . import features as F
from . import statistical
from .rulescore import _SIGNAL_SPEC, Signal, _evidence, _ramp

log = logging.getLogger(__name__)

# Relative weight of each tier in the blended score.
TIER_WEIGHTS: dict[str, float] = {
    "statistical": 0.55,
    "lexicon": 0.17,
    "structure": 0.15,
    "mechanics": 0.07,
    "syntax": 0.06,
}

# (key, label, human_end, machine_end, weight-within-tier)
_STAT_SIGNAL_SPEC: list[tuple[str, str, float, float, float]] = [
    # Strongest zero-shot signal in the literature and in our own measurement.
    ("fast_detect", "Probability-curvature (Fast-DetectGPT)", 0.50, 2.20, 4.0),
    # Robust to plain/simple writing, which is what keeps non-native English
    # from being falsely flagged. Inverted: lower means more machine-like.
    ("binoculars", "Cross-perplexity ratio (Binoculars)", 0.970, 0.850, 3.5),
    ("mean_log_rank", "Token predictability (log-rank)", 1.60, 0.60, 2.0),
    # Register-sensitive and the weakest cross-domain generalizer of the four,
    # so it is deliberately not the lead signal.
    ("perplexity", "Perplexity", 25.0, 6.0, 2.0),
    ("sentence_logp_std", "Per-sentence surprise variation", 0.78, 0.55, 1.5),
]


# How much of the final score the fitted model carries when one exists and its
# training run met the trustworthiness bar. The hand-tuned tier blend keeps the
# remainder: it is interpretable, it degrades predictably on out-of-domain
# text, and it stops a model overfit to one corpus from dominating outright.
FITTED_WEIGHT = 0.60


@dataclass(slots=True)
class Detection:
    score: float
    tier_scores: dict[str, float] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    reliable: bool = True
    note: str = ""
    statistical_available: bool = False
    statistical_error: str | None = None
    truncated: bool = False
    word_count: int = 0
    fitted_used: bool = False
    fitted_score: float | None = None
    heuristic_score: float | None = None

    @property
    def verdict(self) -> str:
        if self.score < 20:
            return "Reads as human"
        if self.score < 40:
            return "Mostly human"
        if self.score < 60:
            return "Mixed signals"
        if self.score < 80:
            return "Likely AI-generated"
        return "Very likely AI-generated"

    def top_signals(self, n: int = 10) -> list[Signal]:
        return sorted(self.signals, key=lambda s: s.score * s.weight, reverse=True)[:n]

    def signals_by_tier(self, tier: str) -> list[Signal]:
        return sorted(
            (s for s in self.signals if s.tier == tier),
            key=lambda s: s.score * s.weight,
            reverse=True,
        )


def _tier_score(signals: list[Signal]) -> float | None:
    """Weighted mean of a tier's signals, or None if the tier is empty."""
    total_weight = sum(s.weight for s in signals)
    if total_weight == 0:
        return None
    return sum(s.score * s.weight for s in signals) / total_weight


def analyze(
    doc: Document, *, use_models: bool = True, use_fitted: bool = True
) -> Detection:
    """Score a document across every available tier."""
    signals: list[Signal] = []

    # --- Tiers B/C/D: cheap, always available -----------------------------
    feats = F.extract(doc)
    for key, label, tier, human_end, ai_end, weight in _SIGNAL_SPEC:
        if key not in feats:
            continue
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

    # --- Tier A: local language models ------------------------------------
    stat_available = False
    stat_error = None
    truncated = False

    if use_models:
        stat = statistical.analyze(doc)
        stat_available = stat.available
        stat_error = stat.error
        truncated = stat.truncated
        if stat.available:
            stat_feats = stat.as_features()
            feats.update(stat_feats)
            for key, label, human_end, ai_end, weight in _STAT_SIGNAL_SPEC:
                if key not in stat_feats:
                    continue
                value = stat_feats[key]
                score = _ramp(value, human_end, ai_end)
                signals.append(
                    Signal(
                        key=key,
                        label=label,
                        value=value,
                        score=score,
                        weight=weight,
                        tier="statistical",
                        evidence=_evidence(label, value, human_end, ai_end, score),
                    )
                )

    # --- Blend, redistributing the weight of any missing tier -------------
    tier_scores: dict[str, float] = {}
    blended = 0.0
    live_weight = 0.0

    for tier, tier_weight in TIER_WEIGHTS.items():
        tier_signals = [s for s in signals if s.tier == tier]
        score = _tier_score(tier_signals)
        if score is None:
            continue
        tier_scores[tier] = round(score * 100, 1)
        blended += score * tier_weight
        live_weight += tier_weight

    raw = blended / live_weight if live_weight else 0.0
    # Mild sharpening — a plain mean is too forgiving of documents that trip
    # many independent signals at once, which is the case we most want caught.
    heuristic = round((raw**0.85) * 100, 1)
    final = heuristic

    # --- Fitted model, when one exists and is worth trusting --------------
    fitted_used = False
    fitted_score: float | None = None

    if use_fitted:
        from ..calibration import fitted as fitted_module

        model = fitted_module.load()
        if model is not None and model.trustworthy:
            try:
                fitted_score = round(model.predict(feats) * 100, 1)
                final = round(
                    FITTED_WEIGHT * fitted_score + (1 - FITTED_WEIGHT) * heuristic, 1
                )
                fitted_used = True
            except Exception:  # noqa: BLE001 — never let scoring die on this
                log.warning("Fitted detector failed; using heuristic score.", exc_info=True)

    result = Detection(
        score=final,
        tier_scores=tier_scores,
        signals=signals,
        statistical_available=stat_available,
        statistical_error=stat_error,
        truncated=truncated,
        word_count=doc.word_count,
        fitted_used=fitted_used,
        fitted_score=fitted_score,
        heuristic_score=heuristic,
    )

    notes: list[str] = []
    if doc.word_count < 150:
        result.reliable = False
        notes.append(
            f"Only {doc.word_count} words. Below ~150 words these statistics are too "
            "noisy for a confident verdict — treat this as indicative only."
        )
    if not stat_available:
        # The caveat applies whether the tier was switched off or failed to
        # load: without sampling-signature evidence, machine text that has been
        # lightly edited for style will slip through.
        reason = "disabled" if not use_models else "unavailable"
        notes.append(
            f"Statistical tier {reason} — score is based on style signals alone, "
            "which will under-detect machine text that has been lightly edited."
        )
    if truncated:
        notes.append("Document was truncated for statistical scoring.")
    result.note = " ".join(notes)

    return result
