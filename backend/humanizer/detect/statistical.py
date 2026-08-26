"""Tier A: model-based detection statistics.

Three signals, all derived from a single pass over the document:

  Fast-DetectGPT   Conditional-probability curvature. Machine text sits at a
                   local maximum of the model's own log-probability surface;
                   human text sits at an arbitrary point. Computed in the
                   analytic form — the expectation and variance over the
                   model's conditional distribution are exact, so no
                   perturbation sampling is required.

  Binoculars       Ratio of the scorer's log-perplexity to the cross-perplexity
                   between scorer and reference. Dividing by cross-perplexity
                   normalizes away "this text is simply written", which is why
                   it does not falsely flag non-native English the way raw
                   perplexity does.

  Burstiness       Spread of per-sentence mean log-probability. Humans spike;
                   machine surprise is uniformly low.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..document import Document
from .backend import TokenStats, get_models


@dataclass(slots=True)
class StatResult:
    fast_detect: float | None = None
    binoculars: float | None = None
    perplexity: float | None = None
    mean_log_rank: float | None = None
    sentence_logp_std: float | None = None
    sentence_logp_cv: float | None = None
    n_tokens: int = 0
    truncated: bool = False
    available: bool = False
    error: str | None = None

    def as_features(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in (
            "fast_detect",
            "binoculars",
            "perplexity",
            "mean_log_rank",
            "sentence_logp_std",
            "sentence_logp_cv",
        ):
            value = getattr(self, name)
            if value is not None and math.isfinite(value):
                out[name] = value
        return out


def fast_detect_score(stats: TokenStats) -> float | None:
    """Conditional-probability curvature. Higher means more machine-like."""
    if stats.n_tokens < 2 or stats.sum_var <= 0:
        return None
    return (stats.sum_logp - stats.sum_mu) / math.sqrt(stats.sum_var)


def binoculars_score(stats: TokenStats) -> float | None:
    """Perplexity / cross-perplexity. *Lower* means more machine-like."""
    if not stats.has_reference or stats.n_tokens < 2:
        return None
    log_ppl = -stats.mean_logp
    log_xppl = stats.sum_xent / stats.n_tokens
    if log_xppl <= 0:
        return None
    return log_ppl / log_xppl


def sentence_burstiness(doc: Document, stats: TokenStats) -> tuple[float, float] | None:
    """(std, cv) of per-sentence mean token log-probability.

    Tokens are assigned to sentences by character offset, so this stays correct
    regardless of how the tokenizer split the text.
    """
    sentences = doc.sentences
    if len(sentences) < 3 or not stats.token_logp:
        return None

    bounds = [(s.start, s.end) for s in sentences]
    totals = [0.0] * len(bounds)
    counts = [0] * len(bounds)

    cursor = 0
    for logp, (start, _end) in zip(stats.token_logp, stats.token_offsets, strict=True):
        # Offsets are monotonic, so advance rather than re-search each time.
        while cursor < len(bounds) and start >= bounds[cursor][1]:
            cursor += 1
        if cursor >= len(bounds):
            break
        if start >= bounds[cursor][0]:
            totals[cursor] += logp
            counts[cursor] += 1

    means = [t / c for t, c in zip(totals, counts, strict=True) if c >= 3]
    if len(means) < 3:
        return None

    mu = sum(means) / len(means)
    variance = sum((m - mu) ** 2 for m in means) / (len(means) - 1)
    std = math.sqrt(variance)
    return std, (std / abs(mu) if mu else 0.0)


def analyze(doc: Document) -> StatResult:
    """Run the full Tier A analysis. Degrades cleanly when models are absent."""
    models = get_models()
    if not models.available:
        return StatResult(available=False, error=models.load_error)

    stats = models.analyze(doc.text)
    if stats is None or stats.n_tokens < 2:
        return StatResult(available=False, error="Not enough tokens to score.")

    result = StatResult(
        fast_detect=fast_detect_score(stats),
        binoculars=binoculars_score(stats),
        perplexity=stats.perplexity,
        mean_log_rank=stats.mean_log_rank,
        n_tokens=stats.n_tokens,
        truncated=stats.truncated,
        available=True,
    )

    burst = sentence_burstiness(doc, stats)
    if burst is not None:
        result.sentence_logp_std, result.sentence_logp_cv = burst

    doc.meta["_token_stats"] = stats
    return result
