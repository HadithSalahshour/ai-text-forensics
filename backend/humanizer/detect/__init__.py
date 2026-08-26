"""Detection: feature extraction, statistical analysis, and scoring.

Four tiers, fused in `ensemble`:

  A  statistical  perplexity, Fast-DetectGPT curvature, Binoculars, burstiness
  B  structure    sentence/paragraph shape, syntactic variation
  C  lexicon      LLM markers, stock phrases, discourse connectives
  D  mechanics    punctuation, specificity, redundancy

Tiers B-D need nothing but spaCy. Tier A loads two local language models and
still makes no network call after the first download.
"""

from .ensemble import Detection, analyze
from .rulescore import RuleResult, Signal, score_document, score_sentences
from .statistical import StatResult

__all__ = [
    "Detection",
    "RuleResult",
    "Signal",
    "StatResult",
    "analyze",
    "score_document",
    "score_sentences",
]
