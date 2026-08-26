"""Claude candidate generator — the only component that uses the network.

Off unless explicitly enabled. When enabled it produces *candidates*, nothing
more: every rewrite it returns goes through the same meaning and fact guards as
a rule-based one, and is accepted only if it lowers the real score. The model
has no privileged path.

Two design choices worth stating:

  Paragraph-level, not sentence-level. Rewriting sentence by sentence produces
  disjointed prose and multiplies the call count. A paragraph is the smallest
  unit where varying rhythm actually means something.

  N variants per call, not N calls. Sonnet 5 and Opus 5 reject non-default
  `temperature`, so the usual way of getting variety is unavailable. Asking for
  several deliberately different rewrites in one structured response is both
  cheaper and more varied than repeated sampling would have been — the model
  can see its own earlier variants and make the next one different.

The prompt carries quantitative targets rather than adjectives. "Write more
naturally" produces the LLM mean; "at least one sentence under 8 words, at
least one over 25, do not use these 6 words" produces something measurably
different.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..config import settings
from ..detect import lexicon
from .guards import extract_citations, extract_entities
from .transforms import Candidate, is_plausible

log = logging.getLogger(__name__)

# USD per million tokens (input, output). Sonnet 5 carries introductory
# pricing through 2026-08-31; the higher standard rate is used for estimates
# so a quote is never optimistic.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),
}


# Thinking tokens count against this too, so it needs headroom well beyond the
# length of the rewrites themselves.
MAX_OUTPUT_TOKENS = 12000


class RewriteVariants(BaseModel):
    """Structured response schema — avoids parsing prose for the rewrites."""

    variants: list[str] = Field(
        description="Distinct rewrites of the passage, each preserving its meaning exactly."
    )


@dataclass(slots=True)
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    @property
    def cost_usd(self) -> float:
        rate_in, rate_out = PRICING.get(self.model, (3.0, 15.0))
        return (self.input_tokens * rate_in + self.output_tokens * rate_out) / 1_000_000

    def summary(self) -> str:
        return (
            f"{self.calls} calls · {self.input_tokens:,} in / {self.output_tokens:,} out "
            f"· ${self.cost_usd:.4f} ({self.model})"
        )


@dataclass(slots=True)
class LLMStatus:
    available: bool
    reason: str = ""
    model: str = ""


SYSTEM_PROMPT = """You rewrite passages so they read as though a person wrote them, \
without changing what they say.

Absolute constraints — a rewrite that breaks any of these is useless:
- Never introduce a fact, number, date, name, statistic, or citation that is not \
already in the passage. Not one.
- Never drop a number, date, name, or citation that IS in the passage.
- Never change what the passage claims. Preserve hedging and strength of claim: \
if the original says "may reduce", do not write "reduces".
- Output prose only. No headings, no bullet points, no commentary about your edits.

What actually makes writing read as human, in priority order:
1. Rhythm. Real writers vary sentence length hard — a four-word sentence next to a \
thirty-five-word one. Uniform mid-length sentences are the single loudest tell.
2. Plain verbs over nominalizations: "we decided" beats "the decision was made".
3. Committing to a claim sometimes, instead of hedging every one.
4. Concrete phrasing over abstract summary — but only using specifics already present.

Do not simply swap flagged words for their most obvious synonyms. That is the \
mistake most rewriters make: the plainest synonym is the most statistically \
predictable one, which makes the text easier to detect, not harder. Restructure \
the sentence instead."""


def _build_prompt(
    text: str,
    *,
    register: str,
    n_variants: int,
    banned: list[str],
    must_keep: list[str],
) -> str:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    lengths = [len(s.split()) for s in sentences]

    parts = [
        f"Rewrite the passage below. Produce {n_variants} distinct rewrites — "
        "genuinely different from each other in structure, not the same rewrite "
        "with different synonyms.",
        f"\nRegister: {register}.",
    ]

    if lengths:
        parts.append(
            f"\nThe passage has {len(lengths)} sentences of "
            f"{min(lengths)}-{max(lengths)} words (mean {sum(lengths) / len(lengths):.0f}). "
            "Each rewrite must include at least one sentence of 8 words or fewer, "
            "and at least one of 25 words or more."
        )

    if banned:
        parts.append(
            "\nDo not use any of these words or phrases: " + ", ".join(sorted(banned)) + "."
        )

    if must_keep:
        parts.append(
            "\nThese must appear in every rewrite, unchanged: "
            + ", ".join(f'"{item}"' for item in sorted(must_keep))
            + "."
        )

    parts.append(f"\n---\n{text}\n---")
    return "\n".join(parts)


def _constraints_for(text: str) -> tuple[list[str], list[str]]:
    """Banned marker vocabulary present in this span, and the specifics that
    must survive it."""
    lowered = text.lower()
    banned = [
        word
        for word, weight in lexicon.GPTISM_WEIGHTS.items()
        if weight >= 0.45 and re.search(rf"\b{re.escape(word)}\b", lowered)
    ]
    banned.extend(
        match for match, _weight, _s, _e in lexicon.phrase_hits(text)
    )

    must_keep: list[str] = []
    must_keep.extend(sorted(extract_citations(text)))
    # Surface numbers exactly as written so the model echoes the original form.
    must_keep.extend(sorted({m.group(0) for m in re.finditer(r"\d[\d.,]*%?", text)}))
    must_keep.extend(sorted(extract_entities(text))[:8])

    return banned[:20], must_keep[:12]


class ClaudeRewriter:
    """Candidate generator backed by the Anthropic API."""

    def __init__(self, model: str | None = None, effort: str | None = None) -> None:
        self.model = model or settings.bulk_model
        self.effort = effort or settings.bulk_effort
        self.usage = Usage(model=self.model)
        self._client = None
        self._status: LLMStatus | None = None

    # -- availability ------------------------------------------------------

    def status(self) -> LLMStatus:
        if self._status is not None:
            return self._status

        if not settings.llm_enabled:
            self._status = LLMStatus(
                False,
                "Claude rewriting is off. Enable with HUMANIZER_LLM_ENABLED=true "
                "(this is the only feature that sends text off your machine).",
            )
            return self._status

        try:
            import anthropic
        except ImportError:
            self._status = LLMStatus(
                False, "anthropic SDK not installed. Run: uv pip install -e '.[llm]'"
            )
            return self._status

        try:
            # A bare client resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or
            # an `ant auth login` profile, in that order.
            self._client = (
                anthropic.Anthropic(api_key=settings.anthropic_api_key)
                if settings.anthropic_api_key
                else anthropic.Anthropic()
            )
        except Exception as exc:  # noqa: BLE001
            self._status = LLMStatus(False, f"No usable Anthropic credentials: {exc}")
            return self._status

        self._status = LLMStatus(True, model=self.model)
        return self._status

    @property
    def available(self) -> bool:
        return self.status().available

    # -- generation --------------------------------------------------------

    def rewrite(
        self, text: str, *, register: str = "professional", n_variants: int = 3
    ) -> list[Candidate]:
        """Return candidate rewrites of one passage. Never raises."""
        if not self.available:
            return []

        banned, must_keep = _constraints_for(text)
        prompt = _build_prompt(
            text,
            register=register,
            n_variants=n_variants,
            banned=banned,
            must_keep=must_keep,
        )

        try:
            response = self._client.messages.parse(
                model=self.model,
                # Generous on purpose. Adaptive thinking is on by default for
                # Sonnet 5 and Opus 5, and `max_tokens` caps thinking *plus*
                # response text. At 4096 this truncated mid-JSON on longer
                # paragraphs, which surfaced as an opaque pydantic parse error
                # and silently wasted the call.
                max_tokens=MAX_OUTPUT_TOKENS,
                system=SYSTEM_PROMPT,
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": prompt}],
                output_format=RewriteVariants,
            )
        except Exception as exc:  # noqa: BLE001 — a failed call must not stop the run
            # The call may still have been billed, so it is counted here even
            # though we got nothing usable back.
            self.usage.calls += 1
            log.warning(
                "Claude rewrite call failed (%s): %s",
                type(exc).__name__,
                str(exc).split("\n")[0][:200],
            )
            return []

        self.usage.calls += 1
        self.usage.input_tokens += response.usage.input_tokens
        self.usage.output_tokens += response.usage.output_tokens

        if response.stop_reason == "refusal":
            log.warning("Claude declined to rewrite this passage.")
            return []
        if response.stop_reason == "max_tokens":
            log.warning(
                "Claude response hit the %d-token cap and was truncated; "
                "discarding this batch.",
                MAX_OUTPUT_TOKENS,
            )
            return []

        parsed = response.parsed_output
        if parsed is None:
            return []

        candidates: list[Candidate] = []
        for index, variant in enumerate(parsed.variants):
            cleaned = variant.strip()
            if not cleaned or cleaned == text.strip():
                continue
            # Single-sentence spans still have to look like sentences; a
            # multi-sentence paragraph is checked by the guards instead.
            if "\n" not in cleaned and not is_plausible(cleaned):
                continue
            candidates.append(Candidate(cleaned, "claude", f"variant {index + 1}"))

        return candidates

    # -- cost control ------------------------------------------------------

    def estimate_cost(self, text: str, *, calls: int) -> float:
        """Pre-flight quote.

        `calls` must be the real upper bound — paragraphs_per_iteration ×
        max_iterations — not a nominal "few passes". The first version of this
        assumed 3 calls while `aggressive` actually issues up to 20, which
        understated a real run by roughly 5x. A cost quote that is wrong in the
        cheap direction is worse than no quote at all.
        """
        # Constants fitted to a measured run: a 370-word, 6-paragraph document
        # at 9 calls used 8,834 input and 8,478 output tokens ($0.154).
        #
        # THINKING_OVERHEAD is the load-bearing term. Adaptive thinking is on by
        # default for Sonnet 5 and Opus 5, and those tokens bill as output — for
        # a short paragraph they dominate the visible rewrites several times
        # over. An earlier version counted only the rewrites and quoted a third
        # of the true cost.
        PROMPT_OVERHEAD = 900  # system prompt + banned/must-keep lists, per call
        THINKING_OVERHEAD = 650  # adaptive-thinking output, per call
        VARIANT_MULTIPLIER = 3.5  # n_variants rewrites of the passage

        paragraphs = max(text.count("\n\n") + 1, 1)
        paragraph_tokens = max(len(text.split()) / paragraphs, 60) * 1.35

        input_tokens = (paragraph_tokens + PROMPT_OVERHEAD) * calls
        output_tokens = (paragraph_tokens * VARIANT_MULTIPLIER + THINKING_OVERHEAD) * calls

        rate_in, rate_out = PRICING.get(self.model, (3.0, 15.0))
        return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000

    def over_budget(self, limit_usd: float | None) -> bool:
        """Whether this run has already spent its allowance."""
        return limit_usd is not None and self.usage.cost_usd >= limit_usd


@dataclass(slots=True)
class RewriterPool:
    """Holds the bulk and escalation rewriters for one run."""

    bulk: ClaudeRewriter = field(default_factory=ClaudeRewriter)
    quality: ClaudeRewriter | None = None

    @classmethod
    def build(cls, *, high_quality: bool = False) -> RewriterPool:
        if high_quality:
            model = settings.quality_model
            return cls(bulk=ClaudeRewriter(model=model), quality=None)
        return cls(bulk=ClaudeRewriter())

    @property
    def usage(self) -> Usage:
        return self.bulk.usage
