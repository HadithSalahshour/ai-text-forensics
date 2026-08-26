"""Claude engine tests.

No API key required. A stub client stands in for the SDK so the whole path —
prompt construction, parsing, guard enforcement, usage accounting, graceful
degradation — is exercised deterministically and for free.

The property these tests exist to protect: **the LLM has no privileged path.**
Its output faces the same guards as a regex transform, and a rewrite that
invents a fact is discarded no matter how good it reads.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from humanizer import ingest, segment
from humanizer.config import settings
from humanizer.humanize import llm, loop

FIXTURES = Path(__file__).parent / "fixtures"


class StubResponse:
    def __init__(self, variants, *, stop_reason="end_turn"):
        self.parsed_output = llm.RewriteVariants(variants=variants) if variants else None
        self.stop_reason = stop_reason
        self.usage = SimpleNamespace(input_tokens=800, output_tokens=400)


class StubMessages:
    def __init__(self, variants, stop_reason="end_turn", error=None):
        self._variants = variants
        self._stop_reason = stop_reason
        self._error = error
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return StubResponse(self._variants, stop_reason=self._stop_reason)


def _rewriter(variants, *, stop_reason="end_turn", error=None, model="claude-sonnet-5"):
    rewriter = llm.ClaudeRewriter(model=model)
    rewriter._status = llm.LLMStatus(True, model=model)
    stub = StubMessages(variants, stop_reason, error)
    rewriter._client = SimpleNamespace(messages=stub)
    return rewriter, stub


# --------------------------------------------------------------------------
# Availability and degradation
# --------------------------------------------------------------------------


def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr(settings, "llm_enabled", False)
    status = llm.ClaudeRewriter().status()
    assert not status.available
    assert "off" in status.reason.lower()


def test_disabled_rewriter_returns_no_candidates(monkeypatch):
    monkeypatch.setattr(settings, "llm_enabled", False)
    assert llm.ClaudeRewriter().rewrite("Some passage of text here.") == []


def test_api_failure_is_swallowed():
    """A failed call must degrade to zero candidates, never crash the run."""
    rewriter, _ = _rewriter(None, error=RuntimeError("connection reset"))
    assert rewriter.rewrite("Some passage of text here.") == []


def test_refusal_returns_no_candidates():
    rewriter, _ = _rewriter(["A rewrite."], stop_reason="refusal")
    assert rewriter.rewrite("Some passage of text here.") == []


# --------------------------------------------------------------------------
# Prompt construction — quantitative targets, not adjectives
# --------------------------------------------------------------------------


def test_prompt_bans_markers_present_in_the_text():
    rewriter, stub = _rewriter(["Rewritten passage that is long enough."])
    rewriter.rewrite(
        "Moreover, we must delve into the intricate and multifaceted realm of design."
    )
    prompt = stub.calls[0]["messages"][0]["content"]
    for marker in ("delve", "intricate", "multifaceted"):
        assert marker in prompt


def test_prompt_pins_numbers_and_citations_verbatim():
    rewriter, stub = _rewriter(["Rewritten passage that is long enough."])
    rewriter.rewrite("Costs rose 12% in 2019 according to prior work (Smith, 2019).")
    prompt = stub.calls[0]["messages"][0]["content"]
    assert "12" in prompt and "2019" in prompt
    assert "(Smith, 2019)" in prompt


def test_prompt_states_sentence_length_targets():
    """Quantitative targets are the point — 'write naturally' returns the mean."""
    rewriter, stub = _rewriter(["Rewritten passage that is long enough."])
    rewriter.rewrite("A passage. With several sentences. Of differing length here.")
    prompt = stub.calls[0]["messages"][0]["content"]
    assert "8 words or fewer" in prompt
    assert "25 words or more" in prompt


def test_system_prompt_forbids_fabrication_and_naive_synonyms():
    assert "Never introduce a fact" in llm.SYSTEM_PROMPT
    assert "most obvious synonyms" in llm.SYSTEM_PROMPT


def test_request_omits_temperature():
    """Sonnet 5 and Opus 5 reject non-default sampling parameters."""
    rewriter, stub = _rewriter(["Rewritten passage that is long enough."])
    rewriter.rewrite("Some passage of text goes here for rewriting.")
    assert "temperature" not in stub.calls[0]
    assert "top_p" not in stub.calls[0]
    assert stub.calls[0]["output_config"]["effort"] == settings.bulk_effort


# --------------------------------------------------------------------------
# Response handling
# --------------------------------------------------------------------------


def test_variants_become_candidates():
    rewriter, _ = _rewriter(
        [
            "The team rebuilt the system. It shipped in March.",
            "In March the team shipped a rebuilt system.",
        ]
    )
    candidates = rewriter.rewrite("The system was rebuilt and shipped in March.")
    assert len(candidates) == 2
    assert all(c.transform == "claude" for c in candidates)


def test_unchanged_variant_is_dropped():
    text = "The system was rebuilt and shipped in March."
    rewriter, _ = _rewriter([text, "In March, a rebuilt system shipped."])
    assert len(rewriter.rewrite(text)) == 1


def test_usage_and_cost_are_tracked():
    rewriter, _ = _rewriter(["A sufficiently long rewritten passage here."])
    rewriter.rewrite("Some passage of text goes here for rewriting.")
    assert rewriter.usage.calls == 1
    assert rewriter.usage.input_tokens == 800
    assert rewriter.usage.output_tokens == 400
    # 800 in @ $3/M + 400 out @ $15/M
    assert rewriter.usage.cost_usd == pytest.approx((800 * 3 + 400 * 15) / 1e6)


def test_cost_estimate_scales_with_length_model_and_calls():
    sonnet = llm.ClaudeRewriter(model="claude-sonnet-5")
    short = sonnet.estimate_cost("word " * 100, calls=9)
    long = sonnet.estimate_cost("word " * 1000, calls=9)
    more_calls = sonnet.estimate_cost("word " * 1000, calls=20)
    opus = llm.ClaudeRewriter(model="claude-opus-5").estimate_cost("word " * 1000, calls=9)
    assert long > short
    assert more_calls > long
    assert opus > long


def test_cost_estimate_tracks_measured_spend():
    """Regression: the estimator assumed a fixed 3 calls while `aggressive`
    issues up to 20, understating a real run by roughly 5x. A quote that errs
    cheap is worse than no quote."""
    sonnet = llm.ClaudeRewriter(model="claude-sonnet-5")
    # Measured: a 370-word document at 'moderate' (9 calls) cost $0.154.
    estimate = sonnet.estimate_cost("word " * 370, calls=9)
    assert 0.05 < estimate < 0.60, estimate


def test_budget_cap_trips_once_spend_exceeds_limit():
    rewriter, _ = _rewriter(["A sufficiently long rewritten passage here."])
    assert not rewriter.over_budget(0.15)
    rewriter.rewrite("Some passage of text goes here for rewriting.")
    rewriter.usage.input_tokens = 10_000_000  # force the cap
    assert rewriter.over_budget(0.15)
    assert not rewriter.over_budget(None)


def test_truncated_response_is_discarded():
    """max_tokens truncation yields invalid JSON. It must be caught explicitly
    rather than surfacing as an opaque parse error that still burns a call."""
    rewriter, _ = _rewriter(["A rewrite."], stop_reason="max_tokens")
    assert rewriter.rewrite("Some passage of text goes here for rewriting.") == []
    assert rewriter.usage.calls == 1  # still counted: the call was billed


def test_failed_call_is_still_counted():
    rewriter, _ = _rewriter(None, error=RuntimeError("boom"))
    rewriter.rewrite("Some passage of text goes here for rewriting.")
    assert rewriter.usage.calls == 1


# --------------------------------------------------------------------------
# The guards apply to Claude exactly as they do to a regex
# --------------------------------------------------------------------------


def test_fabricating_variant_is_rejected_by_the_guards(monkeypatch):
    """A fluent rewrite that invents a statistic must still be discarded."""
    monkeypatch.setattr(settings, "llm_enabled", True)

    source = (
        "Remote work changed how the team operates. Coordination got harder across "
        "time zones, and the meeting load grew before anyone thought to trim it. "
        "Managers had to learn new habits. The shift was uneven across departments."
    )
    doc = segment.build(**ingest.from_text(source))

    fabricated = (
        "Remote work changed how the team operates. Studies show 73% of managers "
        "reported harder coordination across time zones. Meeting load grew by a "
        "third before anyone trimmed it. The shift was uneven."
    )
    rewriter, _ = _rewriter([fabricated])
    monkeypatch.setattr(
        loop, "ClaudeRewriter", lambda **kw: rewriter, raising=False
    )
    monkeypatch.setattr(llm, "ClaudeRewriter", lambda **kw: rewriter)

    result = loop.humanize(doc, use_llm=True, verify_with_models=False)
    assert fabricated not in result.final_text
    assert "73" not in result.final_text


def test_loop_falls_back_cleanly_when_claude_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "llm_enabled", False)
    doc = segment.build(**ingest.from_file(FIXTURES / "generated_sample.txt"))

    local = loop.humanize(doc, strength="moderate", verify_with_models=False)
    requested = loop.humanize(
        doc, strength="moderate", use_llm=True, verify_with_models=False
    )

    assert requested.llm_used is False
    assert requested.final_text == local.final_text
    assert any("unavailable" in n for n in requested.notes)


def test_pricing_table_covers_configured_models():
    for model in (settings.bulk_model, settings.quality_model):
        assert model in llm.PRICING, f"no price for configured model {model}"
