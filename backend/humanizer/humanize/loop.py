"""The adversarial rewrite loop.

Greedy search against our own detector, bounded by the guards. The shape:

    while score > target and iterations remain:
        pick the worst-scoring sentences
        generate candidate rewrites for each
        keep the candidate that lowers the *document* score most,
            provided it passes the meaning and fact guards
        stop early if a full pass produces no accepted edit

Two things make this practical rather than merely correct:

  Document-level objective, not sentence-level. Scoring a candidate in
  isolation misses the point — the heaviest structural signal is *variation*
  in sentence length across the document, so splitting one long sentence helps
  or hurts depending on every other sentence. Candidates are therefore scored
  by substituting them into the whole working document.

  Two-stage evaluation. The inner objective is the rule scorer with the spaCy
  block skipped (~2 ms), so hundreds of candidates per run are affordable. The
  expensive full ensemble — including the two language models — runs once
  before and once after, to report an honest before/after rather than the
  optimizer's own estimate of itself.

That last split matters: reporting the inner objective as the result would be
marking our own homework. The number the user sees comes from the same scorer
that judged the document originally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .. import segment
from ..config import settings
from ..detect import ensemble, rulescore
from ..detect import features as F
from ..document import Document
from . import guards
from .transforms import Candidate, generate_compound, merge_sentences

log = logging.getLogger(__name__)

# How many of the worst sentences to attempt per iteration, by strength.
# `finalists` is how many shortlisted candidates get re-ranked with the
# language models each iteration. Each one costs ~1s, so it trades wall-clock
# directly for alignment with what real detectors measure.
STRENGTH_PRESETS: dict[str, dict] = {
    "light": {
        "per_iteration": 3, "depth": 1, "max_iterations": 2,
        "llm_paragraphs": 2, "finalists": 3,
    },
    "moderate": {
        "per_iteration": 6, "depth": 2, "max_iterations": 3,
        "llm_paragraphs": 3, "finalists": 6,
    },
    "aggressive": {
        "per_iteration": 10, "depth": 2, "max_iterations": 4,
        "llm_paragraphs": 5, "finalists": 10,
    },
}


@dataclass(slots=True)
class Edit:
    sentence_index: int
    original: str
    rewritten: str
    transform: str
    note: str
    similarity: float | None
    objective_delta: float


def _paragraph_scores(doc: Document) -> list[tuple[int, float]]:
    """Mean sentence score per paragraph, for choosing what to hand the LLM."""
    per_sentence = {i: s for i, s, _ in rulescore.score_sentences(doc)}
    rows: list[tuple[int, float]] = []
    for paragraph in doc.paragraphs:
        scores = [per_sentence.get(s.index, 0.0) for s in paragraph.sentences]
        if scores:
            rows.append((paragraph.index, sum(scores) / len(scores)))
    return sorted(rows, key=lambda row: -row[1])


@dataclass(slots=True)
class HumanizeResult:
    original_text: str
    final_text: str
    edits: list[Edit] = field(default_factory=list)
    score_before: float = 0.0
    score_after: float = 0.0
    detection_before: ensemble.Detection | None = None
    detection_after: ensemble.Detection | None = None
    iterations: int = 0
    candidates_considered: int = 0
    candidates_rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    document_similarity: float | None = None
    target_met: bool = False
    notes: list[str] = field(default_factory=list)
    llm_used: bool = False
    llm_usage_summary: str = ""
    llm_cost_usd: float = 0.0

    @property
    def improvement(self) -> float:
        return self.score_before - self.score_after


def objective(text: str) -> float:
    """Cheap style-only score, used to SHORTLIST candidates.

    This is a screening function, not the thing being optimized. It covers the
    style tiers only, which is exactly the set of signals commercial detectors
    weight least — optimizing it alone produces text that satisfies our
    heuristics while still reading as machine-written to a real detector.
    Finalists are re-ranked by `full_objective` below.
    """
    doc = segment.build(text, use_nlp=False)
    feats = F.extract(doc, fast=True)
    return rulescore.score_features(feats, doc.word_count).score


def full_objective(text: str) -> tuple[float, float]:
    """Model-aware score used to CHOOSE among shortlisted candidates.

    Returns (combined, statistical). Runs the language models, so it costs
    ~1s per call — affordable only on a handful of finalists per iteration,
    which is why the two stages exist.

    The statistical tier is weighted above its ensemble share here on purpose.
    A held-out experiment (optimize against Qwen2.5-0.5B, judge with 1.5B)
    showed the style-only objective moved perplexity from 5.6 to 9.0 against a
    human baseline of 17.8 — real progress, but only about a third of the gap,
    while our own score claimed the target was met. Sampling signature is the
    part that generalizes to detectors we cannot see, so the search is pointed
    at it directly.
    """
    doc = segment.build(text, use_nlp=True)
    detection = ensemble.analyze(doc, use_models=True, use_fitted=False)
    statistical_score = detection.tier_scores.get("statistical")
    if statistical_score is None:
        return detection.score, 0.0
    style = detection.score
    return 0.65 * statistical_score + 0.35 * style, statistical_score


def _register_for(doc: Document, override: str | None) -> str:
    """Infer the register so transforms don't informalize a formal document.

    Deliberately does NOT key on nominalization density. That was the first
    version and it was wrong: heavy nominalization is an *LLM* marker, not an
    academic one, so nearly every machine-written document was classified
    academic — which switched off contractions and em-dash merges, the two
    transforms that most help such text. Citations are the honest academic
    signal, because machine prose does not spontaneously produce them.
    """
    if override:
        return override

    feats = F.extract(doc, fast=True)
    if feats.get("citation_rate", 0) > 0.2:
        return "academic"
    if feats.get("first_person_rate", 0) > 1.5 or feats.get("contraction_rate", 0) > 1.0:
        return "casual"
    return "professional"


def _merge_candidates(doc: Document, index: int, register: str) -> list[tuple[Candidate, int]]:
    """Merge candidates pair a sentence with its successor inside the same
    paragraph. Returns (candidate, end_offset_of_second_sentence)."""
    sentences = doc.sentences
    if index + 1 >= len(sentences):
        return []
    first, second = sentences[index], sentences[index + 1]
    if first.para_index != second.para_index:
        return []
    return [(c, second.end) for c in merge_sentences(first.text, second.text, register)]


def humanize(
    doc: Document,
    *,
    register: str | None = None,
    strength: str = "moderate",
    target: float | None = None,
    verify_with_models: bool = True,
    use_llm: bool = False,
    high_quality: bool = False,
    budget_usd: float | None = None,
    seed: int = 0,
) -> HumanizeResult:
    """Rewrite `doc` until it scores below `target` or the search stalls.

    `use_llm` adds Claude as a second candidate source. It changes nothing about
    how candidates are judged: LLM rewrites face the same meaning and fact
    guards and the same document-level objective as rule-based ones, and are
    discarded on the same terms. It is the only option here that sends text off
    this machine.

    `budget_usd` caps API spend for this run. Once reached the loop drops back
    to local transforms and finishes rather than aborting, so you always get a
    usable document.
    """
    preset = STRENGTH_PRESETS.get(strength, STRENGTH_PRESETS["moderate"])
    target = settings.target_score if target is None else target
    chosen_register = _register_for(doc, register)

    rewriter = None
    spent = None  # holds the rewriter after a budget cutoff, for accounting
    if use_llm:
        from .llm import ClaudeRewriter

        rewriter = ClaudeRewriter(
            model=settings.quality_model if high_quality else settings.bulk_model
        )
        if not rewriter.available:
            rewriter = None

    before = ensemble.analyze(doc, use_models=verify_with_models)
    result = HumanizeResult(
        original_text=doc.text,
        final_text=doc.text,
        score_before=before.score,
        detection_before=before,
    )
    result.notes.append(f"register detected: {chosen_register}")
    if use_llm and rewriter is None:
        from .llm import ClaudeRewriter

        result.notes.append(
            f"Claude rewriting requested but unavailable: {ClaudeRewriter().status().reason}"
        )
    elif rewriter is not None:
        result.llm_used = True
        result.notes.append(f"Claude engine: {rewriter.model}")

    working = doc.text
    current_objective = objective(working)

    # Last state confirmed good by the *full* scorer, so a bad iteration can be
    # rolled back rather than shipped.
    last_verified = before
    last_good_text = working

    for iteration in range(preset["max_iterations"]):
        result.iterations = iteration + 1
        edits_before_iteration = len(result.edits)

        if current_objective <= target:
            break

        working_doc = segment.build(working, use_nlp=True)
        ranked = sorted(
            rulescore.score_sentences(working_doc), key=lambda row: -row[1]
        )[: preset["per_iteration"]]

        # Best-first, not first-come. An earlier version accepted the best
        # candidate for each sentence in turn, which made the outcome depend on
        # sentence order: identical transform sets reached measurably different
        # optima. Here every candidate across every targeted sentence is scored
        # against the same baseline, then applied in order of improvement.
        scored: list[tuple[float, int, int, int, Candidate, float | None]] = []

        for sentence_index, _sentence_score, _reasons in ranked:
            if sentence_index >= len(working_doc.sentences):
                continue
            sentence = working_doc.sentences[sentence_index]

            options: list[tuple[Candidate, int]] = [
                (c, sentence.end)
                for c in generate_compound(
                    sentence.text,
                    register=chosen_register,
                    seed=seed + iteration,
                    depth=preset["depth"],
                )
            ]
            options.extend(_merge_candidates(working_doc, sentence_index, chosen_register))

            for candidate, end_offset in options:  # noqa: PLR1702
                result.candidates_considered += 1
                span_original = working[sentence.start : end_offset]

                trial = working[: sentence.start] + candidate.text + working[end_offset:]
                trial_objective = objective(trial)
                if trial_objective >= current_objective:
                    result.candidates_rejected += 1
                    result.rejection_reasons["no improvement"] = (
                        result.rejection_reasons.get("no improvement", 0) + 1
                    )
                    continue

                verdict = guards.check(span_original, candidate.text)
                if not verdict.passed:
                    result.candidates_rejected += 1
                    for reason in verdict.reasons:
                        key = reason.split("(")[0].strip()
                        result.rejection_reasons[key] = (
                            result.rejection_reasons.get(key, 0) + 1
                        )
                    continue

                scored.append(
                    (
                        trial_objective,
                        sentence.start,
                        end_offset,
                        sentence_index,
                        candidate,
                        verdict.similarity,
                    )
                )

        # --- Claude candidates, at paragraph scale ------------------------
        #
        # Whole paragraphs rather than sentences: rhythm is a property of a
        # passage, and a sentence rewritten in isolation reads detached from
        # its neighbours. These compete against the rule-based candidates on
        # exactly the same terms.
        if rewriter is not None and rewriter.over_budget(budget_usd):
            result.notes.append(
                f"Claude budget of ${budget_usd:.2f} reached after "
                f"{rewriter.usage.calls} calls — continuing with local transforms only."
            )
            # Keep the object for usage accounting; only stop calling it.
            # Dropping the reference here lost the spend record entirely.
            spent = rewriter
            rewriter = None

        if rewriter is not None:
            for paragraph_index, _score in _paragraph_scores(working_doc)[
                : preset["llm_paragraphs"]
            ]:
                paragraph = working_doc.paragraphs[paragraph_index]
                if paragraph.word_count < 25:
                    continue

                for candidate in rewriter.rewrite(
                    paragraph.text, register=chosen_register, n_variants=3
                ):
                    result.candidates_considered += 1
                    span_original = working[paragraph.start : paragraph.end]

                    trial = (
                        working[: paragraph.start]
                        + candidate.text
                        + working[paragraph.end :]
                    )
                    trial_objective = objective(trial)
                    if trial_objective >= current_objective:
                        result.candidates_rejected += 1
                        result.rejection_reasons["no improvement"] = (
                            result.rejection_reasons.get("no improvement", 0) + 1
                        )
                        continue

                    verdict = guards.check(span_original, candidate.text)
                    if not verdict.passed:
                        result.candidates_rejected += 1
                        for reason in verdict.reasons:
                            key = "claude: " + reason.split("(")[0].strip()
                            result.rejection_reasons[key] = (
                                result.rejection_reasons.get(key, 0) + 1
                            )
                        continue

                    scored.append(
                        (
                            trial_objective,
                            paragraph.start,
                            paragraph.end,
                            -paragraph_index - 1,  # negative marks a paragraph edit
                            candidate,
                            verdict.similarity,
                        )
                    )

        # Stage 2: re-rank the shortlist with the model-aware objective.
        #
        # The cheap objective can only see style. Promoting finalists on a
        # model-aware score is what stops the search from picking edits that
        # look good to our heuristics while leaving the sampling signature —
        # the thing real detectors key on — untouched.
        scored.sort(key=lambda row: row[0])
        if verify_with_models and scored:
            shortlist = scored[: preset["finalists"]]
            rescored: list[tuple[float, int, int, int, Candidate, float | None]] = []
            for entry in shortlist:
                _, start, end, index, candidate, sim = entry
                trial = working[:start] + candidate.text + working[end:]
                combined, _stat = full_objective(trial)
                rescored.append((combined, start, end, index, candidate, sim))
            rescored.sort(key=lambda row: row[0])
            # Finalists first, then whatever the cheap pass ranked below them.
            scored = rescored + scored[preset["finalists"] :]

        # Apply the best non-overlapping edits, right-to-left so earlier
        # character offsets stay valid as each is spliced in.
        chosen: list[tuple[float, int, int, int, Candidate, float | None]] = []
        claimed: list[tuple[int, int]] = []

        for entry in scored:
            _, start, end, _, _, _ = entry
            if any(start < c_end and c_start < end for c_start, c_end in claimed):
                continue
            chosen.append(entry)
            claimed.append((start, end))

        accepted_any = bool(chosen)
        for trial_objective, start, end, sentence_index, candidate, sim in sorted(
            chosen, key=lambda row: -row[1]
        ):
            span_original = working[start:end]
            working = working[:start] + candidate.text + working[end:]
            result.edits.append(
                Edit(
                    sentence_index=sentence_index,
                    original=span_original,
                    rewritten=candidate.text,
                    transform=candidate.transform,
                    note=candidate.note,
                    similarity=sim,
                    objective_delta=round(trial_objective - current_objective, 2),
                )
            )

        if accepted_any:
            # Edits were scored independently against the same baseline, so the
            # combined effect has to be re-measured rather than assumed.
            current_objective = objective(working)

            # Verify against the real scorer, not just the fast proxy.
            #
            # The proxy covers the style tiers only. That makes it blind to a
            # failure mode we measured: lexical de-marking can *raise* the
            # statistical score, because swapping an unusual word for its most
            # ordinary synonym moves the text toward the model's argmax — the
            # exact curvature signature Fast-DetectGPT keys on. Optimizing the
            # proxy alone drove one sample's statistical tier from 36 to 57
            # while the proxy reported steady progress. One full check per
            # iteration costs about a second and makes the search honest.
            if verify_with_models:
                checkpoint = ensemble.analyze(
                    segment.build(working, use_nlp=True), use_models=True
                )
                if checkpoint.score > last_verified.score:
                    result.notes.append(
                        f"iteration {iteration + 1} reverted: style score improved but "
                        f"the full score rose {last_verified.score:.1f} -> "
                        f"{checkpoint.score:.1f} (statistical tier "
                        f"{last_verified.tier_scores.get('statistical', 0):.0f} -> "
                        f"{checkpoint.tier_scores.get('statistical', 0):.0f})."
                    )
                    working = last_good_text
                    result.edits = result.edits[:edits_before_iteration]
                    current_objective = objective(working)
                    break
                last_verified = checkpoint
                last_good_text = working

        if not accepted_any:
            result.notes.append(
                f"search stalled at iteration {iteration + 1} — no candidate both "
                "improved the score and preserved meaning."
            )
            break

    # --- Honest verification with the full scorer -------------------------
    final_doc = segment.build(working, use_nlp=True)
    after = ensemble.analyze(final_doc, use_models=verify_with_models)

    result.final_text = working
    result.score_after = after.score
    result.detection_after = after
    result.target_met = after.score <= target

    if working != doc.text:
        result.document_similarity = guards.similarity(doc.text, working)
        threshold = (
            settings.min_document_similarity if guards.embeddings_available() else 0.55
        )
        if result.document_similarity < threshold:
            result.notes.append(
                f"WARNING: document-level similarity {result.document_similarity:.3f} "
                f"fell below {threshold:.2f}. Review the diff before using this."
            )

    if not result.edits:
        result.notes.append("No edits applied — nothing both helped and passed the guards.")

    # `spent` survives a mid-run budget cutoff, where `rewriter` is cleared.
    accounting = rewriter or spent
    if accounting is not None and accounting.usage.calls:
        result.llm_usage_summary = accounting.usage.summary()
        result.llm_cost_usd = accounting.usage.cost_usd

    return result
