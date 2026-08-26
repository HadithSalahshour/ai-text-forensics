"""Local HTTP API for the web UI.

Binds to 127.0.0.1 only. There is no auth because there is no remote access:
the server is reachable from this machine and nothing else. If you ever change
`host` to 0.0.0.0 you are exposing every document you analyze to your local
network, and you would need to add authentication first.

Models are loaded once at startup and held, so scoring a document costs ~1s
rather than the ~8s the CLI pays for a cold load each invocation.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from . import export, ingest, segment
from .config import settings
from .detect import ensemble, rulescore
from .humanize import humanize as run_humanize
from .nlp import has_nlp

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    # Warm the local models so the first real request isn't slow.
    try:
        from .detect.backend import get_models

        # Touching `.available` is what triggers the load; the result is only
        # logged because a failure here is not fatal — scoring degrades to the
        # style tiers.
        if not get_models().available:
            log.warning("Local models unavailable: %s", get_models().load_error)
    except Exception:  # noqa: BLE001
        log.warning("Local models unavailable at startup.", exc_info=True)
    yield


app = FastAPI(title="Humanized Writing", version="0.1.0", lifespan=lifespan)

# The Vite dev server runs on a different port; both are loopback-only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class SignalOut(BaseModel):
    key: str
    label: str
    tier: str
    value: float
    score: float
    weight: float
    evidence: str


class SentenceOut(BaseModel):
    index: int
    text: str
    start: int
    end: int
    score: float
    reasons: list[str]


class AnalyzeOut(BaseModel):
    text: str
    source_name: str
    score: float
    verdict: str
    reliable: bool
    note: str
    word_count: int
    sentence_count: int
    paragraph_count: int
    tier_scores: dict[str, float]
    signals: list[SignalOut]
    sentences: list[SentenceOut]
    statistical_available: bool
    fitted_used: bool


class AnalyzeIn(BaseModel):
    text: str
    use_models: bool = True


class HumanizeIn(BaseModel):
    # `register` shadows an attribute on pydantic's BaseModel, so the field is
    # named differently and aliased back to the wire name the UI sends.
    model_config = ConfigDict(populate_by_name=True)

    text: str
    strength: str = "moderate"
    register_hint: str | None = Field(default=None, alias="register")
    use_llm: bool = False
    high_quality: bool = False
    budget_usd: float = 0.15


class EditOut(BaseModel):
    original: str
    rewritten: str
    transform: str
    note: str
    similarity: float | None


class HumanizeOut(BaseModel):
    original_text: str
    final_text: str
    score_before: float
    score_after: float
    verdict_before: str
    verdict_after: str
    target_met: bool
    document_similarity: float | None
    edits: list[EditOut]
    notes: list[str]
    tier_before: dict[str, float]
    tier_after: dict[str, float]
    llm_used: bool
    llm_cost_usd: float
    llm_usage_summary: str


class ExportIn(BaseModel):
    text: str
    format: str = Field(default="txt")
    title: str | None = None


class EstimateIn(BaseModel):
    text: str
    strength: str = "moderate"
    high_quality: bool = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _analyze(text: str, source_name: str, use_models: bool) -> AnalyzeOut:
    if not text.strip():
        raise HTTPException(400, "No text provided.")

    doc = segment.build(text, source_name=source_name)
    detection = ensemble.analyze(doc, use_models=use_models)
    sentence_scores = {i: (s, r) for i, s, r in rulescore.score_sentences(doc)}

    return AnalyzeOut(
        text=doc.text,
        source_name=source_name,
        score=detection.score,
        verdict=detection.verdict,
        reliable=detection.reliable,
        note=detection.note,
        word_count=doc.word_count,
        sentence_count=len(doc.sentences),
        paragraph_count=len(doc.paragraphs),
        tier_scores=detection.tier_scores,
        signals=[
            SignalOut(
                key=s.key,
                label=s.label,
                tier=s.tier,
                value=round(s.value, 3),
                score=round(s.score, 3),
                weight=s.weight,
                evidence=s.evidence,
            )
            for s in detection.top_signals(14)
        ],
        sentences=[
            SentenceOut(
                index=s.index,
                text=s.text,
                start=s.start,
                end=s.end,
                score=sentence_scores.get(s.index, (0.0, []))[0],
                reasons=sentence_scores.get(s.index, (0.0, []))[1],
            )
            for s in doc.sentences
        ],
        statistical_available=detection.statistical_available,
        fitted_used=detection.fitted_used,
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/api/status")
def status() -> dict:
    from .calibration import fitted
    from .detect.backend import get_models
    from .humanize.llm import ClaudeRewriter

    models = get_models()
    llm_status = ClaudeRewriter().status()
    model = fitted.load()

    return {
        "privacy_mode": "claude" if llm_status.available else "local",
        "llm_available": llm_status.available,
        "llm_reason": llm_status.reason,
        "llm_model": settings.bulk_model,
        "quality_model": settings.quality_model,
        "statistical_available": models.available,
        "statistical_error": models.load_error,
        "spacy_available": has_nlp(),
        "calibrated_model": model.summary if model else None,
        "calibrated_in_use": bool(model and model.trustworthy),
        "target_score": settings.target_score,
        "min_reliable_words": settings.min_reliable_words,
        "supported_formats": sorted(ingest.SUPPORTED_SUFFIXES),
        "export_formats": list(export.FORMATS),
    }


@app.post("/api/analyze", response_model=AnalyzeOut)
def analyze(payload: AnalyzeIn) -> AnalyzeOut:
    return _analyze(
        ingest.from_text(payload.text)["text"], "pasted text", payload.use_models
    )


@app.post("/api/upload", response_model=AnalyzeOut)
async def upload(file: UploadFile = File(...)) -> AnalyzeOut:
    import tempfile
    from pathlib import Path

    suffix = Path(file.filename or "input.txt").suffix.lower()
    if suffix not in ingest.SUPPORTED_SUFFIXES:
        raise HTTPException(
            400,
            f"Unsupported format {suffix!r}. Use: {', '.join(sorted(ingest.SUPPORTED_SUFFIXES))}",
        )

    body = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
        handle.write(body)
        handle.flush()
        try:
            payload = ingest.from_file(handle.name)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Could not read that file: {exc}") from exc

    return _analyze(payload["text"], file.filename or "upload", True)


@app.post("/api/humanize", response_model=HumanizeOut)
def humanize_endpoint(payload: HumanizeIn) -> HumanizeOut:
    if not payload.text.strip():
        raise HTTPException(400, "No text provided.")

    doc = segment.build(**ingest.from_text(payload.text))
    result = run_humanize(
        doc,
        strength=payload.strength,
        register=payload.register_hint,
        use_llm=payload.use_llm,
        high_quality=payload.high_quality,
        budget_usd=payload.budget_usd if payload.use_llm else None,
    )

    return HumanizeOut(
        original_text=result.original_text,
        final_text=result.final_text,
        score_before=result.score_before,
        score_after=result.score_after,
        verdict_before=result.detection_before.verdict if result.detection_before else "",
        verdict_after=result.detection_after.verdict if result.detection_after else "",
        target_met=result.target_met,
        document_similarity=result.document_similarity,
        edits=[
            EditOut(
                original=e.original,
                rewritten=e.rewritten,
                transform=e.transform,
                note=e.note,
                similarity=e.similarity,
            )
            for e in result.edits
        ],
        notes=result.notes,
        tier_before=result.detection_before.tier_scores if result.detection_before else {},
        tier_after=result.detection_after.tier_scores if result.detection_after else {},
        llm_used=result.llm_used,
        llm_cost_usd=round(result.llm_cost_usd, 4),
        llm_usage_summary=result.llm_usage_summary,
    )


@app.post("/api/estimate")
def estimate(payload: EstimateIn) -> dict:
    from .humanize.llm import ClaudeRewriter
    from .humanize.loop import STRENGTH_PRESETS

    rewriter = ClaudeRewriter(
        model=settings.quality_model if payload.high_quality else settings.bulk_model
    )
    preset = STRENGTH_PRESETS.get(payload.strength, STRENGTH_PRESETS["moderate"])
    calls = preset["llm_paragraphs"] * preset["max_iterations"]
    return {
        "max_calls": calls,
        "estimate_usd": round(rewriter.estimate_cost(payload.text, calls=calls), 4),
        "model": rewriter.model,
        "available": rewriter.available,
    }


@app.post("/api/export")
def export_endpoint(payload: ExportIn) -> Response:
    try:
        blob = export.export(payload.text, payload.format, title=payload.title)
    except export.ExportError as exc:
        raise HTTPException(400, str(exc)) from exc

    fmt = payload.format.lower().lstrip(".")
    fmt = "md" if fmt == "markdown" else fmt
    return Response(
        content=blob,
        media_type=export.MEDIA_TYPES.get(fmt, "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="humanized.{fmt}"'},
    )


@app.post("/api/scorecard")
def scorecard(payload: AnalyzeIn) -> Response:
    doc = segment.build(**ingest.from_text(payload.text))
    detection = ensemble.analyze(doc, use_models=payload.use_models)
    markdown = export.scorecard_markdown(detection, source_name=doc.source_name)
    return Response(
        content=markdown.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="report.md"'},
    )


def serve() -> None:
    import uvicorn

    settings.ensure_dirs()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
