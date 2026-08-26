"""API and export tests.

Runs against the FastAPI app in-process. The statistical tier is switched off
where it isn't the thing under test, so these stay fast.
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from humanizer import export
from humanizer.api import app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def machine_text() -> str:
    return (FIXTURES / "generated_sample.txt").read_text()


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------


def test_status_reports_capabilities(client):
    body = client.get("/api/status").json()
    assert body["privacy_mode"] in {"local", "claude"}
    assert isinstance(body["supported_formats"], list)
    assert ".docx" in body["supported_formats"]
    assert set(body["export_formats"]) == set(export.FORMATS)


def test_status_privacy_mode_tracks_llm_availability(client):
    body = client.get("/api/status").json()
    assert (body["privacy_mode"] == "claude") == body["llm_available"]


# --------------------------------------------------------------------------
# Analyze
# --------------------------------------------------------------------------


def test_analyze_returns_score_and_sentences(client, machine_text):
    body = client.post("/api/analyze", json={"text": machine_text}).json()
    assert 0 <= body["score"] <= 100
    assert body["verdict"]
    assert body["sentences"]
    assert body["signals"]
    assert body["word_count"] > 100


def test_analyze_sentence_offsets_match_returned_text(client, machine_text):
    """The heatmap paints by offset, so a mismatch here would silently
    highlight the wrong words."""
    body = client.post("/api/analyze", json={"text": machine_text}).json()
    text = body["text"]
    for sentence in body["sentences"]:
        assert text[sentence["start"] : sentence["end"]] == sentence["text"]


def test_analyze_sentences_are_ordered(client, machine_text):
    body = client.post("/api/analyze", json={"text": machine_text}).json()
    starts = [s["start"] for s in body["sentences"]]
    assert starts == sorted(starts)


def test_analyze_rejects_empty_text(client):
    assert client.post("/api/analyze", json={"text": "   "}).status_code == 400


def test_short_text_is_marked_unreliable(client):
    body = client.post("/api/analyze", json={"text": "A very short note indeed."}).json()
    assert body["reliable"] is False
    assert body["note"]


def test_signals_carry_evidence(client, machine_text):
    body = client.post("/api/analyze", json={"text": machine_text}).json()
    for signal in body["signals"]:
        assert signal["evidence"]
        assert 0.0 <= signal["score"] <= 1.0


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------


def test_upload_txt(client):
    with open(FIXTURES / "human_sample.txt", "rb") as handle:
        response = client.post(
            "/api/upload", files={"file": ("human_sample.txt", handle, "text/plain")}
        )
    body = response.json()
    assert response.status_code == 200
    assert body["score"] < 40
    assert body["source_name"] == "human_sample.txt"


def test_upload_rejects_unsupported_format(client):
    response = client.post(
        "/api/upload", files={"file": ("notes.rtf", io.BytesIO(b"x"), "text/rtf")}
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_upload_roundtrips_a_generated_docx(client, tmp_path):
    """Exercises the real docx reader, not a stub."""
    blob = export.to_docx("First paragraph here with enough words.\n\nSecond paragraph.")
    path = tmp_path / "doc.docx"
    path.write_bytes(blob)
    with open(path, "rb") as handle:
        response = client.post("/api/upload", files={"file": ("doc.docx", handle, "")})
    assert response.status_code == 200
    assert "First paragraph" in response.json()["text"]


# --------------------------------------------------------------------------
# Humanize
# --------------------------------------------------------------------------


def test_humanize_improves_and_preserves_meaning(client, machine_text):
    body = client.post(
        "/api/humanize", json={"text": machine_text, "strength": "moderate"}
    ).json()
    assert body["score_after"] < body["score_before"]
    assert body["document_similarity"] > 0.90
    assert body["edits"]
    assert body["llm_used"] is False  # not requested


def test_humanize_does_not_call_the_api_unless_asked(client, machine_text):
    body = client.post("/api/humanize", json={"text": machine_text}).json()
    assert body["llm_cost_usd"] == 0.0
    assert body["llm_usage_summary"] == ""


def test_humanize_rejects_empty_text(client):
    assert client.post("/api/humanize", json={"text": ""}).status_code == 400


# --------------------------------------------------------------------------
# Estimate
# --------------------------------------------------------------------------


def test_estimate_scales_with_strength(client, machine_text):
    light = client.post(
        "/api/estimate", json={"text": machine_text, "strength": "light"}
    ).json()
    aggressive = client.post(
        "/api/estimate", json={"text": machine_text, "strength": "aggressive"}
    ).json()
    assert aggressive["max_calls"] > light["max_calls"]
    assert aggressive["estimate_usd"] > light["estimate_usd"]


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt,magic",
    [("txt", b"Hello"), ("md", b"Hello"), ("docx", b"PK"), ("pdf", b"%PDF")],
)
def test_export_produces_valid_files(client, fmt, magic):
    response = client.post(
        "/api/export", json={"text": "Hello world.\n\nSecond paragraph.", "format": fmt}
    )
    assert response.status_code == 200
    assert response.content.startswith(magic)
    assert f"humanized.{fmt}" in response.headers["content-disposition"]


def test_export_rejects_unknown_format(client):
    response = client.post("/api/export", json={"text": "hi", "format": "rtf"})
    assert response.status_code == 400


def test_pdf_export_escapes_markup():
    """reportlab parses a small HTML subset, so raw angle brackets would be
    swallowed or raise."""
    blob = export.to_pdf("Compare a < b & c > d in the results.")
    assert blob.startswith(b"%PDF")


def test_scorecard_contains_evidence(client, machine_text):
    response = client.post("/api/scorecard", json={"text": machine_text})
    body = response.content.decode()
    assert response.status_code == 200
    assert "# AI-writing report" in body
    assert "## By tier" in body
    assert "## Strongest signals" in body
    assert "proof of authorship" in body


def test_markdown_export_normalizes_paragraphs():
    out = export.to_markdown("One.\n\n\n\nTwo.\n").decode()
    assert out == "One.\n\nTwo.\n"
