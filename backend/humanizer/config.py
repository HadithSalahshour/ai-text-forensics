"""Configuration.

Privacy note: `llm_enabled` gates every outbound network call in the rewrite
path. It defaults to False and must be turned on explicitly, per run. Detection
never consults it because detection never touches the network.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_prefix="HUMANIZER_",
        extra="ignore",
    )

    # --- Server -----------------------------------------------------------
    # Loopback only. Do not change this to 0.0.0.0 without understanding that
    # it exposes your documents to the local network.
    host: str = "127.0.0.1"
    port: int = 8420

    # --- Storage ----------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    model_dir: Path = BACKEND_ROOT / "models"

    # --- Local detector models (Phase 2) ----------------------------------
    scorer_model: str = "Qwen/Qwen2.5-0.5B"
    reference_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    spacy_model: str = "en_core_web_sm"
    device: str = "auto"  # auto | mps | cuda | cpu

    # --- Detector thresholds ----------------------------------------------
    # Below this word count the statistics are too noisy to report a
    # confident number. The UI says so rather than guessing.
    min_reliable_words: int = 150
    target_score: float = 20.0  # humanizer stops once the doc scores under this

    # --- Rewrite engine (Phase 5) -----------------------------------------
    llm_enabled: bool = False
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    bulk_model: str = "claude-sonnet-5"
    quality_model: str = "claude-opus-5"
    bulk_effort: str = "high"
    max_iterations: int = 4
    candidates_per_span: int = 5

    # --- Rewrite guards ---------------------------------------------------
    min_sentence_similarity: float = 0.88
    min_document_similarity: float = 0.93

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
