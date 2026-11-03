"""Copilot settings, read from environment variables and support-copilot/.env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """The checkout root (holds pyproject.toml); the working directory when installed."""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").exists() else Path.cwd()


# Our own .env: rag_core would otherwise read the .env of its own checkout.
PROJECT_ROOT = _project_root()
ENV_FILE = PROJECT_ROOT / ".env"


class CopilotSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # Storage: the SQLite file and the RAG index/uploads live under data_dir.
    data_dir: Path = Path("data")
    database_url: str = ""  # empty -> sqlite file in data_dir
    max_upload_mb: int = 10
    log_level: str = "INFO"
    company_name: str = ""  # signs reply drafts: "служба поддержки <company_name>"

    # Generation (no provider or no key -> retrieval-only, so every draft is escalated)
    llm_provider: Literal["openai", "anthropic", "none"] = "anthropic"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    llm_model: str = ""
    llm_base_url: str = ""  # OpenAI-compatible host (Groq, Gemini, ...); needs LLM_MODEL
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    llm_max_tokens: int = 4000
    llm_timeout_s: float = 60.0

    # Retrieval and ingestion, passed through to rag_core
    index_backend: Literal["numpy", "faiss"] = "faiss"
    embedding_model: str = "intfloat/multilingual-e5-small"
    top_k: int = 5
    min_score: float = 0.75
    chunk_max_chars: int = 500
    chunk_overlap_chars: int = 100
    max_documents: int = 50

    @field_validator("data_dir")
    @classmethod
    def _resolve_data_dir(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def resolved_database_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'copilot.db').as_posix()}"

    @property
    def rag_data_dir(self) -> Path:
        return self.data_dir / "rag"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> CopilotSettings:
    return CopilotSettings()
