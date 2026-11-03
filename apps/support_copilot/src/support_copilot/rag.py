"""The only place that wires rag_core into the copilot."""

from __future__ import annotations

from rag_core.config import Settings as RagSettings
from rag_core.service import RagService, build_service

from support_copilot.config import CopilotSettings


def rag_settings(settings: CopilotSettings) -> RagSettings:
    """rag_core settings built from ours.

    ``_env_file=None`` and an absolute ``data_dir`` keep rag_core away from the
    .env and data folder of its own checkout.
    """
    return RagSettings(
        _env_file=None,
        data_dir=settings.rag_data_dir,
        max_upload_mb=settings.max_upload_mb,
        max_documents=settings.max_documents,
        chunk_max_chars=settings.chunk_max_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
        index_backend=settings.index_backend,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
        min_score=settings.min_score,
        llm_provider=settings.llm_provider,
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
        llm_effort=settings.llm_effort,
        llm_max_tokens=settings.llm_max_tokens,
        llm_timeout_s=settings.llm_timeout_s,
        log_level=settings.log_level,
    )


def build_rag_service(settings: CopilotSettings) -> RagService:
    """The real service: sentence-transformers embedder (loaded lazily), persisted index, LLM."""
    return build_service(rag_settings(settings))
