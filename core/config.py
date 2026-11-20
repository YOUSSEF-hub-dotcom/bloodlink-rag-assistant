"""Application settings, read exclusively from environment variables / ``.env``.

No secret is ever hardcoded here.  ``.env.example`` documents every supported
variable; ``.env`` contains the local development values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE: Path = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    """Validated view of the process environment."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Groq / LLM
    groq_api_key: str = ""
    groq_primary_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "openai/gpt-oss-20b"
    llm_timeout_seconds: int = 30

    # Database
    database_url: str = "postgresql://postgres:PASSWORD@localhost:5432/bloodlink_db"
    rag_db_schema: str = "rag"

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32

    # Chunking
    max_chunk_tokens: int = 400
    chunk_overlap_tokens: int = 40

    # Retrieval
    top_k: int = 5
    score_threshold: float = 0.55
    hnsw_ef_search: int = 40

    # Memory
    history_max_messages: int = 8
    summary_trigger_messages: int = 16
    max_user_message_chars: int = 1000

    # Auth
    auth_enabled: bool = True
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_user_claim: str = "id"

    # API
    cors_allow_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    chat_rate_limit: int = 20
    chat_rate_window_seconds: int = 60

    # Logging
    log_level: str = "INFO"

    @field_validator("embedding_device")
    @classmethod
    def _normalize_device(cls, value: str) -> str:
        device = value.strip().lower()
        if device not in {"cuda", "cpu"}:
            raise ValueError("EMBEDDING_DEVICE must be 'cuda' or 'cpu'")
        return device

    def redacted(self) -> dict[str, Any]:
        """Return settings safe to log (secrets replaced)."""
        data: dict[str, Any] = self.model_dump()
        for key in ("groq_api_key", "jwt_secret"):
            if data.get(key):
                data[key] = "***"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings: Settings = get_settings()
