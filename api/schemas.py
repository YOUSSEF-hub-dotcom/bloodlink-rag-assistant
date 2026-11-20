"""Pydantic request/response models and SSE frame formatting for the API."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from core.config import get_settings

EventName = Literal["meta", "token", "done", "error"]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields (including junk/binary payloads)."""

    model_config = ConfigDict(extra="forbid")


class ChatRequest(StrictModel):
    """Body of ``POST /v1/chat``."""

    conversation_id: UUID | None = None
    message: str = Field(
        min_length=1, max_length=get_settings().max_user_message_chars
    )


class Source(StrictModel):
    """One retrieval source, copied from retrieval metadata."""

    doc_id: str
    section_path: str
    language: str
    score: float


class MetaPayload(StrictModel):
    conversation_id: str
    language: str
    intent: str
    sources: list[Source]


class TokenPayload(StrictModel):
    text: str


class DonePayload(StrictModel):
    message_id: int


class ErrorPayload(StrictModel):
    message: str


class StoredMessage(StrictModel):
    id: int
    role: str
    content: str
    created_at: datetime


class MessagesResponse(StrictModel):
    conversation_id: str
    messages: list[StoredMessage]


class DeleteResponse(StrictModel):
    conversation_id: str
    deleted: bool


class HealthResponse(StrictModel):
    status: str
    device: str | None = None
    embedding_model: str | None = None
    chunks: int | None = None
    groq_models_available: bool | None = None
    error: str | None = None


def format_sse(event: EventName, data: BaseModel | dict[str, Any]) -> str:
    """Render one SSE frame with compact, UTF-8-safe JSON."""
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"


__all__ = [
    "ChatRequest",
    "DeleteResponse",
    "DonePayload",
    "ErrorPayload",
    "EventName",
    "HealthResponse",
    "MessagesResponse",
    "MetaPayload",
    "Source",
    "StoredMessage",
    "TokenPayload",
    "format_sse",
]
