"""FastAPI routes: SSE chat, conversation CRUD, and health."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from api.auth import get_current_user, require_conversation_owner
from api.schemas import (
    ChatRequest,
    DeleteResponse,
    HealthResponse,
    MessagesResponse,
    StoredMessage,
    format_sse,
)
from core.db import check_database
from core.logging import get_logger
from generation.pipeline import Event, run_pipeline
from generation.prompts import canned
from guardrails.rules import detect_language
from memory.store import delete_conversation, list_messages

logger = get_logger(__name__)

PipelineFn = Callable[..., AsyncIterator[Event]]


def get_pipeline() -> PipelineFn:
    """Dependency indirection so tests can inject a fake pipeline."""
    return run_pipeline


def rate_limit_chat(
    request: Request,
    user_id: str | None = Depends(get_current_user),
) -> None:
    """Apply the in-memory per-user/IP limiter stored on ``app.state``."""
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    client_host = request.client.host if request.client else "unknown"
    key = f"user:{user_id}" if user_id else f"ip:{client_host}"
    if not limiter.allow(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many chat requests. Please slow down.",
        )


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Report DB, embedder and Groq availability without exposing secrets."""
    embedder = getattr(request.app.state, "embedder", None)
    availability: dict[str, bool] = getattr(request.app.state, "groq_availability", {})
    try:
        db = await asyncio.to_thread(check_database)
        db_error = None
    except Exception as error:
        logger.warning("health_database_failed", extra={"error_type": type(error).__name__})
        db = {}
        db_error = f"{type(error).__name__}: database unavailable"

    return HealthResponse(
        status="ok" if db_error is None else "degraded",
        device=getattr(embedder, "device", None),
        embedding_model=getattr(embedder, "model_name", None),
        chunks=db.get("chunks"),
        groq_models_available=all(availability.values()) if availability else None,
        error=db_error,
    )


@router.post("/v1/chat")
async def chat(
    body: ChatRequest,
    request: Request,
    user_id: str | None = Depends(get_current_user),
    pipeline: PipelineFn = Depends(get_pipeline),
    _: None = Depends(rate_limit_chat),
) -> StreamingResponse:
    """Stream one chat turn as SSE frames."""
    conversation_id = str(body.conversation_id) if body.conversation_id else None
    if conversation_id:
        require_conversation_owner(conversation_id, user_id, allow_missing=True)

    canned_error = canned("ERROR_TRY_AGAIN", detect_language(body.message))

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event, payload in pipeline(
                body.message,
                conversation_id=conversation_id,
                user_id=user_id,
            ):
                yield format_sse(event, payload)
        except Exception as error:
            logger.exception(
                "chat_stream_failed",
                extra={
                    "conversation_id": conversation_id,
                    "error_type": type(error).__name__,
                },
            )
            yield format_sse("error", {"message": canned_error})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/v1/conversations/{conversation_id}/messages", response_model=MessagesResponse
)
async def get_messages(
    conversation_id: UUID,
    user_id: str | None = Depends(get_current_user),
) -> MessagesResponse:
    """List stored messages for a conversation the caller owns."""
    cid = str(conversation_id)
    require_conversation_owner(cid, user_id)
    messages = await asyncio.to_thread(list_messages, cid)
    return MessagesResponse(
        conversation_id=cid,
        messages=[
            StoredMessage(
                id=message.id,
                role=message.role,
                content=message.content,
                created_at=message.created_at,
            )
            for message in messages
        ],
    )


@router.delete(
    "/v1/conversations/{conversation_id}", response_model=DeleteResponse
)
async def delete_conversation_route(
    conversation_id: UUID,
    user_id: str | None = Depends(get_current_user),
) -> DeleteResponse:
    """Delete a conversation and its messages (owner only)."""
    cid = str(conversation_id)
    require_conversation_owner(cid, user_id)
    deleted = await asyncio.to_thread(delete_conversation, cid)
    return DeleteResponse(conversation_id=cid, deleted=deleted)


__all__ = ["chat", "get_messages", "delete_conversation_route", "health", "router"]
