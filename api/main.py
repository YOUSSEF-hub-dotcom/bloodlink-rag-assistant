"""FastAPI application factory with lifespan loading, CORS and rate limiting."""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import asynccontextmanager
from threading import Lock
from time import monotonic
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from core.config import get_settings
from core.db import bootstrap_schema, close_pool
from core.embedder import get_embedder
from core.logging import get_logger
from generation.llm import verify_models

logger = get_logger(__name__)


class RateLimiter:
    """Simple in-memory sliding-window limiter keyed by user id or client IP."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = max(1, int(limit))
        self.window = max(1, int(window_seconds))
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        """Return whether ``key`` has remaining budget in the current window."""
        now = monotonic()
        with self._lock:
            events = self._hits[key]
            while events and now - events[0] >= self.window:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True


def _cors_origins() -> list[str]:
    raw = get_settings().cors_allow_origins
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load schema, DB pool, embedder and Groq model availability once."""
    bootstrap_schema()
    app.state.embedder = get_embedder()
    app.state.groq_availability = {}
    try:
        app.state.groq_availability = verify_models()
    except Exception as error:
        logger.warning(
            "groq_startup_verification_failed",
            extra={"error_type": type(error).__name__},
        )
    logger.info(
        "api_startup_complete",
        extra={
            "device": app.state.embedder.device,
            "embedding_model": app.state.embedder.model_name,
            "groq_available": all(app.state.groq_availability.values())
            if app.state.groq_availability
            else None,
        },
    )
    try:
        yield
    finally:
        close_pool()


def create_app() -> FastAPI:
    """Build the API app with dependencies ready for tests or uvicorn."""
    settings = get_settings()
    app = FastAPI(title="BloodLink RAG API", version="1.0.0", lifespan=lifespan)
    app.state.rate_limiter = RateLimiter(
        settings.chat_rate_limit, settings.chat_rate_window_seconds
    )
    app.state.embedder = None
    app.state.groq_availability = {}
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.include_router(router)
    return app


app = create_app()


__all__ = ["RateLimiter", "app", "create_app", "lifespan"]
