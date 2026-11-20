"""Groq client with model verification, pre-token retries, fallback, and streaming.

Reasoning models expose a separate ``reasoning`` field.  GPT-OSS uses
``reasoning_effort="low"`` with ``include_reasoning=False``; only
``delta.content`` is ever yielded, so reasoning text cannot reach the user.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq, RateLimitError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_fixed

from core.config import Settings, get_settings
from core.logging import get_logger
from generation.prompts import ERROR_TRY_AGAIN, ChatMessage, Language, canned

logger = get_logger(__name__)

TEMPERATURE = 0.2
MAX_TOKENS = 700
REASONING_EFFORT = "low"
MAX_ATTEMPTS_PER_MODEL = 2
RETRY_DELAY_SECONDS = 0.5

ModelCallback = Callable[[str, bool], None]


class EmptyStreamError(RuntimeError):
    """Raised when a model opens a stream but emits no content token."""


@dataclass(frozen=True)
class _OpenedStream:
    """A stream whose first content token has already arrived."""

    model: str
    first_text: str
    chunks: Iterator[Any]


_client: Groq | None = None
_client_lock = threading.Lock()


def get_client() -> Groq:
    """Return one process-wide official Groq client.

    SDK-level retries are disabled: ``stream_answer`` owns retry policy so it
    can guarantee no restart after the first token has been sent.
    """
    global _client
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = Groq(
                    api_key=settings.groq_api_key,
                    timeout=float(settings.llm_timeout_seconds),
                    max_retries=0,
                )
    return _client


def verify_models(client: Any | None = None) -> dict[str, bool]:
    """Check configured IDs against ``GET /openai/v1/models`` via the SDK.

    Missing configured models are logged as warnings; availability is returned
    for the health endpoint without blocking startup.
    """
    settings = get_settings()
    api = client or get_client()
    response = api.models.list()
    available = {str(getattr(item, "id", "")) for item in response.data}
    configured = (settings.groq_primary_model, settings.groq_fallback_model)
    result = {model: model in available for model in configured}
    for model, is_available in result.items():
        if not is_available:
            logger.warning(
                "groq_model_missing",
                extra={
                    "model": model,
                    "available_model_count": len(available),
                },
            )
    logger.info(
        "groq_models_verified",
        extra={
            "available_model_count": len(available),
            "primary": settings.groq_primary_model,
            "primary_available": result[settings.groq_primary_model],
            "fallback": settings.groq_fallback_model,
            "fallback_available": result[settings.groq_fallback_model],
        },
    )
    return result


def _is_retryable(error: BaseException) -> bool:
    """Only retry pre-token transport, rate-limit, 5xx, or empty-stream errors."""
    if isinstance(error, EmptyStreamError):
        return True
    if isinstance(error, (APITimeoutError, APIConnectionError, RateLimitError)):
        return True
    if isinstance(error, APIStatusError):
        return (error.status_code or 0) >= 500
    return False


def _log_retry(retry_state: Any) -> None:
    """Emit a structured warning when tenacity is about to retry."""
    error = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "groq_attempt_retrying",
        extra={
            "attempt": retry_state.attempt_number,
            "error_type": type(error).__name__ if error else None,
        },
    )


def _request_kwargs(
    messages: Sequence[Mapping[str, str]], model: str, settings: Settings
) -> dict[str, Any]:
    """Build the Groq chat-completion request for plain-text streaming."""
    return {
        "messages": [dict(message) for message in messages],
        "model": model,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "reasoning_effort": REASONING_EFFORT,
        "include_reasoning": False,
        "timeout": settings.llm_timeout_seconds,
    }


def _delta_text(chunk: Any) -> str:
    """Extract only final answer text; reasoning fields are deliberately ignored."""
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    return getattr(delta, "content", None) or ""


def _open_stream(
    client: Any,
    model: str,
    messages: Sequence[Mapping[str, str]],
    settings: Settings,
) -> _OpenedStream:
    """Open a stream and wait for the first content token (before retry point)."""
    stream = client.chat.completions.create(**_request_kwargs(messages, model, settings))
    chunks = iter(stream)
    for chunk in chunks:
        text = _delta_text(chunk)
        if text:
            return _OpenedStream(model=model, first_text=text, chunks=chunks)
    raise EmptyStreamError(f"No content token from {model}")


def _with_retries(
    operation: Callable[[], _OpenedStream], retry_delay: float
) -> _OpenedStream:
    """Run a pre-token operation with at most two attempts."""
    retryer = Retrying(
        stop=stop_after_attempt(MAX_ATTEMPTS_PER_MODEL),
        wait=wait_fixed(retry_delay),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
        before_sleep=_log_retry,
    )
    for attempt in retryer:
        with attempt:
            return operation()
    raise AssertionError("unreachable")  # pragma: no cover - tenacity always returns/raises


def stream_answer(
    messages: Sequence[ChatMessage],
    *,
    language: Language = "en",
    client: Any | None = None,
    primary_model: str | None = None,
    fallback_model: str | None = None,
    retry_delay: float = RETRY_DELAY_SECONDS,
    on_model: ModelCallback | None = None,
) -> Iterator[str]:
    """Stream final-answer text deltas for ``messages``.

    ``primary_model`` is tried first.  Before the first token, retryable
    failures get one short retry, then the fallback model is tried the same
    way.  After the first token there is never a restart: a stream failure
    appends ``ERROR_TRY_AGAIN`` and ends the generator.
    """
    settings = get_settings()
    api = client or get_client()
    primary = primary_model or settings.groq_primary_model
    fallback = fallback_model or settings.groq_fallback_model

    started = perf_counter()
    attempts: list[str] = []
    opened: _OpenedStream | None = None
    served_model: str | None = None
    fallback_used = False

    for model, is_fallback in ((primary, False), (fallback, True)):
        def attempt(active_model: str = model) -> _OpenedStream:
            attempts.append(active_model)
            return _open_stream(api, active_model, messages, settings)

        try:
            opened = _with_retries(attempt, retry_delay)
        except Exception as error:
            logger.warning(
                "groq_model_failed_before_first_token",
                extra={
                    "model": model,
                    "error_type": type(error).__name__,
                    "retryable": _is_retryable(error),
                },
            )
            if not _is_retryable(error):
                break
            continue

        served_model = model
        fallback_used = is_fallback
        break

    if opened is None or served_model is None:
        total_ms = (perf_counter() - started) * 1000
        logger.error(
            "groq_all_models_failed",
            extra={
                "primary": primary,
                "fallback": fallback,
                "attempts": attempts,
                "total_ms": round(total_ms, 1),
            },
        )
        yield canned("ERROR_TRY_AGAIN", language)
        return

    ttft_ms = (perf_counter() - started) * 1000
    logger.info(
        "groq_stream_started",
        extra={
            "model": served_model,
            "fallback_used": fallback_used,
            "ttft_ms": round(ttft_ms, 1),
            "attempts": attempts,
        },
    )
    if on_model:
        on_model(served_model, fallback_used)

    yield opened.first_text
    usage: Any = None
    try:
        for chunk in opened.chunks:
            text = _delta_text(chunk)
            if text:
                yield text
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage:
                usage = chunk_usage
    except Exception as error:
        logger.error(
            "groq_stream_failed_mid_stream",
            extra={"model": served_model, "error_type": type(error).__name__},
        )
        yield canned("ERROR_TRY_AGAIN", language)
    finally:
        total_ms = (perf_counter() - started) * 1000
        logger.info(
            "groq_stream_finished",
            extra={
                "model": served_model,
                "fallback_used": fallback_used,
                "total_ms": round(total_ms, 1),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
        )


__all__ = [
    "EmptyStreamError",
    "get_client",
    "verify_models",
    "stream_answer",
    "ERROR_TRY_AGAIN",
]
