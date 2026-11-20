"""Full chat-turn orchestration for the Phase 5 API.

Order is deliberately fixed:

1. auth (performed by the API dependency) and deterministic input checks
2. emergency regex
3. history load + standalone rewrite
4. intent routing
5. retrieval (hybrid RRF, thresholded)
6. streamed generation
7. persistence, rolling summary, then non-blocking post-checks

The function is an async generator of ``(event, payload)`` pairs so the API can
emit SSE frames as tokens arrive.  Canned paths never call the answer generator.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from time import perf_counter
from typing import Any, Callable

from core.config import get_settings
from core.embedder import get_embedder
from core.logging import get_logger
from generation.llm import stream_answer
from generation.prompts import ChatMessage, Language, build_messages, canned
from guardrails.checks import run_post_checks
from guardrails.rules import check_message, is_emergency
from memory.history import load_context, maybe_roll_summary
from memory.rewrite import QueryAnalysis, analyze_query
from memory.store import add_message, ensure_conversation, new_conversation_id
from retrieval.search import SearchResult, search

logger = get_logger(__name__)

SearchFn = Callable[..., list[SearchResult]]
StreamFn = Callable[..., Iterator[str]]
AnalysisFn = Callable[..., QueryAnalysis]
PostCheckFn = Callable[..., dict[str, list[str]]]
Event = tuple[str, dict[str, Any]]

_CANNED_BY_INTENT: dict[str, str] = {
    "off_topic": "OFF_TOPIC",
    "smalltalk": "GREETING_FALLBACK",
    "prompt_injection": "OFF_TOPIC",
}


def _sources(results: Sequence[SearchResult]) -> list[dict[str, Any]]:
    """Build source metadata from retrieval only, never from the LLM."""
    return [
        {
            "doc_id": str(result.metadata.get("doc_id") or result.id),
            "section_path": result.section_path,
            "language": result.language,
            "score": round(result.score, 4),
        }
        for result in results
    ]


_END = object()


def _next_or_end(iterator: Iterator[str]) -> str | object:
    """Never let StopIteration cross a Future boundary (asyncio forbids it)."""
    try:
        return next(iterator)
    except StopIteration:
        return _END


async def _aiter_sync(iterator: Iterator[str]) -> AsyncIterator[str]:
    """Drain a blocking sync iterator without stalling the event loop."""
    while True:
        item = await asyncio.to_thread(_next_or_end, iterator)
        if item is _END:
            return
        yield item


async def run_pipeline(
    message: str,
    *,
    conversation_id: str | None = None,
    user_id: str | None = None,
    language: Language | None = None,
    client: Any | None = None,
    embedder: Any | None = None,
    search_fn: SearchFn = search,
    stream_fn: StreamFn = stream_answer,
    analysis_fn: AnalysisFn = analyze_query,
    post_check_fn: PostCheckFn = run_post_checks,
) -> AsyncIterator[Event]:
    """Yield SSE-ready events for one complete chat turn."""
    settings = get_settings()
    turn_started = perf_counter()
    check = await asyncio.to_thread(check_message, message)
    if not check.allowed:
        detail = (
            "Message must not be empty."
            if check.reason == "empty_message"
            else f"Message exceeds the {settings.max_user_message_chars} character limit."
        )
        yield ("error", {"message": detail})
        return

    lang: Language = language or check.language
    conv_id = str(conversation_id or new_conversation_id())
    await asyncio.to_thread(ensure_conversation, conv_id, user_id=user_id)

    async def _finalize(answer: str, *, run_checks: bool = False) -> int:
        """Persist text, roll the summary, then run post-checks (in that order)."""
        await asyncio.to_thread(add_message, conv_id, "user", message)
        assistant = await asyncio.to_thread(add_message, conv_id, "assistant", answer)
        summary_triggered = await asyncio.to_thread(
            maybe_roll_summary, conv_id, language=lang, client=client
        )
        findings: dict[str, list[str]] = {}
        if run_checks:
            findings = await asyncio.to_thread(
                post_check_fn,
                answer,
                results,
                language=lang,
                conversation_id=conv_id,
            )
        logger.info(
            "pipeline_turn_finalized",
            extra={
                "conversation_id": conv_id,
                "intent": current_intent,
                "language": lang,
                "summary_triggered": summary_triggered,
                "ungrounded_numbers": len(findings.get("ungrounded_numbers", [])),
                "medical_advice_flags": len(findings.get("medical_advice_flags", [])),
            },
        )
        return int(assistant.id)

    async def _emit_canned(name: str, intent: str) -> AsyncIterator[Event]:
        text = canned(name, lang)  # type: ignore[arg-type]
        yield ("meta", {
            "conversation_id": conv_id,
            "language": lang,
            "intent": intent,
            "sources": [],
        })
        yield ("token", {"text": text})
        message_id = await _finalize(text)
        logger.info(
            "pipeline_canned_response",
            extra={
                "conversation_id": conv_id,
                "intent": intent,
                "language": lang,
                "llm_calls": 0,
                "total_latency_ms": round((perf_counter() - turn_started) * 1000, 1),
            },
        )
        yield ("done", {"message_id": message_id})

    current_intent = "input_check"
    results: list[SearchResult] = []

    # 2. Emergency regex — before any query analysis or Groq call.
    if await asyncio.to_thread(is_emergency, message, lang):
        current_intent = "emergency"
        async for event in _emit_canned("EMERGENCY", "emergency"):
            yield event
        return

    # 3. History + standalone rewrite.
    context = await asyncio.to_thread(load_context, conv_id)
    history: list[ChatMessage] = [
        {"role": item.role, "content": item.content} for item in context.buffer
    ]
    analysis: QueryAnalysis = await asyncio.to_thread(
        analysis_fn, message, history, client=client, language_hint=lang
    )
    standalone = analysis.standalone_question.strip() or message
    if language is None:
        lang = analysis.language
    current_intent = analysis.intent

    # 4. Intent routing — canned replies skip retrieval and answer generation.
    if current_intent in _CANNED_BY_INTENT:
        async for event in _emit_canned(_CANNED_BY_INTENT[current_intent], current_intent):
            yield event
        return

    # 5. Retrieval — the dense threshold still gates all candidates.
    model = embedder or await asyncio.to_thread(get_embedder)
    results = await asyncio.to_thread(
        search_fn, standalone, lang, embedder=model
    )

    if not results:
        async for event in _emit_canned("NO_INFO", "on_topic"):
            yield event
        return

    # 6. Streamed generation.
    yield ("meta", {
        "conversation_id": conv_id,
        "language": lang,
        "intent": current_intent,
        "sources": _sources(results),
    })

    messages = build_messages(
        message,
        lang,
        results,
        history=history,
        summary=context.conversation.summary,
    )
    selected: list[tuple[str, bool]] = []

    def _on_model(model_name: str, is_fallback: bool) -> None:
        selected.append((model_name, is_fallback))

    iterator = stream_fn(
        messages,
        language=lang,
        client=client,
        on_model=_on_model,
    )
    pieces: list[str] = []
    stream_started = perf_counter()
    ttft_ms: float | None = None
    async for delta in _aiter_sync(iterator):
        if not delta:
            continue
        if ttft_ms is None:
            ttft_ms = (perf_counter() - stream_started) * 1000
        pieces.append(delta)
        yield ("token", {"text": delta})

    answer = "".join(pieces).strip() or canned("ERROR_TRY_AGAIN", lang)  # type: ignore[arg-type]
    if not pieces:
        yield ("token", {"text": answer})

    # 7. Persistence -> summary -> post-checks.
    message_id = await _finalize(answer, run_checks=True)
    served_model, fallback_used = selected[0] if selected else (None, False)
    logger.info(
        "pipeline_turn_complete",
        extra={
            "conversation_id": conv_id,
            "intent": current_intent,
            "language": lang,
            "chunks": len(results),
            "top_score": round(results[0].score, 4),
            "model": served_model,
            "fallback_used": fallback_used,
            "llm_calls": 1,
            "ttft_ms": None if ttft_ms is None else round(ttft_ms, 1),
            "total_latency_ms": round((perf_counter() - turn_started) * 1000, 1),
        },
    )
    yield ("done", {"message_id": message_id})


__all__ = ["Event", "_sources", "run_pipeline"]
