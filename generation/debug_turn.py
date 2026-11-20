"""RAG debug CLI with optional Phase 4 conversation memory and query rewriting.

Usage::

    python -m generation.debug_turn "هل ينفع أتبرع وأنا بصوم؟"
    python -m generation.debug_turn "Can I donate?" --lang en
    python -m generation.debug_turn "Can I donate?" --mock-fallback
    python -m generation.debug_turn "طب لو أنا ست؟" --conversation-id <uuid>

Without ``--conversation-id`` the Phase 3 single-turn behavior is unchanged.
With it, the last buffered messages plus the rolling summary are loaded from
``rag.messages``/``rag.conversations``, the follow-up is rewritten with the
fallback model, retrieval uses the standalone question, and both the original
question and streamed answer are persisted.  ``--mock-fallback`` still cannot
be combined with a conversation because rewrite/summary need a real client.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import httpx
from groq import APITimeoutError

from core.config import get_settings
from core.db import bootstrap_schema
from core.embedder import get_embedder
from core.logging import get_logger
from generation.llm import get_client, stream_answer, verify_models
from generation.prompts import ChatMessage, Language, build_messages, canned
from ingestion.parse import detect_language
from memory.history import load_context, maybe_roll_summary
from memory.rewrite import QueryAnalysis, analyze_query
from memory.store import add_message, ensure_conversation, get_conversation
from retrieval.search import SearchResult, search

logger = get_logger(__name__)

SearchFn = Callable[..., list[SearchResult]]
StreamFn = Callable[..., Iterator[str]]
AnalysisFn = Callable[..., QueryAnalysis]
DeltaCallback = Callable[[str], None]
ModelCallback = Callable[[str, bool], None]


@dataclass(frozen=True)
class TurnResult:
    """Everything needed to inspect one debug turn."""

    question: str
    language: Language
    results: list[SearchResult]
    answer: str
    model: str | None
    llm_calls: int
    fallback_used: bool
    availability: dict[str, bool]
    conversation_id: str | None = None
    analysis: QueryAnalysis | None = None
    rewritten_question: str | None = None
    history_messages: int = 0
    summary: str | None = None
    summary_triggered: bool = False

    @property
    def intent(self) -> str | None:
        return self.analysis.intent if self.analysis else None


def _fake_chunk(text: str, reasoning: str | None = None) -> Any:
    """Build a minimal SDK-like streaming chunk without a network call."""
    delta = SimpleNamespace(content=text, reasoning=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


class _MockFallbackClient:
    """Fake Groq client used only by ``--mock-fallback`` to demo fallback."""

    def __init__(self, primary_model: str, fallback_model: str) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.attempts: list[str] = []
        self.chat = SimpleNamespace(completions=self)
        self.models = SimpleNamespace(list=self._list_models)

    def _list_models(self, **kwargs: Any) -> Any:
        return SimpleNamespace(
            data=[
                SimpleNamespace(id=self.primary_model),
                SimpleNamespace(id=self.fallback_model),
            ]
        )

    def create(self, **kwargs: Any) -> Any:
        model = str(kwargs["model"])
        self.attempts.append(model)
        if model == self.primary_model:
            request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            raise APITimeoutError(request=request)
        return iter(
            [
                _fake_chunk("Mock fallback "),
                _fake_chunk("answer.", reasoning="SECRET_REASONING_MUST_NOT_APPEAR"),
            ]
        )


def run_turn(
    question: str,
    *,
    language: Language | None = None,
    embedder: Any | None = None,
    client: Any | None = None,
    mock_fallback: bool = False,
    verify: bool = True,
    retry_delay: float = 0.5,
    search_fn: SearchFn = search,
    stream_fn: StreamFn = stream_answer,
    on_delta: DeltaCallback | None = None,
    on_model: ModelCallback | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
    analysis_fn: AnalysisFn | None = None,
    summary_trigger: int | None = None,
    history_max: int | None = None,
) -> TurnResult:
    """Run one retrieval-plus-generation turn and return the full result.

    ``on_delta`` is invoked as each token arrives so the CLI can stream live.
    No token is emitted before retrieval succeeds; an empty result set emits
    the canned ``NO_INFO`` reply with ``llm_calls=0``.  Memory is opt-in through
    ``conversation_id`` so Phase 3 callers and tests keep their exact behavior.
    """
    settings = get_settings()
    lang: Language = language or detect_language(question)  # type: ignore[assignment]
    api = client
    availability: dict[str, bool] = {}

    if mock_fallback:
        if client is not None:
            raise ValueError("Pass either client or mock_fallback, not both")
        if conversation_id:
            raise ValueError("mock_fallback cannot be combined with --conversation-id")
        api = _MockFallbackClient(settings.groq_primary_model, settings.groq_fallback_model)

    if conversation_id and client is None and not mock_fallback:
        api = get_client()
    memory_client = api

    if verify:
        try:
            availability = verify_models(api)
        except Exception as error:
            # Availability changes; startup verification must not block a turn.
            logger.warning(
                "groq_model_verification_failed",
                extra={"error_type": type(error).__name__},
            )

    analysis: QueryAnalysis | None = None
    rewritten = question
    history: list[ChatMessage] = []
    summary: str | None = None

    if conversation_id:
        ensure_conversation(conversation_id, user_id=user_id)
        context = load_context(conversation_id, history_max=history_max)
        summary = context.conversation.summary
        history = [
            {"role": item.role, "content": item.content} for item in context.buffer
        ]
        analyzer = analysis_fn or analyze_query
        analysis = analyzer(
            question,
            history,
            client=memory_client,
            language_hint=lang,
        )
        rewritten = analysis.standalone_question.strip() or question
        if language is None:
            lang = analysis.language

    model = embedder or get_embedder()
    results = search_fn(rewritten, lang, embedder=model)

    summary_state: dict[str, Any] = {"triggered": False, "summary": summary}

    def _persist_memory(answer: str) -> None:
        if not conversation_id:
            return
        add_message(conversation_id, "user", question)
        add_message(conversation_id, "assistant", answer)
        summary_state["triggered"] = maybe_roll_summary(
            conversation_id,
            language=lang,
            client=memory_client,
            trigger_messages=summary_trigger,
            keep_last=history_max,
        )
        conversation_after = get_conversation(conversation_id)
        summary_state["summary"] = conversation_after.summary if conversation_after else summary

    def _turn_result(
        answer: str,
        *,
        result_results: list[SearchResult],
        served_model: str | None,
        fallback_used: bool,
        llm_calls: int,
    ) -> TurnResult:
        return TurnResult(
            question=question,
            language=lang,
            results=result_results,
            answer=answer,
            model=served_model,
            llm_calls=llm_calls,
            fallback_used=fallback_used,
            availability=availability,
            conversation_id=conversation_id,
            analysis=analysis,
            rewritten_question=rewritten if conversation_id else None,
            history_messages=len(history),
            summary=summary_state["summary"],
            summary_triggered=bool(summary_state["triggered"]),
        )

    if not results:
        answer = canned("NO_INFO", lang)
        logger.info(
            "phase3_turn_no_context",
            extra={"language": lang, "llm_calls": 0, "conversation_id": conversation_id},
        )
        if on_delta:
            on_delta(answer)
        _persist_memory(answer)
        return _turn_result(
            answer,
            result_results=[],
            served_model=None,
            fallback_used=False,
            llm_calls=0,
        )

    messages: list[ChatMessage] = build_messages(
        question, lang, results, history=history, summary=summary
    )
    selected: list[tuple[str, bool]] = []

    def _model_selected(model_name: str, is_fallback: bool) -> None:
        selected.append((model_name, is_fallback))
        if on_model:
            on_model(model_name, is_fallback)

    pieces: list[str] = []
    for delta in stream_fn(
        messages,
        language=lang,
        client=api,
        retry_delay=retry_delay,
        on_model=_model_selected,
    ):
        pieces.append(delta)
        if on_delta:
            on_delta(delta)
    answer = "".join(pieces)
    served_model, fallback_used = selected[0] if selected else (None, False)

    logger.info(
        "phase3_turn_complete",
        extra={
            "language": lang,
            "chunks": len(results),
            "top_score": round(results[0].score, 4),
            "model": served_model,
            "fallback_used": fallback_used,
            "llm_calls": 1,
            "conversation_id": conversation_id,
            "intent": analysis.intent if analysis else None,
        },
    )
    _persist_memory(answer)
    return _turn_result(
        answer,
        result_results=results,
        served_model=served_model,
        fallback_used=fallback_used,
        llm_calls=1,
    )


def main(argv: list[str] | None = None) -> None:
    """Print a human-readable RAG turn debug trace."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run one RAG turn with streaming output.")
    parser.add_argument("question", nargs="+", help="Question text (quote it).")
    parser.add_argument("--lang", choices=("ar", "en"), default=None,
                        help="Force a language (default: detect or rewrite).")
    parser.add_argument("--conversation-id", default=None,
                        help="Enable Phase 4 memory for this UUID conversation.")
    parser.add_argument("--summary-trigger", type=int, default=None,
                        help="Temporary SUMMARY_TRIGGER_MESSAGES override (debug).")
    parser.add_argument("--history-max", type=int, default=None,
                        help="Temporary HISTORY_MAX_MESSAGES override (debug).")
    parser.add_argument("--mock-fallback", action="store_true",
                        help="Force primary-model timeouts and stream from the fallback.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the startup GET /openai/v1/models check.")
    parser.add_argument("--retry-delay", type=float, default=0.5,
                        help="Seconds between pre-token retries (default: 0.5).")
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip()
    if not question:
        parser.error("question must not be empty")

    if args.conversation_id:
        bootstrap_schema()

    print("=" * 100)
    print(f"query: {question}")
    if args.conversation_id:
        print(
            f"conversation_id: {args.conversation_id} | "
            f"summary_trigger: {args.summary_trigger if args.summary_trigger is not None else settings.summary_trigger_messages} | "
            f"history_max: {args.history_max if args.history_max is not None else settings.history_max_messages}"
        )
    print(
        f"mock_fallback: {args.mock_fallback} | verify_models: {not args.no_verify} "
        f"| threshold: {settings.score_threshold} | top_k: {settings.top_k}"
    )
    print("-" * 100)
    print("answer:", flush=True)

    result = run_turn(
        question,
        language=args.lang,
        conversation_id=args.conversation_id,
        summary_trigger=args.summary_trigger,
        history_max=args.history_max,
        mock_fallback=args.mock_fallback,
        verify=not args.no_verify,
        retry_delay=args.retry_delay,
        on_delta=lambda text: print(text, end="", flush=True),
    )

    print()
    print("-" * 100)
    if result.conversation_id:
        print(
            f"rewritten standalone_question: {result.rewritten_question}\n"
            f"rewrite language: {result.language} | intent: {result.intent} | "
            f"history_messages: {result.history_messages}"
        )
    print(
        f"language: {result.language} | llm_calls: {result.llm_calls} "
        f"| model: {result.model or 'none (canned)'} | fallback_used: {result.fallback_used}"
    )
    print(f"chunks: {len(result.results)}")
    for rank, item in enumerate(result.results, start=1):
        print(
            f"[{rank}] score={item.score:.4f} source={item.metadata.get('doc_id')} "
            f"section={item.section_path}"
        )
    if result.conversation_id:
        print(
            f"summary_triggered: {result.summary_triggered} | "
            f"conversations.summary: {result.summary or '(empty)'}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main(sys.argv[1:])
