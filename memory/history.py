"""History buffering and rolling summaries for Phase 4 chat memory.

The buffer is rebuilt from ``rag.messages`` each turn: only text after the
summary watermark is loaded, the oldest items are dropped first when the
message-count or token budget is exceeded, and the LLM receives the result as
plain ``user``/``assistant`` messages.  Summaries are produced only after a
response has been streamed, using the small fallback model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from core.config import get_settings
from core.logging import get_logger
from generation.llm import get_client
from memory import store

if TYPE_CHECKING:  # pragma: no cover - typing only
    from memory.store import Conversation, Message

logger = get_logger(__name__)

HISTORY_TOKEN_BUDGET = 1500
SUMMARY_MAX_WORDS = 150
SUMMARY_MAX_TOKENS = 300
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class BufferMessage:
    """A history item normalized for prompt construction."""

    role: str
    content: str
    id: int | None = None


@dataclass(frozen=True)
class HistoryContext:
    """Everything the turn needs from persisted memory."""

    conversation: "Conversation"
    unsummarized: list["Message"]
    buffer: list[BufferMessage]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate: word/punctuation pieces, floored by 4 chars/token."""
    if not text:
        return 0
    pieces = len(TOKEN_RE.findall(text))
    return max(pieces, (len(text) + 3) // 4, 1)


def _role_of(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "user"))
    return str(getattr(message, "role", "user"))


def _content_of(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _id_of(message: Any) -> int | None:
    if isinstance(message, dict):
        value = message.get("id")
    else:
        value = getattr(message, "id", None)
    return int(value) if value is not None else None


def _to_buffer_message(message: Any, content: str | None = None) -> BufferMessage:
    return BufferMessage(
        role=_role_of(message),
        content=_content_of(message) if content is None else content,
        id=_id_of(message),
    )


def build_history_buffer(
    messages: Sequence[Any],
    *,
    max_messages: int | None = None,
    token_budget: int = HISTORY_TOKEN_BUDGET,
) -> list[BufferMessage]:
    """Keep the newest messages, dropping the oldest first over either limit."""
    settings = get_settings()
    limit = settings.history_max_messages if max_messages is None else max_messages
    limit = max(0, int(limit))
    budget = max(1, int(token_budget))

    kept: list[BufferMessage] = []
    used = 0
    for message in reversed(list(messages)):
        if len(kept) >= limit:
            break
        content = _content_of(message)
        tokens = estimate_tokens(content)
        if kept and used + tokens > budget:
            break
        if not kept and tokens > budget:
            # A single oversized newest message still fits after truncation.
            max_chars = max(1, budget * 4)
            content = content[-max_chars:]
            kept.append(_to_buffer_message(message, content))
            used = estimate_tokens(content)
            break
        kept.append(_to_buffer_message(message, content))
        used += tokens

    kept.reverse()
    return kept


def to_chat_messages(messages: Sequence[BufferMessage]) -> list[dict[str, str]]:
    """Convert buffered history into the SDK's ``role``/``content`` shape."""
    return [{"role": item.role, "content": item.content} for item in messages]


def load_context(
    conversation_id: str,
    *,
    history_max: int | None = None,
    token_budget: int = HISTORY_TOKEN_BUDGET,
) -> HistoryContext:
    """Load the summary watermark and build this turn's history buffer."""
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} does not exist")

    unsummarized = store.list_messages(
        conversation_id, after_id=conversation.summary_upto_message_id or 0
    )
    buffer = build_history_buffer(
        unsummarized, max_messages=history_max, token_budget=token_budget
    )
    return HistoryContext(
        conversation=conversation, unsummarized=unsummarized, buffer=buffer
    )


def _summary_messages(
    messages: Sequence[Any], existing_summary: str | None, language: str
) -> list[dict[str, str]]:
    language_name = "Arabic" if language == "ar" else "English"
    history_lines = [
        f"{_role_of(item)}: {_content_of(item)}" for item in messages
    ]
    previous = (existing_summary or "(none)").strip()
    return [
        {
            "role": "system",
            "content": (
                "You write short continuity summaries for the BloodLink assistant. "
                f"Write in {language_name}, in at most {SUMMARY_MAX_WORDS} words, plain text only. "
                "Keep only what the user asked and what the assistant already answered. "
                "Do not add medical facts, advice, or anything not in the supplied turns."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Current summary:\n{previous}\n\n"
                "Older turns to fold into it:\n"
                + "\n".join(history_lines)
                + "\n\nReturn the updated summary only."
            ),
        },
    ]


def summarize_messages(
    messages: Sequence[Any],
    existing_summary: str | None,
    language: str,
    *,
    client: Any | None = None,
) -> str:
    """Summarize older turns with ``GROQ_FALLBACK_MODEL`` (non-streamed)."""
    if not messages:
        return existing_summary or ""

    settings = get_settings()
    api = client or get_client()
    response = api.chat.completions.create(
        model=settings.groq_fallback_model,
        messages=_summary_messages(messages, existing_summary, language),
        temperature=0,
        max_tokens=SUMMARY_MAX_TOKENS,
    )
    content = response.choices[0].message.content or ""
    summary = " ".join(str(content).split())
    words = summary.split()
    if len(words) > SUMMARY_MAX_WORDS:
        summary = " ".join(words[:SUMMARY_MAX_WORDS]) + " ..."
    return summary


def maybe_roll_summary(
    conversation_id: str,
    *,
    language: str,
    client: Any | None = None,
    trigger_messages: int | None = None,
    keep_last: int | None = None,
) -> bool:
    """Summarize messages beyond the buffer once the trigger count is exceeded.

    Returns ``True`` only when a new summary was persisted.  ``keep_last`` is
    normally ``HISTORY_MAX_MESSAGES`` and can be lowered by debug/test scripts
    to demonstrate the trigger without a long conversation.
    """
    settings = get_settings()
    trigger = settings.summary_trigger_messages if trigger_messages is None else trigger_messages
    keep = settings.history_max_messages if keep_last is None else keep_last
    keep = max(0, int(keep))

    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        return False

    watermark = conversation.summary_upto_message_id or 0
    unsummarized = store.list_messages(conversation_id, after_id=watermark)
    if len(unsummarized) <= trigger:
        return False

    to_summarize = unsummarized[:-keep] if keep else list(unsummarized)
    if not to_summarize:
        return False

    try:
        summary = summarize_messages(
            to_summarize, conversation.summary, language, client=client
        )
    except Exception as error:  # summary must never break the user turn
        logger.warning(
            "rolling_summary_failed",
            extra={
                "conversation_id": conversation_id,
                "error_type": type(error).__name__,
            },
        )
        return False

    if not summary:
        return False

    last_id = int(to_summarize[-1].id)
    store.set_summary(conversation_id, summary, last_id)
    logger.info(
        "rolling_summary_updated",
        extra={
            "conversation_id": conversation_id,
            "messages_summarized": len(to_summarize),
            "upto_message_id": last_id,
            "language": language,
            "summary_words": len(summary.split()),
        },
    )
    return True


__all__ = [
    "BufferMessage",
    "HistoryContext",
    "build_history_buffer",
    "estimate_tokens",
    "load_context",
    "maybe_roll_summary",
    "summarize_messages",
    "to_chat_messages",
]
