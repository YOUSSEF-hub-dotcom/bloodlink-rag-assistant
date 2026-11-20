"""Unit tests for Phase 4 memory, rolling summary, and query rewriting.

No DB, GPU or network: storage functions and Groq clients are monkeypatched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from core.config import get_settings
from generation.prompts import build_messages
from memory.history import (
    BufferMessage,
    build_history_buffer,
    maybe_roll_summary,
)
from memory.rewrite import QueryAnalysis, analyze_query, fallback_analysis
from retrieval.search import SearchResult

FALLBACK = get_settings().groq_fallback_model


class FakeGroqClient:
    """Return scripted non-streamed completion contents and record requests."""

    def __init__(self, contents: list[Any]) -> None:
        self.contents = list(contents)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        content = self.contents.pop(0)
        if isinstance(content, BaseException):
            raise content
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _msg(message_id: int, text: str, role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(id=message_id, role=role, content=text)


def test_history_buffer_drops_oldest_by_count_and_token_budget() -> None:
    messages = [_msg(index, "one two three four five") for index in range(1, 6)]

    by_count = build_history_buffer(messages, max_messages=2, token_budget=100)
    assert [item.id for item in by_count] == [4, 5]

    by_budget = build_history_buffer(messages, max_messages=5, token_budget=12)
    assert [item.id for item in by_budget] == [4, 5]

    oversized = build_history_buffer(
        [_msg(9, "x" * 2000)], max_messages=1, token_budget=10
    )
    assert len(oversized) == 1
    assert oversized[0].id == 9
    assert len(oversized[0].content) <= 40


def test_maybe_roll_summary_triggers_and_stores_watermark(
    monkeypatch: Any,
) -> None:
    from memory import history as history_module

    conversation = SimpleNamespace(
        id="11111111-1111-4111-8111-111111111111",
        summary=None,
        summary_upto_message_id=0,
    )
    messages = [_msg(index, f"older turn {index}") for index in range(1, 6)]
    written: dict[str, Any] = {}

    monkeypatch.setattr(history_module.store, "get_conversation", lambda cid: conversation)
    monkeypatch.setattr(history_module.store, "list_messages", lambda cid, after_id: messages)
    monkeypatch.setattr(
        history_module.store,
        "set_summary",
        lambda cid, summary, upto: written.update(
            conversation_id=cid, summary=summary, upto=upto
        ),
    )

    client = FakeGroqClient(["ملخص المحادثة: شروط التبرع الأساسية."])
    triggered = maybe_roll_summary(
        conversation.id,
        language="ar",
        client=client,
        trigger_messages=2,
        keep_last=2,
    )

    assert triggered is True
    assert written["conversation_id"] == conversation.id
    assert written["upto"] == 3  # ids 1..3 summarized, ids 4..5 stay in buffer
    assert "ملخص المحادثة" in written["summary"]
    assert client.requests[0]["model"] == FALLBACK
    assert client.requests[0]["temperature"] == 0
    assert "Arabic" in str(client.requests[0]["messages"])


def test_maybe_roll_summary_respects_threshold(monkeypatch: Any) -> None:
    from memory import history as history_module

    conversation = SimpleNamespace(
        id="22222222-2222-4222-8222-222222222222",
        summary="existing",
        summary_upto_message_id=1,
    )
    messages = [_msg(2, "turn two"), _msg(3, "turn three")]
    monkeypatch.setattr(history_module.store, "get_conversation", lambda cid: conversation)
    monkeypatch.setattr(history_module.store, "list_messages", lambda cid, after_id: messages)

    def forbidden_store(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("summary must not be stored below the trigger")

    monkeypatch.setattr(history_module.store, "set_summary", forbidden_store)
    client = FakeGroqClient(["SHOULD NOT BE CALLED"])

    assert (
        maybe_roll_summary(
            conversation.id,
            language="en",
            client=client,
            trigger_messages=2,
            keep_last=0,
        )
        is False
    )
    assert client.requests == []


def test_analyze_query_uses_strict_structured_output_and_last_four_history() -> None:
    history = [_msg(index, f"hist-{index}") for index in range(1, 7)]
    answer = json.dumps(
        {
            "standalone_question": "ما هي الشروط الأساسية للتبرع بالدم إذا كنتِ امرأة؟",
            "language": "ar",
            "intent": "on_topic",
        },
        ensure_ascii=False,
    )
    client = FakeGroqClient([answer])

    analysis = analyze_query(
        "طب لو أنا ست؟", history, client=client, language_hint="ar"
    )

    assert analysis.standalone_question == "ما هي الشروط الأساسية للتبرع بالدم إذا كنتِ امرأة؟"
    assert analysis.language == "ar"
    assert analysis.intent == "on_topic"
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["model"] == FALLBACK
    assert request["temperature"] == 0
    assert request["response_format"]["json_schema"]["strict"] is True
    schema = request["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "standalone_question",
        "language",
        "intent",
    }
    prompt = request["messages"][1]["content"]
    assert "hist-6" in prompt and "hist-3" in prompt
    assert "hist-1" not in prompt and "hist-2" not in prompt
    assert "طب لو أنا ست؟" in prompt


def test_analyze_query_retries_best_effort_after_invalid_json() -> None:
    valid = json.dumps(
        {
            "standalone_question": "Can I donate while fasting?",
            "language": "en",
            "intent": "on_topic",
        }
    )
    client = FakeGroqClient(["not json", valid])

    analysis = analyze_query("What about fasting?", client=client, language_hint="en")

    assert analysis.standalone_question == "Can I donate while fasting?"
    assert len(client.requests) == 2
    assert client.requests[0]["response_format"]["json_schema"]["strict"] is True
    assert client.requests[1]["response_format"]["json_schema"]["strict"] is False


def test_analyze_query_falls_back_deterministically() -> None:
    client = FakeGroqClient([RuntimeError("offline"), RuntimeError("offline")])

    analysis = analyze_query("هل ينفع أتبرع وأنا بصوم؟", client=client)

    assert analysis == fallback_analysis("هل ينفع أتبرع وأنا بصوم؟")
    assert analysis.intent == "on_topic"
    assert analysis.language == "ar"
    assert len(client.requests) == 2


def test_build_messages_includes_summary_history_then_original_question() -> None:
    messages = build_messages(
        "طب لو أنا ست؟",
        "ar",
        [],
        history=[{"role": "user", "content": "إيه الشروط الأساسية؟"}],
        summary="ملخص قديم",
    )

    assert messages[0]["role"] == "system"
    assert "CONVERSATION SUMMARY" in messages[0]["content"]
    assert "ملخص قديم" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "إيه الشروط الأساسية؟"}
    assert messages[-1] == {"role": "user", "content": "طب لو أنا ست؟"}


def test_run_turn_with_conversation_wires_rewrite_history_and_persistence(
    monkeypatch: Any,
) -> None:
    import generation.debug_turn as debug_turn

    conversation_id = "33333333-3333-4333-8333-333333333333"
    original = "طب لو أنا ست؟"
    standalone = "ما هي الشروط الأساسية للتبرع بالدم إذا كنتِ امرأة؟"
    seen: dict[str, Any] = {}
    stored: list[tuple[str, str]] = []
    summary_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(debug_turn, "ensure_conversation", lambda *a, **k: None)
    monkeypatch.setattr(
        debug_turn,
        "load_context",
        lambda cid, history_max=None: SimpleNamespace(
            conversation=SimpleNamespace(summary="ملخص قديم"),
            buffer=[BufferMessage(role="user", content="إيه الشروط الأساسية؟", id=1)],
        ),
    )
    monkeypatch.setattr(
        debug_turn,
        "add_message",
        lambda cid, role, content: stored.append((role, content)),
    )
    monkeypatch.setattr(
        debug_turn,
        "get_conversation",
        lambda cid: SimpleNamespace(summary="ملخص جديد"),
    )

    def fake_summary(cid: str, **kwargs: Any) -> bool:
        summary_calls.append({"cid": cid, **kwargs})
        return True

    monkeypatch.setattr(debug_turn, "maybe_roll_summary", fake_summary)

    def fake_analyze(message: str, history: list[Any], **kwargs: Any) -> QueryAnalysis:
        seen["history"] = history
        assert message == original
        return QueryAnalysis(
            standalone_question=standalone, language="ar", intent="on_topic"
        )

    def fake_search(query: str, language: str, **kwargs: Any) -> list[SearchResult]:
        seen["query"] = query
        seen["language"] = language
        return [
            SearchResult(
                id="chunk-1",
                content="الشروط الأساسية للتبرع",
                metadata={"doc_id": "01_donor_eligibility"},
                section_path="الشروط الأساسية",
                language="ar",
                score=0.75,
            )
        ]

    def fake_stream(messages: list[Any], **kwargs: Any):
        seen["messages"] = messages
        yield "إجابة "
        yield "تجريبية"

    result = debug_turn.run_turn(
        original,
        conversation_id=conversation_id,
        client=SimpleNamespace(),
        embedder=object(),
        verify=False,
        analysis_fn=fake_analyze,
        search_fn=fake_search,
        stream_fn=fake_stream,
        history_max=2,
    )

    assert seen["query"] == standalone
    assert seen["language"] == "ar"
    assert seen["history"] == [
        {"role": "user", "content": "إيه الشروط الأساسية؟"}
    ]
    assert seen["messages"][-1] == {"role": "user", "content": original}
    assert result.rewritten_question == standalone
    assert result.intent == "on_topic"
    assert result.answer == "إجابة تجريبية"
    assert result.summary == "ملخص جديد"
    assert result.summary_triggered is True
    assert stored == [
        ("user", original),
        ("assistant", "إجابة تجريبية"),
    ]
    assert summary_calls[0]["cid"] == conversation_id
    assert summary_calls[0]["keep_last"] == 2
