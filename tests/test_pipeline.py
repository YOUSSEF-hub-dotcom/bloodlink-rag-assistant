"""Pipeline routing tests: canned paths must not call the answer generator."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from generation import pipeline as pipeline_module
from generation.prompts import EMERGENCY, NO_INFO, OFF_TOPIC
from memory.rewrite import QueryAnalysis
from retrieval.search import SearchResult

CONVERSATION_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


def _collect(agen: Any) -> list[tuple[str, dict[str, Any]]]:
    async def _inner() -> list[tuple[str, dict[str, Any]]]:
        return [item async for item in agen]

    return asyncio.run(_inner())


def _patch_storage(monkeypatch: Any, order: list[str], assistant_id: int = 42) -> None:
    monkeypatch.setattr(
        pipeline_module,
        "ensure_conversation",
        lambda cid, user_id=None: order.append("ensure"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_context",
        lambda cid: SimpleNamespace(
            conversation=SimpleNamespace(summary=None), buffer=[]
        ),
    )

    def fake_add(cid: str, role: str, content: str) -> SimpleNamespace:
        order.append(role)
        return SimpleNamespace(id=41 if role == "user" else assistant_id)

    monkeypatch.setattr(pipeline_module, "add_message", fake_add)
    monkeypatch.setattr(
        pipeline_module,
        "maybe_roll_summary",
        lambda cid, **kwargs: order.append("summary") or False,
    )


def _forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("answer generation/retrieval must not be called on canned paths")


def _fake_post_checks(order: list[str]) -> Any:
    def check(*args: Any, **kwargs: Any) -> dict[str, list[str]]:
        order.append("post_checks")
        return {"ungrounded_numbers": [], "medical_advice_flags": []}

    return check


def test_emergency_path_skips_rewrite_retrieval_and_generation(monkeypatch: Any) -> None:
    order: list[str] = []
    _patch_storage(monkeypatch, order)

    events = _collect(
        pipeline_module.run_pipeline(
            "أحتاج دم الآن!",
            conversation_id=CONVERSATION_ID,
            embedder=object(),
            analysis_fn=_forbidden,
            search_fn=_forbidden,
            stream_fn=_forbidden,
            post_check_fn=_forbidden,
        )
    )

    assert [name for name, _ in events] == ["meta", "token", "done"]
    assert events[0][1]["intent"] == "emergency"
    assert events[0][1]["sources"] == []
    assert events[1][1]["text"] == EMERGENCY["ar"]
    assert events[2][1]["message_id"] == 42
    assert order == ["ensure", "user", "assistant", "summary"]


def test_off_topic_intent_uses_canned_reply_without_answer_generation(
    monkeypatch: Any,
) -> None:
    order: list[str] = []
    _patch_storage(monkeypatch, order)
    analysis_calls: list[str] = []

    def fake_analysis(message: str, history: list[Any], **kwargs: Any) -> QueryAnalysis:
        analysis_calls.append(message)
        return QueryAnalysis(
            standalone_question=message, language="en", intent="off_topic"
        )

    events = _collect(
        pipeline_module.run_pipeline(
            "What is the weather in Cairo?",
            conversation_id=CONVERSATION_ID,
            embedder=object(),
            analysis_fn=fake_analysis,
            search_fn=_forbidden,
            stream_fn=_forbidden,
            post_check_fn=_forbidden,
        )
    )

    assert analysis_calls == ["What is the weather in Cairo?"]
    assert [name for name, _ in events] == ["meta", "token", "done"]
    assert events[0][1]["intent"] == "off_topic"
    assert events[0][1]["sources"] == []
    assert events[1][1]["text"] == OFF_TOPIC["en"]
    assert order == ["ensure", "user", "assistant", "summary"]


def test_no_context_uses_no_info_without_answer_generation(monkeypatch: Any) -> None:
    order: list[str] = []
    _patch_storage(monkeypatch, order)

    def fake_analysis(message: str, history: list[Any], **kwargs: Any) -> QueryAnalysis:
        return QueryAnalysis(
            standalone_question=message, language="en", intent="on_topic"
        )

    events = _collect(
        pipeline_module.run_pipeline(
            "What is the weather in Cairo?",
            conversation_id=CONVERSATION_ID,
            embedder=object(),
            analysis_fn=fake_analysis,
            search_fn=lambda *args, **kwargs: [],
            stream_fn=_forbidden,
            post_check_fn=_forbidden,
        )
    )

    assert [name for name, _ in events] == ["meta", "token", "done"]
    assert events[0][1]["intent"] == "on_topic"
    assert events[0][1]["sources"] == []
    assert events[1][1]["text"] == NO_INFO["en"]
    assert order == ["ensure", "user", "assistant", "summary"]


def test_grounded_turn_streams_tokens_then_persists_and_post_checks(
    monkeypatch: Any,
) -> None:
    order: list[str] = []
    _patch_storage(monkeypatch, order)
    result = SearchResult(
        id="chunk-1",
        content="Donors must be in good health. Age 18 to 65.",
        metadata={"doc_id": "01_donor_eligibility"},
        section_path="Eligibility > Basic rules",
        language="en",
        score=0.7811,
    )

    def fake_analysis(message: str, history: list[Any], **kwargs: Any) -> QueryAnalysis:
        return QueryAnalysis(
            standalone_question="What are the basic donor eligibility rules?",
            language="en",
            intent="on_topic",
        )

    def fake_stream(messages: list[Any], **kwargs: Any):
        kwargs["on_model"]("openai/gpt-oss-120b", False)
        yield "Donors "
        yield "must be healthy."

    events = _collect(
        pipeline_module.run_pipeline(
            "What are the rules?",
            conversation_id=CONVERSATION_ID,
            embedder=object(),
            analysis_fn=fake_analysis,
            search_fn=lambda *args, **kwargs: [result],
            stream_fn=fake_stream,
            post_check_fn=_fake_post_checks(order),
        )
    )

    assert [name for name, _ in events] == ["meta", "token", "token", "done"]
    assert events[0][1]["sources"] == [
        {
            "doc_id": "01_donor_eligibility",
            "section_path": "Eligibility > Basic rules",
            "language": "en",
            "score": 0.7811,
        }
    ]
    assert "".join(payload["text"] for name, payload in events if name == "token") == (
        "Donors must be healthy."
    )
    assert events[-1][1]["message_id"] == 42
    assert order == ["ensure", "user", "assistant", "summary", "post_checks"]
