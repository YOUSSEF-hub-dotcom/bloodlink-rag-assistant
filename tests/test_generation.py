"""Mocked-Groq tests for Phase 3 (no network, no GPU, no real API key)."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any, Iterator

import httpx
import pytest
from groq import APITimeoutError

from core.config import get_settings
from generation.debug_turn import run_turn
from generation.llm import stream_answer, verify_models
from generation.prompts import (
    ERROR_TRY_AGAIN,
    NO_INFO,
    build_messages,
    canned,
    format_context,
)
from retrieval.search import SearchResult

SETTINGS = get_settings()
PRIMARY = SETTINGS.groq_primary_model
FALLBACK = SETTINGS.groq_fallback_model


def _timeout() -> APITimeoutError:
    return APITimeoutError(
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    )


def _chunk(text: str, reasoning: str | None = None, usage: Any = None) -> Any:
    delta = SimpleNamespace(content=text, reasoning=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage)


class FakeGroqClient:
    """Script per-model outcomes: an exception or an iterable of chunks."""

    def __init__(self, outcomes: dict[str, list[Any]]) -> None:
        self.outcomes = {model: deque(items) for model, items in outcomes.items()}
        self.attempts: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)
        self.models = SimpleNamespace(list=self._list_models)

    def _list_models(self, **kwargs: Any) -> Any:
        return SimpleNamespace(
            data=[SimpleNamespace(id=PRIMARY), SimpleNamespace(id=FALLBACK)]
        )

    def create(self, **kwargs: Any) -> Any:
        model = str(kwargs["model"])
        self.attempts.append(model)
        self.requests.append(kwargs)
        if model not in self.outcomes or not self.outcomes[model]:
            raise AssertionError(f"No scripted outcome for {model}")
        outcome = self.outcomes[model].popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return iter(outcome)


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Rules + context"},
        {"role": "user", "content": "Can I donate?"},
    ]


def test_context_formatting_and_canned_messages() -> None:
    result = SearchResult(
        id="id1",
        content="Donors should be in good health.",
        metadata={"doc_id": 'doc"1'},
        section_path="Eligibility <basic>",
        language="en",
        score=0.9,
    )
    context = format_context([result])
    assert '<chunk source="doc&quot;1" section="Eligibility &lt;basic&gt;">' in context
    assert "Donors should be in good health." in context
    assert context.startswith("<context>\n") and context.endswith("\n</context>")

    messages = build_messages("Can I donate?", "en", [result])
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith('You are "BloodLink Companion"')
    assert "<chunk source=" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Can I donate?"}
    assert "123" in canned("EMERGENCY", "ar")
    assert "123" in canned("EMERGENCY", "en")
    assert NO_INFO["en"].startswith("I don't have")


def test_stream_answer_streams_grounded_text_and_hides_reasoning() -> None:
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=5, total_tokens=17)
    client = FakeGroqClient(
        {
            PRIMARY: [
                [
                    _chunk("Usually ", reasoning="SECRET_REASONING"),
                    _chunk("yes.", usage=usage),
                ]
            ]
        }
    )

    output = "".join(
        stream_answer(
            _messages(), language="en", client=client, retry_delay=0.0
        )
    )

    assert output == "Usually yes."
    assert "SECRET_REASONING" not in output
    assert client.attempts == [PRIMARY]
    request = client.requests[0]
    assert request["model"] == PRIMARY
    assert request["stream"] is True
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 700
    assert request["reasoning_effort"] == "low"
    assert request["include_reasoning"] is False


def test_no_context_uses_canned_no_info_without_llm_call() -> None:
    client = FakeGroqClient({})
    deltas: list[str] = []

    def fake_search(query: str, language: str, **kwargs: Any) -> list[SearchResult]:
        assert query == "What is the weather in Cairo tomorrow?"
        assert language == "en"
        assert "embedder" in kwargs
        return []

    def forbidden_stream(*args: Any, **kwargs: Any) -> Iterator[str]:
        pytest.fail("LLM must not be called when retrieval returns no context")

    result = run_turn(
        "What is the weather in Cairo tomorrow?",
        language="en",
        embedder=object(),
        client=client,
        verify=True,
        search_fn=fake_search,
        stream_fn=forbidden_stream,
        on_delta=deltas.append,
    )

    assert result.answer == NO_INFO["en"]
    assert result.llm_calls == 0
    assert result.model is None
    assert result.results == []
    assert deltas == [NO_INFO["en"]]
    assert client.attempts == []


def test_forced_primary_failure_switches_to_fallback_before_first_token() -> None:
    client = FakeGroqClient(
        {
            PRIMARY: [_timeout(), _timeout()],
            FALLBACK: [[_chunk("Fallback "), _chunk("answer.")]],
        }
    )
    selected: list[tuple[str, bool]] = []

    output = "".join(
        stream_answer(
            _messages(),
            language="en",
            client=client,
            retry_delay=0.0,
            on_model=lambda model, is_fallback: selected.append((model, is_fallback)),
        )
    )

    assert output == "Fallback answer."
    assert client.attempts == [PRIMARY, PRIMARY, FALLBACK]
    assert selected == [(FALLBACK, True)]


def _chunks_then_failure() -> Iterator[Any]:
    yield _chunk("Partial ")
    raise RuntimeError("stream broke after first token")


def test_midstream_failure_appends_error_and_never_restarts() -> None:
    client = FakeGroqClient(
        {
            PRIMARY: [_chunks_then_failure()],
            FALLBACK: [[_chunk("SHOULD NOT BE USED")]],
        }
    )

    output = "".join(
        stream_answer(_messages(), language="en", client=client, retry_delay=0.0)
    )

    assert output == "Partial " + ERROR_TRY_AGAIN["en"]
    assert client.attempts == [PRIMARY]


def test_verify_models_reports_missing_configured_model() -> None:
    client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(id=PRIMARY)])
        )
    )
    availability = verify_models(client)
    assert availability[PRIMARY] is True
    assert availability[FALLBACK] is False
