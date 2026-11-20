"""SSE formatting, JWT verification and API dependency tests (no network/GPU)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jwt
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.auth import get_current_user
from api.main import RateLimiter, create_app
from api.routes import get_pipeline, rate_limit_chat
from api.schemas import format_sse


def test_sse_frame_formatting_is_compact_and_utf8_safe() -> None:
    assert format_sse("token", {"text": "مرحبًا"}) == (
        'event: token\ndata: {"text":"مرحبًا"}\n\n'
    )
    frame = format_sse(
        "meta",
        {
            "conversation_id": "c1",
            "language": "en",
            "intent": "on_topic",
            "sources": [],
        },
    )
    assert frame.startswith("event: meta\ndata: {")
    assert frame.endswith("\n\n")
    assert '"sources":[]' in frame


def test_jwt_bearer_verification_returns_401_and_200(monkeypatch: Any) -> None:
    settings = SimpleNamespace(
        auth_enabled=True,
        jwt_secret="phase5-unit-test-secret-with-32plus-bytes",
        jwt_algorithm="HS256",
        jwt_user_claim="id",
    )
    monkeypatch.setattr("api.auth.get_settings", lambda: settings)

    app = FastAPI()

    @app.get("/whoami")
    def whoami(user_id: str | None = Depends(get_current_user)) -> dict[str, Any]:
        return {"user": user_id}

    client = TestClient(app)

    missing = client.get("/whoami")
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"

    invalid = client.get("/whoami", headers={"Authorization": "Bearer invalid.token.value"})
    assert invalid.status_code == 401

    token = jwt.encode({"id": "user-7"}, settings.jwt_secret, algorithm="HS256")
    allowed = client.get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert allowed.status_code == 200
    assert allowed.json() == {"user": "user-7"}

    wrong_claim = jwt.encode({"sub": "user-7"}, settings.jwt_secret, algorithm="HS256")
    assert client.get(
        "/whoami", headers={"Authorization": f"Bearer {wrong_claim}"}
    ).status_code == 401

    settings.auth_enabled = False
    disabled = client.get("/whoami")
    assert disabled.status_code == 200
    assert disabled.json() == {"user": None}


def test_chat_route_streams_pipeline_events_as_sse() -> None:
    app = create_app()

    async def fake_pipeline(message: str, **kwargs: Any):
        assert message == "Can I donate?"
        yield ("meta", {
            "conversation_id": "c1",
            "language": "en",
            "intent": "on_topic",
            "sources": [],
        })
        yield ("token", {"text": "Usually yes."})
        yield ("done", {"message_id": 42})

    app.dependency_overrides[get_pipeline] = lambda: fake_pipeline
    app.dependency_overrides[get_current_user] = lambda: None
    app.dependency_overrides[rate_limit_chat] = lambda: None
    client = TestClient(app)

    response = client.post("/v1/chat", json={"message": "Can I donate?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == (
        'event: meta\ndata: {"conversation_id":"c1","language":"en","intent":"on_topic","sources":[]}\n\n'
        'event: token\ndata: {"text":"Usually yes."}\n\n'
        'event: done\ndata: {"message_id":42}\n\n'
    )
    app.dependency_overrides.clear()


def test_chat_allows_new_conversation_but_read_requires_existing(monkeypatch: Any) -> None:
    import api.auth as auth_module

    monkeypatch.setattr(auth_module, "get_conversation", lambda cid: None)
    app = create_app()

    async def fake_pipeline(message: str, **kwargs: Any):
        yield ("meta", {
            "conversation_id": kwargs["conversation_id"],
            "language": "en",
            "intent": "on_topic",
            "sources": [],
        })
        yield ("token", {"text": "OK."})
        yield ("done", {"message_id": 1})

    app.dependency_overrides[get_pipeline] = lambda: fake_pipeline
    app.dependency_overrides[get_current_user] = lambda: None
    app.dependency_overrides[rate_limit_chat] = lambda: None
    client = TestClient(app)
    cid = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"

    created = client.post(
        "/v1/chat", json={"conversation_id": cid, "message": "Can I donate?"}
    )
    assert created.status_code == 200
    assert "event: done" in created.text

    missing = client.get(f"/v1/conversations/{cid}/messages")
    assert missing.status_code == 404
    app.dependency_overrides.clear()


def test_rate_limiter_is_keyed_per_caller() -> None:
    limiter = RateLimiter(limit=2, window_seconds=60)
    assert limiter.allow("user:u1") is True
    assert limiter.allow("user:u1") is True
    assert limiter.allow("user:u1") is False
    assert limiter.allow("ip:127.0.0.1") is True
