"""Optional local Streamlit test UI for the BloodLink RAG API.

This is a lightweight developer tool for interacting with a running FastAPI
server in the browser.  It is NOT the production UI.  Start the API first::

    .\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000

Then launch this app::

    .\.venv\Scripts\python.exe -m streamlit run demo_ui.py

The UI posts each turn to ``POST /v1/chat``, reads the SSE response token by
token, and keeps one conversation UUID for the whole session until
"New Conversation" is pressed.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Iterator

import httpx
import streamlit as st

DEFAULT_API_URL = "http://127.0.0.1:8000/v1/chat"


def _init_session() -> None:
    """Create the conversation id and message buffer once per browser session."""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict[str, Any]] = []


def _iter_sse(response: httpx.Response) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse ``event:``/``data:`` frames from an httpx streaming response."""
    event_name = ""
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
        elif line == "" and data_lines:
            raw = "".join(data_lines)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"message": raw}
            if not isinstance(payload, dict):
                payload = {"message": str(payload)}
            yield event_name, payload
            event_name = ""
            data_lines = []


def stream_chat(
    api_url: str, message: str, conversation_id: str, jwt_token: str
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(event, payload)`` pairs from the FastAPI SSE endpoint."""
    headers = {"Accept": "text/event-stream"}
    if jwt_token.strip():
        headers["Authorization"] = f"Bearer {jwt_token.strip()}"

    body = {"conversation_id": conversation_id, "message": message}
    timeout = httpx.Timeout(30.0, read=180.0)
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", api_url, json=body, headers=headers) as response:
            if response.status_code >= 400:
                detail = response.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status_code}: {detail}")
            yield from _iter_sse(response)


def _debug_payload(meta: dict[str, Any]) -> dict[str, Any]:
    """Show only retrieval metadata; never show internal prompts or diagnostics."""
    return {
        "conversation_id": meta.get("conversation_id", st.session_state.conversation_id),
        "intent": meta.get("intent"),
        "language": meta.get("language"),
        "sources": [
            {
                "doc_id": source.get("doc_id"),
                "section_path": source.get("section_path"),
                "score": source.get("score"),
            }
            for source in meta.get("sources", [])
        ],
    }


def _render_debug(meta: dict[str, Any]) -> None:
    with st.expander("Debug / Retrieval Metadata", expanded=False):
        st.json(_debug_payload(meta))


def main() -> None:
    st.set_page_config(page_title="BloodLink RAG — Local Test UI", page_icon="🩸")
    st.title("BloodLink RAG — local test UI")
    st.caption("Optional developer tool for the running FastAPI server, not the production UI.")
    _init_session()

    with st.sidebar:
        st.header("Connection")
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        jwt_token = st.text_input(
            "Bearer JWT (optional)",
            type="password",
            help="Only needed when the server has AUTH_ENABLED=true.",
        )
        st.caption(f"Conversation ID: `{st.session_state.conversation_id}`")
        if st.button("New Conversation", use_container_width=True):
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("meta"):
                _render_debug(message["meta"])

    prompt = st.chat_input("Ask about blood donation…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = ""
    meta: dict[str, Any] = {}
    error: str | None = None
    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            for event, payload in stream_chat(
                api_url, prompt, st.session_state.conversation_id, jwt_token
            ):
                if event == "meta":
                    meta = payload
                elif event == "token":
                    answer += payload.get("text", "")
                    placeholder.markdown(answer + "▌")
                elif event == "error":
                    error = payload.get("message") or "The server reported an error."
        except Exception as exc:  # show connection/server errors in the chat
            error = str(exc)

        if error:
            placeholder.error(error)
            answer = error
        else:
            placeholder.markdown(answer or "(empty response)")
        if meta:
            _render_debug(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "meta": meta}
    )


if __name__ == "__main__":
    main()
