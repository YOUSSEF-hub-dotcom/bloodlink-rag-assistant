"""Conversation and message CRUD in ``rag.conversations`` / ``rag.messages``.

Only user and assistant text is stored here.  Retrieved context, rewrites and
internal diagnostics are deliberately never persisted.  All identifiers are
validated as UUIDs before reaching PostgreSQL, and table names are composed
with ``psycopg.sql.Identifier`` so ``RAG_DB_SCHEMA`` cannot inject SQL.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Sequence

from psycopg import sql

from core.config import get_settings
from core.db import get_pool
from core.logging import get_logger

logger = get_logger(__name__)

Role = Literal["user", "assistant"]
_ROLES = ("user", "assistant")
_CONVERSATION_COLUMNS = (
    "id, user_id, summary, summary_upto_message_id, created_at, updated_at"
)
_MESSAGE_COLUMNS = "id, conversation_id, role, content, created_at"


@dataclass(frozen=True)
class Conversation:
    """One chat conversation and its rolling-summary watermark."""

    id: str
    user_id: str | None
    summary: str | None
    summary_upto_message_id: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Message:
    """One stored chat message (user text or assistant answer only)."""

    id: int
    conversation_id: str
    role: Role
    content: str
    created_at: datetime


def _table(name: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(
        sql.Identifier(get_settings().rag_db_schema), sql.Identifier(name)
    )


def _as_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as error:
        raise ValueError(f"Invalid conversation id: {value!r}") from error


def new_conversation_id() -> str:
    """Return a fresh UUID4 string for a new conversation."""
    return str(uuid.uuid4())


def _conversation_from_row(row: Sequence[Any]) -> Conversation:
    return Conversation(
        id=str(row[0]),
        user_id=row[1],
        summary=row[2],
        summary_upto_message_id=row[3],
        created_at=row[4],
        updated_at=row[5],
    )


def _message_from_row(row: Sequence[Any]) -> Message:
    return Message(
        id=int(row[0]),
        conversation_id=str(row[1]),
        role=str(row[2]),  # type: ignore[arg-type]
        content=str(row[3]),
        created_at=row[4],
    )


def create_conversation(
    conversation_id: str | None = None, *, user_id: str | None = None
) -> Conversation:
    """Create a conversation, or return the existing row on id conflict."""
    conversation_uuid = _as_uuid(conversation_id or new_conversation_id())
    select = sql.SQL("SELECT {cols} FROM {table} WHERE id = %(id)s").format(
        cols=sql.SQL(_CONVERSATION_COLUMNS), table=_table("conversations")
    )
    insert = sql.SQL(
        """
        INSERT INTO {table} (id, user_id)
        VALUES (%(id)s, %(user_id)s)
        ON CONFLICT (id) DO NOTHING
        RETURNING {cols}
        """
    ).format(table=_table("conversations"), cols=sql.SQL(_CONVERSATION_COLUMNS))

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(insert, {"id": conversation_uuid, "user_id": user_id})
            row = cur.fetchone()
            if row is None:
                cur.execute(select, {"id": conversation_uuid})
                row = cur.fetchone()
    assert row is not None
    return _conversation_from_row(row)


def ensure_conversation(
    conversation_id: str | None, *, user_id: str | None = None
) -> Conversation:
    """Return the conversation for ``conversation_id``, creating it if absent."""
    return create_conversation(conversation_id, user_id=user_id)


def get_conversation(conversation_id: str) -> Conversation | None:
    """Fetch one conversation, or ``None`` when it does not exist."""
    statement = sql.SQL("SELECT {cols} FROM {table} WHERE id = %(id)s").format(
        cols=sql.SQL(_CONVERSATION_COLUMNS), table=_table("conversations")
    )
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement, {"id": _as_uuid(conversation_id)})
            row = cur.fetchone()
    return _conversation_from_row(row) if row else None


def add_message(conversation_id: str, role: Role, content: str) -> Message:
    """Append one message and bump the conversation's ``updated_at``."""
    if role not in _ROLES:
        raise ValueError(f"Unsupported message role: {role!r}")
    if not content.strip():
        raise ValueError("message content must not be empty")

    conversation_uuid = _as_uuid(conversation_id)
    insert = sql.SQL(
        """
        INSERT INTO {table} (conversation_id, role, content)
        VALUES (%(cid)s, %(role)s, %(content)s)
        RETURNING {cols}
        """
    ).format(table=_table("messages"), cols=sql.SQL(_MESSAGE_COLUMNS))
    touch = sql.SQL("UPDATE {table} SET updated_at = now() WHERE id = %(cid)s").format(
        table=_table("conversations")
    )

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                insert,
                {"cid": conversation_uuid, "role": role, "content": content},
            )
            row = cur.fetchone()
            cur.execute(touch, {"cid": conversation_uuid})
    assert row is not None
    logger.debug(
        "message_stored",
        extra={
            "conversation_id": str(conversation_uuid),
            "message_id": row[0],
            "role": role,
        },
    )
    return _message_from_row(row)


def list_messages(
    conversation_id: str,
    *,
    after_id: int | None = None,
    limit: int | None = None,
) -> list[Message]:
    """Return messages in ascending id order, optionally after a watermark."""
    conditions = [sql.SQL("conversation_id = %(cid)s")]
    params: dict[str, Any] = {"cid": _as_uuid(conversation_id)}
    if after_id is not None:
        conditions.append(sql.SQL("id > %(after_id)s"))
        params["after_id"] = int(after_id)

    statement = sql.SQL("SELECT {cols} FROM {table} WHERE {where} ORDER BY id ASC").format(
        cols=sql.SQL(_MESSAGE_COLUMNS),
        table=_table("messages"),
        where=sql.SQL(" AND ").join(conditions),
    )
    if limit is not None:
        statement = sql.SQL("{statement} LIMIT %(limit)s").format(statement=statement)
        params["limit"] = int(limit)

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement, params)
            rows = cur.fetchall()
    return [_message_from_row(row) for row in rows]


def count_messages_after(conversation_id: str, after_id: int | None = None) -> int:
    """Count messages not yet covered by a summary watermark."""
    conditions = [sql.SQL("conversation_id = %(cid)s")]
    params: dict[str, Any] = {"cid": _as_uuid(conversation_id)}
    if after_id is not None:
        conditions.append(sql.SQL("id > %(after_id)s"))
        params["after_id"] = int(after_id)

    statement = sql.SQL("SELECT count(*) FROM {table} WHERE {where}").format(
        table=_table("messages"), where=sql.SQL(" AND ").join(conditions)
    )
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement, params)
            row = cur.fetchone()
    assert row is not None
    return int(row[0])


def set_summary(conversation_id: str, summary: str, upto_message_id: int) -> Conversation:
    """Persist a rolling summary and the id of its last covered message."""
    statement = sql.SQL(
        """
        UPDATE {table}
        SET summary = %(summary)s,
            summary_upto_message_id = %(upto)s,
            updated_at = now()
        WHERE id = %(cid)s
        RETURNING {cols}
        """
    ).format(table=_table("conversations"), cols=sql.SQL(_CONVERSATION_COLUMNS))
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                statement,
                {
                    "cid": _as_uuid(conversation_id),
                    "summary": summary,
                    "upto": int(upto_message_id),
                },
            )
            row = cur.fetchone()
    assert row is not None
    return _conversation_from_row(row)


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and its messages (FK cascade)."""
    statement = sql.SQL("DELETE FROM {table} WHERE id = %(cid)s").format(
        table=_table("conversations")
    )
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement, {"cid": _as_uuid(conversation_id)})
            deleted = cur.rowcount > 0
    if deleted:
        logger.info("conversation_deleted", extra={"conversation_id": conversation_id})
    return deleted


__all__ = [
    "Conversation",
    "Message",
    "Role",
    "add_message",
    "count_messages_after",
    "create_conversation",
    "delete_conversation",
    "ensure_conversation",
    "get_conversation",
    "list_messages",
    "new_conversation_id",
    "set_summary",
]
