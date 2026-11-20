"""PostgreSQL connection pool and idempotent ``rag`` schema bootstrap.

All application tables live in the dedicated schema configured by
``RAG_DB_SCHEMA`` (default ``rag``) so that Prisma's ``db push`` on ``public``
cannot touch them.  The pgvector type adapter is registered on every pooled
connection via the pool's ``configure`` callback.
"""

from __future__ import annotations

import threading
from typing import Any

from pgvector.psycopg import register_vector
from psycopg import Connection, connect, sql
from psycopg_pool import ConnectionPool

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def _configure_connection(conn: Connection) -> None:
    """Register the vector type adapter on a freshly created connection."""
    register_vector(conn)


def _ensure_vector_extension() -> None:
    """Create pgvector before opening the pool: its adapter needs the type to exist."""
    with connect(get_settings().database_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


def get_pool() -> ConnectionPool:
    """Return the process-wide synchronous connection pool."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                settings = get_settings()
                _ensure_vector_extension()
                _pool = ConnectionPool(
                    conninfo=settings.database_url,
                    min_size=1,
                    max_size=8,
                    configure=_configure_connection,
                    open=True,
                )
                _pool.wait()
                logger.info("database_pool_ready", extra={"schema": settings.rag_db_schema})
    return _pool


def close_pool() -> None:
    """Close the pool (used on shutdown and by CLI scripts)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def _qualified(table: str) -> sql.Composed:
    """Build a schema-qualified, safely quoted table identifier."""
    return sql.SQL("{}.{}").format(
        sql.Identifier(get_settings().rag_db_schema),
        sql.Identifier(table),
    )


def bootstrap_schema() -> None:
    """Create the extension, schema, tables and indexes if missing."""
    schema = sql.Identifier(get_settings().rag_db_schema)
    chunks = _qualified("chunks")
    conversations = _qualified("conversations")
    messages = _qualified("messages")

    statements: list[sql.Composable] = [
        sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"),
        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema),
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {chunks} (
              id            TEXT PRIMARY KEY,
              doc_id        TEXT NOT NULL,
              language      TEXT NOT NULL CHECK (language IN ('ar','en')),
              section_path  TEXT NOT NULL,
              chunk_index   INT  NOT NULL,
              content       TEXT NOT NULL,
              token_count   INT  NOT NULL,
              content_hash  TEXT NOT NULL,
              metadata      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
              embedding     vector(1024) NOT NULL,
              created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        ).format(chunks=chunks),
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
              ON {chunks} USING hnsw (embedding vector_cosine_ops)
              WITH (m = 16, ef_construction = 64)
            """
        ).format(chunks=chunks),
        sql.SQL("CREATE INDEX IF NOT EXISTS chunks_language_idx ON {chunks} (language)").format(
            chunks=chunks
        ),
        sql.SQL("CREATE INDEX IF NOT EXISTS chunks_doc_idx ON {chunks} (doc_id)").format(chunks=chunks),
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {conversations} (
              id          UUID PRIMARY KEY,
              user_id     TEXT,
              summary     TEXT,
              summary_upto_message_id BIGINT,
              created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        ).format(conversations=conversations),
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {messages} (
              id               BIGSERIAL PRIMARY KEY,
              conversation_id  UUID NOT NULL REFERENCES {conversations}(id) ON DELETE CASCADE,
              role             TEXT NOT NULL CHECK (role IN ('user','assistant')),
              content          TEXT NOT NULL,
              created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        ).format(messages=messages, conversations=conversations),
        sql.SQL("CREATE INDEX IF NOT EXISTS messages_conv_idx ON {messages} (conversation_id, id)").format(
            messages=messages
        ),
    ]

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)

    logger.info(
        "database_schema_ready",
        extra={"schema": get_settings().rag_db_schema, "tables": ["chunks", "conversations", "messages"]},
    )


def check_database() -> dict[str, Any]:
    """Return connectivity and schema facts for verification / health checks."""
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), version()")
            db_row = cur.fetchone()
            assert db_row is not None
            database, version_string = db_row

            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            extension_row = cur.fetchone()
            pgvector_version = extension_row[0] if extension_row else "NOT INSTALLED"

            cur.execute(sql.SQL("SELECT count(*) FROM {}").format(_qualified("chunks")))
            count_row = cur.fetchone()
            assert count_row is not None
            chunk_count = count_row[0]

    return {
        "database": database,
        "server_version": version_string.split(",")[0],
        "pgvector_version": pgvector_version,
        "schema": get_settings().rag_db_schema,
        "chunks": chunk_count,
    }
