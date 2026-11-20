"""Persist chunk rows in ``rag.chunks`` (idempotent upsert + stale deletion)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pgvector import Vector
from psycopg import sql
from psycopg.types.json import Json

from core.config import get_settings
from core.db import get_pool
from core.logging import get_logger
from ingestion.chunk import Chunk

logger = get_logger(__name__)


def _chunks_table() -> sql.Composed:
    return sql.SQL("{}.{}").format(
        sql.Identifier(get_settings().rag_db_schema), sql.Identifier("chunks")
    )


def fetch_existing_hashes() -> dict[str, str]:
    """Return ``{chunk_id: content_hash}`` for every stored chunk."""
    query = sql.SQL("SELECT id, content_hash FROM {}").format(_chunks_table())
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return {row[0]: row[1] for row in cur.fetchall()}


def upsert_chunks(chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
    """Insert or update changed chunks together with their embeddings."""
    if not chunks:
        return 0
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    statement = sql.SQL(
        """
        INSERT INTO {table} (
            id, doc_id, language, section_path, chunk_index,
            content, token_count, content_hash, metadata, embedding, updated_at
        ) VALUES (
            %(id)s, %(doc_id)s, %(language)s, %(section_path)s, %(chunk_index)s,
            %(content)s, %(token_count)s, %(content_hash)s, %(metadata)s,
            %(embedding)s, now()
        )
        ON CONFLICT (id) DO UPDATE SET
            doc_id = EXCLUDED.doc_id,
            language = EXCLUDED.language,
            section_path = EXCLUDED.section_path,
            chunk_index = EXCLUDED.chunk_index,
            content = EXCLUDED.content,
            token_count = EXCLUDED.token_count,
            content_hash = EXCLUDED.content_hash,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding,
            updated_at = now()
        """
    ).format(table=_chunks_table())

    params = [
        {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "language": chunk.language,
            "section_path": chunk.section_path,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "token_count": chunk.token_count,
            "content_hash": chunk.content_hash,
            "metadata": Json(chunk.metadata),
            "embedding": Vector(list(vector)),
        }
        for chunk, vector in zip(chunks, embeddings, strict=True)
    ]

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(statement, params)

    logger.info("chunks_upserted", extra={"count": len(chunks)})
    return len(chunks)


def delete_stale_chunks(doc_id: str, keep_ids: Iterable[str]) -> int:
    """Delete rows for ``doc_id`` whose ids are no longer produced."""
    keep = list(keep_ids)
    if keep:
        statement = sql.SQL(
            "DELETE FROM {table} WHERE doc_id = %(doc_id)s AND NOT (id = ANY(%(keep)s))"
        ).format(table=_chunks_table())
        params = {"doc_id": doc_id, "keep": keep}
    else:
        statement = sql.SQL("DELETE FROM {table} WHERE doc_id = %(doc_id)s").format(
            table=_chunks_table()
        )
        params = {"doc_id": doc_id}

    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement, params)
            deleted = cur.rowcount or 0

    if deleted:
        logger.info("stale_chunks_deleted", extra={"doc_id": doc_id, "deleted": deleted})
    return deleted


def truncate_chunks() -> None:
    """Remove every row from ``rag.chunks`` (used by ``--reset``)."""
    statement = sql.SQL("TRUNCATE TABLE {}").format(_chunks_table())
    with get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement)
    logger.info("chunks_truncated")
