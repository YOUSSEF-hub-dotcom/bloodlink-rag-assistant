"""Hybrid retrieval: dense pgvector cosine + sparse full-text, fused with RRF.

The dense cosine score remains the threshold gate and the public
``SearchResult.score`` value.  Sparse keyword matches (PostgreSQL ``tsvector``
with the ``'simple'`` config, Arabic normalization, and exact blood-type boosts)
only re-rank dense candidates that already passed ``SCORE_THRESHOLD``.  This
keeps out-of-scope queries at zero results with zero LLM calls.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from pgvector import Vector
from psycopg import sql

from core.config import get_settings
from core.db import get_pool
from core.logging import get_logger

logger = get_logger(__name__)

RRF_K = 60
EXACT_BLOOD_TYPE_BOOST = 0.5
BLOOD_TYPE_RE = re.compile(r"(?<![A-Za-z])(AB|A|B|O)\s*([+-])", re.IGNORECASE)

# Tatweel, alef variants, short vowels/sukun, final ya/taa-marbuta -> normalized.
# The sentinel keeps translate() positional; it is removed immediately after.
NORMALIZE_FROM = "ـآأإًٌٍَُِّْىة"
NORMALIZE_TO = "\x01ااا\x01\x01\x01\x01\x01\x01\x01\x01يه"
NORMALIZE_SENTINEL = "\x01"
assert len(NORMALIZE_FROM) == len(NORMALIZE_TO)


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk; ``score`` stays the dense cosine similarity."""

    id: str
    content: str
    metadata: dict[str, Any]
    section_path: str
    language: str
    score: float
    rrf_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None

    @property
    def dense_score(self) -> float:
        """Explicit alias for the threshold-anchored dense score."""
        return self.score


def normalize_fts_text(text: str) -> str:
    """Fold Arabic variants and Unicode minus for sparse keyword matching."""
    translated = text.replace("\u2212", "-").translate(
        str.maketrans(NORMALIZE_FROM, NORMALIZE_TO)
    )
    return translated.replace(NORMALIZE_SENTINEL, "")


def extract_blood_types(text: str) -> list[str]:
    """Return unique blood types such as ``O-`` found in ``text``."""
    normalized = normalize_fts_text(text)
    found = (
        f"{match.group(1).upper()}{match.group(2)}"
        for match in BLOOD_TYPE_RE.finditer(normalized)
    )
    return list(dict.fromkeys(found))


def _quote_ts_term(term: str) -> str:
    """Quote one term for ``to_tsquery`` without allowing query injection."""
    return "'" + term.replace("'", "''") + "'"


def _build_tsquery(query_text: str) -> str:
    """Build a safe OR tsquery so partial keyword matches still contribute."""
    normalized = normalize_fts_text(query_text)
    terms = [token.lower() for token in re.split(r"[^\w+]+", normalized, flags=re.UNICODE) if token]
    terms.extend(blood_type.lower() for blood_type in extract_blood_types(normalized))
    unique_terms = list(dict.fromkeys(terms))
    if not unique_terms:
        return ""
    return " | ".join(_quote_ts_term(term) for term in unique_terms)


def _chunks_table() -> sql.Composed:
    settings = get_settings()
    return sql.SQL("{}.{}").format(
        sql.Identifier(settings.rag_db_schema), sql.Identifier("chunks")
    )


def _normalized_content_sql() -> sql.Composed:
    """SQL expression applying the same Arabic normalization to chunk text."""
    return sql.SQL(
        "replace(translate(content, %(norm_from)s, %(norm_to)s), %(sentinel)s, '')"
    )


def _fetch_candidates(
    vector: Sequence[float],
    language: str | None,
    top_k: int,
    ef_search: int,
    pool: Any,
) -> list[SearchResult]:
    """Run the dense pgvector leg, optionally restricted to a language."""
    where = sql.SQL("WHERE language = %(lang)s") if language else sql.SQL("")
    statement = sql.SQL(
        """
        -- dense pgvector leg
        SELECT id, content, metadata, section_path, language,
               1 - (embedding <=> %(q)s) AS dense_score
        FROM {table}
        {where}
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
        """
    ).format(table=_chunks_table(), where=where)
    params: dict[str, Any] = {"q": Vector(list(vector)), "k": top_k}
    if language:
        params["lang"] = language

    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Must be inside the same transaction as the dense SELECT.
            cur.execute(sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(ef_search)))
            cur.execute(statement, params)
            rows = cur.fetchall()

    results: list[SearchResult] = []
    for row in rows:
        metadata = row[2] if isinstance(row[2], dict) else json.loads(row[2])
        results.append(
            SearchResult(
                id=str(row[0]),
                content=str(row[1]),
                metadata=metadata,
                section_path=str(row[3]),
                language=str(row[4]),
                score=float(row[5]),
            )
        )
    return results


def _contains_exact_blood_type(content: str, blood_type: str) -> bool:
    """Check an exact standalone blood type, so O- does not match O+."""
    pattern = rf"(?<![A-Za-z0-9]){re.escape(blood_type)}(?![A-Za-z0-9])"
    return re.search(pattern, content, re.IGNORECASE) is not None


def _fetch_sparse_scores(
    query_text: str, language: str | None, pool: Any
) -> dict[str, float]:
    """Rank chunks with native full-text search plus exact keyword boosts."""
    tsquery = _build_tsquery(query_text)
    if not tsquery:
        return {}

    normalized_content = _normalized_content_sql()
    match = sql.SQL(
        "to_tsvector('simple', {content}) @@ to_tsquery('simple', %(tsq)s)"
    ).format(content=normalized_content)
    conditions = [match]
    params: dict[str, Any] = {
        "tsq": tsquery,
        "norm_from": NORMALIZE_FROM,
        "norm_to": NORMALIZE_TO,
        "sentinel": NORMALIZE_SENTINEL,
    }
    if language:
        conditions.append(sql.SQL("language = %(lang)s"))
        params["lang"] = language

    statement = sql.SQL(
        """
        -- sparse full-text leg
        SELECT id, content,
               ts_rank_cd(
                   to_tsvector('simple', {content}),
                   to_tsquery('simple', %(tsq)s)
               ) AS sparse_score
        FROM {table}
        WHERE {where}
        """
    ).format(
        content=normalized_content,
        table=_chunks_table(),
        where=sql.SQL(" AND ").join(conditions),
    )

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(statement, params)
            rows = cur.fetchall()

    blood_types = extract_blood_types(query_text)
    scores: dict[str, float] = {}
    for row in rows:
        doc_id = str(row[0])
        content = normalize_fts_text(str(row[1]))
        boost = sum(
            EXACT_BLOOD_TYPE_BOOST
            for blood_type in blood_types
            if _contains_exact_blood_type(content, blood_type)
        )
        scores[doc_id] = float(row[2]) + boost

    logger.debug(
        "sparse_retrieval_complete",
        extra={"hits": len(scores), "language": language, "rrf_k": RRF_K},
    )
    return scores


def _rrf(rank: int) -> float:
    """Reciprocal Rank Fusion contribution for a 1-based rank."""
    return 1.0 / (RRF_K + rank)


def _fuse_results(
    dense: Sequence[SearchResult], sparse_scores: dict[str, float], top_k: int
) -> list[SearchResult]:
    """Fuse dense and sparse ranks with RRF, then cap at ``top_k``."""
    dense_sorted = sorted(dense, key=lambda result: (-result.score, result.id))
    sparse_ordered = sorted(sparse_scores.items(), key=lambda item: (-item[1], item[0]))
    sparse_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(sparse_ordered, 1)}

    fused: list[SearchResult] = []
    for dense_rank, result in enumerate(dense_sorted, 1):
        sparse_rank = sparse_ranks.get(result.id)
        rrf_score = _rrf(dense_rank)
        if sparse_rank is not None:
            rrf_score += _rrf(sparse_rank)
        fused.append(
            replace(
                result,
                rrf_score=rrf_score,
                sparse_rank=sparse_rank,
                sparse_score=sparse_scores.get(result.id),
            )
        )

    fused.sort(
        key=lambda result: (
            -(result.rrf_score or 0.0),
            -result.score,
            result.id,
        )
    )
    return fused[:top_k]


def search_by_vector(
    vector: Sequence[float],
    language: str,
    *,
    query: str | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
    ef_search: int | None = None,
    connection_factory: Callable[[], Any] | None = None,
) -> list[SearchResult]:
    """Search with an already computed query vector (testable without a model).

    ``query`` enables the sparse leg; callers that only have a vector still get
    dense RRF scores and the same ``SearchResult`` shape.
    """
    settings = get_settings()
    top_k = top_k if top_k is not None else settings.top_k
    score_threshold = (
        score_threshold if score_threshold is not None else settings.score_threshold
    )
    ef_search = ef_search if ef_search is not None else settings.hnsw_ef_search
    pool = connection_factory() if connection_factory else get_pool()

    candidates = _fetch_candidates(vector, language, top_k, ef_search, pool)
    best = max((result.score for result in candidates), default=float("-inf"))

    fallback_used = False
    sparse_scope = language
    if best < score_threshold:
        fallback_used = True
        sparse_scope = None
        logger.info(
            "cross_lingual_fallback",
            extra={"language": language, "best_score": round(best, 4), "threshold": score_threshold},
        )
        candidates = _fetch_candidates(vector, None, top_k, ef_search, pool)

    # The dense cosine score is the hard gate; sparse never adds sub-threshold rows.
    passing = [result for result in candidates if result.score >= score_threshold]

    sparse_scores: dict[str, float] = {}
    if query and passing:
        sparse_scores = _fetch_sparse_scores(query, sparse_scope, pool)

    results = _fuse_results(passing, sparse_scores, top_k)
    logger.info(
        "retrieval_complete",
        extra={
            "strategy": "hybrid_rrf",
            "rrf_k": RRF_K,
            "language": language,
            "fallback_used": fallback_used,
            "chunks": len(results),
            "top_score": round(results[0].score, 4) if results else None,
            "top_rrf_score": round(results[0].rrf_score or 0.0, 6) if results else None,
            "sparse_hits": len(sparse_scores),
            "top_k": top_k,
            "threshold": score_threshold,
        },
    )
    return results


def search(
    query: str,
    language: str,
    *,
    embedder: Any | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
    ef_search: int | None = None,
    connection_factory: Callable[[], Any] | None = None,
) -> list[SearchResult]:
    """Embed ``query`` and return hybrid, thresholded results for ``language``."""
    from core.embedder import get_embedder

    model = embedder or get_embedder()
    vector = model.embed_query(query)
    return search_by_vector(
        vector,
        language,
        query=query,
        top_k=top_k,
        score_threshold=score_threshold,
        ef_search=ef_search,
        connection_factory=connection_factory,
    )
