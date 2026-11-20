"""Unit tests for hybrid retrieval (fake DB, fake embedder; no GPU/network)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from retrieval.debug import format_result, main as debug_main
from retrieval.search import (
    RRF_K,
    SearchResult,
    _build_tsquery,
    extract_blood_types,
    normalize_fts_text,
    search,
    search_by_vector,
)

CALL_LANG = "dense_lang"
SPARSE_LANG = "sparse_lang"
SET_LOCAL = "set_local"


def _row(
    score: float,
    language: str,
    section: str = "Section",
    doc_id: str = "01_test",
    chunk_id: str | None = None,
) -> tuple:
    return (
        chunk_id or f"id-{doc_id}-{section}-{score}",
        f"{section}\nbody text with details about the question",
        {"doc_id": doc_id, "section_path": section, "language": language},
        section,
        language,
        score,
    )


def _sparse_row(chunk_id: str, score: float, content: str = "keyword body") -> tuple:
    return (chunk_id, content, score)


class FakeCursor:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def execute(self, statement: Any, params: dict[str, Any] | None = None) -> None:
        text = str(statement)
        if "hnsw.ef_search" in text:
            self.state["events"].append((SET_LOCAL, text))
            return

        language = params.get("lang") if params else None
        if "-- sparse full-text leg" in text:
            self.state["events"].append((SPARSE_LANG, language))
            self.state["last_sparse_query"] = text
            self.state["rows"] = self.state["sparse_by_language"].get(language, [])
            return

        self.state["events"].append((CALL_LANG, language))
        self.state["last_query"] = text
        self.state["rows"] = (
            self.state["by_language"].get(language, []) if language
            else self.state["cross_lingual"]
        )

    def fetchall(self) -> list[tuple]:
        return self.state["rows"]


class FakeConnection:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.state)


class FakePool:
    def __init__(
        self,
        by_language: dict[str, list[tuple]],
        cross_lingual: list[tuple],
        sparse_by_language: dict[str | None, list[tuple]] | None = None,
    ) -> None:
        self.state: dict[str, Any] = {
            "by_language": by_language,
            "cross_lingual": cross_lingual,
            "sparse_by_language": sparse_by_language or {},
            "events": [],
            "rows": [],
            "last_query": "",
            "last_sparse_query": "",
        }

    def connection(self) -> FakeConnection:
        return FakeConnection(self.state)

    @property
    def calls(self) -> list[str | None]:
        return [value for kind, value in self.state["events"] if kind == CALL_LANG]

    @property
    def sparse_calls(self) -> list[str | None]:
        return [value for kind, value in self.state["events"] if kind == SPARSE_LANG]


@dataclass
class FakeEmbedder:
    vector: list[float] = field(default_factory=lambda: [0.1] * 1024)
    calls: list[str] = field(default_factory=list)

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vector


def test_language_filter_threshold_and_no_fallback() -> None:
    pool = FakePool({"ar": [_row(0.80, "ar", "A"), _row(0.50, "ar", "B"), _row(0.30, "ar", "C")]}, [])
    results = search_by_vector([0.1] * 1024, "ar", top_k=5, score_threshold=0.45,
                               ef_search=40, connection_factory=lambda: pool)
    assert [round(r.score, 2) for r in results] == [0.80, 0.50]
    assert pool.calls == ["ar"]
    assert "<=>" in pool.state["last_query"]
    assert "1 - (embedding <" in pool.state["last_query"]


def test_cross_lingual_fallback_once_when_best_below_threshold() -> None:
    pool = FakePool(
        {"ar": [_row(0.20, "ar", "WeakAr")]},
        [_row(0.10, "ar", "WeakAr"), _row(0.72, "en", "StrongEn")],
    )
    results = search_by_vector([0.1] * 1024, "ar", top_k=5, score_threshold=0.45,
                               ef_search=40, connection_factory=lambda: pool)
    assert pool.calls == ["ar", None]
    assert [(r.language, r.score) for r in results] == [("en", 0.72)]


def test_no_results_when_fallback_also_below_threshold() -> None:
    pool = FakePool({"en": [_row(0.10, "en", "Weak")]}, [_row(0.10, "en", "Weak")])
    results = search_by_vector([0.1] * 1024, "en", top_k=5, score_threshold=0.45,
                               ef_search=40, connection_factory=lambda: pool)
    assert results == []
    assert pool.calls == ["en", None]
    # Sparse must not run when the dense gate rejects every candidate.
    assert pool.sparse_calls == []


def test_hnsw_ef_search_is_set_local_inside_transaction() -> None:
    pool = FakePool({"en": [_row(0.9, "en", "A")]}, [])
    search_by_vector([0.1] * 1024, "en", top_k=3, score_threshold=0.45,
                     ef_search=77, connection_factory=lambda: pool)
    set_events = [text for kind, text in pool.state["events"] if kind == SET_LOCAL]
    assert len(set_events) == 1
    assert "SET LOCAL hnsw.ef_search" in set_events[0]
    assert "77" in set_events[0]


def test_search_uses_shared_embedder_and_debug_zero_threshold() -> None:
    embedder = FakeEmbedder()
    pool = FakePool({"en": [_row(0.2, "en", "Low")]}, [])
    results = search("warfarin?", "en", embedder=embedder, top_k=2, score_threshold=0.0,
                     ef_search=40, connection_factory=lambda: pool)
    assert embedder.calls == ["warfarin?"]
    assert len(results) == 1  # debug threshold of 0.0 keeps low-scoring rows
    assert pool.calls == ["en"]
    assert pool.sparse_calls == ["en"]
    assert "ts_rank_cd" in pool.state["last_sparse_query"]


def test_hybrid_rrf_ranking_promotes_sparse_keyword_match() -> None:
    a = _row(0.80, "en", "A", chunk_id="a")
    b = _row(0.70, "en", "B", chunk_id="b")
    c = _row(0.75, "en", "C", chunk_id="c")
    pool = FakePool(
        {"en": [a, b, c]},
        [],
        {
            "en": [
                _sparse_row("c", 0.40),  # sparse rank 1 -> promoted above dense #1
                _sparse_row("b", 0.30),  # sparse rank 2
                _sparse_row("a", 0.20),  # sparse rank 3
            ]
        },
    )

    results = search_by_vector(
        [0.1] * 1024,
        "en",
        query="exact keyword",
        top_k=3,
        score_threshold=0.45,
        ef_search=40,
        connection_factory=lambda: pool,
    )

    assert [result.id for result in results] == ["c", "a", "b"]
    # Public score stays dense even though ordering is by RRF.
    assert [round(result.score, 2) for result in results] == [0.75, 0.80, 0.70]
    assert [(r.sparse_rank, r.sparse_score) for r in results] == [
        (1, 0.40), (3, 0.20), (2, 0.30)
    ]
    assert results[0].dense_score == results[0].score
    expected_c = 1.0 / (RRF_K + 2) + 1.0 / (RRF_K + 1)
    assert abs(results[0].rrf_score - expected_c) < 1e-9


def test_exact_blood_type_boost_breaks_o_minus_vs_o_plus() -> None:
    assert extract_blood_types("دم O− و O+ و A-") == ["O-", "O+", "A-"]
    assert normalize_fts_text("أإآـ") == "ااا"
    tsquery = _build_tsquery("دم O-")
    assert " | " in tsquery
    assert "'o-'" in tsquery

    o_plus = _row(0.90, "ar", "O+", chunk_id="plus")
    o_minus = _row(0.88, "ar", "O-", chunk_id="minus")
    generic = _row(0.86, "ar", "General", chunk_id="generic")
    pool = FakePool(
        {"ar": [o_plus, o_minus, generic]},
        [],
        {
            "ar": [
                _sparse_row("minus", 0.05, "O- فقط"),
                _sparse_row("generic", 0.10, "دم عام"),
                _sparse_row("plus", 0.05, "O+ فقط"),
            ]
        },
    )

    results = search_by_vector(
        [0.1] * 1024,
        "ar",
        query="من يستطيع استقبال دم O-؟",
        top_k=3,
        score_threshold=0.45,
        ef_search=40,
        connection_factory=lambda: pool,
    )

    assert [result.id for result in results] == ["minus", "plus", "generic"]
    assert results[0].sparse_rank == 1
    # 0.05 base + 0.5 exact-match boost for the literal O-.
    assert abs(results[0].sparse_score - 0.55) < 1e-9
    assert results[1].sparse_score == 0.05  # O+ gets no O- boost


def test_format_result_preview_and_debug_cli(monkeypatch: Any, capsys: Any) -> None:
    result = SearchResult(
        "id1",
        "Header\nA long body " * 30,
        {},
        "Section Path",
        "ar",
        0.81234,
        rrf_score=0.03123,
        sparse_rank=2,
        sparse_score=0.0987,
    )
    formatted = format_result(1, result, preview_chars=40)
    assert "dense_score=0.8123" in formatted
    assert "sparse_rank=2" in formatted
    assert "sparse_score=0.0987" in formatted
    assert "rrf_score=0.031230" in formatted
    assert "language=ar" in formatted
    assert "section_path: Section Path" in formatted
    assert "..." in formatted
    assert len(formatted.splitlines()) == 3

    def fake_search(query: str, language: str, **kwargs: Any) -> list[SearchResult]:
        assert query == "test question"
        assert language == "en"
        assert kwargs["score_threshold"] == 0.0
        return [result]

    monkeypatch.setattr("retrieval.debug.search", fake_search)
    debug_main(["test", "question", "--lang", "en", "--k", "3"])
    out = capsys.readouterr().out
    assert "query: test question" in out
    assert "threshold: ignored (debug)" in out
    assert "strategy: hybrid_rrf (k=60)" in out
    assert "dense_score=0.8123" in out
