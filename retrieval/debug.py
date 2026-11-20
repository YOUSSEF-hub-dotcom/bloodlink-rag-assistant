"""Manual hybrid-retrieval tuning CLI: ``python -m retrieval.debug "<question>"``.

Prints the top-k chunks with dense score, sparse rank/score, RRF score, section
path, language and a text preview.  The production ``SCORE_THRESHOLD`` is
ignored here so ``TOP_K`` / ``SCORE_THRESHOLD`` can be tuned manually.
"""

from __future__ import annotations

import argparse
import sys

from core.config import get_settings
from core.logging import get_logger
from ingestion.parse import detect_language
from retrieval.search import RRF_K, SearchResult, search

logger = get_logger(__name__)


def format_result(rank: int, result: SearchResult, preview_chars: int = 180) -> str:
    """Render one hybrid result as a compact, human-readable block."""
    preview = " ".join(result.content.split())
    if len(preview) > preview_chars:
        preview = preview[:preview_chars].rstrip() + " ..."
    sparse_rank = "-" if result.sparse_rank is None else str(result.sparse_rank)
    sparse_score = "-" if result.sparse_score is None else f"{result.sparse_score:.4f}"
    rrf_score = result.rrf_score if result.rrf_score is not None else 1.0 / (RRF_K + rank)
    return (
        f"[{rank}] dense_score={result.dense_score:.4f}  "
        f"sparse_rank={sparse_rank}  sparse_score={sparse_score}  "
        f"rrf_score={rrf_score:.6f}  language={result.language}\n"
        f"    section_path: {result.section_path}\n"
        f"    preview: {preview}"
    )


def main(argv: list[str] | None = None) -> None:
    """Print hybrid retrieval diagnostics for one query."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Debug hybrid retrieval for a question.")
    parser.add_argument("question", nargs="+", help="Question text (quote it).")
    parser.add_argument("--lang", choices=("ar", "en"), default=None,
                        help="Force a language (default: detect from the question).")
    parser.add_argument("--k", type=int, default=settings.top_k,
                        help=f"Top-k to display (default: {settings.top_k}).")
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip()
    language = args.lang or detect_language(question)

    results = search(
        question,
        language,
        top_k=args.k,
        score_threshold=0.0,  # debug ignores the production threshold
    )

    print("=" * 100)
    print(f"query: {question}")
    print(
        f"language: {language} | k: {args.k} | strategy: hybrid_rrf (k={RRF_K}) "
        f"| threshold: ignored (debug)"
    )
    print(f"configured TOP_K: {settings.top_k} | SCORE_THRESHOLD: {settings.score_threshold}")
    print("-" * 100)
    if not results:
        print("no results")
    for rank, result in enumerate(results, start=1):
        print(format_result(rank, result))
    print("=" * 100)


if __name__ == "__main__":
    main(sys.argv[1:])
