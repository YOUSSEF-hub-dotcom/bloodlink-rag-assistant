"""CLI entry point: ``python -m ingestion.run [--reset]``.

Pipeline: bootstrap schema → parse → clean → chunk → diff hashes → embed only
new/changed chunks → upsert → delete stale rows.  Re-running without data or
code changes embeds zero chunks.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

from core.db import bootstrap_schema, close_pool
from core.logging import get_logger
from ingestion.chunk import Chunk, chunk_document, get_tokenizer
from ingestion.clean import clean_markdown
from ingestion.embed import embed_texts
from ingestion.parse import load_documents
from ingestion.store import delete_stale_chunks, fetch_existing_hashes, truncate_chunks, upsert_chunks

logger = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest BloodLink knowledge-base Markdown into pgvector.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Truncate rag.chunks and re-ingest everything.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing .md files (default: repository data/).",
    )
    return parser.parse_args(argv)


def run_ingestion(*, reset: bool = False, data_dir: Path | None = None) -> dict[str, object]:
    """Execute the full ingestion pipeline and return a summary dict."""
    started = time.perf_counter()
    bootstrap_schema()

    documents = load_documents(data_dir)
    tokenizer = get_tokenizer()

    chunks_by_doc: dict[str, list[Chunk]] = defaultdict(list)
    all_chunks: list[Chunk] = []
    for doc in documents:
        cleaned_parts = [(part.language, clean_markdown(part.markdown)[0]) for part in doc.parts]
        doc_chunks = chunk_document(doc, cleaned_parts, tokenizer=tokenizer)
        chunks_by_doc[doc.doc_id].extend(doc_chunks)
        all_chunks.extend(doc_chunks)

    if reset:
        truncate_chunks()
        existing_hashes: dict[str, str] = {}
    else:
        existing_hashes = fetch_existing_hashes()

    changed = [
        chunk
        for chunk in all_chunks
        if existing_hashes.get(chunk.id) != chunk.content_hash
    ]
    skipped = len(all_chunks) - len(changed)

    if changed:
        embeddings = embed_texts([chunk.content for chunk in changed])
        upsert_chunks(changed, embeddings)

    deleted_total = 0
    for doc_id, doc_chunks in chunks_by_doc.items():
        deleted_total += delete_stale_chunks(doc_id, [chunk.id for chunk in doc_chunks])

    duration_seconds = time.perf_counter() - started
    per_language = dict(Counter(chunk.language for chunk in all_chunks).most_common())
    per_file = {
        doc.source_file: {
            "doc_id": doc.doc_id,
            "chunks": len(chunks_by_doc[doc.doc_id]),
            "ar": sum(1 for chunk in chunks_by_doc[doc.doc_id] if chunk.language == "ar"),
            "en": sum(1 for chunk in chunks_by_doc[doc.doc_id] if chunk.language == "en"),
        }
        for doc in documents
    }

    summary: dict[str, object] = {
        "files": len(documents),
        "total_chunks": len(all_chunks),
        "chunks_per_language": per_language,
        "chunks_per_file": per_file,
        "embedded": len(changed),
        "skipped": skipped,
        "deleted": deleted_total,
        "total_tokens": sum(chunk.token_count for chunk in all_chunks),
        "duration_seconds": round(duration_seconds, 3),
        "reset": reset,
    }

    print("=== Ingestion summary ===")
    print(f"files: {summary['files']}")
    print(f"total_chunks: {summary['total_chunks']}")
    print(f"chunks_per_language: {summary['chunks_per_language']}")
    print("chunks_per_file:")
    for source_file, info in per_file.items():
        print(
            f"  - {source_file} (doc_id={info['doc_id']}): "
            f"{info['chunks']} chunks [ar={info['ar']}, en={info['en']}]"
        )
    print(f"chunks embedded: {summary['embedded']}")
    print(f"chunks skipped (unchanged): {summary['skipped']}")
    print(f"stale chunks deleted: {summary['deleted']}")
    print(f"total tokens: {summary['total_tokens']}")
    print(f"duration_seconds: {summary['duration_seconds']}")
    print(f"reset: {summary['reset']}")

    return summary


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        run_ingestion(reset=args.reset, data_dir=args.data_dir)
    finally:
        close_pool()


if __name__ == "__main__":
    main()
