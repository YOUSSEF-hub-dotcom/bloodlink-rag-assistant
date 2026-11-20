"""Structure-aware Markdown chunking with recursive splitting fallback.

Each H2/H3 section becomes one chunk.  The four screening questions are a
numbered list under a single H2 in the current data files, so question sections
are split into one chunk per numbered item (the spec expects one chunk per
question).  Oversized sections fall back to the bge-m3 tokenizer via
``RecursiveCharacterTextSplitter``; Markdown tables are split by whole rows
with the header repeated, never mid-row.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer

from core.config import get_settings
from core.logging import get_logger
from ingestion.clean import VERIFY_SENTINEL, remove_verify_sentinels
from ingestion.parse import ParsedDocument

logger = get_logger(__name__)

HEADERS_TO_SPLIT: list[tuple[str, str]] = [("#", "h1"), ("##", "h2"), ("###", "h3")]
SEPARATORS: list[str] = ["\n\n", "\n", "۔", ". ", "؟", "? ", "، ", " ", ""]
QUESTION_RE = re.compile(r"أسئلة|سؤال|questions?", re.IGNORECASE)
NUMBERED_ITEM_RE = re.compile(r"^[ \t]*(\d+)\.[ \t]+(.*)$")
BOLD_LABEL_RE = re.compile(r"^\*\*(.+?)\*\*[ \t]*(.*)$", re.DOTALL)
TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
TABLE_SEPARATOR_RE = re.compile(r"^[ \t]*\|[\s:|-]+\|[ \t]*$")
LIST_PREFIX_RE = re.compile(r"^[ \t]*(?:[-*]|\d+\.)[ \t]+")
MIN_TOKENS = 20


@dataclass(frozen=True)
class Chunk:
    """A single embedded unit, including its contextual header line."""

    id: str
    doc_id: str
    language: str
    section_path: str
    chunk_index: int
    content: str
    token_count: int
    content_hash: str
    metadata: dict[str, Any]


@lru_cache(maxsize=2)
def get_tokenizer(local_files_only: bool = False) -> Any:
    """Return the bge-m3 tokenizer (the same one used by the embedder)."""
    return AutoTokenizer.from_pretrained(
        get_settings().embedding_model, local_files_only=local_files_only
    )


def count_tokens(text: str, tokenizer: Any | None = None) -> int:
    """Count tokens of the full text, including special tokens like the splitter."""
    tok = tokenizer or get_tokenizer()
    return len(tok.encode(text))


def _make_chunk(
    *,
    doc: ParsedDocument,
    language: str,
    section_path: str,
    chunk_index: int,
    content: str,
    token_count: int,
    has_marker: bool,
    headings: dict[str, str | None],
) -> Chunk:
    """Build a chunk with deterministic id and content hash."""
    raw_id = f"{doc.doc_id}|{language}|{section_path}|{chunk_index}"
    chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
    metadata: dict[str, Any] = dict(doc.metadata)
    metadata.update(
        {
            "language": language,
            "section_path": section_path,
            "h1": headings.get("h1"),
            "h2": headings.get("h2"),
            "h3": headings.get("h3"),
            "chunk_index": chunk_index,
            "token_count": token_count,
            "has_unverified_markers": has_marker,
        }
    )
    return Chunk(
        id=chunk_id,
        doc_id=doc.doc_id,
        language=language,
        section_path=section_path,
        chunk_index=chunk_index,
        content=content,
        token_count=token_count,
        content_hash=hashlib.sha1(content.encode("utf-8")).hexdigest(),
        metadata=metadata,
    )


def _extract_label(text: str, language: str, number: int) -> str:
    """Get the bold label of a numbered question, or a generic fallback."""
    match = BOLD_LABEL_RE.match(text.strip())
    if match:
        return match.group(1).rstrip(":").strip()
    return f"Question {number}" if language == "en" else f"السؤال {number}"


def _split_trailing_paragraph(lines: list[str]) -> tuple[list[str], list[str]]:
    """Peel off a final explanatory paragraph after a numbered list."""
    for index in range(len(lines) - 1, -1, -1):
        if (
            lines[index].strip() == ""
            and index + 1 < len(lines)
            and lines[index + 1].strip()
            and not LIST_PREFIX_RE.match(lines[index + 1])
        ):
            return lines[:index], lines[index + 1 :]
    return lines, []


def _split_question_items(body: str, language: str) -> list[tuple[str, str]]:
    """Turn a numbered question list into (label, text) pairs."""
    items: list[dict[str, Any]] = []
    prefix: list[str] = []
    for line in body.split("\n"):
        match = NUMBERED_ITEM_RE.match(line)
        if match:
            label = _extract_label(match.group(2), language, len(items) + 1)
            remainder = BOLD_LABEL_RE.match(match.group(2).strip())
            first_line = remainder.group(2).strip() if remainder else match.group(2).strip()
            items.append({"label": label, "lines": [first_line] if first_line else []})
        elif items:
            items[-1]["lines"].append(line)
        else:
            prefix.append(line)

    if len(items) < 2:
        return []

    suffix: list[str] = []
    last_lines = items[-1]["lines"]
    items[-1]["lines"], suffix = _split_trailing_paragraph(last_lines)

    intro = "\n".join(line for line in prefix if line.strip()).strip()
    outro = "\n".join(line for line in suffix if line.strip()).strip()
    result: list[tuple[str, str]] = []
    for index, item in enumerate(items):
        text = "\n".join(item["lines"]).strip()
        if index == 0 and intro:
            text = f"{intro}\n\n{text}"
        if index == len(items) - 1 and outro:
            text = f"{text}\n\n{outro}"
        if text:
            result.append((item["label"], text))
    return result


def _split_text(
    text: str, budget: int, overlap: int, tokenizer: Any
) -> list[str]:
    """Recursively split plain text using the bge-m3 tokenizer."""
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=budget,
        chunk_overlap=min(overlap, max(0, budget - 1)),
        separators=SEPARATORS,
    )
    return [piece.strip() for piece in splitter.split_text(text) if piece.strip()]


def _split_table(table_text: str, budget: int, tokenizer: Any) -> list[str]:
    """Split a Markdown table by row groups, repeating the header each time."""
    lines = [line for line in table_text.split("\n") if line.strip()]
    if len(lines) < 3 or not TABLE_SEPARATOR_RE.match(lines[1]):
        return [table_text]

    header = lines[:2]
    pieces: list[str] = []
    current: list[str] = []
    for row in lines[2:]:
        candidate = "\n".join(header + current + [row])
        if not current or count_tokens(candidate, tokenizer) <= budget:
            current.append(row)
        else:
            pieces.append("\n".join(header + current))
            current = [row]
    if current:
        pieces.append("\n".join(header + current))
    return pieces


def _split_body(
    body: str, budget: int, overlap: int, tokenizer: Any
) -> list[str]:
    """Split oversized body text, isolating Markdown table blocks."""
    blocks: list[tuple[str, list[str]]] = []
    buffer: list[str] = []
    in_table = False
    for line in body.split("\n"):
        is_table = TABLE_ROW_RE.match(line) is not None
        if buffer and is_table != in_table:
            blocks.append(("table" if in_table else "text", buffer))
            buffer = []
        in_table = is_table
        buffer.append(line)
    if buffer:
        blocks.append(("table" if in_table else "text", buffer))

    pieces: list[str] = []
    for kind, block in blocks:
        text = "\n".join(block).strip()
        if not text:
            continue
        if kind == "table":
            pieces.extend(_split_table(text, budget, tokenizer))
        else:
            pieces.extend(_split_text(text, budget, overlap, tokenizer))
    return pieces


def chunk_language_part(
    doc: ParsedDocument,
    language: str,
    markdown: str,
    *,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    tokenizer: Any | None = None,
) -> list[Chunk]:
    """Chunk one cleaned language section of one document."""
    settings = get_settings()
    max_tokens = max_tokens if max_tokens is not None else settings.max_chunk_tokens
    overlap_tokens = overlap_tokens if overlap_tokens is not None else settings.chunk_overlap_tokens
    tok = tokenizer or get_tokenizer()

    sections = MarkdownHeaderTextSplitter(HEADERS_TO_SPLIT).split_text(markdown)
    chunks: list[Chunk] = []
    for section in sections:
        headings = {
            key: section.metadata.get(key) for key in ("h1", "h2", "h3")
        }
        body = section.page_content.strip()
        if not body:
            continue
        base_path = " > ".join(
            value for value in (headings["h1"], headings["h2"], headings["h3"]) if value
        )
        if not base_path:
            continue

        candidates: list[tuple[str, str, dict[str, str | None]]] = []
        is_question = headings["h3"] is None and bool(
            QUESTION_RE.search(str(headings["h2"] or ""))
        )
        question_items = _split_question_items(body, language) if is_question else []
        if question_items:
            for label, item_text in question_items:
                candidates.append(
                    (
                        f"{base_path} > {label}",
                        item_text,
                        {**headings, "h3": label},
                    )
                )
        else:
            candidates.append((base_path, body, headings))

        for section_path, raw_body, section_headings in candidates:
            has_marker = VERIFY_SENTINEL in raw_body
            clean_body = remove_verify_sentinels(raw_body).strip()
            if not clean_body:
                continue

            pieces = [clean_body]
            header = f"{section_path}\n"
            if count_tokens(header + clean_body, tok) > max_tokens:
                budget = max(32, max_tokens - count_tokens(header, tok))
                pieces = _split_body(clean_body, budget, overlap_tokens, tok)

            for piece in pieces:
                content = f"{section_path}\n{piece}"
                token_count = count_tokens(content, tok)
                chunk = _make_chunk(
                    doc=doc,
                    language=language,
                    section_path=section_path,
                    chunk_index=len(chunks),
                    content=content,
                    token_count=token_count,
                    has_marker=has_marker,
                    headings=section_headings,
                )
                if token_count < MIN_TOKENS or token_count > max_tokens:
                    logger.warning(
                        "chunk_size_out_of_range",
                        extra={
                            "chunk_id": chunk.id,
                            "doc_id": doc.doc_id,
                            "language": language,
                            "section_path": section_path,
                            "token_count": token_count,
                            "max_tokens": max_tokens,
                        },
                    )
                chunks.append(chunk)
    return chunks


def chunk_document(
    doc: ParsedDocument,
    cleaned_parts: Sequence[tuple[str, str]],
    *,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    tokenizer: Any | None = None,
) -> list[Chunk]:
    """Chunk every cleaned language part of a parsed document."""
    chunks: list[Chunk] = []
    for language, markdown in cleaned_parts:
        chunks.extend(
            chunk_language_part(
                doc,
                language,
                markdown,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                tokenizer=tokenizer,
            )
        )
    return chunks
