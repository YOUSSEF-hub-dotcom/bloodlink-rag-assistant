"""Read the Markdown knowledge base and split each file by language.

The language split is driven by the top-level (``#``) headings: their suffix
``(العربية)`` / ``(English)`` is authoritative, with an Arabic-character ratio
fallback for headings without a suffix.  Front-matter is consumed here and
never becomes part of chunk text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import frontmatter

from core.logging import get_logger

logger = get_logger(__name__)

H1_RE = re.compile(r"^#[ \t]+(.+?)[ \t]*$", re.MULTILINE)
AR_SUFFIX = "(العربية)"
EN_SUFFIX = "(English)"
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06ff]")


@dataclass(frozen=True)
class LanguagePart:
    """One language section of a document, including its H1 heading."""

    language: str
    markdown: str


@dataclass(frozen=True)
class ParsedDocument:
    """A knowledge-base file with front-matter applied and languages split."""

    source_file: str
    doc_id: str
    topic: str
    audience: str
    review_status: str
    languages: tuple[str, ...]
    parts: tuple[LanguagePart, ...]

    @property
    def metadata(self) -> dict[str, object]:
        """Front-matter fields copied into every chunk's metadata."""
        return {
            "source_file": self.source_file,
            "doc_id": self.doc_id,
            "topic": self.topic,
            "audience": self.audience,
            "review_status": self.review_status,
        }


def detect_language(text: str) -> str:
    """Return ``ar`` or ``en`` from a heading suffix or Arabic-letter ratio."""
    stripped = text.strip()
    if stripped.endswith(AR_SUFFIX):
        return "ar"
    if stripped.endswith(EN_SUFFIX):
        return "en"
    letters = sum(1 for char in stripped if char.isalpha())
    arabic = len(ARABIC_CHAR_RE.findall(stripped))
    return "ar" if letters and arabic / letters > 0.30 else "en"


def split_languages(body: str) -> list[LanguagePart]:
    """Split a Markdown body into language parts on top-level H1 headings."""
    matches = list(H1_RE.finditer(body))
    if not matches:
        text = body.strip()
        return [LanguagePart(detect_language(text), text)] if text else []

    parts: list[LanguagePart] = []
    for index, match in enumerate(matches):
        start = 0 if index == 0 else match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            parts.append(LanguagePart(detect_language(match.group(1)), text))
    return parts


def read_document(path: Path) -> ParsedDocument:
    """Parse one Markdown file with ``python-frontmatter``."""
    post = frontmatter.load(path)
    meta = post.metadata
    parts = split_languages(post.content)
    if not parts:
        raise ValueError(f"No language sections found in {path}")

    return ParsedDocument(
        source_file=path.name,
        doc_id=str(meta.get("doc_id") or path.stem),
        topic=str(meta.get("topic", "")),
        audience=str(meta.get("audience", "")),
        review_status=str(meta.get("review_status", "")),
        languages=tuple(part.language for part in parts),
        parts=tuple(parts),
    )


def load_documents(data_dir: Path | None = None) -> list[ParsedDocument]:
    """Load every ``*.md`` file in ``data/``, sorted by filename."""
    directory = data_dir or Path(__file__).resolve().parents[1] / "data"
    documents = [read_document(path) for path in sorted(directory.glob("*.md"))]
    logger.info(
        "knowledge_base_parsed",
        extra={
            "files": len(documents),
            "parts": sum(len(doc.parts) for doc in documents),
            "languages": sorted({lang for doc in documents for lang in doc.languages}),
        },
    )
    return documents
