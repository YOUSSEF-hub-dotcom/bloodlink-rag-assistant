"""Deterministic Markdown cleaning rules applied before chunking.

``[verify]`` markers are replaced by an internal sentinel rather than deleted
outright: this lets the chunker flag only the chunks that actually contained a
marker.  :func:`remove_verify_sentinels` strips the sentinel from final text.
Arabic diacritics, alef/yaa/taa-marbuta variants and markdown structure are
preserved; the stored wording stays exactly as authored (apart from tatweel).
"""

from __future__ import annotations

import re
import unicodedata

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
VERIFY_MARKER_RE = re.compile(r"\[verify[^\]\n]*\]", re.IGNORECASE)
SEPARATOR_LINE_RE = re.compile(r"(?m)^[ \t]*---[ \t]*$")
TRAILING_SPACES_RE = re.compile(r"[ \t]+$", re.MULTILINE)
PUNCTUATION_RE = re.compile(r"[ \t]+([.,،؛؟!:])")

VERIFY_SENTINEL = "__VERIFY_MARKER__"  # plain-text sentinel survives Markdown splitting
TATWEEL = "\u0640"
ODD_WHITESPACE = {
    "\u00a0": " ",  # no-break space
    "\u2007": " ",  # figure space
    "\u202f": " ",  # narrow no-break space
    "\u200b": "",  # zero-width space
    "\ufeff": "",  # byte order mark
}


def remove_html_comments(text: str) -> str:
    """Remove multi-line ``<!-- ... -->`` blocks (e.g. the DEV NOTE in file 04)."""
    return HTML_COMMENT_RE.sub("", text)


def strip_verify_markers(text: str) -> tuple[str, bool]:
    """Replace ``[verify]`` / ``[verify ...]`` notes with an internal sentinel."""
    cleaned, count = VERIFY_MARKER_RE.subn(VERIFY_SENTINEL, text)
    return cleaned, count > 0


def normalize_unicode(text: str) -> str:
    """Apply NFC normalization and map odd spaces to ordinary spaces."""
    text = unicodedata.normalize("NFC", text)
    for source, replacement in ODD_WHITESPACE.items():
        text = text.replace(source, replacement)
    return text


def remove_tatweel(text: str) -> str:
    """Remove Arabic tatweel (U+0640) without touching diacritics or letters."""
    return text.replace(TATWEEL, "")


def remove_separator_lines(text: str) -> str:
    """Drop standalone ``---`` language separators, keeping table separator rows."""
    return SEPARATOR_LINE_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Trim trailing spaces, collapse horizontal runs and 3+ newlines to 2."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in text.split("\n"):
        leading = re.match(r"[ \t]*", line).group(0)
        rest = re.sub(r"[ \t]{2,}", " ", line[len(leading) :]).rstrip()
        lines.append(leading + rest)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_verify_sentinels(text: str) -> str:
    """Remove internal sentinels and repair spacing around punctuation."""
    text = re.sub(r"[ \t]+" + VERIFY_SENTINEL, VERIFY_SENTINEL, text)
    text = text.replace(VERIFY_SENTINEL, "")
    text = PUNCTUATION_RE.sub(r"\1", text)
    text = TRAILING_SPACES_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_markdown(text: str) -> tuple[str, bool]:
    """Run every cleaning rule in order; return (text_with_sentinels, had_markers)."""
    cleaned = remove_html_comments(text)
    cleaned, had_markers = strip_verify_markers(cleaned)
    cleaned = normalize_unicode(cleaned)
    cleaned = remove_tatweel(cleaned)
    cleaned = remove_separator_lines(cleaned)
    return collapse_whitespace(cleaned), had_markers
