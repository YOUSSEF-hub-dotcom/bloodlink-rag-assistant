"""Unit tests for ingestion.clean rules (no network, no GPU)."""

from __future__ import annotations

import unicodedata

from ingestion.clean import (
    VERIFY_SENTINEL,
    clean_markdown,
    collapse_whitespace,
    normalize_unicode,
    remove_html_comments,
    remove_separator_lines,
    remove_tatweel,
    remove_verify_sentinels,
    strip_verify_markers,
)


def test_remove_html_comments_multiline() -> None:
    text = "before\n<!-- DEV NOTE\nline 2 -->\nafter"
    assert remove_html_comments(text) == "before\n\nafter"


def test_strip_verify_markers_flags_and_inserts_sentinel() -> None:
    text = "Profile [verify] page and [verify screen name] again"
    cleaned, had = strip_verify_markers(text)
    assert had is True
    assert cleaned.count(VERIFY_SENTINEL) == 2
    assert "[verify]" not in cleaned.lower()


def test_strip_verify_markers_absent() -> None:
    cleaned, had = strip_verify_markers("no markers here")
    assert cleaned == "no markers here"
    assert had is False


def test_verify_sentinel_survives_markdown_splitter() -> None:
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    from ingestion.chunk import HEADERS_TO_SPLIT

    text, _ = strip_verify_markers("## Heading\nText [verify] here")
    sections = MarkdownHeaderTextSplitter(HEADERS_TO_SPLIT).split_text(text)
    assert any(VERIFY_SENTINEL in section.page_content for section in sections)


def test_normalize_unicode_nfc_and_odd_spaces() -> None:
    decomposed = unicodedata.normalize("NFD", "é")
    text = f"cafe\u00a0{decomposed}"
    cleaned = normalize_unicode(text)
    assert cleaned == "cafe é"
    assert unicodedata.is_normalized("NFC", cleaned)


def test_remove_tatweel_keeps_diacritics_and_letters() -> None:
    text = "التبـرع بِالدَم"  # tatweel after ba, kasra on ba
    cleaned = remove_tatweel(text)
    assert "ـ" not in cleaned
    assert cleaned == "التبرع بِالدَم"


def test_remove_separator_lines_keeps_table_separators() -> None:
    text = "intro\n---\n| a | b |\n|---|---|\n| 1 | 2 |\n---\nend"
    cleaned = remove_separator_lines(text)
    assert "\n---\n" not in cleaned
    assert "|---|---|" in cleaned


def test_collapse_whitespace_and_trim() -> None:
    text = "line one   \r\n\r\n\r\nline two   \n"
    assert collapse_whitespace(text) == "line one\n\nline two"


def test_remove_verify_sentinels_repairs_spacing() -> None:
    text = f"Sign up {VERIFY_SENTINEL}. Done {VERIFY_SENTINEL}!"
    assert remove_verify_sentinels(text) == "Sign up. Done!"


def test_clean_markdown_end_to_end_file04_dev_note() -> None:
    text = "<!-- DEV NOTE: api=/x -->\n# Guide\nBody [verify] text."
    cleaned, had = clean_markdown(text)
    assert "DEV NOTE" not in cleaned
    assert "api=/x" not in cleaned
    assert had is True
    assert "[verify]" not in remove_verify_sentinels(cleaned)
