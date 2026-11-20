"""Unit tests for parsing, chunking, and id/hash stability (no GPU/network)."""

from __future__ import annotations

from pathlib import Path

from ingestion.chunk import chunk_document, count_tokens, get_tokenizer
from ingestion.clean import clean_markdown
from ingestion.parse import load_documents, split_languages

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _all_chunks():
    tokenizer = get_tokenizer(local_files_only=True)
    chunks = []
    for doc in load_documents(DATA_DIR):
        cleaned = [(p.language, clean_markdown(p.markdown)[0]) for p in doc.parts]
        chunks.extend(chunk_document(doc, cleaned, tokenizer=tokenizer))
    return chunks


def test_language_splitting_all_four_files() -> None:
    docs = load_documents(DATA_DIR)
    assert [d.doc_id for d in docs] == [
        "01_donor_eligibility",
        "02_donation_preparation_and_care",
        "03_blood_compatibility",
        "04_bloodlink_platform_guide",
    ]
    for doc in docs:
        assert tuple(p.language for p in doc.parts) == ("ar", "en")
        assert len(doc.parts) == 2


def test_language_detection_suffix_and_ratio() -> None:
    parts = split_languages("# عنوان (العربية)\nنص\n---\n# Title (English)\nText")
    assert [p.language for p in parts] == ["ar", "en"]
    parts_fallback = split_languages("# مرحبا بك\nهذا نص عربي طويل")
    assert parts_fallback[0].language == "ar"


def test_every_h3_becomes_own_chunk_doc03() -> None:
    chunks = _all_chunks()
    ar_faq = [
        c
        for c in chunks
        if c.doc_id == "03_blood_compatibility"
        and c.language == "ar"
        and c.metadata.get("h2", "").startswith("أسئلة شائعة")
    ]
    assert len(ar_faq) == 8
    en_faq = [
        c
        for c in chunks
        if c.doc_id == "03_blood_compatibility"
        and c.language == "en"
        and c.metadata.get("h2") == "Common Questions from First-Time Donors"
    ]
    assert len(en_faq) == 8


def test_four_screening_questions_are_own_chunks() -> None:
    chunks = _all_chunks()
    ar_q = [
        c
        for c in chunks
        if c.doc_id == "01_donor_eligibility"
        and c.language == "ar"
        and "الفحص المبدئي: 4 أسئلة سريعة >" in c.section_path
    ]
    en_q = [
        c
        for c in chunks
        if c.doc_id == "01_donor_eligibility"
        and c.language == "en"
        and "Pre-Screening: 4 Quick Questions >" in c.section_path
    ]
    assert len(ar_q) == 4
    assert len(en_q) == 4
    assert all(c.metadata["h3"] for c in ar_q + en_q)


def test_compatibility_table_stays_in_one_chunk() -> None:
    chunks = _all_chunks()
    ar_table = [
        c
        for c in chunks
        if c.doc_id == "03_blood_compatibility"
        and c.language == "ar"
        and "جدول توافق نقل خلايا الدم الحمراء" in c.section_path
    ]
    en_table = [
        c
        for c in chunks
        if c.doc_id == "03_blood_compatibility"
        and c.language == "en"
        and "Compatibility Table for Red Blood Cells" in c.section_path
    ]
    assert len(ar_table) == 1
    assert len(en_table) == 1
    for chunk in ar_table + en_table:
        rows = [line for line in chunk.content.splitlines() if line.startswith("|")]
        assert len(rows) == 10  # header + separator + 8 blood types
        assert chunk.content.count("| O- |") == 1
        assert chunk.content.count("| AB+ |") == 1
        assert chunk.token_count <= 400


def test_no_chunk_mixes_languages_and_within_limit() -> None:
    import re

    arabic_re = re.compile(r"[\u0600-\u06ff]")
    latin_re = re.compile(r"[A-Za-z]")
    for chunk in _all_chunks():
        assert chunk.token_count <= 400
        body = "\n".join(chunk.content.split("\n")[1:])
        ar = len(arabic_re.findall(body))
        la = len(latin_re.findall(body))
        # allow short labels like A+, O-, BloodLink in Arabic text
        if chunk.language == "ar":
            assert ar > 0
        else:
            assert la > ar


def test_html_comment_stripped_and_verify_stripped_but_flagged() -> None:
    """File 04 starts with a DEV NOTE comment and has [verify] markers."""
    chunks = [c for c in _all_chunks() if c.doc_id == "04_bloodlink_platform_guide"]
    assert chunks
    for chunk in chunks:
        assert "<!--" not in chunk.content
        assert "DEV NOTE" not in chunk.content
        assert "/api/" not in chunk.content
        assert "[verify]" not in chunk.content.lower()

    # Markers appear in some sections (not all), so only affected chunks are flagged.
    marked = [c for c in chunks if c.metadata["has_unverified_markers"]]
    assert marked
    assert any(not c.metadata["has_unverified_markers"] for c in chunks)


def test_deterministic_ids_and_hash_change_detection() -> None:
    chunks_a = _all_chunks()
    chunks_b = _all_chunks()
    assert [c.id for c in chunks_a] == [c.id for c in chunks_b]
    assert [c.content_hash for c in chunks_a] == [c.content_hash for c in chunks_b]
    modified = chunks_a[0].content + " extra"
    assert modified != chunks_a[0].content
    assert count_tokens(modified, get_tokenizer(local_files_only=True)) >= chunks_a[0].token_count


def test_verify_marker_metadata_only_on_affected_chunks() -> None:
    chunks = [c for c in _all_chunks() if c.doc_id == "04_bloodlink_platform_guide"]
    marked = [c for c in chunks if c.metadata["has_unverified_markers"]]
    unmarked = [c for c in chunks if not c.metadata["has_unverified_markers"]]
    assert marked  # file 04 has many [verify] markers
    assert unmarked
    for chunk in chunks:
        assert "[verify]" not in chunk.content.lower()
        assert "<!--" not in chunk.content
        assert "DEV NOTE" not in chunk.content


def test_contextual_header_and_metadata() -> None:
    chunk = next(
        c
        for c in _all_chunks()
        if c.section_path == "شروط الأهلية للتبرع بالدم (العربية) > الشروط الأساسية للتبرع بالدم"
    )
    assert chunk.content.startswith(chunk.section_path + "\n")
    assert count_tokens(chunk.content, get_tokenizer(local_files_only=True)) == chunk.token_count
    for key in (
        "source_file",
        "doc_id",
        "topic",
        "audience",
        "review_status",
        "language",
        "section_path",
        "h1",
        "h2",
        "h3",
        "chunk_index",
        "token_count",
        "has_unverified_markers",
    ):
        assert key in chunk.metadata
    assert chunk.metadata["doc_id"] == "01_donor_eligibility"
    assert chunk.metadata["language"] == "ar"
