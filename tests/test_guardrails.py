"""Unit tests for deterministic guardrails and post-stream checks."""

from __future__ import annotations

import logging

from core.config import get_settings
from generation.prompts import EMERGENCY
from guardrails.checks import extract_numbers, run_post_checks
from guardrails.rules import check_length, check_message, detect_language, is_emergency
from retrieval.search import SearchResult


def test_emergency_regex_arabic_and_english_positives() -> None:
    arabic = [
        "أحتاج دم الآن!",
        "في حالة طارئة، اتصل بالإسعاف",
        "رجل فاقد للوعي بعد حادث",
    ]
    english = [
        "Call an ambulance, this is a medical emergency.",
        "I need blood now.",
        "He is unconscious after a serious accident.",
    ]

    for text in arabic:
        assert is_emergency(text, "ar") is True, text
    for text in english:
        assert is_emergency(text, "en") is True, text


def test_emergency_regex_negatives() -> None:
    negatives = [
        "ما هي الشروط الأساسية للتبرع بالدم؟",
        "هل ينفع أتبرع وأنا بصوم؟",
        "How do I book a donation appointment?",
        "Can I donate after a common cold?",
    ]
    for text in negatives:
        assert is_emergency(text) is False, text


def test_length_cap_and_language_detection() -> None:
    limit = get_settings().max_user_message_chars
    assert check_length("") == "empty_message"
    assert check_length("x" * (limit + 1)) == "message_too_long"
    assert check_length("within limit") is None

    result = check_message("هل ينفع أتبرع وأنا بصوم؟")
    assert result.allowed is True
    assert result.language == "ar"
    assert detect_language("Can I donate?") == "en"
    assert "123" in EMERGENCY["ar"]


def test_post_checks_log_ungrounded_number_and_medical_advice(
    caplog: logging.LogCaptureFixture,
) -> None:
    results = [
        SearchResult(
            id="c1",
            content="Age 18 to 65. Weight over 50 kg.",
            metadata={"doc_id": "01_donor_eligibility"},
            section_path="Eligibility",
            language="en",
            score=0.8,
        )
    ]
    caplog.set_level(logging.WARNING)

    findings = run_post_checks(
        "Age 18 to 99. Take 500 mg for this.",
        results,
        language="en",
        conversation_id="conv-1",
    )

    assert "99" in findings["ungrounded_numbers"]
    assert "500" in findings["ungrounded_numbers"]
    assert "18" not in findings["ungrounded_numbers"]
    assert "dosage" in findings["medical_advice_flags"]
    messages = [record.message for record in caplog.records]
    assert "ungrounded_number" in messages
    assert "medical_advice_flag" in messages


def test_arabic_indic_digits_are_normalized_before_comparison() -> None:
    assert extract_numbers("العمر ١٨ سنة والوزن ٥٠ كجم") == ["18", "50"]
    assert extract_numbers("12.5 جم") == ["12.5"]


def test_post_checks_never_modify_or_block_the_answer() -> None:
    answer = "القرار النهائي للفريق الطبي."
    findings = run_post_checks(answer, [], language="ar")
    assert findings == {"ungrounded_numbers": [], "medical_advice_flags": []}
