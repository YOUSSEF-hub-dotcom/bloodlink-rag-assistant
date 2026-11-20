"""Non-blocking post-stream checks.

These run only after the answer has been streamed.  They never modify or
withhold text: they only emit structured warnings for numbers that are absent
from the retrieved context and for phrases that resemble diagnosis/dosage
advice.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from core.logging import get_logger
from retrieval.search import SearchResult

logger = get_logger(__name__)

_AR_TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫", "01234567890123456789.")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")

_MEDICAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dosage", re.compile(r"\b(?:take|dosage|dose|mg\b|milligrams?)\b", re.IGNORECASE)),
    ("stop_medication", re.compile(r"\bstop (?:taking|your) (?:medication|medicine|drug)\b", re.IGNORECASE)),
    ("diagnosis", re.compile(r"\b(?:you have|you are diagnosed|diagnose you|prescribe)\b", re.IGNORECASE)),
    ("ar_dosage", re.compile(r"جرعة|خذ\s+\d|تناول\s+\d|مليجرام", re.IGNORECASE)),
    ("ar_stop_medication", re.compile(r"توقف\s+عن\s+(?:الدواء|العلاج)|إيقاف\s+الدواء", re.IGNORECASE)),
    ("ar_diagnosis", re.compile(r"أنت\s+مصاب|عندك\s+\w+\s*(?:مرض|التهاب)|أشخّصك|أعطيك\s+تشخيص", re.IGNORECASE)),
)


def normalize_numbers(text: str) -> str:
    """Fold Arabic-Indic/Persian digits and decimal separators to ASCII."""
    normalized = text.translate(_AR_TO_ASCII)
    return _DECIMAL_COMMA_RE.sub(".", normalized)


def extract_numbers(text: str) -> list[str]:
    """Return unique normalized numeric tokens appearing in ``text``."""
    normalized = normalize_numbers(text)
    found: list[str] = []
    for match in _NUMBER_RE.findall(normalized):
        token = match.rstrip(".").lstrip("0") or "0"
        if token not in found:
            found.append(token)
    return found


def check_number_grounding(
    answer: str,
    results: Sequence[SearchResult],
    *,
    language: str,
    conversation_id: str | None = None,
) -> list[str]:
    """Log ``ungrounded_number`` for answer numbers absent from context."""
    context = "\n".join(result.content for result in results)
    context_numbers = set(extract_numbers(context))
    ungrounded = [number for number in extract_numbers(answer) if number not in context_numbers]
    for number in ungrounded:
        logger.warning(
            "ungrounded_number",
            extra={
                "conversation_id": conversation_id,
                "number": number,
                "language": language,
            },
        )
    return ungrounded


def check_medical_advice(
    answer: str,
    *,
    language: str,
    conversation_id: str | None = None,
) -> list[str]:
    """Log ``medical_advice_flag`` for diagnosis/dosage-like phrasing."""
    flags: list[str] = []
    for name, pattern in _MEDICAL_PATTERNS:
        if pattern.search(answer):
            if name not in flags:
                flags.append(name)
                logger.warning(
                    "medical_advice_flag",
                    extra={
                        "conversation_id": conversation_id,
                        "pattern": name,
                        "language": language,
                    },
                )
    return flags


def run_post_checks(
    answer: str,
    results: Sequence[SearchResult],
    *,
    language: str = "en",
    conversation_id: str | None = None,
) -> dict[str, list[str]]:
    """Run both checks safely; return their findings for diagnostics/tests."""
    findings: dict[str, list[str]] = {"ungrounded_numbers": [], "medical_advice_flags": []}
    try:
        findings["ungrounded_numbers"] = check_number_grounding(
            answer, results, language=language, conversation_id=conversation_id
        )
    except Exception as error:  # pragma: no cover - defensive, never block answer
        logger.warning(
            "post_check_failed",
            extra={"check": "number_grounding", "error_type": type(error).__name__},
        )
    try:
        findings["medical_advice_flags"] = check_medical_advice(
            answer, language=language, conversation_id=conversation_id
        )
    except Exception as error:  # pragma: no cover - defensive, never block answer
        logger.warning(
            "post_check_failed",
            extra={"check": "medical_advice", "error_type": type(error).__name__},
        )
    return findings


__all__ = [
    "check_medical_advice",
    "check_number_grounding",
    "extract_numbers",
    "normalize_numbers",
    "run_post_checks",
]
