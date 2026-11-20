"""Deterministic, pre-LLM input rules: length cap, language, emergency regex.

These checks are intentionally cheap and local.  They run before any query
analysis or answer generation so an emergency is answered with the canned
ambulance message without contacting Groq.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from core.config import get_settings
from ingestion.parse import detect_language as _detect_language

Language = Literal["ar", "en"]

# Deliberately strong phrases; a plain "emergency contact" question is not
# treated as a live medical emergency.
EMERGENCY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:"
        r"طوارئ|طارئة|طارئ|إسعاف|اسعاف|الإسعاف|الاسعاف"
        r"|نزيف\s*(?:شديد|غزير)?|ينزف|بحاجة\s+دم\s+(?:الآن|دلوقتي)"
        r"|أحتاج\s+دم\s+(?:الآن|دلوقتي)|محتاج\s+دم\s+دلوقتي"
        r"|إغماء|غيبوبة|فاقد\s+للوعي|ألم\s+صدر|ضيق\s+نفس|اختناق"
        r"|حادث|دهسة|سكتة|جلطة|نوبة\s+قلبية|اتصلوا\s+بالإسعاف"
        r")",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:"
        r"medical\s+emergency|emergency\s+room|in\s+an\s+emergency"
        r"|call\s+(?:an\s+)?ambulance|ambulance\s+now|need\s+blood\s+now"
        r"|severe\s+bleeding|bleeding\s+heavily|unconscious|fainted"
        r"|chest\s+pain|can(?:not|'t)\s+breathe|heart\s+attack|stroke"
        r"|serious\s+accident|call\s+123"
        r")",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class InputCheck:
    """Result of blocking input validation."""

    allowed: bool
    language: Language
    message: str
    reason: str | None = None


def detect_language(text: str) -> Language:
    """Return the supported language tag for ``text``."""
    detected = _detect_language(text)
    return "ar" if detected == "ar" else "en"


def check_length(message: str) -> str | None:
    """Return a rejection reason when ``message`` is empty or too long."""
    stripped = message.strip()
    if not stripped:
        return "empty_message"
    limit = get_settings().max_user_message_chars
    if len(stripped) > limit:
        return "message_too_long"
    return None


def is_emergency(message: str, language: Language | None = None) -> bool:
    """Return ``True`` when a deterministic AR/EN emergency pattern matches."""
    lang = language or detect_language(message)
    patterns = EMERGENCY_PATTERNS[:1] if lang == "ar" else EMERGENCY_PATTERNS[1:]
    return any(pattern.search(message) for pattern in patterns)


def check_message(message: str) -> InputCheck:
    """Run the blocking input checks in spec order."""
    language = detect_language(message)
    reason = check_length(message)
    if reason:
        return InputCheck(allowed=False, language=language, message=message.strip(), reason=reason)
    return InputCheck(allowed=True, language=language, message=message.strip())


__all__ = [
    "EMERGENCY_PATTERNS",
    "InputCheck",
    "check_length",
    "check_message",
    "detect_language",
    "is_emergency",
]
