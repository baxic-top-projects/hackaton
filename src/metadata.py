from __future__ import annotations

import re
from pathlib import Path
from typing import Any


DATE_PATTERNS = [
    r"\b(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b",
    r"\b(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}\b",
    r"\b(?:19|20)\d{2}\b",
]
AUTHOR_PATTERN = r"(?:автор(?:ы)?|authors?|разработал(?:и)?|исполнитель)\s*[:\-]\s*([^\n.;]{3,120})"
CONDITION_PATTERNS = {
    "temperature": r"\b\d{2,4}\s?(?:°C|C|℃)\b",
    "ph": r"\bpH\s?\d+(?:[.,]\d+)?\b",
    "percent": r"\b\d+(?:[.,]\d+)?\s?%",
    "particle_size": r"(?:[+-]\s?\d+\s?(?:мкм|mm|мм)|\b\d+\s?(?:мкм|mm|мм)\b)",
}


def extract_metadata(text: str, source: str, base: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = dict(base or {})
    metadata.setdefault("source", source)
    metadata.setdefault("extension", Path(source).suffix.lower())
    metadata["dates"] = _unique(_find_dates(text))[:5]
    metadata["authors"] = _unique(_find_authors(text))[:5]
    metadata["conditions"] = {
        name: _unique(re.findall(pattern, text, flags=re.IGNORECASE))[:8]
        for name, pattern in CONDITION_PATTERNS.items()
    }
    metadata["has_chinese"] = bool(re.search(r"[\u4e00-\u9fff]", text))
    metadata["has_latin"] = bool(re.search(r"[A-Za-z]", text))
    return metadata


def _find_dates(text: str) -> list[str]:
    values: list[str] = []
    for pattern in DATE_PATTERNS:
        values.extend(re.findall(pattern, text))
    return values


def _find_authors(text: str) -> list[str]:
    return [match.strip() for match in re.findall(AUTHOR_PATTERN, text, flags=re.IGNORECASE)]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
