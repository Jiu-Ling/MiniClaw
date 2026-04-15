"""JSON parsing helpers with consistent fallback semantics.

Four idioms:
  - safe_loads(text, default)        → parse or return default
  - safe_loads_dict(text, default)   → parse or return default dict
  - safe_loads_with_raw(text)        → parse or return {"raw": text}
  - extract_json_object(text)        → LLM output: strip fences, find {...}
"""
from __future__ import annotations

import json
import re
from typing import Any, TypeVar

T = TypeVar("T")

__all__ = [
    "safe_loads",
    "safe_loads_dict",
    "safe_loads_with_raw",
    "extract_json_object",
]


def safe_loads(text: Any, default: T = None) -> Any | T:
    """Strict json.loads with exception → default fallback.

    Use when you just want "parse or give me the default", no type check.
    Accepts None/non-string inputs and returns default for those too.
    """
    if not isinstance(text, (str, bytes, bytearray)) or not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def safe_loads_dict(
    text: Any,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse JSON and enforce dict result. Non-dict → default (or {})."""
    fallback = default if default is not None else {}
    parsed = safe_loads(text, default=None)
    if not isinstance(parsed, dict):
        return dict(fallback)
    return parsed


def safe_loads_with_raw(text: Any) -> dict[str, Any]:
    """Parse JSON, fallback to {"raw": <original>}. Dict passthrough; empty → {}."""
    if text is None or text == "":
        return {}
    if isinstance(text, dict):
        return dict(text)
    if not isinstance(text, (str, bytes, bytearray)):
        return {"raw": str(text)}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"raw": str(text)}
    if not isinstance(parsed, dict):
        return {"raw": str(text)}
    return parsed


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def _find_first_json_object(text: str) -> str | None:
    """Find first balanced {...} block via O(n) brace counting.

    Handles string literals and escapes; no backtracking. Returns substring
    with braces or None if not found.
    """
    start = -1
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue  # unbalanced } before any {; skip
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None


def extract_json_object(
    text: Any,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract JSON object from LLM output (prose + fences).

    Tries: (1) unfenced content, (2) whole text, (3) first {...} block, (4) default.
    Single-quoted JSON is not repaired. Empty {} is a valid result.
    """
    fallback = default if default is not None else {}
    if not isinstance(text, str) or not text:
        return dict(fallback)

    # Try 1: fenced content
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        parsed = safe_loads(candidate, default=None)
        if isinstance(parsed, dict):
            return parsed

    # Try 2: whole-text parse
    parsed = safe_loads(text.strip(), default=None)
    if isinstance(parsed, dict):
        return parsed

    # Try 3: first balanced {...} block via linear scan (ReDoS-safe)
    candidate = _find_first_json_object(text)
    if candidate is not None:
        parsed = safe_loads(candidate, default=None)
        if isinstance(parsed, dict):
            return parsed

    return dict(fallback)
