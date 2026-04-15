"""JSON parsing helpers with consistent fallback semantics.

Three idioms covered:
  - safe_loads(text, default)        — strict parse, default on failure
  - safe_loads_dict(text, default)   — enforces dict type (most common)
  - safe_loads_with_raw(text)        — {"raw": str(text)} wrapper on failure
  - extract_json_object(text)        — LLM-output-aware: strips code fences,
                                       pulls first balanced {...} block

Use extract_json_object for LLM outputs (rewrite, consolidation, future
subagent judge). Use safe_loads_dict for trusted JSON from disk or DB.
Use safe_loads_with_raw only when downstream code wants access to
unparseable input for debugging (tool-call argument parsers).
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
    """Parse JSON, falling back to {"raw": <original>} on any failure.

    Used by tool-call argument parsers in nodes.py / tool_loop.py where
    downstream code still wants access to unparseable input for debugging.
    If the input is already a dict, it is returned unchanged. Empty input
    returns an empty dict.
    """
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
_FIRST_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(
    text: Any,
    default: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract a JSON object from LLM output that may contain prose or fences.

    Tries in order:
      1. Strip markdown code fences and parse the inner content
      2. Parse the whole text directly
      3. Regex out the first {...} balanced block and parse that
      4. Return default (or {}) on all failures

    This is the right helper for LLM outputs. Do NOT use it for trusted
    JSON from disk — use safe_loads_dict instead.

    Deliberate non-feature: single-quoted "JSON" is not repaired. Aggressive
    repair is out of scope; mini models are expected to emit valid JSON.
    """
    fallback = default if default is not None else {}
    if not isinstance(text, str) or not text:
        return dict(fallback)

    # Try 1: fenced content
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1).strip()
        parsed = safe_loads_dict(candidate, default=None)
        if parsed:
            return parsed

    # Try 2: whole-text parse
    parsed = safe_loads_dict(text.strip(), default=None)
    if parsed:
        return parsed

    # Try 3: first balanced {...} block via greedy regex
    obj_match = _FIRST_OBJECT_RE.search(text)
    if obj_match:
        parsed = safe_loads_dict(obj_match.group(0), default=None)
        if parsed:
            return parsed

    return dict(fallback)
