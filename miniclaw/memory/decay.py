"""Fading memory: compress old entries, delete ancient ones.

Called after each turn's memory write. Older memories get shorter;
anything past 30 days is dropped entirely.

Decay tiers:
  - Today (age=0): keep full (already compressed by write-time rules)
  - 1-7 days:      compress each entry to ≤100 chars
  - 7-30 days:     compress each entry to ≤50 chars
  - 30+ days:      delete

Per-thread cap: 5 entries (most recent kept).
"""
from __future__ import annotations

import re
from datetime import date

from miniclaw.memory.files import MemoryFileStore, compress_memory_entry

_MAX_ITEMS_PER_THREAD = 5

# Regex to extract the date (YYYY-MM-DD) from a memory entry line.
# Entries look like: "2026-04-10 06:33: User asked to ...; outcome: ..."
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def decay_memory(store: MemoryFileStore, today: date) -> None:
    """Apply fading decay to all recent_work entries in the memory file."""
    doc = store.read()
    changed = False

    for thread_key in list(doc.recent_work):
        items = doc.recent_work[thread_key]
        decayed: list[str] = []
        for item in items:
            age = _entry_age_days(item, today)
            if age is None:
                # Can't parse date — keep short
                decayed.append(compress_memory_entry(item, max_chars=100))
                changed = True
                continue
            if age > 30:
                changed = True
                continue  # Drop
            if age > 7:
                compressed = compress_memory_entry(item, max_chars=50)
                if compressed != item:
                    changed = True
                decayed.append(compressed)
            elif age > 0:
                compressed = compress_memory_entry(item, max_chars=100)
                if compressed != item:
                    changed = True
                decayed.append(compressed)
            else:
                # Today — already compressed at write time
                decayed.append(item)

        # Cap per thread
        if len(decayed) > _MAX_ITEMS_PER_THREAD:
            decayed = decayed[-_MAX_ITEMS_PER_THREAD:]
            changed = True

        doc.recent_work[thread_key] = decayed

    # Remove empty threads
    empty_threads = [k for k, v in doc.recent_work.items() if not v]
    for k in empty_threads:
        del doc.recent_work[k]
        changed = True

    if changed:
        store.update(
            long_term_facts=doc.long_term_facts,
            recent_work=doc.recent_work,
        )


def _entry_age_days(entry: str, today: date) -> int | None:
    """Parse the date from an entry and return its age in days."""
    match = _DATE_RE.search(entry)
    if not match:
        return None
    try:
        entry_date = date.fromisoformat(match.group(1))
        return (today - entry_date).days
    except ValueError:
        return None
