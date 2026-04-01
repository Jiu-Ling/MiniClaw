---
name: memory
description: Memory management guide — when and how to update MEMORY.md, search memory, and manage long-term facts. Load this when editing memory or debugging memory behavior.
discoverable: true
---

# Memory

## Structure

MiniClaw uses two memory layers:

- `.miniclaw/MEMORY.md` — Long-term facts (preferences, project context, relationships). Always loaded into context.
- SQLite — Runtime memory items, thread digests, and recovery state. Managed automatically.

## When to Update MEMORY.md

Write important facts immediately using `edit_file` or `write_file`:
- User preferences ("I prefer dark mode")
- Project context ("The API uses OAuth2")
- Relationships ("Alice is the project lead")
- Conventions ("We use uv for Python package management")

## What Belongs in Memory

- User preferences
- Project conventions
- Durable facts
- Short summaries of recent work

## What Does NOT Belong in Memory

- Persona text already covered by `SOUL.md`
- Raw transient logs that only matter for one turn
- Repeated instructions already enforced by runtime

## Searching Past Context

For runtime memory stored in SQLite, use the built-in memory search capabilities. For `.miniclaw/MEMORY.md`:

- Small file: use `read_file`, then search in-memory
- Large file: use the `shell` tool for targeted search

Examples:
```bash
grep -i "keyword" .miniclaw/MEMORY.md
```

## Auto-consolidation

Old conversations are automatically summarized and stored in SQLite when the session grows large. Long-term facts are extracted to MEMORY.md. You don't need to manage this.

## Notes

- Keep memory concise and editable
- Prefer summarizing recent work in a way that remains useful after context compaction
