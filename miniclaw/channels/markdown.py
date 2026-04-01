"""Convert standard Markdown (LLM output) to Telegram-compatible HTML.

HTML parse_mode only needs & < > escaping in non-code text.
Far more reliable than MarkdownV2 which requires 20+ special char escapes.
"""
from __future__ import annotations

import re

_CODE_BLOCK_RE = re.compile(r"```(\w*)\n([\s\S]*?)```")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def markdown_to_html(text: str) -> str:
    if not text:
        return text

    # Phase 1: Extract and protect code blocks
    blocks: list[str] = []

    def _save_block(m: re.Match) -> str:
        lang = m.group(1).strip()
        body = _html_escape(m.group(2))
        idx = len(blocks)
        if lang:
            blocks.append(f'<pre><code class="language-{lang}">{body}</code></pre>')
        else:
            blocks.append(f"<pre><code>{body}</code></pre>")
        return f"\x00BLOCK{idx}\x00"

    text = _CODE_BLOCK_RE.sub(_save_block, text)

    # Phase 2: Extract and protect inline code
    inlines: list[str] = []

    def _save_inline(m: re.Match) -> str:
        idx = len(inlines)
        inlines.append(f"<code>{_html_escape(m.group(1))}</code>")
        return f"\x00INLINE{idx}\x00"

    text = _INLINE_CODE_RE.sub(_save_inline, text)

    # Phase 3: Escape remaining HTML special chars
    text = _html_escape(text)

    # Phase 4: Convert formatting
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _STRIKE_RE.sub(r"<s>\1</s>", text)
    text = _LINK_RE.sub(r'<a href="\2">\1</a>', text)

    # Phase 5: Restore protected spans
    for idx, code in enumerate(inlines):
        text = text.replace(f"\x00INLINE{idx}\x00", code)
    for idx, block in enumerate(blocks):
        text = text.replace(f"\x00BLOCK{idx}\x00", block)

    return text


def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("'", "&#x27;").replace('"', "&quot;")
