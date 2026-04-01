from __future__ import annotations

from collections.abc import Iterable

from miniclaw.capabilities.contracts import CapabilityEntry, CapabilityIndex


def render_capability_sections(index: CapabilityIndex) -> str:
    sections: list[str] = []

    guidance = _render_guidance(index)
    if guidance:
        sections.append(guidance)

    skills = _render_skills(index.skills)
    if skills:
        sections.append(skills)

    tools = _render_tools(index)
    if tools:
        sections.append(tools)

    mcp = _render_mcp(index.mcp_servers)
    if mcp:
        sections.append(mcp)

    return "\n\n".join(sections)


def _render_guidance(index: CapabilityIndex) -> str:
    lines = [hint.strip() for hint in index.selection_hints if hint.strip()]
    has_inventory = any(
        [
            index.visible_tools,
            index.discoverable_tools,
            index.skills,
            index.mcp_servers,
        ]
    )
    if not lines and has_inventory:
        lines = [
            "Visible tools can be used directly.",
            "Use activation tools to reveal additional skills or MCP capabilities when needed.",
        ]
    if index.activation_tools:
        if lines:
            lines.append("")
        lines.append("Activation tools:")
        lines.extend(f"- {name}" for name in index.activation_tools if str(name).strip())

    if not lines:
        return ""

    return "\n".join(["## Capability Guidance", *lines])


def _render_skills(skills: Iterable[CapabilityEntry]) -> str:
    lines = [_render_skill_entry(entry) for entry in skills]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    header = (
        "## Skills Summary\n"
        "When a task matches a discoverable skill below, load it with `load_skill_tools` "
        "BEFORE using generic tools. Skills contain specialized instructions and APIs "
        "that are more effective than general-purpose approaches."
    )
    return "\n".join([header, *lines])


def _render_tools(index: CapabilityIndex) -> str:
    lines = [_render_entry(entry) for entry in index.discoverable_tools]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    return "\n".join(["## Tools Summary", *lines])


def _render_mcp(mcp_servers: Iterable[CapabilityEntry]) -> str:
    lines: list[str] = []
    for entry in mcp_servers:
        rendered = _render_entry(entry)
        if not rendered:
            continue
        lines.append(rendered)

    if not lines:
        return ""

    return "\n".join(["## MCP Summary", *lines])


def _render_entry(entry: CapabilityEntry) -> str:
    description = entry.description.strip()

    parts = [f"- {entry.name}"]
    if description:
        parts.append(f": {description}")

    children = [child.strip() for child in entry.children if child.strip()]
    if children:
        parts.append(f" (tools: {', '.join(children)})")

    return "".join(parts)


def _render_skill_entry(entry: CapabilityEntry) -> str:
    rendered = _render_entry(entry)
    if not rendered:
        return ""
    if entry.active or entry.always_active:
        return f"{rendered} [active]"
    return f"{rendered} [discoverable]"
