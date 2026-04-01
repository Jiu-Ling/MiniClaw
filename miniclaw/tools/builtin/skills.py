from __future__ import annotations

import json

from miniclaw.skills.loader import SkillLoader
from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool


def build_list_skills_tool(loader: SkillLoader) -> RegisteredTool:
    def execute(call: ToolCall) -> ToolResult:
        del call
        payload = [skill.model_dump(mode="json") for skill in loader.list_skills()]
        return ToolResult(content=json.dumps(payload, ensure_ascii=True, indent=2))

    return RegisteredTool(
        spec=ToolSpec(
            name="list_skills",
            description=(
                "Inspect the skill index when you need to choose which skill to activate "
                "before loading a full SKILL.md."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            source="builtin",
        ),
        executor=execute,
    )


def build_load_skill_tool(loader: SkillLoader) -> RegisteredTool:
    def execute(call: ToolCall) -> ToolResult:
        skill_name = str(call.arguments.get("name", "")).strip()
        if not skill_name:
            return ToolResult(content="name is required", is_error=True)

        loaded = loader.load_skill(skill_name)
        if loaded is None:
            return ToolResult(content=f"skill not found: {skill_name}", is_error=True)

        return ToolResult(content=loaded.content, metadata={"path": str(loaded.path)})

    return RegisteredTool(
        spec=ToolSpec(
            name="load_skill",
            description=(
                "Load one skill's full SKILL.md after the index shows it is the right tool "
                "for the current task."
            ),
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            source="builtin",
        ),
        executor=execute,
    )
