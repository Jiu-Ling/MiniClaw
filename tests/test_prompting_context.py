from miniclaw.prompting.context import ContextBuilder


def test_recommended_plan_section_rendered():
    state = {
        "plan_summary": "Investigate then patch",
        "subagent_briefs": [
            {"role": "researcher", "task": "find issue", "expected_output": "RCA"},
            {"role": "executor", "task": "apply fix", "depends_on": [0]},
        ],
        "executor_notes": "Start with researcher first",
    }
    sections = ContextBuilder._build_planner_sections(state)
    text = "\n\n".join(sections)
    assert "## Recommended Plan" in text
    assert "Investigate then patch" in text
    assert "researcher" in text
    assert "find issue" in text
    assert "Start with researcher" in text


def test_recommended_plan_omitted_when_no_summary():
    sections = ContextBuilder._build_planner_sections({})
    assert sections == []


def test_main_agent_system_message_has_cache_control():
    builder = ContextBuilder(
        system_prompt="you are an agent",
        skills_loader=None, tool_registry=None, mcp_registry=None,
        history_char_budget=10000, max_history_messages=50,
    )
    messages = builder.build_provider_messages({
        "messages": [{"role": "user", "content": "hi"}],
        "memory_context": "some memory",
        "user_input": "hi",
    })
    system_msg = messages[0]
    parts = getattr(system_msg, "content_parts", None) or []
    assert parts, "system message must use content_parts"
    assert any(
        isinstance(p, dict) and p.get("cache_control", {}).get("type") == "ephemeral"
        for p in parts
    ), f"static part must have cache_control: ephemeral; got {parts}"
