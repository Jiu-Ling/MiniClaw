from __future__ import annotations

from miniclaw.runtime.nodes import make_planner


def test_planner_returns_empty_dict_when_route_not_planned() -> None:
    # New planner skips entirely when route != "planned" (T10: slim planner)
    planner = make_planner()

    result = planner({"route": "simple", "user_input": "Say hello"})

    assert result == {}


def test_planner_returns_subagent_briefs_when_no_provider() -> None:
    # New planner with no provider returns empty briefs (advisory; executor runs directly)
    planner = make_planner()

    result = planner(
        {
            "route": "planned",
            "user_input": "Build planner and executor nodes",
            "memory_context": "",
        }
    )

    assert result["plan_summary"] == ""
    assert result["subagent_briefs"] == []
    assert result["executor_notes"] == ""
