from __future__ import annotations

import pytest

from miniclaw.runtime.nodes import make_classify


@pytest.mark.parametrize(
    ("user_input", "expected_route", "needs_clarification"),
    [
        ("Say hello", "simple", False),
        # Rule-based fallback classifies these as simple intent
        ("Read the README file", "simple", False),
        ("Load the skill for postgres support", "simple", False),
        # Planned markers trigger planned route
        (
            "Implement Postgres persistence and update tests",
            "planned",
            False,
        ),
    ],
)
def test_classify_routes_request_kinds(
    user_input: str,
    expected_route: str,
    needs_clarification: bool,
) -> None:
    # make_classify() with no args uses rule-based fallback:
    # only returns simple or planned
    classify = make_classify()

    result = classify(
        {
            "user_input": user_input,
            "active_capabilities": {
                "skills": [],
                "tools": [],
                "mcp_servers": [],
                "mcp_tools": [],
            },
        }
    )

    assert result["route"] == expected_route
    assert result["needs_clarification"] is needs_clarification


def test_classify_rule_based_planned_markers() -> None:
    # "implement" and "update tests" both trigger planned route
    classify = make_classify()

    result = classify(
        {
            "user_input": "Implement the feature and update tests",
            "active_capabilities": {},
        }
    )

    assert result["route"] == "planned"
    assert result["needs_clarification"] is False


def test_classify_rule_based_simple_input() -> None:
    classify = make_classify()

    result = classify(
        {
            "user_input": "Say hello",
            "active_capabilities": {},
        }
    )

    assert result["route"] == "simple"
    assert result["needs_clarification"] is False
