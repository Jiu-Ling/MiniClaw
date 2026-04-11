from __future__ import annotations

import pytest

from miniclaw.runtime.nodes import route_after_classify, route_after_load_context, route_after_planner, route_on_error


def test_route_classify_clarify():
    assert route_after_classify({"needs_clarification": True}) == "clarify"


def test_route_classify_to_load_context():
    assert route_after_classify({"needs_clarification": False}) == "load_context"


def test_route_classify_error():
    assert route_after_classify({"last_error": "boom"}) == "error_handler"


def test_route_load_context_simple():
    assert route_after_load_context({"route": "simple"}) == "agent"


def test_route_load_context_planned():
    assert route_after_load_context({"route": "planned"}) == "planner"


def test_route_load_context_error():
    assert route_after_load_context({"last_error": "oops"}) == "error_handler"


def test_route_planner_ok():
    assert route_after_planner({}) == "agent"


def test_route_planner_error():
    assert route_after_planner({"last_error": "planner failed: x"}) == "error_handler"


def test_route_on_error_ok():
    fn = route_on_error("complete")
    assert fn({}) == "complete"
    assert fn({"last_error": ""}) == "complete"


def test_route_on_error_error():
    fn = route_on_error("complete")
    assert fn({"last_error": "something failed"}) == "error_handler"
