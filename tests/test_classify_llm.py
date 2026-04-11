from __future__ import annotations

import json
from miniclaw.runtime.nodes import make_classify, clarify, error_handler


class FakeClassifyProvider:
    def __init__(self, intent="simple", reason=""):
        self._intent = intent
        self._reason = reason

    class capabilities:
        vision = False

    async def achat(self, messages, *, model=None, tools=None):
        from miniclaw.providers.contracts import ChatResponse
        return ChatResponse(
            content="",
            provider="fake",
            tool_calls=[{
                "id": "call_1",
                "function": {
                    "name": "classify_intent",
                    "arguments": json.dumps({"intent": self._intent, "reason": self._reason}),
                },
            }],
        )


class FailingProvider:
    class capabilities:
        vision = False

    async def achat(self, messages, *, model=None, tools=None):
        raise ConnectionError("down")


def test_classify_simple_intent():
    c = make_classify(mini_provider=FakeClassifyProvider(intent="simple"))
    result = c({"user_input": "查一下天气", "messages": []})
    assert result["route"] == "simple"
    assert result["needs_clarification"] is False


def test_classify_planned_intent():
    c = make_classify(mini_provider=FakeClassifyProvider(intent="planned"))
    result = c({"user_input": "实现一个搜索系统", "messages": []})
    assert result["route"] == "planned"


def test_classify_clarify_intent():
    c = make_classify(mini_provider=FakeClassifyProvider(intent="clarify", reason="缺少操作对象"))
    result = c({"user_input": "改一下", "messages": []})
    assert result["route"] == "clarify"
    assert result["needs_clarification"] is True
    assert "缺少操作对象" in result["clarification_reason"]


def test_classify_mini_fails_falls_back_to_main():
    c = make_classify(mini_provider=FailingProvider(), main_provider=FakeClassifyProvider(intent="planned"))
    result = c({"user_input": "build something", "messages": []})
    assert result["route"] == "planned"


def test_classify_both_fail_falls_back_to_rules():
    c = make_classify(mini_provider=FailingProvider(), main_provider=FailingProvider())
    result = c({"user_input": "implement a feature", "messages": []})
    assert result["route"] == "planned"


def test_classify_no_provider_uses_rules():
    c = make_classify()
    result = c({"user_input": "read the README", "messages": []})
    assert result["route"] == "simple"


def test_classify_extracts_last_turn_summary():
    c = make_classify(mini_provider=FakeClassifyProvider(intent="simple"))
    messages = [
        {"role": "user", "content": "查天气"},
        {"role": "assistant", "content": "武汉 15°C"},
    ]
    result = c({"user_input": "继续", "messages": messages})
    assert result["route"] == "simple"


def test_clarify_with_reason():
    result = clarify({"user_input": "改一下", "clarification_reason": "缺少修改的目标对象"})
    assert "缺少修改的目标对象" in result["response_text"]
    assert result["last_error"] == ""


def test_clarify_without_reason():
    result = clarify({"user_input": "那个", "clarification_reason": ""})
    assert "那个" in result["response_text"]


def test_error_handler_planner_failed():
    result = error_handler({"last_error": "planner failed: connection refused"})
    assert "规划失败" in result["response_text"]


def test_error_handler_validation_failed():
    result = error_handler({"last_error": "planner validation failed: no execution task"})
    assert "不符合要求" in result["response_text"]


def test_error_handler_generic():
    result = error_handler({"last_error": "something broke"})
    assert "something broke" in result["response_text"]


