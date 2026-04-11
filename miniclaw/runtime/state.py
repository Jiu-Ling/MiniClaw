from typing import Any, NotRequired, Required, TypedDict

from pydantic import BaseModel, ConfigDict, Field


class ActiveCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list)
    mcp_tools: list[str] = Field(default_factory=list)


class RuntimeMessage(TypedDict, total=False):
    role: Required[str]
    content: Required[str]
    name: NotRequired[str]
    tool_call_id: NotRequired[str]
    tool_calls: NotRequired[list[dict[str, Any]]]
    content_parts: NotRequired[list[dict[str, Any]]]


class RuntimeMetadata(TypedDict, total=False):
    thread_id: str
    channel: str
    chat_id: str


class RuntimeUsage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class FleetRun(TypedDict, total=False):
    fleet_id: str
    sub_id: str
    role: str
    status: str
    summary: str
    rounds_used: int
    usage: dict[str, int]


class RuntimeState(TypedDict, total=False):
    thread_id: str
    runtime_metadata: RuntimeMetadata
    user_input: str
    user_content_parts: list[dict[str, Any]]
    messages: list[RuntimeMessage]
    memory_context: str
    active_capabilities: ActiveCapabilities
    response_text: str
    usage: RuntimeUsage
    last_error: str
    planner_context: str
    plan_summary: str
    subagent_briefs: list[dict[str, Any]]
    executor_notes: str
    fleet_runs: list[FleetRun]
    needs_clarification: bool
    clarification_reason: str
    route: str
