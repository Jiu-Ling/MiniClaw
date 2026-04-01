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


class PlannedTask(TypedDict):
    id: str
    title: str
    kind: str
    status: str
    worker_role: str
    parallel_group: str


class WorkerRun(TypedDict, total=False):
    id: str
    task_id: str
    role: str
    status: str
    summary: str
    result: str
    error: str


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
    request_kind: str
    needs_plan: bool
    context_profile: str
    planner_context: str
    plan_summary: str
    plan_steps: list[str]
    tasks: list[PlannedTask]
    worker_runs: list[WorkerRun]
    orchestration_status: str
    aggregated_worker_context: str
    executor_notes: str
    execution_mode: str
    suggested_capabilities: list[str]
    needs_clarification: bool
    clarification_reason: str
    route: str
