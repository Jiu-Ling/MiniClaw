from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from miniclaw.memory import build_memory_context
from miniclaw.memory.files import MemoryFileStore
from miniclaw.observability.contracts import NoopTracer, TraceContext
from miniclaw.persistence.memory_store import MemoryItem, MemoryStore
from miniclaw.prompting import ContextBuilder
from miniclaw.prompting.context import prompt_trace_scope
from miniclaw.runtime.checkpoint import AsyncSQLiteCheckpointer
from miniclaw.runtime.graph import build_graph
from miniclaw.runtime.state import ActiveCapabilities, RuntimeState, RuntimeUsage

if TYPE_CHECKING:
    from miniclaw.config.settings import Settings
    from miniclaw.memory.indexer import MemoryIndexer
    from miniclaw.memory.retriever import HybridRetriever
    from miniclaw.providers.contracts import ChatProvider
    from miniclaw.tools.registry import ToolRegistry

THREAD_SUMMARY_KIND = "thread_summary"
DURABLE_MEMORY_KINDS = {"fact", "preference", "project"}
SUMMARY_CONSOLIDATION_THRESHOLD = 3
SUMMARY_FETCH_LIMIT = 10


@dataclass(slots=True)
class TurnResult:
    thread_id: str
    response_text: str
    last_error: str
    usage: RuntimeUsage
    checkpoint_id: str | None = None


@dataclass(slots=True)
class StreamEvent:
    kind: str
    # kinds: "status", "chunk", "result",
    #        "thinking", "model_text", "tool_calling", "tool_done"
    text: str = ""
    result: TurnResult | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ResumeResult:
    thread_id: str
    response_text: str
    last_error: str
    usage: RuntimeUsage
    active_capabilities: ActiveCapabilities
    checkpoint_id: str | None = None
    message_count: int = 0


class RuntimeService:
    def __init__(
        self,
        *,
        settings: Settings,
        provider: ChatProvider,
        memory_store: MemoryStore,
        memory_file_store: MemoryFileStore | None = None,
        tool_registry: ToolRegistry | None = None,
        thread_control_store: object | None = None,
        memory_indexer: MemoryIndexer | None = None,
        retriever: HybridRetriever | None = None,
        mini_provider: object | None = None,
        tracer: object | None = None,
        clock: object | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.memory_store = memory_store
        self.memory_file_store = memory_file_store
        self.tool_registry = tool_registry
        self.thread_control_store = thread_control_store
        self.memory_indexer = memory_indexer
        self.retriever = retriever
        self.mini_provider = mini_provider
        self.tracer = tracer or NoopTracer()
        self.clock = clock if callable(clock) else _utc_now
        self._compression_synced_threads: set[str] = set()
        if memory_file_store is not None:
            setattr(self.memory_store, "memory_file_store", memory_file_store)

    def with_messaging_bridge(self, bridge: object) -> "RuntimeService":
        bound = copy.copy(self)
        tool_registry = getattr(self, "tool_registry", None)
        if tool_registry is None:
            return bound
        clone = tool_registry.clone()
        send_tool = clone.get("send") or clone.get("send_message")
        if send_tool is None:
            return bound
        from miniclaw.tools.builtin.send import build_send_tool

        clone.replace(build_send_tool(messaging_bridge=bridge))
        bound.tool_registry = clone
        return bound

    def get_thread_control(self, *, thread_id: str):
        if self.thread_control_store is None:
            raise RuntimeError("thread control store is not configured")
        return self.thread_control_store.get(thread_id)

    def set_thread_stopped(self, *, thread_id: str, stopped: bool = True):
        if self.thread_control_store is None:
            raise RuntimeError("thread control store is not configured")
        return self.thread_control_store.set_stopped(thread_id, stopped)

    def retry_last_turn(
        self,
        *,
        thread_id: str,
        runtime_metadata: Mapping[str, object] | None = None,
    ) -> TurnResult:
        checkpoint = self.resume_thread(thread_id=thread_id)
        if checkpoint is None:
            raise RuntimeError(f"no checkpoint found for thread={thread_id}")
        user_input, user_content_parts = self._extract_latest_user_turn(thread_id=thread_id)
        return self.run_turn(
            thread_id=thread_id,
            user_input=user_input,
            runtime_metadata=runtime_metadata,
            user_content_parts=user_content_parts or None,
        )

    def resume_run(
        self,
        *,
        thread_id: str,
        user_input: str,
        runtime_metadata: Mapping[str, object] | None = None,
    ) -> TurnResult:
        checkpoint = self.resume_thread(thread_id=thread_id)
        if checkpoint is None:
            raise RuntimeError(f"no checkpoint found for thread={thread_id}")
        result = self.run_turn(
            thread_id=thread_id,
            user_input=user_input,
            runtime_metadata=runtime_metadata,
        )
        if not str(result.last_error).strip():
            self.set_thread_stopped(thread_id=thread_id, stopped=False)
        return result

    def run_turn(
        self,
        *,
        thread_id: str,
        user_input: str,
        runtime_metadata: Mapping[str, object] | None = None,
        user_content_parts: list[dict[str, object]] | None = None,
    ) -> TurnResult:
        run_context = _safe_start_run(
            self.tracer,
            name="runtime.turn",
            thread_id=thread_id,
            metadata={"mode": "sync"},
        )
        checkpointer = AsyncSQLiteCheckpointer(self.settings.sqlite_path)
        graph = build_graph(
            settings=self.settings,
            provider=_TracingProviderProxy(
                provider=self.provider,
                tracer=self.tracer,
                parent_context=run_context,
            ),
            mini_provider=self.mini_provider,
            memory_store=self.memory_store,
            tool_registry=_wrap_tool_registry(
                self.tool_registry,
                tracer=self.tracer,
                parent_context=run_context,
            ),
            retriever=self.retriever,
            indexer=getattr(self, "memory_indexer", None),
            memory_token_budget=getattr(self.settings, "memory_token_budget", 2000),
        )
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = app.get_state(config)
        snapshot_values = snapshot.values if snapshot is not None else {}
        final_status = "error"
        final_output: dict[str, Any] = {}
        try:
            with prompt_trace_scope(tracer=self.tracer, context=run_context):
                result = app.invoke(
                    self._build_initial_state(
                        thread_id=thread_id,
                        user_input=user_input,
                        runtime_metadata=runtime_metadata,
                        user_content_parts=user_content_parts,
                        snapshot_values=snapshot_values,
                    ),
                    config,
                )
        except Exception as exc:
            final_output = {"error": str(exc)}
            _safe_record_event(
                self.tracer,
                run_context,
                name="runtime.error",
                payload=final_output,
                status="error",
            )
            _safe_finish_run(self.tracer, run_context, status=final_status, output=final_output)
            raise
        snapshot = app.get_state(config)
        checkpoint_id = None
        if snapshot is not None:
            checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")

        turn_result = TurnResult(
            thread_id=thread_id,
            response_text=str(result.get("response_text", "")),
            last_error=str(result.get("last_error", "")),
            usage=result.get("usage", {}),
            checkpoint_id=str(checkpoint_id) if checkpoint_id is not None else None,
        )
        self._remember_turn(
            thread_id=thread_id,
            user_input=user_input,
            result=turn_result,
            trace_context=run_context,
        )
        final_status = "error" if turn_result.last_error else "ok"
        final_output = {
            "response_text": turn_result.response_text,
            "last_error": turn_result.last_error,
            "usage": turn_result.usage,
            "checkpoint_id": turn_result.checkpoint_id,
            "orchestration_status": str(result.get("orchestration_status", "")),
            "worker_run_count": len(result.get("worker_runs", [])) if isinstance(result.get("worker_runs", []), list) else 0,
        }
        _safe_record_event(
            self.tracer,
            run_context,
            name="orchestration.status",
            payload={
                "status": str(result.get("orchestration_status", "")),
                "worker_run_count": len(result.get("worker_runs", [])) if isinstance(result.get("worker_runs", []), list) else 0,
            },
            status=final_status,
        )
        _safe_record_event(
            self.tracer,
            run_context,
            name="runtime.result",
            payload=final_output,
            status=final_status,
        )
        _safe_finish_run(self.tracer, run_context, status=final_status, output=final_output)
        return turn_result

    def run_turn_stream(
        self,
        *,
        thread_id: str,
        user_input: str,
        runtime_metadata: Mapping[str, object] | None = None,
        user_content_parts: list[dict[str, object]] | None = None,
    ) -> Iterator[StreamEvent]:
        run_context = _safe_start_run(
            self.tracer,
            name="runtime.turn.stream",
            thread_id=thread_id,
            metadata={"mode": "stream"},
        )
        yield StreamEvent(kind="thinking", text="🤔 MiniClaw is thinking...")
        _safe_record_event(
            self.tracer,
            run_context,
            name="stream.status",
            payload={"text": "thinking", "phase": "start"},
        )
        prepared_state = self._prepare_stream_state(
            thread_id=thread_id,
            user_input=user_input,
            runtime_metadata=runtime_metadata,
            user_content_parts=user_content_parts,
        )
        traced_provider = _TracingProviderProxy(
            provider=self.provider,
            tracer=self.tracer,
            parent_context=run_context,
        )
        traced_registry = _wrap_tool_registry(
            self.tool_registry,
            tracer=self.tracer,
            parent_context=run_context,
        )
        stream_method = getattr(traced_provider, "astream_text", None)
        if callable(stream_method) and self.tool_registry is None:
            yielded = False
            chunk_index = 0
            try:
                provider_messages = self._build_provider_messages(prepared_state, trace_context=run_context)
                for chunk in _iterate_async_iterator_sync(
                    stream_method(provider_messages, model=self.settings.model)
                ):
                    if not chunk:
                        continue
                    yielded = True
                    _safe_record_event(
                        self.tracer,
                        run_context,
                        name="stream.chunk",
                        payload={"index": chunk_index, "text": chunk},
                    )
                    chunk_index += 1
                    yield StreamEvent(kind="chunk", text=chunk)
                    prepared_state["response_text"] = str(prepared_state.get("response_text", "")) + chunk
                if yielded:
                    final_state = dict(prepared_state)
                    final_state["messages"] = list(prepared_state.get("messages", [])) + [
                        {"role": "assistant", "content": str(prepared_state.get("response_text", ""))}
                    ]
                    final_state["last_error"] = ""
                    final_state["usage"] = {}
                    result = self._persist_final_state(
                        thread_id=thread_id,
                        final_state=final_state,
                        trace_context=run_context,
                    )
                    final_payload = {
                        "response_text": result.response_text,
                        "last_error": result.last_error,
                        "usage": result.usage,
                        "checkpoint_id": result.checkpoint_id,
                    }
                    _safe_record_event(
                        self.tracer,
                        run_context,
                        name="runtime.result",
                        payload=final_payload,
                        status="ok",
                    )
                    _safe_finish_run(self.tracer, run_context, status="ok", output=final_payload)
                    yield StreamEvent(kind="result", result=result)
                    return
            except Exception:
                pass

        yield from self._run_status_stream_fallback(
            thread_id=thread_id,
            prepared_state=prepared_state,
            run_context=run_context,
            provider=traced_provider,
            tool_registry=traced_registry,
        )

    def _run_status_stream_fallback(
        self,
        *,
        thread_id: str,
        prepared_state: dict[str, Any],
        run_context: TraceContext,
        provider: object,
        tool_registry: object | None,
    ) -> Iterator[StreamEvent]:
        from miniclaw.runtime import nodes as runtime_nodes

        max_rounds = self.settings.max_tool_rounds
        max_errors = self.settings.max_consecutive_tool_errors
        max_result_chars = self.settings.max_tool_result_chars
        debug = getattr(self.settings, "debug", False)

        loop_state = dict(prepared_state)
        usage: RuntimeUsage = {}
        final_state: dict[str, Any] | None = None
        consecutive_errors = 0
        tool_history: list[dict[str, str]] = []  # accumulated tool call records

        try:
            for iteration in range(max_rounds):
                messages = self._build_provider_messages(loop_state, trace_context=run_context)
                visible_tools = runtime_nodes._build_provider_tools(
                    tool_registry,
                    runtime_nodes._coerce_active_capabilities(loop_state.get("active_capabilities")),
                )
                response = runtime_nodes._run_provider_sync(
                    runtime_nodes._invoke_provider(
                        provider,
                        messages,
                        model=self.settings.model,
                        tools=visible_tools,
                    )
                )
                usage = runtime_nodes._merge_usage(usage, response)

                if not response.tool_calls:
                    # Final response — no more tool calls
                    final_state = runtime_nodes._finish_response(loop_state, response, usage)
                    final_text = str(final_state.get("response_text", ""))
                    if final_text:
                        yield StreamEvent(kind="chunk", text=final_text)
                    break

                # Model returned tool calls — yield model's intermediate text + tool info
                model_text = str(response.content or "").strip()
                if model_text:
                    yield StreamEvent(kind="model_text", text=model_text)

                # Yield tool_calling events for each tool call
                for raw_call in response.tool_calls:
                    function = raw_call.get("function", {})
                    tool_name = str(function.get("name", "")).strip() or "tool"
                    tool_args = function.get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            import json as _json
                            tool_args = _json.loads(tool_args)
                        except Exception:
                            tool_args = {"raw": tool_args}
                    yield StreamEvent(
                        kind="tool_calling",
                        text=tool_name,
                        metadata={"tool_name": tool_name, "arguments": tool_args if debug else None},
                    )

                # Execute tool calls
                loop_state = runtime_nodes._apply_tool_calls(
                    loop_state, response, tool_registry, max_result_chars=max_result_chars,
                )

                # Extract tool results and yield tool_done events
                tool_messages = [
                    msg for msg in loop_state.get("messages", [])
                    if (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")) == "tool"
                ]
                recent_tool_msgs = tool_messages[-len(response.tool_calls):] if response.tool_calls else []
                for i, raw_call in enumerate(response.tool_calls):
                    function = raw_call.get("function", {})
                    tool_name = str(function.get("name", "")).strip() or "tool"
                    tool_args = function.get("arguments", {})
                    if isinstance(tool_args, str):
                        try:
                            import json as _json
                            tool_args = _json.loads(tool_args)
                        except Exception:
                            tool_args = {"raw": tool_args}
                    tool_result_text = ""
                    if i < len(recent_tool_msgs):
                        msg = recent_tool_msgs[i]
                        tool_result_text = msg.get("content", "") if isinstance(msg, dict) else str(getattr(msg, "content", ""))
                    record = {"tool_name": tool_name}
                    if debug:
                        record["arguments"] = str(tool_args)
                        record["result"] = tool_result_text[:500]
                    tool_history.append(record)
                    yield StreamEvent(
                        kind="tool_done",
                        text=tool_name,
                        metadata={
                            "tool_name": tool_name,
                            "arguments": tool_args if debug else None,
                            "result": tool_result_text[:500] if debug else None,
                        },
                    )

                # Check consecutive errors
                if recent_tool_msgs and runtime_nodes._is_error_content(recent_tool_msgs[-1]):
                    consecutive_errors += 1
                    if consecutive_errors >= max_errors:
                        final_state = runtime_nodes._error_state(
                            loop_state,
                            f"tool call failed {max_errors} consecutive times",
                            usage,
                        )
                        break
                else:
                    consecutive_errors = 0

            if final_state is None:
                final_state = runtime_nodes._error_state(
                    loop_state,
                    f"tool loop round limit reached after {max_rounds} rounds",
                    usage,
                )
        except Exception as exc:
            final_state = runtime_nodes._error_state(loop_state, runtime_nodes._format_error(exc), usage)

        result = self._persist_final_state(
            thread_id=thread_id,
            final_state=final_state,
            trace_context=run_context,
        )
        final_status = "error" if result.last_error else "ok"
        _safe_finish_run(self.tracer, run_context, status=final_status, output={
            "response_text": result.response_text,
            "last_error": result.last_error,
        })
        yield StreamEvent(kind="result", result=result, metadata={"tool_history": tool_history})

    def resume_thread(self, *, thread_id: str) -> ResumeResult | None:
        checkpointer = AsyncSQLiteCheckpointer(self.settings.sqlite_path)
        checkpoint = checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
        if checkpoint is None:
            return None

        checkpoint_id = checkpoint.config.get("configurable", {}).get("checkpoint_id")
        if checkpoint_id is None:
            return None

        channel_values = checkpoint.checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        usage = channel_values.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        active_capabilities = channel_values.get("active_capabilities")

        return ResumeResult(
            thread_id=thread_id,
            response_text=str(channel_values.get("response_text", "")),
            last_error=str(channel_values.get("last_error", "")),
            usage=usage,
            active_capabilities=ActiveCapabilities.model_validate(active_capabilities or {}),
            checkpoint_id=str(checkpoint_id),
            message_count=len(messages) if isinstance(messages, list) else 0,
        )

    def reset_thread(self, *, thread_id: str) -> None:
        checkpointer = AsyncSQLiteCheckpointer(self.settings.sqlite_path)
        checkpointer.delete_thread(thread_id)
        if self.thread_control_store is not None:
            self.thread_control_store.clear(thread_id)
        prune = getattr(self.memory_store, "prune", None)
        if callable(prune):
            prune(thread_id)

    def _extract_latest_user_turn(self, *, thread_id: str) -> tuple[str, list[dict[str, object]]]:
        checkpointer = AsyncSQLiteCheckpointer(self.settings.sqlite_path)
        checkpoint = checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
        if checkpoint is None:
            raise RuntimeError(f"no checkpoint found for thread={thread_id}")

        messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
        if not isinstance(messages, list):
            raise RuntimeError(f"no retryable user message found for thread={thread_id}")

        for message in reversed(messages):
            role, content, content_parts = self._extract_message_payload(message)
            if role == "user":
                text = self._extract_message_text(content, content_parts)
                if text or content_parts:
                    return text, content_parts

        raise RuntimeError(f"no retryable user message found for thread={thread_id}")

    @staticmethod
    def _extract_message_payload(message: object) -> tuple[str | None, object, list[dict[str, object]]]:
        if isinstance(message, dict):
            raw_parts = message.get("content_parts", [])
            content_parts = [dict(part) for part in raw_parts if isinstance(part, dict)]
            role = message.get("role")
            return (str(role) if role is not None else None), message.get("content", ""), content_parts

        raw_parts = getattr(message, "content_parts", [])
        content_parts = [dict(part) for part in raw_parts if isinstance(part, dict)]
        role = getattr(message, "role", None)
        content = getattr(message, "content", "")
        return (str(role) if role is not None else None), content, content_parts

    @staticmethod
    def _extract_message_text(content: object, content_parts: list[dict[str, object]]) -> str:
        if isinstance(content, str):
            text = RuntimeService._strip_runtime_metadata(content.strip())
            if text:
                return text
        for part in content_parts:
            if part.get("type") == "text":
                text = RuntimeService._strip_runtime_metadata(str(part.get("text", "")).strip())
                if text:
                    return text
        return ""

    @staticmethod
    def _strip_runtime_metadata(text: str) -> str:
        metadata_marker = "\n\n## Runtime Metadata"
        if metadata_marker in text:
            return text.split(metadata_marker, 1)[0]
        return text

    def _build_initial_state(
        self,
        *,
        thread_id: str,
        user_input: str,
        runtime_metadata: Mapping[str, object] | None,
        user_content_parts: list[dict[str, object]] | None,
        snapshot_values: Mapping[str, object],
    ) -> dict[str, object]:
        resolved_runtime_metadata = {"thread_id": thread_id}
        if runtime_metadata is not None:
            resolved_runtime_metadata.update(
                {str(key): value for key, value in runtime_metadata.items() if value is not None}
            )
        resolved_runtime_metadata.setdefault("clock", self.clock().isoformat())
        return {
            "thread_id": thread_id,
            "runtime_metadata": resolved_runtime_metadata,
            "user_input": user_input,
            "user_content_parts": list(user_content_parts or []),
            "messages": list(snapshot_values.get("messages", [])),
            "memory_context": str(snapshot_values.get("memory_context", "")),
            "active_capabilities": snapshot_values.get("active_capabilities") or ActiveCapabilities(),
            "response_text": "",
            "usage": {},
            "last_error": "",
            "request_kind": str(snapshot_values.get("request_kind", "")),
            "needs_plan": bool(snapshot_values.get("needs_plan", False)),
            "context_profile": str(snapshot_values.get("context_profile", "")),
            "planner_context": str(snapshot_values.get("planner_context", "")),
            "plan_summary": str(snapshot_values.get("plan_summary", "")),
            "plan_steps": list(snapshot_values.get("plan_steps", [])),
            "tasks": list(snapshot_values.get("tasks", [])),
            "worker_runs": list(snapshot_values.get("worker_runs", [])),
            "orchestration_status": str(snapshot_values.get("orchestration_status", "")),
            "aggregated_worker_context": str(snapshot_values.get("aggregated_worker_context", "")),
            "executor_notes": str(snapshot_values.get("executor_notes", "")),
            "execution_mode": str(snapshot_values.get("execution_mode", "")),
            "suggested_capabilities": list(snapshot_values.get("suggested_capabilities", [])),
        }

    def _prepare_stream_state(
        self,
        *,
        thread_id: str,
        user_input: str,
        runtime_metadata: Mapping[str, object] | None,
        user_content_parts: list[dict[str, object]] | None,
    ) -> dict[str, Any]:
        checkpointer = AsyncSQLiteCheckpointer(self.settings.sqlite_path)
        config = {"configurable": {"thread_id": thread_id}}
        graph = build_graph(
            settings=self.settings,
            provider=self.provider,
            mini_provider=self.mini_provider,
            memory_store=self.memory_store,
            tool_registry=self.tool_registry,
        )
        app = graph.compile(checkpointer=checkpointer)
        snapshot = app.get_state(config)
        snapshot_values = snapshot.values if snapshot is not None else {}
        state = self._build_initial_state(
            thread_id=thread_id,
            user_input=user_input,
            runtime_metadata=runtime_metadata,
            user_content_parts=user_content_parts,
            snapshot_values=snapshot_values,
        )
        from miniclaw.runtime.nodes import ingest

        state.update(ingest(state))
        thread_runtime_metadata = state["runtime_metadata"]
        state["memory_context"] = build_memory_context(
            self.memory_store,
            str(thread_runtime_metadata.get("thread_id", thread_id)),
        )
        return state

    def _build_provider_messages(
        self,
        state: Mapping[str, Any],
        *,
        trace_context: TraceContext | None = None,
    ) -> list[Any]:
        context_builder = ContextBuilder(
            system_prompt=self.settings.system_prompt,
            skills_loader=self.tool_registry.skill_loader if self.tool_registry is not None else None,
            tool_registry=self.tool_registry,
            mcp_registry=self.tool_registry.mcp_registry if self.tool_registry is not None else None,
            history_char_budget=getattr(self.settings, "history_char_budget", None),
            compress_keep_recent_turns=getattr(self.settings, "compress_keep_recent_turns", None),
            compress_turn_summary_chars=getattr(self.settings, "compress_turn_summary_chars", None),
        )
        if trace_context is None:
            messages = context_builder.build_provider_messages(state)
        else:
            with prompt_trace_scope(tracer=self.tracer, context=trace_context):
                messages = context_builder.build_provider_messages(state)

        compression_summary = getattr(context_builder, "_last_compression_summary", "")
        if compression_summary and self.memory_file_store is not None:
            thread_id = str(state.get("thread_id", ""))
            if thread_id and thread_id not in self._compression_synced_threads:
                self._compression_synced_threads.add(thread_id)
                self._sync_compression_to_memory(
                    thread_id=thread_id,
                    summary=compression_summary,
                    trace_context=trace_context,
                )

        return messages

    def _sync_compression_to_memory(
        self,
        *,
        thread_id: str,
        summary: str,
        trace_context: TraceContext | None = None,
    ) -> None:
        thread_key = f"thread:{thread_id}"
        try:
            self.memory_file_store.append_recent_work(thread_key, summary)
            _safe_record_event(
                self.tracer,
                trace_context,
                name="memory.compression_sync",
                payload={"thread_id": thread_id, "summary_length": len(summary)},
            )
        except Exception:
            pass

    def _persist_final_state(
        self,
        *,
        thread_id: str,
        final_state: dict[str, Any],
        trace_context: TraceContext | None = None,
    ) -> TurnResult:
        checkpointer = AsyncSQLiteCheckpointer(self.settings.sqlite_path)
        graph = StateGraph(RuntimeState)
        graph.add_node("persist", lambda state: state)
        graph.add_edge(START, "persist")
        graph.add_edge("persist", END)
        app = graph.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        result = app.invoke(dict(final_state), config)
        snapshot = app.get_state(config)
        checkpoint_id = None
        if snapshot is not None:
            checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")
        turn_result = TurnResult(
            thread_id=thread_id,
            response_text=str(result.get("response_text", "")),
            last_error=str(result.get("last_error", "")),
            usage=result.get("usage", {}),
            checkpoint_id=str(checkpoint_id) if checkpoint_id is not None else None,
        )
        self._remember_turn(
            thread_id=thread_id,
            user_input=self._resolve_user_input(final_state),
            result=turn_result,
            trace_context=trace_context,
        )
        return turn_result

    def _remember_turn(
        self,
        *,
        thread_id: str,
        user_input: str,
        result: TurnResult,
        trace_context: TraceContext | None = None,
    ) -> None:
        append_fact = getattr(self.memory_store, "append_fact", None)
        if not callable(append_fact):
            return

        memory_span = None
        if trace_context is not None:
            memory_span = _safe_start_span(
                self.tracer,
                trace_context,
                name="memory.record_turn",
                metadata={"thread_id": thread_id},
            )
        now = self.clock()
        summary = self._build_thread_digest(now=now, user_input=user_input, result=result)
        timestamp = self._format_timestamp(now)
        _safe_record_event(
            self.tracer,
            memory_span or trace_context,
            name="memory.thread_summary",
            payload={
                "thread_id": thread_id,
                "summary": summary,
                "timestamp": timestamp,
                "failed": bool(str(result.last_error).strip()),
            },
        )

        self._safe_memory_write(
            lambda: append_fact(
                thread_id,
                summary,
                THREAD_SUMMARY_KIND,
                {
                    "source": "runtime",
                    "thread_id": thread_id,
                    "timestamp": timestamp,
                    "failed": bool(str(result.last_error).strip()),
                },
            )
        )

        extracted = self._extract_durable_facts(user_input=user_input, timestamp=timestamp)
        if extracted:
            _safe_record_event(
                self.tracer,
                memory_span or trace_context,
                name="memory.extracted_facts",
                payload={"facts": extracted},
            )
        high_importance = False
        for item in extracted:
            importance = str(item["metadata"].get("importance", "")).strip().lower()
            high_importance = high_importance or importance == "high"
            self._safe_memory_write(
                lambda payload=item: append_fact(
                    thread_id,
                    str(payload["content"]),
                    str(payload["kind"]),
                    dict(payload["metadata"]),
                )
            )

        if self.memory_file_store is None:
            _safe_finish_span(
                self.tracer,
                memory_span,
                status="ok",
                output={"extracted_fact_count": len(extracted), "consolidated": False},
            )
            return
        if not self._should_consolidate(thread_id=thread_id, high_importance=high_importance):
            _safe_finish_span(
                self.tracer,
                memory_span,
                status="ok",
                output={"extracted_fact_count": len(extracted), "consolidated": False},
            )
            return
        consolidate_span = None
        if memory_span is not None:
            consolidate_span = _safe_start_span(
                self.tracer,
                memory_span,
                name="memory.consolidate",
                metadata={"thread_id": thread_id},
            )
        self._safe_memory_write(lambda: self._consolidate_thread_memory(thread_id, trace_context=consolidate_span))
        _safe_finish_span(
            self.tracer,
            consolidate_span,
            status="ok",
            output={"thread_id": thread_id},
        )
        _safe_finish_span(
            self.tracer,
            memory_span,
            status="ok",
            output={"extracted_fact_count": len(extracted), "consolidated": True},
        )

    def _should_consolidate(self, *, thread_id: str, high_importance: bool) -> bool:
        if high_importance:
            return True
        summaries = self._list_recent_by_kind(
            thread_id,
            THREAD_SUMMARY_KIND,
            limit=SUMMARY_CONSOLIDATION_THRESHOLD,
        )
        return len(summaries) >= SUMMARY_CONSOLIDATION_THRESHOLD

    def _consolidate_thread_memory(self, thread_id: str, trace_context: TraceContext | None = None) -> None:
        if self.memory_file_store is None:
            return

        document = self.memory_file_store.read()
        summaries = list(
            reversed(
                self._list_recent_by_kind(
                    thread_id,
                    THREAD_SUMMARY_KIND,
                    limit=self.memory_file_store.recent_work_limit,
                )
            )
        )
        durable_facts = list(reversed(self._list_recent_durable_facts(thread_id, limit=SUMMARY_FETCH_LIMIT)))

        long_term_facts = self._merge_fact_lines(
            document.long_term_facts,
            [item.content for item in durable_facts],
        )
        recent_work = dict(document.recent_work)
        if summaries:
            recent_work[f"thread:{thread_id}"] = [item.content for item in summaries]

        self.memory_file_store.update(
            long_term_facts=long_term_facts,
            recent_work=recent_work,
        )
        self._write_daily_md(thread_id=thread_id, summaries=summaries)
        _safe_record_event(
            self.tracer,
            trace_context,
            name="memory.consolidated",
            payload={
                "thread_id": thread_id,
                "long_term_facts": long_term_facts,
                "recent_work": recent_work,
            },
        )

    def _write_daily_md(
        self,
        *,
        thread_id: str,
        summaries: list,
    ) -> None:
        if not summaries:
            return
        settings = self.settings
        runtime_dir = getattr(settings, "runtime_dir", None)
        if runtime_dir is None:
            return

        today = self.clock().strftime("%Y-%m-%d")
        memory_dir = runtime_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        daily_path = memory_dir / f"{today}.md"

        if not daily_path.exists():
            daily_path.write_text(f"# {today}\n", encoding="utf-8")

        section = f"\n### thread:{thread_id}\n"
        section += "\n".join(f"- [summary] {s.content}" for s in summaries)
        section += "\n"

        with daily_path.open("a", encoding="utf-8") as f:
            f.write(section)

        if self.memory_indexer is not None:
            mark_dirty = getattr(self.memory_indexer, "mark_dirty", None)
            if callable(mark_dirty):
                mark_dirty(f"{today}.md")

    def _list_recent_durable_facts(self, thread_id: str, *, limit: int) -> list[MemoryItem]:
        items = self._list_recent_items(thread_id, limit=limit)
        return [item for item in items if item.kind in DURABLE_MEMORY_KINDS]

    def _list_recent_by_kind(self, thread_id: str, kind: str, *, limit: int) -> list[MemoryItem]:
        list_recent_by_kind = getattr(self.memory_store, "list_recent_by_kind", None)
        if callable(list_recent_by_kind):
            return list_recent_by_kind(thread_id, kind, limit)
        return [item for item in self._list_recent_items(thread_id, limit=limit * 3) if item.kind == kind][:limit]

    def _list_recent_items(self, thread_id: str, *, limit: int) -> list[MemoryItem]:
        list_recent = getattr(self.memory_store, "list_recent", None)
        if not callable(list_recent):
            return []
        return list_recent(thread_id, limit)

    def _extract_durable_facts(
        self,
        *,
        user_input: str,
        timestamp: str,
    ) -> list[dict[str, object]]:
        text = " ".join(user_input.split())
        if not text:
            return []

        normalized = text.rstrip(". ")
        lowered = normalized.lower()
        items: list[dict[str, object]] = []

        if "remember that " in lowered:
            remembered = normalized[lowered.index("remember that ") + len("remember that ") :]
            extracted = self._classify_durable_fact(remembered, importance="high", timestamp=timestamp)
            if extracted is not None:
                items.append(extracted)
            return items

        if lowered.startswith("prefer "):
            extracted = self._classify_durable_fact(normalized, importance="normal", timestamp=timestamp)
            if extracted is not None:
                items.append(extracted)

        return items

    def _classify_durable_fact(
        self,
        text: str,
        *,
        importance: str,
        timestamp: str,
    ) -> dict[str, object] | None:
        normalized = self._canonicalize_fact_text(text)
        lowered = normalized.lower()
        if not normalized:
            return None

        kind = "fact"
        if lowered.startswith("prefer "):
            kind = "preference"
        elif lowered.startswith("use ") or lowered.startswith("only support "):
            kind = "project"

        return {
            "content": normalized,
            "kind": kind,
            "metadata": {
                "importance": importance,
                "source": "runtime",
                "timestamp": timestamp,
            },
        }

    _DIGEST_OUTCOME_MAX_CHARS = 300

    def _build_thread_digest(
        self,
        *,
        now: datetime,
        user_input: str,
        result: TurnResult,
    ) -> str:
        user_summary = self._summarize_user_input(user_input)
        if str(result.last_error).strip():
            outcome = f"failed: {result.last_error}"
        else:
            outcome = result.response_text.strip()
        if not outcome:
            outcome = "completed"
        if len(outcome) > self._DIGEST_OUTCOME_MAX_CHARS:
            outcome = outcome[: self._DIGEST_OUTCOME_MAX_CHARS] + "..."
        return f"{self._format_timestamp(now)}: User asked to {user_summary}; outcome: {outcome}"

    def _resolve_user_input(self, final_state: Mapping[str, Any]) -> str:
        user_input = str(final_state.get("user_input", "")).strip()
        if user_input:
            return user_input
        messages = final_state.get("messages", [])
        if isinstance(messages, list):
            for message in reversed(messages):
                role, content, content_parts = self._extract_message_payload(message)
                if role == "user":
                    return self._extract_message_text(content, content_parts)
        return ""

    @staticmethod
    def _summarize_user_input(user_input: str) -> str:
        text = " ".join(user_input.split()).rstrip(". ")
        if not text:
            return "continue the thread"
        text = text[:96].strip()
        if text:
            text = text[0].lower() + text[1:]
        return text

    @staticmethod
    def _canonicalize_fact_text(text: str) -> str:
        normalized = " ".join(text.split()).strip()
        if not normalized:
            return ""
        normalized = re.sub(r"^(please|that)\s+", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"^we\s+(use|only support)\s+", r"\1 ", normalized, flags=re.IGNORECASE)
        normalized = normalized.rstrip(". ")
        if normalized:
            normalized = normalized[0].upper() + normalized[1:]
        return f"{normalized}."

    @staticmethod
    def _merge_fact_lines(existing: list[str], incoming: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for line in [*existing, *incoming]:
            normalized = " ".join(str(line).split()).strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(normalized)
        return merged

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _safe_memory_write(action) -> None:
        try:
            action()
        except Exception:
            return


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_start_run(
    tracer: object,
    *,
    name: str,
    thread_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TraceContext:
    try:
        return tracer.start_run(name=name, thread_id=thread_id, metadata=metadata)
    except Exception:
        return NoopTracer().start_run(name=name, thread_id=thread_id, metadata=metadata)


def _safe_finish_run(
    tracer: object,
    context: TraceContext | None,
    *,
    status: str,
    output: Mapping[str, Any] | None = None,
) -> None:
    if context is None:
        return
    try:
        tracer.finish_run(context, status=status, output=output)
    except Exception:
        return


def _safe_start_span(
    tracer: object,
    parent: TraceContext | None,
    *,
    name: str,
    metadata: Mapping[str, Any] | None = None,
) -> TraceContext | None:
    if parent is None:
        return None
    try:
        return tracer.start_span(parent, name=name, metadata=metadata)
    except Exception:
        return NoopTracer().start_span(parent, name=name, metadata=metadata)


def _safe_finish_span(
    tracer: object,
    context: TraceContext | None,
    *,
    status: str,
    output: Mapping[str, Any] | None = None,
) -> None:
    if context is None:
        return
    try:
        tracer.finish_span(context, status=status, output=output)
    except Exception:
        return


def _safe_record_event(
    tracer: object,
    context: TraceContext | None,
    *,
    name: str,
    payload: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    status: str | None = None,
) -> None:
    if context is None:
        return
    try:
        tracer.record_event(context, name=name, payload=payload, metadata=metadata, status=status)
    except Exception:
        return



def _wrap_tool_registry(
    tool_registry: object | None,
    *,
    tracer: object,
    parent_context: TraceContext,
) -> object | None:
    if tool_registry is None:
        return None
    return _TracingToolRegistryProxy(
        tool_registry=tool_registry,
        tracer=tracer,
        parent_context=parent_context,
    )


class _TracingProviderProxy:
    def __init__(self, *, provider: object, tracer: object, parent_context: TraceContext) -> None:
        self._provider = provider
        self._tracer = tracer
        self._parent_context = parent_context
        self.capabilities = getattr(provider, "capabilities", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    async def achat(
        self,
        messages: list[Any],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ):
        span = _safe_start_span(
            self._tracer,
            self._parent_context,
            name="provider.chat",
            metadata={
                "model": model,
                "message_count": len(messages),
                "tool_count": len(tools or []),
            },
        )
        _safe_record_event(
            self._tracer,
            span or self._parent_context,
            name="provider.request",
            payload={
                "messages": _serialize_messages(messages),
                "model": model,
                "tools": [dict(tool) for tool in tools or []],
            },
        )
        try:
            response = await self._provider.achat(messages, model=model, tools=tools)
        except Exception as exc:
            _safe_record_event(
                self._tracer,
                span or self._parent_context,
                name="provider.error",
                payload={"error": str(exc)},
                status="error",
            )
            _safe_finish_span(self._tracer, span, status="error", output={"error": str(exc)})
            raise
        _safe_record_event(
            self._tracer,
            span or self._parent_context,
            name="provider.response",
            payload=_serialize_response(response),
        )
        _safe_finish_span(
            self._tracer,
            span,
            status="ok",
            output={
                "response_text": str(getattr(response, "content", "")),
                "tool_call_count": len(getattr(response, "tool_calls", []) or []),
                "usage": _serialize_usage(getattr(response, "usage", None)),
            },
        )
        return response

    async def astream_text(
        self,
        messages: list[Any],
        *,
        model: str | None = None,
    ):
        span = _safe_start_span(
            self._tracer,
            self._parent_context,
            name="provider.stream",
            metadata={"model": model, "message_count": len(messages)},
        )
        _safe_record_event(
            self._tracer,
            span or self._parent_context,
            name="provider.request",
            payload={"messages": _serialize_messages(messages), "model": model},
        )
        chunk_count = 0
        try:
            async for chunk in self._provider.astream_text(messages, model=model):
                _safe_record_event(
                    self._tracer,
                    span or self._parent_context,
                    name="provider.stream.chunk",
                    payload={"index": chunk_count, "text": str(chunk)},
                )
                chunk_count += 1
                yield str(chunk)
        except Exception as exc:
            _safe_record_event(
                self._tracer,
                span or self._parent_context,
                name="provider.error",
                payload={"error": str(exc)},
                status="error",
            )
            _safe_finish_span(self._tracer, span, status="error", output={"error": str(exc)})
            raise
        _safe_finish_span(self._tracer, span, status="ok", output={"chunk_count": chunk_count})


class _TracingToolRegistryProxy:
    def __init__(self, *, tool_registry: object, tracer: object, parent_context: TraceContext) -> None:
        self._tool_registry = tool_registry
        self._tracer = tracer
        self._parent_context = parent_context
        self.skill_loader = getattr(tool_registry, "skill_loader", None)
        self.mcp_registry = getattr(tool_registry, "mcp_registry", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool_registry, name)

    def execute(self, call: object, active_capabilities: object | None = None):
        tool_name = str(getattr(call, "name", "")).strip() or "tool"
        arguments = dict(getattr(call, "arguments", {}) or {})
        spec = None
        get_tool = getattr(self._tool_registry, "get", None)
        if callable(get_tool):
            registered = get_tool(tool_name)
            spec = getattr(registered, "spec", None)
        span = _safe_start_span(
            self._tracer,
            self._parent_context,
            name=f"tool.call.{tool_name}",
            metadata={
                "tool_name": tool_name,
                "tool_source": getattr(spec, "source", None),
            },
        )
        _safe_record_event(
            self._tracer,
            span or self._parent_context,
            name="tool.call",
            payload={
                "tool_name": tool_name,
                "tool_source": getattr(spec, "source", None),
                "arguments": arguments,
            },
        )
        try:
            result = self._tool_registry.execute(call, active_capabilities)
        except Exception as exc:
            _safe_record_event(
                self._tracer,
                span or self._parent_context,
                name="tool.error",
                payload={"tool_name": tool_name, "error": str(exc)},
                status="error",
            )
            _safe_finish_span(self._tracer, span, status="error", output={"error": str(exc)})
            raise
        _safe_record_event(
            self._tracer,
            span or self._parent_context,
            name="tool.result",
            payload={
                "tool_name": tool_name,
                "content": str(getattr(result, "content", "")),
                "is_error": bool(getattr(result, "is_error", False)),
                "metadata": dict(getattr(result, "metadata", {}) or {}),
            },
        )
        _safe_finish_span(
            self._tracer,
            span,
            status="error" if bool(getattr(result, "is_error", False)) else "ok",
            output={
                "is_error": bool(getattr(result, "is_error", False)),
                "content": str(getattr(result, "content", "")),
            },
        )
        return result


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        if hasattr(message, "model_dump"):
            serialized.append(message.model_dump(exclude_none=True))
            continue
        if isinstance(message, Mapping):
            serialized.append({str(key): value for key, value in message.items()})
            continue
        serialized.append(
            {
                "role": str(getattr(message, "role", "")),
                "content": getattr(message, "content", None),
                "content_parts": list(getattr(message, "content_parts", []) or []),
                "tool_calls": list(getattr(message, "tool_calls", []) or []),
                "name": getattr(message, "name", None),
                "tool_call_id": getattr(message, "tool_call_id", None),
            }
        )
    return serialized


def _serialize_response(response: object) -> dict[str, Any]:
    return {
        "provider": str(getattr(response, "provider", "")),
        "model": getattr(response, "model", None),
        "content": str(getattr(response, "content", "")),
        "content_parts": [dict(part) for part in getattr(response, "content_parts", []) or []],
        "tool_calls": [dict(call) for call in getattr(response, "tool_calls", []) or []],
        "usage": _serialize_usage(getattr(response, "usage", None)),
        "raw": _normalize_mapping(getattr(response, "raw", {})),
    }


def _serialize_usage(usage: object) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    if isinstance(usage, Mapping):
        return {str(key): value for key, value in usage.items()}
    return {}


def _normalize_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _iterate_async_iterator_sync(async_iterator) -> Iterator[str]:
    from queue import Queue
    from threading import Thread

    queue: Queue[tuple[str, object | None]] = Queue()

    async def produce() -> None:
        try:
            async for item in async_iterator:
                queue.put(("chunk", str(item)))
            queue.put(("done", None))
        except Exception as exc:
            queue.put(("error", exc))

    def runner() -> None:
        asyncio.run(produce())

    thread = Thread(target=runner, daemon=True)
    thread.start()
    try:
        while True:
            kind, payload = queue.get()
            if kind == "chunk":
                yield str(payload)
                continue
            if kind == "done":
                return
            if kind == "error":
                raise payload  # type: ignore[misc]
    finally:
        thread.join()
