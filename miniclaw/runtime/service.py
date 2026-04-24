from __future__ import annotations

from collections.abc import Iterator, Mapping
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.base import Checkpoint
from langgraph.graph import END, START, StateGraph

from miniclaw.memory.files import MemoryFileStore
from miniclaw.observability.contracts import NoopTracer, TraceContext
from miniclaw.observability.safe import (
    safe_finish_run,
    safe_finish_span,
    safe_record_event,
    safe_start_run,
    safe_start_span,
)
from miniclaw.persistence.memory_store import MemoryItem, MemoryStore
from miniclaw.prompting.context import prompt_trace_scope
from miniclaw.runtime.checkpoint import AsyncSQLiteCheckpointer
from miniclaw.runtime.graph import build_graph
from miniclaw.runtime.state import ActiveCapabilities, RuntimeState, RuntimeUsage

if TYPE_CHECKING:
    from miniclaw.config.settings import Settings
    from miniclaw.memory.indexer import MemoryIndexer
    from miniclaw.memory.retriever import HybridRetriever
    from miniclaw.observability.contracts import Tracer
    from miniclaw.providers.contracts import ChatProvider, ChatResponse
    from miniclaw.providers.openai_compat import OpenAICompatibleProvider
    from miniclaw.runtime.background import BackgroundScheduler
    from miniclaw.runtime.thread_control import SQLiteThreadControlStore
    from miniclaw.tools.contracts import ToolCall, ToolResult
    from miniclaw.tools.messaging import MessagingBridge
    from miniclaw.tools.registry import ToolRegistry

THREAD_SUMMARY_KIND = "thread_summary"
SUMMARY_CONSOLIDATION_THRESHOLD = 3


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
        thread_control_store: SQLiteThreadControlStore | None = None,
        memory_indexer: MemoryIndexer | None = None,
        retriever: HybridRetriever | None = None,
        mini_provider: OpenAICompatibleProvider | None = None,
        tracer: TraceContext | None = None,
        clock: Callable[[], datetime] | None = None,
        background_scheduler: BackgroundScheduler | None = None,
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
        self.background_scheduler = background_scheduler
        self._checkpointer = AsyncSQLiteCheckpointer(settings.sqlite_path)
        self._persist_app = self._build_persist_app()
        self._compression_synced_threads: set[str] = set()
        if memory_file_store is not None:
            setattr(self.memory_store, "memory_file_store", memory_file_store)

    def _build_persist_app(self):
        """Build a minimal graph used only for persisting state to checkpoint."""
        graph = StateGraph(RuntimeState)
        graph.add_node("persist", lambda state: state)
        graph.add_edge(START, "persist")
        graph.add_edge("persist", END)
        return graph.compile(checkpointer=self._checkpointer)

    def with_messaging_bridge(self, bridge: MessagingBridge) -> "RuntimeService":
        bound = copy.copy(self)
        tool_registry = self.tool_registry
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
        run_context = safe_start_run(
            self.tracer,
            name="runtime.turn",
            thread_id=thread_id,
            metadata={"mode": "sync"},
        )
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
            indexer=self.memory_indexer,
            memory_token_budget=self.settings.memory_token_budget,
            tracer=self.tracer,
        )
        app = graph.compile(checkpointer=self._checkpointer)
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        langchain_callbacks = _build_langchain_callbacks(self.tracer)
        if langchain_callbacks:
            config["callbacks"] = langchain_callbacks
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
                        turn_trace=run_context,
                    ),
                    config,
                )
        except Exception as exc:
            final_output = {"error": str(exc)}
            safe_record_event(
                self.tracer,
                run_context,
                name="runtime.error",
                payload=final_output,
                status="error",
            )
            safe_finish_run(self.tracer, run_context, status=final_status, output=final_output)
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
        fleet_runs = result.get("fleet_runs", [])
        final_output = {
            "response_text": turn_result.response_text,
            "last_error": turn_result.last_error,
            "usage": turn_result.usage,
            "checkpoint_id": turn_result.checkpoint_id,
            "fleet_run_count": len(fleet_runs) if isinstance(fleet_runs, list) else 0,
        }
        safe_record_event(
            self.tracer,
            run_context,
            name="fleet.status",
            payload={
                "fleet_run_count": len(fleet_runs) if isinstance(fleet_runs, list) else 0,
            },
            status=final_status,
        )
        safe_record_event(
            self.tracer,
            run_context,
            name="runtime.result",
            payload=final_output,
            status=final_status,
        )
        safe_finish_run(self.tracer, run_context, status=final_status, output=final_output)
        return turn_result

    def run_turn_stream(
        self,
        *,
        thread_id: str,
        user_input: str,
        runtime_metadata: Mapping[str, object] | None = None,
        user_content_parts: list[dict[str, object]] | None = None,
    ) -> Iterator[StreamEvent]:
        from queue import Queue
        from threading import Thread

        run_context = safe_start_run(
            self.tracer,
            name="runtime.turn.stream",
            thread_id=thread_id,
            metadata={"mode": "stream"},
        )
        yield StreamEvent(kind="thinking", text="🤔 MiniClaw is thinking...")

        event_queue: Queue[StreamEvent | None] = Queue()

        def on_event(raw: dict[str, Any]) -> None:
            kind = str(raw.get("kind", ""))
            text = str(raw.get("text", ""))
            metadata = {k: v for k, v in raw.items() if k not in ("kind", "text")}
            event_queue.put(StreamEvent(kind=kind, text=text, metadata=metadata or None))

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
            indexer=self.memory_indexer,
            memory_token_budget=self.settings.memory_token_budget,
            on_event=on_event,
            tracer=self.tracer,
        )
        app = graph.compile(checkpointer=self._checkpointer)
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        langchain_callbacks = _build_langchain_callbacks(self.tracer)
        if langchain_callbacks:
            config["callbacks"] = langchain_callbacks

        snapshot = app.get_state(config)
        snapshot_values = snapshot.values if snapshot is not None else {}
        initial_state = self._build_initial_state(
            thread_id=thread_id,
            user_input=user_input,
            runtime_metadata=runtime_metadata,
            user_content_parts=user_content_parts,
            snapshot_values=snapshot_values,
            turn_trace=run_context,
        )

        result_holder: list[dict[str, Any] | None] = [None]
        error_holder: list[Exception | None] = [None]

        def run_graph() -> None:
            try:
                from miniclaw.prompting.context import prompt_trace_scope
                with prompt_trace_scope(tracer=self.tracer, context=run_context):
                    result_holder[0] = app.invoke(initial_state, config)
            except Exception as exc:
                error_holder[0] = exc
            finally:
                event_queue.put(None)

        thread = Thread(target=run_graph, daemon=True)
        thread.start()

        while True:
            event = event_queue.get()
            if event is None:
                break
            yield event

        thread.join()

        if error_holder[0] is not None:
            final_output = {"error": str(error_holder[0])}
            safe_finish_run(self.tracer, run_context, status="error", output=final_output)
            yield StreamEvent(
                kind="result",
                result=TurnResult(
                    thread_id=thread_id,
                    response_text="",
                    last_error=str(error_holder[0]),
                    usage={},
                ),
            )
            return

        result_state = result_holder[0] or {}
        snapshot = app.get_state(config)
        checkpoint_id = None
        if snapshot is not None:
            checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")

        turn_result = TurnResult(
            thread_id=thread_id,
            response_text=str(result_state.get("response_text", "")),
            last_error=str(result_state.get("last_error", "")),
            usage=result_state.get("usage", {}),
            checkpoint_id=str(checkpoint_id) if checkpoint_id is not None else None,
        )
        self._remember_turn(
            thread_id=thread_id,
            user_input=user_input,
            result=turn_result,
            trace_context=run_context,
        )
        final_status = "error" if turn_result.last_error else "ok"
        safe_finish_run(self.tracer, run_context, status=final_status, output={
            "response_text": turn_result.response_text,
            "last_error": turn_result.last_error,
        })
        yield StreamEvent(kind="result", result=turn_result)

    def resume_thread(self, *, thread_id: str) -> ResumeResult | None:
        checkpoint = self._checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
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
        self._checkpointer.delete_thread(thread_id)
        if self.thread_control_store is not None:
            self.thread_control_store.clear(thread_id)
        self.memory_store.prune(thread_id)

    def _extract_latest_user_turn(self, *, thread_id: str) -> tuple[str, list[dict[str, object]]]:
        checkpoint = self._checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
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

        raw_parts = message.content_parts if hasattr(message, "content_parts") else []
        content_parts = [dict(part) for part in raw_parts if isinstance(part, dict)]
        role = message.role if hasattr(message, "role") else None
        content = message.content if hasattr(message, "content") else ""
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
        turn_trace: TraceContext | None = None,
    ) -> dict[str, object]:
        resolved_runtime_metadata = {"thread_id": thread_id}
        if runtime_metadata is not None:
            resolved_runtime_metadata.update(
                {str(key): value for key, value in runtime_metadata.items() if value is not None}
            )
        resolved_runtime_metadata.setdefault("clock", self.clock().isoformat())
        # Stash the turn-level trace_id / run_id so downstream nodes and
        # agent/subagent span builders share one trace. Underscore keys are
        # stripped from tool span payloads by `_safe_copy_for_trace` so they
        # never leak into persisted trace records.
        if turn_trace is not None:
            resolved_runtime_metadata["_turn_trace_id"] = turn_trace.trace_id
            resolved_runtime_metadata["_turn_run_id"] = turn_trace.run_id
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
            "planner_context": str(snapshot_values.get("planner_context", "")),
            "plan_summary": str(snapshot_values.get("plan_summary", "")),
            "subagent_briefs": list(snapshot_values.get("subagent_briefs", [])),
            "executor_notes": str(snapshot_values.get("executor_notes", "")),
            "fleet_runs": list(snapshot_values.get("fleet_runs", [])),
        }

    def _persist_final_state(
        self,
        *,
        thread_id: str,
        final_state: dict[str, Any],
        trace_context: TraceContext | None = None,
    ) -> TurnResult:
        config = {"configurable": {"thread_id": thread_id}}
        result = self._persist_app.invoke(dict(final_state), config)
        snapshot = self._persist_app.get_state(config)
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
        now = self.clock()
        ts = self._format_timestamp(now)
        user_short = " ".join(user_input.split())[:120]
        outcome = str(result.last_error).strip() or str(result.response_text).strip()[:120] or "completed"
        digest = f"{ts}: {user_short} | {outcome}"

        self._safe_memory_write(
            lambda: self.memory_store.append_fact(
                thread_id, digest, THREAD_SUMMARY_KIND,
                {"source": "runtime", "thread_id": thread_id, "timestamp": ts},
            )
        )

        if not self._should_trigger_consolidation(thread_id):
            return

        from miniclaw.memory.consolidation import llm_consolidate, regex_consolidate_fallback
        from miniclaw.runtime.background import BackgroundJob

        memory_path = self.settings.runtime_dir / "MEMORY.md"
        daily_dir = self.settings.runtime_dir / "memory"
        threshold = self.settings.memory_consolidation_trigger_threshold
        digests = [
            item.content for item in
            self.memory_store.list_recent_by_kind(thread_id, THREAD_SUMMARY_KIND, threshold)
        ]

        provider = self._select_consolidation_provider()
        model = self._resolve_consolidation_model()

        def _run():
            import asyncio
            asyncio.run(llm_consolidate(
                thread_id=thread_id, provider=provider, model=model,
                memory_path=memory_path, digests=digests,
                recent_exchanges=[], daily_dir=daily_dir,
                indexer=self.memory_indexer,
                critical_max=self.settings.memory_critical_facts_max,
            ))

        def _fallback(exc):
            regex_consolidate_fallback(
                thread_id=thread_id, memory_path=memory_path, digests=digests,
                daily_dir=daily_dir,
            )

        job = BackgroundJob(
            fn=_run if provider else lambda: _fallback(None),
            kind="memory.consolidate",
            metadata={"thread_id": thread_id},
            on_failure=_fallback,
            parent_trace=trace_context,
        )
        self._dispatch_background(job)

    def _should_trigger_consolidation(self, thread_id: str) -> bool:
        if not self.settings.memory_consolidation_enabled:
            return False
        threshold = self.settings.memory_consolidation_trigger_threshold
        summaries = self.memory_store.list_recent_by_kind(thread_id, THREAD_SUMMARY_KIND, threshold)
        return len(summaries) >= threshold

    def _select_consolidation_provider(self):
        tier = self.settings.memory_consolidation_model_tier
        if tier == "mini":
            return self.mini_provider
        if tier == "main":
            return self.provider
        return self.mini_provider or self.provider

    def _resolve_consolidation_model(self) -> str:
        tier = self.settings.memory_consolidation_model_tier
        if tier == "main":
            return self.settings.model
        mini = self.settings.mini_model
        return mini if mini else self.settings.model

    def _list_recent_by_kind(self, thread_id: str, kind: str, *, limit: int) -> list[MemoryItem]:
        return self.memory_store.list_recent_by_kind(thread_id, kind, limit)

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
    def _format_timestamp(value: datetime) -> str:
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _safe_memory_write(action) -> None:
        try:
            action()
        except Exception:
            return

    def _dispatch_background(self, job) -> None:
        """Submit a BackgroundJob to the scheduler, or run it synchronously.

        When background_scheduler is configured, submits the job. On submit
        failure (e.g., queue full raising, although the implementation
        returns False), or when no scheduler is set, runs job.fn() directly
        and routes exceptions through job.on_failure. Never raises.
        """
        if self.background_scheduler is not None:
            try:
                self.background_scheduler.submit(job)
                return
            except Exception:
                import logging
                logging.getLogger(__name__).exception(
                    "scheduler.submit failed; falling back to sync for kind=%s",
                    job.kind,
                )
        try:
            job.fn()
        except Exception as exc:
            if job.on_failure is not None:
                try:
                    job.on_failure(exc)
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception(
                        "on_failure raised during sync fallback"
                    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _wrap_tool_registry(
    tool_registry: ToolRegistry | None,
    *,
    tracer: Tracer,
    parent_context: TraceContext,
) -> _TracingToolRegistryProxy | None:
    if tool_registry is None:
        return None
    return _TracingToolRegistryProxy(
        tool_registry=tool_registry,
        tracer=tracer,
        parent_context=parent_context,
    )


class _TracingProviderProxy:
    def __init__(self, *, provider: ChatProvider, tracer: Tracer, parent_context: TraceContext) -> None:
        self._provider = provider
        self._tracer = tracer
        self._parent_context = parent_context
        self.capabilities = provider.capabilities if hasattr(provider, "capabilities") else None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    async def achat(
        self,
        messages: list[Any],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ):
        request_data = {
            "messages": _serialize_messages(messages),
            "model": model,
            "tools": [dict(tool) for tool in tools or []],
        }
        span = safe_start_span(
            self._tracer,
            self._parent_context,
            name="provider.chat",
            metadata={"model": model, "message_count": len(messages), "tool_count": len(tools or [])},
            inputs=request_data,
            run_type="llm",
        )
        try:
            response = await self._provider.achat(messages, model=model, tools=tools)
        except Exception as exc:
            safe_finish_span(self._tracer, span, status="error", outputs={"error": str(exc)})
            raise
        safe_finish_span(self._tracer, span, status="ok", outputs=_serialize_response(response))
        return response

    async def astream_text(
        self,
        messages: list[Any],
        *,
        model: str | None = None,
    ):
        request_data = {
            "messages": _serialize_messages(messages),
            "model": model,
        }
        span = safe_start_span(
            self._tracer,
            self._parent_context,
            name="provider.stream",
            metadata={"model": model, "message_count": len(messages)},
            inputs=request_data,
            run_type="llm",
        )
        collected_text = ""
        try:
            async for chunk in self._provider.astream_text(messages, model=model):
                collected_text += str(chunk)
                yield str(chunk)
        except Exception as exc:
            safe_finish_span(self._tracer, span, status="error", outputs={"error": str(exc)})
            raise
        safe_finish_span(self._tracer, span, status="ok", outputs={"content": collected_text})


class _TracingToolRegistryProxy:
    def __init__(self, *, tool_registry: ToolRegistry, tracer: Tracer, parent_context: TraceContext) -> None:
        self._tool_registry = tool_registry
        self._tracer = tracer
        self._parent_context = parent_context
        self.skill_loader = tool_registry.skill_loader
        self.mcp_registry = tool_registry.mcp_registry

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool_registry, name)

    def execute(self, call: ToolCall, active_capabilities: ActiveCapabilities | None = None) -> ToolResult:
        tool_name = call.name.strip() or "tool"
        registered = self._tool_registry.get(tool_name)
        tool_source = registered.spec.source if registered is not None else None
        span = safe_start_span(
            self._tracer,
            self._parent_context,
            name=f"tool.call.{tool_name}",
            metadata={"tool_name": tool_name, "tool_source": tool_source},
            inputs={"tool_name": tool_name, "arguments": dict(call.arguments)},
        )
        try:
            result = self._tool_registry.execute(call, active_capabilities)
        except Exception as exc:
            safe_finish_span(self._tracer, span, status="error", outputs={"error": str(exc)})
            raise
        safe_finish_span(
            self._tracer,
            span,
            status="error" if result.is_error else "ok",
            outputs={"content": result.content, "is_error": result.is_error, "metadata": dict(result.metadata)},
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
                "role": str(message.role),
                "content": message.content,
                "content_parts": list(message.content_parts or []),
                "tool_calls": list(message.tool_calls or []),
                "name": message.name,
                "tool_call_id": message.tool_call_id,
            }
        )
    return serialized


def _serialize_response(response: ChatResponse) -> dict[str, Any]:
    return {
        "provider": response.provider,
        "model": response.model,
        "content": response.content,
        "content_parts": [dict(part) for part in response.content_parts],
        "tool_calls": [dict(call) for call in response.tool_calls],
        "usage": response.usage.model_dump(exclude_none=True) if response.usage is not None else {},
        "raw": dict(response.raw),
    }


def _normalize_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def _build_langchain_callbacks(tracer: Tracer) -> list[Any]:
    """Build LangChain callbacks for LangGraph node-level tracing.

    When a LangSmithTracer is active, injects a LangChainTracer that records
    each graph node's input/output state and duration as nested spans.
    """
    from miniclaw.observability.langsmith import LangSmithTracer
    from miniclaw.observability.composite import CompositeTracer

    tracers_to_check: list[object] = []
    if isinstance(tracer, CompositeTracer):
        tracers_to_check.extend(tracer._tracers)
    else:
        tracers_to_check.append(tracer)

    for t in tracers_to_check:
        if isinstance(t, LangSmithTracer):
            try:
                from langchain_core.tracers import LangChainTracer

                return [LangChainTracer(
                    client=t.client,
                    project_name=t.project,
                )]
            except ImportError:
                return []
    return []


