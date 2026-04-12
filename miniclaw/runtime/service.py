from __future__ import annotations

from collections.abc import Iterator, Mapping
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.base import Checkpoint
from langgraph.graph import END, START, StateGraph

from miniclaw.memory.files import MemoryFileStore
from miniclaw.observability.contracts import NoopTracer, TraceContext
from miniclaw.persistence.memory_store import MemoryItem, MemoryStore
from miniclaw.prompting.context import prompt_trace_scope
from miniclaw.runtime.checkpoint import AsyncSQLiteCheckpointer
from miniclaw.runtime.graph import build_graph
from miniclaw.runtime.state import ActiveCapabilities, RuntimeState, RuntimeUsage

if TYPE_CHECKING:
    from miniclaw.config.settings import Settings
    from miniclaw.memory.indexer import MemoryIndexer
    from miniclaw.memory.retriever import HybridRetriever
    from miniclaw.providers.contracts import ChatProvider, ChatResponse
    from miniclaw.tools.contracts import ToolCall, ToolResult
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
        mini_provider: OpenAICompatibleProvider | None = None,
        tracer: TraceContext | None = None,
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

    def with_messaging_bridge(self, bridge: object) -> "RuntimeService":
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
        run_context = _safe_start_run(
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
        fleet_runs = result.get("fleet_runs", [])
        final_output = {
            "response_text": turn_result.response_text,
            "last_error": turn_result.last_error,
            "usage": turn_result.usage,
            "checkpoint_id": turn_result.checkpoint_id,
            "fleet_run_count": len(fleet_runs) if isinstance(fleet_runs, list) else 0,
        }
        _safe_record_event(
            self.tracer,
            run_context,
            name="fleet.status",
            payload={
                "fleet_run_count": len(fleet_runs) if isinstance(fleet_runs, list) else 0,
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
        from queue import Queue
        from threading import Thread

        run_context = _safe_start_run(
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
            _safe_finish_run(self.tracer, run_context, status="error", output=final_output)
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
        _safe_finish_run(self.tracer, run_context, status=final_status, output={
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
        append_fact = self.memory_store.append_fact

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
        # Decay old memories (fading: older → shorter → gone after 30 days)
        if self.memory_file_store is not None:
            from miniclaw.memory.decay import decay_memory
            self._safe_memory_write(
                lambda: decay_memory(self.memory_file_store, self.clock().date())
            )
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
        runtime_dir = settings.runtime_dir
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
            self.memory_indexer.mark_dirty(f"{today}.md")

    def _list_recent_durable_facts(self, thread_id: str, *, limit: int) -> list[MemoryItem]:
        items = self._list_recent_items(thread_id, limit=limit)
        return [item for item in items if item.kind in DURABLE_MEMORY_KINDS]

    def _list_recent_by_kind(self, thread_id: str, kind: str, *, limit: int) -> list[MemoryItem]:
        return self.memory_store.list_recent_by_kind(thread_id, kind, limit)

    def _list_recent_items(self, thread_id: str, *, limit: int) -> list[MemoryItem]:
        return self.memory_store.list_recent(thread_id, limit)

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

    _DIGEST_OUTCOME_MAX_CHARS = 120

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
    inputs: Mapping[str, Any] | None = None,
    run_type: str | None = None,
) -> TraceContext | None:
    if parent is None:
        return None
    try:
        return tracer.start_span(parent, name=name, metadata=metadata, inputs=inputs, run_type=run_type)
    except Exception:
        return NoopTracer().start_span(parent, name=name, metadata=metadata)


def _safe_finish_span(
    tracer: object,
    context: TraceContext | None,
    *,
    status: str,
    output: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
) -> None:
    if context is None:
        return
    try:
        tracer.finish_span(context, status=status, output=output, outputs=outputs)
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
    def __init__(self, *, provider: ChatProvider, tracer: object, parent_context: TraceContext) -> None:
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
        span = _safe_start_span(
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
            _safe_finish_span(self._tracer, span, status="error", output={"error": str(exc)})
            raise
        _safe_finish_span(self._tracer, span, status="ok", outputs=_serialize_response(response))
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
        span = _safe_start_span(
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
            _safe_finish_span(self._tracer, span, status="error", output={"error": str(exc)})
            raise
        _safe_finish_span(self._tracer, span, status="ok", outputs={"content": collected_text})


class _TracingToolRegistryProxy:
    def __init__(self, *, tool_registry: ToolRegistry, tracer: object, parent_context: TraceContext) -> None:
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
        span = _safe_start_span(
            self._tracer,
            self._parent_context,
            name=f"tool.call.{tool_name}",
            metadata={"tool_name": tool_name, "tool_source": tool_source},
            inputs={"tool_name": tool_name, "arguments": dict(call.arguments)},
        )
        try:
            result = self._tool_registry.execute(call, active_capabilities)
        except Exception as exc:
            _safe_finish_span(self._tracer, span, status="error", output={"error": str(exc)})
            raise
        _safe_finish_span(
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


def _build_langchain_callbacks(tracer: object) -> list[Any]:
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


