from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from miniclaw.config.settings import Settings, _default_sqlite_path
from miniclaw.cron.service import CronService
from miniclaw.heartbeat.service import HeartbeatService
from miniclaw.memory.embedding import OllamaEmbedder
from miniclaw.memory.files import MemoryFileStore
from miniclaw.memory.indexer import MemoryIndexer
from miniclaw.memory.retriever import HybridRetriever
from miniclaw.mcp.registry import MCPRegistry
from miniclaw.observability.factory import build_tracer
from miniclaw.persistence.factory import build_memory_store
from miniclaw.persistence.memory_store import MemoryStore
from miniclaw.prompting import ContextBuilder
from miniclaw.providers.openai_compat import OpenAICompatibleProvider
from miniclaw.runtime.checkpoint import AsyncSQLiteCheckpointer
from miniclaw.runtime.service import ResumeResult, RuntimeService
from miniclaw.runtime.state import ActiveCapabilities
from miniclaw.runtime.thread_control import SQLiteThreadControlStore
from miniclaw.skills.loader import SkillLoader
from miniclaw.tools.builtin.activation import (
    build_add_mcp_server_tool,
    build_load_mcp_tools_tool,
    build_load_skill_tools_tool,
    build_reload_mcp_servers_tool,
)
from miniclaw.tools.builtin.cron import build_cron_tool
from miniclaw.tools.builtin.filesystem import build_read_file_tool, build_write_file_tool
from miniclaw.tools.builtin.memory_search import build_search_memory_tool
from miniclaw.tools.builtin.send import build_send_tool
from miniclaw.tools.builtin.skills import build_list_skills_tool, build_load_skill_tool
from miniclaw.tools.builtin.shell import build_shell_tool
from miniclaw.tools.builtin.web import build_web_search_tool
from miniclaw.tools.builtin.spawn_subagent import build_spawn_subagent_tool
from miniclaw.tools.builtin.heartbeat_manage import build_manage_heartbeat_tool
from miniclaw.tools.messaging import MessagingBridge
from miniclaw.tools.search import SearchBackend, build_search_backend
from miniclaw.tools.registry import RegisteredTool
from miniclaw.tools.registry import ToolRegistry
from miniclaw.observability.contracts import TraceContext

if TYPE_CHECKING:
    from miniclaw.cron.types import CronJob
    from miniclaw.observability.contracts import Tracer
    from miniclaw.runtime.background import BackgroundScheduler




def _resolve_cache_strategy(
    *,
    base_url: str,
    explicit: str,
    enable_legacy_prompt_cache: bool,
) -> str:
    """Resolve the effective cache_strategy from settings + base_url.

    Order: explicit (non-"auto") wins; otherwise auto-detect from base_url;
    fall back to "anthropic" if legacy flag set; else "none".
    """
    if explicit != "auto":
        return explicit
    url = (base_url or "").lower()
    if "anthropic.com" in url or "dashscope" in url:
        return "anthropic"
    if "openai.com" in url:
        return "openai_auto"
    if enable_legacy_prompt_cache:
        return "anthropic"
    return "none"


def build_settings() -> Settings:
    return Settings()


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_skill_loader(
    *,
    workspace: Path | None = None,
    builtin_skills_dir: Path | None = None,
    settings: Settings | None = None,
) -> SkillLoader:
    resolved_settings = settings
    resolved_workspace = workspace or (resolved_settings.runtime_dir if resolved_settings is not None else _workspace_root())
    resolved_builtin_skills_dir = builtin_skills_dir or (_workspace_root() / "miniclaw" / "skills" / "builtin")
    return SkillLoader(
        workspace=resolved_workspace,
        builtin_skills_dir=resolved_builtin_skills_dir,
        workspace_skills_dir=resolved_settings.user_skills_dir if resolved_settings is not None else None,
    )


def build_tool_registry(
    *,
    workspace: Path | None = None,
    skill_loader: SkillLoader | None = None,
    mcp_registry: MCPRegistry | None = None,
    cron_service: CronService | None = None,
    search_backend: SearchBackend | None = None,
    messaging_bridge: MessagingBridge | None = None,
    settings: Settings | None = None,
    provider: OpenAICompatibleProvider | None = None,
    tracer: TraceContext | None = None,
    retriever: HybridRetriever | None = None,
    heartbeat_file: Path | None = None,
) -> ToolRegistry:
    resolved_settings = settings
    resolved_workspace = workspace or _workspace_root()
    resolved_loader = skill_loader or build_skill_loader(workspace=resolved_workspace, settings=resolved_settings)
    resolved_mcp_registry = mcp_registry or build_mcp_registry(resolved_settings)
    resolved_cron_service = cron_service
    resolved_search_backend = search_backend if search_backend is not None else build_search_backend(resolved_settings)
    registry = ToolRegistry(
        skill_loader=resolved_loader,
        mcp_registry=resolved_mcp_registry,
    )
    spawn_subagent_tool = build_spawn_subagent_tool(
        provider=provider,
        settings=resolved_settings,
        tool_registry=registry,
        tracer=tracer,
    )
    resolved_max_read_file_bytes = resolved_settings.max_read_file_bytes if resolved_settings else 32 * 1024
    builtin_tools = [
        build_read_file_tool(workspace=resolved_workspace, max_bytes=resolved_max_read_file_bytes),
        build_write_file_tool(),
        build_cron_tool(cron_service=resolved_cron_service),
        build_list_skills_tool(resolved_loader),
        build_load_skill_tool(resolved_loader),
        build_load_skill_tools_tool(resolved_loader),
        build_load_mcp_tools_tool(resolved_mcp_registry),
        build_add_mcp_server_tool(resolved_mcp_registry, resolved_settings.mcp_config_path if resolved_settings else _workspace_root() / ".miniclaw" / "mcp.json"),
        build_reload_mcp_servers_tool(resolved_mcp_registry, resolved_settings.mcp_config_path if resolved_settings else _workspace_root() / ".miniclaw" / "mcp.json"),
        build_shell_tool(workspace=resolved_workspace),
        build_web_search_tool(search_backend=resolved_search_backend),
        build_send_tool(messaging_bridge=messaging_bridge),
    ]
    builtin_tools.append(spawn_subagent_tool)
    if heartbeat_file is not None:
        builtin_tools.append(build_manage_heartbeat_tool(heartbeat_file=heartbeat_file))
    if retriever is not None:
        builtin_tools.append(
            build_search_memory_tool(
                retriever=retriever,
                default_top_k=resolved_settings.memory_search_top_k if resolved_settings else 10,
            )
        )
    _CORE_TOOLS = {
        "shell", "read_file", "write_file", "send", "web_search",
        "cron", "manage_heartbeat",
        "load_skill_tools", "load_mcp_tools",
        "spawn_subagent",
    }
    for tool in builtin_tools:
        if tool.spec.name in _CORE_TOOLS:
            registry.register(_mark_always_active(tool))
        else:
            registry.register(_mark_discoverable(tool))
    return registry


def build_mcp_registry(settings: Settings | None = None) -> MCPRegistry:
    from miniclaw.mcp.config import register_servers_from_config

    registry = MCPRegistry()
    resolved_settings = settings
    if resolved_settings is not None:
        register_servers_from_config(registry, resolved_settings.mcp_config_path)
    return registry


def build_cron_service(
    settings: Settings | None = None,
    *,
    on_notify: Any | None = None,
    runtime_factory: Callable[[], RuntimeService] | None = None,
) -> CronService:
    resolved_settings = settings or build_settings()
    return CronService(
        resolved_settings.cron_store_path,
        on_job=_build_scheduled_runtime_callback(resolved_settings, runtime_factory=runtime_factory),
        on_notify=on_notify,
    )


def build_heartbeat_service(
    settings: Settings | None = None,
    *,
    provider: OpenAICompatibleProvider | None = None,
    on_execute: Any | None = None,
    on_notify: Any | None = None,
    runtime_factory: Callable[[], RuntimeService] | None = None,
    mini_provider: OpenAICompatibleProvider | None = None,
) -> HeartbeatService:
    resolved_settings = settings or build_settings()
    resolved_provider = provider or build_provider(resolved_settings)
    resolved_mini_provider = mini_provider or build_mini_provider(resolved_settings)

    return HeartbeatService(
        workspace=resolved_settings.runtime_dir,
        provider=resolved_provider,
        model=resolved_settings.model,
        heartbeat_model=resolved_settings.mini_model,
        heartbeat_provider=resolved_mini_provider,
        on_execute=on_execute or _build_heartbeat_execute_callback(resolved_settings, runtime_factory=runtime_factory),
        on_notify=_build_heartbeat_notify_callback(resolved_settings, on_notify),
        interval_s=resolved_settings.heartbeat_interval_s,
    )


def _mark_always_active(tool: RegisteredTool) -> RegisteredTool:
    metadata = dict(tool.spec.metadata)
    metadata.update({"always_active": True, "discoverable": True})
    return RegisteredTool(spec=tool.spec.model_copy(update={"metadata": metadata}), executor=tool.executor)


def _mark_discoverable(tool: RegisteredTool) -> RegisteredTool:
    metadata = dict(tool.spec.metadata)
    metadata.update({"discoverable": True, "always_active": False})
    return RegisteredTool(spec=tool.spec.model_copy(update={"metadata": metadata}), executor=tool.executor)


def build_thread_control_store(sqlite_path: Path | None = None) -> SQLiteThreadControlStore:
    resolved_path = sqlite_path or _resolve_sqlite_path_from_env()
    return SQLiteThreadControlStore(resolved_path)


def build_provider(settings: Settings | None = None) -> OpenAICompatibleProvider:
    resolved_settings = settings or build_settings()
    cache_strategy = _resolve_cache_strategy(
        base_url=resolved_settings.base_url,
        explicit=resolved_settings.cache_strategy,
        enable_legacy_prompt_cache=resolved_settings.enable_prompt_cache,
    )
    if resolved_settings.enable_prompt_cache and resolved_settings.cache_strategy == "auto":
        import warnings
        warnings.warn(
            "MINICLAW_ENABLE_PROMPT_CACHE is deprecated; "
            "set MINICLAW_CACHE_STRATEGY=anthropic explicitly. "
            "This setting will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
    return OpenAICompatibleProvider(
        api_key=resolved_settings.api_key,
        base_url=resolved_settings.base_url,
        model=resolved_settings.model,
        cache_strategy=cache_strategy,
    )


def build_mini_provider(settings: Settings) -> OpenAICompatibleProvider | None:
    if not settings.mini_model or not settings.mini_model_base_url:
        return None
    return OpenAICompatibleProvider(
        api_key=settings.mini_model_api_key or settings.api_key,
        base_url=settings.mini_model_base_url,
        model=settings.mini_model,
    )


def build_context_builder(settings: Settings | None = None) -> ContextBuilder:
    resolved_settings = settings or build_settings()
    tool_registry = build_tool_registry(workspace=resolved_settings.runtime_dir, settings=resolved_settings)
    return ContextBuilder(
        system_prompt=resolved_settings.system_prompt,
        tool_registry=tool_registry,
        mcp_registry=tool_registry.mcp_registry,
        skills_loader=tool_registry.skill_loader,
    )


def build_memory_file_store(settings: Settings | None = None) -> MemoryFileStore:
    resolved_settings = settings or build_settings()
    return MemoryFileStore(resolved_settings.sqlite_path.parent / "MEMORY.md")


def build_embedder(settings: Settings) -> OllamaEmbedder:
    return OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        dims=settings.embedding_dims,
    )


def build_memory_indexer(settings: Settings, embedder: OllamaEmbedder) -> MemoryIndexer:
    memory_dir = settings.runtime_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    return MemoryIndexer(
        db_path=settings.sqlite_path,
        embedder=embedder,
        memory_dir=memory_dir,
    )


def build_retriever(settings: Settings, embedder: OllamaEmbedder) -> HybridRetriever:
    return HybridRetriever(
        db_path=settings.sqlite_path,
        embedder=embedder,
    )


def build_runtime_service(
    settings: Settings | None = None,
    *,
    provider: OpenAICompatibleProvider | None = None,
    mini_provider: OpenAICompatibleProvider | None = None,
    memory_store: MemoryStore | None = None,
    memory_file_store: MemoryFileStore | None = None,
    tool_registry: ToolRegistry | None = None,
    mcp_registry: MCPRegistry | None = None,
    cron_service: CronService | None = None,
    search_backend: SearchBackend | None = None,
    messaging_bridge: MessagingBridge | None = None,
    thread_control_store: SQLiteThreadControlStore | None = None,
    tracer: TraceContext | None = None,
    langsmith_client: Any | None = None,
    memory_indexer: MemoryIndexer | None = None,
    retriever: HybridRetriever | None = None,
    background_scheduler: BackgroundScheduler | None = None,
) -> RuntimeService:
    resolved_settings = settings or build_settings()
    resolved_memory_store = memory_store or build_memory_store(resolved_settings.sqlite_path)
    resolved_provider = provider or build_provider(resolved_settings)
    resolved_mini_provider = mini_provider or build_mini_provider(resolved_settings)
    resolved_memory_file_store = memory_file_store or build_memory_file_store(resolved_settings)
    resolved_mcp_registry = mcp_registry or build_mcp_registry(resolved_settings)
    resolved_tracer = tracer or build_tracer(
        resolved_settings,
        langsmith_client=langsmith_client,
    )

    # Auto-build retriever and indexer if not provided
    resolved_retriever = retriever
    resolved_indexer = memory_indexer
    if resolved_retriever is None or resolved_indexer is None:
        try:
            embedder = build_embedder(resolved_settings)
            if resolved_retriever is None:
                resolved_retriever = build_retriever(resolved_settings, embedder)
            if resolved_indexer is None:
                resolved_indexer = build_memory_indexer(resolved_settings, embedder)
        except Exception:
            pass  # Ollama not available — degrade gracefully

    resolved_tool_registry = tool_registry or build_tool_registry(
        workspace=_workspace_root(),
        mcp_registry=resolved_mcp_registry,
        cron_service=cron_service,
        search_backend=search_backend,
        messaging_bridge=messaging_bridge,
        settings=resolved_settings,
        provider=resolved_provider,
        tracer=resolved_tracer,
        heartbeat_file=resolved_settings.heartbeat_file,
        retriever=resolved_retriever,
    )
    resolved_thread_control_store = thread_control_store or build_thread_control_store(
        resolved_settings.sqlite_path
    )
    return RuntimeService(
        settings=resolved_settings,
        provider=resolved_provider,
        mini_provider=resolved_mini_provider,
        memory_store=resolved_memory_store,
        memory_file_store=resolved_memory_file_store,
        tool_registry=resolved_tool_registry,
        thread_control_store=resolved_thread_control_store,
        memory_indexer=resolved_indexer,
        retriever=resolved_retriever,
        tracer=resolved_tracer,
        background_scheduler=background_scheduler,
    )


def initialize_local_storage(sqlite_path: Path | None = None) -> Path:
    import json as _json

    resolved_path = sqlite_path or build_settings().sqlite_path
    runtime_dir = resolved_path.parent
    (runtime_dir / "skills").mkdir(parents=True, exist_ok=True)
    (runtime_dir / "cron").mkdir(parents=True, exist_ok=True)
    (runtime_dir / "memory").mkdir(parents=True, exist_ok=True)
    heartbeat_file = runtime_dir / "HEARTBEAT.md"
    if not heartbeat_file.exists():
        heartbeat_file.write_text(
            "# Heartbeat\n\nAdd periodic tasks below that MiniClaw should review on each heartbeat tick.\n",
            encoding="utf-8",
        )
    mcp_config_file = runtime_dir / "mcp.json"
    if not mcp_config_file.exists():
        mcp_config_file.write_text(
            _json.dumps({"servers": []}, indent=2) + "\n",
            encoding="utf-8",
        )
    build_memory_store(resolved_path)
    AsyncSQLiteCheckpointer(resolved_path)
    return resolved_path


def load_latest_checkpoint(
    thread_id: str,
    sqlite_path: Path | None = None,
) -> ResumeResult | None:
    resolved_path = sqlite_path or _resolve_sqlite_path_from_env()
    checkpointer = AsyncSQLiteCheckpointer(resolved_path)
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


def _resolve_sqlite_path_from_env() -> Path:
    raw_value = os.environ.get("MINICLAW_SQLITE_PATH", "").strip()
    if raw_value:
        return Path(raw_value)
    return _default_sqlite_path()


def _make_runtime_builder(
    settings: Settings,
    runtime_factory: Callable[[], RuntimeService] | None = None,
) -> Callable[[], RuntimeService]:
    if runtime_factory is not None:
        return runtime_factory
    return lambda: build_runtime_service(settings)


def _build_scheduled_runtime_callback(
    settings: Settings,
    *,
    runtime_factory: Callable[[], RuntimeService] | None = None,
):
    build_runtime = _make_runtime_builder(settings, runtime_factory)

    async def _run(job: CronJob) -> str | None:
        runtime = build_runtime()
        payload = job.payload
        raw_message = (payload.message or "").strip()
        job_name = job.name.strip()

        # Build a task directive instead of passing raw prompt
        user_input = (
            f"[Scheduled task: {job_name}]\n"
            f"Execute the following task now. Do NOT re-create or re-schedule this task — it is already scheduled.\n\n"
            f"Task: {raw_message}"
        )

        channel = (payload.channel or "").strip()
        chat_id = (payload.chat_id or "").strip()
        message_thread_id = (payload.message_thread_id or "").strip()
        runtime_metadata: dict[str, Any] = {
            "thread_id": f"cron:{job.id}",
            "channel": channel or "cron",
        }
        if chat_id:
            runtime_metadata["chat_id"] = chat_id
        if message_thread_id:
            runtime_metadata["message_thread_id"] = message_thread_id

        result = runtime.run_turn(
            thread_id=f"cron:{job.id}",
            user_input=user_input,
            runtime_metadata=runtime_metadata,
        )
        return result.response_text or result.last_error or None

    return _run


def _build_heartbeat_execute_callback(
    settings: Settings,
    *,
    runtime_factory: Callable[[], RuntimeService] | None = None,
):
    build_runtime = _make_runtime_builder(settings, runtime_factory)

    async def _run(tasks: str) -> str | None:
        runtime = build_runtime()
        user_input = (
            "[Heartbeat task]\n"
            "Execute the following tasks now. Do NOT re-create or re-schedule these tasks.\n\n"
            f"Tasks:\n{tasks}"
        )
        result = runtime.run_turn(
            thread_id="heartbeat",
            user_input=user_input,
            runtime_metadata={"thread_id": "heartbeat", "channel": "heartbeat"},
        )
        return result.response_text or result.last_error or None

    return _run


def _build_heartbeat_notify_callback(settings: Settings, on_notify: Any | None = None):
    if on_notify is not None:
        return on_notify
    if not settings.heartbeat_chat_id:
        return None

    async def _notify(message: str) -> None:
        return None

    return _notify
