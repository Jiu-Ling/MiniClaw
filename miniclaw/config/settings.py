from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic.networks import AnyHttpUrl


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_sqlite_path() -> Path:
    return _repo_root() / ".miniclaw" / "miniclaw.sqlite3"


def _default_trace_dir() -> Path:
    return _default_sqlite_path().parent / "traces"


class Settings(BaseSettings):
    """Runtime settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="MINICLAW_",
        case_sensitive=False,
        extra="ignore",
        env_file=_repo_root() / ".env",
        env_file_encoding="utf-8",
    )

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        trace_mode: str | None = None,
        trace_dir: Path | str | None = None,
        trace_full_content: bool | None = None,
        trace_max_chars: int | None = None,
        **values: Any,
    ) -> None:
        data: dict[str, Any] = dict(values)
        if api_key is not None:
            data["api_key"] = api_key
        if base_url is not None:
            data["base_url"] = base_url
        if model is not None:
            data["model"] = model
        if trace_mode is not None:
            data["trace_mode"] = trace_mode
        if trace_dir is not None:
            data["trace_dir"] = trace_dir
        if trace_full_content is not None:
            data["trace_full_content"] = trace_full_content
        if trace_max_chars is not None:
            data["trace_max_chars"] = trace_max_chars
        super().__init__(**data)

    api_key: str = Field(...)
    base_url: str = Field(...)
    model: str = Field(...)
    sqlite_path: Path = Field(default_factory=_default_sqlite_path)
    default_thread_id: str = "default"
    telegram_bot_token: str | None = None
    heartbeat_chat_id: str | None = None
    heartbeat_message_thread_id: str | None = None
    log_level: str = "INFO"
    debug: bool = True
    system_prompt: str = ""
    trace_mode: Literal["off", "local", "langsmith", "both"] = "both"
    trace_dir: Path = Field(default_factory=_default_trace_dir)
    trace_full_content: bool = True
    trace_max_chars: int = 4000
    search_provider: Literal["none", "brave", "tavily", "searxng", "jina"] = "none"
    search_api_key: str | None = None
    search_base_url: str | None = None
    search_proxy: str | None = None
    search_max_results: int = 5

    # Runtime limits
    max_tool_rounds: int = 16
    max_consecutive_tool_errors: int = 4
    max_tool_result_chars: int = 16_000
    subagent_max_tool_result_chars: int = 2_000
    max_read_file_bytes: int = 32 * 1024
    history_char_budget: int = 12_000

    # Context compression
    compress_keep_recent_turns: int = 2
    compress_turn_summary_chars: int = 200
    max_history_messages: int = 4

    # Embedding
    embedding_model: str = "bge-m3"
    embedding_dims: int = 1024
    ollama_base_url: str = "http://localhost:11434"

    # Memory retrieval
    memory_search_top_k: int = 5
    memory_token_budget: int = 2000

    # Heartbeat
    heartbeat_interval_s: int = 900

    # Background scheduler (fire-and-forget jobs; see runtime/background.py)
    background_max_queue: int = 32
    background_stop_timeout_s: float = 5.0

    # Memory rewrite (Phase 3)
    memory_rewrite_enabled: bool = True
    memory_rewrite_model_tier: Literal["mini", "main", "auto"] = "auto"
    memory_rewrite_timeout_s: float = 10.0
    memory_rewrite_recent_exchanges: int = 2

    # Memory consolidation (Phase 4)
    memory_consolidation_enabled: bool = True
    memory_consolidation_model_tier: Literal["mini", "main", "auto"] = "auto"
    memory_consolidation_timeout_s: float = 30.0
    memory_consolidation_trigger_threshold: int = 3
    memory_critical_facts_max: int = 12

    # Prompt caching strategy.
    # "auto" → bootstrap detects from base_url (anthropic/dashscope → "anthropic",
    #          openai → "openai_auto", unknown → "none").
    # Explicit values bypass detection.
    cache_strategy: Literal["auto", "anthropic", "openai_auto", "none"] = "auto"

    # DEPRECATED: use cache_strategy instead. Removed in a future release.
    # When True (and cache_strategy is "auto"), bootstrap maps to "anthropic".
    enable_prompt_cache: bool = False

    # Mini model — shared by classify intent + heartbeat judgment
    mini_model: str | None = None
    mini_model_base_url: str | None = None
    mini_model_api_key: str | None = None

    @property
    def runtime_dir(self) -> Path:
        return self.sqlite_path.parent

    @property
    def user_skills_dir(self) -> Path:
        return self.runtime_dir / "skills"

    @property
    def heartbeat_file(self) -> Path:
        return self.runtime_dir / "HEARTBEAT.md"

    @property
    def cron_store_path(self) -> Path:
        return self.runtime_dir / "cron" / "jobs.json"

    @property
    def mcp_config_path(self) -> Path:
        return self.runtime_dir / "mcp.json"

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return str(AnyHttpUrl(value))

    @field_validator("default_thread_id")
    @classmethod
    def validate_default_thread_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("default_thread_id must not be empty")
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        allowed_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if normalized not in allowed_levels:
            raise ValueError(f"log_level must be one of: {', '.join(sorted(allowed_levels))}")
        return normalized

    @field_validator("trace_mode")
    @classmethod
    def validate_trace_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed_modes = {"off", "local", "langsmith", "both"}
        if normalized not in allowed_modes:
            raise ValueError(f"trace_mode must be one of: {', '.join(sorted(allowed_modes))}")
        return normalized

    @field_validator("trace_max_chars")
    @classmethod
    def validate_trace_max_chars(cls, value: int) -> int:
        if value < 0:
            raise ValueError("trace_max_chars must be greater than or equal to 0")
        return value

    @field_validator("search_provider")
    @classmethod
    def validate_search_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"none", "brave", "tavily", "searxng", "jina"}
        if normalized not in allowed:
            raise ValueError(f"search_provider must be one of: {', '.join(sorted(allowed))}")
        return normalized

    @field_validator("search_max_results")
    @classmethod
    def validate_search_max_results(cls, value: int) -> int:
        if value < 1:
            raise ValueError("search_max_results must be greater than or equal to 1")
        return value
