from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True)
class CronSchedule:
    kind: Literal["at", "every", "cron"]
    at_ms: int | None = None
    every_ms: int | None = None
    expr: str | None = None
    tz: str | None = None


@dataclass(slots=True)
class CronPayload:
    kind: Literal["agent_turn"] = "agent_turn"
    message: str = ""
    deliver: bool = True
    channel: str | None = None
    chat_id: str | None = None
    message_thread_id: str | None = None


@dataclass(slots=True)
class CronRunRecord:
    run_at_ms: int
    status: Literal["ok", "error", "skipped"]
    duration_ms: int = 0
    error: str | None = None


@dataclass(slots=True)
class CronJobState:
    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: Literal["ok", "error", "skipped"] | None = None
    last_error: str | None = None
    run_history: list[CronRunRecord] = field(default_factory=list)


@dataclass(slots=True)
class CronJob:
    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False


@dataclass(slots=True)
class CronStore:
    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)
