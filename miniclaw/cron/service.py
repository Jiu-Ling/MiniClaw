from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from miniclaw.cron.types import CronJob, CronJobState, CronPayload, CronSchedule, CronStore


def _now_ms() -> int:
    return int(time.time() * 1000)


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None
    if schedule.kind == "every":
        if not schedule.every_ms or schedule.every_ms <= 0:
            return None
        return now_ms + schedule.every_ms
    if schedule.kind == "cron" and schedule.expr:
        try:
            from zoneinfo import ZoneInfo

            from croniter import croniter

            base_time = now_ms / 1000
            tz = ZoneInfo(schedule.tz) if schedule.tz else datetime.now().astimezone().tzinfo
            base_dt = datetime.fromtimestamp(base_time, tz=tz)
            cron = croniter(schedule.expr, base_dt)
            next_dt = cron.get_next(datetime)
            return int(next_dt.timestamp() * 1000)
        except Exception:
            return None
    return None


def _parse_at_to_ms(raw_at: str) -> int:
    text = raw_at.strip()
    if not text:
        raise ValueError("at must not be empty")
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("at must be an ISO datetime string") from exc
    return int(dt.timestamp() * 1000)


class CronService:
    _MAX_RUN_HISTORY = 20

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
        on_notify: Callable[[CronJob, str], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self.store_path = Path(store_path)
        self.on_job = on_job
        self.on_notify = on_notify
        self._store: CronStore | None = None
        self._running = False
        self._timer_task: Any | None = None
        self._loop: Any | None = None
        self._pending_arm = False

    def add_job(
        self,
        *,
        name: str,
        message: str,
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        tz: str | None = None,
        at: str | None = None,
        deliver: bool = True,
        channel: str | None = None,
        chat_id: str | None = None,
        message_thread_id: str | None = None,
    ) -> CronJob:
        job_name = name.strip() or "scheduled-task"
        payload_message = message.strip()
        if not payload_message:
            raise ValueError("message is required")
        if deliver and (channel or "").strip().lower() == "telegram" and not (chat_id or "").strip():
            raise ValueError("chat_id is required for telegram cron delivery")
        schedule = self._build_schedule(every_seconds=every_seconds, cron_expr=cron_expr, tz=tz, at=at)
        now_ms = _now_ms()
        job = CronJob(
            id=uuid.uuid4().hex[:12],
            name=job_name,
            schedule=schedule,
            payload=CronPayload(
                message=payload_message,
                deliver=deliver,
                channel=channel,
                chat_id=(chat_id.strip() if isinstance(chat_id, str) and chat_id.strip() else None),
                message_thread_id=(
                    message_thread_id.strip()
                    if isinstance(message_thread_id, str) and message_thread_id.strip()
                    else None
                ),
            ),
            state=CronJobState(next_run_at_ms=_compute_next_run(schedule, now_ms)),
            created_at_ms=now_ms,
            updated_at_ms=now_ms,
            delete_after_run=schedule.kind == "at",
        )
        store = self._load_store()
        store.jobs.append(job)
        self._save_store()
        if self._running:
            self._arm_timer()
        return job

    def list_jobs(self) -> list[CronJob]:
        return list(self._load_store().jobs)

    def remove_job(self, job_id: str) -> bool:
        store = self._load_store()
        remaining = [job for job in store.jobs if job.id != job_id]
        if len(remaining) == len(store.jobs):
            return False
        store.jobs = remaining
        self._save_store()
        if self._running:
            self._arm_timer()
        return True

    def status(self) -> dict[str, Any]:
        jobs = self.list_jobs()
        return {"jobs": len(jobs), "enabled": sum(1 for job in jobs if job.enabled)}

    async def start(self) -> None:
        import asyncio

        self._running = True
        self._loop = asyncio.get_running_loop()
        self._load_store()
        self._arm_timer()

    def stop(self) -> None:
        self._running = False
        self._loop = None
        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None

    async def trigger_due_jobs(self, *, now_ms: int | None = None) -> list[str | None]:
        store = self._load_store()
        current_ms = now_ms or _now_ms()
        results: list[str | None] = []
        for job in list(store.jobs):
            if not job.enabled or not job.state.next_run_at_ms or current_ms < job.state.next_run_at_ms:
                continue
            results.append(await self._execute_job(job, current_ms=current_ms))
        self._save_store()
        self._arm_timer()
        return results

    def _build_schedule(
        self,
        *,
        every_seconds: int | None,
        cron_expr: str | None,
        tz: str | None,
        at: str | None,
    ) -> CronSchedule:
        populated = sum(
            1
            for value in (
                every_seconds if every_seconds is not None else None,
                cron_expr.strip() if isinstance(cron_expr, str) else None,
                at.strip() if isinstance(at, str) else None,
            )
            if value
        )
        if populated != 1:
            raise ValueError("exactly one of every_seconds, cron_expr, or at is required")
        if every_seconds is not None:
            if every_seconds <= 0:
                raise ValueError("every_seconds must be greater than 0")
            return CronSchedule(kind="every", every_ms=every_seconds * 1000)
        if cron_expr:
            return CronSchedule(kind="cron", expr=cron_expr.strip(), tz=(tz or "").strip() or None)
        assert at is not None
        return CronSchedule(kind="at", at_ms=_parse_at_to_ms(at))

    def _load_store(self) -> CronStore:
        if self._store is not None:
            return self._store
        if not self.store_path.exists():
            self._store = CronStore()
            return self._store
        text = self.store_path.read_text(encoding="utf-8").strip()
        if not text:
            self._store = CronStore()
            return self._store
        raw = json.loads(text)
        jobs = [
            CronJob(
                id=item["id"],
                name=item["name"],
                enabled=item.get("enabled", True),
                schedule=CronSchedule(
                    kind=item["schedule"]["kind"],
                    at_ms=item["schedule"].get("atMs"),
                    every_ms=item["schedule"].get("everyMs"),
                    expr=item["schedule"].get("expr"),
                    tz=item["schedule"].get("tz"),
                ),
                payload=CronPayload(
                    kind=item.get("payload", {}).get("kind", "agent_turn"),
                    message=item.get("payload", {}).get("message", ""),
                    deliver=item.get("payload", {}).get("deliver", True),
                    channel=item.get("payload", {}).get("channel"),
                    chat_id=item.get("payload", {}).get("chatId"),
                    message_thread_id=item.get("payload", {}).get("messageThreadId"),
                ),
                state=CronJobState(
                    next_run_at_ms=item.get("state", {}).get("nextRunAtMs"),
                    last_run_at_ms=item.get("state", {}).get("lastRunAtMs"),
                    last_status=item.get("state", {}).get("lastStatus"),
                    last_error=item.get("state", {}).get("lastError"),
                ),
                created_at_ms=item.get("createdAtMs", 0),
                updated_at_ms=item.get("updatedAtMs", 0),
                delete_after_run=item.get("deleteAfterRun", False),
            )
            for item in raw.get("jobs", [])
        ]
        self._store = CronStore(version=raw.get("version", 1), jobs=jobs)
        return self._store

    def _save_store(self) -> None:
        store = self._load_store()
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": store.version,
            "jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "enabled": job.enabled,
                    "schedule": {
                        "kind": job.schedule.kind,
                        "atMs": job.schedule.at_ms,
                        "everyMs": job.schedule.every_ms,
                        "expr": job.schedule.expr,
                        "tz": job.schedule.tz,
                    },
                    "payload": {
                        "kind": job.payload.kind,
                        "message": job.payload.message,
                        "deliver": job.payload.deliver,
                        "channel": job.payload.channel,
                        "chatId": job.payload.chat_id,
                        "messageThreadId": job.payload.message_thread_id,
                    },
                    "state": {
                        "nextRunAtMs": job.state.next_run_at_ms,
                        "lastRunAtMs": job.state.last_run_at_ms,
                        "lastStatus": job.state.last_status,
                        "lastError": job.state.last_error,
                        "runHistory": [],
                    },
                    "createdAtMs": job.created_at_ms,
                    "updatedAtMs": job.updated_at_ms,
                    "deleteAfterRun": job.delete_after_run,
                }
                for job in store.jobs
            ],
        }
        self.store_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _arm_timer(self) -> None:
        import asyncio

        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None
        if not self._running:
            return

        # Resolve the event loop: prefer current thread's loop, fall back
        # to the loop captured at start().
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._loop

        if loop is None:
            self._pending_arm = True
            return

        next_wake = min(
            (
                job.state.next_run_at_ms
                for job in self._load_store().jobs
                if job.enabled and job.state.next_run_at_ms is not None
            ),
            default=None,
        )
        if next_wake is None:
            return
        delay_s = max(0, (next_wake - _now_ms()) / 1000)

        async def _tick() -> None:
            await asyncio.sleep(delay_s)
            if self._running:
                await self.trigger_due_jobs()

        # If called from a worker thread (no running loop in this thread),
        # schedule _arm_timer on the main loop thread instead of creating
        # the coroutine here — avoids "coroutine was never awaited".
        in_loop_thread = False
        try:
            in_loop_thread = asyncio.get_running_loop() is loop
        except RuntimeError:
            pass

        if in_loop_thread:
            self._timer_task = loop.create_task(_tick())
        else:
            loop.call_soon_threadsafe(self._arm_timer)

    async def _execute_job(self, job: CronJob, *, current_ms: int) -> str | None:
        started_ms = _now_ms()
        result: str | None = None
        error: str | None = None
        try:
            if self.on_job is not None:
                result = await self.on_job(job)
            if result and self.on_notify is not None and job.payload.deliver:
                await self.on_notify(job, result)
            job.state.last_status = "ok"
        except Exception as exc:
            error = str(exc)
            job.state.last_status = "error"
            job.state.last_error = error
        finally:
            finished_ms = _now_ms()
            job.state.last_run_at_ms = current_ms
            if job.state.last_status == "ok":
                job.state.last_error = None
            if job.delete_after_run:
                self._load_store().jobs = [item for item in self._load_store().jobs if item.id != job.id]
            else:
                job.state.next_run_at_ms = _compute_next_run(job.schedule, current_ms)
                job.updated_at_ms = finished_ms
        return result if error is None else error
