from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from miniclaw.bootstrap import (
    build_cron_service,
    build_heartbeat_service,
    build_mini_provider,
    build_runtime_service,
    build_settings,
    initialize_local_storage,
)
from miniclaw.channels.access import ChannelAccessStore
from miniclaw.channels.cli.channel import CLIChannel
from miniclaw.channels.loop import MessageLoop
from miniclaw.channels.telegram.channel import TelegramChannel
from miniclaw.commands.builtin import register_builtin_commands
from miniclaw.commands.registry import CommandRegistry
from miniclaw.cron.service import CronService
from miniclaw.runtime.background import BackgroundScheduler
from miniclaw.utils.jsonx import safe_loads

app = typer.Typer(
    name="miniclaw",
    help="MiniClaw runtime CLI",
    add_completion=False,
)
telegram_app = typer.Typer(
    name="telegram",
    help="Telegram channel commands.",
    add_completion=False,
)
app.add_typer(telegram_app, name="telegram")

trace_app = typer.Typer(name="trace", help="Trace inspection commands.", add_completion=False)
app.add_typer(trace_app, name="trace")


@trace_app.command("tail")
def trace_tail(
    path: Path = typer.Argument(..., help="Trace JSONL file path"),
    follow: bool = typer.Option(True, "--follow/--no-follow"),
) -> None:
    """Pretty-print a trace JSONL file with hierarchical indentation."""
    import time

    if not path.exists():
        typer.echo(f"trace file not found: {path}", err=True)
        raise typer.Exit(1)

    span_depth: dict[str, int] = {}

    def _depth_for(rec: dict) -> int:
        kind = rec.get("kind", "")
        span_id = rec.get("span_id") or ""
        parent = rec.get("parent_span_id")
        if parent and parent in span_depth:
            return span_depth[parent] + 1
        if kind == "run_start":
            return 0
        return span_depth.get(span_id, 0)

    def _print_record(rec: dict) -> None:
        kind = rec.get("kind", "")
        name = rec.get("name", "")
        span_id = rec.get("span_id") or ""
        depth = _depth_for(rec)
        if kind in ("run_start", "span_start"):
            span_depth[span_id] = depth
        prefix = "  " * depth
        marker = {
            "run_start": "▶",
            "run_finish": "■",
            "span_start": "├",
            "span_finish": "└",
            "event": "•",
        }.get(kind, "·")
        status = rec.get("status", "")
        suffix = f" [{status}]" if status else ""
        typer.echo(f"{prefix}{marker} {name}{suffix}")

    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parsed = safe_loads(line)
            if parsed is None:
                continue
            _print_record(parsed)

        if not follow:
            return

        try:
            while True:
                line = fh.readline()
                if not line:
                    time.sleep(0.2)
                    continue
                line = line.strip()
                if line:
                    parsed = safe_loads(line)
                    if parsed is not None:
                        _print_record(parsed)
        except KeyboardInterrupt:
            return


@app.callback()
def main() -> None:
    """MiniClaw runtime CLI."""
    return


def _resolve_thread_id(thread_id: str | None) -> str:
    if thread_id is not None:
        return thread_id
    return build_settings().default_thread_id


@app.command()
def init(
    sqlite_path: Path | None = typer.Option(
        None,
        "--sqlite-path",
        help="Override the local SQLite storage path.",
    ),
) -> None:
    """Initialize local runtime data storage."""
    path = initialize_local_storage(sqlite_path)
    typer.echo(f"Initialized MiniClaw storage at {path}")


@app.command()
def chat(
    message: str = typer.Argument(..., help="User message to send."),
    thread_id: str | None = typer.Option(
        None,
        "--thread-id",
        help="Runtime thread identifier.",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Stream text output when supported by the provider.",
    ),
) -> None:
    """Send one message through the runtime."""
    from rich.console import Console
    from rich.markdown import Markdown

    runtime = build_runtime_service()
    resolved_thread_id = _resolve_thread_id(thread_id)
    result = runtime.run_turn(thread_id=resolved_thread_id, user_input=message)
    Console().print(Markdown(result.response_text or result.last_error or ""))


@app.command()
def resume(
    thread_id: str | None = typer.Option(
        None,
        "--thread-id",
        help="Runtime thread identifier.",
    ),
) -> None:
    """Show the latest checkpointed state for a thread."""
    resolved_thread_id = _resolve_thread_id(thread_id)
    runtime = build_runtime_service()
    checkpoint = runtime.resume_thread(thread_id=resolved_thread_id)
    if checkpoint is None:
        typer.echo(f"No checkpoint found for thread={resolved_thread_id}", err=True)
        raise typer.Exit(code=1)
    _render_checkpoint(checkpoint)


@app.command()
def repl(
    thread_id: str | None = typer.Option(
        None,
        "--thread-id",
        help="Runtime thread identifier.",
    ),
    stream: bool = typer.Option(
        False,
        "--stream",
        help="Stream text output when supported by the provider.",
    ),
) -> None:
    """Run a multi-turn REPL on one thread."""
    resolved_thread_id = _resolve_thread_id(thread_id)
    settings = build_settings()
    runtime = build_runtime_service(settings)
    channel = CLIChannel(thread_id=resolved_thread_id)
    commands = CommandRegistry()
    register_builtin_commands(commands)
    access = ChannelAccessStore(settings.sqlite_path)
    access.initialize()
    loop = MessageLoop(
        channel=channel,
        runtime=runtime,
        command_registry=commands,
        access_store=access,
    )
    asyncio.run(loop.run())


@app.command()
def pair(
    code: str = typer.Argument(None),
    list_pending: bool = typer.Option(False, "--list", help="List pending pairing requests."),
) -> None:
    """Authorize a channel for access."""
    settings = build_settings()
    access = ChannelAccessStore(settings.sqlite_path)
    access.initialize()
    if list_pending:
        pending = access.list_pending()
        if not pending:
            typer.echo("No pending pairing requests.")
            return
        for p in pending:
            typer.echo(f"[{p['status']}] {p['channel_id']} code={p['pair_code']} ({p['created_at']})")
        return
    if not code:
        typer.echo("Usage: miniclaw pair <code> or miniclaw pair --list")
        return
    channel_id = access.authorize(code)
    if channel_id:
        typer.echo(f"Authorized channel {channel_id}")
    else:
        typer.echo("Invalid or expired code.", err=True)
        raise typer.Exit(code=1)


@app.command()
def unpair(
    channel_id: str = typer.Argument(..., help="Channel ID to revoke access for."),
) -> None:
    """Revoke channel access."""
    settings = build_settings()
    access = ChannelAccessStore(settings.sqlite_path)
    access.initialize()
    access.revoke(channel_id)
    typer.echo(f"Revoked access for {channel_id}")


@telegram_app.command("polling")
def telegram_polling(
    timeout: int = typer.Option(
        30,
        "--timeout",
        min=1,
        help="Long-poll timeout in seconds.",
    ),
    offset: int | None = typer.Option(
        None,
        "--offset",
        help="Initial Telegram update offset.",
    ),
    once: bool = typer.Option(
        False,
        "--once",
        help="Process a single polling iteration and exit.",
    ),
) -> None:
    """Run the Telegram bot in polling mode."""
    asyncio.run(_run_telegram_polling(timeout=timeout, offset=offset, once=once))


async def _run_telegram_polling(*, timeout: int, offset: int | None, once: bool) -> None:
    """Run Telegram polling plus cron and heartbeat supervision."""
    settings = build_settings()
    if not settings.telegram_bot_token:
        typer.echo(
            "error: MINICLAW_TELEGRAM_BOT_TOKEN is required to start Telegram polling.",
            err=True,
        )
        raise typer.Exit(code=1)

    background_scheduler = BackgroundScheduler(
        name="miniclaw-bg",
        max_queue=settings.background_max_queue,
    )
    background_scheduler.start()

    mini_provider = build_mini_provider(settings)

    channel = TelegramChannel(bot_token=settings.telegram_bot_token, poll_timeout=timeout)

    # Build messaging bridge from channel — tools use this to send messages
    from miniclaw.channels.bridge import ChannelMessagingBridge
    # Default channel_id for bridge; actual channel_id is injected per-turn via ToolContext
    bridge = ChannelMessagingBridge(channel=channel, channel_id="")

    _shared_cron_service: CronService | None = None

    def _runtime_factory() -> object:
        return build_runtime_service(
            settings,
            cron_service=_shared_cron_service,
            messaging_bridge=bridge,
            mini_provider=mini_provider,
            background_scheduler=background_scheduler,
        )

    cron_service = build_cron_service(
        settings,
        on_notify=_build_cron_notify(channel),
        runtime_factory=_runtime_factory,
    )
    _shared_cron_service = cron_service

    runtime = build_runtime_service(
        settings,
        cron_service=cron_service,
        mini_provider=mini_provider,
        messaging_bridge=bridge,
        background_scheduler=background_scheduler,
    )

    commands = CommandRegistry()
    register_builtin_commands(commands)

    access = ChannelAccessStore(settings.sqlite_path)
    access.initialize()

    loop = MessageLoop(
        channel=channel,
        runtime=runtime,
        command_registry=commands,
        access_store=access,
        messaging_bridge=bridge,
    )

    heartbeat_service = build_heartbeat_service(
        settings,
        on_notify=_build_heartbeat_notify(channel, settings),
        runtime_factory=_runtime_factory,
        mini_provider=mini_provider,
    )

    typer.echo("MiniClaw Telegram polling started", err=True)
    try:
        await cron_service.start()
        await heartbeat_service.start()
        await loop.run()
    except KeyboardInterrupt:
        typer.echo("MiniClaw Telegram polling stopped", err=True)
    finally:
        cron_service.stop()
        heartbeat_service.stop()
        background_scheduler.stop(
            wait=True,
            timeout=settings.background_stop_timeout_s,
        )


def _build_cron_notify(channel: TelegramChannel):
    from miniclaw.channels.contracts import OutboundMessage

    async def _notify(job: object, text: str) -> None:
        payload = getattr(job, "payload", None)
        chan = str(getattr(payload, "channel", "") or "").strip().lower()
        chat_id = str(getattr(payload, "chat_id", "") or "").strip()
        thread_id = str(getattr(payload, "message_thread_id", "") or "").strip()
        if chan != "telegram" or not chat_id:
            return
        channel_id = f"tg:{chat_id}" + (f":{thread_id}" if thread_id else "")
        await channel.send_message(OutboundMessage(
            channel_id=channel_id,
            text=text,
        ))

    return _notify


def _build_heartbeat_notify(channel: TelegramChannel, settings: object):
    from miniclaw.channels.contracts import OutboundMessage

    async def _notify(text: str) -> None:
        chat_id = str(getattr(settings, "heartbeat_chat_id", "") or "").strip()
        thread_id = str(getattr(settings, "heartbeat_message_thread_id", "") or "").strip()
        if not chat_id:
            return
        channel_id = f"tg:{chat_id}" + (f":{thread_id}" if thread_id else "")
        await channel.send_message(OutboundMessage(
            channel_id=channel_id,
            text=text,
        ))

    return _notify


@app.command("graph")
def graph_command(
    fmt: str = typer.Option("mermaid", "--format", "-f", help="mermaid | ascii | png"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file path (for png)"),
) -> None:
    """Export the runtime graph topology."""
    from unittest.mock import MagicMock

    from miniclaw.runtime.graph import build_graph

    # Build the graph with stub providers — the topology does not depend on
    # provider behavior, so we can render it without real credentials.
    from miniclaw.config.settings import Settings

    settings = Settings(
        api_key="stub",
        base_url="https://stub.example.com/v1",
        model="stub-model",
        trace_mode="off",
    )
    stub_provider = MagicMock(name="StubProvider")
    stub_memory = MagicMock(name="StubMemoryStore")
    stub_memory.recent_messages.return_value = []

    graph = build_graph(
        settings=settings,
        provider=stub_provider,
        memory_store=stub_memory,
        tool_registry=None,
    )
    drawable = graph.compile().get_graph()

    if fmt == "mermaid":
        typer.echo(drawable.draw_mermaid())
    elif fmt == "ascii":
        try:
            typer.echo(drawable.draw_ascii())
        except Exception as exc:
            typer.echo(f"ASCII render failed: {exc}", err=True)
            typer.echo("Hint: install grandalf or use --format mermaid", err=True)
            raise typer.Exit(1)
    elif fmt == "png":
        try:
            png_bytes = drawable.draw_mermaid_png()
        except Exception as exc:
            typer.echo(f"PNG render failed: {exc}", err=True)
            typer.echo("Hint: install pyppeteer or use --format mermaid and pipe to mmdc", err=True)
            raise typer.Exit(1)
        target = output or Path("miniclaw-graph.png")
        target.write_bytes(png_bytes)
        typer.echo(f"Wrote {target}")
    else:
        typer.echo(f"Unknown format: {fmt}", err=True)
        raise typer.Exit(2)


def _render_checkpoint(checkpoint: object) -> None:
    response_text = str(getattr(checkpoint, "response_text", ""))
    if response_text:
        typer.echo(response_text)

    last_error = str(getattr(checkpoint, "last_error", ""))
    thread_id = str(getattr(checkpoint, "thread_id", ""))
    checkpoint_id = str(getattr(checkpoint, "checkpoint_id", ""))
    message_count = getattr(checkpoint, "message_count", 0)
    summary = f"thread={thread_id} checkpoint={checkpoint_id} messages={message_count}"
    if last_error:
        typer.echo(f"error: {last_error}", err=True)
        typer.echo(summary)
        raise typer.Exit(code=1)
    typer.echo(summary)
