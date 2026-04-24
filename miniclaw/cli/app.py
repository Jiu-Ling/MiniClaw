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

memory_app = typer.Typer(name="memory", help="Memory management commands.", add_completion=False)
app.add_typer(memory_app, name="memory")


@memory_app.command("rebuild")
def memory_rebuild(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Drop and rebuild the memory index with parent-child chunking.

    Destructive: drops memory_chunks, memory_parents, memory_fts, memory_vec,
    and memory_dirty_files tables, then re-indexes all daily md files and
    MEMORY.md under ~/.miniclaw/memory/ with the new parent-child schema.

    Requires Ollama to be running for embedding generation.
    """
    from miniclaw.bootstrap import build_embedder, build_memory_indexer, build_settings

    settings = build_settings()
    memory_dir = settings.runtime_dir / "memory"
    memory_md = settings.runtime_dir / "MEMORY.md"

    md_files = sorted(memory_dir.glob("*.md")) if memory_dir.is_dir() else []
    has_memory_md = memory_md.is_file()

    typer.echo(f"Memory dir:  {memory_dir}")
    typer.echo(f"MEMORY.md:   {memory_md} ({'exists' if has_memory_md else 'missing'})")
    typer.echo(f"Daily files: {len(md_files)}")
    typer.echo(f"SQLite:      {settings.sqlite_path}")
    typer.echo()
    typer.echo("This will DROP all memory index tables and rebuild from scratch.")

    if not yes:
        typer.confirm("Continue?", abort=True)

    # Backup MEMORY.md before destructive operation
    if has_memory_md:
        bak = memory_md.with_suffix(".md.bak")
        import shutil
        shutil.copy2(memory_md, bak)
        typer.echo(f"Backed up MEMORY.md → {bak.name}")

    try:
        embedder = build_embedder(settings)
        indexer = build_memory_indexer(settings, embedder)
    except Exception as exc:
        typer.echo(f"Failed to initialize embedder/indexer: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Rebuilding index (this may take a minute for embedding)...")
    typer.echo("NOTE: Requires Ollama running at the configured ollama_base_url.")
    try:
        asyncio.run(indexer.rebuild_all())
    except Exception as exc:
        error_msg = str(exc) or f"{type(exc).__name__} (no message)"
        typer.echo(f"Rebuild failed: {error_msg}", err=True)
        raise typer.Exit(code=1)

    # Count results
    import sqlite3
    with sqlite3.connect(settings.sqlite_path) as conn:
        parents = conn.execute("SELECT count(*) FROM memory_parents").fetchone()[0]
        children = conn.execute("SELECT count(*) FROM memory_chunks").fetchone()[0]

    typer.echo(f"Done: {parents} parents, {children} children indexed.")
    typer.echo("Memory index is ready. Restart miniclaw to use the new index.")


@trace_app.command("tail")
def trace_tail(
    path: Path = typer.Argument(..., help="Trace JSONL file path"),
    follow: bool = typer.Option(True, "--follow/--no-follow"),
    tool: str | None = typer.Option(None, "--tool", help="Filter to tool.<name> spans (exact name after 'tool.' prefix)."),
    kind: str | None = typer.Option(None, "--kind", help="Filter by kind(s), comma-separated (e.g. span_finish,event)."),
    min_ms: int = typer.Option(0, "--min-ms", help="Only show spans whose duration is >= N ms."),
    status: str | None = typer.Option(None, "--status", help="Filter by status value (e.g. ok, error)."),
    color: bool = typer.Option(True, "--color/--no-color", help="Enable or disable colored output."),
) -> None:
    """Pretty-print a trace JSONL file with duration, tool args, cache stats, and filters."""
    import sys
    import time

    from miniclaw.cli.trace_renderer import SpanState, format_record

    if not path.exists():
        typer.echo(f"trace file not found: {path}", err=True)
        raise typer.Exit(1)

    # Disable color when stdout is not a TTY
    use_color = color and sys.stdout.isatty()

    # Parse filter sets
    kind_filter: set[str] | None = None
    if kind:
        kind_filter = {k.strip() for k in kind.split(",") if k.strip()}

    span_state: dict[str, SpanState] = {}

    def _process_line(raw: str) -> None:
        raw = raw.strip()
        if not raw:
            return
        parsed = safe_loads(raw)
        if not isinstance(parsed, dict):
            return
        line = format_record(
            parsed,
            span_state,
            color=use_color,
            min_ms=min_ms,
            tool_filter=tool,
            kind_filter=kind_filter,
            status_filter=status,
        )
        if line is not None:
            typer.echo(line)

    with path.open("r") as fh:
        for raw_line in fh:
            _process_line(raw_line)

        if not follow:
            return

        try:
            while True:
                raw_line = fh.readline()
                if not raw_line:
                    time.sleep(0.2)
                    continue
                _process_line(raw_line)
        except KeyboardInterrupt:
            return


@trace_app.command("summary")
def trace_summary(
    path: Path = typer.Argument(..., help="Trace JSONL file path"),
    top_n: int = typer.Option(10, "--top-n", help="Top N entries per category"),
) -> None:
    """Aggregate stats from a trace JSONL file: tools, spans, cache, errors."""
    from miniclaw.cli.trace_renderer import aggregate_trace, render_summary

    if not path.exists():
        typer.echo(f"trace file not found: {path}", err=True)
        raise typer.Exit(1)

    def _records():
        with path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parsed = safe_loads(line)
                if isinstance(parsed, dict):
                    yield parsed

    summary = aggregate_trace(_records())
    typer.echo(render_summary(summary, top_n=top_n))


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
