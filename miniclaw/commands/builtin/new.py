from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("new", description="Start a fresh session")
def cmd_new(ctx: CommandContext) -> CommandResult:
    runtime = ctx.runtime_service
    if runtime is None:
        return CommandResult(text="Runtime not available.")
    reset = getattr(runtime, "reset_thread", None)
    if callable(reset):
        reset(thread_id=ctx.thread_id)
    set_stopped = getattr(runtime, "set_thread_stopped", None)
    if callable(set_stopped):
        set_stopped(thread_id=ctx.thread_id, stopped=False)
    return CommandResult(text=f"Started a fresh session for thread={ctx.thread_id}")
