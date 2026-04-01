from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("clear", description="Clear current thread checkpoint")
def cmd_clear(ctx: CommandContext) -> CommandResult:
    runtime = ctx.runtime_service
    if runtime is None:
        return CommandResult(text="Runtime not available.")
    reset = getattr(runtime, "reset_thread", None)
    if callable(reset):
        reset(thread_id=ctx.thread_id)
    return CommandResult(text=f"Cleared checkpoint for thread={ctx.thread_id}")
