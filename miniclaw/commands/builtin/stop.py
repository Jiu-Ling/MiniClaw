from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("stop", description="Stop current thread")
def cmd_stop(ctx: CommandContext) -> CommandResult:
    runtime = ctx.runtime_service
    if runtime is None:
        return CommandResult(text="Runtime not available.")
    runtime.set_thread_stopped(thread_id=ctx.thread_id, stopped=True)
    return CommandResult(text=f"Stopped thread={ctx.thread_id}. Use /resume_run to continue or /new to start over.")
