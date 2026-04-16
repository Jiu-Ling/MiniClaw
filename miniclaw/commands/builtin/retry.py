from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("retry", description="Retry the last turn")
def cmd_retry(ctx: CommandContext) -> CommandResult:
    runtime = ctx.runtime_service
    if runtime is None:
        return CommandResult(text="Runtime not available.")
    try:
        result = runtime.retry_last_turn(thread_id=ctx.thread_id)
        response = str(result.response_text)
        error = str(result.last_error)
        return CommandResult(text=response or error or "Retried.")
    except RuntimeError as exc:
        return CommandResult(text=str(exc))
