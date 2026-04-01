from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("retry", description="Retry the last turn")
def cmd_retry(ctx: CommandContext) -> CommandResult:
    runtime = ctx.runtime_service
    if runtime is None:
        return CommandResult(text="Runtime not available.")
    retry = getattr(runtime, "retry_last_turn", None)
    if not callable(retry):
        return CommandResult(text="Retry is not available.")
    try:
        result = retry(thread_id=ctx.thread_id)
        response = str(getattr(result, "response_text", ""))
        error = str(getattr(result, "last_error", ""))
        return CommandResult(text=response or error or "Retried.")
    except RuntimeError as exc:
        return CommandResult(text=str(exc))
