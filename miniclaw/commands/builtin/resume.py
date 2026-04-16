from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("resume_run", description="Resume from checkpoint with optional message")
def cmd_resume_run(ctx: CommandContext) -> CommandResult:
    runtime = ctx.runtime_service
    if runtime is None:
        return CommandResult(text="Runtime not available.")
    user_input = ctx.args.strip() or "Continue from the latest state."
    try:
        result = runtime.resume_run(thread_id=ctx.thread_id, user_input=user_input)
        response = str(result.response_text)
        error = str(result.last_error)
        return CommandResult(text=response or error or "Resumed.")
    except RuntimeError as exc:
        return CommandResult(text=str(exc))
