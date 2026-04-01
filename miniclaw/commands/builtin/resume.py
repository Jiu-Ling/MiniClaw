from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("resume_run", description="Resume from checkpoint with optional message")
def cmd_resume_run(ctx: CommandContext) -> CommandResult:
    runtime = ctx.runtime_service
    if runtime is None:
        return CommandResult(text="Runtime not available.")
    resume_run = getattr(runtime, "resume_run", None)
    if not callable(resume_run):
        return CommandResult(text="Resume is not available.")
    user_input = ctx.args.strip() or "Continue from the latest state."
    try:
        result = resume_run(thread_id=ctx.thread_id, user_input=user_input)
        response = str(getattr(result, "response_text", ""))
        error = str(getattr(result, "last_error", ""))
        return CommandResult(text=response or error or "Resumed.")
    except RuntimeError as exc:
        return CommandResult(text=str(exc))
