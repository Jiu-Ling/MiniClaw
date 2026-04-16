from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("status", description="Show thread checkpoint status", aliases=["resume"])
def cmd_status(ctx: CommandContext) -> CommandResult:
    runtime = ctx.runtime_service
    if runtime is None:
        return CommandResult(text="Runtime not available.")
    checkpoint = runtime.resume_thread(thread_id=ctx.thread_id)
    if checkpoint is None:
        return CommandResult(text=f"No checkpoint found for thread={ctx.thread_id}")
    response_text = str(checkpoint.response_text)
    last_error = str(checkpoint.last_error)
    message_count = checkpoint.message_count
    checkpoint_id = str(checkpoint.checkpoint_id or "")
    lines = [f"Thread: {ctx.thread_id}", f"Checkpoint: {checkpoint_id}", f"Messages: {message_count}"]
    if last_error:
        lines.append(f"Last error: {last_error}")
    if response_text:
        preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
        lines.append(f"Last response: {preview}")
    return CommandResult(text="\n".join(lines))
