from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("ping", description="Check bot status and show current thread ID")
def cmd_ping(ctx: CommandContext) -> CommandResult:
    return CommandResult(text=f"🏓 pong\nThread: {ctx.thread_id}\nChannel: {ctx.channel}")
