from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("help", description="Show available commands")
def cmd_help(ctx: CommandContext) -> CommandResult:
    registry = ctx.registry
    if registry is not None:
        return CommandResult(text=registry.build_help_text())
    return CommandResult(text="MiniClaw — use /help for commands")
