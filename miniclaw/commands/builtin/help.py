from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("help", description="Show available commands", aliases=["start"])
def cmd_help(ctx: CommandContext) -> CommandResult:
    registry = ctx.registry
    if registry is not None:
        build_help = getattr(registry, "build_help_text", None)
        if callable(build_help):
            return CommandResult(text=build_help())
    return CommandResult(text="MiniClaw — use /help for commands")
