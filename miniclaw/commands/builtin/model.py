from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("model", description="Show current model configuration")
def cmd_model(ctx: CommandContext) -> CommandResult:
    settings = ctx.settings
    if settings is None:
        return CommandResult(text="Settings not available.")
    model = settings.model
    base_url = settings.base_url
    mini = settings.mini_model or "not configured"
    mini_url = settings.mini_model_base_url or "N/A"
    lines = [
        f"Main model: {model}",
        f"Base URL: {base_url}",
        f"Mini model: {mini}",
        f"Mini model URL: {mini_url}",
    ]
    return CommandResult(text="\n".join(lines))
