from __future__ import annotations

from miniclaw.commands.decorator import command
from miniclaw.commands.registry import CommandContext, CommandResult


@command("model", description="Show current model configuration")
def cmd_model(ctx: CommandContext) -> CommandResult:
    settings = ctx.settings
    if settings is None:
        return CommandResult(text="Settings not available.")
    model = getattr(settings, "model", "unknown")
    base_url = getattr(settings, "base_url", "unknown")
    mini = getattr(settings, "mini_model", None) or "not configured"
    mini_url = getattr(settings, "mini_model_base_url", None) or "N/A"
    lines = [
        f"Main model: {model}",
        f"Base URL: {base_url}",
        f"Mini model: {mini}",
        f"Mini model URL: {mini_url}",
    ]
    return CommandResult(text="\n".join(lines))
