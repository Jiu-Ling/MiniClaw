"""Decorator for declaring command metadata."""
from __future__ import annotations

from miniclaw.commands.registry import CommandHandler, CommandMeta

# Module-level collector for auto-discovery
_REGISTERED_COMMANDS: list[tuple[CommandHandler, CommandMeta]] = []


def command(
    name: str,
    *,
    description: str,
    aliases: list[str] | None = None,
    hidden: bool = False,
):
    """Decorator to declare a command with metadata.

    Usage:
        @command("status", description="Show thread status")
        def cmd_status(ctx: CommandContext) -> CommandResult:
            ...
    """
    def decorator(fn: CommandHandler) -> CommandHandler:
        meta = CommandMeta(
            name=name,
            description=description,
            aliases=aliases or [],
            hidden=hidden,
        )
        fn._command_meta = meta  # type: ignore[attr-defined]
        _REGISTERED_COMMANDS.append((fn, meta))
        return fn
    return decorator


def collect_commands() -> list[tuple[CommandHandler, CommandMeta]]:
    """Return all commands registered via @command decorator."""
    return list(_REGISTERED_COMMANDS)
