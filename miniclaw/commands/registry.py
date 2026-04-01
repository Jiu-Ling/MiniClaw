from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommandContext:
    thread_id: str
    channel: str
    settings: object | None = None
    runtime_service: object | None = None
    args: str = ""  # text after the command name
    registry: object | None = None  # CommandRegistry ref for help command


@dataclass(frozen=True)
class CommandMeta:
    name: str
    description: str
    aliases: list[str] = field(default_factory=list)
    hidden: bool = False  # hidden commands don't show in help


@dataclass(frozen=True)
class CommandResult:
    text: str
    handled: bool = True


CommandHandler = Callable[[CommandContext], CommandResult]


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, CommandHandler] = {}
        self._meta: dict[str, CommandMeta] = {}

    def register(self, handler: CommandHandler, meta: CommandMeta) -> None:
        key = meta.name.strip().lower()
        self._commands[key] = handler
        self._meta[key] = meta
        for alias in meta.aliases:
            alias_key = alias.strip().lower()
            self._commands[alias_key] = handler
            # aliases share same meta but don't appear separately

    def match(self, text: str) -> str | None:
        text = text.strip()
        if not text.startswith("/"):
            return None
        command = text.split()[0].removeprefix("/").strip().lower()
        return command if command in self._commands else None

    def extract_args(self, text: str) -> str:
        """Extract text after the command name."""
        parts = text.strip().split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""

    def execute(self, name: str, context: CommandContext) -> CommandResult:
        handler = self._commands.get(name.strip().lower())
        if handler is None:
            return CommandResult(text=f"Unknown command: /{name}", handled=False)
        return handler(context)

    def list_commands(self) -> list[str]:
        return sorted(self._meta.keys())

    def get_meta(self, name: str) -> CommandMeta | None:
        return self._meta.get(name.strip().lower())

    def build_help_text(self) -> str:
        lines = ["MiniClaw commands:\n"]
        for name in sorted(self._meta):
            meta = self._meta[name]
            if meta.hidden:
                continue
            aliases = ""
            if meta.aliases:
                aliases = f" (alias: {', '.join('/' + a for a in meta.aliases)})"
            lines.append(f"/{meta.name} — {meta.description}{aliases}")
        return "\n".join(lines)
