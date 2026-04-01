"""Built-in commands package.

Import all command modules to trigger @command decorator registration,
then provide register_builtin_commands() for the entry point.
"""
from __future__ import annotations

# Import all command modules to trigger @command registration
from miniclaw.commands.builtin import clear, help, model, new, ping, resume, retry, status, stop  # noqa: F401

from miniclaw.commands.decorator import collect_commands
from miniclaw.commands.registry import CommandRegistry


def register_builtin_commands(registry: CommandRegistry) -> None:
    """Discover and register all built-in commands."""
    for handler, meta in collect_commands():
        registry.register(handler, meta)
