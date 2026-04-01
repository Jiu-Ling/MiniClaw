from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool

_DANGEROUS_COMMANDS = {
    "chmod",
    "chown",
    "dd",
    "mkfs",
    "mount",
    "rm",
    "rmdir",
    "scp",
    "ssh",
    "sudo",
    "su",
    "umount",
}

_DANGEROUS_TOKENS = (";", "`", "$(", ">", "<", "\n", "\r")


def build_shell_tool(*, workspace: Path, timeout: float = 60.0) -> RegisteredTool:
    root = Path(workspace).resolve()

    def execute(call: ToolCall) -> ToolResult:
        command = str(call.arguments.get("command", "")).strip()
        if not command:
            return ToolResult(content="command is required", is_error=True)
        if _looks_dangerous(command):
            return ToolResult(content=f"command blocked by safety guard: {command}", is_error=True)

        try:
            completed = subprocess.run(
                command,
                cwd=root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                content=_format_timeout(command=command, workspace=root, timeout=timeout),
                is_error=True,
                metadata={"command": command, "cwd": str(root), "timeout_seconds": timeout},
            )

        content = _format_result(command=command, workspace=root, completed=completed)
        return ToolResult(
            content=content,
            is_error=completed.returncode != 0,
            metadata={
                "command": command,
                "cwd": str(root),
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )

    return RegisteredTool(
        spec=ToolSpec(
            name="shell",
            description=(
                "Run one simple command inside the workspace root when local shell access "
                "is the right next step. Use it for bounded, non-destructive commands only."
            ),
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
            source="builtin",
            metadata={"discoverable": True},
        ),
        executor=execute,
    )


def _looks_dangerous(command: str) -> bool:
    # if any(token in command for token in _DANGEROUS_TOKENS):
    #     return True

    try:
        first_token = shlex.split(command, posix=True)[0]
    except (ValueError, IndexError):
        return True

    return Path(first_token).name in _DANGEROUS_COMMANDS


def _format_result(*, command: str, workspace: Path, completed: subprocess.CompletedProcess[str]) -> str:
    stdout = completed.stdout.rstrip()
    stderr = completed.stderr.rstrip()
    return "\n".join(
        [
            f"Command: {command}",
            f"Working directory: {workspace}",
            f"Exit code: {completed.returncode}",
            "Stdout:",
            stdout if stdout else "<empty>",
            "Stderr:",
            stderr if stderr else "<empty>",
        ]
    )


def _format_timeout(*, command: str, workspace: Path, timeout: float) -> str:
    return "\n".join(
        [
            f"Command: {command}",
            f"Working directory: {workspace}",
            f"Status: timed out after {timeout} seconds",
            "Stdout:",
            "<empty>",
            "Stderr:",
            "<empty>",
        ]
    )


__all__ = ["build_shell_tool"]
