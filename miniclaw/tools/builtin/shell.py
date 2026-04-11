from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool

# Commands that mutate state are blocked — shell is read-only. Model must use
# write_file for creates/modifications (scoped to user sandbox).
_DANGEROUS_COMMANDS = {
    # Privilege / network / filesystem-mutating classics
    "chmod", "chown", "dd", "mkfs", "mount", "umount",
    "scp", "ssh", "sudo", "su",
    # Deletion
    "rm", "rmdir", "unlink", "shred",
    # Writes / copies / moves / creates
    "cp", "mv", "touch", "mkdir", "tee", "install", "ln", "rsync",
    # In-place editors
    "patch",
    # Package managers (state-mutating)
    "apt", "apt-get", "yum", "dnf", "pacman", "brew", "pip", "npm", "pnpm", "yarn",
    # Git state-changing subcommands are handled via _looks_dangerous below,
    # since "git" itself is a valid read-only tool (git log/diff/status).
}

# Characters that enable shell redirection, command chaining, subshells, or
# pipelines (pipes chain commands, and only checking the first segment would
# let `true | rm foo` through). Any of these makes the command unsafe.
_DANGEROUS_TOKENS = (">", "<", ";", "|", "&", "`", "$(", "\n", "\r")

# Arg patterns that turn a read-only command into a write. Detected via
# token-level inspection (not just substring match) to avoid false positives.
_WRITE_FLAG_PATTERNS = {
    "sed": {"-i", "--in-place"},
    "awk": {"-i"},
    "perl": {"-i"},
    "find": {"-delete", "-exec", "-execdir"},
    "xargs": set(),  # xargs can chain any command; block entirely
}

_BLOCKED_GIT_SUBCOMMANDS = {
    "add", "am", "apply", "branch", "checkout", "cherry-pick", "clean", "clone",
    "commit", "config", "fetch", "gc", "init", "merge", "mv", "pull", "push",
    "rebase", "reset", "restore", "revert", "rm", "stash", "submodule", "switch",
    "tag", "update-ref", "worktree",
}


def build_shell_tool(*, workspace: Path, timeout: float = 60.0) -> RegisteredTool:
    root = Path(workspace).resolve()

    def execute(call: ToolCall) -> ToolResult:
        command = str(call.arguments.get("command", "")).strip()
        if not command:
            return ToolResult(content="command is required", is_error=True)
        if _looks_dangerous(command):
            return ToolResult(
                content=(
                    f"command blocked: shell is read-only.\n"
                    f"  blocked: {command}\n"
                    f"Use read-only commands (ls/cat/head/tail/grep/find/git log/git diff/etc.) "
                    f"for inspection, and use the write_file tool for all creates and modifications "
                    f"(scoped to your user sandbox)."
                ),
                is_error=True,
            )

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
                "Run ONE read-only shell command inside the workspace root. "
                "Allowed: ls, cat, head, tail, grep, find (without -delete/-exec), "
                "wc, sort, uniq, diff, git log/diff/show/status/blame, etc. "
                "BLOCKED: any write, copy, move, delete, redirect, pipe-chain, subshell, "
                "or package manager. For all file writes use the write_file tool "
                "(scoped to your user sandbox). No '>', no '|', no ';', no '&&'."
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
    # Reject any shell redirection / chaining / subshell metacharacter.
    if any(token in command for token in _DANGEROUS_TOKENS):
        return True

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return True
    if not tokens:
        return True

    first = Path(tokens[0]).name
    if first in _DANGEROUS_COMMANDS:
        return True

    # Tool-specific write flags (e.g., sed -i, find -delete, xargs anything)
    if first in _WRITE_FLAG_PATTERNS:
        blocked_flags = _WRITE_FLAG_PATTERNS[first]
        if not blocked_flags:
            return True  # entirely blocked (xargs)
        for tok in tokens[1:]:
            if tok in blocked_flags or any(tok.startswith(f + "=") for f in blocked_flags):
                return True

    # Git: allow read-only subcommands (log, diff, show, status, blame, etc.)
    if first == "git" and len(tokens) >= 2:
        sub = tokens[1].lstrip("-")
        if sub in _BLOCKED_GIT_SUBCOMMANDS:
            return True

    return False


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
