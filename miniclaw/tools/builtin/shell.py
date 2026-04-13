from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool

# Commands whose invocation indicates state mutation or side effects. Blocked
# anywhere in a pipeline/chain/subshell. The model is expected to use
# write_file for writes inside its sandbox; shell handles queries + redirects.
_DANGEROUS_COMMANDS = {
    # Irrecoverable / system-level destruction only
    "dd", "mkfs", "mount", "umount", "shred",
    # Deletion (the classic footguns)
    "rm", "rmdir", "unlink",
    # NOTE: cp / mv / ln / tee / install — allowed. AI uses these routinely.
    # NOTE: chmod / chown — allowed. Needed for executable bits, script setup.
    # NOTE: sudo / su — allowed. User can decline at the permission prompt.
    # NOTE: ssh / scp / rsync / curl / wget — allowed. Network ops are fine.
    # NOTE: apt / yum / dnf / pacman / brew / pip / npm / pnpm / yarn / uv —
    #       allowed. Package managers are frequently needed.
    # NOTE: mkdir / touch / echo-redirect — allowed; benign.
    # NOTE: git state-changing subcommands still handled via _check_segment.
}

# Tool-specific flags that turn a read-only command into a write. Detected
# via token-level inspection (not substring match).
_WRITE_FLAG_PATTERNS: dict[str, set[str]] = {
    "sed": {"-i", "--in-place"},
    "awk": {"-i"},
    "perl": {"-i"},
    "find": {"-delete"},  # -exec kept allowed; find -exec is too common in read-only use
}

_BLOCKED_GIT_SUBCOMMANDS = {
    "am", "apply", "checkout", "cherry-pick", "clean", "clone",
    "commit", "config", "fetch", "gc", "init", "merge", "mv", "pull", "push",
    "rebase", "reset", "restore", "revert", "rm", "stash", "submodule", "switch",
    "tag", "update-ref", "worktree",
}

# Shell operators that chain commands. We split on these and check each
# segment's first token. This catches `echo x | rm foo` without blocking the
# operators themselves.
_CHAIN_SPLIT_RE = re.compile(r"\|\||&&|\||;|&(?!>)")

# Subshell / command-substitution patterns. We extract the inner commands
# and check them recursively so `echo $(rm foo)` is still caught.
_DOLLAR_PAREN_RE = re.compile(r"\$\(([^()]*)\)")
_BACKTICK_RE = re.compile(r"`([^`]*)`")


def build_shell_tool(*, workspace: Path, timeout: float = 60.0) -> RegisteredTool:
    root = Path(workspace).resolve()

    def execute(call: ToolCall) -> ToolResult:
        command = str(call.arguments.get("command", "")).strip()
        if not command:
            return ToolResult(content="command is required", is_error=True)
        if _looks_dangerous(command):
            return ToolResult(
                content=(
                    f"command blocked: a destructive command is present in the pipeline.\n"
                    f"  blocked: {command}\n"
                    f"Filter checks each segment of pipes/chains/subshells for "
                    f"commands like rm, chmod, sudo, git commit/push/reset, "
                    f"tee, install, and sed -i / find -delete / etc. "
                    f"Pipes, redirects, and chains themselves are fine — only the "
                    f"specific blocked commands are rejected. For sandbox-scoped "
                    f"file creation consider the write_file tool."
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
                "Run a shell command inside the workspace root. Pipes ( | ), "
                "redirects ( > >> < ), chains ( && || ; ), and subshells "
                "( $() ) are ALL supported — use them freely for text processing, "
                "inspection, and file I/O. The filter only blocks specific "
                "destructive commands (rm, chmod, sudo, "
                "git state-changing subcommands). Prefer the write_file tool for "
                "creating files inside your sandbox when the path is known; use "
                "shell for queries, redirects, and pipelines."
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
    """Command-name-based filter.

    Pipes, redirects, chains, and subshells are allowed; we just check every
    command in every pipeline/chain/subshell segment against the denylist.
    This way `cat foo | grep bar > /tmp/out` works, but
    `echo foo | rm -rf /` is still blocked at the `rm` segment.
    """
    # Extract and recursively check subshell contents first. A blocked
    # command anywhere inside $(...) or `...` is still blocked.
    for inner in _DOLLAR_PAREN_RE.findall(command):
        if _looks_dangerous(inner):
            return True
    for inner in _BACKTICK_RE.findall(command):
        if _looks_dangerous(inner):
            return True
    # Strip subshells from the outer command so we don't double-process them
    # in the segment check below (and so shlex doesn't choke on unbalanced
    # parens when they contained quotes).
    outer = _BACKTICK_RE.sub(" ", _DOLLAR_PAREN_RE.sub(" ", command))

    # Split by shell chain operators and check each segment.
    segments = [s.strip() for s in _CHAIN_SPLIT_RE.split(outer)]
    for segment in segments:
        if not segment:
            continue
        if _check_segment(segment):
            return True
    return False


def _check_segment(segment: str) -> bool:
    """Check a single pipeline segment against the command denylist."""
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        # Unbalanced quotes etc. — reject to stay safe.
        return True
    if not tokens:
        return False

    first = Path(tokens[0]).name

    # Skip redirect-target tokens if the "command" is actually just a redirect
    # target leftover (e.g., after split on '|', we shouldn't see redirects).
    if first in _DANGEROUS_COMMANDS:
        return True

    # Tool-specific write flags
    if first in _WRITE_FLAG_PATTERNS:
        blocked_flags = _WRITE_FLAG_PATTERNS[first]
        for tok in tokens[1:]:
            if tok in blocked_flags or any(tok.startswith(f + "=") for f in blocked_flags):
                return True

    # Git: block state-changing subcommands
    if first == "git" and len(tokens) >= 2:
        # Skip leading flags to find the subcommand (e.g., `git -C dir status`)
        sub = None
        for tok in tokens[1:]:
            if tok.startswith("-"):
                continue
            sub = tok
            break
        if sub and sub in _BLOCKED_GIT_SUBCOMMANDS:
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
