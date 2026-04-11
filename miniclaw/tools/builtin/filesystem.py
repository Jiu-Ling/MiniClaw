from __future__ import annotations

from pathlib import Path

from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool

MAX_READ_FILE_BYTES = 32 * 1024  # 32 KB
MAX_WRITE_FILE_BYTES = 1024 * 1024  # 1 MB
_WRITE_MODES = frozenset({"create", "overwrite", "append"})


def build_read_file_tool(*, workspace: Path, max_bytes: int = MAX_READ_FILE_BYTES) -> RegisteredTool:
    root = Path(workspace).resolve()
    limit = max_bytes

    def execute(call: ToolCall) -> ToolResult:
        raw_path = str(call.arguments.get("path", "")).strip()
        if not raw_path:
            return ToolResult(content="path is required", is_error=True)

        target = (root / raw_path).resolve()
        if target != root and root not in target.parents:
            return ToolResult(content="path escapes workspace", is_error=True)
        if not target.is_file():
            return ToolResult(content=f"file not found: {raw_path}", is_error=True)

        file_size = target.stat().st_size
        if file_size > limit:
            return ToolResult(
                content=(
                    f"File too large: {raw_path} is {file_size:,} bytes (limit {limit:,}).\n"
                    f"Use the `shell` tool to inspect this file instead. Examples:\n"
                    f"  head -n 50 '{raw_path}'          # first 50 lines\n"
                    f"  tail -n 50 '{raw_path}'          # last 50 lines\n"
                    f"  grep -n 'pattern' '{raw_path}'   # search for pattern\n"
                    f"  wc -l '{raw_path}'               # line count\n"
                    f"  sed -n '100,200p' '{raw_path}'   # lines 100-200\n"
                    f"See the 'large-file' skill for more strategies."
                ),
                is_error=True,
            )

        return ToolResult(content=target.read_text(encoding="utf-8"))

    return RegisteredTool(
        spec=ToolSpec(
            name="read_file",
            description=(
                "Read one known UTF-8 text file from the workspace. "
                "Use this when you already know the path and need the file body, "
                "not for binary files or directory browsing."
            ),
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
            source="builtin",
        ),
        executor=execute,
    )


def build_write_file_tool(*, max_bytes: int = MAX_WRITE_FILE_BYTES) -> RegisteredTool:
    """Write/create/append a file inside the current user's sandbox directory.

    The sandbox root is read from `ToolCall.context["user_sandbox"]`, which
    `MessageLoop` populates per-turn from the inbound message's sender.
    The tool rejects absolute paths and paths that would escape the sandbox.
    """
    limit = max_bytes

    def execute(call: ToolCall) -> ToolResult:
        raw_path = str(call.arguments.get("path", "")).strip()
        if not raw_path:
            return ToolResult(content="path is required", is_error=True)

        content = call.arguments.get("content")
        if not isinstance(content, str):
            return ToolResult(content="content is required and must be a string", is_error=True)

        mode = str(call.arguments.get("mode", "overwrite")).strip().lower() or "overwrite"
        if mode not in _WRITE_MODES:
            return ToolResult(
                content=f"invalid mode: {mode} (expected: create, overwrite, append)",
                is_error=True,
            )

        encoded_size = len(content.encode("utf-8"))
        if encoded_size > limit:
            return ToolResult(
                content=f"content too large: {encoded_size:,} bytes (limit {limit:,})",
                is_error=True,
            )

        sandbox_raw = str(call.context.get("user_sandbox", "")).strip()
        if not sandbox_raw:
            return ToolResult(
                content=(
                    "no user_sandbox configured for this turn — write_file is "
                    "only available when a user sandbox has been provisioned "
                    "by the channel layer"
                ),
                is_error=True,
            )

        sandbox = Path(sandbox_raw).resolve()
        if Path(raw_path).is_absolute():
            return ToolResult(
                content=f"path must be relative to the sandbox: {raw_path}",
                is_error=True,
            )

        target = (sandbox / raw_path).resolve()
        if target != sandbox and sandbox not in target.parents:
            return ToolResult(
                content=f"path escapes sandbox: {raw_path}",
                is_error=True,
            )

        try:
            sandbox.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ToolResult(content=f"could not create sandbox: {exc}", is_error=True)

        if mode == "create" and target.exists():
            return ToolResult(
                content=f"file already exists (mode=create): {raw_path}",
                is_error=True,
            )

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "append":
                with target.open("a", encoding="utf-8") as fh:
                    fh.write(content)
            else:
                target.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(content=f"write failed: {exc}", is_error=True)

        try:
            relative = target.relative_to(sandbox)
        except ValueError:
            relative = Path(raw_path)
        final_size = target.stat().st_size if target.exists() else encoded_size
        return ToolResult(
            content=f"wrote {final_size:,} bytes to {relative} ({mode})",
            metadata={
                "path": str(relative),
                "mode": mode,
                "bytes": final_size,
                "sandbox": str(sandbox),
            },
        )

    return RegisteredTool(
        spec=ToolSpec(
            name="write_file",
            description=(
                "Create, overwrite, or append to a UTF-8 text file inside YOUR "
                "user sandbox. The `path` is relative to the sandbox root — "
                "absolute paths and paths escaping the sandbox are rejected. "
                "Modes: 'create' fails if the file exists; 'overwrite' replaces "
                "any existing content (default); 'append' adds to the end. "
                "Use this tool for all file writes; the shell tool is read-only."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path under your sandbox directory",
                    },
                    "content": {
                        "type": "string",
                        "description": "UTF-8 text content to write",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["create", "overwrite", "append"],
                        "description": "Write mode (default: overwrite)",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            source="builtin",
        ),
        executor=execute,
    )
