from __future__ import annotations

from pathlib import Path

from miniclaw.tools.contracts import ToolCall, ToolResult, ToolSpec
from miniclaw.tools.registry import RegisteredTool

MAX_READ_FILE_BYTES = 32 * 1024  # 32 KB


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
