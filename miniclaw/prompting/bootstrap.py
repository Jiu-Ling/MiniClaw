from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

BOOTSTRAP_FILE_NAMES: tuple[str, ...] = ("SOUL.md", "USER.md")


class BootstrapFile(BaseModel):
    """Loaded prompt bootstrap file."""

    model_config = ConfigDict(extra="forbid")

    name: str
    path: Path
    content: str


class BootstrapLoader:
    """Load workspace-level bootstrap prompt files in a stable order."""

    def __init__(
        self,
        *,
        workspace: Path | None = None,
        file_names: tuple[str, ...] = BOOTSTRAP_FILE_NAMES,
    ) -> None:
        self.workspace = Path(workspace) if workspace is not None else self._default_workspace()
        self.file_names = file_names

    def load(self) -> list[BootstrapFile]:
        files: list[BootstrapFile] = []
        for file_name in self.file_names:
            file_path = self.workspace / file_name
            if not file_path.is_file():
                continue
            files.append(
                BootstrapFile(
                    name=file_name,
                    path=file_path,
                    content=file_path.read_text(encoding="utf-8").strip(),
                )
            )
        return files

    @staticmethod
    def _default_workspace() -> Path:
        return Path(__file__).resolve().parents[2]
