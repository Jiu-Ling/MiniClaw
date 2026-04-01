from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from miniclaw.skills.contracts import LoadedSkill, SkillSummary


class SkillsLoader:
    """Discover and load skills from workspace and builtin locations."""

    def __init__(
        self,
        *,
        workspace: Path,
        builtin_skills_dir: Path | None = None,
        workspace_skills_dir: Path | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace_skills = (
            Path(workspace_skills_dir) if workspace_skills_dir is not None else self.workspace / "skills"
        )
        self.builtin_skills = Path(builtin_skills_dir) if builtin_skills_dir is not None else None

    def list_skills(self) -> list[SkillSummary]:
        discovered: dict[str, SkillSummary] = {}

        for root, source in (
            (self.builtin_skills, "builtin"),
            (self.workspace_skills, "workspace"),
        ):
            if root is None or not root.exists():
                continue

            for skill_dir in sorted(root.iterdir(), key=lambda item: item.name):
                skill_file = skill_dir / "SKILL.md"
                if not skill_dir.is_dir() or not skill_file.is_file():
                    continue

                metadata = self._read_metadata(skill_file)
                description = metadata.get("description", "").strip() or skill_dir.name
                discovered[skill_dir.name] = SkillSummary(
                    name=skill_dir.name,
                    description=description,
                    path=skill_file,
                    source=source,
                    available=True,
                    metadata=metadata,
                )

        return [discovered[name] for name in sorted(discovered)]

    def load_skill(self, name: str) -> LoadedSkill | None:
        for root in (self.workspace_skills, self.builtin_skills):
            if root is None:
                continue

            skill_file = root / name / "SKILL.md"
            if not skill_file.is_file():
                continue

            return LoadedSkill(
                name=name,
                path=skill_file,
                content=skill_file.read_text(encoding="utf-8"),
                metadata=self._read_metadata(skill_file),
            )

        return None

    def build_active_skills_block(self, active_skill_names: Iterable[str] | None = None) -> str:
        loaded_skills = self._load_discovered_skills()
        active_names = {str(name).strip() for name in (active_skill_names or []) if str(name).strip()}
        active_skills = [
            skill
            for skill in loaded_skills
            if self._is_always_active(skill.metadata) or skill.name in active_names
        ]
        if not active_skills:
            return ""

        sections = [self._format_active_skill(skill) for skill in active_skills]
        return "\n\n".join(["## Active Skills", *sections])

    def build_skill_summary_block(self, active_skill_names: Iterable[str] | None = None) -> str:
        summaries = self.list_skills()
        if not summaries:
            return ""

        active_names = {str(name).strip() for name in (active_skill_names or []) if str(name).strip()}
        lines = []
        for summary in summaries:
            loaded = self.load_skill(summary.name)
            if loaded is not None and (
                self._is_always_active(loaded.metadata) or loaded.name in active_names
            ):
                continue

            description = summary.description.strip() or summary.name
            lines.append(f"- {summary.name}: {description}")

        if not lines:
            return ""

        return "\n".join(["## Skills Summary", *lines])

    def _load_discovered_skills(self) -> list[LoadedSkill]:
        skills: list[LoadedSkill] = []
        for summary in self.list_skills():
            loaded = self.load_skill(summary.name)
            if loaded is not None:
                skills.append(loaded)
        return skills

    @staticmethod
    def _is_always_active(metadata: dict[str, str]) -> bool:
        value = metadata.get("always_active", "")
        return value.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _format_active_skill(skill: LoadedSkill) -> str:
        body = _strip_frontmatter(skill.content).strip()
        return f"### {skill.name}\n{body}" if body else f"### {skill.name}"

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, str]:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}

        try:
            end_index = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
        except StopIteration:
            return {}

        metadata: dict[str, str] = {}
        for raw_line in lines[1:end_index]:
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            cleaned_value = value.strip().strip('"').strip("'")
            metadata[key.strip()] = cleaned_value
        return metadata


def _strip_frontmatter(content: str) -> str:
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return content
    try:
        end_index = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        return content
    return "".join(lines[end_index + 1:])


SkillLoader = SkillsLoader
