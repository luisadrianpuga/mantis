from __future__ import annotations

from datetime import datetime
from pathlib import Path


class IdentityManager:
    def __init__(self, root: str = ".mantis/identity") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.persona_path = self.root / "persona.md"
        self.mission_path = self.root / "mission.md"
        self.principles_path = self.root / "principles.md"
        self.journal_path = self.root / "journal.md"

    def load_identity_block(self) -> str:
        persona = self._read(self.persona_path)
        mission = self._read(self.mission_path)
        principles = self._read(self.principles_path)
        journal_tail = self._recent_journal_lines(20)

        return "\n\n".join(
            [
                "IDENTITY:\n" + (persona or "(identity missing)"),
                "MISSION:\n" + (mission or "(mission missing)"),
                "PRINCIPLES:\n" + (principles or "(principles missing)"),
                "RECENT JOURNAL:\n" + (journal_tail or "(journal empty)"),
            ]
        ).strip()

    def append_journal(self, entry: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.journal_path.exists():
            self.journal_path.write_text("# Mantis Journal\n", encoding="utf-8")

        timestamp = datetime.utcnow().isoformat()
        with open(self.journal_path, "a", encoding="utf-8") as handle:
            handle.write(f"\n[{timestamp}] {entry.strip()}\n")

    def get_identity_sections(self, recent_lines: int = 20) -> dict[str, str]:
        return {
            "persona": self._read(self.persona_path),
            "mission": self._read(self.mission_path),
            "principles": self._read(self.principles_path),
            "recent_journal": self._recent_journal_lines(recent_lines),
        }

    def _recent_journal_lines(self, max_lines: int) -> str:
        content = self._read(self.journal_path)
        if not content:
            return ""
        lines = [line for line in content.splitlines() if line.strip()]
        return "\n".join(lines[-max_lines:])

    @staticmethod
    def _read(path: Path) -> str:
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
