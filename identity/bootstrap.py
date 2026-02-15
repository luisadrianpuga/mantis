from __future__ import annotations

import socket
from pathlib import Path

from identity.manager import IdentityManager
from providers.detection import has_anthropic, has_ollama, has_openai


def bootstrap_identity(root: str = ".mantis/identity", workspace: str | None = None) -> IdentityManager:
    identity = IdentityManager(root=root)
    templates_dir = Path(__file__).parent / "templates"

    _copy_if_missing(templates_dir / "persona.md", identity.persona_path)
    _copy_if_missing(templates_dir / "mission.md", identity.mission_path)
    _copy_if_missing(templates_dir / "principles.md", identity.principles_path)

    journal_was_missing = not identity.journal_path.exists()
    _copy_if_missing(templates_dir / "journal.md", identity.journal_path)

    if journal_was_missing:
        providers = f"OpenAI={'YES' if has_openai() else 'NO'}, Anthropic={'YES' if has_anthropic() else 'NO'}, Ollama={'YES' if has_ollama() else 'NO'}"
        identity.append_journal(
            f"Bootstrapped identity on host={socket.gethostname()} workspace={workspace or str(Path.cwd())} providers=({providers})"
        )

    return identity


def _copy_if_missing(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
