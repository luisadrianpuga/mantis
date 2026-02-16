import shutil
import subprocess
from pathlib import Path
from typing import List


def run_tests(_: str = "") -> str:
    command = _detect_test_command()
    if not command:
        return "No supported test command detected (pytest, npm test, make test). EXIT_CODE: 127"

    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except Exception as exc:
        return f"Test execution failed: {exc}. EXIT_CODE: 1"

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    merged = "\n".join(part for part in [stdout, stderr] if part)
    if not merged:
        merged = "(no output)"

    return f"COMMAND: {' '.join(command)}\nEXIT_CODE: {completed.returncode}\nOUTPUT:\n{merged}"


def _detect_test_command() -> List[str] | None:
    cwd = Path(".")
    if shutil.which("pytest") and (cwd.joinpath("tests").exists() or list(cwd.glob("test_*.py"))):
        return ["pytest", "-q"]

    if cwd.joinpath("package.json").exists() and shutil.which("npm"):
        return ["npm", "test", "--", "--runInBand"]

    if cwd.joinpath("Makefile").exists() and shutil.which("make"):
        return ["make", "test"]

    return None
