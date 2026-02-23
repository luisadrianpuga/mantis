#!/usr/bin/env python3
"""
Three-rule emergent agent
ATTEND -> ASSOCIATE -> ACT -> (side effects loop back)
Memory: ChromaDB (vector) + SQLite FTS5 (keyword) + MEMORY.md (markdown)
Safety: Lane Queue (serial FIFO per source)
"""
import hashlib
import json
import os
import re
import select
import shlex
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
import chromadb
import httpx
from dotenv import load_dotenv
from docs.assets.start_up_logo import start_up_logo as START_UP_LOGO

# -- Config -------------------------------------------------------------------
load_dotenv()


def _env_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


LLM_BASE = os.getenv("LLM_BASE", "http://localhost:8001/v1")
MODEL = os.getenv("MODEL", "Qwen2.5-14B-Instruct-Q4_K_M.gguf")
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", ".agent/memory"))
SOUL_PATH = Path(os.getenv("SOUL_PATH", "SOUL.md"))
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hash").lower()
AUTONOMOUS_INTERVAL_SEC = int(os.getenv("AUTONOMOUS_INTERVAL_SEC", "900"))
WATCH_PATH = os.getenv("WATCH_PATH", ".")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
SHELL_LOG = Path(os.getenv("SHELL_LOG", ".agent/shell.log"))
MAX_LLM_TIMEOUT = int(os.getenv("MAX_LLM_TIMEOUT", "120"))
MAX_AUTO_REPAIR_ATTEMPTS = int(os.getenv("MAX_AUTO_REPAIR_ATTEMPTS", "1"))
MIN_FAILURES_BEFORE_SKILL_UPDATE = int(
    os.getenv("MIN_FAILURES_BEFORE_SKILL_UPDATE", "3")
)
SKILL_UPDATE_WINDOW = int(os.getenv("SKILL_UPDATE_WINDOW", "50"))
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "24000"))
MAX_SYSTEM_CHARS = int(os.getenv("MAX_SYSTEM_CHARS", "12000"))
MAX_HISTORY_MSG_CHARS = int(os.getenv("MAX_HISTORY_MSG_CHARS", "3000"))
MAX_USER_INPUT_CHARS = int(os.getenv("MAX_USER_INPUT_CHARS", "4000"))
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
DISCORD_CHANNEL_ID = _env_int("DISCORD_CHANNEL_ID", 0)
DISCORD_ACTIVITY_FEED = _env_bool("DISCORD_ACTIVITY_FEED", True)

DB_PATH = MEMORY_DIR / "fts.db"
MEMORY_MD = Path(".agent/MEMORY.md")
SKILLS_DIR = Path(".agent/skills")
TASKS_PATH = Path("tasks.md")
EMBED_DIM = 384

print(START_UP_LOGO)

# -- Boot ---------------------------------------------------------------------
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"


def _ensure_shell_journal_hooked() -> None:
    """
    Auto-install PROMPT_COMMAND hook into ~/.bashrc for shared shell awareness.
    Idempotent; safe to call on every boot.
    """
    bashrc = Path.home() / ".bashrc"
    marker = "# mantis shell journal"
    journal_path = str(SHELL_LOG.expanduser().resolve())
    hook = (
        f"\n{marker}\n"
        "_mantis_log_command() {\n"
        "    local last_cmd\n"
        "    last_cmd=$(history 1 | sed \"s/^[ ]*[0-9]*[ ]*//\" 2>/dev/null)\n"
        "    if [ -n \"$last_cmd\" ]; then\n"
        f"        echo \"$(date -u +%FT%T) [you]: $last_cmd\" >> {journal_path} 2>/dev/null\n"
        "    fi\n"
        "}\n"
        "export PROMPT_COMMAND='_mantis_log_command'\n"
    )
    try:
        existing = bashrc.read_text(encoding="utf-8") if bashrc.exists() else ""
        if marker not in existing:
            with open(bashrc, "a", encoding="utf-8") as f:
                f.write(hook)
            print("[setup] shell journal hooked into ~/.bashrc")
            print("[setup] run: source ~/.bashrc  (or open a new terminal)")
    except Exception as e:
        print(f"[setup] could not hook ~/.bashrc: {e} — add PROMPT_COMMAND manually")


_ensure_shell_journal_hooked()


def _ensure_computer_use_deps() -> None:
    """Auto-install scrot and xdotool if not present."""
    import shutil

    missing = [tool for tool in ["scrot", "xdotool"] if not shutil.which(tool)]
    if not missing:
        return

    print(f"[setup] installing computer use deps: {', '.join(missing)}")
    try:
        result = subprocess.run(
            f"sudo apt install -y {' '.join(missing)}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            print("[setup] computer use deps installed")
        else:
            print(f"[setup] install failed: {result.stderr[:200]}")
    except Exception as e:
        print(
            "[setup] could not install deps: "
            f"{e} — run: sudo apt install -y scrot xdotool"
        )


_ensure_computer_use_deps()


def _ensure_playwright_firefox() -> None:
    """Ensure Playwright and Firefox are installed."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            browser.close()
    except Exception:
        print("[setup] installing playwright firefox...")
        try:
            subprocess.run(
                "playwright install firefox",
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            print("[setup] playwright firefox ready")
        except Exception as e:
            print(f"[setup] playwright install failed: {e}")


_ensure_playwright_firefox()


def _ensure_tasks_file() -> None:
    """Create a default tasks.md file on first boot."""
    if TASKS_PATH.exists():
        return
    default = """# Mantis Tasks

- name: morning briefing
  schedule: daily 8am
  prompt: Run the now command, check todo_list.txt if it exists, and give a brief morning summary.

- name: news digest
  schedule: daily 9am
  prompt: Fetch top headlines by running: curl -sL https://feeds.apnews.com/rss/apf-topnews | grep -oP '(?<=<title>)[^<]+' | head -6

- name: disk check
  schedule: daily 11pm
  prompt: Check disk usage with df -h. If any partition is over 80% full, alert the user clearly.

- name: soul review
  schedule: weekly sunday 10am
  prompt: Review this week's memory. What did you learn about the user's preferences, working style, or goals? Propose 1-3 short additions to SOUL.md as a numbered list. Do not write anything yet - wait for the user to say 'approve soul'.

- name: learn review
  schedule: weekly sunday 11am
  prompt: You are reviewing your own command execution history to improve your skills.
    Step 1 - Run: COMMAND: echo "reviewing outcomes"
    Step 2 - The system will show you recent_outcome_log in context. Analyze it.
    Step 3 - For each skill that had failures or empty results, read the skill file.
    Step 4 - Rewrite the skill file with WRITE: to fix the failing patterns.
    Step 5 - Write one line to MEMORY.md summarizing what changed and why.
    Focus on news.md first - it has the most execution history.
    Only update skills where you have clear evidence of what works better.
    Do not remove working commands. Add alternatives above failing ones.
"""
    try:
        TASKS_PATH.write_text(default, encoding="utf-8")
        print("[tasks] created default tasks.md")
    except Exception as e:
        print(f"[tasks] could not create tasks.md: {e}")


_ensure_tasks_file()
db = chromadb.PersistentClient(str(MEMORY_DIR))
collection = db.get_or_create_collection("events")

_encoder = None
_encoder_lock = threading.Lock()


def _hash_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    out = []
    for i in range(dim):
        digest = hashlib.sha256(f"{i}:{text}".encode("utf-8")).digest()
        n = int.from_bytes(digest[:4], "big", signed=False)
        out.append((n / 2_147_483_647.5) - 1.0)
    return out


def _llm_embed(text: str) -> list[float] | None:
    try:
        r = httpx.post(
            f"{LLM_BASE}/embeddings",
            json={"model": MODEL, "input": text},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            emb = data.get("data", [{}])[0].get("embedding")
            if isinstance(emb, list) and len(emb) > 0:
                return emb
    except Exception:
        pass
    return None


_llm_embed_available: bool | None = None
_llm_embed_lock = threading.Lock()


def embed_text(text: str) -> list[float]:
    global _llm_embed_available

    if EMBEDDING_BACKEND == "sentence-transformers":
        if _encoder is None:
            with _encoder_lock:
                from sentence_transformers import SentenceTransformer
                globals()["_encoder"] = SentenceTransformer("all-MiniLM-L6-v2")
        return _encoder.encode(text).tolist()

    if EMBEDDING_BACKEND == "llm" or EMBEDDING_BACKEND == "hash":
        # Try LLM embeddings first, fall back to hash
        with _llm_embed_lock:
            if _llm_embed_available is None:
                probe = _llm_embed("test")
                _llm_embed_available = probe is not None
                if _llm_embed_available:
                    print("[embeddings] using llm endpoint")
                else:
                    print("[embeddings] llm endpoint unavailable, using hash fallback")

        if _llm_embed_available:
            result = _llm_embed(text)
            if result is not None:
                return result

    return _hash_embed(text)


# -- SQLite FTS5 setup --------------------------------------------------------
def _get_fts_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memories
        USING fts5(text, source, ts)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS command_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            cmd TEXT NOT NULL,
            result TEXT NOT NULL,
            outcome TEXT NOT NULL,
            source TEXT NOT NULL,
            skill TEXT,
            duration_ms INTEGER
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_command_outcomes_ts ON command_outcomes(ts)"
    )
    conn.commit()
    return conn


fts_conn = _get_fts_conn()
fts_lock = threading.Lock()


def fts_store(text: str, source: str, ts: str) -> None:
    with fts_lock:
        fts_conn.execute(
            "INSERT INTO memories(text, source, ts) VALUES (?, ?, ?)",
            (text, source, ts),
        )
        fts_conn.commit()


def fts_search(query: str, k: int = TOP_K) -> list[str]:
    with fts_lock:
        rows = fts_conn.execute(
            "SELECT text FROM memories WHERE memories MATCH ? ORDER BY rank LIMIT ?",
            (query, k),
        ).fetchall()
    return [r[0] for r in rows]


def classify_outcome(result: str) -> str:
    text = (result or "").strip()
    lower = text.lower()
    if "timed out" in lower:
        return "timeout"
    if text in {"", "(no output)"}:
        return "empty"
    failure_markers = ("error", "failed", "traceback", "not found", "permission denied")
    if any(marker in lower for marker in failure_markers):
        return "fail"
    return "success"


def store_command_outcome(
    cmd: str,
    result: str,
    source: str,
    *,
    skill: str = "",
    duration_ms: int | None = None,
    ts: str | None = None,
) -> str:
    outcome = classify_outcome(result)
    when = ts or datetime.utcnow().isoformat()
    with fts_lock:
        fts_conn.execute(
            """
            INSERT INTO command_outcomes(ts, cmd, result, outcome, source, skill, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (when, cmd[:4000], result[:4000], outcome, source, skill, duration_ms),
        )
        fts_conn.commit()
    return outcome


def recent_command_outcomes(limit: int = 50) -> list[dict]:
    with fts_lock:
        rows = fts_conn.execute(
            """
            SELECT ts, cmd, outcome, source, COALESCE(skill, ''), COALESCE(duration_ms, 0)
            FROM command_outcomes
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "ts": r[0],
            "cmd": r[1],
            "outcome": r[2],
            "source": r[3],
            "skill": r[4],
            "duration_ms": r[5],
        }
        for r in rows
    ]


def outcome_digest(limit: int = 25) -> str:
    rows = recent_command_outcomes(limit)
    if not rows:
        return "No command outcomes recorded yet."
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["outcome"]] += 1
    latest = rows[0]
    parts = [
        f"recent_outcomes={len(rows)}",
        f"success={counts.get('success', 0)}",
        f"empty={counts.get('empty', 0)}",
        f"fail={counts.get('fail', 0)}",
        f"timeout={counts.get('timeout', 0)}",
        f"latest={latest['outcome']}:{_command_preview(latest['cmd'], 60)}",
    ]
    return " | ".join(parts)


def recent_outcome_lines(limit: int = 12) -> str:
    rows = recent_command_outcomes(limit)
    if not rows:
        return "(none)"
    lines = []
    for row in rows:
        lines.append(
            f"- {row['ts']} [{row['outcome']}] ({row['source']}) {_command_preview(row['cmd'], 90)}"
        )
    return "\n".join(lines)


def _cmd_failure_count(cmd: str, window: int = SKILL_UPDATE_WINDOW) -> int:
    """
    Count how many times a command failed/empty/timed out in recent outcomes.
    Match on normalized first 200 chars to absorb minor whitespace variance.
    """
    key = " ".join((cmd or "").strip().split())[:200]
    if not key:
        return 0
    rows = recent_command_outcomes(window)
    count = 0
    for row in rows:
        row_key = " ".join((row.get("cmd") or "").strip().split())[:200]
        if row_key == key and row.get("outcome") in {"fail", "empty", "timeout"}:
            count += 1
    return count


def _skill_update_eligible(cmd: str) -> tuple[bool, str]:
    """
    A command must fail repeatedly before skill removals/deprecations are allowed.
    Set MIN_FAILURES_BEFORE_SKILL_UPDATE=0 to disable this gate.
    """
    if MIN_FAILURES_BEFORE_SKILL_UPDATE <= 0:
        return True, "gate disabled by config"
    count = _cmd_failure_count(cmd, SKILL_UPDATE_WINDOW)
    if count >= MIN_FAILURES_BEFORE_SKILL_UPDATE:
        return True, f"confirmed: {count} failures in last {SKILL_UPDATE_WINDOW} outcomes"
    return (
        False,
        "insufficient evidence: only "
        f"{count} failure(s), need {MIN_FAILURES_BEFORE_SKILL_UPDATE}",
    )


def _is_skill_write(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    return normalized.startswith(".agent/skills/") or "/skills/" in f"/{normalized}/"


def _is_todo_write(path: str) -> bool:
    normalized = path.replace("\\", "/").strip().lower()
    return normalized.endswith("/todo_list.txt") or normalized == "todo_list.txt"


def _looks_shell_like_task(text: str) -> bool:
    lower = (text or "").strip().lower()
    if not lower:
        return True
    if any(marker in lower for marker in ("command:", "read:", "write:", "fetch:", "skill:")):
        return True
    shell_markers = (
        "&&",
        "||",
        ";",
        "|",
        "`",
        "$(",
        "sudo ",
        " apt ",
        " pip ",
        " python ",
        " bash",
        " sh ",
        " echo ",
        " curl ",
        " wget ",
        " rm ",
        " chmod ",
        " chown ",
    )
    return any(marker in f" {lower} " for marker in shell_markers)


def _is_valid_todo_content(content: str) -> tuple[bool, str]:
    """
    Require todo_list.txt to be simple task text (bullets/checklist/numbered lines),
    not shell-like command fragments.
    """
    raw_lines = content.splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    if not lines:
        return False, "todo_list.txt cannot be empty"

    allowed_prefixes = ("- ", "* ", "[ ] ", "[x] ", "[X] ")
    for line in lines:
        task_text = line
        if line.startswith(allowed_prefixes):
            task_text = line.split(" ", 1)[1].strip() if " " in line else ""
        elif re.match(r"^\d+\.\s+", line):
            task_text = re.sub(r"^\d+\.\s+", "", line).strip()
        else:
            return False, f"invalid task format: `{line[:60]}`"

        if len(task_text) < 3:
            return False, f"task too short: `{line[:60]}`"
        if _looks_shell_like_task(task_text):
            return False, f"shell-like task rejected: `{line[:60]}`"

    return True, "ok"


def _guarded_todo_write(wpath: str, wcontent: str, reply: str) -> tuple[str, bool]:
    """
    Block malformed todo_list.txt writes so command-like content does not corrupt tasks.
    Returns (updated_reply, was_blocked).
    """
    valid, reason = _is_valid_todo_content(wcontent)
    if valid:
        return reply, False

    ui_print(f"\n  [todo write blocked]\n  {reason}\n")
    discord_event("warn", f"todo write blocked — {reason[:180]}")
    attend(
        f"todo_list.txt write blocked: {reason}. "
        "Write todo_list.txt as plain task lines (bullets/checklist/numbered), "
        "not commands.",
        source="tool",
    )
    updated_reply = reply[: reply.index("WRITE:")].strip() or "(write blocked)"
    return updated_reply, True


def _extract_deprecated_commands(old_content: str, new_content: str) -> list[str]:
    """
    Find COMMAND lines present in old content but removed from new content.
    """
    old_commands = [c.strip() for c in re.findall(r"COMMAND:\s*(.+)", old_content)]
    new_commands = [c.strip() for c in re.findall(r"COMMAND:\s*(.+)", new_content)]
    new_set = {" ".join(c.split())[:200] for c in new_commands if c}
    removed = []
    for cmd in old_commands:
        key = " ".join(cmd.split())[:200]
        if key and key not in new_set:
            removed.append(cmd)
    return removed


def _guarded_skill_write(wpath: str, wcontent: str, reply: str) -> tuple[str, bool]:
    """
    Block skill writes that remove commands without enough failure evidence.
    Returns (updated_reply, was_blocked).
    """
    existing = read_file(wpath)
    if existing.startswith("(read error"):
        return reply, False

    removed_cmds = _extract_deprecated_commands(existing, wcontent)
    if not removed_cmds:
        return reply, False

    blocked: list[tuple[str, str]] = []
    for cmd in removed_cmds:
        eligible, reason = _skill_update_eligible(cmd)
        if not eligible:
            blocked.append((cmd, reason))

    if not blocked:
        return reply, False

    msg_parts = [f"`{_command_preview(c, 60)}`: {r}" for c, r in blocked]
    block_msg = "; ".join(msg_parts)
    ui_print(f"\n  [skill write blocked]\n  {block_msg}\n")
    discord_event("warn", f"skill write blocked — {block_msg[:200]}")
    attend(
        f"skill write to {wpath} blocked. Insufficient failure evidence for: "
        f"{block_msg}. Need {MIN_FAILURES_BEFORE_SKILL_UPDATE} failures each. "
        "Add alternatives above existing commands instead of removing them.",
        source="tool",
    )
    updated_reply = reply[: reply.index("WRITE:")].strip() or "(write blocked)"
    return updated_reply, True


# -- MEMORY.md ----------------------------------------------------------------
def md_append(text: str, source: str) -> None:
    ts = datetime.utcnow().isoformat()
    line = f"\n- [{ts}] ({source}) {text[:300]}"
    with open(MEMORY_MD, "a", encoding="utf-8") as f:
        f.write(line)


def md_tail(n: int = 6) -> list[str]:
    if not MEMORY_MD.exists():
        return []
    lines = MEMORY_MD.read_text(encoding="utf-8").strip().splitlines()
    return [l.strip("- ").strip() for l in lines[-n:] if l.strip()]


def load_tasks() -> list[dict]:
    """Parse tasks.md into a list of task dicts."""
    if not TASKS_PATH.exists():
        return []
    try:
        text = TASKS_PATH.read_text(encoding="utf-8")
    except Exception:
        return []

    tasks = []
    current: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("- name:"):
            if current.get("name") and current.get("schedule") and current.get("prompt"):
                tasks.append(current)
            current = {"name": line[7:].strip()}
        elif line.startswith("schedule:"):
            current["schedule"] = line[9:].strip()
        elif line.startswith("prompt:"):
            current["prompt"] = line[7:].strip()
        elif current.get("prompt") and line and not line.startswith("-"):
            current["prompt"] += f" {line}"
    if current.get("name") and current.get("schedule") and current.get("prompt"):
        tasks.append(current)
    return tasks


def _next_fire(schedule: str, now: datetime) -> datetime | None:
    """Parse natural language schedule and return next fire datetime."""
    s = schedule.lower().strip()

    # daily HH:MM or daily Xam/pm
    m = re.search(r"daily\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        meridiem = m.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    # weekly <weekday> HH:MM or Xam/pm
    days = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for day_name, day_num in days.items():
        if day_name in s:
            m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s)
            if not m:
                return None
            hour = int(m.group(1))
            minute = int(m.group(2) or 0)
            meridiem = m.group(3)
            if meridiem == "pm" and hour != 12:
                hour += 12
            elif meridiem == "am" and hour == 12:
                hour = 0

            days_ahead = (day_num - now.weekday()) % 7
            if days_ahead == 0:
                candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if candidate > now:
                    return candidate
                days_ahead = 7
            candidate = (now + timedelta(days=days_ahead)).replace(
                hour=hour,
                minute=minute,
                second=0,
                microsecond=0,
            )
            return candidate
    return None


# -- Conversation history -----------------------------------------------------
_chat_history: list[dict] = []
_history_lock = threading.Lock()


def history_append(role: str, content: str, source: str = "user") -> None:
    # Track user, autonomous, and tool exchanges; ignore low-signal noise lanes.
    if source not in {"user", "autonomous", "tool"}:
        return
    with _history_lock:
        _chat_history.append({"role": role, "content": content})
        if len(_chat_history) > MAX_HISTORY * 2:
            _chat_history[:] = _chat_history[-MAX_HISTORY * 2:]


def history_snapshot() -> list[dict]:
    with _history_lock:
        return list(_chat_history[-MAX_HISTORY:])


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 18] + "\n...[truncated]"


def _message_chars(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += len(m.get("content", "") or "")
    return total


def _compact_messages_for_context(
    system_text: str,
    history: list[dict],
    user_text: str,
    *,
    max_total_chars: int = MAX_PROMPT_CHARS,
) -> list[dict]:
    system_trimmed = _truncate_text(system_text, MAX_SYSTEM_CHARS)
    user_trimmed = _truncate_text(user_text, MAX_USER_INPUT_CHARS)

    compact_history: list[dict] = []
    for msg in history[-MAX_HISTORY:]:
        role = msg.get("role", "user")
        content = _truncate_text(msg.get("content", ""), MAX_HISTORY_MSG_CHARS)
        compact_history.append({"role": role, "content": content})

    messages = [{"role": "system", "content": system_trimmed}]
    messages.extend(compact_history)
    messages.append({"role": "user", "content": user_trimmed})

    while len(messages) > 2 and _message_chars(messages) > max_total_chars:
        # Drop oldest non-system history first.
        del messages[1]

    if _message_chars(messages) > max_total_chars:
        available_for_system = max(1000, max_total_chars - len(user_trimmed) - 200)
        messages[0]["content"] = _truncate_text(messages[0]["content"], available_for_system)

    return messages


def _is_context_overflow_error(err: Exception) -> bool:
    text = str(err).lower()
    return ("exceeds the available context size" in text) or ("maximum context length" in text)


# -- Lane Queue ---------------------------------------------------------------
class LaneQueue:
    """One queue per lane (source). Each lane processes serially."""

    def __init__(self):
        self._lanes: dict[str, Queue] = defaultdict(Queue)
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    def put(self, event: dict):
        lane = event.get("source", "user")
        self._lanes[lane].put(event)

    def drain(self) -> list[dict]:
        events = []
        for q in self._lanes.values():
            while True:
                try:
                    events.append(q.get_nowait())
                except Empty:
                    break
        return events

    def lock(self, source: str) -> threading.Lock:
        return self._locks[source]


lane_queue = LaneQueue()
_event_loop_stop = threading.Event()
_watcher_started = False
_watcher_lock = threading.Lock()
_last_seen_fs: dict[str, float] = {}
_fs_lock = threading.Lock()
_FS_DEBOUNCE_SEC = 10.0
_shell_journal_lock = threading.Lock()
_last_shell_journal_line = ""
_console_lock = threading.Lock()
_input_waiting = threading.Event()
_discord_client = None
_discord_loop = None
_discord_ready = threading.Event()
_discord_channel = None
_task_fire_times: dict[str, datetime] = {}
_task_lock = threading.Lock()
_recent_inputs: list[str] = []
_recent_inputs_lock = threading.Lock()
_RECENT_INPUT_WINDOW = 5
_last_event_loop_error_msg = ""
_last_event_loop_error_ts = 0.0
_suppressed_event_loop_errors = 0
_META_REPLIES = {
    "(running...)",
    "(ran command)",
    "(read file)",
    "(searched)",
    "(fetched)",
    "(loaded skill)",
    "(clicked)",
    "(typed)",
    "(took screenshot)",
    "(wrote file)",
}
_DISCORD_ICONS = {
    "run": "⚡",
    "done": "✅",
    "error": "❌",
    "read": "📋",
    "write": "💾",
    "search": "🔍",
    "fetch": "🌐",
    "skill": "🧠",
    "task": "🕐",
    "shell": "🖥️",
    "info": "ℹ️",
    "warn": "⚠️",
}
_PHANTOM_PHRASES = (
    "has been updated to mark",
    "file has been updated",
    "tasks as completed",
    "have been marked as complete",
)
PRIORITY_SOURCES = {"user", "discord"}
BACKGROUND_SOURCES = {"autonomous", "filesystem", "shell_journal", "learning"}
_MEMORY_NOISE_PATTERNS = (
    "since inception",
    "since you last ran this command",
    "moon phase",
    "day progress:",
    "weather in severn",
    "load average:",
    "cpu temp:",
    "waxing",
    "waning",
)
_FILLER_PHRASES = (
    "is there anything",
    "would you like",
    "let me know",
    "please let me know",
    "feel free to ask",
    "if you need",
    "if you have any",
    "how can i assist",
    "how can i help",
    "i'm here to help",
    "i'm ready to",
    "shall i proceed",
    "do you want me to",
    "would you like me to",
    "anything else",
    "further assistance",
    "further details",
)


def ui_print(message: str = "", redraw_prompt: bool = True) -> None:
    with _console_lock:
        if message:
            print(message, flush=True)
        else:
            print("", flush=True)
        if redraw_prompt and _input_waiting.is_set():
            print("you: ", end="", flush=True)


def _start_discord() -> None:
    """Start Discord client and route channel messages into the event bus."""
    global _discord_client, _discord_loop, _discord_channel

    if not DISCORD_TOKEN or DISCORD_CHANNEL_ID <= 0:
        return

    try:
        import asyncio
        import discord
    except Exception as e:
        ui_print(f"[discord] unavailable: {e}")
        return

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        global _discord_client, _discord_loop, _discord_channel
        _discord_client = client
        _discord_loop = asyncio.get_running_loop()
        _discord_channel = client.get_channel(DISCORD_CHANNEL_ID)
        if _discord_channel is None:
            try:
                _discord_channel = await client.fetch_channel(DISCORD_CHANNEL_ID)
            except Exception:
                _discord_channel = None
        _discord_ready.set()
        ui_print(f"[discord] connected as {client.user}")
        threading.Thread(target=_discord_boot_message, daemon=True).start()

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        if message.channel.id != DISCORD_CHANNEL_ID:
            return
        if not message.content:
            return
        if message.content.lower().startswith("approve soul"):
            attend(
                "The user approved the soul update proposal. "
                "Write the approved additions to SOUL.md now using WRITE:. "
                "Append to the existing content, do not replace it.",
                source="discord",
            )
            return
        attend(message.content, source="discord")

    try:
        asyncio.run(client.start(DISCORD_TOKEN))
    except Exception as e:
        ui_print(f"[discord] stopped: {e}")


def discord_post(message: str) -> None:
    """Post a message back to the configured Discord channel."""
    if not message or not _discord_client or not _discord_loop:
        return
    if not _discord_ready.is_set() or DISCORD_CHANNEL_ID <= 0:
        return

    import asyncio

    async def _send():
        global _discord_channel
        channel = _discord_channel or _discord_client.get_channel(DISCORD_CHANNEL_ID)
        if channel is None:
            try:
                channel = await _discord_client.fetch_channel(DISCORD_CHANNEL_ID)
                _discord_channel = channel
            except Exception as e:
                ui_print(f"[discord] send failed (channel): {e}")
                return
        if channel is None:
            return
        for i in range(0, len(message), 1900):
            try:
                await channel.send(message[i : i + 1900])
            except Exception as e:
                ui_print(f"[discord] send failed: {e}")
                return

    try:
        fut = asyncio.run_coroutine_threadsafe(_send(), _discord_loop)
        fut.add_done_callback(lambda f: f.exception())
    except Exception as e:
        ui_print(f"[discord] scheduling failed: {e}")


def discord_event(kind: str, message: str) -> None:
    """Post a lightweight activity event to Discord."""
    if not DISCORD_ACTIVITY_FEED or not message:
        return
    icon = _DISCORD_ICONS.get(kind, "•")
    discord_post(f"{icon} {message}")


def _discord_boot_message() -> None:
    """Post a one-time status summary after Discord connects."""
    if not _discord_ready.wait(timeout=10):
        return
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    tasks = load_tasks()
    task_names = ", ".join(t.get("name", "") for t in tasks if t.get("name")) or "none"
    skills = sorted(SKILLS_DIR.glob("*.md")) if SKILLS_DIR.exists() else []
    skill_names = ", ".join(s.stem for s in skills) if skills else "none"
    msg = (
        f"🟢 **mantis online** - {now_utc}\n"
        f"tasks: {task_names}\n"
        f"skills: {skill_names}\n"
        f"model: {MODEL}"
    )
    discord_post(msg)


# -- Soul ---------------------------------------------------------------------
def load_soul() -> str:
    if SOUL_PATH.exists():
        return SOUL_PATH.read_text(encoding="utf-8").strip()
    return "You are a sharp, minimal, autonomous agent with memory and terminal access."


# -- Tools --------------------------------------------------------------------
ASYNC_COMMAND_PREFIXES = [
    "pip install",
    "playwright install",
    "npm install",
    "apt install",
    "wget",
    "curl",
    "git clone",
]

INTERACTIVE_COMMANDS = [
    "htop",
    "top",
    "vim",
    "vi",
    "nano",
    "less",
    "more",
    "man",
    "watch",
    "tail -f",
    "journalctl",
    "xdotool selectwindow",
    "xdotool behave",
]

SAFE_ALTERNATIVES = {
    "htop": "cat /proc/loadavg && free -h && df -h",
    "top": "cat /proc/loadavg && free -h && df -h",
}


def _log_to_shell_journal(cmd: str, result: str, actor: str = "mantis") -> None:
    ts = datetime.utcnow().isoformat()
    try:
        SHELL_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SHELL_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts} [{actor}]: {cmd}\n")
            f.write(f"{ts} [result]: {result[:500]}\n")
    except Exception:
        pass


def _is_shell_log_path(path: str) -> bool:
    try:
        return Path(path).resolve() == SHELL_LOG.expanduser().resolve()
    except Exception:
        return path.replace("\\", "/").endswith("shell.log")


def _is_shell_journal_noise(cmd: str) -> bool:
    """Ignore shell setup/meta commands from shared journal events."""
    c = (cmd or "").strip().lower()
    if not c:
        return True

    noise_prefixes = (
        "export term=",
        "export ps1=",
        "export prompt_command=",
        "source ~/.bashrc",
        ". ~/.bashrc",
        "_mantis_log_command",
        "history 1",
    )
    if any(c.startswith(prefix) for prefix in noise_prefixes):
        return True

    if "__mantis_done__" in c or "mantis_done" in c:
        return True

    # ignore raw function-definition fragments from .bashrc hooks
    if c in {"{", "}", "fi", "then", "do", "done"}:
        return True

    return False


def _strip_ansi(text: str) -> str:
    cleaned = text
    cleaned = re.sub(r"\x1b\[[0-9;]*[mGKHF]", "", cleaned)
    cleaned = re.sub(r"\x1b\][^\x07]*\x07", "", cleaned)
    return cleaned.strip()


def _make_noninteractive(cmd: str) -> str:
    """Prepend DEBIAN_FRONTEND for apt install/upgrade commands."""
    stripped = (cmd or "").strip()
    if not stripped:
        return cmd
    if "debian_frontend=" in stripped.lower():
        return cmd
    if re.match(r"^(sudo\s+)?apt(-get)?\s+(install|upgrade|dist-upgrade)\b", stripped):
        return (
            "DEBIAN_FRONTEND=noninteractive "
            "DEBCONF_NONINTERACTIVE_SEEN=true "
            f"{stripped}"
        )
    return cmd


def run_script(cmd: str, timeout: int = 30) -> str:
    """
    Write command text to a temp bash script and execute it atomically.
    Handles multi-line commands, subshells, awk, heredocs, and pipelines.
    """
    cmd = _make_noninteractive(cmd)
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sh",
            delete=False,
            dir="/tmp",
            prefix="mantis_",
        ) as f:
            f.write("#!/bin/bash\n")
            f.write("set -o pipefail\n")
            f.write(cmd)
            if not cmd.endswith("\n"):
                f.write("\n")
            script_path = f.name

        result = subprocess.run(
            ["bash", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout + result.stderr).strip()
        output = _strip_ansi(output)
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"
    except Exception as e:
        return f"(script error: {e})"
    finally:
        if script_path:
            try:
                Path(script_path).unlink(missing_ok=True)
            except Exception:
                pass


def _record_outcome_and_learn(
    cmd: str,
    result: str,
    source: str,
    *,
    duration_ms: int | None = None,
    skill: str = "",
) -> str:
    outcome = store_command_outcome(
        cmd,
        result,
        source,
        skill=skill,
        duration_ms=duration_ms,
    )
    preview = _command_preview(cmd, 90)
    attend(
        (
            f"command outcome [{outcome}] source={source} "
            f"skill={skill or '-'} cmd={preview} result={result[:220]}"
        ),
        source="learning",
    )
    return outcome


def run_script_async(
    cmd: str,
    trigger_source: str = "tool",
    *,
    skill: str = "",
) -> None:
    """Run script asynchronously and feed result back as a shell event."""

    def _run():
        started = time.time()
        result = run_script(cmd, timeout=60)
        elapsed = round(time.time() - started, 1)
        duration_ms = int((time.time() - started) * 1000)
        preview = _command_preview(cmd, 120)
        _log_to_shell_journal(cmd, result, actor="mantis")
        outcome = _record_outcome_and_learn(
            cmd,
            result,
            trigger_source,
            duration_ms=duration_ms,
            skill=skill,
        )
        attend(f"shell result for `{cmd[:80]}`:\n{result}", source="shell")
        if outcome in {"timeout", "fail"}:
            discord_event("error", f"`{preview}` - {result[:140]}")
        else:
            discord_event("shell", f"`{preview}` done ({elapsed}s)\n{result[:300]}")

    threading.Thread(target=_run, daemon=True).start()


def read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return f"(read error: {e})"


def write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {path}"
    except Exception as e:
        return f"(write error: {e})"


def _flatten_command(cmd: str) -> str:
    """
    Convert multi-line command text into a single logical line.
    This avoids partial execution when models emit formatted shell blocks.
    """
    lines = cmd.strip().splitlines()
    if len(lines) <= 1:
        return cmd.strip()

    parts = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts.append(stripped)
    return " ".join(parts).strip()


def _normalize_command(cmd: str) -> str:
    """Normalize model-emitted command text before validation/execution."""
    cleaned = cmd or ""
    cleaned = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", cleaned)
    cleaned = cleaned.strip()

    # Strip optional fenced wrappers if model leaked markdown.
    cleaned = re.sub(r"^```(?:bash|sh|shell)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    if cleaned.startswith("$"):
        cleaned = cleaned[1:].strip()

    # If stray closing parens leaked from surrounding prose, trim extras.
    while cleaned.endswith(")") and cleaned.count(")") > cleaned.count("("):
        cleaned = cleaned[:-1].rstrip()

    return cleaned.strip()


def parse_command(reply: str) -> str | None:
    tool_markers = "READ|WRITE|SCREENSHOT|CLICK|TYPE|SEARCH|FETCH|SKILL"
    m = re.search(
        rf"COMMAND:\s*\n?(.*?)(?=\n(?:{tool_markers}):|\Z)",
        reply,
        re.DOTALL,
    )
    if not m:
        return None
    cmd = _normalize_command(m.group(1).strip())
    if not cmd:
        return None
    if "\n" in cmd:
        cmd = _flatten_command(cmd)
    if cmd == "(":
        return None
    return cmd


def parse_command_fallback(reply: str) -> str | None:
    """
    Fallback parser: extract first fenced shell block when COMMAND: is missing.
    """
    m = re.search(r"```(?:bash|sh|shell)?\s*\n(.+?)\n```", reply, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    block = m.group(1).strip()
    if not block:
        return None
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None
    if lines[0].lstrip().startswith("$"):
        lines[0] = lines[0].lstrip()[1:].strip()
    cmd = _normalize_command("\n".join(lines).strip())
    if "\n" in cmd:
        cmd = _flatten_command(cmd)
    if cmd == "(":
        return None
    if len(cmd) < 4:
        return None
    return cmd or None


def _is_safe_command(cmd: str) -> bool:
    """Basic safety check for script execution."""
    compact = cmd.strip()
    if len(compact) < 2:
        return False
    if not re.search(r"[A-Za-z0-9]", compact):
        return False
    if ":(){ :|:& };:" in compact:
        return False
    if re.search(r"(^|[;\s])rm\s+-rf\s+/\s*($|[;\s])", compact):
        return False
    return True


def _is_phantom_completion(text: str) -> bool:
    lower = (text or "").lower()
    return any(phrase in lower for phrase in _PHANTOM_PHRASES)


def _is_duplicate_input(text: str) -> bool:
    key = (text or "").strip()[:200]
    if not key:
        return False
    with _recent_inputs_lock:
        if key in _recent_inputs:
            return True
        _recent_inputs.append(key)
        if len(_recent_inputs) > _RECENT_INPUT_WINDOW:
            _recent_inputs.pop(0)
    return False


def _is_memory_noise(text: str) -> bool:
    lower = (text or "").lower()
    return any(pattern in lower for pattern in _MEMORY_NOISE_PATTERNS)


def _is_worth_posting(reply: str, source: str) -> bool:
    """
    Return True only if this reply has enough value to post to Discord.
    Direct Discord responses always post. Background noise never does.
    """
    # Always respond to Luis directly.
    if source == "discord":
        return True

    # Meta replies are tool scaffolding, not content.
    if reply in _META_REPLIES:
        return False

    # Too short to be meaningful.
    if len(reply.strip()) < 30:
        return False

    # Pure filler - question fishing or offers to help.
    lower = reply.lower().strip()
    filler_count = sum(1 for phrase in _FILLER_PHRASES if phrase in lower)
    if filler_count >= 2:
        return False

    # Ends with a question but has no actual content before it.
    lines = [l.strip() for l in reply.strip().splitlines() if l.strip()]
    if len(lines) == 1 and lines[0].endswith("?"):
        return False

    return True


def _is_incomplete_command(cmd: str) -> bool:
    """Reject partial shell fragments that are likely to hang or misfire."""
    stripped = cmd.strip()
    if not stripped:
        return True

    if re.fullmatch(r"[\(\)\[\]\{\}\"'`|;&\\\s]+", stripped):
        return True

    trivial = {"(", ")", "{", "}", "[", "]", "'", '"', "`", "|", "||", "&&", ";"}
    if stripped in trivial:
        return True

    if stripped.endswith(("|", "||", "&&", "\\", "; do", "; then")):
        return True

    if stripped.count("(") != stripped.count(")"):
        return True
    if stripped.count("{") != stripped.count("}"):
        return True

    # Unbalanced quotes are almost always incomplete input in this context.
    if stripped.count("'") % 2 != 0:
        return True
    if stripped.count('"') % 2 != 0:
        return True

    first = stripped.split()[0].lower() if stripped.split() else ""
    if first in {"if", "for", "while", "until", "case", "select", "function"} and "\n" not in cmd:
        return True

    return False


def _command_preview(cmd: str, limit: int = 120) -> str:
    """One-line preview for logs/discord."""
    compact = " ".join(cmd.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _should_attempt_command_repair(cmd: str, result: str, outcome: str) -> bool:
    if outcome not in {"fail", "timeout"}:
        return False
    if not cmd.strip() or len(cmd) > 5000:
        return False
    lower = (result or "").lower()
    markers = (
        "syntax error",
        "unexpected token",
        "unterminated",
        "parse error",
        "awk:",
        "sed:",
        "bash:",
    )
    return any(m in lower for m in markers)


def _repair_command_once(
    broken_cmd: str,
    error_output: str,
    user_input: str,
    memory: list[str],
) -> str | None:
    memory_block = "\n".join(f"- {m}" for m in memory[:6]) or "(none)"
    messages = [
        {
            "role": "system",
            "content": (
                "You repair broken shell commands.\n"
                "Return exactly one tool line in this format:\n"
                "COMMAND: <fixed command>\n"
                "Rules:\n"
                "- Keep the user's intent identical.\n"
                "- Prefer portable bash/awk syntax.\n"
                "- Output only one complete runnable command.\n"
                "- No markdown fences, no explanations."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User intent: {user_input}\n\n"
                f"Recent memory:\n{memory_block}\n\n"
                f"Broken command:\n{broken_cmd}\n\n"
                f"Error output:\n{error_output[:800]}"
            ),
        },
    ]
    try:
        r = httpx.post(
            f"{LLM_BASE}/chat/completions",
            json={"model": MODEL, "messages": messages, "max_tokens": 220},
            timeout=min(MAX_LLM_TIMEOUT, 45),
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

    repaired = parse_command(content) or parse_command_fallback(content)
    if not repaired:
        return None
    repaired = _normalize_command(repaired)
    if "\n" in repaired:
        repaired = _flatten_command(repaired)
    if repaired == broken_cmd:
        return None
    if _is_incomplete_command(repaired) or not _is_safe_command(repaired):
        return None
    return repaired


def parse_read(reply: str) -> str | None:
    m = re.search(r"READ:\s*(.+)", reply)
    return m.group(1).strip() if m else None


def parse_write(reply: str) -> tuple[str, str] | None:
    m = re.search(r"WRITE:\s*(.+?)\n(.*)", reply, re.DOTALL)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def parse_screenshot(reply: str) -> str | None:
    m = re.search(r"SCREENSHOT:\s*(.+)", reply)
    return m.group(1).strip() if m else None


def parse_click(reply: str) -> tuple[int, int] | None:
    m = re.search(r"CLICK:\s*(\d+)\s+(\d+)", reply)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_type(reply: str) -> str | None:
    m = re.search(r"TYPE:\s*(.+)", reply)
    return m.group(1).strip() if m else None


def parse_search(reply: str) -> str | None:
    m = re.search(r"SEARCH:\s*(.+)", reply)
    return m.group(1).strip() if m else None


def parse_fetch(reply: str) -> str | None:
    m = re.search(r"FETCH:\s*(https?://\S+)", reply)
    return m.group(1).strip() if m else None


def parse_skill(reply: str) -> str | None:
    m = re.search(r"SKILL:\s*(\S+)", reply)
    return m.group(1).strip() if m else None


def take_screenshot(path: str) -> str:
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            f"scrot {shlex.quote(str(p))}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return f"screenshot saved to {p}"
        return f"screenshot failed: {result.stderr.strip()}"
    except Exception as e:
        return f"screenshot error: {e}"


def mouse_click(x: int, y: int) -> str:
    try:
        result = subprocess.run(
            f"xdotool mousemove {x} {y} click 1",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return f"clicked {x},{y}"
        return f"click failed: {result.stderr.strip()}"
    except Exception as e:
        return f"click error: {e}"


def keyboard_type(text: str) -> str:
    try:
        safe = text.replace("'", "'\\''")
        result = subprocess.run(
            f"xdotool type --clearmodifiers '{safe}'",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return f"typed: {text}"
        return f"type failed: {result.stderr.strip()}"
    except Exception as e:
        return f"type error: {e}"


def _search_tier1(query: str) -> str:
    """DuckDuckGo instant answer API."""
    try:
        safe_query = urllib.parse.quote_plus(query)
        url = (
            "https://api.duckduckgo.com/"
            f"?q={safe_query}&format=json&no_html=1&skip_disambig=1"
        )
        result = subprocess.run(
            f"curl -sL {shlex.quote(url)}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return ""

        data = json.loads(result.stdout)
        parts: list[str] = []
        abstract = data.get("Abstract")
        if isinstance(abstract, str) and abstract.strip():
            parts.append(abstract.strip())

        related = data.get("RelatedTopics", [])
        if isinstance(related, list):
            for topic in related[:3]:
                if isinstance(topic, dict) and isinstance(topic.get("Text"), str):
                    parts.append(topic["Text"])
                elif isinstance(topic, dict) and isinstance(topic.get("Topics"), list):
                    for nested in topic["Topics"][:2]:
                        if isinstance(nested, dict) and isinstance(nested.get("Text"), str):
                            parts.append(nested["Text"])

        return "\n---\n".join(parts[:5]) if parts else ""
    except Exception:
        return ""


def _search_tier2(query: str) -> str:
    """Playwright headless scrape of DuckDuckGo results page."""
    try:
        from playwright.sync_api import sync_playwright

        safe_query = urllib.parse.quote_plus(query)
        url = f"https://duckduckgo.com/?q={safe_query}&ia=web"
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000)
            page.wait_for_timeout(2000)
            snippets = page.eval_on_selector_all(
                "[data-result='snippet'], .result__snippet, .OgdwYG",
                "els => els.map(e => e.innerText).filter(t => t.length > 20)",
            )
            browser.close()
            if not snippets:
                return ""
            return "\n---\n".join(snippets[:5])
    except Exception:
        return ""


def web_search(query: str) -> str:
    """
    Tiered search — fast API first, then Playwright scrape when thin.
    """
    result = _search_tier1(query)
    if len(result) >= 100:
        return f"search results for '{query}':\n{result}"

    ui_print("  [tier 1 thin, trying playwright scrape...]")
    result = _search_tier2(query)
    if result:
        return f"search results for '{query}' (scraped):\n{result}"

    return f"no results found for: {query}"


def web_fetch(url: str) -> str:
    try:
        result = subprocess.run(
            f"curl -sL --max-time 10 -A 'Mozilla/5.0' {shlex.quote(url)}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"fetch failed: {result.stderr.strip()}"

        text = re.sub(r"<[^>]+>", " ", result.stdout)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return f"fetch produced no text: {url}"
        return text[:2000] if len(text) > 2000 else text
    except Exception as e:
        return f"fetch error: {e}"


def load_skill(source: str) -> str:
    """Load a skill from URL or local path, persist it, and return summary."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    if source.startswith("http://") or source.startswith("https://"):
        content = web_fetch(source)
        if content.startswith("fetch"):
            return f"skill load failed: {content}"
        name = source.rstrip("/").split("/")[-1].split("?")[0].strip() or "skill.md"
        if not name.endswith(".md"):
            name = f"{name}.md"
        skill_path = SKILLS_DIR / name
        try:
            skill_path.write_text(content, encoding="utf-8")
        except Exception as e:
            return f"skill load failed: {e}"
        return f"skill loaded from {source} -> saved to {skill_path}\n\n{content[:500]}"

    try:
        p = Path(source)
        content = p.read_text(encoding="utf-8")
        dest = SKILLS_DIR / p.name
        if p.resolve() != dest.resolve():
            dest.write_text(content, encoding="utf-8")
        return f"skill loaded from {source}\n\n{content[:500]}"
    except Exception as e:
        return f"skill load failed: {e}"


def load_all_skills() -> str:
    """Load all skill files and combine compactly for the system prompt."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skills = sorted(SKILLS_DIR.glob("*.md"))
    if not skills:
        return ""

    parts = []
    for skill in skills:
        try:
            content = skill.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if content:
            parts.append(f"### Skill: {skill.stem}\n{content[:600]}")
    return "\n\n".join(parts) if parts else ""


# -- Autonomous ---------------------------------------------------------------
_prompt_index = 0

HEARTBEAT_PROMPTS = [
    "COMMAND: now",
]


def _primary_user_name_from_soul() -> str | None:
    soul = load_soul()
    m = re.search(r"Primary user:\s*(.+)", soul, flags=re.IGNORECASE)
    if not m:
        return None
    name = m.group(1).strip()
    if not name or name.lower() in {"[name]", "name", "unknown", "n/a"}:
        return None
    return name


def next_heartbeat_prompt() -> str:
    global _prompt_index
    hour = datetime.utcnow().hour
    user_name = _primary_user_name_from_soul()

    if 6 <= hour < 10:
        prefix = "Good morning. "
    elif 18 <= hour < 22:
        prefix = "End of day check. "
    else:
        prefix = ""

    prompt = HEARTBEAT_PROMPTS[_prompt_index % len(HEARTBEAT_PROMPTS)]
    _prompt_index += 1
    if user_name:
        prompt = prompt.replace("the user", user_name).replace("user", user_name)
    return f"{prefix}{prompt}"


def autonomous_loop():
    while not _event_loop_stop.is_set():
        _event_loop_stop.wait(AUTONOMOUS_INTERVAL_SEC)
        if _event_loop_stop.is_set():
            break
        attend(next_heartbeat_prompt(), source="autonomous")


def task_scheduler_loop():
    """Check tasks.md periodically and fire due tasks as autonomous events."""
    while not _event_loop_stop.is_set():
        _event_loop_stop.wait(60)
        if _event_loop_stop.is_set():
            break
        try:
            now = datetime.utcnow()
            tasks = load_tasks()
            for task in tasks:
                name = task["name"]
                with _task_lock:
                    next_fire = _task_fire_times.get(name)
                    if next_fire is None:
                        next_fire = _next_fire(task["schedule"], now)
                        if next_fire:
                            _task_fire_times[name] = next_fire
                        continue
                    if now >= next_fire:
                        ui_print(f"\n[tasks] firing: {name}\n")
                        discord_event("task", f"task fired: {name}")
                        attend(task["prompt"], source="autonomous")
                        _task_fire_times[name] = _next_fire(task["schedule"], now)
        except Exception as e:
            ui_print(f"[tasks] error: {e}")


# -- Filesystem watcher -------------------------------------------------------
def _should_ignore_fs_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized.endswith(".pyc"):
        return True
    noisy_parts = ["/.agent/", "/__pycache__/", "/.git/", "/venv/"]
    return any(part in f"/{normalized}/" for part in noisy_parts)


def _debounced_fs_path(path: str) -> bool:
    now = time.time()
    with _fs_lock:
        last = _last_seen_fs.get(path, 0.0)
        if now - last < _FS_DEBOUNCE_SEC:
            return True
        _last_seen_fs[path] = now
    return False


def start_watcher():
    global _watcher_started
    with _watcher_lock:
        if _watcher_started:
            return
        _watcher_started = True

    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except Exception as e:
        ui_print(f"[watcher] disabled (watchdog unavailable): {e}")
        return

    class MantisHandler(FileSystemEventHandler):
        def on_created(self, event):
            self._handle(event, "created")

        def on_modified(self, event):
            self._handle(event, "modified")

        def on_deleted(self, event):
            self._handle(event, "deleted")

        def _handle(self, event, event_type: str):
            global _last_shell_journal_line
            if event.is_directory:
                return
            path = str(getattr(event, "src_path", ""))
            if not path:
                return

            if _is_shell_log_path(path) and event_type == "modified":
                try:
                    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
                    if lines:
                        last = lines[-1]
                        with _shell_journal_lock:
                            if last == _last_shell_journal_line:
                                return
                            _last_shell_journal_line = last
                        if "[you]:" in last:
                            cmd_part = last.split("[you]:", 1)[-1].strip()
                            if _is_shell_journal_noise(cmd_part):
                                return
                            attend(f"you ran in terminal: {last}", source="shell_journal")
                except Exception:
                    pass
                return

            if Path(path).name == "tasks.md" and event_type in {"created", "modified"}:
                with _task_lock:
                    _task_fire_times.clear()
                ui_print("[tasks] reloaded tasks.md")
                return

            normalized = path.replace("\\", "/")
            if "/skills/" in f"/{normalized}/" and event_type in {"created", "modified"}:
                try:
                    content = Path(path).read_text(encoding="utf-8").strip()
                    if content:
                        attend(
                            f"new skill available: {Path(path).stem}\n{content[:300]}",
                            source="skill",
                        )
                except Exception:
                    pass
                return

            if _should_ignore_fs_path(path):
                return
            if _debounced_fs_path(path):
                return
            attend(f"file {event_type}: {path}", source="filesystem")

    observer = Observer()
    try:
        observer.schedule(MantisHandler(), WATCH_PATH, recursive=True)
        observer.start()
    except Exception as e:
        ui_print(f"[watcher] failed to start on {WATCH_PATH}: {e}")
        return
    ui_print(f"[watcher] watching: {WATCH_PATH}")

    try:
        while not _event_loop_stop.is_set():
            time.sleep(1)
    finally:
        observer.stop()
        observer.join(timeout=5)


# -- Event processing ---------------------------------------------------------
def process_events_once() -> None:
    all_events = lane_queue.drain()

    # Process user and discord first, background last.
    priority = [e for e in all_events if e.get("source") in PRIORITY_SOURCES]
    background = [e for e in all_events if e.get("source") not in PRIORITY_SOURCES]

    for event in priority + background:
        source = event.get("source", "user")
        with lane_queue.lock(source):
            context = associate(event)
            if not context:
                continue
            reply = act(context)
            if source in {
                "autonomous",
                "shell",
                "shell_journal",
                "search",
                "skill",
                "discord",
            }:
                ui_print(f"\n[mantis]: {reply}\n")
            elif source not in {"agent_echo", "tool"}:
                ui_print(f"\nagent: {reply}\n")
            if source in {"discord", "autonomous", "shell", "shell_journal", "search", "skill"}:
                if _is_worth_posting(reply, source):
                    discord_post(reply)


def _report_event_loop_error(err: Exception) -> None:
    """Rate-limit repeated event loop error logs to avoid console spam."""
    global _last_event_loop_error_msg
    global _last_event_loop_error_ts
    global _suppressed_event_loop_errors

    now = time.time()
    msg = str(err).strip() or err.__class__.__name__
    same_error = msg == _last_event_loop_error_msg and (now - _last_event_loop_error_ts) < 10

    if same_error:
        _suppressed_event_loop_errors += 1
        return

    if _suppressed_event_loop_errors > 0:
        ui_print(f"[error] suppressed {_suppressed_event_loop_errors} repeated event loop errors")
        _suppressed_event_loop_errors = 0

    _last_event_loop_error_msg = msg
    _last_event_loop_error_ts = now
    ui_print(f"\n[error] event loop recovered: {msg}\n")


def event_loop():
    while not _event_loop_stop.is_set():
        try:
            process_events_once()
        except Exception as e:
            _report_event_loop_error(e)
        time.sleep(0.2)


# -- Rule 1: ATTEND -----------------------------------------------------------
def attend(text: str, source: str = "user"):
    lane_queue.put(
        {
            "text": text,
            "source": source,
            "ts": datetime.utcnow().isoformat(),
        }
    )


# -- Rule 2: ASSOCIATE --------------------------------------------------------
def associate(event: dict) -> dict | None:
    text = event["text"]
    source = event.get("source", "user")
    ts = event["ts"]

    # Drop duplicate autonomous/tool inputs to prevent heartbeat pileups.
    if source in {"autonomous", "tool"} and _is_duplicate_input(text):
        return None

    if source == "agent_echo":
        if _is_phantom_completion(text):
            # Don't store hallucinated completion claims in memory.
            return None
        fts_store(text, source, ts)
        md_append(text, source)
        return None

    if source == "learning":
        vec = embed_text(text)
        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            embeddings=[vec],
            metadatas=[{"ts": ts, "source": source}],
        )
        fts_store(text, source, ts)
        md_append(text, source)
        return None

    # Don't pollute recall with system status noise.
    if _is_memory_noise(text):
        # Keep markdown audit history, but skip vector and FTS storage.
        md_append(text, source)
        if source in {"autonomous", "tool"}:
            return None
        return {"input": text, "memory": [], "source": source}

    vec = embed_text(text)

    vec_results = collection.query(query_embeddings=[vec], n_results=TOP_K)
    vec_recall = vec_results["documents"][0] if vec_results["documents"] else []

    safe_query = re.sub(r"[^\w\s]", " ", text).strip()
    fts_recall = fts_search(safe_query) if safe_query else []
    md_recall = md_tail(4)

    seen = set()
    recalled = []
    for item in vec_recall + fts_recall + md_recall:
        if item and item not in seen:
            seen.add(item)
            recalled.append(item)
    recalled = recalled[: TOP_K * 2]

    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[text],
        embeddings=[vec],
        metadatas=[{"ts": ts, "source": source}],
    )
    fts_store(text, source, ts)
    md_append(text, source)

    return {"input": text, "memory": recalled, "source": source}


# -- Rule 3: ACT --------------------------------------------------------------
def act(context: dict) -> str:
    soul = load_soul()
    skills_block = load_all_skills()
    if len(skills_block) > 4000:
        skills_block = skills_block[:4000] + "\n...[skills truncated]"
    memory_block = "\n".join(f"- {m}" for m in context["memory"]) or "(none yet)"
    if len(memory_block) > 2000:
        memory_block = memory_block[:2000] + "\n...[truncated]"
    source = context.get("source", "user")
    active_skill: str = ""

    system = f"{soul}\n\n"
    if skills_block:
        system += f"---\n## Loaded Skills\n{skills_block}\n\n"
    system += f"---\nRelevant memory:\n{memory_block}\n\n"
    if source in {"autonomous", "learning", "shell", "skill"}:
        outcomes_block = outcome_digest(25)
        recent_outcomes_block = recent_outcome_lines(8)
        if len(outcomes_block) + len(recent_outcomes_block) > 3000:
            recent_outcomes_block = recent_outcomes_block[:2000] + "\n...[truncated]"
        system += (
            f"Recent command outcomes:\n{outcomes_block}\n\n"
            f"Recent outcome log:\n{recent_outcomes_block}\n\n"
        )
    system += (
        "---\n"
        "Tools - emit one or more per reply if needed:\n"
        "  COMMAND: <shell command>\n"
        "  READ: <filepath>\n"
        "  WRITE: <filepath>\n<full file content>\n\n"
        "  SCREENSHOT: <filepath>\n"
        "  CLICK: <x> <y>\n"
        "  TYPE: <text>\n\n"
        "  SEARCH: <query>\n"
        "  FETCH: <url>\n"
        "  SKILL: <url-or-path>\n\n"
        "For COMMAND:, emit one complete runnable command/script only.\n"
        "Never emit partial fragments like `(`, `awk '`, `if`, or unclosed quotes.\n"
        "Tool result will be fed back. Otherwise reply directly."
    )

    history = history_snapshot()
    messages = _compact_messages_for_context(
        system,
        history,
        context["input"],
        max_total_chars=MAX_PROMPT_CHARS,
    )

    try:
        r = httpx.post(
            f"{LLM_BASE}/chat/completions",
            json={"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS},
            timeout=MAX_LLM_TIMEOUT,
        )
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"].strip()
    except httpx.HTTPStatusError as e:
        if _is_context_overflow_error(e):
            compact_messages = _compact_messages_for_context(
                system,
                history[-2:],
                context["input"],
                max_total_chars=max(4000, MAX_PROMPT_CHARS // 3),
            )
            try:
                r = httpx.post(
                    f"{LLM_BASE}/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": compact_messages,
                        "max_tokens": min(MAX_TOKENS, 256),
                    },
                    timeout=MAX_LLM_TIMEOUT,
                )
                r.raise_for_status()
                reply = r.json()["choices"][0]["message"]["content"].strip()
            except Exception as retry_err:
                return f"(llm context overflow; compact retry failed: {retry_err})"
        else:
            return f"(llm http error: {e})"
    except httpx.TimeoutException:
        return (
            f"(llm timeout after {MAX_LLM_TIMEOUT}s; will continue running and retry on next event)"
        )
    except httpx.RequestError as e:
        return f"(llm request error: {e})"
    except Exception as e:
        return f"(llm error: {e})"

    cmd = parse_command(reply)
    if not cmd:
        cmd = parse_command_fallback(reply)
    # Redirect simple `cat <path>` to READ so file content stays in-turn.
    if cmd:
        try:
            cat_parts = shlex.split(cmd.strip())
        except ValueError:
            cat_parts = []
        if len(cat_parts) == 2 and cat_parts[0] == "cat":
            cmd = None
            reply = f"READ: {cat_parts[1]}"
    if cmd:
        if _is_incomplete_command(cmd):
            preview = _command_preview(cmd, 80)
            ui_print(f"\n  [skipped incomplete command: {preview}]")
            attend(
                f"command skipped: incomplete shell fragment `{preview}`. Emit a complete runnable COMMAND.",
                source="tool",
            )
            discord_event("warn", f"skipped incomplete command `{preview}`")
            cmd = None
            reply = "(skipped)"
    if cmd:
        if not _is_safe_command(cmd):
            preview = _command_preview(cmd, 80)
            ui_print(f"\n  [skipped unsafe command: {preview}]")
            attend("command skipped: failed safety check", source="tool")
            discord_event("warn", "skipped unsafe command")
            cmd = None
            reply = "(skipped)"
    if cmd:
        stripped = cmd.strip().lower()
        base = stripped.split()[0] if stripped else ""
        if any(stripped.startswith(c) for c in INTERACTIVE_COMMANDS):
            alt = SAFE_ALTERNATIVES.get(base)
            if alt:
                ui_print(f"\n  [blocked interactive: {_command_preview(cmd, 80)} -> using: {alt}]")
                discord_event(
                    "warn",
                    f"blocked interactive `{_command_preview(cmd, 60)}`, using safe alternative",
                )
                cmd = alt
            else:
                preview = _command_preview(cmd, 80)
                ui_print(f"\n  [blocked interactive command: {preview}]")
                attend(
                    f"blocked: {cmd} is interactive — suggest a non-interactive alternative",
                    source="tool",
                )
                discord_event("warn", f"blocked interactive `{preview}`")
                reply = reply[: reply.index("COMMAND:")].strip() or f"(blocked: {cmd})"
                cmd = None
        if cmd is None:
            # Already handled blocked command path.
            pass
        else:
            normalized = cmd.strip().lower()
            is_long = ("\n" in cmd) or any(
                normalized.startswith(prefix) for prefix in ASYNC_COMMAND_PREFIXES
            )
            if is_long:
                preview = _command_preview(cmd, 120)
                ui_print(f"\n  [running async script: {preview}]")
                discord_event("run", f"`{preview}`")
                run_script_async(cmd, trigger_source=source, skill=active_skill)
                reply = "(running...)"
            else:
                preview = _command_preview(cmd, 120)
                ui_print(f"\n  [running: {preview}]")
                started = time.time()
                discord_event("run", f"`{preview}`")
                result = run_script(cmd, timeout=30)
                elapsed = round(time.time() - started, 1)
                duration_ms = int((time.time() - started) * 1000)
                ui_print(f"  {result[:200]}\n")
                _log_to_shell_journal(cmd, result, actor="mantis")
                outcome = _record_outcome_and_learn(
                    cmd,
                    result,
                    source,
                    duration_ms=duration_ms,
                    skill=active_skill,
                )
                attend(f"command result for `{cmd[:80]}`:\n{result}", source="tool")
                discord_event("done", f"`{preview}` done ({elapsed}s)\n{result[:300]}")

                repaired_result = None
                if MAX_AUTO_REPAIR_ATTEMPTS > 0 and _should_attempt_command_repair(
                    cmd, result, outcome
                ):
                    repaired_cmd = _repair_command_once(
                        cmd,
                        result,
                        context["input"],
                        context.get("memory", []),
                    )
                    if repaired_cmd:
                        repaired_preview = _command_preview(repaired_cmd, 120)
                        ui_print(f"\n  [auto-repair retry: {repaired_preview}]")
                        discord_event(
                            "warn",
                            f"auto-repair retry for `{preview}` -> `{repaired_preview}`",
                        )
                        retry_start = time.time()
                        repaired_result = run_script(repaired_cmd, timeout=30)
                        retry_elapsed = round(time.time() - retry_start, 1)
                        retry_duration_ms = int((time.time() - retry_start) * 1000)
                        ui_print(f"  {repaired_result[:200]}\n")
                        _log_to_shell_journal(repaired_cmd, repaired_result, actor="mantis")
                        _record_outcome_and_learn(
                            repaired_cmd,
                            repaired_result,
                            source,
                            duration_ms=retry_duration_ms,
                            skill=active_skill,
                        )
                        attend(
                            f"command retry result for `{repaired_cmd[:80]}`:\n{repaired_result}",
                            source="tool",
                        )
                        discord_event(
                            "done",
                            f"`{repaired_preview}` done ({retry_elapsed}s)\n{repaired_result[:300]}",
                        )
                reply = "(ran command)"

    read_path = parse_read(reply)
    if read_path:
        ui_print(f"\n  [reading: {read_path}]")
        contents = read_file(read_path)
        ui_print(f"  ({len(contents)} chars)\n")
        discord_event("read", f"`{read_path}` ({len(contents)} chars)")

        # Inject file contents into context and continue this turn.
        enriched_input = (
            f"{context['input']}\n\n"
            f"[contents of {read_path}]:\n"
            f"{_truncate_text(contents, 3000)}"
        )
        new_context = dict(context)
        new_context["input"] = enriched_input
        new_context["_read_depth"] = context.get("_read_depth", 0) + 1

        # Guard against infinite read loops.
        if new_context["_read_depth"] <= 3:
            return act(new_context)
        attend(f"file contents of {read_path}:\n{contents}", source="tool")
        return "(read file - max depth reached)"

    write_result = parse_write(reply)
    if write_result:
        wpath, wcontent = write_result
        was_blocked = False
        if _is_skill_write(wpath):
            reply, was_blocked = _guarded_skill_write(wpath, wcontent, reply)
        if not was_blocked and _is_todo_write(wpath):
            reply, was_blocked = _guarded_todo_write(wpath, wcontent, reply)

        if not was_blocked:
            ui_print(f"\n  [writing: {wpath}]")
            discord_event("write", f"`{wpath}`")
            result = write_file(wpath, wcontent)
            ui_print(f"  {result}\n")
            discord_event("done", result[:300])
            attend(f"file write result: {result}", source="tool")
            reply = reply[: reply.index("WRITE:")].strip() or "(wrote file)"

    screenshot_path = parse_screenshot(reply)
    if screenshot_path:
        ui_print(f"\n  [screenshot: {screenshot_path}]")
        discord_event("info", f"screenshot `{screenshot_path}`")
        result = take_screenshot(screenshot_path)
        ui_print(f"  {result}\n")
        discord_event("done", result[:300])
        attend(f"screenshot result: {result}", source="tool")
        reply = reply[: reply.index("SCREENSHOT:")].strip() or "(took screenshot)"

    click_coords = parse_click(reply)
    if click_coords:
        x, y = click_coords
        ui_print(f"\n  [click: {x},{y}]")
        discord_event("info", f"click `{x},{y}`")
        result = mouse_click(x, y)
        ui_print(f"  {result}\n")
        discord_event("done", result[:300])
        attend(f"click result: {result}", source="tool")
        reply = reply[: reply.index("CLICK:")].strip() or "(clicked)"

    type_text = parse_type(reply)
    if type_text:
        ui_print(f"\n  [type: {type_text}]")
        discord_event("info", f"type `{type_text[:80]}`")
        result = keyboard_type(type_text)
        ui_print(f"  {result}\n")
        discord_event("done", result[:300])
        attend(f"type result: {result}", source="tool")
        reply = reply[: reply.index("TYPE:")].strip() or "(typed)"

    search_query = parse_search(reply)
    if search_query:
        ui_print(f"\n  [searching: {search_query}]")
        discord_event("search", f"`{search_query[:120]}`")
        result = web_search(search_query)
        ui_print(f"  ({len(result)} chars)\n")
        discord_event("done", f"search returned {len(result)} chars")

        # Inject search results into context and continue this turn.
        enriched_input = (
            f"{context['input']}\n\n"
            f"[search results for '{search_query}']:\n"
            f"{_truncate_text(result, 2000)}"
        )
        new_context = dict(context)
        new_context["input"] = enriched_input
        new_context["_search_depth"] = context.get("_search_depth", 0) + 1

        if new_context["_search_depth"] <= 2:
            return act(new_context)
        attend(f"search results: {result}", source="search")
        return "(searched - max depth reached)"

    fetch_url = parse_fetch(reply)
    if fetch_url:
        ui_print(f"\n  [fetching: {fetch_url}]")
        discord_event("fetch", f"`{fetch_url}`")
        result = web_fetch(fetch_url)
        ui_print(f"  ({len(result)} chars)\n")
        discord_event("done", f"fetch returned {len(result)} chars")

        # Inject fetched content into context and continue this turn.
        enriched_input = (
            f"{context['input']}\n\n"
            f"[fetched content from {fetch_url}]:\n"
            f"{_truncate_text(result, 2000)}"
        )
        new_context = dict(context)
        new_context["input"] = enriched_input
        new_context["_fetch_depth"] = context.get("_fetch_depth", 0) + 1

        if new_context["_fetch_depth"] <= 2:
            return act(new_context)
        attend(f"fetched content from {fetch_url}:\n{result}", source="search")
        return "(fetched - max depth reached)"

    skill_source = parse_skill(reply)
    if skill_source:
        ui_print(f"\n  [loading skill: {skill_source}]")
        discord_event("skill", f"loading `{skill_source}`")
        result = load_skill(skill_source)
        ui_print(f"  ({len(result)} chars)\n")
        skill_name = Path(skill_source).name if "://" not in skill_source else skill_source.split("/")[-1]
        active_skill = Path(skill_name).stem
        discord_event("done", f"skill ready: {skill_name[:80]}")
        attend(f"skill loaded: {result}", source="skill")
        reply = reply[: reply.index("SKILL:")].strip() or "(loaded skill)"

    # Store exchange in history.
    history_append("user", context["input"], source=source)
    history_append("assistant", reply, source=source)

    attend(f"agent: {reply}", source="agent_echo")
    return reply


# -- Main loop ----------------------------------------------------------------
def input_loop():
    """Non-blocking stdin loop so autonomous output can print immediately."""
    while not _event_loop_stop.is_set():
        _input_waiting.set()
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if not ready:
            continue

        line = sys.stdin.readline()
        if line == "":
            # EOF; avoid tight loops and allow graceful shutdown.
            time.sleep(0.1)
            continue

        user_input = line.strip()
        if not user_input:
            ui_print()
            continue
        if user_input.lower() in {"exit", "quit"}:
            _event_loop_stop.set()
            break
        attend(user_input, source="user")


def main():
    threading.Thread(target=event_loop, daemon=True).start()
    threading.Thread(target=autonomous_loop, daemon=True).start()
    threading.Thread(target=task_scheduler_loop, daemon=True).start()
    threading.Thread(target=start_watcher, daemon=True).start()
    threading.Thread(target=input_loop, daemon=True).start()
    threading.Thread(target=_start_discord, daemon=True).start()

    ui_print("agent ready. ctrl+c to exit.\n", redraw_prompt=False)
    print("you: ", end="", flush=True)
    _input_waiting.set()
    try:
        _event_loop_stop.wait()
    except KeyboardInterrupt:
        _event_loop_stop.set()
        ui_print("\nbye.", redraw_prompt=False)


if __name__ == "__main__":
    main()
