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
import threading
import time
import urllib.parse
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
import chromadb
import httpx
import pexpect
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


LLM_BASE = os.getenv("LLM_BASE", "http://localhost:8001/v1")
MODEL = os.getenv("MODEL", "Qwen2.5-14B-Instruct-Q4_K_M.gguf")
MEMORY_DIR = Path(os.getenv("MEMORY_DIR", ".agent/memory"))
SOUL_PATH = Path(os.getenv("SOUL_PATH", "SOUL.md"))
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hash").lower()
AUTONOMOUS_INTERVAL_SEC = int(os.getenv("AUTONOMOUS_INTERVAL_SEC", "300"))
WATCH_PATH = os.getenv("WATCH_PATH", ".")
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
SHELL_LOG = Path(os.getenv("SHELL_LOG", ".agent/shell.log"))
MAX_LLM_TIMEOUT = int(os.getenv("MAX_LLM_TIMEOUT", "120"))
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
DISCORD_CHANNEL_ID = _env_int("DISCORD_CHANNEL_ID", 0)

DB_PATH = MEMORY_DIR / "fts.db"
MEMORY_MD = Path(".agent/MEMORY.md")
SKILLS_DIR = Path(".agent/skills")
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


# -- Conversation history -----------------------------------------------------
_chat_history: list[dict] = []
_history_lock = threading.Lock()


def history_append(role: str, content: str, source: str = "user") -> None:
    # only track user <-> agent exchanges, not tool/filesystem noise
    if source not in {"user", "autonomous"}:
        return
    with _history_lock:
        _chat_history.append({"role": role, "content": content})
        if len(_chat_history) > MAX_HISTORY * 2:
            _chat_history[:] = _chat_history[-MAX_HISTORY * 2:]


def history_snapshot() -> list[dict]:
    with _history_lock:
        return list(_chat_history[-MAX_HISTORY:])


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

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        if message.channel.id != DISCORD_CHANNEL_ID:
            return
        if not message.content:
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


# -- Soul ---------------------------------------------------------------------
def load_soul() -> str:
    if SOUL_PATH.exists():
        return SOUL_PATH.read_text(encoding="utf-8").strip()
    return "You are a sharp, minimal, autonomous agent with memory and terminal access."


# -- Tools --------------------------------------------------------------------
_shell: pexpect.spawn | None = None
_shell_lock = threading.Lock()
_shell_io_lock = threading.Lock()
_SHELL_PROMPT = "__MANTIS_DONE__"

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


def _reset_shell() -> None:
    global _shell
    with _shell_lock:
        if _shell is not None:
            try:
                _shell.terminate(force=True)
            except Exception:
                pass
        _shell = None


def _get_shell() -> pexpect.spawn:
    global _shell
    with _shell_lock:
        if _shell is None or not _shell.isalive():
            shell = pexpect.spawn(
                "/bin/bash",
                encoding="utf-8",
                timeout=None,
            )
            shell.sendline("export TERM=dumb")
            shell.sendline(f'export PS1="{_SHELL_PROMPT} "')
            shell.expect(_SHELL_PROMPT, timeout=5)
            _shell = shell
        return _shell


def _shell_output(output: str, cmd: str) -> str:
    lines = output.splitlines()
    if lines and lines[0].strip() == cmd.strip():
        lines = lines[1:]
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\x1b\[[0-9;]*[mGKHF]", "", cleaned)
    cleaned = re.sub(r"\x1b\][^\x07]*\x07", "", cleaned)
    cleaned = cleaned.strip()
    return cleaned or "(no output)"


def _run_shell_command(cmd: str, timeout: int) -> str:
    with _shell_io_lock:
        shell = _get_shell()
        shell.sendline(cmd)
        shell.expect(_SHELL_PROMPT, timeout=timeout)
        return _shell_output(shell.before.strip(), cmd)


def run_command_async(cmd: str) -> None:
    """Run a command in the persistent shell and feed result back as events."""

    def _run():
        try:
            result = _run_shell_command(cmd, timeout=60)
            _log_to_shell_journal(cmd, result, actor="mantis")
            attend(f"shell result for `{cmd}`:\n{result}", source="shell")
        except pexpect.TIMEOUT:
            _reset_shell()
            result = "timed out after 60s"
            _log_to_shell_journal(cmd, result, actor="mantis")
            attend(f"shell result for `{cmd}`: {result}", source="shell")
        except Exception as e:
            _reset_shell()
            result = f"error — {e}"
            _log_to_shell_journal(cmd, result, actor="mantis")
            attend(f"shell result for `{cmd}`: {result}", source="shell")

    threading.Thread(target=_run, daemon=True).start()


def run_command(cmd: str) -> str:
    """Synchronous command execution over the persistent shell."""
    try:
        result = _run_shell_command(cmd, timeout=30)
        _log_to_shell_journal(cmd, result, actor="mantis")
        return result
    except pexpect.TIMEOUT:
        _reset_shell()
        result = "(timed out after 30s; shell session reset)"
        _log_to_shell_journal(cmd, result, actor="mantis")
        return result
    except Exception as e:
        _reset_shell()
        result = f"(error: {e})"
        _log_to_shell_journal(cmd, result, actor="mantis")
        return result


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


def parse_command(reply: str) -> str | None:
    m = re.search(r"COMMAND:\s*(.+)", reply)
    return m.group(1).strip() if m else None


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
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0]
    if first.startswith("$"):
        first = first[1:].strip()
    return first or None


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
    # Unfinished work
    "What has the user asked you to do that isn't finished yet? Be specific and surface it.",
    # Hardware awareness
    "Check system health. COMMAND: now && cat /proc/loadavg && free -h",
    # Todo check
    "Is there a todo list? If so, read todo_list.txt and remind the user of anything incomplete.",
    # Memory synthesis
    "Review recent memory. What's the single most important thing the user is working on right now?",
    # Open question
    "Based on recent memory, what's one thing you're uncertain about that would help to clarify with the user?",
    # File awareness
    "What files have changed recently that the user might care about?",
    # Reminder surface
    "Did the user mention anything they wanted to follow up on later? Surface it now.",
    # Curiosity
    "You're curious about something the user mentioned recently. Search for it. SEARCH: <topic from memory>",
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
                            attend(f"you ran in terminal: {last}", source="shell_journal")
                except Exception:
                    pass
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
    events = lane_queue.drain()
    for event in events:
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
                discord_post(reply)


def event_loop():
    while not _event_loop_stop.is_set():
        try:
            process_events_once()
        except Exception as e:
            ui_print(f"\n[error] event loop recovered: {e}\n")
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

    if source == "agent_echo":
        fts_store(text, source, ts)
        md_append(text, source)
        return None

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
    memory_block = "\n".join(f"- {m}" for m in context["memory"]) or "(none yet)"
    source = context.get("source", "user")

    system = f"{soul}\n\n"
    if skills_block:
        system += f"---\n## Loaded Skills\n{skills_block}\n\n"
    system += (
        "---\n"
        f"Relevant memory:\n{memory_block}\n\n"
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
        "Tool result will be fed back. Otherwise reply directly."
    )

    messages = [{"role": "system", "content": system}]
    messages.extend(history_snapshot())
    messages.append({"role": "user", "content": context["input"]})

    r = httpx.post(
        f"{LLM_BASE}/chat/completions",
        json={"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS},
        timeout=MAX_LLM_TIMEOUT,
    )
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"].strip()

    cmd = parse_command(reply)
    cmd_from_fallback = False
    if not cmd:
        cmd = parse_command_fallback(reply)
        cmd_from_fallback = cmd is not None
    if cmd:
        stripped = cmd.strip().lower()
        base = stripped.split()[0] if stripped else ""
        if any(stripped.startswith(c) for c in INTERACTIVE_COMMANDS):
            alt = SAFE_ALTERNATIVES.get(base)
            if alt:
                ui_print(f"\n  [blocked interactive: {cmd} -> using: {alt}]")
                cmd = alt
            else:
                ui_print(f"\n  [blocked interactive command: {cmd}]")
                attend(
                    f"blocked: {cmd} is interactive — suggest a non-interactive alternative",
                    source="tool",
                )
                reply = reply[: reply.index("COMMAND:")].strip() or f"(blocked: {cmd})"
                cmd = None
        if cmd is None:
            # Already handled blocked command path.
            pass
        else:
            normalized = cmd.strip().lower()
            is_long = any(normalized.startswith(prefix) for prefix in ASYNC_COMMAND_PREFIXES)
            if is_long:
                ui_print(f"\n  [running async: {cmd}]")
                run_command_async(cmd)
                reply = "(running...)"
            else:
                ui_print(f"\n  [running: {cmd}]")
                result = run_command(cmd)
                ui_print(f"  {result}\n")
                attend(f"command result for `{cmd}`:\n{result}", source="tool")
                reply = "(ran command)"

    read_path = parse_read(reply)
    if read_path:
        ui_print(f"\n  [reading: {read_path}]")
        contents = read_file(read_path)
        ui_print(f"  ({len(contents)} chars)\n")
        attend(f"file contents of {read_path}:\n{contents}", source="tool")
        reply = reply[: reply.index("READ:")].strip() or "(read file)"

    write_result = parse_write(reply)
    if write_result:
        wpath, wcontent = write_result
        ui_print(f"\n  [writing: {wpath}]")
        result = write_file(wpath, wcontent)
        ui_print(f"  {result}\n")
        attend(f"file write result: {result}", source="tool")
        reply = reply[: reply.index("WRITE:")].strip() or "(wrote file)"

    screenshot_path = parse_screenshot(reply)
    if screenshot_path:
        ui_print(f"\n  [screenshot: {screenshot_path}]")
        result = take_screenshot(screenshot_path)
        ui_print(f"  {result}\n")
        attend(f"screenshot result: {result}", source="tool")
        reply = reply[: reply.index("SCREENSHOT:")].strip() or "(took screenshot)"

    click_coords = parse_click(reply)
    if click_coords:
        x, y = click_coords
        ui_print(f"\n  [click: {x},{y}]")
        result = mouse_click(x, y)
        ui_print(f"  {result}\n")
        attend(f"click result: {result}", source="tool")
        reply = reply[: reply.index("CLICK:")].strip() or "(clicked)"

    type_text = parse_type(reply)
    if type_text:
        ui_print(f"\n  [type: {type_text}]")
        result = keyboard_type(type_text)
        ui_print(f"  {result}\n")
        attend(f"type result: {result}", source="tool")
        reply = reply[: reply.index("TYPE:")].strip() or "(typed)"

    search_query = parse_search(reply)
    if search_query:
        ui_print(f"\n  [searching: {search_query}]")
        result = web_search(search_query)
        ui_print(f"  ({len(result)} chars)\n")
        attend(f"search results: {result}", source="search")
        reply = reply[: reply.index("SEARCH:")].strip() or "(searched)"

    fetch_url = parse_fetch(reply)
    if fetch_url:
        ui_print(f"\n  [fetching: {fetch_url}]")
        result = web_fetch(fetch_url)
        ui_print(f"  ({len(result)} chars)\n")
        attend(f"fetched content from {fetch_url}:\n{result}", source="search")
        reply = reply[: reply.index("FETCH:")].strip() or "(fetched)"

    skill_source = parse_skill(reply)
    if skill_source:
        ui_print(f"\n  [loading skill: {skill_source}]")
        result = load_skill(skill_source)
        ui_print(f"  ({len(result)} chars)\n")
        attend(f"skill loaded: {result}", source="skill")
        reply = reply[: reply.index("SKILL:")].strip() or "(loaded skill)"

    # store exchange in history (user and autonomous only)
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
