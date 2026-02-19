#!/usr/bin/env python3
"""
Three-rule emergent agent
ATTEND -> ASSOCIATE -> ACT -> (side effects loop back)
Memory: ChromaDB (vector) + SQLite FTS5 (keyword) + MEMORY.md (markdown)
Safety: Lane Queue (serial FIFO per source)
"""
import hashlib
import os
import re
import select
import sqlite3
import sys
import threading
import time
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

DB_PATH = MEMORY_DIR / "fts.db"
MEMORY_MD = Path(".agent/MEMORY.md")
EMBED_DIM = 384

print(START_UP_LOGO)

# -- Boot ---------------------------------------------------------------------
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


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


def ui_print(message: str = "", redraw_prompt: bool = True) -> None:
    with _console_lock:
        if message:
            print(message, flush=True)
        else:
            print("", flush=True)
        if redraw_prompt and _input_waiting.is_set():
            print("you: ", end="", flush=True)


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
            shell.sendline(f'export PS1="{_SHELL_PROMPT} "')
            shell.expect(_SHELL_PROMPT, timeout=5)
            _shell = shell
        return _shell


def _shell_output(output: str, cmd: str) -> str:
    lines = output.splitlines()
    if lines and lines[0].strip() == cmd.strip():
        lines = lines[1:]
    cleaned = "\n".join(lines).strip()
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


def parse_read(reply: str) -> str | None:
    m = re.search(r"READ:\s*(.+)", reply)
    return m.group(1).strip() if m else None


def parse_write(reply: str) -> tuple[str, str] | None:
    m = re.search(r"WRITE:\s*(.+?)\n(.*)", reply, re.DOTALL)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


# -- Autonomous ---------------------------------------------------------------
_prompt_index = 0

HEARTBEAT_PROMPTS = [
    # Unfinished work
    "What has the user asked you to do that isn't finished yet? Be specific and surface it.",
    # Hardware awareness
    "Check system health. COMMAND: uptime && df -h && free -h",
    # Todo check
    "Is there a todo list? If so read it and remind the user of anything incomplete. READ: todo_list.txt",
    # Memory synthesis
    "Review recent memory. What's the single most important thing the user is working on right now?",
    # Open question
    "Based on recent memory, what's one thing you're uncertain about that would help to clarify with the user?",
    # File awareness
    "What files have changed recently that the user might care about?",
    # Reminder surface
    "Did the user mention anything they wanted to follow up on later? Surface it now.",
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
            if source in {"autonomous", "shell", "shell_journal"}:
                ui_print(f"\n[mantis]: {reply}\n")
            elif source not in {"agent_echo", "tool"}:
                ui_print(f"\nagent: {reply}\n")


def event_loop():
    while not _event_loop_stop.is_set():
        process_events_once()
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
    memory_block = "\n".join(f"- {m}" for m in context["memory"]) or "(none yet)"
    source = context.get("source", "user")

    system = (
        f"{soul}\n\n"
        "---\n"
        f"Relevant memory:\n{memory_block}\n\n"
        "---\n"
        "Tools - emit exactly one per reply if needed:\n"
        "  COMMAND: <shell command>\n"
        "  READ: <filepath>\n"
        "  WRITE: <filepath>\n<full file content>\n\n"
        "Tool result will be fed back. Otherwise reply directly."
    )

    messages = [{"role": "system", "content": system}]
    messages.extend(history_snapshot())
    messages.append({"role": "user", "content": context["input"]})

    r = httpx.post(
        f"{LLM_BASE}/chat/completions",
        json={"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS},
        timeout=60,
    )
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"].strip()

    cmd = parse_command(reply)
    if cmd:
        normalized = cmd.strip().lower()
        is_long = any(normalized.startswith(prefix) for prefix in ASYNC_COMMAND_PREFIXES)
        if is_long:
            ui_print(f"\n  [running async: {cmd}]")
            run_command_async(cmd)
        else:
            ui_print(f"\n  [running: {cmd}]")
            result = run_command(cmd)
            ui_print(f"  {result}\n")
            attend(f"command result for `{cmd}`:\n{result}", source="tool")
        reply = reply[: reply.index("COMMAND:")].strip() or "(ran command)"

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
