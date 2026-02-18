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
import sqlite3
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import chromadb
import httpx
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

DB_PATH = MEMORY_DIR / "fts.db"
MEMORY_MD = Path(".agent/MEMORY.md")
EMBED_DIM = 384

print(START_UP_LOGO)
# -- Boot ---------------------------------------------------------------------
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
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


def _get_sentence_transformer():
    global _encoder
    if _encoder is not None:
        return _encoder
    with _encoder_lock:
        if _encoder is not None:
            return _encoder
        # Lazy import to avoid startup crashes on unsupported CPU instruction sets.
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return _encoder


def embed_text(text: str) -> list[float]:
    if EMBEDDING_BACKEND == "sentence-transformers":
        return _get_sentence_transformer().encode(text).tolist()
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
def run_command(cmd: str) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        out = result.stdout.strip() or result.stderr.strip() or "(no output)"
        return f"$ {cmd}\n{out}"
    except subprocess.TimeoutExpired:
        return f"$ {cmd}\n(timed out after 30s)"
    except Exception as e:
        return f"$ {cmd}\n(error: {e})"


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


def autonomous_prompt_for_hour(hour_utc: int) -> str:
    if 6 <= hour_utc < 10:
        return (
            "It's morning. Reflect on the day ahead using recent memory and identify "
            "one concrete priority."
        )
    if 18 <= hour_utc < 22:
        return (
            "It's evening. Review what was done today and explicitly note unresolved "
            "items to revisit."
        )
    return (
        "You're in an idle reflection window. Synthesize recent memories into a short "
        "summary and store one durable insight."
    )


def autonomous_loop():
    while not _event_loop_stop.is_set():
        _event_loop_stop.wait(AUTONOMOUS_INTERVAL_SEC)
        if _event_loop_stop.is_set():
            break
        prompt = autonomous_prompt_for_hour(datetime.utcnow().hour)
        attend(prompt, source="autonomous")


def _should_ignore_fs_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized.endswith(".pyc"):
        return True
    noisy_parts = ["/.agent/", "/__pycache__/", "/.git/"]
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
            if event.is_directory:
                return
            path = str(getattr(event, "src_path", ""))
            if not path or _should_ignore_fs_path(path):
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


def process_events_once() -> None:
    events = lane_queue.drain()
    for event in events:
        source = event.get("source", "user")
        with lane_queue.lock(source):
            context = associate(event)
            if not context:
                continue
            reply = act(context)
            if source == "autonomous":
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

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": context["input"]},
    ]

    r = httpx.post(
        f"{LLM_BASE}/chat/completions",
        json={"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS},
        timeout=60,
    )
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"].strip()

    cmd = parse_command(reply)
    if cmd:
        ui_print(f"\n  [running: {cmd}]")
        result = run_command(cmd)
        ui_print(f"  {result}\n")
        attend(f"command result: {result}", source="tool")
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

    attend(f"agent: {reply}", source="agent_echo")
    return reply


# -- Main loop ----------------------------------------------------------------
def main():
    threading.Thread(target=event_loop, daemon=True).start()
    threading.Thread(target=autonomous_loop, daemon=True).start()
    threading.Thread(target=start_watcher, daemon=True).start()

    ui_print("agent ready. ctrl+c to exit.\n", redraw_prompt=False)
    while True:
        try:
            _input_waiting.set()
            user_input = input("you: ").strip()
            _input_waiting.clear()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                _event_loop_stop.set()
                break

            attend(user_input, source="user")
        except KeyboardInterrupt:
            _input_waiting.clear()
            _event_loop_stop.set()
            ui_print("\nbye.", redraw_prompt=False)
            break


if __name__ == "__main__":
    main()
