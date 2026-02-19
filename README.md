# Mantis

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/mantis-logo.png">
    <img src="docs/assets/mantis-logo.png" alt="Mantis logo" width="500">
  </picture>
</p>

Mantis is a local-first Python agent runtime with a three-rule loop:
`ATTEND -> ASSOCIATE -> ACT`.

Current capabilities include:
- Hybrid memory (Chroma vectors + SQLite FTS5 + markdown log)
- Persistent shell execution (sync + async)
- Shared shell journal awareness (`.agent/shell.log`)
- Autonomous heartbeat prompts (rotating, time-modified)
- Filesystem watcher ingestion
- Linux computer-use tools (`SCREENSHOT`, `CLICK`, `TYPE`)
- Tiered web search (`SEARCH`) and page fetch (`FETCH`)
- Skill loading from local files or URLs (`SKILL`)

## Scope

This repo is a single executable runtime (`mantis.py`) for local experimentation and prototyping. It is not packaged as a production service.

## Runtime model

For each event, Mantis runs:
1. `ATTEND`: enqueue an event with source + timestamp.
2. `ASSOCIATE`: recall related memory and persist the new event.
3. `ACT`: call the LLM and optionally execute one or more tool directives.

Events are processed in per-source serialized lanes (FIFO per source).

## Event sources

Current sources in code:
- `user`
- `autonomous`
- `filesystem`
- `tool`
- `search`
- `skill`
- `shell`
- `shell_journal`
- `agent_echo`

## Memory

Memory layers:
- ChromaDB collection `events` in `MEMORY_DIR`
- SQLite FTS5 table `memories` in `MEMORY_DIR/fts.db`
- Markdown append-only log `.agent/MEMORY.md`

Recall merges vector + keyword + markdown tail, deduplicates, then injects into prompt as relevant memory.

## Tool directives

Mantis can execute these directives from model output:
- `COMMAND: <shell command>`
- `READ: <filepath>`
- `WRITE: <filepath>` + full content
- `SCREENSHOT: <filepath>`
- `CLICK: <x> <y>`
- `TYPE: <text>`
- `SEARCH: <query>`
- `FETCH: <url>`
- `SKILL: <url-or-path>`

Tool outcomes are fed back into the event bus and become memory.

## Shell model

Mantis uses a persistent `/bin/bash` via `pexpect`:
- state survives across commands (`cd`, env vars, etc.)
- short commands run synchronously
- long commands run asynchronously and report back on completion

Long-command prefixes currently include installs/downloads (for example `pip install`, `npm install`, `git clone`).

## Shared shell journal

Mantis writes command/result entries to `SHELL_LOG` (default `.agent/shell.log`) and auto-installs a `.bashrc` hook to log user shell commands.

That creates cross-awareness:
- user commands -> journal -> watcher -> `shell_journal` events
- mantis commands -> journal entries for auditability

## Autonomous behavior

Autonomous loop fires every `AUTONOMOUS_INTERVAL_SEC` (default 300s) with rotating prompts, including:
- unfinished work
- system health checks
- todo checks
- memory synthesis
- open questions
- file awareness
- reminders
- curiosity search prompt

Time-of-day is a modifier:
- morning UTC adds `Good morning.`
- evening UTC adds `End of day check.`

If `Primary user:` exists in soul content, prompts personalize user references.

## Web research

`SEARCH` is tiered:
1. DuckDuckGo instant answer API (fast)
2. Playwright headless Firefox scrape of DDG results when tier 1 is too thin (`<100` chars)

`FETCH` pulls a specific URL with `curl`, strips HTML tags, and trims text for context.

## Skills system

Skills live in `.agent/skills/*.md` and persist across restarts.

Ways to load:
- `SKILL: https://.../skill.md` (fetch + save)
- `SKILL: .agent/skills/local.md` (read local + copy if needed)

Loaded skills are injected into the system prompt under `## Loaded Skills` (trimmed per skill for context control).

Filesystem changes under `/skills/` trigger immediate skill events.

## Computer use (Linux)

Mantis supports blind-first computer control via:
- `scrot` for screenshots
- `xdotool` for mouse/keyboard actions

Boot setup:
- defaults `DISPLAY=:0` when unset
- attempts to install missing `scrot`/`xdotool`
- attempts to ensure Playwright Firefox is available

## Input/output behavior

CLI input is non-blocking (`select.select`) in a dedicated thread, so autonomous/shell/search/skill messages can print without waiting for user keystrokes.

Output prefixes:
- `agent:` for direct user-facing responses
- `[mantis]:` for autonomous/self-triggered streams (autonomous, shell, shell_journal, search, skill)

Exit with `Ctrl+C` or `exit` / `quit`.

## Requirements

- Python 3.10+
- Reachable OpenAI-compatible API endpoint for:
  - `POST /v1/chat/completions`
  - optionally `POST /v1/embeddings`
- Python deps in `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python3 mantis.py
```

## Additional runtime tools (system-level)

Some capabilities rely on non-Python tools:
- `scrot`
- `xdotool`
- `curl`
- Playwright Firefox runtime (`playwright install firefox`)

`mantis.py` attempts auto-setup for some of these on boot, but manual install may still be needed depending on permissions/environment.

## Configuration

Environment variables are loaded from `.env` via `python-dotenv`.

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE` | `http://localhost:8001/v1` | Base URL for model API |
| `MODEL` | `Qwen2.5-14B-Instruct-Q4_K_M.gguf` | Model sent to API |
| `MEMORY_DIR` | `.agent/memory` | Chroma + SQLite storage path |
| `SOUL_PATH` | `SOUL.md` | Soul prompt file |
| `TOP_K` | `4` | Retrieval depth |
| `MAX_TOKENS` | `512` | Completion max tokens |
| `MAX_LLM_TIMEOUT` | `120` | LLM request timeout (seconds) |
| `EMBEDDING_BACKEND` | `hash` | `hash`, `llm`, or `sentence-transformers` |
| `AUTONOMOUS_INTERVAL_SEC` | `300` | Autonomous heartbeat interval |
| `WATCH_PATH` | `.` | Filesystem watcher root |
| `MAX_HISTORY` | `10` | Recent chat messages kept in context |
| `SHELL_LOG` | `.agent/shell.log` | Shared shell journal path |

## Soul file note

Repo file is `soul.md` (lowercase), while default config uses `SOUL.md`.
On case-sensitive filesystems, either:
- set `SOUL_PATH=soul.md`, or
- rename file to `SOUL.md`.

## Repository layout

- `mantis.py` — runtime
- `soul.md` — soul prompt/instructions
- `requirements.txt` — Python dependencies
- `docs/assets/` — logo/startup art

## Security and limits

This runtime is intentionally permissive:
- shell commands run as current OS user
- file writes are unconstrained within process permissions
- external network calls are allowed by tool directives
- no built-in policy engine or sandbox inside `mantis.py`

Use in controlled environments.

## Inspiration

- OpenClaw: https://github.com/OpenClaw/OpenClaw
- PicoClaw: https://github.com/sipeed/picoclaw
