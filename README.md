# Mantis

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/mantis-logo.png">
    <img src="docs/assets/mantis-logo.png" alt="Mantis logo" width="500">
  </picture>
</p>

Inspired by Craig Reynolds' Boids, Mantis applies a minimal local-rule approach to emergent behavior: simple rules, repeated continuously, produce adaptive autonomous behavior.

Mantis is a local-first Python agent runtime built around a three-rule loop:
`ATTEND -> ASSOCIATE -> ACT`.

It runs as a single process (`mantis.py`) for local experimentation and autonomous workflows.

## Inspiration: Boids -> 3 Rules

Boids showed that complex flocking can emerge from a few local rules.
Mantis takes the same design principle for agent autonomy:

1. `ATTEND`: capture signals (user input, tools, files, autonomous prompts).
2. `ASSOCIATE`: connect the signal to memory and context.
3. `ACT`: decide and execute the next concrete step.

Autonomy comes from this loop running continuously across many event sources, not from a single giant planner pass.

## What It Does (Current)

- Hybrid memory:
  - Chroma vectors (`MEMORY_DIR`)
  - SQLite FTS5 (`MEMORY_DIR/fts.db`)
  - append-only markdown log (`.agent/MEMORY.md`)
- Event-driven runtime with per-source FIFO lanes
- Autonomous heartbeat prompts (`AUTONOMOUS_INTERVAL_SEC`)
- Scheduled tasks from `tasks.md` (daily/weekly natural-language schedules)
- Command execution with:
  - sync and async paths
  - safety checks for unsafe/incomplete commands
  - one-shot auto-repair on syntax-like failures
  - outcome tracking in SQLite (`command_outcomes` table)
- Shared shell journal awareness via `.agent/shell.log` + `.bashrc` hook
- Filesystem watcher ingestion
- Web tools: `SEARCH` (tiered DDG) and `FETCH`
- Linux computer-use tools: `SCREENSHOT`, `CLICK`, `TYPE`
- Skills loaded from local files or URLs (`.agent/skills/*.md`)
- Optional Discord bridge (inbound prompts + outbound agent/activity feed)

## Quick Start

### 1) Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 2) Configure environment

Create `.env` (optional but recommended), example:

```bash
LLM_BASE=http://localhost:8001/v1
MODEL=Qwen2.5-14B-Instruct-Q4_K_M.gguf
SOUL_PATH=soul.md
```

### 3) Run Mantis

```bash
python3 mantis.py
```

Exit with `Ctrl+C`, or type `exit` / `quit`.

## How To Use

### Interactive use

Start `mantis.py`, then type natural language requests at the `you:` prompt.
Mantis can answer directly or choose tool directives internally (`COMMAND`, `READ`, `WRITE`, etc.).

Output prefixes:
- `agent:` direct reply to user input
- `[mantis]:` autonomous or non-user-triggered streams (`autonomous`, `shell`, `search`, `skill`, `discord`, etc.)

### Autonomous loop

Every `AUTONOMOUS_INTERVAL_SEC` (default `300`), Mantis self-prompts across rotating themes:
- unfinished work
- system/todo checks
- memory synthesis
- reminders/open questions
- curiosity search

Time-of-day modifiers are UTC-based (`Good morning.` / `End of day check.`).

### Scheduled tasks (`tasks.md`)

`tasks.md` is parsed at runtime and checked every minute. Supported schedule patterns:
- `daily 8am`
- `daily 14:30`
- `weekly sunday 10am`

If `tasks.md` does not exist, Mantis creates a default one on boot.

### Skills

Skills live in `.agent/skills/*.md` and are injected into the system prompt (truncated per skill).
They can be loaded via:
- local path: `SKILL: .agent/skills/my_skill.md`
- URL: `SKILL: https://.../my_skill.md`

Skill updates are guarded: removing previously known commands requires repeated failure evidence (`MIN_FAILURES_BEFORE_SKILL_UPDATE`).

### Discord mode (optional)

Set `DISCORD_TOKEN` and `DISCORD_CHANNEL_ID` to enable:
- inbound channel messages -> Mantis events
- outbound replies and activity events
- special phrase: `approve soul` triggers a soul-write instruction

## Runtime Model

For each event:
1. `ATTEND`: enqueue event with source + timestamp
2. `ASSOCIATE`: recall related memory and persist new event
3. `ACT`: call the LLM and execute optional tool directives

Events are serialized per source lane (FIFO per source), improving concurrency without cross-source race conditions.

## Tool Directives

The agent can emit:
- `COMMAND: <shell command>`
- `READ: <filepath>`
- `WRITE: <filepath>` followed by full file content
- `SCREENSHOT: <filepath>`
- `CLICK: <x> <y>`
- `TYPE: <text>`
- `SEARCH: <query>`
- `FETCH: <url>`
- `SKILL: <url-or-path>`

Tool results are fed back into the event bus and memory.

## Shell Execution Model

Mantis executes commands by writing a temporary bash script and running it non-interactively.

- Short commands: synchronous execution
- Long/install-like commands (for example `pip install`, `npm install`, `apt install`, `git clone`, multiline scripts): async execution
- Interactive commands are blocked or replaced with safe alternatives where configured
- Unsafe/incomplete command fragments are skipped

Mantis also logs command/result pairs to `SHELL_LOG`.

## System Requirements

- Python `3.10+`
- OpenAI-compatible API:
  - `POST /v1/chat/completions`
  - optional `POST /v1/embeddings` when using LLM embeddings
- Python packages in `requirements.txt`

System tools used by some features:
- `curl`
- `scrot` (screenshots)
- `xdotool` (mouse/keyboard automation)
- Playwright Firefox runtime (`playwright install firefox`)

`mantis.py` attempts setup for some Linux tools on startup; manual install may still be required.

## Configuration

Environment variables are loaded from `.env` via `python-dotenv`.

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE` | `http://localhost:8001/v1` | Base URL for LLM API |
| `MODEL` | `Qwen2.5-14B-Instruct-Q4_K_M.gguf` | Model name sent to API |
| `MEMORY_DIR` | `.agent/memory` | Chroma and SQLite storage path |
| `SOUL_PATH` | `SOUL.md` | Soul prompt file |
| `TOP_K` | `4` | Recall depth per retrieval source |
| `MAX_TOKENS` | `512` | Completion max tokens |
| `MAX_LLM_TIMEOUT` | `120` | LLM request timeout (seconds) |
| `EMBEDDING_BACKEND` | `hash` | `hash`, `llm`, or `sentence-transformers` |
| `AUTONOMOUS_INTERVAL_SEC` | `300` | Autonomous heartbeat interval |
| `WATCH_PATH` | `.` | Filesystem watcher root |
| `MAX_HISTORY` | `10` | Messages kept in rolling history |
| `SHELL_LOG` | `.agent/shell.log` | Shared shell journal path |
| `MAX_AUTO_REPAIR_ATTEMPTS` | `1` | Max automatic command repair retries |
| `MIN_FAILURES_BEFORE_SKILL_UPDATE` | `3` | Evidence threshold for removing skill commands |
| `SKILL_UPDATE_WINDOW` | `50` | Outcome window size for skill update gating |
| `MAX_PROMPT_CHARS` | `24000` | Total prompt budget before compaction |
| `MAX_SYSTEM_CHARS` | `12000` | System message char cap during compaction |
| `MAX_HISTORY_MSG_CHARS` | `3000` | Per-history-message char cap |
| `MAX_USER_INPUT_CHARS` | `4000` | User input char cap in prompt |
| `DISCORD_TOKEN` | `` | Discord bot token (optional) |
| `DISCORD_CHANNEL_ID` | `0` | Target Discord channel ID (optional) |
| `DISCORD_ACTIVITY_FEED` | `true` | Enables Discord activity events |

## Soul File Note

This repo currently has `soul.md` (lowercase), while default config points to `SOUL.md`.
On case-sensitive filesystems, set:

```bash
SOUL_PATH=soul.md
```

## Repository Layout

- `mantis.py` - runtime
- `soul.md` - soul prompt/instructions
- `tasks.md` - scheduled autonomous tasks
- `requirements.txt` - Python dependencies
- `docs/assets/` - logo/startup art

## Security Notes

This runtime is intentionally permissive:
- shell commands run as the current OS user
- file writes are unconstrained within process permissions
- network calls are possible through tool directives
- no internal sandbox or policy engine is enforced in `mantis.py`

Use in controlled environments.

## Inspiration

- OpenClaw: https://github.com/OpenClaw/OpenClaw
- PicoClaw: https://github.com/sipeed/picoclaw
