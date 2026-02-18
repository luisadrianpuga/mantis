# Mantis

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/mantis-logo.png">
    <img src="docs/assets/mantis-logo.png" alt="Mantis logo" width="500">
  </picture>
</p>

Mantis is a local-first Python agent runtime that implements a three-stage loop:
`ATTEND -> ASSOCIATE -> ACT`.

The system combines:
- OpenAI-compatible chat completion calls for reasoning
- hybrid memory retrieval (vector + keyword + markdown log)
- lightweight tool execution (`COMMAND`, `READ`, `WRITE`)
- optional autonomous periodic reflection
- optional filesystem event ingestion

## Scope

This repository currently provides a single executable runtime (`mantis.py`) intended for local experimentation and research prototyping. It is not packaged as a production service.

## Runtime Model

For each event, Mantis performs:
1. `ATTEND`: queue an event tagged with source and timestamp.
2. `ASSOCIATE`: retrieve related memories from three stores and persist the new event.
3. `ACT`: call the configured LLM and optionally execute one tool action.

### Event Sources

Supported event sources in the current implementation:
- `user` (interactive CLI input)
- `filesystem` (watchdog events)
- `autonomous` (timer-triggered prompts)
- `tool` (tool output fed back to loop)
- `agent_echo` (agent response persisted to memory)

### Memory Subsystems

Mantis persists and recalls memory through:
- ChromaDB persistent collection (`events`) for vector retrieval
- SQLite FTS5 table (`memories`) for keyword retrieval
- markdown append-only log at `.agent/MEMORY.md`

The retrieved context is deduplicated and provided to the LLM as "Relevant memory".

## Tool Interface

The model can emit at most one primary tool directive per response using strict markers:
- `COMMAND: <shell command>`
- `READ: <filepath>`
- `WRITE: <filepath>` followed by full file content

Tool results are re-injected as events (`source="tool"`) and influence subsequent turns.

## Requirements

- Python 3.10+
- A reachable OpenAI-compatible `/v1/chat/completions` endpoint
- Dependencies in `requirements.txt`

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python mantis.py
```

## Configuration

Configuration is loaded from environment variables (`.env` supported via `python-dotenv`).

Default values in code:

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE` | `http://localhost:8001/v1` | Base URL for chat completion API |
| `MODEL` | `Qwen2.5-14B-Instruct-Q4_K_M.gguf` | Model identifier sent in request payload |
| `MEMORY_DIR` | `.agent/memory` | ChromaDB and SQLite storage directory |
| `SOUL_PATH` | `SOUL.md` | System prompt file path |
| `TOP_K` | `4` | Retrieval depth per store |
| `MAX_TOKENS` | `512` | `max_tokens` for completion call |
| `EMBEDDING_BACKEND` | `hash` | `hash` or `sentence-transformers` |
| `AUTONOMOUS_INTERVAL_SEC` | `300` | Period for autonomous prompts |
| `WATCH_PATH` | `.` | Root path for filesystem watcher |

### Embedding Backend Notes

- `hash` backend is dependency-free and deterministic.
- `sentence-transformers` backend is supported in code but requires installing `sentence-transformers` separately (not included in `requirements.txt`).

## Execution Characteristics and Limits

- Tool command execution uses `subprocess.run(..., shell=True, timeout=30)`.
- File writes can target arbitrary paths accessible to the process.
- No authentication, sandboxing, or policy engine is implemented inside `mantis.py`.
- Error handling for the LLM call is minimal (`raise_for_status()` then parse JSON).

These properties make the current runtime suitable for controlled local environments, not untrusted multi-tenant deployment.

## Repository Layout

- `mantis.py`: main runtime
- `requirements.txt`: Python dependencies
- `docs/assets/`: static assets (logo, startup banner module)
- `soul.md`: example prompt text file in this repo (note: default lookup path in code is `SOUL.md` unless overridden)

## Scientific Writing and Evaluation Guidance

If this project is used for scientific workflows, treat the runtime as an experimental system. Recommended practice:
- state the exact commit hash and environment variables used in experiments
- fix model version and endpoint implementation for reproducibility
- log all tool invocations and side effects as part of experimental records
- report known threats to validity (prompt drift, retrieval noise, nondeterministic model outputs)
- separate exploratory qualitative findings from quantitative claims

## Inspiration

- OpenClaw: https://github.com/OpenClaw/OpenClaw
- PicoClaw: https://github.com/sipeed/picoclaw
