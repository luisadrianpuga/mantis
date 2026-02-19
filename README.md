<<<<<<< HEAD
# Mantis — Personal AI Runtime

<p align="center">
  <img src="docs/assets/mantis-logo.png" alt="Mantis Logo" width="420"/>
</p>

<p align="center">
  <strong>AUTOMATE • THINK • EXECUTE</strong>
</p>

---

**Mantis** is a minimal, extensible personal AI runtime for autonomous task execution using **OpenAI, Claude, or local LLMs** with vector memory and tool orchestration.

Bring your own model. Run your own agent. Own your AI stack.

---

## Features

* OpenAI-compatible API
* Works with **OpenAI, Claude, or Local LLMs**
* Autonomous agent loop
* Planner-first project execution for repo-level goals
* Git-native dev workflow (branch, diff, commit)
* Tool orchestration system
* Vector memory (RAG)
* Runs fully local if desired
* Minimal, hackable architecture

---

## Quick start
```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8001
```

Then connect the React UI to `http://localhost:8001/v1/chat/completions`.
Open docs at `http://localhost:8001/docs`.

Defaults:
- Provider auto-detection order: OpenAI -> Anthropic -> Ollama
- Optional explicit model override: `MANTIS_MODEL`
- Vector memory path: `.mantis/chroma` (override with `MANTIS_CHROMA_DIR`)

Tool calls: the agent emits `TOOL: tool_name | input`.

Runtime toolset includes:
- Workspace awareness: `workspace.tree`, `workspace.read_file`, `workspace.search`, `workspace.snapshot`
- Code editing: `create_file`, `write_file`, `patch_file`, `delete_file`
- Validation: `run_tests` (auto-detects `pytest`, `npm test`, `make test`)
- Git tools: `git.status`, `git.create_branch`, `git.diff`, `git.checkout`, `git.log`, `git.commit`
- Utility: `python`, `http`

Safety controls:
- `MANTIS_DEV_MODE=true` enables planner-driven project loops.
- `MANTIS_ALLOW_FILE_WRITE=false` by default disables file mutation tools.
- `MANTIS_SANDBOX=true` restricts file writes to `./workspace/`.
- All successful file mutations are logged to `.mantis/changes/`.
- Resumable plans are persisted in `.mantis/plans/`.
- Task queue is persisted in `.mantis/queue.json`.
- Execution metrics are appended to `.mantis/metrics.jsonl`.
- Worker includes autonomous repo task discovery and enqueues goals without user prompts.

Try this to validate loop + tools:
```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mantis-local-agent","messages":[{"role":"user","content":"fetch https://example.com and summarize it"}]}'
```

## Install
```bash
git clone <repo>
cd mantis
cp .env.example .env
bash scripts/install.sh
python mantis.py chat
```

## Cloud Mode (No Ollama)
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
python mantis.py api
```

## Local Mode (Ollama)
```bash
ollama serve
ollama pull qwen2.5:7b-instruct
python mantis.py worker
```

## Hybrid Mode
Mantis routes by model prefix:
- `gpt*` -> OpenAI
- `claude*` -> Anthropic
- anything else -> Ollama

You can run all providers together and select per-request with the `model` field.

---

## Why Mantis exists

Most AI products today are:

* APIs
* chat apps
* hosted platforms

Mantis is a **personal AI runtime**.

Software that sits between:

* you
* your tools
* your data
* your models

Goals:

* Bring Your Own LLM (OpenAI / Claude / Local)
* Avoid vendor lock-in
* Run locally and privately
* Learn how agent systems actually work
* Provide a clean reference architecture for builders
=======
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
>>>>>>> origin/rewrite/autonomy

## Scope

This repository currently provides a single executable runtime (`mantis.py`) intended for local experimentation and research prototyping. It is not packaged as a production service.

<<<<<<< HEAD
```mermaid
flowchart TB

User[User / Apps]

subgraph API Layer
API[OpenAI-Compatible API]
end

subgraph Mantis Runtime
Agent[Agent Runtime Loop]
Prompt[Prompt Builder]
Memory[Vector Memory]
Tools[Tool Router]
end

subgraph Model Providers
LLM[OPENAI • Claude • Local LLM]
end

subgraph Memory Layer
VectorDB[(Vector Database)]
end

subgraph Tools
Python[Python Executor]
Browser[Browser Automation]
HTTP[HTTP / APIs]
end

User --> API
API --> Agent

Agent --> Prompt
Agent --> Memory
Agent --> Tools

Prompt --> LLM
Memory --> VectorDB

Tools --> Python
Tools --> Browser
Tools --> HTTP
=======
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
>>>>>>> origin/rewrite/autonomy

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

<<<<<<< HEAD
<p>Mantis is designed as a personal AI runtime.
The system sits between users, tools, data, and model providers, enabling autonomous task execution with full control over models and integrations.</p>

<a href='docs/diagrams/FULL_UML_Diagram.mermaid'>Full diagram</a>

---
=======
Run:
>>>>>>> origin/rewrite/autonomy

```bash
python mantis.py
```

## Configuration

<<<<<<< HEAD
1. **Request enters the API**
   Messages from UI, CLI, or integrations are received via an OpenAI-compatible endpoint.

2. **Context is assembled**
   The runtime gathers recent conversation and retrieves relevant long-term memories from the vector database.

3. **The agent decides the next action**
   The LLM chooses whether to:
=======
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
>>>>>>> origin/rewrite/autonomy

### Embedding Backend Notes

<<<<<<< HEAD
4. **Tools execute real actions**
   Mantis can run code, fetch data, or automate tasks and feed results back into the loop.
=======
- `hash` backend is dependency-free and deterministic.
- `sentence-transformers` backend is supported in code but requires installing `sentence-transformers` separately (not included in `requirements.txt`).
>>>>>>> origin/rewrite/autonomy

## Execution Characteristics and Limits

<<<<<<< HEAD
This transforms a chat model into a persistent, tool-using personal AI runtime.

---

## Bring Your Own LLM

Mantis is model-agnostic.

Use whichever provider you prefer:

* OpenAI
* Anthropic Claude
* Ollama / Local GGUF models
* LM Studio
* Future providers

Your runtime. Your model. Your choice.
=======
- Tool command execution uses `subprocess.run(..., shell=True, timeout=30)`.
- File writes can target arbitrary paths accessible to the process.
- No authentication, sandboxing, or policy engine is implemented inside `mantis.py`.
- Error handling for the LLM call is minimal (`raise_for_status()` then parse JSON).

These properties make the current runtime suitable for controlled local environments, not untrusted multi-tenant deployment.

## Repository Layout
>>>>>>> origin/rewrite/autonomy

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

<<<<<<< HEAD
Mantis is inspired by:

* OpenClaw
  [https://github.com/OpenClaw/OpenClaw](https://github.com/OpenClaw/OpenClaw)

* PicoClaw
  [https://github.com/sipeed/picoclaw](https://github.com/sipeed/picoclaw)

These projects helped shape the modern personal-AI runtime pattern.

---

## How Mantis differs

Mantis distills the core architecture down to its essentials.

Instead of a large feature-heavy platform, Mantis focuses on the smallest set of components required to build a personal AI runtime.

### What we kept

From OpenClaw:

* Autonomous agent loop
* Local-first workflow
* Tool orchestration

From PicoClaw:

* OpenAI-compatible API
* Modular tool mindset
* Gateway-friendly architecture

### What we simplified

Mantis prioritizes:

* clarity
* minimalism
* hackability
* learnability

It is a **reference implementation**, not a feature race.

---

## Project status

Currently in the **design and architecture phase**.

### Phase 1 — MVP

* OpenAI-compatible API
* Local LLM integration
* Agent loop
* Basic tools
* Vector memory

### Phase 2

* Browser automation
* Task scheduling
* Background agents

### Phase 3

* Plugin / skill system
* Multi-agent workflows
* Deployment options

---

## Planned modules

| Module | Responsibility                  |
| ------ | ------------------------------- |
| API    | OpenAI-compatible interface     |
| Agent  | Autonomous reasoning loop       |
| Tools  | Code execution and integrations |
| Memory | Vector storage and retrieval    |

---

## Contributing

Mantis is designed to be a learning-friendly open-source project.

Contributions welcome:

* architecture ideas
* documentation
* tooling
* early implementation

---

## License

MIT License
=======
- OpenClaw: https://github.com/OpenClaw/OpenClaw
- PicoClaw: https://github.com/sipeed/picoclaw
>>>>>>> origin/rewrite/autonomy
