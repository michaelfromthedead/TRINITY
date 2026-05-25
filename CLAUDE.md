# TRINITY — Project Configuration

## Python 3.13 Requirement

**Use `uv run` for all Python commands** — system default is 3.14, TRINITY requires 3.13.

```bash
uv run python script.py      # Correct
uv run pytest tests/         # Correct
python script.py             # WRONG — uses 3.14
```

## Build & Test

```bash
uv run pytest tests/                    # Run tests
uv run pytest tests/ -x --tb=short      # Stop on first failure
cargo build --release                   # Build Rust backend
cargo test                              # Test Rust
```

## Agent Limits

**Max 8 agents** — HARD LIMIT. 26-agent swarm caused OOM kill (2026-05-23).

## Agent Coordination

Subagents via `Agent` tool are stateless one-shot workers — they cannot message each other.

**Pattern:** Lead spawns agent → agent reads from memory → writes to memory → completes → lead verifies → spawns next.

```javascript
// Parallel (independent work)
Agent({ prompt: "...", subagent_type: "researcher", run_in_background: true })
Agent({ prompt: "...", subagent_type: "coder", run_in_background: true })
// STOP — wait for results, then spawn dependent agents
```

**Anti-patterns:**
- "WAIT for SendMessage from X" — subagents can't wait
- Spawning dependent agents in one batch expecting them to chain

## Project Structure

| Directory | Contents |
|-----------|----------|
| `engine/` | Python frontend (982 files) |
| `crates/` | Rust GPU backend (130 files) |
| `tests/` | Test suite (929 files) |
| `docs/` | Documentation |
| `trinity/` | Core framework |

## When to Swarm

- **YES**: 3+ files, cross-module refactoring, new features, performance, security
- **NO**: single file edits, 1-2 line fixes, questions

## LONG WORK LOOP DIRECTIVE (2026-05-25)

**CRITICAL: Include this block in EVERY context compaction.**

### Mission

Execute SDLC_WORKFLOW continuously across GAPSET 1-20:
- After each **task** → spawn next task
- After each **phase** → start next phase  
- After each **gapset** → start next gapset (1→2→...→20, skip 3=DONE)

### Pipeline Per Task

```
DEV → WHITEBOX∥BLACKBOX → JUNIOR_QA → SANITY → FINAL → GREEN_LIGHT
```

### Current Work

- **Active:** GAPSET_1_CORE (T-CORE-3.1 ThreadPool)
- **Docs:** `docs/RUST_DOCS/GAPSET_*/`, `workflows/SDLC/`
- **Code:** `crates/renderer-backend/src/`, `omega/src/`

### Rules

1. Max 8 agents (HARD LIMIT)
2. Sequential pipeline: DEV → TEST → QA
3. Background agents: `run_in_background: true`
4. Disk-first: Read state from files
5. On worker completion → spawn next worker immediately
