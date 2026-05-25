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
