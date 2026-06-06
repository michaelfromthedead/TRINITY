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

### CRON Job

- **Schedule:** Every 5 minutes (`*/5 * * * *`)
- **Type:** Durable, recurring
- **Spawn if busy:** YES — spawn new work thread even if current worker exists
- **Auto-expires:** 2026-06-09

### Mission

Fix NEEDS_WORK directories from MEGA_PYTHON_REPORT_V2 until all reach GREEN_LIGHT (>99%):
- Read `docs/PYTHON_DOCS/MEGA_PYTHON_REPORT_V2.md` for current status
- For each NEEDS_WORK directory: DIAGNOSE → FIX → VERIFY
- Mark GREEN_LIGHT when >99% pass rate achieved

### Pipeline Per Directory

```
DIAGNOSE (find failures) → FIX (code fixes) → VERIFY (run tests) → GREEN_LIGHT
```

### Current Work (2026-06-02)

**PYTHON_DOCS SDLC: 27/35 GREEN_LIGHT (77%)**

| Priority | Directory | Pass% | Issue |
|----------|-----------|-------|-------|
| P2 | dialogue_dsp | 85.9% | DSP time effects |
| P2 | mixing_spatial | 95.1% | Mixer RMS |
| P3 | crowds_facial | 98.8% | FaceCaptureRetargeter |
| P3 | abilities_ai_camera | 99.0% | Camera edge cases |
| P3 | world | 99.0% | Phase1 verification |
| P3 | trinity_decorators_part1 | 98.4% | ECS relation tests |
| P3 | trinity_descriptors | 99.8% | Version decoder |

**Tracker:** `docs/PYTHON_DOCS/MEGA_PYTHON_REPORT_V2.md`

### Rules

1. Max 8 agents (HARD LIMIT)
2. Sequential pipeline: DEV → TEST → QA
3. Background agents: `run_in_background: true`
4. Disk-first: Read state from files
5. On worker completion → spawn next worker immediately
6. **Spawn regardless of existing workers** — each cron tick spawns unblocked work

### On Cron Wake-Up

1. Read `docs/PYTHON_DOCS/MEGA_PYTHON_REPORT_V2.md` for NEEDS_WORK list
2. Pick lowest pass% directory that isn't being worked on
3. Spawn DIAGNOSE agent: run tests, identify failing test patterns
4. Spawn FIX agent: fix code based on diagnosis
5. Run VERIFY: `uv run pytest tests/<dir>/ -q --tb=short`
6. If >99%: mark GREEN_LIGHT. Else: re-run FIX cycle
