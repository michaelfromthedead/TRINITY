# PYTHON_DOCS SDLC CRON — Worker Pool

## STATUS: ACTIVE

**Created:** 2026-06-02
**Cron Job:** `ac74bcba`
**Schedule:** Every 5 minutes (`*/5 * * * *`)
**Directories:** 35 (engine/animation, engine/audio, engine/gameplay, etc.)

---

## PURPOSE

Execute SDLC_WORKFLOW for PYTHON_DOCS implementation tasks across 35 directories.
Many modules are already implemented — cron focuses on VERIFICATION (writing tests) and QA.

---

## TRACKER FILES

| File | Purpose |
|------|---------|
| `docs/PYTHON_DOCS/SDLC_TRACKER.json` | State tracker (directories, tasks, workers) |
| `docs/PYTHON_DOCS/INPROGRESS.md` | Human-readable progress log |
| `docs/PYTHON_DOCS/<dir>/PHASE_N_TODO.md` | Task acceptance criteria |
| `docs/PYTHON_DOCS/<dir>/PHASE_N_ARCH.md` | Architecture specs |

---

## PIPELINE PER TASK

```
┌─────────────────────────────────────────────────────────────┐
│ PRESTEP: Create branch task/T-XX-N.N, init INPROGRESS      │
├─────────────────────────────────────────────────────────────┤
│                           │                                 │
│                           ▼                                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ DEV: Code implementation (SKIP if code exists)         │ │
│ └─────────────────────────────────────────────────────────┘ │
│                           │                                 │
│                           ▼                                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ TEST_UNIT: WHITEBOX ∥ BLACKBOX (parallel)              │ │
│ │  - WHITEBOX: Full source access, internal tests        │ │
│ │  - BLACKBOX: Cleanroom, contract-only tests            │ │
│ └─────────────────────────────────────────────────────────┘ │
│                           │                                 │
│                           ▼                                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ QA_UNIT: JUNIOR → SANITY → FINAL (sequential)          │ │
│ │  - JUNIOR: Hypercritical, high-recall findings         │ │
│ │  - SANITY: Filter false positives                      │ │
│ │  - FINAL: Binding verdict                              │ │
│ └─────────────────────────────────────────────────────────┘ │
│                           │                                 │
│                           ▼                                 │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ VERDICT                                                 │ │
│ │  - GREEN_LIGHT → Mark [x], merge, next task            │ │
│ │  - FIX → Re-run DEV+TEST+QA (max 3 cycles)             │ │
│ │  - REWRITE → ARCH_DEV on main, fresh branch            │ │
│ │  - ESCALATE → Pause, report to human                   │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## VERIFICATION MODE

Most PYTHON_DOCS modules are **already implemented**. The cron prioritizes:

1. **Check if code exists** → Skip DEV stage
2. **Check if tests exist** → If not, spawn TEST_UNIT
3. **Run tests** → Verify acceptance criteria
4. **QA review** → Validate implementation quality

```bash
# Check implementation status
ls engine/animation/graph/*.py           # Code exists?
ls tests/animation/graph/test_*.py       # Tests exist?
uv run python -m pytest tests/animation/graph/ -v  # Tests pass?
```

---

## SPAWN PATTERNS

### DEV (skip if code exists)
```python
Agent(
    description="DEV T-XX-N.N",
    subagent_type="coder",
    run_in_background=True,
    prompt="SDLC DEV for T-XX-N.N. Read WORKER_DEV.md. CODE ONLY."
)
```

### TEST_UNIT (parallel)
```python
Agent(description="WHITEBOX T-XX-N.N", subagent_type="tester", ...)
Agent(description="BLACKBOX T-XX-N.N", subagent_type="tester", ...)
```

### QA_UNIT (sequential)
```python
Agent(description="JUNIOR T-XX-N.N", subagent_type="reviewer", ...)
# Wait for completion
Agent(description="SANITY T-XX-N.N", subagent_type="reviewer", ...)
# Wait for completion
Agent(description="FINAL T-XX-N.N", subagent_type="reviewer", ...)
```

---

## HARD RULES

1. **Max 8 agents** — HARD LIMIT (per CLAUDE.md)
2. **Pool size 4** — Leave headroom for other work
3. **Python 3.13** — Use `uv run python` for ALL commands
4. **Disk-first** — Read tracker before spawning
5. **Sequential QA** — JUNIOR → SANITY → FINAL, no shortcuts
6. **No fabrication** — Every result from actual command output
7. **Background agents** — `run_in_background: true` for all spawns

---

## CRON TICK LOGIC

```
1. Read SDLC_TRACKER.json
2. Count active workers in pool.active_workers
3. If >= 4 active: STOP (pool full)
4. Find current IN_PROGRESS directory
5. Find current task and stage
6. Determine next action based on stage
7. Spawn worker(s) to fill available slots
8. Update tracker with new workers
9. Report status
```

---

## DIRECTORY ORDER

Process in alphabetical order:
1. engine_animation_crowds_facial ✅
2. engine_animation_graph_ik 🔄
3. engine_animation_motionmatching_procedural
4. engine_animation_skeletal_systems
5. engine_audio_adaptive_core
... (35 total)

---

## REFERENCES

- `workflows/SDLC/SDLC_WORKFLOW.json` — Full state machine
- `workflows/SDLC/WORKER_*.md` — Role-specific instructions
- `workflows/SHARED/WORKER_PROTOCOL.md` — Non-negotiable rules

---

*Created: 2026-06-02*
*For: PYTHON_DOCS Implementation Verification*
