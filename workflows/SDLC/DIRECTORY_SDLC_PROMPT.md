# DIRECTORY SDLC WORKER — Generic Prompt

**Purpose:** Execute SDLC workflow on a single RDC-processed directory.
**Input:** Directory name from the tracker (e.g., `engine_rendering_framegraph`)
**Output:** Implemented code according to PHASE_N_TODO.md specifications

---

## USAGE

```
DIRECTORY SDLC WORKER — Process: <DIRECTORY_NAME>
```

Replace `<DIRECTORY_NAME>` with the target directory from FOLDER_TODO_TRACKER.json.

---

## PHASE 0: KNOWLEDGE INJECTION (MANDATORY)

Before ANY work, read these files IN FULL:

### Project Context
1. **docs/specs/TRINITY_LATEST.md** — Full project knowledge
2. **CLAUDE.md** — Development rules and Python 3.13 requirement

### Directory-Specific RDC Outputs
3. **docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/<DIRECTORY_NAME>/PROJECT.md** — Scope, goals, constraints
4. **docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/<DIRECTORY_NAME>/CLARIFICATION.md** — Design philosophy
5. **ALL PHASE_N_ARCH.md files** in the directory — Architecture decisions
6. **ALL PHASE_N_TODO.md files** in the directory — Implementation tasks

---

## PHASE 1: INVENTORY

1. List all PHASE_N_TODO.md files in the directory
2. Count total tasks across all phases
3. Identify dependencies between phases (Phase N+1 may depend on Phase N)
4. Report: "Found N phases with M total tasks"

---

## PHASE 2: EXECUTE PHASES SEQUENTIALLY

For each PHASE_N_TODO.md (in order 1, 2, 3, ...):

### 2a. Read Phase Context
- Read PHASE_N_ARCH.md for architecture decisions
- Read PHASE_N_TODO.md for task list

### 2b. For Each Task in the Phase

1. **Understand** — Read the task description and acceptance criteria
2. **Locate** — Find the target file(s) in the codebase
3. **Implement** — Write/modify code to satisfy the task
4. **Test** — Run tests if specified: `uv run python -m pytest <path> -v`
5. **Verify** — Check acceptance criteria are met
6. **Mark Done** — Update tracker with task completion

### 2c. Phase Completion Check
- All tasks in phase complete?
- Tests pass?
- If yes: Move to next phase
- If blocked: Report blocker and stop

---

## PHASE 3: UPDATE TRACKER

After completing work on the directory:

1. Update `workflows/SDLC/FOLDER_TODO_TRACKER.json`:
   - Set directory status to appropriate value
   - Record phases completed
   - Record tasks completed
   - Add timestamp

2. Report completion:
```
DIRECTORY SDLC COMPLETE: <DIRECTORY_NAME>
Phases: N/N complete
Tasks: M/M complete
Tests: X passed, Y failed
Status: GREEN_LIGHT | PARTIAL | BLOCKED
Next: <next_pending_directory or ALL_DONE>
```

---

## HARD RULES

1. **Python 3.13** — Use `uv run python` for ALL Python commands
2. **Sequential phases** — Complete Phase N before starting Phase N+1
3. **No fabrication** — Only implement what's specified in TODO files
4. **Test everything** — Run tests after each significant change
5. **Track progress** — Update FOLDER_TODO_TRACKER.json after each session

---

## TASK STATES

| State | Meaning |
|-------|---------|
| `PENDING` | Not started |
| `IN_PROGRESS` | Currently being worked on |
| `DONE` | Complete and verified |
| `BLOCKED` | Cannot proceed, needs resolution |
| `SKIPPED` | Intentionally skipped (document why) |

---

## DIRECTORY STATES

| State | Meaning |
|-------|---------|
| `PENDING` | No work started |
| `IN_PROGRESS` | At least one phase started |
| `PARTIAL` | Some phases complete, others pending |
| `DONE` | All phases and tasks complete |
| `BLOCKED` | Cannot proceed, needs resolution |

---

## EXAMPLE INVOCATION

```
DIRECTORY SDLC WORKER — Process: engine_rendering_framegraph

Read RDC outputs from docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/engine_rendering_framegraph/
Execute all PHASE_N_TODO tasks sequentially.
Update tracker when complete.
```

---

*Created: 2026-05-23*
*For: Post-RDC Implementation Workflow*
