# SDLC PROGRESSOR CRON

**Purpose:** Execute the next task in the current directory.
**Frequency:** Every 5 minutes (or on-demand)
**State File:** `workflows/SDLC/FOLDER_TODO_TRACKER.json`

---

## LOGIC

```
1. Read FOLDER_TODO_TRACKER.json
2. IF current_directory is null:
   → SKIP (Starter cron handles it)
   → Report: "No directory in progress. Waiting for Starter."
3. IF current_directory is set:
   a. Read RDC outputs from docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/<current_directory>/
   b. Find current phase's PHASE_N_TODO.md
   c. Find next pending task in that phase
   d. IF task found:
      - Execute task (implement code, run tests)
      - Update tracker: tasks_done++
      - Report: "TASK DONE: <task>. Phase X: Y/Z tasks."
   e. IF phase complete:
      - phases_done++
      - current_phase++
      - IF more phases: continue
      - IF all phases done:
        * Set directory.status = "DONE"
        * Set directory.completed_at = now()
        * Set current_directory = null
        * Update summary counts
        * Report: "DIRECTORY COMPLETE: <name>. Progress: X/35."
```

---

## INVOCATION PROMPT

```
SDLC PROGRESSOR — Execute next task

**CRITICAL: Python 3.13 Required** — Use `uv run python` for ALL Python commands.

1. Read workflows/SDLC/FOLDER_TODO_TRACKER.json
2. If current_directory is null: Report "NO WORK. Waiting for Starter." and stop
3. If current_directory is set:
   a. Read docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/<current_directory>/PHASE_<N>_TODO.md
   b. Find next pending task
   c. Execute ONE task:
      - DEV: Implement the code
      - TEST: Run tests with `uv run python -m pytest <path> -v`
      - QA: Syntax check with `uv run python -m py_compile <file>`
   d. Update FOLDER_TODO_TRACKER.json:
      - Increment tasks_done
      - If phase complete: increment phases_done, advance current_phase
      - If all phases complete: set status=DONE, clear current_directory
   e. Report: "TASK: <name>. Phase: X/Y. Directory: <name>. Overall: Z/35."

Reference: workflows/SDLC/DIRECTORY_SDLC_PROMPT.md for full instructions.
```

---

*Created: 2026-05-23*
