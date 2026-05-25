# SDLC STARTER CRON

**Purpose:** Pick the next pending directory and start SDLC processing.
**Frequency:** Every 30 minutes (or on-demand)
**State File:** `workflows/SDLC/FOLDER_TODO_TRACKER.json`

---

## LOGIC

```
1. Read FOLDER_TODO_TRACKER.json
2. IF current_directory is NOT null:
   → SKIP (Progressor cron handles it)
   → Report: "Directory in progress: <name>. Skipping."
3. IF current_directory is null:
   → Find first directory with status == "PENDING"
   → IF found:
     a. Set current_directory = directory.name
     b. Set directory.status = "IN_PROGRESS"
     c. Set directory.started_at = now()
     d. Update summary counts
     e. Write FOLDER_TODO_TRACKER.json
     f. Report: "STARTED: <name>. Progress: X/35."
   → IF not found (all done):
     f. Report: "ALL DIRECTORIES COMPLETE. 35/35 done."
```

---

## INVOCATION PROMPT

```
SDLC STARTER — Check for work

1. Read workflows/SDLC/FOLDER_TODO_TRACKER.json
2. If current_directory is set: Report "IN PROGRESS: <name>" and stop
3. If no current_directory:
   - Find first PENDING directory
   - Set it to IN_PROGRESS
   - Set current_directory, current_phase=1, current_task=null
   - Update tracker
   - Report: "STARTED: <name>. Phases: N. Progress: X/35."
4. If no PENDING directories: Report "ALL COMPLETE: 35/35"
```

---

*Created: 2026-05-23*
