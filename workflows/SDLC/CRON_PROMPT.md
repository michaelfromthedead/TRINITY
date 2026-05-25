# SDLC CRON SYSTEM — Session Bootstrap

**DO NOT MODIFY THIS FILE WITHOUT EXPLICIT USER PERMISSION.**
**DO NOT "IMPROVE" OR "ENHANCE" THE PROMPTS.**
**EXECUTE EXACTLY AS WRITTEN.**

---

## WHAT THIS IS

A **worker-pool cron system** that processes 35 directories through the SDLC_WORKFLOW. Instead of stage-gated single-worker crons, this system maintains a **pool of N parallel worker slots** and fills them on every tick with whatever work is available.

**Key principles:**
- **Fan-out (B):** All tasks within a phase spawn in parallel. DEV for all tasks at once. TEST pairs for all tasks at once.
- **Worker pool (D):** N slots (default 4). Crons fill empty slots — no stage gating, no waiting for one slow worker.
- **QA sequential:** JUNIOR → SANITY → FINAL remain sequential (data-dependent chain), but multiple directories can be in QA simultaneously.

---

## ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKER POOL (N=4 slots)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Cron tick (every 2 min) → count running workers → fill gaps     │
│                                                                  │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐                │
│  │ Slot 1 │  │ Slot 2 │  │ Slot 3 │  │ Slot 4 │                │
│  │ Dir A  │  │ Dir A  │  │ Dir B  │  │ Dir C  │                │
│  │ T1.1   │  │ T1.2   │  │ T1.1   │  │ T1.1   │                │
│  │ DEV    │  │ DEV    │  │ DEV    │  │ DEV    │                │
│  └────────┘  └────────┘  └────────┘  └────────┘                │
│                                                                  │
│  Per-task state: PENDING → DEV(running) → DEV(done) →            │
│  TEST(running) → TEST(done) → QA_JUNIOR → QA_SENIOR →            │
│  ACCEPTANCE                                                      │
│                                                                  │
│  QA chain: JUNIOR(done) → SANITY(running) → FINAL(pending)       │
│  Multiple directories process simultaneously                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## TRACKER SCHEMA (v2)

`workflows/SDLC/FOLDER_TODO_TRACKER.json` is restructured for per-task, per-directory tracking:

```json
{
  "version": "2.0.0",
  "pool": {
    "max_slots": 4,
    "active_workers": {}
  },
  "directories": [
    {
      "name": "engine_animation_crowds_facial",
      "status": "IN_PROGRESS",
      "phase": 1,
      "phase_count": 3,
      "qa_cycle_counter": 0,
      "rewrite_counter": 0,
      "tasks": [
        {"id": "T1.1", "stage": "DONE", "worker_id": null},
        {"id": "T1.2", "stage": "DONE", "worker_id": null},
        {"id": "T1.3", "stage": "TEST", "worker_id": "a23e4d2a", "spawned_at": "..."},
        {"id": "T1.5", "stage": "PENDING", "worker_id": null}
      ]
    }
  ]
}
```

### Task Stage States
| Stage | Meaning |
|-------|---------|
| `PENDING` | Not started |
| `DEV` | DEV worker running |
| `DEV_DONE` | DEV complete, ready for TEST |
| `TEST` | WHITEBOX+BLACKBOX running |
| `TEST_DONE` | Tests complete, ready for QA |
| `QA_JUNIOR` | JUNIOR_QA running |
| `QA_JUNIOR_DONE` | Findings produced |
| `QA_SENIOR` | SANITY+FINAL running |
| `QA_SENIOR_DONE` | Verdict emitted |
| `ACCEPTING` | Verdict being executed |
| `DONE` | Task complete |
| `BLOCKED` | Cannot proceed |

---

## STEP 1: CREATE THE 5 UNIFORM POOL-FILLER CRONS

All 5 crons share the same logic. They are staggered in time so a check happens every ~2 minutes.

```
CronCreate({
  cron: "0,10,20,30,40,50 * * * *",
  recurring: true,
  prompt: "SDLC WORKER POOL — Fill empty slots (every 10 min at :00)\n\nPython 3.13: use `uv run python` for ALL commands. Max workers: 4.\n\n1. Read workflows/SDLC/FOLDER_TODO_TRACKER.json\n2. Count currently running workers (tasks with stage=DEV|TEST|QA_JUNIOR|QA_SENIOR|ACCEPTING and worker_id set, plus pool.active_workers entries).\n3. If running >= max_slots (4): Report \"POOL FULL: N/4 slots active.\" and stop.\n4. If running < max_slots: scan for work to fill remaining slots.\n\nWORK SCAN (in priority order):\n\nA. ACTIVE DIRECTORIES — for each IN_PROGRESS directory:\n   a. Find tasks at stage PENDING → spawn DEV worker. Set stage=DEV, record worker_id.\n   b. Find tasks at stage DEV_DONE → spawn WHITEBOX+BLACKBOX pair. Set stage=TEST.\n   c. Find tasks at stage TEST_DONE → spawn JUNIOR_QA. Set stage=QA_JUNIOR.\n   d. Find tasks at stage QA_JUNIOR_DONE → spawn SANITY then FINAL. Set stage=QA_SENIOR.\n   e. Find tasks at stage QA_SENIOR_DONE → execute verdict. Set stage=ACCEPTING.\n      - GREEN_LIGHT: stage=DONE\n      - FIX: qa_cycle_counter++, stage=PENDING (re-enter), if >=3: BLOCKED\n      - REWRITE: rewrite_counter++, stage=PENDING (re-enter), if >=2: BLOCKED\n      - ESCALATE: stage=BLOCKED\n   f. Check phase completion: if all tasks DONE, advance phase. If all phases DONE, directory=DONE.\n\nB. START NEW DIRECTORY — if slots still available and no work found in active dirs:\n   - Find first PENDING directory, set IN_PROGRESS, create task list from PHASE_N_TODO.md\n   - Spawn DEV workers for all PENDING tasks (fan-out within phase)\n\nSPAWNING RULES:\n- DEV: subagent_type \"coder\", name \"dev-<dir>-<task>\", include WORKER_DEV.md + protocol + task TODO + ARCH\n- WHITEBOX+BLACKBOX: spawn in single message, both bg. Include WORKER_TESTDEV_*.md roles.\n- JUNIOR_QA: subagent_type \"reviewer\", name \"qa-junior-<dir>-<task>\"\n- SANITY+FINAL: sequential spawns, name \"qa-sanity-<dir>-<task>\" / \"qa-final-<dir>-<task>\"\n- BLACKBOX is CLEANROOM: list forbidden files explicitly in prompt\n\n5. Update tracker with all spawned worker_ids and new stages.\n6. Report: \"POOL: N/4 slots filled. Active: <dir:task:stage list>.\"\n\nReference: workflows/SDLC/SDLC_WORKFLOW.json, workflows/SDLC/WORKER_*.md, workflows/SHARED/WORKER_PROTOCOL.md"
})

CronCreate({
  cron: "2,12,22,32,42,52 * * * *",
  recurring: true,
  prompt: "SDLC WORKER POOL — Fill empty slots (every 10 min at :02)\n\n[SAME LOGIC AS CRON :00 ABOVE]\n\nPython 3.13. Max workers: 4. Read tracker. Count running. If full, stop. Else scan for work: active dirs first (DEV→TEST→QA_JUNIOR→QA_SENIOR→ACCEPTANCE), then start new directory. Spawn workers to fill slots. Update tracker. Report pool status."
})

CronCreate({
  cron: "4,14,24,34,44,54 * * * *",
  recurring: true,
  prompt: "SDLC WORKER POOL — Fill empty slots (every 10 min at :04)\n\n[SAME LOGIC AS CRON :00 ABOVE]\n\nPython 3.13. Max workers: 4. Read tracker. Count running. If full, stop. Else scan for work: active dirs first, then new directory. Spawn to fill slots. Update tracker. Report pool status."
})

CronCreate({
  cron: "6,16,26,36,46,56 * * * *",
  recurring: true,
  prompt: "SDLC WORKER POOL — Fill empty slots (every 10 min at :06)\n\n[SAME LOGIC AS CRON :00 ABOVE]\n\nPython 3.13. Max workers: 4. Read tracker. Count running. If full, stop. Else scan for work: active dirs first, then new directory. Spawn to fill slots. Update tracker. Report pool status."
})

CronCreate({
  cron: "8,18,28,38,48,58 * * * *",
  recurring: true,
  prompt: "SDLC WORKER POOL — Fill empty slots (every 10 min at :08)\n\n[SAME LOGIC AS CRON :00 ABOVE]\n\nPython 3.13. Max workers: 4. Read tracker. Count running. If full, stop. Else scan for work: active dirs first, then new directory. Spawn to fill slots. Update tracker. Report pool status."
})
```

---

## STEP 2: TRIGGER INITIAL FILL

After creating crons, run the pool fill manually to start:

```
SDLC POOL FILL — Initial

1. Read workflows/SDLC/FOLDER_TODO_TRACKER.json
2. If no active directories: find first PENDING, set IN_PROGRESS
3. Read PHASE_N_TODO.md, create task list in tracker
4. Fill pool slots with DEV workers for all tasks in the phase
5. Report pool status
```

---

## STEP 3: CONFIRM

```
SDLC WORKER POOL ACTIVE (B+D)
- Pool size: 4 slots
- Crons: 5 staggered checkers (every ~2 min)
- Job IDs: <list>
- Active directories: <count>
- Current workers: <N>/4
- Progress: X/35 directories
```

---

## HARD RULES

1. **Python 3.13** — Always use `uv run python` for ALL Python commands
2. **Spawn workers, don't do work** — Crons spawn via Agent tool, never implement directly
3. **Pool limit** — Max 4 concurrent workers. Don't exceed.
4. **Fan-out within phase** — DEV tasks spawn in parallel. TEST pairs spawn in parallel per task.
5. **QA sequential** — JUNIOR → SANITY → FINAL within each task. Multiple tasks/dirs can QA simultaneously though.
6. **No Rust** — All Rust/crates tasks translocated to RUST_BACKLOG.md
7. **Update tracker** — Record worker_id and stage on every transition
8. **Self-contained prompts** — Every worker gets role doc, protocol, TODO entry, ARCH, file paths
9. **Loop limits** — FIX: max 3. REWRITE: max 2. Exceeded → BLOCKED.

---

## TROUBLESHOOTING

### Pool is always full
Workers may be slow or stuck. Check active_workers for long-running entries. Manually check if a worker hung.

### Worker completed but stage didn't advance
When a worker's notification arrives, the cron that handles it updates the tracker. If you see a gap, manually advance the stage.

### "ALL COMPLETE: 35/35"
Done.

### Crons not firing
Session-only. Re-run this document on new sessions.

---

## CREATED

- **v1:** 2026-05-23 — 2-cron (Starter + Progressor)
- **v2:** 2026-05-23 — 5-cron stage-gated swarm
- **v3:** 2026-05-23 — 5-cron self-healing
- **v4:** 2026-05-23 — Worker pool (B+D): fan-out per phase + N-slot pool
- **Directories:** 35 | **Phases:** 143 | **Pool:** 4 slots | **Crons:** 5 staggered

---

**REMINDER: DO NOT MODIFY WITHOUT EXPLICIT USER PERMISSION.**
