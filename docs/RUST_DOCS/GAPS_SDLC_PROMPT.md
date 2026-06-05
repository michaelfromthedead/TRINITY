# GAPS_SDLC_PROMPT — Spawn-First Model

**Trigger:** Cron every 10 minutes (`7,17,27,37,47,57 * * * *`)
**Rule:** EVERY cycle MUST spawn new DEV agents. Never skip. Never "pool is full." Always spawn.

---

## UNCONDITIONAL RULES

1. **Every cycle spawns DEVs.** If the current gapset has `[-]` or `[~]` tasks, spawn DEV agents for them. Target: 4-6 per cycle.
2. **Process completions AFTER spawning, not instead of.** Advance in-flight tasks to next stage, but only after new DEVs are spawned.
3. **Disk is truth.** Agent state cannot be tracked in memory. Check files on disk:
   - `[-]` count in gapset PHASE_N_TODO.md → tasks needing DEV
   - Files in `crates/renderer-backend/tests/` → completed TEST agents
   - Build health: `cargo check -p renderer-backend 2>&1 | tail -1`

---

## WORKFLOW PER INVOCATION

### Step 1 — Find current gapset
Check `docs/gap_sets/GAPS_SDLC_TODO.md`. First gapset NOT marked DONE. Skip GAPSET_3_BRIDGE (100% DONE).

### Step 2 — Count remaining work
```bash
grep -c '\[-\]' docs/gap_sets/<GAPSET>/PHASE_N_TODO.md
grep -c '\[~\]' docs/gap_sets/<GAPSET>/PHASE_N_TODO.md
```
If both are 0: mark gapset DONE in GAPS_SDLC_TODO.md, advance to next gapset, return to Step 2.

### Step 3 — SPAWN DEV AGENTS (MANDATORY)
Pick up to 6 unblocked `[-]` or `[~]` tasks. Spawn DEV agents. ALL in ONE message. `run_in_background: true`.

Each DEV prompt: read PHASE_N_TODO.md for the task, read PHASE_N_ARCH.md for architecture, read WORKER_DEV.md for role. CODE ONLY. No tests.

**If you spawn 0 agents, you have failed the cycle. Always spawn.**

### Step 4 — Process any recent completions
If prior-cycle DEV agents completed (their output files exist on disk):
- Run `cargo check` — if passes, advance to TEST_UNIT (spawn WHITEBOX + BLACKBOX)
- If TEST agents completed → advance to JUNIOR_QA
- If JUNIOR completed → advance to SENIOR_QA_SANITY
- If SANITY completed → advance to SENIOR_QA_FINAL
- If FINAL returned GREEN_LIGHT → toggle [x] in gapset PHASE_N_TODO.md

### Step 5 — Report
```
<GAPSET>: spawned N DEVs: <task ids>. X [-] and Y [~] remaining.
Advanced: Z threads to next stage. Build: PASS/FAIL.
```

---

## SPAWN PATTERNS

### DEV spawn
```
Agent(description:"DEV <TASK>", name:"dev-<task>", subagent_type:"general-purpose",
  run_in_background:true, prompt:"SDLC DEV for <TASK> in <GAPSET>. CODE ONLY. No tests.
  Read PHASE_N_TODO.md for task. Read PHASE_N_ARCH.md for architecture.
  Read WORKER_DEV.md for role. Produce code. Report with files changed.")
```

### WHITEBOX + BLACKBOX spawn (same message, both background)
```
Agent(description:"WHITEBOX <TASK>", name:"wb-<task>", subagent_type:"general-purpose",
  run_in_background:true, prompt:"WHITEBOX tests for <TASK>. Full source access. ...")
Agent(description:"BLACKBOX <TASK>", name:"bb-<task>", subagent_type:"general-purpose",
  run_in_background:true, prompt:"BLACKBOX tests for <TASK>. CLEANROOM. Cannot read src/. ...")
```

### QA spawns (sequential)
JUNIOR → SANITY → FINAL, each waiting for prior to complete.

---

## GAPSET ORDER
1→2→3(SKIP)→4→5→6→7→8→9→10→11→12→13→14→15→16→17→18→19→20
