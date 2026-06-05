# WGPU SDLC CRON — Worker Pool

## ⛔ STATUS: COMPLETE — DO NOT EXECUTE ⛔

**Completed:** 2026-05-31
**Result:** 256/256 tasks GREEN_LIGHT across 7 phases
**This workflow is archived. No action needed on trigger.**

---

~~**Purpose:** Execute SDLC_WORKFLOW for WGPU implementation (256 tasks, 7 phases).~~
~~**Frequency:** Every 5 minutes (`*/5 * * * *`)~~
~~**Model:** Worker pool — 4 slots, any stage (DEV/TEST/QA).~~

---

## INVOCATION PROMPT

```
WGPU SDLC — Worker Pool Cycle

**HARD CAP: 8 agents max. Never exceed.**
**DISK-FIRST: Read tracker, count running agents, verify before reporting.**

1. READ docs/WGPU_DOCS/RDC_OUTPUT/WGPU_SDLC_TRACKER.md
2. READ current PHASE_N_TODO.md for active phase
3. IDENTIFY next unblocked task (check prereqs are [x] DONE)
4. DETERMINE stage needed:
   - No in-flight work → spawn DEV
   - DEV done → spawn WHITEBOX + BLACKBOX (parallel)
   - TEST done → spawn JUNIOR_QA
   - JUNIOR done → spawn SANITY
   - SANITY done → spawn FINAL
   - FINAL verdict → execute outcome (GREEN_LIGHT/FIX/REWRITE/ESCALATE)
5. SPAWN worker with role-specific prompt
6. UPDATE tracker with in-flight task
7. REPORT: "WGPU SDLC: Task T-WGPU-P<N>.<X>.<Y> — Stage <STAGE> — Worker spawned"

Reference: workflows/SDLC/CRON_WGPU_SDLC.md
```

---

## SPAWN PATTERNS

### DEV (1 slot)
```
Agent(description:"DEV T-WGPU-P1.X.Y", name:"dev-wgpu-p1xy", subagent_type:"coder",
  run_in_background:true, prompt:"
SDLC DEV for T-WGPU-P1.X.Y

Read:
- workflows/SDLC/WORKER_DEV.md
- workflows/SHARED/WORKER_PROTOCOL.md
- docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_ARCH.md
- docs/WGPU_DOCS/RDC_OUTPUT/PHASE_1_CORE_TODO.md (task T-WGPU-P1.X.Y)

CODE ONLY. No tests. Deliverable: <deliverable from TODO>
Acceptance: <criteria from TODO>
")
```

### WHITEBOX + BLACKBOX (2 slots, parallel)
```
Agent(description:"WHITEBOX T-WGPU-P1.X.Y", name:"wb-wgpu-p1xy", subagent_type:"tester",
  run_in_background:true, prompt:"
SDLC WHITEBOX for T-WGPU-P1.X.Y. Full source access.
Read: workflows/SDLC/WORKER_TESTDEV_WHITEBOX.md, DEV output.
Write tests exercising internals, edge cases, code paths.
")
Agent(description:"BLACKBOX T-WGPU-P1.X.Y", name:"bb-wgpu-p1xy", subagent_type:"tester",
  run_in_background:true, prompt:"
SDLC BLACKBOX for T-WGPU-P1.X.Y. CLEANROOM — cannot read implementation.
Read: workflows/SDLC/WORKER_TESTDEV_BLACKBOX.md, PHASE_N_ARCH.md, task TODO only.
Write tests from public contract only. Do NOT read DEV source files.
")
```

### QA_UNIT (3 slots, sequential)
```
Agent(description:"JUNIOR T-WGPU-P1.X.Y", name:"junior-wgpu-p1xy", subagent_type:"reviewer",
  run_in_background:true, prompt:"
SDLC JUNIOR_QA for T-WGPU-P1.X.Y. Hypercritical stance.
Read: workflows/SDLC/WORKER_QA_JUNIOR.md, DEV code, both tests.
Produce: findings list with severity (Critical/High/Medium/Low).
")
```

---

## TASK DEPENDENCY RULES

Before spawning DEV for a task:
1. Check prereqs in PHASE_N_TODO.md
2. Verify prereq tasks are marked [x] in tracker
3. If prereqs not done → skip to next unblocked task
4. If no unblocked tasks in phase → check if phase complete → advance to next phase

---

## PHASE ORDER

```
PHASE_1_CORE → PHASE_2_RESOURCES → PHASE_3_PIPELINES → 
PHASE_4_SYNCHRONIZATION → PHASE_5_RAY_TRACING → 
PHASE_6_ADVANCED → PHASE_7_INTEGRATION
```

---

## VERDICTS

| Verdict | Action |
|---------|--------|
| GREEN_LIGHT | Mark task [x], update tracker, advance to next task |
| FIX | Re-spawn DEV+TEST with findings, re-run QA_UNIT |
| REWRITE | ARCH_DEV amends, fresh branch, restart task |
| ESCALATE | Pause, report to human |

---

*Created: 2026-05-27*
*For: WGPU SDLC Implementation*
