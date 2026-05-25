# PHASE_4_PARALLEL_WALKTHROUGH — Parallel Sibling Execution

**Scenario:** Phase 4's T-04.1 (Executor) and T-04.2 (Optimizer) spawn in parallel after T-04.3 (Error Reporter) lands. One sibling PASSes; the other FAILs. Demonstrate aggregate verdict logic.

**Purpose:** Trace the PARALLEL_BATCH_UNIT semantics. Show how mixed sibling outcomes resolve.

---

## Setup

Phases 1-3 PHASE_GREEN_LIGHT. Phase 4 entered. T-04.3 (Error Reporter) just completed: TASK_PASS.

QUEEN now needs to spawn T-04.1 + T-04.2 (parallel siblings per `PHASE_MODEL.md`).

---

## QUEEN orchestration (T+3:15)

```
QUEEN: T-04.3 PASS at T+3:00. Workspace_manifest updated.
       T-04.1 and T-04.2 are parallel-with-siblings; collecting into PARALLEL_BATCH_UNIT.

       Spawning PHASE_TASK_UNIT(T-04.1)  ← in background, run_in_background=true
       Spawning PHASE_TASK_UNIT(T-04.2)  ← in background, run_in_background=true

       Awaiting both unit verdicts.
```

QUEEN's batch state:
```
batch_id: phase4_runtime_batch_1
units: [
  {id: "T-04.1", status: "running", started: T+3:15},
  {id: "T-04.2", status: "running", started: T+3:15}
]
aggregate_pending: true
```

QUEEN does NOT poll. Per `LANG_DEV_V2_WORKFLOW.json` and CLAUDE.md "ALWAYS use `run_in_background: true` ... After spawning, STOP — do NOT add more tool calls or check status." QUEEN waits for unit completion notifications.

---

## T-04.1 outcome (T+4:00)

PHASE_EXECUTOR for T-04.1:
- Reads `boss_level_1_executor_rules.md`
- Reads `<library>_decisions.json` for atom catalog
- Implements per-atom executor for all 11 pandas_mini atoms
- Writes BOSS_LEVEL_1/{executor.py, ...}
- e2e_test.py runs canonical input → produces correct DataFrame

PHASE_QA for T-04.1:
- All 11 atoms have executor implementations ✓
- e2e_test.py exit 0 ✓
- Trace collection works ✓
- Verdict: **TASK_PASS**

QUEEN updates batch state:
```
units: [
  {id: "T-04.1", status: "pass", completed: T+4:00},
  {id: "T-04.2", status: "running"}  ← still running
]
```

QUEEN waits for T-04.2.

---

## T-04.2 outcome (T+4:30)

PHASE_EXECUTOR for T-04.2:
- Reads `boss_level_2_optimizer_rules.md`
- Implements 5 optimization rules
- Writes BOSS_LEVEL_2/{optimizer.py, optimization_rules.py, cost_model.py, test_optimizer.py, correctness_suite.py}

PHASE_QA for T-04.2:
- test_optimizer.py: 32 tests pass ✓
- correctness_suite.py runs 20 plans:
  - Plans 1-19: `optimized.execute() == original.execute()` ✓
  - **Plan 20: `optimized` produces 4 rows; `original` produces 5 rows** ← golden rule violation
- The failing plan: `LoadCSV, Filter('a>0'), Sort('b'), Filter('c<10'), Head(5)` — Optimizer's filter combination rule combined the two Filters but reordered them incorrectly when one filter clause referenced a column that Sort modified.
- Critical finding: "Filter combination rule violates golden rule on plan #20. Output diff: original[5] != optimized[4]. Per `PHASE_04_CONTRACT.md#T-04.2`: 'Any golden-rule violation → TASK_FAIL_ESCALATE (no retry — this is correctness, not bug).'"
- Verdict: **TASK_FAIL_ESCALATE**

QUEEN updates batch state:
```
units: [
  {id: "T-04.1", status: "pass"},
  {id: "T-04.2", status: "fail_escalate"}  ← critical
]
```

---

## QUEEN aggregate (T+4:32)

QUEEN consults aggregate rule (per `LANG_DEV_V2_WORKFLOW.json` §units.PARALLEL_BATCH_UNIT.aggregate_rule):

> "All siblings TASK_PASS → batch PASS. Any sibling TASK_FAIL_ESCALATE → batch HOLD. Mix of PASS and FAIL_RETRY → wait for retries to settle, then aggregate."

T-04.1 PASS + T-04.2 ESCALATE → batch HOLD.

But: T-04.4 (Debugger) depends on T-04.1, NOT on T-04.2. T-04.1 is PASS. Should T-04.4 proceed?

Per dependency graph:
- T-04.4 has `depends_on: ["T-04.1"]` (per `phase_model.PHASE_4.tasks.T-04.4`)
- T-04.4 does NOT depend on T-04.2

So T-04.4 CAN spawn. But Phase 4 PHASE_GREEN_LIGHT requires ALL tasks PASS — T-04.2 ESCALATE blocks the phase verdict regardless of T-04.4 outcome.

QUEEN's choice: spawn T-04.4 anyway (so its work is not lost waiting for T-04.2 fix), OR pause now and wait for human intervention on T-04.2.

**Per workflow design:** spawn T-04.4 anyway. The phase verdict is computed at phase boundary, not at each task. Spawning T-04.4 in parallel with the human's debugging-T-04.2 work is a productive use of time.

```
QUEEN: T-04.2 TASK_FAIL_ESCALATE at T+4:32. Critical: golden rule violation on plan #20.
       Phase 4 verdict: PHASE_HOLD (cannot reach PHASE_GREEN_LIGHT until T-04.2 resolves).

       However, T-04.4 (Debugger) depends only on T-04.1 (PASS). Spawning T-04.4 to
       continue useful work in parallel with human intervention on T-04.2.

       Workflow status: PHASE_4_HOLD. Awaiting:
       - Human resolution of T-04.2 (Optimizer golden-rule violation)
       - T-04.4 completion (independent)
```

---

## T-04.4 outcome (T+5:30)

PHASE_EXECUTOR + PHASE_QA for T-04.4: TASK_PASS.

QUEEN updates batch state:
```
units: [
  {id: "T-04.1", status: "pass"},
  {id: "T-04.2", status: "fail_escalate"},
  {id: "T-04.4", status: "pass"}
]
phase_status: "hold"
```

T-04.4's outputs are committed. The Debugger is functionally complete.

The phase still HOLDs because T-04.2 has not resolved.

---

## Human intervention (T+5:45)

User reads QUEEN's report. Inspects T-04.2 PHASE_QA verdict and BOSS_LEVEL_2/optimization_rules.py.

Identifies: filter_combination rule's `combine()` function uses Python `dict.update()` to merge filter clauses, which loses ordering. When two filters reference columns affected by an intervening Sort, the combined filter runs before Sort instead of after.

Fix: filter_combination rule must check that no atom between the two filters modifies any column referenced by either filter. If such an atom exists, do NOT combine.

User edits BOSS_LEVEL_2/optimization_rules.py. Adds regression test to test_optimizer.py covering plan #20.

User: `re-engage T-04.2`

QUEEN re-spawns PHASE_EXECUTOR for T-04.2:
- `task_id: "T-04.2"`
- `prior_retry_findings: [golden-rule violation finding from prior verdict]`
- `prior_attempt_outputs: [BOSS_LEVEL_2/...]` (existing files; executor inspects + iterates)

**Note:** Per workflow, this is a `TASK_FAIL_ESCALATE` resumption, NOT a retry. The retry counter was already at 0 (TASK_FAIL_ESCALATE on critical findings doesn't allow retry). User intervention "unsticks" it by editing the workspace.

Actually per `LANG_DEV_V2_WORKFLOW.json` §loop_limits.task_retry_counter_semantics: TASK_FAIL_ESCALATE pauses the workflow. Resumption after human intervention restarts T-04.2 fresh (counter resets to 0).

PHASE_EXECUTOR for T-04.2 (resumed):
- Reads existing BOSS_LEVEL_2/ files (now containing user's fix)
- Runs correctness_suite.py: all 20 plans PASS this time ✓
- Returns completion report noting "outputs already exist; verified per acceptance"

PHASE_QA for T-04.2:
- correctness_suite.py: 20/20 PASS ✓
- test_optimizer.py: 33 tests pass (32 original + 1 new regression test) ✓
- Verdict: **TASK_PASS**

---

## Phase 4 verdict (T+6:15)

QUEEN updates batch state:
```
units: [
  {id: "T-04.1", status: "pass"},
  {id: "T-04.2", status: "pass"},  ← resolved after human intervention
  {id: "T-04.4", status: "pass"}
]
phase_status: "complete"
```

All 4 Phase 4 tasks PASS. QUEEN emits PHASE_GREEN_LIGHT for Phase 4. Advances to METHODOLOGY_INTEGRATION.

---

## Bookkeeping

`workspace_manifest.json` Phase 4 summary:
```json
{
  "phase": 4,
  "name": "RUNTIME",
  "status": "complete",
  "tasks": [
    {"task_id": "T-04.3", "status": "pass", "attempts": 1, "duration_min": 15},
    {"task_id": "T-04.1", "status": "pass", "attempts": 1, "duration_min": 45, "parallel_batch_id": "phase4_runtime_batch_1"},
    {"task_id": "T-04.2", "status": "pass", "attempts": 2, "duration_min": 75, "parallel_batch_id": "phase4_runtime_batch_1", "human_intervention": true, "intervention_at": "T+5:45", "intervention_summary": "Fixed filter_combination rule's column-overlap check"},
    {"task_id": "T-04.4", "status": "pass", "attempts": 1, "duration_min": 60}
  ],
  "completed_at": "T+6:15",
  "duration_min": 195,
  "parallelism_notes": "T-04.1 + T-04.2 ran in parallel (PARALLEL_BATCH); T-04.4 spawned during T-04.2 hold to preserve forward progress"
}
```

INPROGRESS.md prepended:
```
## 2026-04-19 T+6:15 — LANG_DEV_V2 Phase 4 PHASE_GREEN_LIGHT (with human intervention)

T-04.3: PASS (15min)
T-04.1: PASS (45min) — parallel with T-04.2
T-04.2: PASS after intervention (75min total) — golden-rule violation on filter_combination rule
        Critical finding at T+4:32; human fix at T+5:45; re-QA PASS at T+6:15
T-04.4: PASS (60min) — spawned during T-04.2 hold; depends only on T-04.1

Aggregate parallelism saved ~30min vs serial.
Human intervention saved methodology — golden rule violation would have caused
METHODOLOGY_INTEGRATOR to FAIL Gate 3 if uncaught.
```

---

## What this scenario teaches

**1. Parallel batches use dependency-graph-aware spawning.**
T-04.4 has only T-04.1 as a dependency, not T-04.2 or T-04.3. So T-04.4 can spawn during T-04.2's hold without violating dependency semantics. This makes the workflow resilient to single-task failures.

**2. Critical findings short-circuit retry.**
Golden-rule violations are not eligible for retry per the contract. The reasoning: a golden-rule violation isn't a "bug we need to debug" — it's a correctness violation that requires architectural reconsideration. Retry without architectural change would just re-violate.

**3. Human intervention is a first-class workflow concept.**
The workflow doesn't pretend humans are unnecessary. ESCALATE pauses; human edits; re-engagement resumes. Bookkeeping records the intervention.

**4. PARALLEL_BATCH_UNIT aggregate is conservative.**
Even though T-04.1 and T-04.4 PASSed, the phase HOLDs because T-04.2 ESCALATEd. The phase verdict is "all-or-nothing" — a partially-passing phase is not a passing phase.

**5. "Save useful work" is implicit policy.**
QUEEN spawning T-04.4 during T-04.2's hold means the Debugger work isn't lost. If QUEEN had paused entirely on T-04.2 ESCALATE, T-04.4 work would queue behind T-04.2 resolution unnecessarily. The workflow encodes the productive default.

---

## Edge case: both siblings ESCALATE

If T-04.1 AND T-04.2 both ESCALATE in parallel:
- Batch HOLDs
- T-04.4 cannot spawn (depends on T-04.1)
- Phase 4 entirely BLOCKED
- Human must intervene on both before workflow can advance

---

*End of PHASE_4_PARALLEL_WALKTHROUGH.*
