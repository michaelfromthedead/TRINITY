# RECOVERY_MODEL — T-01.1.1 Spawn Semantics

**Purpose:** Document the decision for how T-01.1.1 RECOVER is modeled in v2.

**Decision:** T-01.1.1 is the **same PHASE_EXECUTOR worker invoked with `recovery_mode: true`** — NOT a separate worker role.

**Rationale:**

1. **Avoid worker proliferation.** Adding a dedicated RECOVERY_EXECUTOR role doubles maintenance surface (role doc, spawn semantics, failure modes) for a capability that is a parametric variation of PHASE_EXECUTOR.

2. **Recovery discipline lives in the STEP doc.** `STEP 1.1 - RECOVER OPS.md` fully specifies the 4 recovery strategies and their application order. The methodology is the source of truth; the worker is a generic executor of methodology-defined procedures.

3. **Contract already covers the deliverables.** `contracts/PHASE_01_CONTRACT.md#T-01.1.1` specifies recovery_log.md + updated primitives_catalog.json outputs with acceptance commands. No additional contract needed.

4. **Consistent with v2's generic-worker pattern.** PHASE_EXECUTOR is already generic across all 16+ tasks; recovery is one more mode, not a special case warranting a new worker.

---

## Spawn semantics

**Trigger:** T-01.1 reaches `task_retry_counter == 2` without TASK_PASS (i.e., two consecutive PHASE_QA FAILs on T-01.1).

**QUEEN behavior:** Instead of emitting `TASK_FAIL_ESCALATE` for T-01.1 immediately, QUEEN consults `recovery_attempts` counter for T-01.1:
- If `recovery_attempts < 3`: spawn PHASE_EXECUTOR for T-01.1.1 with `recovery_mode: true`
- If `recovery_attempts == 3`: emit `TASK_FAIL_ESCALATE` for T-01.1 (recovery exhausted)

**PHASE_EXECUTOR in recovery mode:**
- Reads `STEP 1.1 - RECOVER OPS.md` (the sole source doc for recovery)
- Reads T-01.1's partial outputs (`primitives_catalog.json`, `deconstruction_notes.md`)
- Reads T-01.1's STEP_QA findings (passed in prior_retry_findings)
- Reads `recovery_log.md` if prior recovery attempts occurred (so it can pick an untried strategy)
- Selects ONE recovery strategy from the 4 in `STEP 1.1`:
  1. Fresh re-read of source examples
  2. Compound-primitive check
  3. Cross-domain analogy
  4. Tier escalation
- Applies the strategy; updates `primitives_catalog.json`; appends to `recovery_log.md`
- Runs T-01.1's acceptance commands on the updated catalog
- Returns completion report

**PHASE_QA for T-01.1.1:**
- Runs against `PHASE_01_CONTRACT.md#T-01.1.1`
- Key verification: did the strategy apply produce a catalog that now passes T-01.1's acceptance commands?
- TASK_PASS for T-01.1.1 implies T-01.1 is now ready for a fresh PHASE_QA run (which should pass)

**Post-T-01.1.1:**
- If T-01.1.1 TASK_PASS AND T-01.1 re-QA PASSes: T-01.1 marked TASK_PASS; Phase 1 advances to T-01.2. `recovery_attempts` retained on workspace_manifest.json for transparency.
- If T-01.1.1 TASK_PASS but T-01.1 re-QA still FAILs (strategy worked partially): QUEEN spawns T-01.1.1 again (next strategy); `recovery_attempts++`
- If T-01.1.1 TASK_FAIL: counts toward `recovery_attempts`; spawn again if budget remains

---

## Bookkeeping

`workspace_manifest.json` task entry for T-01.1 when recovery occurred:

```json
{
  "task_id": "T-01.1",
  "status": "pass",
  "attempts": 2,
  "recovery_attempts": 2,
  "recovery_log_path": "STEP_01/recovery_log.md",
  "outputs": [...],
  "qa_verdict_log": [
    "TASK_FAIL_RETRY at <ISO>: <finding summary>",
    "TASK_FAIL_RETRY at <ISO>: <finding summary>",
    "Recovery triggered; T-01.1.1 spawned at <ISO>",
    "T-01.1.1 TASK_PASS at <ISO> (strategy: compound-primitive check)",
    "T-01.1 re-QA: TASK_PASS at <ISO>"
  ]
}
```

T-01.1.1 gets its own entry only if it was spawned:

```json
{
  "task_id": "T-01.1.1",
  "status": "pass",
  "attempts": 1,
  "strategy_applied": "compound-primitive check",
  "outputs": [
    {"path": "STEP_01/recovery_log.md", "sha256": "..."},
    {"path": "STEP_01/primitives_catalog.json", "sha256": "..."}
  ]
}
```

---

## Verdict semantics

T-01.1.1 uses the same verdicts as any other task (TASK_PASS / TASK_FAIL_RETRY / TASK_FAIL_ESCALATE). Recovery-specific semantics are layered on by QUEEN's spawn orchestration, not by the verdict vocabulary.

`TASK_SKIP_BY_DESIGN` for T-01.1.1 means T-01.1 passed on first attempt (or within retry budget) and recovery was never triggered — the default happy path.

---

## Why NOT a separate RECOVERY_EXECUTOR worker

Considered and rejected:

| Argument for separate worker | Counter |
|---|---|
| "Recovery is cognitively different from forward-execution" | Recovery is procedural (apply strategy → verify). The cognitive work is in the STEP 1.1 source doc, not in the worker. |
| "Recovery might need different tools" | Same tools (read files, write JSON + markdown, run acceptance commands). |
| "Role separation aids maintenance" | More roles = more docs + more contracts + more edge cases. Parametric mode is simpler. |
| "QA semantics might differ" | They don't. PHASE_QA verifies outputs against the contract; the contract for T-01.1.1 is just as concrete as any other task. |
| "SDLC has DEV + TESTDEV separation for similar reasons" | That separation exists because DEV and TESTDEV have INCOMPATIBLE context access (DEV sees all code; BLACKBOX sees none). Recovery has no such incompatibility. |

---

*End of RECOVERY_MODEL.*
