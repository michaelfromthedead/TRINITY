# PHASE_MODEL — LANG_DEV_V2 Phase Structure

**Purpose:** Define the 4-phase grouped task model that the v2 state machine implements. Replaces v1's flat 16-step model.

**Authoritative source:** `workflows/LANG_DEV/LANGS_DEV_RDC/PHASE_<1-4>_{ARCH,TODO}.md`. This document compresses + structures the RDC content; on any conflict, RDC wins.

---

## 1. Top-level structure

```
PRESTEP (input validation, workspace init)
   │
   ▼
PHASE 1 ─ DECONSTRUCTION ─ understand the domain
   │
   ▼
PHASE 2 ─ DESIGN ─ specify atoms + composition
   │
   ▼
PHASE 3 ─ IMPLEMENTATION ─ build compiler pipeline
   │
   ▼
PHASE 4 ─ RUNTIME ─ executor / optimizer / errors / debugger
   │
   ▼
METHODOLOGY_INTEGRATION (shuffle test + compression check + e2e demo)
   │
   ▼
METHODOLOGY_GREEN_LIGHT (or escalate)
```

Phase ordering is strictly serial. Within phases, see per-phase parallelism rules below.

---

## 2. Per-phase task hierarchy

### Phase 1 — DECONSTRUCTION

| Task | Sub-of | Parallelism | Mandatory? | Source docs |
|---|---|---|---|---|
| T-01.1 — Deconstruction Ops | — | sequential | yes | `STEP 1`, `context.md` |
| T-01.1.1 — Recover Ops | T-01.1 | on-demand (triggers on T-01.1 FAIL after retry) | conditional | `STEP 1.1` |
| T-01.2 — Deconstruction Objs | — | sequential (after T-01.1 PASS) | yes | `STEP 2` |
| T-01.3 — Deconstruction Types | — | sequential (after T-01.2 PASS) | yes | `STEP 3` |

**Phase verdict:** PHASE_GREEN_LIGHT requires T-01.1, T-01.2, T-01.3 all PASS. T-01.1.1 is independent — its presence/absence in the run does not alter the phase verdict structure (it's a recovery aid for T-01.1).

### Phase 2 — DESIGN

| Task | Sub-of | Parallelism | Mandatory? | Source docs |
|---|---|---|---|---|
| T-02.1 — Atomics | — | sequential | yes | `STEP 4`, `context.md` (for primitive classification context) |
| T-02.2 — Decisions Schema (STEP 5A) | — | sequential (after T-02.1 PASS) | yes | `STEP 5 - DECISIONS SCHEMA` |
| T-02.3 — Bag Grammar (STEP 5B) | — | sequential (after T-02.2 PASS) | yes | `STEP 5 - BAG GRAMMAR` |
| T-02.4 — Rulesets and Defaults | — | sequential (after T-02.3 PASS) | yes | `STEP 6` |
| T-02.4.1 — The Conundrum | T-02.4 | sequential within T-02.4 | yes | `STEP 6.1` |
| T-02.4.2 — Pre-Lexer | T-02.4 | sequential within T-02.4 (after T-02.4.1 PASS) | yes | `STEP 6.2` |

**STEP 5 ordering note (binding, per COURT #1 SYNTHESIS in `LANGS_DEV_RDC/INPROGRESS.md`):**
- `STEP_05A` = DECISIONS SCHEMA (format layer; records what STEP 4 produced)
- `STEP_05B` = BAG GRAMMAR (operator layer; layered on top of the schema)
- v1 had these reversed. v2 enforces this ordering as a hard rule.

**Phase verdict:** All six tasks PASS.

### Phase 3 — IMPLEMENTATION

| Task | Sub-of | Parallelism | Mandatory? | Source docs |
|---|---|---|---|---|
| T-03.1 — Lexer | — | sequential | yes | `STEP 7` |
| T-03.1.1 — Validator | T-03.1 | sequential (after T-03.1 PASS) | yes | `STEP 7.1` |
| T-03.2 — Parser | — | sequential (after T-03.1.1 PASS) | yes | `STEP 8` |
| T-03.3 — Typer | — | sequential (after T-03.2 PASS) | yes | `STEP 9` |
| T-03.4 — Classifier | — | sequential (after T-03.3 PASS) | yes | `STEP 10` |
| T-03.5 — Solver | — | sequential (after T-03.4 PASS) | yes | `STEP 11` |

**Phase verdict:** All six tasks PASS. T-03.5's acceptance includes the Solver-level shuffle test (necessary but not sufficient — methodology-level shuffle test runs at integration).

### Phase 4 — RUNTIME

| Task | Sub-of | Parallelism | Mandatory? | Source docs |
|---|---|---|---|---|
| T-04.3 — Error Reporter | — | sequential (runs first; defines unified Error type) | yes | `boss_level_3` |
| T-04.1 — Executor | — | parallel-with-siblings (after T-04.3 PASS) | yes | `boss_level_1` |
| T-04.2 — Optimizer | — | parallel-with-siblings (after T-04.3 PASS) | yes | `boss_level_2` |
| T-04.4 — Debugger | — | sequential (after T-04.1 PASS) | yes | `boss_level_4` |

**Note on Phase 4 ordering rationale:**
- T-04.3 must define the unified Error type before other stages emit errors against it
- T-04.4 (Debugger) depends on T-04.1 (Executor) per `LANGS_DEV_RDC/PHASE_04_RUNTIME_TODO.md` opening note
- T-04.2 (Optimizer) is independent of T-04.1 and T-04.4 — can parallel with them
- Trade-off: strict-serial is simpler; parallel-Phase-4 is faster. v2 declares the parallelism, but QUEEN may choose serial execution if context permits.

**Phase verdict:** All four tasks PASS.

---

## 3. Verdict cascade

### Per-task verdicts

| Verdict | Meaning | Action |
|---|---|---|
| `TASK_PASS` | PHASE_QA verdict PASS against contract acceptance | Append to workspace_manifest; mark task complete; advance to next task per dependency graph |
| `TASK_FAIL_RETRY` | PHASE_QA FAIL with retries remaining | Re-spawn PHASE_EXECUTOR with findings as correction directive; retry_counter++ |
| `TASK_FAIL_ESCALATE` | PHASE_QA FAIL + retry_counter at limit | Pause workflow; report; await human |
| `TASK_SKIP_BY_DESIGN` | On-demand task not triggered (e.g., T-01.1.1 when T-01.1 passes first attempt) | No spawn occurred; phase verdict not affected |

### Per-phase verdicts

| Verdict | Meaning | Action |
|---|---|---|
| `PHASE_GREEN_LIGHT` | All mandatory tasks PASS; no escalations pending | Advance to next phase |
| `PHASE_HOLD` | At least one task FAIL_ESCALATE; phase work otherwise complete | Workflow paused; phase incomplete |
| `PHASE_BLOCKED` | A task cannot start because a prior dependency FAIL_ESCALATEd | Cascading hold |

### Methodology verdict

| Verdict | Meaning |
|---|---|
| `METHODOLOGY_GREEN_LIGHT` | All 4 phases PHASE_GREEN_LIGHT + METHODOLOGY_INTEGRATOR PASS (shuffle test + compression + e2e demo) |
| `METHODOLOGY_INCOMPLETE` | Any phase HOLD/BLOCKED; OR all phases PASS but METHODOLOGY_INTEGRATOR FAIL |
| `ABORTED` | Human aborts mid-flight |

---

## 4. Loop limits

| Counter | Limit | Reset trigger | Behavior at limit |
|---|---|---|---|
| `task_retry_counter` | 2 | Each task entry | TASK_FAIL_ESCALATE |
| `phase_retry_counter` | 0 (no per-phase retry; only per-task) | n/a | n/a |
| `methodology_integration_retry_counter` | 1 | METHODOLOGY_INTEGRATOR entry | If shuffle test or compression fails twice, METHODOLOGY_INCOMPLETE — do not infinitely re-run integration |

Rationale: per-task retry only. Phase-level retry (rerun an entire phase from scratch) is not modeled — if a phase produces enough failures to need that, escalation is the correct response.

---

## 5. Mandatory vs on-demand semantics

**Mandatory tasks** (the default): must reach TASK_PASS before downstream tasks can proceed. Cannot be skipped.

**On-demand tasks** (T-01.1.1 RECOVER is the only one in v2): spawned only when a triggering condition is met. Triggering condition for T-01.1.1: T-01.1 reaches retry limit without TASK_PASS. PHASE_EXECUTOR for T-01.1.1 receives T-01.1's last failed output + STEP_QA findings as recovery context. After T-01.1.1 PASS, T-01.1 is re-spawned (recovery succeeded) — NOT advanced past.

**Decision (per `LANG_DEV_V2_BUILDOUT_TODO.md` T-4.4):** T-01.1.1 is the same PHASE_EXECUTOR worker invoked with `recovery_mode: true` parameter, not a separate worker doc. Recovery discipline is encoded in `STEP 1.1` source doc + the contract — workers do not need a separate role doc for it.

---

## 6. Cross-phase artifact dependencies

Each phase consumes its predecessors' outputs. Artifact catalog (full version in `contracts/ARTIFACT_CATALOG.md`):

```
Phase 1 produces:
  workspace/STEP_01/{primitives_catalog.json, tier_compression.md, deconstruction_notes.md, recovery_log.md (conditional)}
  workspace/STEP_02/{object_hierarchy.json, object_operation_matrix.md}
  workspace/STEP_03/{type_signatures.json, composition_graph.dot, type_algebra.md}

Phase 2 consumes Phase 1; produces:
  workspace/STEP_04/{atoms_draft.json, port_types_draft.json, phases_draft.json, pcfg_weights_initial.json}
  <workspace>/<library>_decisions.json   ← AUTHORITATIVE; root, not in workspace/
  workspace/STEP_05B/bag_grammar_spec.md
  workspace/STEP_06/ruleset_spec.md
  workspace/STEP_06_1/conundrum_resolution.md
  workspace/STEP_06_2/token_inventory.md

Phase 3 consumes Phases 1-2; produces:
  workspace/STEP_07/{lexer.py, tokens.py, test_lexer.py}
  workspace/STEP_07_1/{validator.py, vocabulary.py, levenshtein.py, test_validator.py}
  workspace/STEP_08/{parser.py, cst.py, test_parser.py}
  workspace/STEP_09/{typer.py, types.py, atom_catalog.py, test_typer.py}
  workspace/STEP_10/{classifier.py, ast.py, column_analysis.py, test_classifier.py}
  workspace/STEP_11/{solver.py, constraints.py, execution_plan.py, test_solver.py}

Phase 4 consumes Phase 3 + decisions.json; produces:
  workspace/BOSS_LEVEL_1/{executor.py, execution_context.py, atom_executors.py, trace.py, test_executor.py}
  workspace/BOSS_LEVEL_2/{optimizer.py, optimization_rules.py, cost_model.py, test_optimizer.py}
  workspace/BOSS_LEVEL_3/{error_reporter.py, unified_error.py, rendering.py, fuzzy_match.py, test_error_reporter.py}
  workspace/BOSS_LEVEL_4/{debugger.py, debug_session.py, snapshot.py, visualization.py, commands.py, test_debugger.py}

METHODOLOGY_INTEGRATION consumes everything; produces:
  workspace/METHODOLOGY_REPORT.md
  workspace/SHUFFLE_TEST_RESULTS.json
  workspace/COMPRESSION_REPORT.md
  workspace/E2E_DEMO_OUTPUT.md
```

Across all phases, `workspace_manifest.json` records every artifact's path + producing-task + consuming-tasks + checksum.

---

## 7. Comparison to v1

| Aspect | v1 (0.1.0-SUPERSEDED) | v2 (2.0.0-DRAFT) |
|---|---|---|
| Top-level units | 16 flat phases | 4 grouped phases + 1 integration step |
| Sub-tasks | None (T-02.4.1 is just STEP_06_01 in flat array) | Explicit (T-02.4.1, T-02.4.2 nested under T-02.4) |
| Parallelism | Forbidden (`monolithic_serial`) | Per-task declaration (sequential / parallel-with-siblings / on-demand) |
| Phase 4 | Excluded by hard rule | Included |
| STEP 5A/B | Reversed | Per COURT #1 SYNTHESIS |
| Acceptance gate | Per-step QA only | Per-task QA + per-phase verdict + methodology-level integrator |
| Retry model | Per-phase (16 separate counters) | Per-task |
| Recovery (T-01.1.1) | Same as any other phase (always runs) | On-demand (only on T-01.1 FAIL) |

---

*End of PHASE_MODEL.*
