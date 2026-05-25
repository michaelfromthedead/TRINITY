# LANG_DEV_V2 — Entry Point

**Status:** v2.0.0-DRAFT — buildout complete; first engagement pending.

**What this is:** A swarm workflow for executing the language-development methodology consolidated in `workflows/LANG_DEV/LANGS_DEV_RDC/`. Given a target programming library + pre-generated nexus reports, produces a fully-functional DSL covering the library's domain (compiler pipeline + runtime support).

**Authoritative methodology source:** `workflows/LANG_DEV/LANGS_DEV_RDC/` (RDC pass output, do not edit).

**Trigger:** `LANG_DEV_V2_WORKFLOW` or `LANG_DEV_V2`.

**v2 vs v1:** v1 (`workflows/LANG_DEV/`, v0.1.0-SUPERSEDED) had ten architectural defects. v2 is a from-scratch rebuild against the RDC corpus. See `WHY_V2.md` for the defect list. v1 is preserved as a historical artifact.

---

## The 4-phase pipeline

```
PRESTEP — validate inputs (target_library, nexus_reports_dir, workspace_dir, step_source_dir)
   │
   ▼
PHASE 1 — DECONSTRUCTION
  Find primitives, objects, type signatures.
  Tasks: T-01.1 (Deconstruction Ops), T-01.1.1 (on-demand RECOVER), T-01.2 (Objs), T-01.3 (Types)
  Source docs: STEP 1 + 1.1 + 2 + 3 + context.md
   │
   ▼
PHASE 2 — DESIGN
  Specify atoms, decisions schema, bag grammar, ruleset, tokens.
  Tasks: T-02.1 (Atomics), T-02.2 (5A Schema), T-02.3 (5B Grammar), T-02.4 (Rulesets) → 02.4.1 (Conundrum) → 02.4.2 (Pre-Lexer)
  Source docs: STEP 4 + 5-DECISIONS + 5-BAG_GRAMMAR + 6 + 6.1 + 6.2
  Note: STEP 5A = DECISIONS_SCHEMA, STEP 5B = BAG_GRAMMAR (per COURT #1 SYNTHESIS — v1 had reversed)
   │
   ▼
PHASE 3 — IMPLEMENTATION
  Build the compiler pipeline.
  Tasks: T-03.1 (Lexer), T-03.1.1 (Validator), T-03.2 (Parser), T-03.3 (Typer), T-03.4 (Classifier), T-03.5 (Solver — Solver-level shuffle test)
  Source docs: STEP 7 + 7.1 + 8 + 9 + 10 + 11
   │
   ▼
PHASE 4 — RUNTIME
  Build executor, optimizer, error reporter, debugger.
  Tasks: T-04.3 (Error Reporter; first), then (T-04.1 Executor || T-04.2 Optimizer in parallel), then T-04.4 (Debugger)
  Source docs: boss_level_1 + 2 + 3 + 4
   │
   ▼
METHODOLOGY_INTEGRATION
  Gate 1: methodology-level shuffle test (12+ bags × 100 iterations)
  Gate 2: compression ratio check (≥18:1 production target; ≥1.5:1 reference target)
  Gate 3: end-to-end demo (full pipeline + golden-rule check)
   │
   ▼
METHODOLOGY_GREEN_LIGHT → CLEAN_RUN
```

Each phase has a contract (`contracts/PHASE_<N>_CONTRACT.md`) binding tasks to concrete output files + acceptance commands.

---

## Quickstart

**Prerequisites:**
1. A target library to deconstruct (e.g., `pandas`)
2. Pre-generated nexus reports for it (8 reports: GRAVITY, GRAMMAR, VERBS, TIERS, CLASSIFICATION, GENESIS, GENERATOR, COMPRESSION) at `<target>_nexus/`
3. The 22 methodology source docs at `LANG_DEV_STEPS/` (or override via `step_source_dir` parameter)

**Engagement:**
```
User: LANG_DEV_V2_WORKFLOW
QUEEN: [reads required docs]
       "LANG_DEV_V2_WORKFLOW mode engaged. Inputs: target=<pending>, nexus=<pending>, workspace=<pending>. Ready."

User: target=/path/to/pandas, workspace=/path/to/pandas_workspace
QUEEN: [PRESTEP validation: target ✓, nexus auto-derived ✓ 8/8 reports, workspace ✓ created, step_source ✓ 22/22 docs]
       "PRESTEP: all inputs validated. Workspace initialized. Ready to spawn Phase 1. Confirm to proceed."

User: proceed
QUEEN: [spawns PHASE_EXECUTOR for T-01.1 with multi-doc reads: STEP 1 + context.md]
       ...
```

See `walkthroughs/HAPPY_PATH_WALKTHROUGH.md` for a full mental trace.

---

## Key documents

### Specifications
| Doc | Purpose |
|---|---|
| `LANG_DEV_V2_WORKFLOW.json` | State machine — modes, flow, roles, verdicts, hard rules |
| `WHY_V2.md` | The 10 v1 defects + v2 resolutions (architectural rationale) |
| `spec/PHASE_MODEL.md` | 4-phase task hierarchy with sub-task structure + parallelism declarations |
| `spec/INPUT_CONTRACT.md` | The 4 required inputs + PRESTEP validation sequence |
| `spec/STEP_DOC_INVENTORY.md` | Per-task source-doc read list (multi-doc) |
| `spec/RECOVERY_MODEL.md` | T-01.1.1 RECOVER spawn semantics decision |
| `spec/REFERENCE_LIBRARY.md` | The integration-test target library spec |
| `spec/SHUFFLE_TEST_SPEC.md` | Methodology-level Gate 1 spec |
| `spec/COMPRESSION_SPEC.md` | Methodology-level Gate 2 spec |
| `spec/E2E_DEMO_SPEC.md` | Methodology-level Gate 3 spec |

### Contracts (per-phase deliverable bindings)
| Doc | Purpose |
|---|---|
| `contracts/PHASE_01_CONTRACT.md` | T-01.1 / T-01.1.1 / T-01.2 / T-01.3 deliverables + acceptance commands |
| `contracts/PHASE_02_CONTRACT.md` | T-02.1 / T-02.2 (5A) / T-02.3 (5B) / T-02.4 / T-02.4.1 / T-02.4.2 |
| `contracts/PHASE_03_CONTRACT.md` | T-03.1 / T-03.1.1 / T-03.2 / T-03.3 / T-03.4 / T-03.5 (incl. Solver shuffle test) |
| `contracts/PHASE_04_CONTRACT.md` | T-04.3 / T-04.1 / T-04.2 / T-04.4 (parallel-aware) |
| `contracts/ARTIFACT_CATALOG.md` | Cross-phase artifact index + workspace_manifest.json schema |

### Worker docs
| Doc | Purpose |
|---|---|
| `WORKER_PHASE_EXECUTOR.md` | Per-task worker — multi-doc reads, contract-bound outputs |
| `WORKER_PHASE_QA.md` | Per-task verifier — bounded by contract; binding verdict |
| `WORKER_METHODOLOGY_INTEGRATOR.md` | Terminal acceptance worker — runs Gates 1-3 |

### Walkthroughs
| Doc | Purpose |
|---|---|
| `walkthroughs/HAPPY_PATH_WALKTHROUGH.md` | Full canonical run (incl. realistic Gate 2 escalation) |
| `walkthroughs/RECOVERY_WALKTHROUGH.md` | T-01.1.1 on-demand spawn after T-01.1 retry exhaustion |
| `walkthroughs/SHUFFLE_FAIL_WALKTHROUGH.md` | All phases PASS but methodology-level Gate 1 fails (cross-stage bug) |
| `walkthroughs/PHASE_4_PARALLEL_WALKTHROUGH.md` | Parallel siblings; one PASS one ESCALATE; aggregate logic |

### Test scaffolding
| Path | Purpose |
|---|---|
| `test_target/README.md` | Layout for reference library + nexus + sample data |
| `test_target/validators/*.py` | Workspace manifest, decisions schema, phase-output validators |
| `test_target/sample_data/small.csv` | E2E demo input |

### Buildout
| Doc | Purpose |
|---|---|
| `LANG_DEV_V2_BUILDOUT_TODO.md` | 32-task buildout TODO (all parts complete) |
| `KNOWN_LIMITATIONS.md` | Buildout limitations + first-run prep notes |

---

## Hard rules (workflow-specific, beyond shared protocol)

1. **`step_5_ordering`:** STEP_05A = DECISIONS_SCHEMA, STEP_05B = BAG_GRAMMAR (per COURT #1 SYNTHESIS)
2. **`nexus_reports_required`:** PRESTEP cannot complete without all 8 nexus reports
3. **`phase_4_required`:** METHODOLOGY_GREEN_LIGHT cannot be emitted without Phase 4 complete
4. **`methodology_acceptance`:** CLEAN_RUN requires METHODOLOGY_INTEGRATOR Gates 1-3 PASS
5. **`multi_doc_authority`:** PHASE_EXECUTOR full-reads every doc in source_docs array; ambiguities surfaced not silenced

Plus all shared `WORKER_PROTOCOL.md` rules (no-fabrication, no-no-verify, INPROGRESS prepend-only, etc.).

---

## Relationship to other workflows

- **Parallel-altitude to** SDLC, RDC, RECON, BOOK family, ORGANIZE, LANG_DEV (v1)
- **Consumes** the RDC output corpus at `workflows/LANG_DEV/LANGS_DEV_RDC/` (one-time, at design time)
- **Does not auto-trigger** any other workflow
- **Does not mix** with any other active workflow
- **Open question:** whether v2 should remain a workflow at all, or whether the RDC's `PHASE_*_TODO.md` outputs should be handed directly to SDLC. See `LANG_DEV_V2_BUILDOUT_TODO.md` Open Questions §1.

---

## Shared infrastructure

- `workflows/SHARED/WORKER_QUEEN.md` — QUEEN orchestrator role
- `workflows/SHARED/WORKER_PROTOCOL.md` — non-negotiable worker contract
- `workflows/SHARED/WORKER.md` — master role index (will be updated to include v2 workers)

---

*End of README.*
