# WHY V2 — The Ten Architectural Defects of LANG_DEV v0.1.0

**Purpose:** Document the architectural defects in `workflows/LANG_DEV/` (v0.1.0-SUPERSEDED) that motivated a from-scratch v2 rebuild against the RDC corpus at `workflows/LANG_DEV/LANGS_DEV_RDC/`.

**Authoritative source for all "RDC says X" claims below:** the RDC output files in `workflows/LANG_DEV/LANGS_DEV_RDC/`. Every defect cites a specific source.

**Migration path:** None. v1 is preserved as a historical artifact. v2 is functionally a replacement, not a successor — different state machine, different worker contracts, different acceptance model.

---

## Defect 1 — Wrong shape: 16 flat phases vs 4 grouped phases with sub-tasks

**v1 says (`workflows/LANG_DEV/LANG_DEV_WORKFLOW.json` §`phases_serial_order`):**
A flat 16-element array — `STEP_01, STEP_01_01, STEP_02, STEP_03, STEP_04, STEP_05A, STEP_05B, STEP_06, STEP_06_01, STEP_06_02, STEP_07, STEP_07_01, STEP_08, STEP_09, STEP_10, STEP_11`. Every element is a "phase." Every phase is mandatory.

**RDC says (`LANGS_DEV_RDC/PROJECT.md` §"Phase structure (4 phases, serial)"):**
Four phases — DECONSTRUCTION, DESIGN, IMPLEMENTATION, RUNTIME — each with named tasks (T-01.1, T-01.2, T-01.3, etc.) and sub-tasks (T-02.4.1, T-02.4.2; T-03.1.1; T-01.1.1 on-demand). The 16 STEP docs are SOURCES, not phases.

**v2 resolution:** `spec/PHASE_MODEL.md` defines 4 phases with task hierarchy. State machine encodes phase-then-task structure, not flat-step structure. PHASE-level verdicts gate progress, not STEP-level verdicts.

---

## Defect 2 — Phase 4 (RUNTIME) excluded entirely

**v1 says (`workflows/LANG_DEV/LANG_DEV_WORKFLOW.json` §`hard_rules.boss_level_roles_not_integrated`):**
> "the 4 boss_level_*_rules.md docs are NOT workers in this workflow. They may belong to a downstream runtime-execution workflow (not yet designed). See known_tbds."

**RDC says (`LANGS_DEV_RDC/PHASE_04_RUNTIME_ARCH.md` §1):**
> "Phase 4 is **the runtime**. Phase 3 produces plans; Phase 4 consumes them. ... Phase 4 is the END of the methodology. After Phase 4, you have a runnable DSL for your library."

`LANGS_DEV_RDC/PROJECT.md` §"Produced outputs" lists `Executor + Optimizer + Error Reporter + Debugger` as Phase 4 deliverables — not optional.

**v2 resolution:** Phase 4 is in scope. The `boss_level_roles_not_integrated` rule is dropped. `contracts/PHASE_04_CONTRACT.md` binds T-04.1 through T-04.4 to concrete deliverables. METHODOLOGY_GREEN_LIGHT requires Phase 4 complete.

---

## Defect 3 — STEP 5 ordering reversed

**v1 says (`workflows/LANG_DEV/LANG_DEV_WORKFLOW.json` §`phase_map`):**
- `STEP_05A` = `STEP 5 - BAG GRAMMAR.md` ("BAG GRAMMAR")
- `STEP_05B` = `STEP 5 - DECISIONS SCHEMA.md` ("DECISIONS SCHEMA")

**RDC says (`LANGS_DEV_RDC/INPROGRESS.md` §"COURT #1: Two-STEP-5 conflict" → §"QUEEN RULING: SYNTHESIS"):**
> "**Synthesis resolution:** `STEP_05A` = `STEP 5 - DECISIONS SCHEMA.md` — the format layer. `STEP_05B` = `STEP 5 - BAG GRAMMAR.md` — the linguistic-operator layer. Both are required sub-phases of STEP 5. Serial ordering: 05A → 05B (format first, operators second)"

Rationale (per the same ruling): the DECISIONS_SCHEMA records what STEP 4 (atoms) produced; the BAG_GRAMMAR adds operators layered on top of that schema. Format precedes operators.

**v2 resolution:** `STEP_05A` = DECISIONS_SCHEMA, `STEP_05B` = BAG_GRAMMAR. Hard rule `step_5_ordering` enforces this in `LANG_DEV_V2_WORKFLOW.json`. `contracts/PHASE_02_CONTRACT.md` honors the ruling explicitly with a `grep`-able acceptance check.

---

## Defect 4 — Nexus reports missing from input contract

**v1 says (`workflows/LANG_DEV/LANG_DEV_WORKFLOW.json` §`trigger.parameters_required`):**
Three required inputs — `target_library`, `workspace_dir`, `step_source_dir`. Nexus reports not mentioned anywhere in the workflow.

**RDC says (`LANGS_DEV_RDC/PROJECT.md` §"Required inputs"):**
> "1. **Target library** ... 2. **Nexus reports** (upstream, from prior library-analysis process) — at minimum GRAVITY, GRAMMAR, VERBS, TIERS, CLASSIFICATION. Typical location: `<target>_nexus/` directory ..."

`LANGS_DEV_RDC/PHASE_02_DESIGN_ARCH.md` §2.2 cites nexus signal → schema field mapping: `GRAMMAR → port_types, TIERS → phases, GRAVITY/WEIGHT → atoms`. STEP 1 cannot deconstruct without nexus reports.

**v2 resolution:** `spec/INPUT_CONTRACT.md` adds `nexus_reports_dir` as a required input. Hard rule `nexus_reports_required` blocks PRESTEP advancement if nexus reports are absent or incomplete. `LANG_DEV_V2_WORKFLOW.json` enumerates the required nexus report set.

---

## Defect 5 — Single-doc authority too narrow

**v1 says (`workflows/LANG_DEV/LANG_DEV_WORKFLOW.json` §`roles.STEP_EXECUTOR`):**
> "Reads the assigned STEP source doc fully ... per-step domain knowledge lives in the STEP source doc, not in this worker's role doc."

Each spawn reads ONE STEP doc.

**RDC says (`LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_TODO.md` T-01.1):**
> "**Source doc:** `STEP 1 - DECONSTRUCTION OPS.md` + `context.md` (extended rules)"

`context.md` is a retrofit (per `LANGS_DEV_RDC/INVENTORY.md` Pass 21) that adds the 5-type primitive classification, multi-tier basis, hidden-primitive check, and two-path check. STEP 1 alone is incomplete without it.

Similarly, `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_TODO.md` T-03.1.1 requires the Validator to read both `STEP 7.1` AND `<library>_decisions.json` (from Phase 2). Cross-phase artifact reads are not single-doc.

**v2 resolution:** PHASE_EXECUTOR's input includes `source_docs` (array, not scalar) AND `prior_phase_outputs`. `WORKER_PHASE_EXECUTOR.md` enforces full-read of every doc in the array. `spec/STEP_DOC_INVENTORY.md` enumerates per-task read lists.

---

## Defect 6 — `monolithic_serial` overreach

**v1 says (`workflows/LANG_DEV/LANG_DEV_WORKFLOW.json` §`hard_rules.monolithic_serial`):**
> "phases execute strictly in phases_serial_order. No parallelism across phases. Per user directive."

**RDC says (`LANGS_DEV_RDC/PHASE_04_RUNTIME_ARCH.md` §1):**
> "Phase 4's four components are mostly independent (can be built in parallel), but they share contracts: All consume ExecutionPlan, All emit structured errors consumable by Error Reporter, All are implementable without cross-references (good decoupling)"

`LANGS_DEV_RDC/PHASE_04_RUNTIME_TODO.md` opening note:
> "Tasks are mostly independent (can run in parallel or serial). Error Reporter (T-04.3) depends on all other stages having defined their error types. Debugger (T-04.4) depends on Executor (T-04.1)."

The constraint is dependency-shaped, not a blanket "no parallelism."

**v2 resolution:** Phase-level serial ordering preserved (Phase 1 → 2 → 3 → 4). Within Phase 4, parallelism allowed per dependency graph (T-04.3 first, then T-04.1 / T-04.2 / T-04.4 in parallel). `phase_model` block in `LANG_DEV_V2_WORKFLOW.json` declares per-task `parallelism: sequential | parallel-with-siblings | on-demand`.

---

## Defect 7 — No methodology-level acceptance

**v1 says:** Per-phase QA only. Verdicts are PHASE_GREEN_LIGHT / PHASE_RETRY / PHASE_ESCALATE; workflow verdict CLEAN_RUN means "all 16 phases PASS-ed." No definition of methodology correctness beyond per-doc completion criteria.

**RDC says (`LANGS_DEV_RDC/PROJECT.md` §"Success criteria"):**
> "A methodology run is successful if the produced DSL: 1. **Passes the shuffle test** ... 2. **Achieves ~20:1 compression** at the main semantic level ... 3. **Validates via `decisions.json`** — referential integrity holds ... 4. **Handles errors gracefully** ... 5. **Is self-assembling** ... 6. **Has a canonical, deterministic output** — same input → same output, always."

`LANGS_DEV_RDC/CLARIFICATION.md` §6 calls the shuffle test "the methodology's **acid test**." `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_TODO.md` T-03.5 makes it a Solver acceptance gate. But shuffle invariance on a real produced DSL — across the full pipeline — needs a methodology-level test, not just a Solver-level one.

**v2 resolution:** New `METHODOLOGY_INTEGRATOR` worker (Part 4 T-4.3) runs after Phase 4 with: (a) shuffle test on produced DSL with diverse bags, (b) compression ratio check ≥ ~20:1, (c) end-to-end demo. New verdict `METHODOLOGY_GREEN_LIGHT` gated by these. Hard rule `methodology_acceptance` blocks CLEAN_RUN without them.

---

## Defect 8 — Generic deliverables (not bound to concrete artifacts)

**v1 says (`workflows/LANG_DEV/LANG_DEV_WORKFLOW.json` §`roles.STEP_EXECUTOR.produces`):**
> "concrete output files per the STEP doc's declared output shape (e.g., primitives.json for STEP_01, atoms catalog for STEP_04, lexer.py for STEP_07)"

The "for example" is the entire specification. Workers infer outputs from STEP docs at runtime.

**RDC says (`LANGS_DEV_RDC/PHASE_*_TODO.md`):**
Every task names specific output files with full paths AND an acceptance command. Examples:
- T-01.1 outputs `workspace/STEP_01/primitives_catalog.json`, acceptance: `jq '.primitives | length' workspace/STEP_01/primitives_catalog.json` returns integer 5-15
- T-02.2 outputs `<workspace>/<library>_decisions.json`, acceptance: `python validate_decisions.py <library>_decisions.json` returns "Validation OK. Schema v1.0.0 compliant."
- T-03.5 outputs `workspace/STEP_11/{solver.py, constraints.py, execution_plan.py, test_solver.py}`, acceptance: `python -m pytest workspace/STEP_11/test_solver.py -v --tb=short` with shuffle tests passing

**v2 resolution:** `contracts/PHASE_<N>_CONTRACT.md` files (Part 2) bind every task to its concrete artifact set + acceptance command, copied verbatim from the RDC TODOs. PHASE_QA's authority is bounded by the contract, not by free interpretation of the STEP doc.

---

## Defect 9 — No on-demand spawn semantics

**v1 says (`workflows/LANG_DEV/LANG_DEV_WORKFLOW.json` §`flow.per_phase`):**
> "For each phase_id in phases_serial_order (16 iterations) ..."

Every phase runs every time. No conditional spawning.

**RDC says (`LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_TODO.md` T-01.1.1 header):**
> "Not started (runs on demand)"

T-01.1.1 RECOVER runs ONLY when T-01.1 cannot reach completion criteria. If T-01.1 passes first attempt, T-01.1.1 never spawns.

**v2 resolution:** Each task in `phase_model` declares its `parallelism` field as one of `sequential | parallel-with-siblings | on-demand`. On-demand tasks are wired to a triggering condition (e.g., T-01.1.1 triggers on T-01.1 FAIL after retry exhaustion). Methodology verdict accounts for skipped-by-design vs failed.

---

## Defect 10 — RDC explicitly targets v0.2.0+ (not v0.1.0)

**v1 says:** Self-identifies as `0.1.0-DRAFT`, "synthesized from LANGS_DEVELOPMENT corpus; several design questions explicitly deferred as known_tbds; answers emerge by running it." Author acknowledges the draft nature.

**RDC says (`LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_TODO.md` and every other PHASE_*_TODO header):**
> "**Consumed by:** `LANG_DEV_WORKFLOW` v0.2.0+ — each T- task drives one step-executor spawn."

The RDC author already knew, at the time of producing the consolidation pass, that v0.1.0 needed replacement and produced a corpus designed to drive that replacement.

`LANGS_DEV_RDC/PROJECT.md` §"Downstream consumption":
> "The methodology's 4-phase TODO docs drive the `LANG_DEV_WORKFLOW` (v0.2.0+) which executes this methodology as an AI swarm."

**v2 resolution:** v2 is the answer to that explicit version target. Numbered v2.0.0-DRAFT (not 0.2.0) because v2 is a from-scratch rebuild, not an incremental version bump from 0.1.0. Major-version jump signals incompatible state machine, incompatible worker contracts, incompatible verdict semantics.

---

## What v1 got right (preserved in v2)

Not everything in v1 was wrong. The following design choices carry forward unchanged:

- **QUEEN-as-orchestrator** (Claude in main conversation, not a spawned agent)
- **Workspace-discipline** (`workspace_manifest.json` as single source of truth; intermediate outputs in `workspace_dir/`)
- **No-fabrication discipline** (every output traces to source analysis or prior-phase output)
- **STEP-doc-immutability** (source docs are read-only)
- **No-no-verify** (never bypass git hooks)
- **INPROGRESS prepend-only**
- **Full-read-or-skip** discipline (no partial scans)
- **Same-doc-authority** principle (just expanded to multi-doc-authority in v2 — workers don't substitute their own aesthetic judgment for what the source docs prescribe)
- **Honest-ambiguity** discipline (when source disagrees, surface it; don't silently pick)
- **No auto-trigger of other workflows**
- **No workflow mixing**

---

## Cross-reference table

| Defect | v1 location | RDC source | v2 resolution |
|---|---|---|---|
| 1. Wrong shape | `LANG_DEV_WORKFLOW.json` §phases_serial_order | `PROJECT.md` §Phase structure | `spec/PHASE_MODEL.md` |
| 2. Phase 4 missing | `LANG_DEV_WORKFLOW.json` §hard_rules.boss_level_roles_not_integrated | `PHASE_04_RUNTIME_ARCH.md` §1 | `contracts/PHASE_04_CONTRACT.md` |
| 3. STEP 5 reversed | `LANG_DEV_WORKFLOW.json` §phase_map | `INPROGRESS.md` §COURT #1 SYNTHESIS | `contracts/PHASE_02_CONTRACT.md` + hard rule |
| 4. Nexus inputs | `LANG_DEV_WORKFLOW.json` §trigger.parameters_required | `PROJECT.md` §Required inputs | `spec/INPUT_CONTRACT.md` |
| 5. Single-doc authority | `LANG_DEV_WORKFLOW.json` §roles.STEP_EXECUTOR | `PHASE_01_DECONSTRUCTION_TODO.md` T-01.1 | `WORKER_PHASE_EXECUTOR.md` |
| 6. Monolithic-serial | `LANG_DEV_WORKFLOW.json` §hard_rules.monolithic_serial | `PHASE_04_RUNTIME_ARCH.md` §1 | `phase_model.<task>.parallelism` field |
| 7. No methodology acceptance | `LANG_DEV_WORKFLOW.json` §verdicts | `PROJECT.md` §Success criteria | `WORKER_METHODOLOGY_INTEGRATOR.md` |
| 8. Generic deliverables | `LANG_DEV_WORKFLOW.json` §roles.STEP_EXECUTOR.produces | `PHASE_*_TODO.md` (every task) | `contracts/PHASE_<N>_CONTRACT.md` |
| 9. No on-demand spawn | `LANG_DEV_WORKFLOW.json` §flow.per_phase | `PHASE_01_DECONSTRUCTION_TODO.md` T-01.1.1 | `phase_model.<task>.parallelism: on-demand` |
| 10. v0.2.0+ target | `LANG_DEV_WORKFLOW.json` §version | `PHASE_*_TODO.md` (every header) | v2.0.0-DRAFT version + DEPRECATED.md |

---

*End of WHY_V2.*
