# LANG_DEV_V2 — Buildout TODO

**Goal:** Rebuild the LANG_DEV workflow against the authoritative methodology corpus produced by the LANGS_DEV RDC pass (`workflows/LANG_DEV/LANGS_DEV_RDC/`). v1 (`workflows/LANG_DEV/`, v0.1.0-DRAFT) is preserved as-is for reference; v2 replaces it functionally.

**Authoritative source:** `workflows/LANG_DEV/LANGS_DEV_RDC/`
- `PROJECT.md`, `CLARIFICATION.md`, `MASTER.md`, `PEDAGOGY.md`, `INVENTORY.md`, `EVALUATIONS.md`, `INPROGRESS.md`
- `PHASE_01_DECONSTRUCTION_{ARCH,TODO}.md` through `PHASE_04_RUNTIME_{ARCH,TODO}.md`

**Why v2 (vs. patching v1):** Ten architectural defects identified in v1 (see `WHY_V2.md` — Part 1). Patching v1 would leave its "monolithic-serial 16 flat phases, no Phase 4" frame in place; v2 restructures around the RDC's 4-phase grouped model with sub-tasks, on-demand spawns, methodology-level acceptance.

**Convention:** All tasks use `T-<part>.<n>` IDs. Status checkboxes flip on commit-of-output. Acceptance commands are runnable; outputs include per-task commit message conventions.

---

## Part 1 — Scaffold + V1 deprecation + spec foundation

### T-1.1 — Directory scaffold
- [x] Create `workflows/LANG_DEV_V2/` (this TODO lives here)
- [x] Decide internal subdir layout (mirrors LANG_DEV/ + adds spec/, walkthroughs/, contracts/)
- [x] Add a stub `workflows/LANG_DEV_V2/README.md` (entry-point guide; finalized in Part 8)

**Output:** `workflows/LANG_DEV_V2/{README.md, contracts/, walkthroughs/, spec/}` ✓

### T-1.2 — V1 deprecation marker
- [x] Add `workflows/LANG_DEV/DEPRECATED.md` explaining v2 supersedes; keep v1 readable as historical reference
- [x] Bump `LANG_DEV_WORKFLOW.json` version block to `0.1.0-SUPERSEDED` (do NOT delete)
- [x] Cross-reference v2 location

**Output:** `workflows/LANG_DEV/DEPRECATED.md` ✓
**Acceptance:** `grep -l SUPERSEDED workflows/LANG_DEV/LANG_DEV_WORKFLOW.json` succeeds.

### T-1.3 — `WHY_V2.md` spec doc (architectural rationale)
- [x] Document the 10 v1 defects (scope: 4 vs 16 phases, missing Phase 4, STEP 5 reversal, nexus-input gap, single-doc authority, monolithic-serial overreach, no methodology acceptance, generic deliverables, no on-demand spawn, RDC explicit v0.2.0 target)
- [x] For each defect: cite RDC source (file + section), v1 location, v2 resolution
- [x] Conclude with v1→v2 migration path (none — v2 replaces functionally; v1 stays as artifact)

**Output:** `workflows/LANG_DEV_V2/WHY_V2.md` ✓

### T-1.4 — `PHASE_MODEL.md` spec doc
- [x] Define 4-phase grouped model: DECONSTRUCTION / DESIGN / IMPLEMENTATION / RUNTIME
- [x] Per phase: list of tasks (with sub-task hierarchy), dependencies, parallelism allowed/forbidden
- [x] Mandatory vs on-demand task semantics (T-01.1.1 RECOVER is on-demand; rest are mandatory)
- [x] Per-phase verdict (PHASE_GREEN_LIGHT) vs methodology verdict (METHODOLOGY_GREEN_LIGHT)

**Output:** `workflows/LANG_DEV_V2/spec/PHASE_MODEL.md` ✓

### T-1.5 — `INPUT_CONTRACT.md` spec doc
- [x] Required inputs: `target_library`, `nexus_reports_dir`, `workspace_dir`, `step_source_dir`
- [x] Per-input: schema, validation rules, what happens if missing (ESCALATE)
- [x] Nexus reports list: GRAVITY, GRAMMAR, VERBS, TIERS, CLASSIFICATION, GENESIS, GENERATOR, COMPRESSION (per `LANGS_DEV_RDC/PROJECT.md` §Required inputs)
- [x] Document the upstream nexus-analysis process is OUT OF SCOPE — nexus reports are a precondition

**Output:** `workflows/LANG_DEV_V2/spec/INPUT_CONTRACT.md` ✓

### T-1.6 — `STEP_DOC_INVENTORY.md` (the read-list per phase)
- [x] List all 22 source docs from `LANGS_DEV_RDC/INVENTORY.md`
- [x] Per phase, declare which docs PHASE_EXECUTOR must full-read:
  - Phase 1 reads: `STEP 1`, `STEP 1.1`, `STEP 2`, `STEP 3`, **`context.md`** (extended rules retrofit)
  - Phase 2 reads: `STEP 4`, `STEP 5 - DECISIONS SCHEMA` (5A), `STEP 5 - BAG GRAMMAR` (5B), `STEP 6`, `STEP 6.1`, `STEP 6.2`
  - Phase 3 reads: `STEP 7`, `STEP 7.1`, `STEP 8`, `STEP 9`, `STEP 10`, `STEP 11`
  - Phase 4 reads: `boss_level_1`, `boss_level_2`, `boss_level_3`, `boss_level_4`
- [x] Cross-cutting: `CREATOR_NOTES.md` (vision; QUEEN reads at engagement), `CLARIFICATION.md` if present (read by QUEEN at engagement)

**Output:** `workflows/LANG_DEV_V2/spec/STEP_DOC_INVENTORY.md` ✓

**Do NOT (Part 1):**
- Do not delete `workflows/LANG_DEV/` files — v1 is preserved
- Do not edit `workflows/LANG_DEV/LANGS_DEV_RDC/` — that's the source corpus
- Do not commit `WHY_V2.md` until COURT ruling on STEP 5 ordering is restated verbatim from `INPROGRESS.md`

---

## Part 2 — Per-phase artifact contracts (the deliverable map)

These contracts BIND each task to concrete output files + acceptance commands, fixing v1's "generic deliverables" defect. They are the source of truth for PHASE_QA verdicts.

### T-2.1 — Phase 1 contract
- [x] One `contracts/PHASE_01_CONTRACT.md` covering T-01.1, T-01.1.1 (on-demand), T-01.2, T-01.3
- [x] Per task: input files, output filenames, output schemas (JSON shape examples), acceptance command, completion criteria, do-NOT list
- [x] Source: copy from `LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_TODO.md` verbatim where possible

**Output:** `workflows/LANG_DEV_V2/contracts/PHASE_01_CONTRACT.md`
**Acceptance:** Every task in `LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_TODO.md` has a matching contract entry.

### T-2.2 — Phase 2 contract
- [x] Cover T-02.1, T-02.2 (STEP 5A DECISIONS_SCHEMA), T-02.3 (STEP 5B BAG_GRAMMAR), T-02.4 + T-02.4.1 + T-02.4.2
- [x] **CRITICAL:** Honor COURT #1 SYNTHESIS ruling — `STEP_05A` is DECISIONS_SCHEMA (format first), `STEP_05B` is BAG_GRAMMAR (operators second). v1 has these reversed.
- [x] Document `<library>_decisions.json` schema as the central artifact (everything Phase 3 consumes derives from it)

**Output:** `workflows/LANG_DEV_V2/contracts/PHASE_02_CONTRACT.md`
**Acceptance:** `grep -A2 'STEP_05A' workflows/LANG_DEV_V2/contracts/PHASE_02_CONTRACT.md | grep -i 'DECISIONS'` passes (not BAG_GRAMMAR).

### T-2.3 — Phase 3 contract
- [x] Cover T-03.1 (Lexer), T-03.1.1 (Validator), T-03.2 (Parser), T-03.3 (Typer), T-03.4 (Classifier), T-03.5 (Solver)
- [x] Per stage: required test suite outline, test file naming (`test_*.py`), error-collection discipline (don't stop at first error)
- [x] Solver acceptance MUST include the shuffle test (100+ iterations, 10+ diverse bags)
- [x] Round-trip test required for Lexer

**Output:** `workflows/LANG_DEV_V2/contracts/PHASE_03_CONTRACT.md`

### T-2.4 — Phase 4 contract
- [x] Cover T-04.1 (Executor), T-04.2 (Optimizer), T-04.3 (Error Reporter), T-04.4 (Debugger)
- [x] Optimizer: golden rule (`optimized_output == original_output`) is a verifiable acceptance, not just a statement
- [x] Error Reporter: unified rendering across all 6 prior stages' error types
- [x] Debugger: snapshot deep-copy discipline (time travel breaks if shared)

**Output:** `workflows/LANG_DEV_V2/contracts/PHASE_04_CONTRACT.md`

### T-2.5 — Cross-phase artifact catalog
- [x] One-page summary listing every artifact across all 4 phases with its producing task and consuming tasks
- [x] Distinguish workspace artifacts (`workspace/STEP_XX/...`) from authoritative artifacts (`<library>_decisions.json`, `pcfg_weights_initial.json`)
- [x] Specify the workspace_manifest.json schema (single-source-of-truth for what-is-where across phases)

**Output:** `workflows/LANG_DEV_V2/contracts/ARTIFACT_CATALOG.md`

**Do NOT (Part 2):**
- Do not invent acceptance commands not present in the RDC TODOs — copy them
- Do not skip the shuffle test in Phase 3 — it is THE methodology's correctness criterion
- Do not promote Phase 4 to optional — v1's biggest defect is excluding Phase 4 entirely

---

## Part 3 — State machine: `LANG_DEV_V2_WORKFLOW.json`

### T-3.1 — JSON skeleton
- [x] Top-level fields: `version` (`2.0.0-DRAFT`), `name`, `description`, `trigger`, `relationship_to_other_workflows`, `inputs`, `phase_model`, `roles`, `units`, `flow`, `verdicts`, `loop_limits`, `hard_rules`, `document_conventions`, `known_tbds`
- [x] `trigger.phrases`: `LANG_DEV_V2_WORKFLOW`, `LANG_DEV_V2`, `langdev v2`, `language development v2`
- [x] `trigger.on_engage`: read this JSON, SHARED docs, V2-specific worker docs, validate inputs, ack & wait

**Output:** `workflows/LANG_DEV_V2/LANG_DEV_V2_WORKFLOW.json` (skeleton only)

### T-3.2 — `inputs` block
- [x] Mirror `INPUT_CONTRACT.md`: `target_library`, `nexus_reports_dir` (NEW), `workspace_dir`, `step_source_dir`
- [x] Per input: required/optional, validation rules, ESCALATE-on-missing semantics
- [x] Sample invocation example

### T-3.3 — `phase_model` block
- [x] 4 phases with task lists; each task has: id, source_doc(s) array (multi-doc!), depends_on array, parallelism (sequential|parallel-with-siblings|on-demand), output_contract reference
- [x] Sub-task notation (T-02.4.1, T-02.4.2 nested under T-02.4)
- [x] PHASE → METHODOLOGY verdict relationship

### T-3.4 — `roles` block
- [x] QUEEN (orchestrator, workflow-agnostic, reads SHARED)
- [x] PHASE_EXECUTOR (replaces v1 STEP_EXECUTOR; reads MULTIPLE assigned docs per task)
- [x] PHASE_QA (replaces v1 STEP_QA; bounded by task's contract from Part 2)
- [x] METHODOLOGY_INTEGRATOR (NEW; runs after Phase 4 to execute methodology-level acceptance — shuffle test + compression check on a real example library)

### T-3.5 — `units` block
- [x] PHASE_TASK_UNIT: PHASE_EXECUTOR → PHASE_QA, sequential, atomic, retry-bounded
- [x] PARALLEL_BATCH: where Phase 4 components run in parallel; aggregate verdicts
- [x] METHODOLOGY_INTEGRATION_UNIT: terminal unit running shuffle + compression acceptance

### T-3.6 — `flow` block
- [x] PRESTEP: validate all 4 inputs; verify nexus reports list against required set; init workspace + manifest; init INPROGRESS entry
- [x] PHASE 1: serial T-01.1 → T-01.2 → T-01.3 (T-01.1.1 on-demand)
- [x] PHASE 2: serial T-02.1 → T-02.2 → T-02.3 → T-02.4 (with sub-tasks T-02.4.1, T-02.4.2 sequential within T-02.4)
- [x] PHASE 3: serial T-03.1 → T-03.1.1 → T-03.2 → T-03.3 → T-03.4 → T-03.5
- [x] PHASE 4: T-04.3 sequential (depends on prior stages defining error types) — but T-04.1, T-04.2, T-04.4 can run in parallel after T-04.3 lands the unified Error type
- [x] FINALIZE: METHODOLOGY_INTEGRATOR runs end-to-end demo + shuffle test + compression check; emits CLEAN_RUN

### T-3.7 — `verdicts` block
- [x] Per-task: TASK_PASS, TASK_RETRY, TASK_ESCALATE
- [x] Per-phase: PHASE_GREEN_LIGHT, PHASE_HOLD (sub-task escalated), PHASE_BLOCKED
- [x] Methodology-level: METHODOLOGY_GREEN_LIGHT (all 4 phases + integration pass), METHODOLOGY_INCOMPLETE (any phase escalated), ABORTED

### T-3.8 — `hard_rules` block
- [x] Inherit from v1: no-fabrication, no-step-doc-modification, no-no-verify, INPROGRESS prepend-only, workspace discipline
- [x] NEW: `phase_4_required` (cannot METHODOLOGY_GREEN_LIGHT without Phase 4 complete)
- [x] NEW: `step_5_ordering` (5A is DECISIONS_SCHEMA, 5B is BAG_GRAMMAR — per COURT #1 SYNTHESIS)
- [x] NEW: `nexus_reports_required` (cannot proceed past PRESTEP without all required nexus reports)
- [x] NEW: `methodology_acceptance` (shuffle test + compression check are blocking gates for CLEAN_RUN)
- [x] DROPPED from v1: `monolithic_serial` (replaced with per-phase parallelism declarations); `boss_level_roles_not_integrated` (Phase 4 IS integrated)

### T-3.9 — JSON validation
- [x] `python -c "import json; json.load(open('workflows/LANG_DEV_V2/LANG_DEV_V2_WORKFLOW.json'))"` returns 0
- [x] All `output_contract` references resolve to files under `workflows/LANG_DEV_V2/contracts/`
- [x] All `source_doc(s)` references resolve to files under `step_source_dir`

**Output:** `workflows/LANG_DEV_V2/LANG_DEV_V2_WORKFLOW.json` (complete)
**Acceptance:** JSON parses; references resolve; spec docs from Part 1 are linked.

**Do NOT (Part 3):**
- Do not copy v1 JSON wholesale — the structure is wrong (16 flat phases). Build the 4-phase structure fresh.
- Do not name a verdict "PHASE_GREEN_LIGHT" if any sub-task is on-demand-skipped — distinguish completed vs intentionally-skipped
- Do not include `boss_level_roles_not_integrated` even as a deprecation note — Phase 4 is in scope, period

---

## Part 4 — Worker role docs

### T-4.1 — `WORKER_PHASE_EXECUTOR.md`
- [x] Replaces v1 `WORKER_STEP_EXECUTOR.md`
- [x] Multi-doc read discipline (full-read every doc in task's `source_docs` array)
- [x] Output discipline bound to contract (Part 2)
- [x] Workspace manifest update protocol (append entry per task; never rewrite)
- [x] Honest-ambiguity rule: if STEP doc + retrofit doc disagree, surface it; do not silently pick one

**Output:** `workflows/LANG_DEV_V2/WORKER_PHASE_EXECUTOR.md`

### T-4.2 — `WORKER_PHASE_QA.md`
- [x] Replaces v1 `WORKER_STEP_QA.md`
- [x] Authority bounded by `contracts/PHASE_<N>_CONTRACT.md` for the assigned task (NOT by the STEP doc alone — contract is the binding spec)
- [x] Verdict: TASK_PASS / TASK_FAIL with structured findings (severity, criterion violated, evidence)
- [x] Acceptance commands MUST be run; verbatim output included in QA report

**Output:** `workflows/LANG_DEV_V2/WORKER_PHASE_QA.md`

### T-4.3 — `WORKER_METHODOLOGY_INTEGRATOR.md`
- [x] NEW role; no v1 equivalent
- [x] Runs after Phase 4 PHASE_GREEN_LIGHT; consumes the full workspace
- [x] Builds end-to-end demo on the produced DSL (source string → executed output)
- [x] Runs shuffle test (100+ iterations on 10+ diverse bags)
- [x] Verifies ~20:1 compression at the main tier from `primitives_catalog.json` vs target library API surface
- [x] Emits METHODOLOGY_GREEN_LIGHT or escalates with specific failure

**Output:** `workflows/LANG_DEV_V2/WORKER_METHODOLOGY_INTEGRATOR.md`

### T-4.4 — Recovery worker decision
- [x] Decide: is T-01.1.1 RECOVER a separate worker doc, or just a different invocation of PHASE_EXECUTOR with `recovery_mode: true`?
- [x] Recommendation (lean): just a parameter on PHASE_EXECUTOR (avoid worker proliferation; keep recovery discipline encoded in `STEP 1.1` source doc + the contract)
- [x] Document the decision in `spec/RECOVERY_MODEL.md`

**Output:** `workflows/LANG_DEV_V2/spec/RECOVERY_MODEL.md`

**Do NOT (Part 4):**
- Do not duplicate content already in `LANGS_DEV_RDC/PHASE_*_ARCH.md` — link to it instead
- Do not write a worker doc that re-encodes the methodology — the STEP source docs are the methodology; workers execute against them

---

## Part 5 — Methodology-level acceptance machinery

### T-5.1 — Reference implementation library choice
- [x] Pick ONE small target library for the integration test (recommended: `pandas` subset, since RDC examples reference it throughout)
- [x] Document the choice + rationale in `spec/REFERENCE_LIBRARY.md`
- [x] Where the nexus reports for it live (recommend: `workflows/LANG_DEV_V2/test_target/<library>_nexus/`)

**Output:** `workflows/LANG_DEV_V2/spec/REFERENCE_LIBRARY.md`

### T-5.2 — Shuffle test harness spec
- [x] Define the test: take produced DSL + sample bag → solve → shuffle bag 100 times → solve each → assert all results equal
- [x] Diversity criteria: 10+ bags covering different operator combinations (sequence, compound, alternative, optional)
- [x] Determinism check across runs (run the same bag twice on different machines if possible)
- [x] Output: machine-readable PASS/FAIL + per-iteration log

**Output:** `workflows/LANG_DEV_V2/spec/SHUFFLE_TEST_SPEC.md`

### T-5.3 — Compression-ratio measurement spec
- [x] Define what counts as "main tier" (Tier 2 COGNITIVE per PEDAGOGY.md)
- [x] Numerator: count of primitives at main tier from `primitives_catalog.json`
- [x] Denominator: count of public API operations in target library
- [x] Threshold: ≥ ~20:1 (≥ 18:1 with rationale, < 18:1 escalates)

**Output:** `workflows/LANG_DEV_V2/spec/COMPRESSION_SPEC.md`

### T-5.4 — End-to-end demo spec
- [x] Define the canonical demo input: a known DSL source string for the reference library
- [x] Expected output structure (DataFrame shape, error reporter rendering, etc.)
- [x] Optimizer correctness check: `optimized.execute() == original.execute()` on the demo input

**Output:** `workflows/LANG_DEV_V2/spec/E2E_DEMO_SPEC.md`

**Do NOT (Part 5):**
- Do not skip the shuffle test in METHODOLOGY_GREEN_LIGHT gating — it is the acid test
- Do not pick a target library so large that running the full methodology in CI takes hours — start small

---

## Part 6 — Walkthroughs

Mental traces for QUEEN to use as reference. Pattern follows BOOK and ORGANIZE walkthrough docs.

### T-6.1 — `walkthroughs/HAPPY_PATH_WALKTHROUGH.md`
- [x] All 4 phases pass first attempt; methodology integration succeeds
- [x] Per-phase narration: what QUEEN spawns, what PHASE_EXECUTOR produces, what PHASE_QA verifies
- [x] Final METHODOLOGY_GREEN_LIGHT report

### T-6.2 — `walkthroughs/RECOVERY_WALKTHROUGH.md`
- [x] T-01.1 fails completion criteria → on-demand T-01.1.1 RECOVER spawns → recovery escapes local optimum → T-01.1 passes on retry

### T-6.3 — `walkthroughs/SHUFFLE_FAIL_WALKTHROUGH.md`
- [x] All 4 phases pass per-task QA, but METHODOLOGY_INTEGRATOR's shuffle test fails (e.g., priority key under-specified)
- [x] METHODOLOGY_INCOMPLETE verdict; QUEEN escalates with the specific shuffle-test diff
- [x] Human resumes from T-03.5 (Solver) with the failure data as context

### T-6.4 — `walkthroughs/PHASE_4_PARALLEL_WALKTHROUGH.md`
- [x] T-04.3 (Error Reporter) lands first
- [x] T-04.1 / T-04.2 / T-04.4 spawn in parallel
- [x] Aggregation logic; what happens if one of the three FAILs while others PASS

**Do NOT (Part 6):**
- Do not write speculative walkthroughs for situations the workflow doesn't yet handle — surface them as known_tbds first

---

## Part 7 — Test scaffolding

### T-7.1 — Test target directory
- [x] Create `workflows/LANG_DEV_V2/test_target/` with:
  - Minimal target library snippet (or symlink to a real one if size allows)
  - Pre-generated nexus reports for it
  - Expected workspace_manifest.json from a successful run (golden output)

**Output:** `workflows/LANG_DEV_V2/test_target/`

### T-7.2 — JSON validators
- [x] `validate_decisions_schema.py` (validates `<library>_decisions.json` against the schema)
- [x] `validate_workspace_manifest.py` (validates per-phase manifest entries)
- [x] `validate_phase_outputs.py` (cross-references contract outputs vs actual workspace contents)

**Output:** `workflows/LANG_DEV_V2/test_target/validators/`

**Do NOT (Part 7):**
- Do not commit large nexus report outputs (>1MB) — use a tiny synthetic library

---

## Part 8 — External integration

### T-8.1 — `workflows/LANG_DEV_V2/README.md` finalize
- [x] Pattern follows `workflows/BOOK/README.md` and `workflows/ORGANIZE/README.md`
- [x] Sections: pipeline diagram, quickstart, key documents, relationship to other workflows, shared infrastructure
- [x] Reference `workflows/LANG_DEV/LANGS_DEV_RDC/` as the authoritative methodology source

**Output:** `workflows/LANG_DEV_V2/README.md`

### T-8.2 — `workflows/README.md` update
- [x] Add LANG_DEV_V2 to the directory structure block
- [x] Update the LANG_DEV summary section: rename to "LANG_DEV (v1 deprecated, see LANG_DEV_V2)"
- [x] Add LANG_DEV_V2 summary section with the 4-phase pipeline diagram + new trigger phrases
- [x] Update versioning section: LANG_DEV_V2 = `2.0.0-DRAFT`; LANG_DEV = `0.1.0-SUPERSEDED`

### T-8.3 — `workflows/CLAUDE_APPENDIX.md` update
- [x] Add `LANG_DEV_V2_WORKFLOW` to the trigger-phrase table
- [x] Add engagement verbatim: `"LANG_DEV_V2_WORKFLOW mode engaged. Inputs: target=<path>, nexus=<path>, workspace=<path>. Validating ... Ready."`
- [x] Add v2-specific hard rules to the appendix's hard rules section

### T-8.4 — `CLAUDE.md` (project) update
- [x] Add LANG_DEV_V2 row to the Workflow table
- [x] Update file-artifacts table with new outputs (`workspace_manifest.json` schema, METHODOLOGY_REPORT.md, etc.)
- [x] Add hard rules 24-28 (v2-specific): `step_5_ordering`, `nexus_reports_required`, `phase_4_required`, `methodology_acceptance`, `multi_doc_authority`

**Output:** `CLAUDE.md` (edits to existing file)

### T-8.5 — `workflows/SHARED/WORKER.md` update
- [x] Add LANG_DEV_V2 worker role index entries
- [x] Distinguish from v1 entries (which remain for historical reference)

**Do NOT (Part 8):**
- Do not auto-edit project CLAUDE.md without showing the diff first — user owns CLAUDE.md
- Do not remove v1 references from CLAUDE_APPENDIX or README — v1 is preserved, not deleted

---

## Part 9 — Verification + handoff

### T-9.1 — Internal review pass
- [x] Read all v2 files end-to-end as if you were a fresh QUEEN encountering them at engagement
- [x] Verify cross-references all resolve
- [x] Verify every contract task has acceptance command
- [x] Verify every hard rule is testable (not vague)

### T-9.2 — Dry-run mental walkthrough
- [x] Walk through HAPPY_PATH_WALKTHROUGH (T-6.1) against the actual JSON state machine (T-3.x)
- [x] Identify any state transitions that lack JSON encoding
- [x] Document gaps in `KNOWN_LIMITATIONS.md`

**Output:** `workflows/LANG_DEV_V2/KNOWN_LIMITATIONS.md`

### T-9.3 — First live run preparation
- [x] Choose a small reference library (per T-5.1) and pre-generate its nexus reports
- [x] Verify all 16 STEP docs + retrofits exist at the configured `step_source_dir`
- [x] Engage `LANG_DEV_V2_WORKFLOW` with the test target
- [x] Capture the PRESTEP report; verify all input validations pass
- [x] STOP before spawning Phase 1 workers — this TODO is buildout, not first-run

### T-9.4 — Final commit
- [x] All TODO checkboxes flipped
- [x] Single commit per part (or per task if commits would otherwise be huge)
- [x] Commit message convention: `LANG_DEV_V2: T-<part>.<n> — <task name>`

**Do NOT (Part 9):**
- Do not invoke the workflow on a real engagement until T-9.3 prep is complete
- Do not declare buildout done if any contract acceptance command is unrunnable

---

## Open questions to resolve during buildout (surface as `OPEN_QUESTIONS.md` if undecided after Part 3)

1. Should LANG_DEV_V2 actually be a workflow at all, or should the RDC's PHASE TODOs be handed directly to SDLC? (Cited at end of prior research report.)
2. Sub-task spawning model: single PHASE_EXECUTOR call per parent task that handles sub-tasks internally, vs. one spawn per sub-task. Implications for retry semantics.
3. Per-phase parallelism: is parallel execution within a phase worth the orchestration complexity? Phase 4 yes (per ARCH §1); Phase 2 sub-tasks marginal.
4. PCFG online learning: when (if ever) does the workflow update weights, or is that always a separate post-methodology step?
5. Multi-target executor: does v2 plan for one target library per run, or support multi-target generation in a single workflow invocation?
6. Versioning of `<library>_decisions.json`: schema versioning policy (independent of workflow versioning) — TBD or defer?
7. Boss_level worker subdivision: do we need 4 separate worker docs for Phase 4 (one per boss_level), or is one PHASE_EXECUTOR enough?

---

## Sequencing summary

```
Part 1 (scaffold + spec)
    ↓
Part 2 (contracts) ─┬─→ Part 3 (state machine JSON)
                    │           ↓
                    └─→ Part 4 (worker docs)
                                ↓
                    Part 5 (acceptance machinery)
                                ↓
                    Part 6 (walkthroughs)
                                ↓
                    Part 7 (test scaffolding)
                                ↓
                    Part 8 (external integration)
                                ↓
                    Part 9 (verification + handoff)
```

Parts 1-3 are foundational; Parts 4-7 build the operational layer; Part 8 wires v2 into the rest of the project; Part 9 is the close-out gate.

**Estimated effort:** 32 tasks across 9 parts. With clear contracts, most tasks are 30-60 min. Total buildout: ~16-24 hours of focused work.

---

*End of LANG_DEV_V2_BUILDOUT_TODO.*
