# STEP_DOC_INVENTORY — Source Doc Read List per Task

**Purpose:** Bind every PHASE_EXECUTOR spawn to the specific source docs it must full-read. Fixes v1's "single-doc authority too narrow" defect.

**Authoritative sources:**
- `workflows/LANG_DEV/LANGS_DEV_RDC/INVENTORY.md` (the full 22-doc manifest)
- `workflows/LANG_DEV/LANGS_DEV_RDC/PHASE_<1-4>_TODO.md` (per-task source-doc references)
- `workflows/LANG_DEV/LANGS_DEV_RDC/EVALUATIONS.md` (per-doc contribution record)

---

## 1. The 22 source docs

| # | Filename | Role | Type |
|---|---|---|---|
| 1 | `STEP 1 - DECONSTRUCTION OPS.md` | Foundation | STEP |
| 2 | `STEP 1.1 - RECOVER OPS.md` | Recovery | STEP (retrofit) |
| 3 | `STEP 2 - DECONSTRUCTION OBJS.md` | Foundation | STEP |
| 4 | `STEP 3 - DECONSTRUCTION TYPES.md` | Foundation | STEP |
| 5 | `STEP 4 - ATOMICS.md` | Design | STEP |
| 6 | `STEP 5 - DECISIONS SCHEMA.md` | Design (5A) | STEP |
| 7 | `STEP 5 - BAG GRAMMAR.md` | Design (5B) | STEP |
| 8 | `STEP 6 - RULESETS AND DEFAULTS.md` | Design | STEP |
| 9 | `STEP 6.1 - THE_CONUNDRUM.md` | Design (philosophical) | STEP |
| 10 | `STEP 6.2 - PRE LEXER.md` | Design (token bridge) | STEP |
| 11 | `STEP 7 - LEXER.md` | Implementation | STEP |
| 12 | `STEP 7.1 - VALIDATOR.md` | Implementation | STEP |
| 13 | `STEP 8 - PARSER.md` | Implementation | STEP |
| 14 | `STEP 9 - TYPER.md` | Implementation | STEP |
| 15 | `STEP 10 - CLASSIFIER.md` | Implementation | STEP |
| 16 | `STEP 11 - SOLVER.md` | Implementation | STEP |
| 17 | `boss_level_1_executor_rules.md` | Runtime | BOSS_LEVEL |
| 18 | `boss_level_2_optimizer_rules.md` | Runtime | BOSS_LEVEL |
| 19 | `boss_level_3_error_reporter_rules.md` | Runtime | BOSS_LEVEL |
| 20 | `boss_level_4_debugger_rules.md` | Runtime | BOSS_LEVEL |
| 21 | `context.md` | Extended deconstruction rules | RETROFIT |
| 22 | `CREATOR_NOTES.md` | Vision | META |

**Notes on STEP 5 ordering:**
- Per COURT #1 SYNTHESIS in `LANGS_DEV_RDC/INPROGRESS.md`: `STEP_05A` is DECISIONS SCHEMA (file 6 above), `STEP_05B` is BAG GRAMMAR (file 7 above)
- Format-then-operators ordering: schema records what STEP 4 atomics produced; grammar layers operators on top
- v1 had these reversed; v2 enforces this ordering as a hard rule

**Notes on retrofits (per `LANGS_DEV_RDC/INVENTORY.md` §Retrofits noted):**
- `STEP 1.1` arrived 4h51m after `STEP 1` — recovery patch
- `context.md` arrived 2 days after main cluster — reads as APPENDIX to STEP 1 (multi-tier basis, primitive classification, hidden-primitive check, two-path check)
- `STEP 5 - DECISIONS SCHEMA.md` arrived 2 days after `STEP 5 - BAG GRAMMAR.md` — format spec complementing the philosophical doc
- All retrofits EXTEND prior content; none replaces.

---

## 2. Per-task read list

PHASE_EXECUTOR full-reads every doc in its task's `source_docs` array. Cross-phase artifacts (workspace files from prior phases) are listed in `prior_phase_outputs` rather than `source_docs` — they are inputs but not methodology specifications.

### Phase 1 — DECONSTRUCTION

| Task | source_docs (full-read) | prior_phase_outputs (read) |
|---|---|---|
| T-01.1 — Deconstruction Ops | `STEP 1 - DECONSTRUCTION OPS.md`, **`context.md`** | (none — first task) |
| T-01.1.1 — Recover Ops (on-demand) | `STEP 1.1 - RECOVER OPS.md` | T-01.1 partial outputs (primitives_catalog.json, deconstruction_notes.md) + STEP_QA findings |
| T-01.2 — Deconstruction Objs | `STEP 2 - DECONSTRUCTION OBJS.md` | T-01.1 outputs |
| T-01.3 — Deconstruction Types | `STEP 3 - DECONSTRUCTION TYPES.md` | T-01.1 + T-01.2 outputs |

**Why T-01.1 needs context.md:** per `LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_TODO.md` T-01.1 scope, the task requires the 5-type primitive classification (UNIVERSAL/STRUCTURAL/BRIDGE/GOAL/PHILOSOPHICAL), multi-tier basis (Tier 0-3), hidden-primitive check, and two-path check — all of which are in `context.md`, not `STEP 1`.

### Phase 2 — DESIGN

| Task | source_docs (full-read) | prior_phase_outputs (read) |
|---|---|---|
| T-02.1 — Atomics | `STEP 4 - ATOMICS.md`, `context.md` (re-read for primitive classification context) | All Phase 1 outputs |
| T-02.2 — Decisions Schema (5A) | `STEP 5 - DECISIONS SCHEMA.md` | T-02.1 outputs |
| T-02.3 — Bag Grammar (5B) | `STEP 5 - BAG GRAMMAR.md` | `<library>_decisions.json` from T-02.2 |
| T-02.4 — Rulesets and Defaults | `STEP 6 - RULESETS AND DEFAULTS.md` | `<library>_decisions.json` + bag_grammar_spec.md |
| T-02.4.1 — The Conundrum | `STEP 6.1 - THE_CONUNDRUM.md` | T-02.4 outputs |
| T-02.4.2 — Pre-Lexer | `STEP 6.2 - PRE LEXER.md` | T-02.4 + T-02.4.1 outputs |

**Why T-02.1 re-reads context.md:** primitive classification is referenced again when designing atoms (UNIVERSAL primitives → core atoms; STRUCTURAL → wildcard/passthrough patterns; BRIDGE → cross-domain atoms; GOAL → result-returning atoms; PHILOSOPHICAL → mutually exclusive alternatives). Per `LANGS_DEV_RDC/CLARIFICATION.md` §4.

### Phase 3 — IMPLEMENTATION

| Task | source_docs (full-read) | prior_phase_outputs (read) |
|---|---|---|
| T-03.1 — Lexer | `STEP 7 - LEXER.md` | token_inventory.md (from T-02.4.2), bag_grammar_spec.md (from T-02.3) |
| T-03.1.1 — Validator | `STEP 7.1 - VALIDATOR.md` | `<library>_decisions.json` (for KNOWN_ATOMS), tokens from T-03.1 |
| T-03.2 — Parser | `STEP 8 - PARSER.md` | bag_grammar_spec.md, validated tokens from T-03.1.1 |
| T-03.3 — Typer | `STEP 9 - TYPER.md` | `<library>_decisions.json` (for atom catalog), CST from T-03.2 |
| T-03.4 — Classifier | `STEP 10 - CLASSIFIER.md` | ruleset_spec.md (from T-02.4), Typed CST from T-03.3 |
| T-03.5 — Solver | `STEP 11 - SOLVER.md` | pcfg_weights_initial.json (from T-02.1), AST from T-03.4 |

### Phase 4 — RUNTIME

| Task | source_docs (full-read) | prior_phase_outputs (read) |
|---|---|---|
| T-04.3 — Error Reporter | `boss_level_3_error_reporter_rules.md` | All stage error types from Phase 3 (LexerError, ValidationMessage, ParseError, TypeError, SemanticError, SolverError) |
| T-04.1 — Executor | `boss_level_1_executor_rules.md` | `<library>_decisions.json` (atom catalog), ExecutionPlan shape from T-03.5 |
| T-04.2 — Optimizer | `boss_level_2_optimizer_rules.md` | ExecutionPlan + AST dependencies from T-03.4 |
| T-04.4 — Debugger | `boss_level_4_debugger_rules.md` | ExecutionPlan, Executor's atom_executors registry |

### Methodology Integration

| Phase | source_docs (full-read) | prior_phase_outputs (read) |
|---|---|---|
| METHODOLOGY_INTEGRATOR | (none — uses contract spec docs only) | All workspace artifacts; especially `<library>_decisions.json`, solver test suite, executor, primitives_catalog.json |

---

## 3. At-engagement reads (QUEEN, before any task spawn)

Per `INPUT_CONTRACT.md` §6 step 7, QUEEN reads these into main-conversation context at engagement:

| Doc | Why |
|---|---|
| `workflows/LANG_DEV_V2/LANG_DEV_V2_WORKFLOW.json` | The workflow itself |
| `workflows/SHARED/WORKER_QUEEN.md` | Orchestrator role |
| `workflows/SHARED/WORKER_PROTOCOL.md` | Non-negotiable worker contract |
| `workflows/SHARED/WORKER.md` | Master role index |
| `workflows/LANG_DEV_V2/WORKER_PHASE_EXECUTOR.md` | What workers it will spawn |
| `workflows/LANG_DEV_V2/WORKER_PHASE_QA.md` | What QA it will spawn |
| `workflows/LANG_DEV_V2/WORKER_METHODOLOGY_INTEGRATOR.md` | Terminal acceptance role |
| `step_source_dir/CREATOR_NOTES.md` | The methodology's vision (north star) |
| `workflows/LANG_DEV/LANGS_DEV_RDC/CLARIFICATION.md` | Philosophical framing (mental models per phase) |

QUEEN does NOT read the 16 STEP docs at engagement — those are read by PHASE_EXECUTOR per-task. CREATOR_NOTES + CLARIFICATION are the only methodology source docs QUEEN reads, because they are workflow-wide orientation rather than per-task spec.

---

## 4. Multi-doc read discipline

PHASE_EXECUTOR's `source_docs` parameter is an array. Discipline:

1. **Full-read every doc in the array, in array order.** No partial scans.
2. **Order matters when docs disagree.** First doc in array is the primary spec; subsequent docs EXTEND or REFINE it (retrofits). On unresolved conflict between docs, surface the conflict in the completion report — do not silently pick.
3. **Re-read on retry.** If TASK_FAIL_RETRY occurs, the re-spawned PHASE_EXECUTOR full-reads the source docs again (don't trust prior context).
4. **Cross-reference is explicit.** PHASE_EXECUTOR's completion report MUST cite which doc + section every output entry traces to. "Per STEP 1 §3" or "per context.md §Multi-tier basis" — not "per the spec."

PHASE_QA's authority is bounded by the source docs in this array PLUS the contract from `contracts/PHASE_<N>_CONTRACT.md`. Findings cite specific source-doc lines and contract-criterion violations.

---

## 5. Doc-to-task reverse index

For tracing which tasks a given doc influences (useful when editing a STEP doc to understand impact):

| Doc | Read by tasks |
|---|---|
| `STEP 1` | T-01.1 |
| `STEP 1.1` | T-01.1.1 (on-demand) |
| `STEP 2` | T-01.2 |
| `STEP 3` | T-01.3 |
| `STEP 4` | T-02.1 |
| `STEP 5 - DECISIONS SCHEMA` (5A) | T-02.2 |
| `STEP 5 - BAG GRAMMAR` (5B) | T-02.3 |
| `STEP 6` | T-02.4 |
| `STEP 6.1` | T-02.4.1 |
| `STEP 6.2` | T-02.4.2 |
| `STEP 7` | T-03.1 |
| `STEP 7.1` | T-03.1.1 |
| `STEP 8` | T-03.2 |
| `STEP 9` | T-03.3 |
| `STEP 10` | T-03.4 |
| `STEP 11` | T-03.5 |
| `boss_level_1` | T-04.1 |
| `boss_level_2` | T-04.2 |
| `boss_level_3` | T-04.3 |
| `boss_level_4` | T-04.4 |
| `context.md` | T-01.1, T-02.1 (re-read) |
| `CREATOR_NOTES.md` | (QUEEN at engagement) |

`CLARIFICATION.md` (in `LANGS_DEV_RDC/`, not in `step_source_dir`) is read by QUEEN at engagement; not by individual tasks.

---

## 6. Comparison to v1

| Aspect | v1 | v2 |
|---|---|---|
| Source docs declared | 16 (the 16 STEPs) | 22 (16 STEPs + 4 boss_level + context + CREATOR_NOTES) |
| Per-spawn read count | 1 | 1+ (array; T-01.1 reads 2; T-02.1 reads 2; rest read 1) |
| context.md acknowledged | No | Yes (T-01.1, T-02.1) |
| boss_level docs included | No (excluded by hard rule) | Yes (T-04.1-04.4) |
| CREATOR_NOTES read | No | Yes (QUEEN at engagement) |
| Retrofits handled | No (v1's flat phase model treats STEP 1.1 same as STEP 1) | Yes (T-01.1.1 on-demand triggered by T-01.1 FAIL) |
| Cross-doc conflict surfacing | Implicit | Explicit (PHASE_EXECUTOR must surface in completion report) |

---

*End of STEP_DOC_INVENTORY.*
