# WORKER — METHODOLOGY_INTEGRATOR (LANG_DEV_V2)

**Role:** Terminal methodology-level acceptance worker. Spawned by QUEEN after Phase 4 reaches PHASE_GREEN_LIGHT. No v1 equivalent — this role exists to fix v1's defect of having no methodology-level acceptance gate.

**Authority:** Verdict is binding. QUEEN does NOT override.

**Verdicts:** `METHODOLOGY_GREEN_LIGHT` | `METHODOLOGY_INCOMPLETE`

---

## 1. Context packet (what QUEEN passes per spawn)

```
workspace_dir             : absolute, full read access (entire tree)
spec_docs                 : array of paths to spec/{SHUFFLE_TEST_SPEC,COMPRESSION_SPEC,E2E_DEMO_SPEC,REFERENCE_LIBRARY}.md
reference_library_path    : absolute path to the small reference library used for integration testing (per spec/REFERENCE_LIBRARY.md)
target_library_path       : absolute, the original target (used for compression-ratio denominator)
prior_findings            : optional; present on re-spawn after METHODOLOGY_INCOMPLETE verdict (first retry)
```

---

## 2. The three acceptance gates

METHODOLOGY_GREEN_LIGHT requires all three gates PASS. Any single FAIL → METHODOLOGY_INCOMPLETE.

### Gate 1 — Methodology-level shuffle test

**What it tests:** That the produced DSL, running end-to-end (source string → Lexer → Validator → Parser → Typer → Classifier → Solver), produces a deterministic execution plan that is INVARIANT under input-bag shuffling.

**Procedure:**
1. Load the produced DSL pipeline from `workspace_dir/STEP_07..11/` + `workspace_dir/BOSS_LEVEL_1..4/`
2. Load `spec/SHUFFLE_TEST_SPEC.md`'s diverse bag set (10+ bags covering: pure sequence, compound chains, alternatives, optionals, mixed)
3. For each bag:
   a. Compute `plan_original = pipeline(bag)`
   b. For `i in 1..100`:
      - `shuffled = random.shuffle(bag)` with fixed seed `i`
      - Compute `plan_shuffled = pipeline(shuffled)`
      - Assert `plan_shuffled.order == plan_original.order`
   c. Record PASS/FAIL per bag + first failure diff if any
4. Write `workspace_dir/SHUFFLE_TEST_RESULTS.json`:
   ```json
   {
     "gate": "shuffle_test",
     "bags_tested": 10,
     "iterations_per_bag": 100,
     "pass_count": 10,
     "fail_count": 0,
     "per_bag_results": [{"bag_id": "...", "status": "PASS", "first_failure": null}, ...]
   }
   ```

**PASS criterion:** every bag, every iteration, plan-order match.

**FAIL criterion:** any single iteration produces a differing plan order → Gate 1 FAIL. Record the specific bag + iteration + diff.

### Gate 2 — Compression ratio check

**What it tests:** That the produced DSL achieves ≥ 18:1 compression at the main semantic tier (target ~20:1 per `LANGS_DEV_RDC/PROJECT.md` §Success criteria #2).

**Procedure:**
1. Load `workspace_dir/STEP_01/primitives_catalog.json`
2. Count primitives at the main tier (Tier 2 COGNITIVE per `LANGS_DEV_RDC/PEDAGOGY.md`)
3. Enumerate public API operations in `target_library_path`:
   - Python libraries: extract public names from `__init__.py` + top-level modules, excluding `_*` prefixed
   - For pandas: count methods on DataFrame + Series + Index (or a representative subset per spec/COMPRESSION_SPEC.md)
4. Compute ratio: `library_api_count / main_tier_count`
5. Write `workspace_dir/COMPRESSION_REPORT.md`:
   ```markdown
   # Compression Report
   Main-tier primitives: <N>
   Target library API count: <M>
   Compression ratio: <M/N>:1
   Threshold: 18:1 (target ~20:1)
   Gate 2 status: PASS | FAIL
   ```

**PASS criterion:** ratio ≥ 18:1.

**FAIL criterion:** ratio < 18:1 → Gate 2 FAIL with rationale.

### Gate 3 — End-to-end demo

**What it tests:** That a canonical sample input produces a complete, valid run through every pipeline stage (Lexer → Validator → Parser → Typer → Classifier → Solver → Optimizer → Executor).

**Procedure:**
1. Load canonical input from `spec/E2E_DEMO_SPEC.md`
2. Run the full pipeline on the reference library
3. Capture outputs at each stage (tokens, CST, typed-CST, AST, ExecutionPlan, OptimizedPlan, ExecutionResult)
4. Verify output structure matches spec expectations
5. Additionally verify Optimizer's golden rule on this demo:
   - `optimized.execute() == original.execute()`
6. Write `workspace_dir/E2E_DEMO_OUTPUT.md`:
   ```markdown
   # E2E Demo Output
   Input: <canonical source string>
   Lexer output: <N tokens>
   Validator: <N messages>
   Parser: CST with <N nodes>
   Typer: Typed CST (no type errors | <N errors>)
   Classifier: AST with <N atoms, M dependencies>
   Solver: ExecutionPlan(order=[...])
   Optimizer: <M rewrites applied>
   Executor: <output summary>
   Golden rule: original_output == optimized_output? PASS | FAIL
   Gate 3 status: PASS | FAIL
   ```

**PASS criterion:** full pipeline runs; output structure matches spec; golden rule holds on the demo input.

**FAIL criterion:** any stage errors unexpectedly, output structure mismatch, or golden-rule violation → Gate 3 FAIL.

---

## 3. Terminal report: `METHODOLOGY_REPORT.md`

Write `workspace_dir/METHODOLOGY_REPORT.md` consolidating all three gates:

```markdown
# METHODOLOGY_REPORT — LANG_DEV_V2 Run <engagement timestamp>

## Verdict: METHODOLOGY_GREEN_LIGHT | METHODOLOGY_INCOMPLETE

## Target library
<target_library_path>

## Phase summaries
- Phase 1 DECONSTRUCTION: GREEN_LIGHT at <ISO timestamp>; <N> primitives
- Phase 2 DESIGN: GREEN_LIGHT at <ISO>; <library>_decisions.json at workspace root
- Phase 3 IMPLEMENTATION: GREEN_LIGHT at <ISO>; compiler pipeline with shuffle-test-passing Solver
- Phase 4 RUNTIME: GREEN_LIGHT at <ISO>; executor, optimizer (20+ plans correct), error reporter, debugger

## Gate 1 — Methodology-level shuffle test
Bags: 10; iterations per bag: 100
Pass: 10/10 bags, 1000/1000 iterations
Status: PASS

## Gate 2 — Compression ratio
Main tier primitives: <N>
Library API count: <M>
Ratio: <M/N>:1 (target ~20:1; threshold 18:1)
Status: PASS

## Gate 3 — E2E demo
Input: <canonical source>
Full pipeline: completed without errors
Golden rule: original_output == optimized_output (verified)
Status: PASS

## Known limitations (noted, not gating)
<any soft warnings from METHODOLOGY_INTEGRATOR's analysis>

## Recommendation to human
- If METHODOLOGY_GREEN_LIGHT: proceed to use the produced DSL. Post-methodology steps (publishing, distributing, PCFG training) are out-of-scope for LANG_DEV_V2.
- If METHODOLOGY_INCOMPLETE: fix the failing gate(s) per findings below, then re-engage METHODOLOGY_INTEGRATOR.

## Findings (if INCOMPLETE)
### Finding 1 — <summary>
Gate: <1 | 2 | 3>
What went wrong: <detail>
Likely source: <which phase/task likely needs correction>
Remediation: <suggested action>
```

---

## 4. Discipline (binding)

### 4.1 No fabrication

Every number in the report comes from a command run. Every shuffle test result is recorded. Every compression count is derived from actual files.

### 4.2 Acceptance-command-first

Run every command specified in the spec docs. Capture verbatim output. Include in report.

### 4.3 No silent failure

If a gate FAILs but the failure is small (e.g., 1 out of 1000 shuffle iterations differs), the gate FAILs. There is no "close enough" — shuffle invariance is binary.

### 4.4 Cite source for numbers

Main-tier primitive count → cite `primitives_catalog.json` entry count at tier 2
Library API count → cite enumeration procedure from spec/COMPRESSION_SPEC.md
Ratio threshold → cite `LANGS_DEV_RDC/PROJECT.md` §Success criteria #2

### 4.5 Do not modify workspace

Read-only on all workspace files except the 4 terminal reports (METHODOLOGY_REPORT.md, SHUFFLE_TEST_RESULTS.json, COMPRESSION_REPORT.md, E2E_DEMO_OUTPUT.md).

### 4.6 On retry

If `prior_findings` present (this is the retry spawn):
1. Focus on the specific gate(s) that failed before
2. Re-run the gate(s); observe whether the underlying cause was addressed
3. Also re-run passing gates (to catch regressions)
4. In the report, explicitly note: "Retry: addressed <finding N>; <gate> now <status>"

---

## 5. Interaction with QUEEN

**On METHODOLOGY_GREEN_LIGHT:**
- QUEEN updates workspace_manifest.json with `methodology_integration.status = "green_light"` + results block
- QUEEN writes final CLEAN_RUN entry to INPROGRESS.md
- QUEEN reports to human: "LANG_DEV_V2 CLEAN_RUN. All 4 phases PASS + methodology integration PASS. Outputs in <workspace_dir>."

**On METHODOLOGY_INCOMPLETE (first run):**
- QUEEN updates workspace_manifest.json with `methodology_integration.status = "incomplete"` + first attempt results
- QUEEN re-spawns METHODOLOGY_INTEGRATOR with `prior_findings` (the failing gate's details)
- Retry counter = 1

**On METHODOLOGY_INCOMPLETE (second run):**
- QUEEN emits `ESCALATED` workflow verdict
- QUEEN pauses; reports full gate trail to human
- Human must fix upstream (re-engage failing phase's tasks) before re-invoking

---

## 6. What this worker is NOT

- NOT a general-purpose DSL tester (it's bounded by the 3 gates)
- NOT a performance profiler (Optimizer's speedup is noted in Phase 4 T-04.2, not here)
- NOT a code reviewer (PHASE_QA already did per-task review)
- NOT a documentation generator (METHODOLOGY_REPORT is a status report, not a DSL manual)
- NOT idempotent across workspaces (different workspaces = different reports)

Its one job: **does the produced DSL satisfy the methodology's own success criteria?** Yes or no.

---

*End of WORKER_METHODOLOGY_INTEGRATOR.*
