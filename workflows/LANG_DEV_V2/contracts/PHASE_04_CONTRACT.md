# PHASE_04_CONTRACT — RUNTIME

**Purpose:** Bind every Phase 4 task (Executor, Optimizer, Error Reporter, Debugger) to concrete output files + acceptance commands.

**Authoritative methodology source:** `workflows/LANG_DEV/LANGS_DEV_RDC/PHASE_04_RUNTIME_TODO.md` and `..._ARCH.md`.

**Phase ordering (per dependency graph in `PHASE_MODEL.md`):**
- T-04.3 (Error Reporter) MUST land first — it defines the unified Error type all stages will emit against
- T-04.1 (Executor) and T-04.2 (Optimizer) can spawn in parallel after T-04.3 PASS
- T-04.4 (Debugger) depends on T-04.1 (Executor) per `LANGS_DEV_RDC/PHASE_04_RUNTIME_TODO.md` opening note

**Phase verdict:** PHASE_GREEN_LIGHT iff all four T-04.x tasks reach TASK_PASS.

---

## T-04.3 — Error Reporter (runs first)

**Source docs:**
- `boss_level_3_error_reporter_rules.md`

**Inputs:**
- All Phase 3 stage error types (LexerError, ValidationMessage, ParseError, TypeError, SemanticError, SolverError) — must already exist in `workspace_dir/STEP_*/`
- (Future) ExecutionError from T-04.1 — adapter added when T-04.1 lands

**Required outputs:**

| File | Purpose |
|---|---|
| `BOSS_LEVEL_3/error_reporter.py` | Render dispatch |
| `BOSS_LEVEL_3/unified_error.py` | Error base type with severity, category, location, message, suggestion, context, related_errors |
| `BOSS_LEVEL_3/rendering.py` | Box-drawn format, line pointers, ASCII art |
| `BOSS_LEVEL_3/fuzzy_match.py` | Levenshtein wrapper (shared with Validator) |
| `BOSS_LEVEL_3/test_error_reporter.py` | Test suite |

**Completion criteria:**

1. Unified Error type adapts every stage error type from Phase 3 (Lexer/Validator/Parser/Typer/Classifier/Solver)
2. Severity enum: `ERROR`, `WARNING`, `HINT`
3. Category enum: `SYNTAX`, `TYPE`, `SEMANTIC`, `CONSTRAINT`, `RUNTIME`
4. Box-drawn render format with line pointers (Unicode box-drawing chars)
5. Levenshtein-based "Did you mean?" suggestions for unknown names (distance 1-2 → "Did you mean X?"; 3+ → "Similar: X, Y, Z")
6. Error collection (gather all before rendering)
7. Error grouping (cascading errors collapsed when one explains the others)
8. Priority ordering (most-important errors first)
9. Color-aware output (terminal color codes if `sys.stdout.isatty()`, plain if file/pipe)
10. Every rendered error answers WHAT / WHERE / WHY / HOW
11. Test suite covers: each stage's error type, multi-error display, "Did you mean" suggestions, cascading collapse, color/plain rendering

**Acceptance command:**
```bash
cd workspace_dir/BOSS_LEVEL_3 && python -m pytest test_error_reporter.py -v --tb=short
# Expected: all tests pass; renderings consistent across stages
```

**Do NOT:**
- Do not silently swallow errors
- Do not produce stage-specific rendering — single unified format
- Do not over-suggest (distance 3+ means "similar", not "did you mean")

**Verdict:** All tests pass → `TASK_PASS`.

---

## T-04.1 — Executor (parallel-after-T-04.3)

**Source docs:**
- `boss_level_1_executor_rules.md`

**Inputs:**
- `<library>_decisions.json` (atom catalog → executor registry)
- ExecutionPlan shape from T-03.5
- Runtime config (file paths, flags)

**Required outputs:**

| File | Purpose |
|---|---|
| `BOSS_LEVEL_1/executor.py` | Core execution loop |
| `BOSS_LEVEL_1/execution_context.py` | ExecutionContext, ExecutionResult |
| `BOSS_LEVEL_1/atom_executors.py` | Registry mapping `atom_name → executor_fn` |
| `BOSS_LEVEL_1/trace.py` | TraceEntry + pretty-printer |
| `BOSS_LEVEL_1/test_executor.py` | Test suite |
| `BOSS_LEVEL_1/e2e_test.py` | End-to-end demo (ExecutionPlan → real library output) |

**Completion criteria:**

1. ExecutionContext holds: `current_df`, `current_grouped`, `config`, `trace` (list), `errors` (list)
2. Core loop iterates `ExecutionPlan.order`
3. Per-atom executor function for EVERY atom in `decisions.json` — registry completeness validated at startup
4. Argument extraction helpers (`get_string_arg`, `get_number_arg`, `get_list_arg`, etc.)
5. TraceEntry per atom: `atom_name, rows_before, rows_after, columns, time_ms, notes`
6. Pretty-printed execution trace (table + waterfall)
7. Per-atom try/except wrapping → ExecutionError with `step` + `atom` context
8. Empty DataFrame handled as warning, not error
9. Grouped state transitions supported (GroupBy → grouped; Agg consumes grouped)
10. Integration test: full E2E from ExecutionPlan to actual library output (DataFrame, etc.)

**Acceptance commands:**
```bash
cd workspace_dir/BOSS_LEVEL_1 && python -m pytest test_executor.py -v --tb=short
# Expected: all tests pass

cd workspace_dir && python BOSS_LEVEL_1/e2e_test.py
# Expected: produces valid library output (e.g., 5-row DataFrame for the sample plan)
```

**Do NOT:**
- Do not have executor gaps (every decisions.json atom needs an implementation)
- Do not skip trace collection
- Do not let errors crash silently — always catch + wrap with context
- Do not mutate the ExecutionPlan (it is immutable input)

**Verdict:** All tests + e2e pass → `TASK_PASS`.

---

## T-04.2 — Optimizer (parallel-after-T-04.3)

**Source docs:**
- `boss_level_2_optimizer_rules.md`

**Inputs:**
- ExecutionPlan from T-03.5
- AST dependencies from T-03.4 (for column-dependency respect)

**Required outputs:**

| File | Purpose |
|---|---|
| `BOSS_LEVEL_2/optimizer.py` | Optimization pass driver |
| `BOSS_LEVEL_2/optimization_rules.py` | Per-rule implementations |
| `BOSS_LEVEL_2/cost_model.py` | Row-count estimation |
| `BOSS_LEVEL_2/test_optimizer.py` | Test suite |
| `BOSS_LEVEL_2/correctness_suite.py` | Golden-rule verification |

**Completion criteria:**

1. **Rule 1: Filter pushdown** — move filters before expensive operations, respecting column dependencies
2. **Rule 2: Filter combination** — `Filter("a>0"), Filter("b<10")` → `Filter("(a>0) and (b<10)")`
3. **Rule 3: Sort pushdown (push LATE)** — sort after filters reduce data, but NOT past Head/Tail
4. **Rule 4: Compute combination** — multiple `Compute`s → `ComputeMulti`
5. **Rule 5: Dead code elimination** — computed columns never used → remove
6. OptimizationRule interface: `apply`, `guard`, `preserve-correctness invariant`
7. OptimizationPass (sequence of rules, fixed-point iteration or bounded passes)
8. OptimizedPlan = original plan + rewrite log
9. Cost model for speedup estimation (row-count waterfall)
10. **Disable-all-optimizations escape hatch** present (env var or config flag)
11. Test suite covers: each rule in isolation, rule interaction, **correctness preservation** (golden-rule), counterexamples where rule should NOT apply
12. **Golden rule verified:** for 20+ diverse plans, `optimized.execute() == original.execute()` (this is THE Optimizer acceptance gate)

**Acceptance commands:**
```bash
cd workspace_dir/BOSS_LEVEL_2 && python -m pytest test_optimizer.py -v --tb=short
# Expected: all tests pass

cd workspace_dir && python BOSS_LEVEL_2/correctness_suite.py
# Expected: 20+ plans pass; for each: 'CORRECT: optimized == original'
```

**Do NOT:**
- Do not implement a rule without correctness-preservation proof or test
- Do not skip the disable-optimizations flag
- Do not push Sort past Head/Tail (semantic violation)
- Do not violate the golden rule under any circumstance

**Verdict:**
- All tests + correctness_suite (20+ plans) pass → `TASK_PASS`
- Any golden-rule violation → `TASK_FAIL_ESCALATE` (no retry — this is correctness, not bug)

---

## T-04.4 — Debugger (sequential-after-T-04.1)

**Source docs:**
- `boss_level_4_debugger_rules.md`

**Inputs:**
- ExecutionPlan from T-03.5
- Executor from T-04.1

**Required outputs:**

| File | Purpose |
|---|---|
| `BOSS_LEVEL_4/debugger.py` | Top-level debugger entry |
| `BOSS_LEVEL_4/debug_session.py` | DebugSession state |
| `BOSS_LEVEL_4/snapshot.py` | Snapshot with deep-copy semantics |
| `BOSS_LEVEL_4/visualization.py` | Pipeline view, waterfall |
| `BOSS_LEVEL_4/commands.py` | Interactive command dispatch |
| `BOSS_LEVEL_4/test_debugger.py` | Test suite |
| `BOSS_LEVEL_4/demo_session.py` | Scripted demo |

**Completion criteria:**

1. DebugSession state: `plan, config, current_step, status (READY|RUNNING|PAUSED|DONE), snapshots, breakpoints, watches`
2. Snapshot: deep-copy of DataFrame per step (no shared state)
3. Stepping commands: `next, prev, continue, run, goto N`
4. Data inspection: `data, diff, stats, compare A B`
5. Breakpoints: at step (`b 3`), at atom type (`b Filter`), conditional (`b when rows < 100`), on-change (`b on column_add`)
6. Time travel: restore any saved snapshot
7. Visualization: ASCII pipeline view (current step highlighted), row count waterfall histogram
8. Interactive REPL (if interactive mode) OR scripted API (if not — preferred for testing)
9. Test suite covers: step forward, step back, jump, data inspection, all 4 breakpoint types, visualization, time-travel correctness

**Acceptance commands:**
```bash
cd workspace_dir/BOSS_LEVEL_4 && python -m pytest test_debugger.py -v --tb=short
# Expected: all tests pass

cd workspace_dir && python BOSS_LEVEL_4/demo_session.py
# Expected: scripted demo runs; output shows interactive debugging working
```

**Do NOT:**
- Do not share snapshots (must be deep-copied; shared state breaks time travel)
- Do not couple debugger to production executor path (keep it opt-in)
- Do not skip visualization (debugger UX matters)

**Verdict:** All tests + demo run → `TASK_PASS`.

---

## Phase 4 verdict gate

When all four T-04.x tasks `TASK_PASS`:
- The methodology has produced a runnable DSL
- QUEEN appends Phase 4 summary to `workspace_manifest.json`
- QUEEN writes Phase 4 completion entry to `INPROGRESS.md`
- QUEEN emits `PHASE_GREEN_LIGHT`; advances to **METHODOLOGY_INTEGRATION** (METHODOLOGY_INTEGRATOR worker spawn)

**The phase-level gate does NOT yet emit `METHODOLOGY_GREEN_LIGHT`** — that requires METHODOLOGY_INTEGRATOR to pass:
- Methodology-level shuffle test (full DSL, not just Solver)
- Compression ratio check (≥ ~20:1)
- E2E demo (full source-to-output pipeline working)

See `WORKER_METHODOLOGY_INTEGRATOR.md` for the integration spec.

---

*End of PHASE_04_CONTRACT.*
