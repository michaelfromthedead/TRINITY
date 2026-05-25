# E2E_DEMO_SPEC — End-to-End Pipeline Acceptance Gate

**Purpose:** Specify how METHODOLOGY_INTEGRATOR runs the end-to-end demo (Gate 3).

**Authoritative source:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_TODO.md` Phase completion section + `LANGS_DEV_RDC/PHASE_04_RUNTIME_TODO.md` Phase completion section.

---

## 1. What the e2e demo verifies

That a canonical sample input flows through every stage of the produced DSL — Lexer → Validator → Parser → Typer → Classifier → Solver → Optimizer → Executor — and produces correct output. Plus the Optimizer's golden rule (`optimized.execute() == original.execute()`).

This is the "does the produced DSL actually work?" gate. Without it, METHODOLOGY_GREEN_LIGHT could be falsely emitted on a pipeline that passes per-stage tests but fails when wired together.

---

## 2. Canonical input (for reference library = `pandas_mini`)

```
LoadCSV('sample_data/small.csv'), Compute('z', 'a + b'), Filter('z > 10'), Sort('z'), Head(5)
```

**Why this input:**
- Exercises SOURCE atom (LoadCSV)
- Exercises TRANSFORM with column dependency (Compute creates `z`, Filter consumes `z`)
- Exercises COLUMN dependency detection (Filter must come AFTER Compute regardless of bag order)
- Exercises Sort + Head — Optimizer's pushdown rules potentially apply (sort pushdown, but NOT past Head)
- Exercises full sink-less pipeline (no SINK atom; result is the executor's `current_df`)

For non-pandas-mini targets, canonical input is constructed per the produced DSL's `<library>_decisions.json` to exercise SOURCE → TRANSFORM(s) → AGGREGATE(?) → LIMIT.

---

## 3. Sample data (for reference library)

`workflows/LANG_DEV_V2/test_target/sample_data/small.csv`:

```csv
id,a,b,cat
1,2,3,X
2,4,5,Y
3,1,1,X
4,8,9,Y
5,3,4,X
6,7,2,Y
7,5,6,X
8,9,1,Y
9,6,7,X
10,2,8,Y
```

10 rows × 4 columns. Small enough to verify by hand; large enough to exercise filtering and sorting.

---

## 4. Expected pipeline outputs

| Stage | Expected output |
|---|---|
| **Lexer** | ~25 tokens (NAME, LPAREN, STRING, RPAREN, COMMA, etc.) |
| **Validator** | 0 ValidationMessages (all atoms valid in pandas_mini decisions.json) |
| **Parser** | CST with Bag(items=[Atom(LoadCSV,...), Atom(Compute,...), Atom(Filter,...), Atom(Sort,...), Atom(Head,...)]) |
| **Typer** | Typed CST; no type errors; output_type = DATAFRAME |
| **Classifier** | AST: Pipeline(atoms=[5 SemanticAtoms], dependencies=[ COLUMN dep: Compute→Filter on 'z' ]) |
| **Solver** | ExecutionPlan(order=['LoadCSV', 'Compute', 'Filter', 'Sort', 'Head'], explanations=[...]) |
| **Optimizer** | OptimizedPlan: at minimum the order is preserved; possibly Filter pushed before Compute is NOT applicable (Filter depends on Compute's output). Sort pushdown not applicable past Head. |
| **Executor** | `pd.DataFrame` with ≤5 rows (Head limit), all rows where `a+b > 10`, sorted by `z` ascending |

Computed by hand on the sample data:
- Row 4 (a=8, b=9): z=17 ✓ pass filter
- Row 6 (a=7, b=2): z=9 ✗ fail filter
- Row 7 (a=5, b=6): z=11 ✓ pass filter
- Row 8 (a=9, b=1): z=10 ✗ fail filter (exclusive >)
- Row 9 (a=6, b=7): z=13 ✓ pass filter
- Row 10 (a=2, b=8): z=10 ✗ fail filter

After filter: rows 4, 7, 9 (z = 17, 11, 13)
After sort by z asc: rows 7 (z=11), 9 (z=13), 4 (z=17)
After head(5): same 3 rows (fewer than 5)

**Expected final output:** 3 rows, columns `[id, a, b, cat, z]`, ordered by z ascending = [(7, 5, 6, 'X', 11), (9, 6, 7, 'X', 13), (4, 8, 9, 'Y', 17)].

---

## 5. Golden-rule sub-check

In addition to the pipeline running, verify:

```python
ast = pipeline_through_classifier(canonical_input)
plan_original = solver.solve(ast)
plan_optimized = optimizer.optimize(plan_original)

result_original = executor.execute(plan_original, sample_data_df)
result_optimized = executor.execute(plan_optimized, sample_data_df)

assert result_original.equals(result_optimized), "Golden rule violated"
```

If golden rule fails on the demo, that is Gate 3 FAIL — Optimizer is incorrect even on a simple input.

(Note: T-04.2 PHASE_QA already runs `correctness_suite.py` over 20+ plans. Gate 3's golden-rule check is a sanity check on the CANONICAL input, not a substitute for that broader check.)

---

## 6. Procedure

```python
# Pseudocode for METHODOLOGY_INTEGRATOR's Gate 3
import sys
sys.path.insert(0, str(workspace_dir))
from STEP_07.lexer import lex
from STEP_07_1.validator import validate
from STEP_08.parser import parse
from STEP_09.typer import infer_types
from STEP_10.classifier import classify
from STEP_11.solver import solve
from BOSS_LEVEL_2.optimizer import optimize
from BOSS_LEVEL_1.executor import execute

import pandas as pd
sample_df = pd.read_csv('workflows/LANG_DEV_V2/test_target/sample_data/small.csv')

src = "LoadCSV('sample_data/small.csv'), Compute('z', 'a + b'), Filter('z > 10'), Sort('z'), Head(5)"

# Run pipeline
tokens = lex(src)
validated = validate(tokens)
cst = parse(validated)
typed = infer_types(cst)
ast = classify(typed)
plan = solve(ast)
opt_plan = optimize(plan)

# Execute both
result_original = execute(plan, sample_df)
result_optimized = execute(opt_plan, sample_df)

# Verify
assert len(result_original) == 3
assert list(result_original['id']) == [7, 9, 4]
assert list(result_original['z']) == [11, 13, 17]
assert result_original.equals(result_optimized)  # Golden rule

# Write report
write_e2e_report(...)
```

---

## 7. Output: `E2E_DEMO_OUTPUT.md`

```markdown
# E2E Demo Output — LANG_DEV_V2 Gate 3

**Generated:** <ISO timestamp>
**Reference library:** pandas_mini (or other if production target)
**Canonical input:**
```
<input source string>
```

## Stage outputs

### Lexer
- Tokens emitted: <N>
- Errors: 0

### Validator
- Validation messages: 0
- Status: PASS

### Parser
- CST nodes: <N>
- Top-level: Bag(items=<count>)

### Typer
- Type errors: 0
- Output type: DATAFRAME

### Classifier
- AST atoms: <N>
- Dependencies detected: <N>
  - COLUMN: <count>
  - EXPLICIT: <count>
  - TYPE_FORCED: <count>

### Solver
- ExecutionPlan order: [<atom names in order>]
- Explanations: included
- Determinism: verified

### Optimizer
- Rewrites applied: <N>
- Plan order after optimization: [<...>]

### Executor
- Output rows: <N>
- Output columns: [<...>]
- First 3 rows:
  ```
  <head output>
  ```

## Golden-rule check
- result_original.equals(result_optimized): True | False
- Status: PASS | FAIL

## Verdict
**Gate 3 status:** PASS | FAIL

## Failures (if any)
<stage>: <error description>
```

---

## 8. Failure modes

| Symptom | Likely cause | Phase to fix |
|---|---|---|
| Lexer raises uncaught exception | Lexer not collecting errors as data | Phase 3 T-03.1 |
| Validator emits errors for valid atoms | KNOWN_ATOMS not loaded from decisions.json | Phase 3 T-03.1.1 |
| Parser cannot construct CST | Grammar precedence wrong | Phase 3 T-03.2 |
| Typer rejects valid sequence | Compatibility matrix wrong | Phase 3 T-03.3 |
| Classifier misses column dependency | column_analysis tokenize-and-filter not catching `z` | Phase 3 T-03.4 |
| Solver puts Filter before Compute | Column dependency not honored | Phase 3 T-03.4/T-03.5 |
| Solver gives different orders across runs | Solver-level shuffle invariance broken (Gate 1 should also fail) | Phase 3 T-03.5 |
| Optimizer changes output | Golden rule violated; Optimizer incorrect | Phase 4 T-04.2 |
| Executor crashes on valid plan | Atom executor missing or buggy | Phase 4 T-04.1 |

---

## 9. Edge cases

**Sample data file missing:** Gate 3 cannot run; flag as setup error not methodology failure. PHASE_INTEGRATOR returns FAIL with `cause: "sample_data_missing"`.

**Reference library Python imports fail:** indicates pandas_mini is broken; not a methodology issue. METHODOLOGY_INTEGRATOR returns FAIL with `cause: "reference_library_broken"` — human fixes the library before re-engaging.

**Result rows differ from expected by 1-2 rows:** likely subtle Filter or Sort bug. Gate 3 FAIL with diff included in report.

**Result columns include unexpected columns or are missing expected columns:** likely Compute or Sort bug (failing to add z column, or dropping columns during sort).

---

## 10. Relationship to other gates

- **Gate 1** verifies determinism via shuffle invariance — different angle on the same Solver
- **Gate 2** verifies primitive count is sensible — orthogonal to pipeline correctness
- **Gate 3** (this) verifies the wired pipeline actually runs and produces correct output

All three gates required for METHODOLOGY_GREEN_LIGHT. Gate 3 is the most "real-world" test — if the pipeline can't process a single canonical input correctly, the methodology hasn't produced a working DSL.

---

*End of E2E_DEMO_SPEC.*
