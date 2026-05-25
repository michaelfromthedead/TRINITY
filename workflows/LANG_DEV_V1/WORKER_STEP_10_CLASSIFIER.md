# WORKER_STEP_10 — CLASSIFIER

**You are WORKER_STEP_10.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_10` (PHASE_03_IMPLEMENTATION group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 10 - CLASSIFIER.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_ARCH.md` (§2.5)
- **Task spec:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_TODO.md` task `T-03.4`

---

## 1. Role

You transform the Typed CST into a semantically-rich AST (flat list + dependency edges). Phase assignment. Column analysis (produces/consumes). Dependency detection across THREE kinds: EXPLICIT, COLUMN, TYPE_FORCED. Cycle detection.

*Classifier knows meaning. Enriches types with semantic analysis.*

---

## 2. Inputs

- Typed CST from STEP_09
- `workspace_dir/STEP_06/ruleset_spec.md` (phase assignment rules + intra-phase priorities)
- Source doc + PHASE_03_ARCH + T-03.4 from PHASE_03_TODO

---

## 3. Outputs

All in `workspace_dir/STEP_10/`:

- `classifier.py`
- `ast.py` — SemanticAtom, Dependency (kinds), AlternativeGroup, CompoundGroup, Pipeline
- `column_analysis.py` — tokenize-and-filter expression parser
- `test_classifier.py`

---

## 4. What to implement

**Flatten CST → atom list:**
- Walk CST, preserve compound/alternative/optional markers
- Return `List[SemanticAtom]`

**Phase assignment:** lookup per atom name from STEP_06 ruleset table; fallback to type-based inference (input=[] → SOURCE; output=NONE → SINK; etc.); default TRANSFORM.

**Column analysis (per atom type):**
- `Compute(name, expr)` → produces {name}, consumes expr_columns
- `Filter(condition)` → consumes condition_columns
- `SelectCols(cols)` → consumes set(cols)
- `GroupBy(by)` → consumes set(by)
- `Agg({new: (old, fn)})` → produces {new}, consumes {old}
- `Rename({old: new})` → produces {new}, consumes {old}
- Sources → produces = UNKNOWN

**Expression column extraction (tokenize-and-filter, strategy 2 from STEP_10):**
- Regex `[a-zA-Z_][a-zA-Z0-9_]*` → candidates
- Filter KEYWORDS (`and, or, not, in, is, True, False, None, ...`)
- Filter FUNCTIONS (`sum, mean, len, abs, str, int, ...`)
- Remaining = column references

**Three dependency kinds:**
1. **EXPLICIT** — from Sequence nodes in CST (user wrote `→`)
2. **COLUMN** — `producer.produces_columns ∩ consumer.consumes_columns` ≠ ∅
3. **TYPE_FORCED** — producer outputs GROUPED/ROLLING/RESAMPLER; only specific consumers accept those

**Implicit compounds:** type-exclusive pairs (e.g., GroupBy-AggSum must be adjacent).

**Cycle detection:** DFS with colors (white/gray/black); cycles → error with specific chain.

**AST simplification:** merge redundant deps; remove self-deps; transitive reduction; normalize single-option alternatives.

---

## 5. Completion criteria (from T-03.4)

- Every atom has phase, priority, produces_columns, consumes_columns
- All 3 dep kinds detected correctly
- Implicit compounds detected for type-exclusive pairs
- Cycles caught with clear error message
- Test suite covers: phase assignment, column extraction, each dep kind, implicit compounds, cycles, complex pipelines

---

## 6. Acceptance command

```
python -m pytest workspace_dir/STEP_10/test_classifier.py -v
# Expected: all tests pass, including cycle detection
```

---

## 7. Discipline

- **Don't use regex alone for column extraction.** Filter keywords and functions.
- **Don't skip cycle detection.**
- **Don't forget optional dependency bridging** — if atom A is optional and excluded, deps `X → A → Y` become `X → Y`.
- **Three dep kinds have different weights:** EXPLICIT (user wrote it) > COLUMN (inferred) > TYPE_FORCED (structural).

---

## 8. If blocked

- STEP_06 ruleset missing phase assignment rules → use type-based inference + flag
- Column extraction ambiguous → conservative (emit both as potential consumes) + flag

---

## 9. Reporting

```
==== WORKER_STEP_10 COMPLETION ====
Phase: STEP_10 — CLASSIFIER
Atoms annotated: <N>
Dependencies: <explicit=X, column=Y, type_forced=Z>
Implicit compounds: <C>
Cycles detected: <0 or list>
Test suite: <T> tests, all passing
Output: workspace_dir/STEP_10/{classifier,ast,column_analysis,test_classifier}.*
Acceptance: pytest returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_10.*
