# PHASE_03_CONTRACT — IMPLEMENTATION

**Purpose:** Bind every Phase 3 task (the compiler pipeline) to concrete output files + acceptance commands.

**Authoritative methodology source:** `workflows/LANG_DEV/LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_TODO.md` and `..._ARCH.md`.

**Phase verdict:** PHASE_GREEN_LIGHT iff T-03.1, T-03.1.1, T-03.2, T-03.3, T-03.4, T-03.5 all reach TASK_PASS. Solver-level shuffle test (T-03.5) is part of phase verdict; methodology-level shuffle test runs at integration (METHODOLOGY_INTEGRATOR).

**Cross-stage discipline (binding):**
- Errors are data; collect, don't raise (every stage)
- Every node carries a span (line/column/offset)
- Stages don't run downstream on prior failure
- Test files use pytest naming: `test_<stage>.py`
- Determinism is sacred — no hash-iteration, no random, no time-based decisions

---

## T-03.1 — Lexer

**Source docs:**
- `STEP 7 - LEXER.md`

**Inputs:**
- `STEP_06_2/token_inventory.md`
- `STEP_05B/bag_grammar_spec.md` (operator disambiguation hints)

**Required outputs (paths under `workspace_dir/`):**

| File | Purpose |
|---|---|
| `STEP_07/lexer.py` | Tokenizer implementation |
| `STEP_07/tokens.py` | TokenType enum, Token + Span + LexerError dataclasses |
| `STEP_07/test_lexer.py` | Test suite |

**Completion criteria:**

1. All token types from `token_inventory.md` are recognized
2. NAME (`[A-Z][a-zA-Z0-9]*`) and IDENT (`[a-z_][a-zA-Z0-9_]*`) distinguished
3. String literals support escape sequences (`\n`, `\t`, `\\`, `\"`)
4. Number literals (integer + float) recognized
5. Whitespace skipped with line/column tracking
6. Errors collected (not raised); span field accurate (offset, line, column)
7. Round-trip property holds: concatenating token values reconstructs source modulo whitespace
8. Test suite covers: empty input, whitespace-only, each token type alone, complex valid input, each error class, span accuracy, round-trip

**Acceptance command:**
```bash
cd workspace_dir/STEP_07 && python -m pytest test_lexer.py -v --tb=short
# Expected: all tests pass
```

**Do NOT:**
- Do not skip span tracking (downstream error reporting depends on it)
- Do not stop at first lexer error
- Do not hardcode token inventory in code — derive from `token_inventory.md` (load at runtime, or generate code from it)

**Verdict:** All tests pass → `TASK_PASS`; test failure → `TASK_FAIL_RETRY` with failure list.

---

## T-03.1.1 — Validator (between Lexer and Parser)

**Source docs:**
- `STEP 7.1 - VALIDATOR.md`

**Inputs:**
- `<library>_decisions.json` (KNOWN_ATOMS source)
- T-03.1 token stream
- Reserved identifier list (from T-02.4.2 token_inventory.md or separate config)

**Required outputs:**

| File | Purpose |
|---|---|
| `STEP_07_1/validator.py` | Validation logic |
| `STEP_07_1/vocabulary.py` | KNOWN_ATOMS (loaded from decisions.json), RESERVED_IDENTIFIERS, DEPRECATED_ATOMS |
| `STEP_07_1/levenshtein.py` | Distance function + find_similar helper |
| `STEP_07_1/test_validator.py` | Test suite |

**Completion criteria:**

1. KNOWN_ATOMS derived dynamically from `<library>_decisions.json`
2. ATOMS_BY_PHASE map built (for contextual help in error messages)
3. Reserved identifier check (returns ERROR for forbidden names like `df`, `pd`, `list`)
4. Levenshtein DP implementation (O(mn) standard algorithm)
5. `find_similar(name, candidates, max_distance=2)` returns sorted matches
6. ValidationMessage with `severity` + `suggestion` + `context` fields
7. Errors collected (not raised); multiple validation errors per pass
8. Extensibility hook for custom atoms (function or registry pattern)
9. Test suite covers: valid atoms, unknown atoms, typos with suggestions, reserved identifiers, deprecated atoms (if non-empty), multiple errors

**Acceptance command:**
```bash
cd workspace_dir/STEP_07_1 && python -m pytest test_validator.py -v --tb=short
# Expected: all tests pass, including typo-suggestion tests (e.g., 'Fliter' → 'Filter')
```

**Do NOT:**
- Do not bypass this stage in the pipeline — it is a distinct gate
- Do not hardcode KNOWN_ATOMS — load from decisions.json
- Do not make suggestions beyond Levenshtein distance 3 (too speculative)

**Verdict:** All tests pass → `TASK_PASS`.

---

## T-03.2 — Parser

**Source docs:**
- `STEP 8 - PARSER.md`

**Inputs:**
- `STEP_05B/bag_grammar_spec.md` (precedence)
- T-03.1.1 validated token stream

**Required outputs:**

| File | Purpose |
|---|---|
| `STEP_08/parser.py` | Recursive-descent parser with precedence climbing |
| `STEP_08/cst.py` | CST node types (Bag, Sequence, Compound, Alternative, Optional, Group, Atom, literals, Arg) |
| `STEP_08/test_parser.py` | Test suite |

**Completion criteria:**

1. Parser infrastructure helpers present: `peek`, `advance`, `check`, `match`, `expect`, `at_end`
2. Precedence climbing: one function per precedence level (compound tightest → alternative loosest)
3. Atom parsing: `NAME '(' arg_list? ')'`
4. Argument parsing supports positional + named (`name=value`)
5. Literal parsing: string, number, bool, list, dict
6. Group parsing: `'(' bag ')'`
7. Bag parsing: comma-separated top-level
8. Error recovery via synchronization (COMMA, RPAREN sync points)
9. Span combination: parent node covers first child start to last child end
10. Test suite covers: atoms, bags, each operator, precedence, grouping, error recovery, span combination

**Acceptance command:**
```bash
cd workspace_dir/STEP_08 && python -m pytest test_parser.py -v --tb=short
# Expected: all tests pass
```

**Do NOT:**
- Do not strip spans from CST nodes
- Do not silently skip errors
- Do not confuse CST and AST (CST preserves syntax; AST is built in T-03.4)

**Verdict:** All tests pass → `TASK_PASS`.

---

## T-03.3 — Typer

**Source docs:**
- `STEP 9 - TYPER.md`

**Inputs:**
- `<library>_decisions.json` (atom catalog)
- T-03.2 CST

**Required outputs:**

| File | Purpose |
|---|---|
| `STEP_09/typer.py` | Type inference + checking |
| `STEP_09/types.py` | PortType enum, AtomType, AtomSignature, TypedNode, TypeError |
| `STEP_09/atom_catalog.py` | Built dynamically from decisions.json |
| `STEP_09/test_typer.py` | Test suite |

**Completion criteria:**

1. PortType enum derived from `decisions.json[port_types]`
2. AtomSignature built per atom in catalog
3. TypedNode wraps CST node + adds `input_types` + `output_type`
4. Type inference dispatch covers: Atom, Sequence, Compound, Alternative, Optional, Group, Bag
5. Sequence check: `left.output` compatible with `right.input[0]`
6. Compound check: same as sequence + adjacency marker
7. Alternative check: outputs match (or warning for union)
8. Optional: inner type + `is_optional` flag
9. Argument type checking per signature params (missing required, duplicate, unknown, wrong-type → errors)
10. Source/sink validation (sources have no input; sinks have no output)
11. Type compatibility matrix: exact → wildcard ANY → union → safe coercion → fail
12. Test suite covers: each atom type, valid/invalid sequences, alternatives, optionals, argument checks, source/sink violations

**Acceptance command:**
```bash
cd workspace_dir/STEP_09 && python -m pytest test_typer.py -v --tb=short
```

**Do NOT:**
- Do not let type errors silently propagate — stop type-checking that subtree
- Do not invent types not in decisions.json
- Do not skip source/sink validation

**Verdict:** All tests pass → `TASK_PASS`.

---

## T-03.4 — Classifier

**Source docs:**
- `STEP 10 - CLASSIFIER.md`

**Inputs:**
- `STEP_06/ruleset_spec.md` (phase assignment table + intra-phase priorities)
- T-03.3 Typed CST

**Required outputs:**

| File | Purpose |
|---|---|
| `STEP_10/classifier.py` | Semantic enrichment |
| `STEP_10/ast.py` | SemanticAtom, Dependency (kinds), AlternativeGroup, CompoundGroup, Pipeline (the AST) |
| `STEP_10/column_analysis.py` | Tokenize-and-filter column extraction |
| `STEP_10/test_classifier.py` | Test suite |

**Completion criteria:**

1. CST flattened to linear atom list (preserving compound/alternative/optional markers)
2. Phase assignment per atom (from ruleset table; inference fallback)
3. Intra-phase priority lookup
4. Column extraction strategy: regex `[a-zA-Z_][a-zA-Z0-9_]*` → filter KEYWORDS + FUNCTIONS → remaining = column refs (per `STEP 10 §3.2`)
5. KEYWORDS set defined: `and, or, not, in, is, True, False, None, if, else, for`
6. FUNCTIONS set defined: `sum, mean, min, max, abs, len, str, int, float, list`
7. Three dependency kinds detected: EXPLICIT (from Sequence), COLUMN (producer-consumer column overlap), TYPE_FORCED (port-type adjacency)
8. Implicit compound detection (type-exclusive pairs like GroupBy-Agg)
9. Cycle detection via DFS-with-colors (or Tarjan's)
10. AST simplification (transitive reduction; remove redundant deps)
11. Test suite covers: phase assignment, column extraction, each dep kind, implicit compounds, cycle detection, complex pipelines

**Acceptance command:**
```bash
cd workspace_dir/STEP_10 && python -m pytest test_classifier.py -v --tb=short
# Expected: all tests pass, including cycle-detection tests
```

**Do NOT:**
- Do not use regex alone for column extraction — must filter keywords/functions
- Do not skip cycle detection
- Do not forget to bridge dependencies across optional atoms

**Verdict:** All tests pass → `TASK_PASS`.

---

## T-03.5 — Solver (THE acid test lives here)

**Source docs:**
- `STEP 11 - SOLVER.md`

**Inputs:**
- `STEP_04/pcfg_weights_initial.json` (uniform to start)
- T-03.4 AST

**Required outputs:**

| File | Purpose |
|---|---|
| `STEP_11/solver.py` | Constraint-satisfaction + topological sort |
| `STEP_11/constraints.py` | OrderConstraint, AdjacencyConstraint, ConstraintGraph |
| `STEP_11/execution_plan.py` | ExecutionPlan, Decision, Explanation |
| `STEP_11/test_solver.py` | Test suite (INCLUDES SHUFFLE TEST) |

**Completion criteria:**

1. Constraint collection from dependencies, compounds, phases, priorities
2. Constraint graph construction
3. Alternative resolution (configurable; default first-available)
4. Optional resolution: default exclude; bridge dependencies over excluded atoms
5. Compound handling via super-nodes (collapse → sort → expand)
6. Topological sort: Kahn's algorithm with priority-queue tie-breaker
7. **Fully-specified priority key** (no ties possible): `(phase.value, priority, -pcfg_score, name, str(args))`
8. Result validation: all order + adjacency constraints satisfied
9. Explanation generation (why each atom is at its position)
10. Unsatisfiable constraint detection with actionable error
11. PCFG weight loading + context determination + scoring
12. **SHUFFLE TEST** (T-03.5 acceptance, mandatory):
   - 10+ diverse bags
   - 100+ shuffle iterations per bag
   - Assertion: `solve(bag) == solve(shuffle(bag))` for every iteration
13. Test suite covers: constraint collection, alternative + optional resolution, compound super-nodes, priority key tie-breaking, unsatisfiable detection, **shuffle invariance**

**Acceptance command:**
```bash
cd workspace_dir/STEP_11 && python -m pytest test_solver.py -v --tb=short -k 'shuffle'
# Expected: all shuffle tests pass
cd workspace_dir/STEP_11 && python -m pytest test_solver.py -v --tb=short
# Expected: all tests pass
```

**Do NOT:**
- Do not use `random` in tie-breaking (priority key must be hard-specified)
- Do not use hash-based iteration (deterministic sort required everywhere)
- Do not skip the shuffle test — this is the methodology's acid test
- Do not collapse compounds before collecting their internal constraints

**Verdict:**
- All tests + shuffle invariance pass → `TASK_PASS`
- Shuffle test fail → `TASK_FAIL_RETRY` (Critical — this is the acid test)
- Other test failures → `TASK_FAIL_RETRY` with failure list

---

## Phase 3 verdict gate

When all six T-03.x tasks `TASK_PASS`:
- Run integration test (final check before Phase 4):
  ```bash
  cd workspace_dir && python -c "
  # Source string → execution plan e2e
  src = \"LoadCSV('data.csv'), Compute('z', 'x+y'), Filter('z > 10'), Head(5)\"
  from STEP_07.lexer import lex
  from STEP_07_1.validator import validate
  from STEP_08.parser import parse
  from STEP_09.typer import infer_types
  from STEP_10.classifier import classify
  from STEP_11.solver import solve
  tokens = lex(src)
  validated = validate(tokens)
  cst = parse(validated)
  typed = infer_types(cst)
  ast = classify(typed)
  plan = solve(ast)
  print('PASS:', plan.order)
  "
  ```
- QUEEN appends Phase 3 summary to `workspace_manifest.json`
- QUEEN writes Phase 3 completion entry to `INPROGRESS.md`
- QUEEN emits `PHASE_GREEN_LIGHT`; advances to Phase 4

If e2e integration fails after per-stage tests pass, escalate as cross-stage integration bug (likely a contract mismatch between adjacent stages).

---

*End of PHASE_03_CONTRACT.*
