# WORKER_STEP_08 — PARSER

**You are WORKER_STEP_08.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_08` (PHASE_03_IMPLEMENTATION group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 8 - PARSER.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_ARCH.md` (§2.3)
- **Task spec:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_TODO.md` task `T-03.2`

---

## 1. Role

You build the Concrete Syntax Tree (CST) from validated tokens. Precedence climbing with recursive descent. Preserve ALL syntax for error messages and round-trip.

*Parser knows structure. Doesn't know meaning.*

---

## 2. Inputs

- Validated token stream from STEP_07_01
- `workspace_dir/STEP_05B/bag_grammar_spec.md` (grammar + precedence)
- Source doc + PHASE_03_ARCH + T-03.2 from PHASE_03_TODO

---

## 3. Outputs

All in `workspace_dir/STEP_08/`:

- `parser.py`
- `cst.py` — CST node types (Bag, Sequence, Compound, Alternative, Optional, Group, Atom, literals, Arg)
- `test_parser.py`

---

## 4. What to implement

**Parser infrastructure:** `peek()`, `peek_type()`, `advance()`, `check(type)`, `match(types...)`, `expect(type, msg)`, `at_end()`.

**Precedence climbing** (per STEP_05B precedence):
```
parse_bag()           // top-level (comma-separated)
  parse_alternative() // level 1 (|)
    parse_sequence()  // level 2 (→)
      parse_compound()// level 3 (-)
        parse_unary() // level 4 (?)
          parse_primary() // atoms, groups
```

**Atom parsing:** `NAME '(' arg_list? ')'` — preserves all syntax.

**Argument parsing:** positional + named (`name=value`). Literals: STRING, NUMBER, BOOL, LIST, DICT.

**Group:** `'(' bag ')'` — preserves parens in CST.

**Span combination:** parent node's span = first child's start → last child's end.

**Error recovery:** synchronize to COMMA, RPAREN; missing-token insertion for obvious cases.

---

## 5. Completion criteria (from T-03.2)

- Parses every valid input per STEP_05B grammar
- Precedence correct (compound tightest → sequence → alternative loosest)
- Left-associative chains produce expected tree
- Error recovery produces partial CSTs + multiple errors
- Test suite covers: atoms, bags, each operator, precedence, grouping, errors

---

## 6. Acceptance command

```
python -m pytest workspace_dir/STEP_08/test_parser.py -v
# Expected: all tests pass, including precedence tests
```

---

## 7. Discipline

- **Don't strip spans.** Every CST node has a span.
- **Don't silently skip errors.** Collect them.
- **CST, not AST.** Preserve parens, commas, everything. AST is STEP_10's job.
- **Left-associative by default.** `A → B → C` parses as `(A → B) → C`.
- **Unambiguous grammar.** If your STEP_05B grammar is ambiguous, report it.

---

## 8. If blocked

- STEP_07_01 token stream malformed → escalate
- Grammar ambiguity detected → escalate; may need STEP_05B revision

---

## 9. Reporting

```
==== WORKER_STEP_08 COMPLETION ====
Phase: STEP_08 — PARSER
Node types: <N>
Test suite: <T> tests, all passing
Precedence tests: all passing
Output: workspace_dir/STEP_08/{parser,cst,test_parser}.*
Acceptance: pytest returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_08.*
