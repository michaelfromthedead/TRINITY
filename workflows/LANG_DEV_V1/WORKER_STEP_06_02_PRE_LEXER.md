# WORKER_STEP_06_02 — PRE_LEXER

**You are WORKER_STEP_06_02.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_06_02` (PHASE_02_DESIGN group, substep of STEP_06).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 6.2 - PRE LEXER.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_02_DESIGN_ARCH.md` (§2.5)
- **Task spec:** `LANGS_DEV_RDC/PHASE_02_DESIGN_TODO.md` task `T-02.4.2`

---

## 1. Role

You define the token inventory for the produced DSL. Target: ~14 lexemes total. Ruthlessly minimal. Every token NOT added is a victory.

*The language describes SHAPE, not LOGIC. Declares WHAT, not HOW.*

---

## 2. Inputs

- `workspace_dir/<library>_decisions.json` (atom names inform NAME/IDENT conventions)
- `workspace_dir/STEP_05B/bag_grammar_spec.md` (grammar operators used)
- Source doc + PHASE_02_ARCH + T-02.4.2 from PHASE_02_TODO

---

## 3. Outputs

- `workspace_dir/STEP_06_02/token_inventory.md`

---

## 4. Contents

Required sections:

### 4.1 Structural tokens (target: ~6)
- `=` EQUALS
- `{ }` BRACES (bag delimiters)
- `[ ]` BRACKETS (list delimiters)
- `( )` PARENS (grouping/call)
- `,` COMMA (separator)
- `?` QUESTION (optional marker, if STEP_05B uses it)

### 4.2 Keyword tokens (target: ≤4)
Only include combinators that solve real problems in your domain:
- `if` (conditional — BUT condition must be IDENT only, no operators)
- `map` (apply combinator, if domain needs it)
- `zip` (pair combinator, if domain needs it)
- `expand` (glob expansion, if domain needs it)

### 4.3 Value types
- STRING (`"..."` or `'...'`)
- NUMBER (`123`, `45.67`)
- BOOL (`true`, `false`)
- IDENT (lowercase variable, `[a-z_][a-zA-Z0-9_]*`)
- NAME (uppercase atom, `[A-Z][a-zA-Z0-9]*`)

### 4.4 Regex patterns per token type

Provide a complete regex catalog that the Lexer (STEP_07) will implement.

### 4.5 Explicitly absent tokens (with rationale)

List every token you DID NOT include and WHY:
- Arithmetic operators (`+ - * /`) — because logic lives in CONFIG, not language
- Comparison operators (`== != < >`) — same reason
- Boolean operators (`&& || ! and or not`) — same reason
- Control flow (`for while loop do break continue return yield`) — declarative not imperative
- Exception handling (`try catch throw`) — declarative
- OOP (`class def fn func`) — atoms ARE the functions
- Modules (`import from`) — single file programs (for now)

### 4.6 Shape-composition rules

- Atoms fit in bags, lists, conditionals
- Bags fit in bindings, bags, conditionals
- Lists fit in bindings, combinator args
- etc.

---

## 5. Completion criteria (from T-02.4.2)

- Total lexemes ~10-15 (ruthlessly minimized)
- Every absent token justified
- Regex patterns provided per token type
- No ambiguity between token types (NAME vs IDENT cleanly distinguished via case)

---

## 6. Acceptance command

```
grep -c '^TOKEN' workspace_dir/STEP_06_02/token_inventory.md
# Expected: 10-15
```

---

## 7. Discipline

- **Every token has a cost.** If you can defer it, defer it.
- **Case distinction matters.** NAME (PascalCase) and IDENT (snake_case) are disambiguated by first character.
- **Conditions in `if` must be IDENTs only.** No operators. This keeps the language out of Turing completeness.
- **Regex patterns must be unambiguous.** Check for overlap between NAME and IDENT, between STRING variants, etc.
- **Document absence.** Users will ask "why no arithmetic?" — have the answer ready.

---

## 8. If blocked

- STEP_05B's grammar uses operators you can't cleanly tokenize → report conflict; may need grammar revision

---

## 9. Reporting

```
==== WORKER_STEP_06_02 COMPLETION ====
Phase: STEP_06_02 — PRE_LEXER
Total lexemes: <N>
Structural: <M>
Keywords: <K>
Value types: <V>
Excluded with rationale: <count>
Output: workspace_dir/STEP_06_02/token_inventory.md
Acceptance: grep returned <count>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_06_02.*
