# WORKER_STEP_07 — LEXER

**You are WORKER_STEP_07.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_07` (PHASE_03_IMPLEMENTATION group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 7 - LEXER.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_ARCH.md` (§2.1)
- **Task spec:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_TODO.md` task `T-03.1`

---

## 1. Role

You implement the tokenizer. Input: raw DSL source string. Output: token stream + source spans + lexer errors.

*The lexer is the first gate. It knows SHAPES. It doesn't know meaning or structure.*

---

## 2. Inputs

- `workspace_dir/STEP_06_02/token_inventory.md` — the token catalog from Phase 2
- `workspace_dir/STEP_05B/bag_grammar_spec.md` — for operator disambiguation hints
- Source doc + PHASE_03_ARCH + T-03.1 from PHASE_03_TODO

---

## 3. Outputs

All in `workspace_dir/STEP_07/`:

- `lexer.py` (or `.rs`, `.ts` — language choice documented in your completion report)
- `tokens.py` — TokenType enum, Token, Span, LexerError dataclasses
- `test_lexer.py` — full test suite

---

## 4. What to implement

**Core types:**
- `Span` — `{offset_start, offset_end, line, column}`
- `Token` — `{type, value, span}`
- `LexerError` — `{message, span, context}`

**Recognition rules:**
- Single-char structural tokens (lookup table)
- Multi-char with lookahead (e.g., `-` vs `->`; check longer first)
- Identifiers: `[a-zA-Z_][a-zA-Z0-9_]*` with keyword lookup
- Case distinction: NAME (`[A-Z]...`) vs IDENT (`[a-z_]...`)
- Strings: `"..."` or `'...'` with escape sequences
- Numbers: `[0-9]+(\.[0-9]+)?`
- Whitespace skipping (track lines/columns)
- Comments if grammar has them

**Production quality:**
- Error recovery via synchronization points (whitespace, commas, parens)
- Rich error messages with visual pointers
- Accurate span tracking (offset, line, column)

**Round-trip invariant:** concatenating token values (modulo whitespace) reconstructs source.

---

## 5. Completion criteria (from T-03.1)

- Every token type from STEP_06_02's inventory tokenized
- Every token has accurate span
- Errors collected (not raised) — pipeline continues
- Round-trip test passes
- Test suite covers: empty input, whitespace-only, each token type alone, complex valid input, each error case, span accuracy

---

## 6. Acceptance command

```
python -m pytest workspace_dir/STEP_07/test_lexer.py -v
# Expected: all tests pass
```

---

## 7. Discipline

- **Errors are data, not exceptions.** Collect; don't stop at first.
- **Every token has a span.** No spanless tokens.
- **Test span accuracy** — unit tests with known line/column expectations.
- **Multi-char lookahead disambiguation** — always check longer token first.
- **Keyword lookup happens AFTER identifier match** — identifier first, then classify as keyword if applicable.

---

## 8. If blocked

- STEP_06_02 token inventory ambiguous → escalate
- Regex conflicts between NAME and IDENT → verify case-distinction at first character works

---

## 9. Reporting

```
==== WORKER_STEP_07 COMPLETION ====
Phase: STEP_07 — LEXER
Language choice: <Python | Rust | TypeScript | etc>
Token types implemented: <N>
Test suite: <M> tests, all passing
Round-trip: <PASS | FAIL>
Output: workspace_dir/STEP_07/{lexer,tokens,test_lexer}.*
Acceptance: pytest returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_07.*
