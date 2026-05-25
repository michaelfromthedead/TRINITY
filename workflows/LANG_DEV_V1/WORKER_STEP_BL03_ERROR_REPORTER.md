# WORKER_STEP_BL03 — ERROR_REPORTER

**You are WORKER_STEP_BL03.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_BL03` (PHASE_04_RUNTIME group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/boss_level_3_error_reporter_rules.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_04_RUNTIME_ARCH.md` (§2.3)
- **Task spec:** `LANGS_DEV_RDC/PHASE_04_RUNTIME_TODO.md` task `T-04.3`

---

## 1. Role

You implement the unified error rendering across ALL pipeline stages (Lexer, Validator, Parser, Typer, Classifier, Solver, Executor). Every stage's errors share the same format: box-drawn, line-pointed, Levenshtein-suggesting, severity-categorized.

*Error message is a helping hand, not punishment.*

---

## 2. Inputs

- Error types from all prior stages (LexerError, ValidationMessage, ParseError, TypeError, SemanticError, SolverError, ExecutionError)
- Source doc + PHASE_04_ARCH + T-04.3 from PHASE_04_TODO

---

## 3. Outputs

All in `workspace_dir/BOSS_LEVEL_3/`:

- `error_reporter.py`
- `unified_error.py` — Error base type
- `rendering.py` — box-drawn format
- `fuzzy_match.py` — Levenshtein (shared with STEP_07_01 Validator)
- `test_error_reporter.py`

---

## 4. What to implement

**Unified Error type:**
```python
@dataclass
class Error:
    severity: ERROR | WARNING | HINT
    category: SYNTAX | TYPE | SEMANTIC | CONSTRAINT | RUNTIME
    location: (line, column, offset, file?)
    main_message: str
    explanation: str
    suggestion: Optional[str]
    context: Optional[str]
    related_errors: List[Error]
```

**Adapters from stage-specific error types** to unified Error.

**Box-drawn render format:**
```
┌─ <Severity> <Category> ──────────────────────────┐
│                                                  │
│   <line> │ <source line>                         │
│          │ <pointer (^^^^)>                      │
│                                                  │
│ <main message>                                   │
│                                                  │
│ Hint: <suggestion>                               │
│ Note: <context>                                  │
└──────────────────────────────────────────────────┘
```

**Levenshtein suggestions:** distance 1-2 → "Did you mean X?"; distance 3+ → "Similar: X, Y, Z".

**Error collection + grouping:** gather all errors; group cascading errors under primary cause.

**Priority ordering:** most-important errors first (Critical > High > Medium > Low).

**Rendering modes:** colored (terminal) vs plain (file output).

---

## 5. Completion criteria (from T-04.3)

- Errors from every stage render in identical format
- Every rendered error answers WHAT/WHERE/WHY/HOW
- Levenshtein suggestions work (distance ≤ 2)
- Multiple errors can be displayed at once
- Test suite covers: syntax, type, semantic, runtime errors; unknown-atom-with-suggestion; unknown-column; cascading errors

---

## 6. Acceptance command

```
python -m pytest workspace_dir/BOSS_LEVEL_3/test_error_reporter.py -v
# Expected: all tests pass; renderings consistent across stages
```

---

## 7. Discipline

- **Don't produce stage-specific rendering.** Unified format ALWAYS.
- **Don't silently swallow errors.**
- **Every error answers WHAT/WHERE/WHY/HOW.**
- **Don't over-suggest.** Distance 3+ = "similar", not "did you mean."
- **Color for terminal, plain for files** — detect output target.
- **Related errors grouped.** Cascading errors (one causes many) shown under primary.

---

## 8. If blocked

- Stage-specific error types incompatible with unification → define adapters explicitly
- Levenshtein library unavailable → implement DP algorithm from scratch (shared with STEP_07_01)

---

## 9. Reporting

```
==== WORKER_STEP_BL03 COMPLETION ====
Phase: STEP_BL03 — ERROR_REPORTER
Stages unified: 7 (Lexer, Validator, Parser, Typer, Classifier, Solver, Executor)
Categories: 5 (SYNTAX, TYPE, SEMANTIC, CONSTRAINT, RUNTIME)
Test suite: <T> tests, all passing
Output: workspace_dir/BOSS_LEVEL_3/{error_reporter,unified_error,rendering,fuzzy_match,test_error_reporter}.*
Acceptance: pytest returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_BL03.*
