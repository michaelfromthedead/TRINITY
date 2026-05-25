# WORKER_STEP_07_01 — VALIDATOR

**You are WORKER_STEP_07_01.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_07_01` (PHASE_03_IMPLEMENTATION group, substep of STEP_07).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 7.1 - VALIDATOR.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_ARCH.md` (§2.2)
- **Task spec:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_TODO.md` task `T-03.1.1`

---

## 1. Role

You implement the Vocabulary Gate — a stage between Lexer and Parser. Lexer knows SHAPES; Parser knows STRUCTURE; neither knows MEANING. You bridge: is this NAME a known atom? Is this IDENT a reserved identifier? Levenshtein-based typo suggestions.

*Validator is a guide, not a gatekeeper. Errors are opportunities for education.*

---

## 2. Inputs

- Token stream from STEP_07 Lexer
- `workspace_dir/<library>_decisions.json` (source of KNOWN_ATOMS)
- Reserved identifier list (from STEP_06_02 or separate config)
- Source doc + PHASE_03_ARCH + T-03.1.1 from PHASE_03_TODO

---

## 3. Outputs

All in `workspace_dir/STEP_07_01/`:

- `validator.py`
- `vocabulary.py` — KNOWN_ATOMS (from decisions.json), RESERVED_IDENTIFIERS, DEPRECATED_ATOMS
- `levenshtein.py` — distance + find_similar
- `test_validator.py`

---

## 4. What to implement

**Vocabulary:**
- KNOWN_ATOMS = derived from `decisions.json` atoms array
- ATOMS_BY_PHASE = grouped for contextual help messages
- RESERVED_IDENTIFIERS = `{df, pd, np, list, dict, set, for, while, class, def, import, ...}`
- DEPRECATED_ATOMS = map: `{old_name: {replacement, message, version}}` (optional, can be empty)

**Levenshtein:**
- Standard DP algorithm
- `find_similar(name, candidates, max_distance=2)` returns sorted matches
- Distance 1 = high confidence suggestion; distance 2 = confident; distance 3+ = show as "Similar" not "Did you mean"

**Validation logic:**
- NAME token → check against KNOWN_ATOMS; emit ValidationMessage(ERROR) if not found + suggestion if found
- IDENT token → check against RESERVED_IDENTIFIERS; emit ValidationMessage(ERROR) if reserved
- KNOWN_ATOMS match → check DEPRECATED_ATOMS; emit WARNING if deprecated

**Error rendering** (preview; full rendering is STEP_BL03's job):
```
Unknown atom 'Fliter'
Hint: Did you mean 'Filter'?
Note: TRANSFORM atoms: Filter, Compute, Sort, ...
```

---

## 5. Completion criteria (from T-03.1.1)

- Every NAME token validated against KNOWN_ATOMS
- Every IDENT token validated against RESERVED_IDENTIFIERS
- Typo suggestions work (test: `Fliter` → `Filter`)
- Deprecation warnings (if DEPRECATED_ATOMS non-empty)
- Collects all errors (doesn't stop at first)
- Test suite covers: valid atoms, unknown atoms, typos, reserved identifiers, deprecated atoms, multiple errors

---

## 6. Acceptance command

```
python -m pytest workspace_dir/STEP_07_01/test_validator.py -v
# Expected: all tests pass, including typo suggestion tests
```

---

## 7. Discipline

- **Don't bypass this stage.** It's a distinct pipeline step between Lexer and Parser.
- **Don't hardcode KNOWN_ATOMS.** Derive from `decisions.json` at runtime.
- **Levenshtein bounds:** distance ≤ 2 for confident suggestions; ≤ 3 for "similar" listing; never suggest beyond 3.
- **Extensibility:** stub a `custom_atoms` parameter for domain-specific vocabulary extensions.

---

## 8. If blocked

- `decisions.json` incomplete → escalate
- Reserved identifiers list unclear → use sensible Python defaults + flag for human review

---

## 9. Reporting

```
==== WORKER_STEP_07_01 COMPLETION ====
Phase: STEP_07_01 — VALIDATOR
Known atoms loaded: <N>
Reserved identifiers: <M>
Deprecated atoms: <K> (may be 0)
Test suite: <T> tests, all passing
Output: workspace_dir/STEP_07_01/{validator,vocabulary,levenshtein,test_validator}.*
Acceptance: pytest returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_07_01.*
