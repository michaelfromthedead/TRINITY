# WORKER_STEP_09 — TYPER

**You are WORKER_STEP_09.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_09` (PHASE_03_IMPLEMENTATION group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 9 - TYPER.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_ARCH.md` (§2.4)
- **Task spec:** `LANGS_DEV_RDC/PHASE_03_IMPLEMENTATION_TODO.md` task `T-03.3`

---

## 1. Role

You annotate the CST with type information. Check port compatibility. Validate atom signatures + argument types. Catch source-in-wrong-position and sink-in-wrong-position errors.

*Typer knows compatibility. Doesn't know meaning beyond types.*

---

## 2. Inputs

- CST from STEP_08
- `workspace_dir/<library>_decisions.json` (atom catalog with port types)
- Source doc + PHASE_03_ARCH + T-03.3 from PHASE_03_TODO

---

## 3. Outputs

All in `workspace_dir/STEP_09/`:

- `typer.py`
- `types.py` — PortType enum, AtomType, AtomSignature, TypedNode, TypeError
- `atom_catalog.py` — built from decisions.json at runtime
- `test_typer.py`

---

## 4. What to implement

**Core types:**
- `PortType` enum derived from decisions.json port_types
- `AtomSignature = {name, inputs, output, params, phase}`
- `TypedNode` = CST node + input_types + output_type
- `TypeError = {message, span, expected_type, actual_type, source_node, target_node}`

**Atom catalog:** map atom_name → AtomSignature built from decisions.json.

**Type inference dispatch:**
- Atom → lookup signature, check args, return TypedAtom
- Sequence(L, R) → check L.output compatible with R.input[0]
- Compound(L, R) → same as Sequence + adjacency marker
- Alternative(L, R) → verify outputs match (or emit union warning)
- Optional(inner) → inner type, is_optional=true
- Group(inner) → pass-through
- Bag → type each item

**Compatibility matrix:** exact match → wildcard ANY → union → subtype → safe coercion → fail.

**Source/sink validation:**
- Source atoms (input=[]) cannot be RIGHT of `→`
- Sink atoms (output=NONE) cannot be LEFT of `→`

**Argument type checking:** per signature params — missing required, duplicate, unknown, wrong type = errors.

---

## 5. Completion criteria (from T-03.3)

- Every atom resolves against catalog
- Type mismatches produce errors with expected/actual/source/target
- Valid sequences pass; invalid sequences fail with clear errors
- Source/sink position violations caught
- Argument type mismatches caught
- Test suite covers: each atom type, valid/invalid sequences, alternatives, optionals, argument checks

---

## 6. Acceptance command

```
python -m pytest workspace_dir/STEP_09/test_typer.py -v
# Expected: all tests pass
```

---

## 7. Discipline

- **Don't let type errors silently propagate.** Stop type-checking the erroring subtree; mark it for error reporting.
- **Unknown types = errors.** If a signature references a type not in decisions.json, escalate.
- **Source/sink validation is not optional.**
- **Union types for atoms accepting multiple input types** (e.g., Agg accepts GROUPED | ROLLING | RESAMPLER).

---

## 8. If blocked

- decisions.json incomplete (undefined types referenced) → escalate
- CST from STEP_08 malformed → escalate

---

## 9. Reporting

```
==== WORKER_STEP_09 COMPLETION ====
Phase: STEP_09 — TYPER
Atom signatures loaded: <N>
Port types: <M>
Test suite: <T> tests, all passing
Output: workspace_dir/STEP_09/{typer,types,atom_catalog,test_typer}.*
Acceptance: pytest returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_09.*
