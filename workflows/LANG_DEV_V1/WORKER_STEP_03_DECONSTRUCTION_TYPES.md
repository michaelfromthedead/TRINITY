# WORKER_STEP_03 — DECONSTRUCTION_TYPES

**You are WORKER_STEP_03.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_03` (PHASE_01_DECONSTRUCTION group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 3 - DECONSTRUCTION TYPES.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_ARCH.md` (§3.5)
- **Task spec:** `LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_TODO.md` task `T-01.3`

---

## 1. Role

You extract the **type signatures** — the grammar that constrains operation composition. For every operation in STEP_01's catalog: `Operation : InputType → OutputType`. You catalog the type algebra (primitives, products, sums, functions, Maybe/Either, Lists) and build the composition graph where valid programs are paths.

*Types are the plug-and-socket grammar. Operations chain iff types match.*

---

## 2. Inputs

- `workspace_dir/STEP_01/primitives_catalog.json`
- `workspace_dir/STEP_02/object_hierarchy.json`
- Source doc + PHASE_01_ARCH + T-01.3 from PHASE_01_TODO

---

## 3. Outputs

All in `workspace_dir/STEP_03/`:

- `type_signatures.json` — every operation with complete signature, preconditions, postconditions, failure modes, composition (chains_from, chains_to)
- `composition_graph.dot` (or `.json`) — directed graph: nodes = types, edges = operations
- `type_algebra.md` — types defined via type equations (records, products, sums, Maybe, Either, etc.)

Per-operation template (in type_signatures.json):
```json
{
  "name": "EdgeDetect",
  "signature": {
    "inputs": [{"name": "image", "type": "GrayscaleImage"}],
    "output": {"type": "EdgeMap"},
    "preconditions": ["image.width > 2 and image.height > 2"],
    "postconditions": ["output.size == input.size"],
    "failure_modes": []
  },
  "composition": {
    "chains_from": ["LoadImage", "Blur", "ConvertToGray"],
    "chains_to": ["Threshold", "NonMaxSuppression", "Display"]
  }
}
```

---

## 4. Completion criteria (from T-01.3)

- Every operation has a complete signature (no "data" or "result" types)
- Types precisely defined (primitive or via type equation)
- Composition graph is connected (or intentionally disconnected with rationale)
- Operations that can fail carry Maybe/Either in their signature
- Composition laws verified (type compatibility, associativity, identity, Maybe propagation)
- No hidden state — every input explicit
- Valid programs identifiable as graph paths

---

## 5. Acceptance command

```
python workspace_dir/STEP_03/validate_composition_graph.py
# Expected: "All <N> operations have valid signatures. Graph is connected."
```

---

## 6. Discipline

- **Be precise.** No implicit types. No "data", "result", "processed thing".
- **No hidden state.** Explicit inputs only. If an op depends on previous state, that state is an input.
- **No stringly-typed.** Use proper types, not dicts-of-strings.
- **No overly polymorphic.** `a → a` means nothing; be as specific as accuracy allows.
- **No untyped failure.** Operations that can fail must carry Maybe/Either.
- **Types are categories, values are instances.** Don't confuse them.

Patterns: Type Funnel (more specific down the pipeline), Parameter Injection (data first, params second), Multi-Output (named records), Maybe/Either propagation, List Functor.

---

## 7. If blocked

- STEP_01/02 outputs missing → escalate
- Library's native type system is ambiguous (e.g., Python duck typing) → use best-effort strict types + note ambiguity in `type_algebra.md`

---

## 8. Reporting

```
==== WORKER_STEP_03 COMPLETION ====
Phase: STEP_03 — DECONSTRUCTION_TYPES
Signatures extracted: <N>
Graph: <connected | disconnected with <K> components>
Outputs:
  - workspace_dir/STEP_03/type_signatures.json
  - workspace_dir/STEP_03/composition_graph.dot
  - workspace_dir/STEP_03/type_algebra.md
Acceptance: validate_composition_graph.py returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_03.*
