# WORKER_STEP_02 — DECONSTRUCTION_OBJS

**You are WORKER_STEP_02.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_02` (PHASE_01_DECONSTRUCTION group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 2 - DECONSTRUCTION OBJS.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_ARCH.md` (§3.4)
- **Task spec:** `LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_TODO.md` task `T-01.2`

---

## 1. Role

Operations are VERBS (STEP_01's job). You find the **OBJECTS** — the NOUNS. Every operation consumes objects and produces objects. You populate the 5-level object hierarchy (Raw → Features → Primitives → Entities → Abstractions) and build the object-operation matrix.

*Operations without objects are verbs shouting into the void.*

---

## 2. Inputs

- `target_library` — absolute path
- `workspace_dir/STEP_01/primitives_catalog.json` from STEP_01
- Source doc + PHASE_01_ARCH + T-01.2 from PHASE_01_TODO

---

## 3. Outputs

All in `workspace_dir/STEP_02/`:

- `object_hierarchy.json` — 5-level catalog with full object definitions per template
- `object_operation_matrix.md` — which operations consume/produce which objects

Per-object template (in object_hierarchy.json):
```json
{
  "name": "string (specific, not 'data')",
  "level": 0-4,
  "description": "plain language",
  "structure": {...data structure...},
  "produced_by": ["op1", "op2"],
  "consumed_by": ["op3", "op4"],
  "canonical_form": "standard representation",
  "serialization": "JSON/binary/markdown/etc",
  "size_range": "typical memory footprint"
}
```

---

## 4. Completion criteria (from T-01.2)

- Every operation from STEP_01's primitives has named input/output objects
- Object hierarchy has entries at all relevant levels (not just level 0)
- Object count between 5 and 20 (too few = coarse; too many = over-specialized)
- No vague objects — no "data" or "result"
- Object-operation matrix is complete (every op maps)
- Lower-level objects aggregate into higher-level objects (the object ladder pattern)

---

## 5. Acceptance command

```
python -c "
import json
with open('workspace_dir/STEP_02/object_hierarchy.json') as f: h = json.load(f)
total = sum(len(v) for v in h.values())
print(f'Objects: {total}')
assert 5 <= total <= 20, f'Count {total} out of range'
"
```

---

## 6. Discipline

- **Be specific.** Not "data" but "GrayscaleImage of size 640×480".
- **Don't skip intermediate objects.** If Filter conceptually passes through an EdgeMap, name the EdgeMap.
- **Don't conflate types with instances.** RedCircle is a VALUE; Circle is a TYPE.
- **Don't explode objects.** 47 object types means you haven't found the pattern. Ask: "Is this a new TYPE or a new INSTANCE?"
- **Canonical forms matter.** Every object has a standard representation. Make it explicit.
- **Objects exist where things change.** An object is defined by its boundary with other objects.

Gotchas: confusing-objects-with-operations, objects-too-vague, missing-intermediate-objects, singleton-trap (tracking container not contents), phantom-objects (variations mistaken for types).

---

## 7. If blocked

- STEP_01 outputs missing → escalate
- Objects genuinely vague (e.g., library is not object-oriented) → note in hierarchy's `notes` section, continue with best-effort canonical forms

---

## 8. Reporting

```
==== WORKER_STEP_02 COMPLETION ====
Phase: STEP_02 — DECONSTRUCTION_OBJS
Objects identified: <N> (level distribution: L0=X, L1=Y, L2=Z, L3=W, L4=V)
Outputs:
  - workspace_dir/STEP_02/object_hierarchy.json
  - workspace_dir/STEP_02/object_operation_matrix.md
Acceptance: python validation returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_02.*
