# PHASE_01_CONTRACT — DECONSTRUCTION

**Purpose:** Bind every Phase 1 task to concrete output files + acceptance commands. PHASE_QA's authority is bounded by this contract (not by free interpretation of the STEP doc).

**Authoritative methodology source:** `workflows/LANG_DEV/LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_TODO.md` and `..._ARCH.md`. PHASE_EXECUTOR full-reads source docs from `step_source_dir`; PHASE_QA verifies against THIS contract.

**Phase verdict:** PHASE_GREEN_LIGHT iff T-01.1, T-01.2, T-01.3 all reach TASK_PASS. T-01.1.1 is on-demand and does not gate the phase verdict.

---

## T-01.1 — Deconstruction Ops

**Source docs (PHASE_EXECUTOR full-reads):**
- `STEP 1 - DECONSTRUCTION OPS.md` (foundation)
- `context.md` (extended rules retrofit — multi-tier basis, 5-type primitive classification, hidden-primitive check, two-path check)

**Inputs:**
- `target_library` (read-only)
- `nexus_reports_dir/{GRAVITY,VERBS,TIERS,CLASSIFICATION,GENESIS,GENERATOR,COMPRESSION}.md` (read-only)

**Required outputs (paths relative to `workspace_dir/`):**

| File | Format | Required fields |
|---|---|---|
| `STEP_01/primitives_catalog.json` | JSON | `primitives` array; each: `{name, tier (0-3), type (UNIVERSAL/STRUCTURAL/BRIDGE/GOAL/PHILOSOPHICAL), inputs, outputs, derivation, evidence (nexus signal ref), examples}` |
| `STEP_01/tier_compression.md` | Markdown | Compression ratio per tier transition (Tier 0→1, 1→2, 2→3) with derivation |
| `STEP_01/deconstruction_notes.md` | Markdown | Narrative notes, outliers, recovery flags |
| `STEP_01/recovery_log.md` | Markdown (CONDITIONAL — only if T-01.1.1 ran) | Strategies attempted, what changed |

**Completion criteria (PHASE_QA verifies all):**

1. `primitives_catalog.json` is valid JSON
2. `len(primitives_catalog.primitives)` is between 5 and 15 inclusive at the main tier
3. All five primitive types have been considered (count of each may be zero, but `tier_compression.md` must show that each type was searched for)
4. Hidden primitives surfaced (index, broadcast, assignment, iteration) — completion report must list each as "found at <tier>" or "absent in this domain because <reason>"
5. Two-path check applied — completion report cites whether the domain has both sum-based AND order-based paths
6. Compression measured at every tier transition; ratio at main tier ≥ 18:1 (target ~20:1; <18:1 escalates with rationale)
7. Sparsity verifiable — completion report must state "most library APIs in `target_library` are expressible as combinations of primitives" with evidence count

**Acceptance command (PHASE_QA runs):**
```bash
jq -e '.primitives | length | select(. >= 5 and . <= 15)' workspace_dir/STEP_01/primitives_catalog.json
# Expected: prints the count (5-15); exit 0
```

**Plus structural QA:**
- `python -c "import json; d=json.load(open('workspace_dir/STEP_01/primitives_catalog.json')); assert all('tier' in p and 'type' in p and 'evidence' in p for p in d['primitives'])"` exits 0

**Do NOT:**
- Do not fabricate primitives without nexus-signal evidence (every entry needs a `evidence` field citing a specific nexus report)
- Do not dismiss zero-cost operations (STRUCTURAL primitives are essential)
- Do not stop at one tier (find primitives at every tier 0-3 even if some tiers are sparse)
- Do not proceed to T-01.2 until completion criteria met
- Do not skip the acceptance command

**Verdict mapping:**
- All criteria met + acceptance commands exit 0 → `TASK_PASS`
- Any Critical finding (fabrication, missing required output file) → `TASK_FAIL_RETRY` if retry budget remains, else `TASK_FAIL_ESCALATE`
- High/Medium findings (sparse tier coverage, weak nexus evidence) → `TASK_PASS` with warnings if non-numerous; else `TASK_FAIL_RETRY`

---

## T-01.1.1 — Recover Ops (on-demand)

**Trigger:** T-01.1 reaches retry limit without TASK_PASS.

**Source docs:**
- `STEP 1.1 - RECOVER OPS.md`

**Inputs:**
- T-01.1's partial `primitives_catalog.json` and `deconstruction_notes.md`
- T-01.1's STEP_QA findings (which criterion(a) failed)
- The tier that got stuck

**Required outputs:**

| File | Format | Required fields |
|---|---|---|
| `STEP_01/recovery_log.md` | Markdown | Strategy attempted (1 of 4), what changed, prior-state diff |
| `STEP_01/primitives_catalog.json` | JSON | UPDATED catalog (revised; replaces prior) |

**Recovery strategies (apply ONE per T-01.1.1 spawn, in priority order):**

1. Fresh re-read of source examples with deliberate "what did I miss" lens
2. Compound-primitive check: is a claimed "atom" actually a combination of others?
3. Cross-domain analogy: how does another domain handle this shape?
4. Tier escalation: move from computational to cognitive, or cognitive to goal

**Completion criteria:**

1. `recovery_log.md` exists and names exactly ONE strategy attempted
2. `primitives_catalog.json` is updated (timestamp newer than recovery start)
3. After recovery, T-01.1 acceptance command passes — OR — the recovery_log explicitly escalates: "this domain doesn't decompose at the attempted tier, human judgment needed"

**Acceptance command:**
```bash
# Re-run T-01.1's acceptance commands; they must now pass
jq -e '.primitives | length | select(. >= 5 and . <= 15)' workspace_dir/STEP_01/primitives_catalog.json
```

**Do NOT:**
- Do not invent primitives to force completion
- Do not retry the same strategy that failed (track prior strategies in `recovery_log.md`)
- Do not mix multiple strategies in a single spawn — one strategy per recovery pass; spawn again if needed
- Do not run if T-01.1 has not actually escalated

**Verdict mapping:**
- Recovery strategy applied + T-01.1 acceptance now passes → `TASK_PASS` (then T-01.1 is marked `TASK_PASS` and Phase 1 advances to T-01.2)
- Strategy applied but T-01.1 still fails → another T-01.1.1 spawn with next strategy (max 3 total recovery attempts)
- All 4 strategies exhausted → `TASK_FAIL_ESCALATE` for T-01.1; phase HOLD

---

## T-01.2 — Deconstruction Objs

**Source docs:**
- `STEP 2 - DECONSTRUCTION OBJS.md`

**Inputs:**
- T-01.1's `primitives_catalog.json` (every primitive needs input/output objects identified)
- `target_library` (for object structure analysis)

**Required outputs:**

| File | Format | Required fields |
|---|---|---|
| `STEP_02/object_hierarchy.json` | JSON | 5-level hierarchy: `{level_0_raw: [...], level_1_features: [...], level_2_primitives: [...], level_3_entities: [...], level_4_abstractions: [...]}`; each object: `{name, level, description, structure, produced_by (op list), consumed_by (op list), canonical_form, serialization, size_range}` |
| `STEP_02/object_operation_matrix.md` | Markdown | Matrix: rows = objects, columns = operations; cells = "consumes"/"produces"/blank |

**Completion criteria:**

1. Every operation in `primitives_catalog.json` has named input object(s) AND output object(s) in `object_hierarchy.json`
2. Total object count is between 5 and 20 inclusive (too few = too coarse; too many = too fine-grained)
3. No object named "data", "result", "output", or other vague names
4. Object-operation matrix is complete (no gaps for primitives in catalog)
5. Each level is well-formed (each level's objects aggregate lower-level objects)

**Acceptance command:**
```bash
python -c "
import json
h = json.load(open('workspace_dir/STEP_02/object_hierarchy.json'))
total = sum(len(v) for v in h.values() if isinstance(v, list))
assert 5 <= total <= 20, f'Object count {total} out of range [5,20]'
prims = json.load(open('workspace_dir/STEP_01/primitives_catalog.json'))
all_objs = {o['name'] for level in h.values() if isinstance(level, list) for o in level}
for p in prims['primitives']:
    for io in p.get('inputs', []) + p.get('outputs', []):
        assert io in all_objs, f'Object {io!r} from primitive {p[\"name\"]!r} not in hierarchy'
print(f'PASS: {total} objects, all primitives mapped')
"
```

**Do NOT:**
- Do not use vague names (`data`, `result`, `processed_thing`)
- Do not skip intermediate objects (if Filter takes DataFrame and produces DataFrame but actually passes through an EdgeMap, name the EdgeMap)
- Do not conflate types with instances (`RedCircle` is a value, `Circle` is a type)
- Do not explode objects to 47 — that means you haven't found the pattern

**Verdict mapping:**
- All criteria met + acceptance command exits 0 → `TASK_PASS`
- Vague-name finding → `TASK_FAIL_RETRY` (Critical — vague names are non-negotiable)
- Object count out of [5,20] → `TASK_FAIL_RETRY` with guidance to merge or split

---

## T-01.3 — Deconstruction Types

**Source docs:**
- `STEP 3 - DECONSTRUCTION TYPES.md`

**Inputs:**
- `primitives_catalog.json` (T-01.1)
- `object_hierarchy.json` (T-01.2)

**Required outputs:**

| File | Format | Required fields |
|---|---|---|
| `STEP_03/type_signatures.json` | JSON | Per operation: `{name, signature: {inputs: [{name, type}], output: {type}, preconditions: [...], postconditions: [...], failure_modes: [...]}, composition: {chains_from: [...], chains_to: [...]}}` |
| `STEP_03/composition_graph.dot` (or `.json`) | Graphviz DOT or JSON adjacency | Nodes = types; edges = operations |
| `STEP_03/type_algebra.md` | Markdown | Type definitions via type equations |

**Completion criteria:**

1. Every operation in `primitives_catalog.json` has a complete signature in `type_signatures.json`
2. No type field contains `data`, `result`, or `Any` except where rationale is given
3. Composition graph is connected (or explicitly disconnected with rationale) — every type appears as input or output of at least one operation
4. All operations that can fail carry `Maybe` or `Either` in their output type
5. Composition laws verified: associativity (provable for chained ops), identity (where applicable), Maybe propagation
6. No hidden state (every input is explicit in the signature)
7. Valid programs identifiable as paths through the composition graph

**Acceptance command:**
```bash
python -c "
import json
sigs = json.load(open('workspace_dir/STEP_03/type_signatures.json'))
prims = json.load(open('workspace_dir/STEP_01/primitives_catalog.json'))
prim_names = {p['name'] for p in prims['primitives']}
sig_names = {s['name'] for s in sigs}
missing = prim_names - sig_names
assert not missing, f'Missing signatures for: {missing}'
for s in sigs:
    for io in s['signature']['inputs'] + [s['signature']['output']]:
        assert 'type' in io and io['type'] not in ('data', 'result'), f'Vague type in {s[\"name\"]}'
print(f'PASS: {len(sigs)} signatures, all valid')
"
```

**Do NOT:**
- Do not use implicit types (`data`, `result`, `processed thing`)
- Do not hide state — every input explicit in signature
- Do not be overly polymorphic (`a → a` is meaningless)
- Do not leave failure untyped (use Maybe/Either)

**Verdict mapping:**
- All criteria met + acceptance command exits 0 → `TASK_PASS`
- Missing signatures → `TASK_FAIL_RETRY`
- Vague-type finding → `TASK_FAIL_RETRY` (Critical)

---

## Phase 1 verdict gate

When T-01.1, T-01.2, T-01.3 all `TASK_PASS`:
- QUEEN appends Phase 1 summary to `workspace_manifest.json` (`{phase: 1, tasks: [01.1, 01.1.1?, 01.2, 01.3], outputs: [...full path list], completed_at: ISO}`)
- QUEEN writes Phase 1 completion entry to `INPROGRESS.md` at project root (prepend-only)
- QUEEN emits `PHASE_GREEN_LIGHT` for Phase 1
- QUEEN advances to Phase 2 (T-02.1) per `PHASE_MODEL.md`

If any task ESCALATEs:
- QUEEN emits `PHASE_HOLD` for Phase 1
- Workflow paused; human notified with full task verdict trail
- Recovery: human edits source docs / nexus reports / target_library context, then re-engages workflow

---

*End of PHASE_01_CONTRACT.*
