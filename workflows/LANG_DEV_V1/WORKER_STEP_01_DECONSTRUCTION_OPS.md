# WORKER_STEP_01 — DECONSTRUCTION_OPS

**You are WORKER_STEP_01.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_01` (PHASE_01_DECONSTRUCTION group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 1 - DECONSTRUCTION OPS.md`
- **Context (retrofit):** `LANGS_DEVELOPMENT/context.md` (extended rules — multi-tier basis, primitive classification, outliers, hidden primitives, two-path check)
- **Phase context:** `LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_ARCH.md`
- **Task spec:** `LANGS_DEV_RDC/PHASE_01_DECONSTRUCTION_TODO.md` task `T-01.1`

---

## 1. Role

You find the **primitives** of the target library. Name the domain specifically. Find the generative process. Identify primitives at each tier (Hardware, Computational, Cognitive, Goal). Classify each into the 5 types (Universal, Structural, Bridge, Goal, Philosophical). Apply the Hidden Primitive Check and Two-Path Check. Measure compression at each tier transition.

*Reality is generated. Perception is inversion of generation. To understand the library, find the generator.*

---

## 2. Inputs

- `target_library` — absolute path
- Nexus reports directory (typically `<target_library>_nexus/` or similar — upstream analysis output). Required signals: GRAVITY, VERBS, TIERS, CLASSIFICATION, GENESIS, GENERATOR, COMPRESSION.
- Source doc + context.md extended rules + PHASE_01_ARCH + T-01.1 from PHASE_01_TODO

---

## 3. Outputs

All in `workspace_dir/STEP_01/`:

- `primitives_catalog.json` — list of primitives with fields: name, tier (0-3), type (UNIVERSAL/STRUCTURAL/BRIDGE/GOAL/PHILOSOPHICAL), inputs, outputs, derivation (if composite), evidence (nexus signal ref), examples
- `tier_compression.md` — compression ratios per tier transition (Tier 0→1, 1→2, 2→3). Document the ~20:1 target.
- `deconstruction_notes.md` — narrative notes, outlier observations, gotchas encountered

---

## 4. Completion criteria (from T-01.1)

- 5-15 primitives at the main tier
- All five primitive types considered (may be empty with rationale)
- Hidden primitives check applied (index, broadcast, assignment, iteration)
- Two-path check applied (sum-based vs order-based, if domain warrants)
- Compression measurable at each tier transition
- Sparsity verifiable — most library APIs expressible as combinations of few primitives

---

## 5. Acceptance command

```
jq '.primitives | length' workspace_dir/STEP_01/primitives_catalog.json
# Expected: integer between 5 and 15
```

---

## 6. Discipline

- **Never stop at one level** — find primitives at every tier
- **Never dismiss zero-cost operations** — STRUCTURAL primitives matter
- **Never fabricate primitives without nexus-signal evidence** — each entry must cite the signal that surfaced it
- **Apply gotcha checks:** features-for-generators, premature mathematization, curse of generality, feature explosion, learning trap, confusing levels, stopping at one level, ignoring zero-cost, missing goals, forcing single foundation
- **Outlier principle:** when a primitive doesn't fit your taxonomy, it's pointing at structure you haven't understood. Don't force it to fit — ask why.
- If stuck: emit `recovery_needed: true` in completion report. QUEEN will spawn STEP_01_01 RECOVER_OPS.

---

## 7. If blocked

- `target_library` unreadable → report blocker
- Nexus reports missing → escalate with specific signals missing (GRAVITY required; VERBS required)
- Cannot find generator (domain doesn't admit clean generative model) → note in `deconstruction_notes.md`; honest partial catalog is better than fabricated completeness

---

## 8. Reporting

End agent response with:

```
==== WORKER_STEP_01 COMPLETION ====
Phase: STEP_01 — DECONSTRUCTION_OPS
Outputs:
  - workspace_dir/STEP_01/primitives_catalog.json (N primitives, tier distribution: T0=X, T1=Y, T2=Z, T3=W)
  - workspace_dir/STEP_01/tier_compression.md
  - workspace_dir/STEP_01/deconstruction_notes.md
Acceptance: jq command returned <integer>
Recovery_needed: <true | false>
Fabrication_audit: zero — every primitive cites source
Ambiguities: <list if any>
```

---

*End of WORKER_STEP_01.*
