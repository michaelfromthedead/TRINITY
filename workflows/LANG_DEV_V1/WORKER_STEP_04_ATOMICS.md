# WORKER_STEP_04 — ATOMICS

**You are WORKER_STEP_04.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_04` (PHASE_02_DESIGN group).

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 4 - ATOMICS.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_02_DESIGN_ARCH.md` (§2.1)
- **Task spec:** `LANGS_DEV_RDC/PHASE_02_DESIGN_TODO.md` task `T-02.1`

---

## 1. Role

You design the **atoms** — self-assembling primitives with typed ports, phases, and PCFG weights. For each primitive from STEP_01, define: identity + inputs (typed ports) + outputs (typed ports) + phase + execution-stub. You also define the port types enum (~5-10) and phases enum (~3-8). Output: atoms+ports+phases draft that STEP_05A will materialize into decisions.json.

*The bag is the interface. Atoms snap together automatically. Shuffling the bag produces the same result — THE acid test.*

---

## 2. Inputs

- `workspace_dir/STEP_01/primitives_catalog.json`
- `workspace_dir/STEP_02/object_hierarchy.json`
- `workspace_dir/STEP_03/type_signatures.json`
- Source doc + PHASE_02_ARCH + T-02.1 from PHASE_02_TODO

---

## 3. Outputs

All in `workspace_dir/STEP_04/`:

- `atoms_draft.json` — full atom list with name, description, inputs (port types), outputs (port types), phase
- `port_types_draft.json` — port type enum with name, description, examples (derived from type signatures)
- `phases_draft.json` — phase list with name, order (0-indexed), description
- `pcfg_weights_initial.json` — uniform initial weights file (structure: `{"Start": {...}, "AfterX": {...}}`)

---

## 4. Completion criteria (from T-02.1)

- Every primitive from STEP_01 has a corresponding atom (1:1 or 1:many)
- 5-10 distinct port types (not fewer, not more)
- 3-8 phases with monotone order numbers
- Every atom.phase references a defined phase
- Every atom.inputs/outputs references defined port types
- PCFG weights file present with uniform initial distribution
- Acid test acknowledged (shuffle invariance will be verified in STEP_11)

---

## 5. Acceptance command

```
python workspace_dir/STEP_04/validate_atoms.py
# Expected: "Atoms: N. Port types: M. Phases: P. Referential integrity: OK."
```

---

## 6. Discipline

- **Atoms are self-contained.** No hidden state, no hidden dependencies.
- **Port types are finite, distinguishable, meaningful.** 5-10 is the sweet spot.
- **Phases have semantic justification.** "SOURCE before SINK" because causally, not arbitrarily.
- **PCFG weights go in a file, not as code constants.** Fake self-assembly (hardcoded `if name == X: priority = 1`) is forbidden.
- **Known limitation:** port types don't capture column-level semantic dependencies (Filter uses column Compute creates). That's STEP_10's job.
- **Constraint = liberation.** Tighter operation-set → better self-assembly. Don't over-generalize.

Gotchas: type-explosion (>15 port types is a smell), fake-self-assembly (hardcoded if-else disguised), forgetting-PCFG (weights file must exist even if uniform).

---

## 7. If blocked

- STEP_01/02/03 outputs missing → escalate
- Port types genuinely ambiguous → propose 2 alternatives in drafts; STEP_05A will resolve during schema materialization

---

## 8. Reporting

```
==== WORKER_STEP_04 COMPLETION ====
Phase: STEP_04 — ATOMICS
Atoms: <N>
Port types: <M>
Phases: <P>
Outputs:
  - workspace_dir/STEP_04/atoms_draft.json
  - workspace_dir/STEP_04/port_types_draft.json
  - workspace_dir/STEP_04/phases_draft.json
  - workspace_dir/STEP_04/pcfg_weights_initial.json
Acceptance: validate_atoms.py returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_04.*
