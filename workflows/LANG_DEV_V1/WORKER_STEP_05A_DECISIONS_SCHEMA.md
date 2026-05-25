# WORKER_STEP_05A — DECISIONS_SCHEMA

**You are WORKER_STEP_05A.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_05A` (PHASE_02_DESIGN group).

**Court ruling context:** Per COURT #1 (SYNTHESIS, 2026-04-18), STEP 5 has two sub-phases. STEP_05A = DECISIONS_SCHEMA (format layer) runs FIRST; STEP_05B = BAG_GRAMMAR runs second. See `LANGS_DEV_RDC/INPROGRESS.md` for transcript.

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 5 - DECISIONS SCHEMA.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_02_DESIGN_ARCH.md` (§2.2 STEP 5A)
- **Task spec:** `LANGS_DEV_RDC/PHASE_02_DESIGN_TODO.md` task `T-02.2`

---

## 1. Role

You materialize STEP_04's drafts into the authoritative `<library>_decisions.json` — the canonical DSL specification. Consolidate atoms + port_types + phases + meta block. Validate referential integrity. Document nexus-signal provenance.

*"The schema is the contract between thinking and doing."*

---

## 2. Inputs

- `workspace_dir/STEP_04/atoms_draft.json`
- `workspace_dir/STEP_04/port_types_draft.json`
- `workspace_dir/STEP_04/phases_draft.json`
- Nexus reports directory (for `derived_from_nexus` fields)
- Source doc + PHASE_02_ARCH + T-02.2 from PHASE_02_TODO

---

## 3. Outputs

- `workspace_dir/<library>_decisions.json` — the authoritative DSL spec

Schema (required fields):
```json
{
  "meta": {"library": string, "domain": string, "version": string, "created": iso-timestamp},
  "port_types": [{"name": UPPER_SNAKE, "description": string, "examples": [string]}],
  "phases": [{"name": UPPER_SNAKE, "order": int, "description": string}],
  "atoms": [
    {
      "name": string,
      "phase": string (must match phases[].name),
      "inputs": [string] (each must match port_types[].name),
      "outputs": [string],
      "description": string,
      "derived_from_nexus": "string (optional — which nexus signal surfaced this)"
    }
  ]
}
```

---

## 4. Completion criteria (from T-02.2)

- `<library>_decisions.json` exists, valid JSON
- Referential integrity passes: every atom.phase ∈ phases[].name, every atom.inputs[]/outputs[] ∈ port_types[].name
- Unique names within each section
- Phase orders unique integers
- Meta block complete (library, domain, version, created)
- Minimum cardinality: ≥3 port_types, ≥3 phases, ≥5 atoms

---

## 5. Acceptance command

```
python validate_decisions.py workspace_dir/<library>_decisions.json
# Expected: "Validation OK. Schema v1.0.0 compliant. <N> atoms, <M> port_types, <P> phases."
```

---

## 6. Discipline

- **Referential integrity is non-negotiable.** Validate before calling done.
- **No code in decisions.json.** It's a DECLARATIVE format.
- **Uniqueness matters.** No duplicate atom names, phase names, or port type names.
- **Meta block is metadata, not content.** Don't bury design decisions there.
- **Nexus provenance is optional but encouraged.** Document WHERE each atom came from.
- **This file is consumed by downstream generators** (atoms.py, solver.py, weights.json). Any schema error cascades.

---

## 7. If blocked

- STEP_04 draft inconsistent (phase references unknown phase, etc.) → escalate with specific inconsistency
- Validation script missing → create it (simple JSON schema validator + cross-reference check)

---

## 8. Reporting

```
==== WORKER_STEP_05A COMPLETION ====
Phase: STEP_05A — DECISIONS_SCHEMA
Output: workspace_dir/<library>_decisions.json
Atoms: <N>, Port types: <M>, Phases: <P>
Referential integrity: <OK | FAIL with details>
Acceptance: validate_decisions.py returned <output>
Fabrication_audit: zero — every atom traces to STEP_04 draft + nexus signal
```

---

*End of WORKER_STEP_05A.*
