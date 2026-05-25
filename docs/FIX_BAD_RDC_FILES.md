# FIX_BAD_RDC_FILES — Audit & Correction Tracker

**Created:** 2026-05-23
**Status:** IN PROGRESS
**Owner:** Python swarm

---

## Problem Statement

The `docs/investigation/` source files (167) were processed into `docs/INVESTIGATION_PHASE_X_OUTPUT/` (35 directories, 127 files) using a **non-compliant RDC process**. The outputs are missing:

1. **MASTER.md** — Running consolidated document (required intermediate)
2. **PEDAGOGY.md** — Concept evolution log (required intermediate)
3. **EVALUATIONS.md** — Per-document evaluation (required intermediate)
4. **INVENTORY.md** — Temporal manifest (required intermediate)
5. **Per-phase TODO files** — Should be one per ARCH, not combined

The existing outputs may have **concept loss** or **structural incoherence** because the proper RDC_WORKFLOW phases were skipped.

---

## Correction Strategy

**Approach:** Re-run RDC_WORKFLOW on each investigation directory, producing compliant outputs.

**Work Unit:** One directory = one RDC_WORKFLOW execution (see CRON_JOB_PROMPT.md)

**Output Location:** `docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/<subsystem>/`

---

## Audit Checklist

### Source Directories (docs/investigation/)

| # | Source File | Has Output? | Needs Reprocess? | Status |
|---|-------------|-------------|------------------|--------|
| 1 | engine_core_math.md | Yes | TBD | ⏳ PENDING |
| 2 | engine_core_ecs.md | Yes | TBD | ⏳ PENDING |
| 3 | engine_core_scheduler.md | Yes | TBD | ⏳ PENDING |
| 4 | engine_core_tasks.md | Yes | TBD | ⏳ PENDING |
| ... | (167 total) | ... | ... | ... |

### Output Directories (docs/INVESTIGATION_PHASE_X_OUTPUT/)

| # | Directory | Files | Missing Per Spec | Status |
|---|-----------|-------|------------------|--------|
| 1 | engine_animation_crowds_facial | 4 | MASTER, PEDAGOGY, EVALUATIONS, INVENTORY | ⏳ PENDING |
| 2 | engine_animation_graph_ik | 4 | MASTER, PEDAGOGY, EVALUATIONS, INVENTORY | ⏳ PENDING |
| 3 | engine_animation_motionmatching_procedural | 4 | MASTER, PEDAGOGY, EVALUATIONS, INVENTORY | ⏳ PENDING |
| ... | (35 total) | ... | ... | ... |

---

## Progress Log

### 2026-05-23

- **00:XX** — Identified problem: RDC outputs missing required intermediates
- **00:XX** — Created FIX_BAD_RDC_FILES.md (this document)
- **00:XX** — Created CRON_JOB_PROMPT.md (reprocessing workflow)

---

## Completion Criteria

- [ ] All 35 directories reprocessed through compliant RDC_WORKFLOW
- [ ] Each output directory contains: PROJECT.md, CLARIFICATION.md, MASTER.md, PEDAGOGY.md, EVALUATIONS.md, INVENTORY.md, PHASE_N_ARCH.md (per phase), PHASE_N_TODO.md (per phase)
- [ ] QA_COMPLETENESS passes (zero concept loss)
- [ ] QA_COHERENCE passes (zero structural issues)
- [ ] Old outputs archived or deleted

---

## Notes

- Source: `docs/investigation/*.md` (167 files)
- Bad outputs: `docs/INVESTIGATION_PHASE_X_OUTPUT/` (35 dirs)
- Corrected outputs: `docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/` (TBD)
- Workflow spec: `workflows/RDC/RDC_WORKFLOW.json`
