# RDC CRON JOB — 5-Minute Loop Prompt

**Purpose:** Spawn a FULL RDC_WORKFLOW on one investigation directory every 5 minutes.
**Usage:** Use with `/schedule` or manual loop. Inject project knowledge, then process.

---

## THE PROMPT (copy everything below the line)

---

```
RDC_WORKFLOW CRON JOB — Process Next Investigation Directory

You are QUEEN. Execute a FULL compliant RDC_WORKFLOW.

## PHASE 0: PROJECT KNOWLEDGE INJECTION (MANDATORY)

Before ANY RDC work, you MUST read and internalize these files IN FULL:

1. **Read workflows/RDC/RDC_WORKFLOW.json** (506 lines) — the workflow spec
2. **Read workflows/RDC/WORKER_SCRIBE.md** — SCRIBE worker contract
3. **Read workflows/RDC/WORKER_TAXONOMIST.md** — TAXONOMIST worker contract
4. **Read workflows/RDC/WORKER_QA_COMPLETENESS.md** — QA completeness contract
5. **Read workflows/RDC/WORKER_QA_COHERENCE.md** — QA coherence contract
6. **Read workflows/SHARED/WORKER_QUEEN.md** — your QUEEN contract
7. **Read workflows/SHARED/WORKER_PROTOCOL.md** — universal worker protocol
8. **Read docs/TRINITY_LATEST.md** (2349 lines) — FULL PROJECT KNOWLEDGE
9. **Read docs/investigation/GRAND_SYNTHESIS.md** — synthesis of all investigations
10. **Read docs/FIX_BAD_RDC_FILES.md** — current audit state

DO NOT PROCEED until all 10 files are read IN FULL. No skipping. No summarizing.

## PHASE 1: SELECT TARGET

1. Read docs/FIX_BAD_RDC_FILES.md
2. Find first directory with status "⏳ PENDING"
3. If no pending directories: Report "ALL DIRECTORIES COMPLETE" and stop
4. Set TARGET = that directory path

## PHASE 2: INVENTORY

1. List ALL source .md files related to TARGET subsystem in docs/investigation/
2. Read EACH file IN FULL (every line, no truncation, no summarization)
3. Extract per file:
   - Filename
   - Timestamps/dates
   - All section headings
   - Key concepts defined
   - Dependencies mentioned
   - Architecture decisions
4. Determine temporal ordering
5. Write: docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/<subsystem>/INVENTORY.md

Report to user: "INVENTORY complete. N source files found. Proceeding to SCRIBE_LOOP."

## PHASE 3: SCRIBE_LOOP (Sequential, One Doc at a Time)

Initialize:
- MASTER.md = empty structured document
- PEDAGOGY.md = header only
- EVALUATIONS.md = header only

For EACH source doc in temporal order:

1. Read source doc IN FULL (again if needed)
2. Identify ALL concepts, decisions, constraints, dependencies
3. For each concept:
   - If NEW to MASTER: INSERT in structurally appropriate location
   - If REVISES existing: OVERWRITE in place, log to PEDAGOGY (prior + new + reason)
   - If CONTRADICTS without supersession: FLAG with conflict marker
4. Append to EVALUATIONS.md:
   - Doc name
   - Concepts added (list)
   - Concepts updated (list)
   - Concepts unchanged (list)
   - Conflicts flagged (list)
5. Write updated MASTER.md

CRITICAL RULES:
- Sequential. One doc at a time. In temporal order.
- Never delete concepts. Only overwrite or mark deprecated.
- Every concept from every source doc MUST land somewhere.
- No fabrication. Only what's in the source docs.

After final doc: Report "SCRIBE_LOOP complete. N concepts in MASTER. M conflicts flagged."

## PHASE 4: COURT (if conflicts)

If conflicts were flagged:

For each conflict:
1. Spawn 4 ADVOCATE workers in ONE message (run_in_background: true):
   - ADVOCATE_A1, ADVOCATE_A2 (argue for value A)
   - ADVOCATE_B1, ADVOCATE_B2 (argue for value B)
2. Wait for all 4 briefs
3. Apply decision criteria (from RDC_WORKFLOW.json §court_mechanism.decision_criteria_ordered):
   - Explicit supersession
   - Temporal primacy
   - Evidentiary weight
   - Architectural consistency
   - Load-bearing-ness
4. Rule: SIDE_A_WINS | SIDE_B_WINS | SYNTHESIS | DEFER | ESCALATE
5. Update MASTER.md with ruling + back-reference
6. Log to PEDAGOGY.md

If no conflicts: Skip to PHASE 5.

## PHASE 5: TAXONOMY

1. Read final MASTER.md IN FULL
2. Discover natural phase boundaries from content structure
3. Create:
   - PROJECT.md: scope, goals, constraints, acceptance criteria
   - CLARIFICATION.md: philosophical framing, design rationale
   - PHASE_1_ARCH.md, PHASE_2_ARCH.md, ... (one per discovered phase)
   - PHASE_1_TODO.md, PHASE_2_TODO.md, ... (one per phase, NOT combined)

Phase discovery rules:
- Phases reflect dependency ordering (N+1 depends on N outputs)
- Each phase is a coherent work unit
- TODO items are concrete, actionable, with acceptance criteria

Write to: docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/<subsystem>/

## PHASE 6: QA_UNIT

### QA_COMPLETENESS

Spawn QA_COMPLETENESS worker:
- Input: all source docs + all output docs + PEDAGOGY + MASTER
- Check: EVERY concept from EVERY source doc appears in:
  - Output documents, OR
  - PEDAGOGY.md as superseded/deprecated, OR
  - Court entry as resolved
- Output: completeness report with any MISSING concepts

If MISSING concepts found → SCRIBE_REVISIT (re-read specific source docs, re-upsert)

### QA_COHERENCE

Spawn QA_COHERENCE worker:
- Input: all output docs + MASTER
- Check:
  - Cross-references valid
  - Dependency ordering correct
  - No orphaned concepts
  - TODO items trace to ARCH decisions
  - No scope leaks between phases
- Output: coherence report with any structural issues

If structural issues found → RETAXONOMIZE (re-carve with corrections)

### VERDICT

- Both pass → GREEN_LIGHT
- Loop limit (3 cycles) → ESCALATE to human

## PHASE 7: FINALIZE

On GREEN_LIGHT:

1. Write completion summary to docs/FIX_BAD_RDC_FILES.md:
   - Directory: <path>
   - Source docs processed: N
   - Concepts in MASTER: N
   - Conflicts resolved: N
   - Phases discovered: N
   - QA verdict: GREEN_LIGHT
   - Output files: [list with line counts]
   - Timestamp: YYYY-MM-DD HH:MM

2. Update status in FIX_BAD_RDC_FILES.md: ⏳ PENDING → ✅ DONE

3. Report:
   "RDC CRON JOB COMPLETE
   Directory: <path>
   Status: GREEN_LIGHT
   Next pending: <next directory or NONE>
   
   Schedule next run in 5 minutes? Y/N"

## OUTPUT FILE STRUCTURE

docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/<subsystem>/
├── INVENTORY.md          # Temporal manifest
├── MASTER.md             # Consolidated running doc
├── PEDAGOGY.md           # Concept evolution log
├── EVALUATIONS.md        # Per-doc evaluation
├── PROJECT.md            # Scope/goals/constraints
├── CLARIFICATION.md      # Philosophical framing
├── PHASE_1_ARCH.md       # Phase 1 architecture
├── PHASE_1_TODO.md       # Phase 1 tasks
├── PHASE_2_ARCH.md       # Phase 2 architecture
├── PHASE_2_TODO.md       # Phase 2 tasks
└── ...                   # More phases as discovered

## HARD RULES (NEVER VIOLATE)

From RDC_WORKFLOW.json:
- no_greenlight_without_full_qa_unit: true
- scribe_loop_is_sequential_not_parallel: true
- master_upsert_never_deletes_only_overwrites: true
- pedagogy_is_append_only: true
- no_fabricated_concepts: true
- conflicts_go_through_court_not_silently_resolved: true
- taxonomist_discovers_phases_from_content: true

## FAILURE MODES

If you cannot complete:
1. Write partial state to FIX_BAD_RDC_FILES.md
2. Mark status as "🔴 BLOCKED: <reason>"
3. Report blocker to user
4. Do NOT proceed to next directory
```

---

## SCHEDULING

To run as 5-minute loop:

```
/schedule in 5 minutes: RDC CRON JOB — paste prompt from docs/CRON_JOB_PROMPT.md
```

Or manual invocation:
1. Paste the prompt above
2. Wait for completion
3. Review outputs
4. Repeat

---

## KNOWLEDGE INJECTION SUMMARY

The prompt requires reading **10 files** before ANY RDC work:

| File | Lines | Purpose |
|------|-------|---------|
| RDC_WORKFLOW.json | 506 | Workflow spec |
| WORKER_SCRIBE.md | ~200 | SCRIBE contract |
| WORKER_TAXONOMIST.md | ~250 | TAXONOMIST contract |
| WORKER_QA_COMPLETENESS.md | ~150 | QA completeness contract |
| WORKER_QA_COHERENCE.md | ~200 | QA coherence contract |
| WORKER_QUEEN.md | ~300 | QUEEN contract |
| WORKER_PROTOCOL.md | ~100 | Universal protocol |
| TRINITY_LATEST.md | 2349 | **FULL PROJECT KNOWLEDGE** |
| GRAND_SYNTHESIS.md | 214 | Investigation synthesis |
| FIX_BAD_RDC_FILES.md | ~100 | Current audit state |

**Total pre-read: ~4,369 lines of context**

This ensures the agent has FULL project knowledge before processing.

---

*Created: 2026-05-23*
*For: Python Investigation Correction Cron Loop*
