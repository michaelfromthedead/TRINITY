# COMPOSITOR vs. TAXONOMIST — Deliberate Comparison

**Purpose:** Document the structural similarities and differences between BOOK_CONSOLIDATION's COMPOSITOR role and RDC_WORKFLOW's TAXONOMIST role. Informs the open refactoring question in `BOOK_WORKFLOW_DISSERTATION.md` §16.2.

**References:**
- `workflows/BOOK/WORKER_COMPOSITOR.md` — COMPOSITOR role doc
- `workflows/RDC/WORKER_TAXONOMIST.md` — TAXONOMIST role doc
- `workflows/BOOK/BOOK_CONSOLIDATION.json` — BOOK_CONSOLIDATION workflow spec
- `workflows/RDC/RDC_WORKFLOW.json` — RDC_WORKFLOW spec (for context)
- Dissertation §16.2: "COMPOSITOR vs. TAXONOMIST refactoring"

---

## 1. What is identical

### 1.1 Role in the pipeline

Both roles occupy the exact same structural position:

```
SCRIBE_LOOP → [COURT (conditional)] → COMPOSITOR / TAXONOMIST → QA_UNIT
```

Both receive the final MASTER.md (after all SCRIBE passes and COURT resolutions) as their primary input. Both produce the document set that the downstream workflow consumes. Both are the "carving" step that transforms a consolidated single-file representation into a structured multi-file output.

### 1.2 The discovery-over-prescription principle

Both COMPOSITOR and TAXONOMIST are explicitly prohibited from predetermined structure.

**TAXONOMIST (from WORKER_TAXONOMIST.md §2):**
> "Phase structure is NOT predetermined — you discover it from content."
> "A common TAXONOMIST mistake: Predetermined phase structure (assumed before reading MASTER)."

**COMPOSITOR (from WORKER_COMPOSITOR.md §2):**
> "Chapter structure is NOT predetermined — you discover it from content."
> "You MUST NOT assume a chapter structure in advance."

This is the same philosophy operating on different material. Both roles use content-driven boundary discovery as the primary carving method. The heuristics differ in surface language (conceptual coherence vs. layer separation, SDLC-consumability vs. reader-meaningful unit) but are structurally parallel.

### 1.3 The MASTER-scanning algorithm structure

Both roles follow the same basic algorithm:
1. Read MASTER end-to-end
2. Identify natural groupings / clusters / phases from content signals
3. Determine ordering (dependency order — not source order, not arbitrary)
4. Verify the output set covers all MASTER content (self-check)
5. Produce the output set

Neither role is permitted to drop concepts silently. Both have the same "no silent dropping" discipline.

### 1.4 Faithful-carve discipline

Both roles carve from MASTER; they do not author new content. Both explicitly forbid:

- Introducing new claims not in MASTER
- Editing content substantially during carving
- Dropping concepts the worker deems "unimportant"

Both preserve court back-references in carved output.

### 1.5 Report structure

Both produce equivalent report blocks:
- Files produced (list)
- Discovery narrative (phase discovery vs. chapter discovery)
- Confidence level (HIGH / MEDIUM / LOW)
- Coverage verification (self-check before QA)
- Outstanding items for QA

### 1.6 Inputs

Both receive identical context inputs:
- MASTER.md (final state)
- PEDAGOGY.md (for evolution context)
- EVALUATIONS.md (for source-coverage understanding)
- INPROGRESS.md court entries (for resolved-conflict context)

COMPOSITOR additionally receives BOOK_MANIFEST.json (for genre/structure context). TAXONOMIST has no equivalent input because RDC has no per-project configuration analogous to BOOK_MANIFEST.

---

## 2. What is different

### 2.1 Output shape — the fundamental structural difference

This is the deepest difference. It determines everything else.

**TAXONOMIST** produces an engineering document set:
- `PROJECT.md` — project-wide scope (always exactly one)
- `PHASE_<N>_<NAME>_ARCH.md` — architectural context per phase
- `PHASE_<N>_<NAME>_TODO.md` — actionable task list per phase
- `CLARIFICATION.md` — philosophical/meta framing (always exactly one)

**COMPOSITOR** produces a manuscript document set:
- `chapters/CH_<NN>_<TITLE>.md` — one chapter file per discovered chapter
- `STRUCTURE.md` — the manuscript skeleton (TOC + chapter summaries + section listings + dependency map)

TAXONOMIST's output has a fixed document type inventory (PROJECT, ARCH×N, TODO×N, CLARIFICATION). COMPOSITOR's output has one variable-length array (chapter files) and one fixed summary file.

The TAXONOMIST split (ARCH vs. TODO) is a semantic split — separating architectural context from actionable tasks within each phase. COMPOSITOR has no equivalent semantic split within a chapter — the chapter is the chapter.

### 2.2 Discovery target — phases vs. chapters

**TAXONOMIST** discovers **phases**:
- A phase is a bounded engineering work block that SDLC_WORKFLOW can execute
- Phase granularity is calibrated to SDLC feasibility (weeks-to-months of work per phase)
- Phase boundaries are about deliverables, stack layers, and SDLC-consumability
- A phase that is "too big" causes SDLC workers to drown in scope

**COMPOSITOR** discovers **chapters**:
- A chapter is a reader-meaningful conceptual unit in a manuscript
- Chapter granularity is calibrated to reader cognition and manuscript genre
- Chapter boundaries are about conceptual coherence, pedagogical sequencing, and manuscript arc
- A chapter that is "too big" means the reader can't absorb it as a unit

The granularity criteria are structurally parallel (both bounded units for a downstream consumer), but the downstream consumers are radically different (SDLC worker vs. human reader), so the calibration criteria are different.

### 2.3 Dependency semantics — engineering vs. manuscript

**TAXONOMIST** phase dependencies reflect engineering dependencies:
- Phase N+1 depends on Phase N if the code or systems from Phase N are required to execute Phase N+1
- Dependencies are typically about artifacts (libraries, compiled binaries, test harnesses)
- Cycles are impossible in a well-structured engineering project

**COMPOSITOR** chapter dependencies reflect conceptual dependencies:
- Chapter N+1 depends on Chapter N if the reader must understand concepts from Chapter N to follow Chapter N+1
- Dependencies are about knowledge state, not physical artifacts
- The reader's mental model is what propagates, not compilation outputs
- Cycles can appear in manuscript drafts (author assumed X was already explained when it wasn't) — these are content problems, not just structural problems

### 2.4 Section hierarchy — novel to COMPOSITOR

TAXONOMIST produces a flat structure within each phase: ARCH.md contains prose sections, TODO.md contains task entries. There is no deep section hierarchy specification — SDLC workers consume the TODO task list directly.

COMPOSITOR produces an explicit section hierarchy within each chapter file (H1 chapter title, H2 sections, H3 subsections). This hierarchy is:
- Specified in WORKER_COMPOSITOR.md §3
- Listed in STRUCTURE.md per-chapter section listings
- Validated by QA_COHERENCE (section misplacement check)

The section hierarchy serves downstream BOOK_STORYBOARD workers, who produce per-section storyboard entries. TAXONOMIST has no equivalent because SDLC workers work at the task level, not the section level.

### 2.5 The ARCH/TODO split — absent in COMPOSITOR

TAXONOMIST's core semantic operation is splitting each phase into two documents: architectural context (ARCH) vs. actionable tasks (TODO). This separation serves the SDLC pattern where DEV workers need architectural context and task lists independently.

COMPOSITOR has no ARCH/TODO split. A chapter file is unified — prose, argumentation, and examples in a single document. There is no equivalent of the architectural-vs-task distinction in manuscript material.

This means TAXONOMIST's output requires two documents per phase for N phases (2N+2 documents), while COMPOSITOR's output requires one document per chapter for N chapters (N+1 documents including STRUCTURE.md).

### 2.6 The structural skeleton document — STRUCTURE.md vs. PROJECT.md

Both produce one "master view" document:
- TAXONOMIST: `PROJECT.md` (project-wide scope, goals, constraints, phase overview)
- COMPOSITOR: `STRUCTURE.md` (manuscript skeleton: TOC, chapter summaries, section listings, dependency map)

These documents are structurally analogous but semantically different:

| Aspect | PROJECT.md | STRUCTURE.md |
|---|---|---|
| Purpose | Project scope + phase navigation | Manuscript skeleton + structural reference |
| Content | Goals, constraints, non-goals, phase overview | TOC, chapter summaries, section listings, dependency map |
| Audience | SDLC workers (QUEEN, DEV, QA) | BOOK_STORYBOARD workers, BOOK_EDITORIAL workers |
| Dependency map? | No (phases have known order) | Yes (chapters have discovered dependency order) |
| Section listings? | No | Yes (explicit per-chapter) |

COMPOSITOR's STRUCTURE.md is more elaborate than TAXONOMIST's PROJECT.md because downstream BOOK workflows (particularly STORYBOARD and EDITORIAL) need richer structural metadata than SDLC's phase-list-plus-scope.

### 2.7 CLARIFICATION.md — absent from COMPOSITOR

TAXONOMIST produces CLARIFICATION.md — a philosophical/pedagogical framing document for the engineering project. This captures "why it looks this way" and "what we learned" context.

COMPOSITOR produces no equivalent. In BOOK_CONSOLIDATION, the pedagogical framing IS the manuscript — it doesn't need a separate meta-document. The PEDAGOGY.md (produced by SCRIBE_LOOP) serves the archaeological record function; STRUCTURE.md serves the navigation function; the chapter files serve the content function.

### 2.8 Downstream consumer and validation semantics

**TAXONOMIST** → **SDLC_WORKFLOW**:
- QA_COHERENCE checks: TODO↔ARCH references, phase dependency ordering, cross-references, scope separation between phases
- The carved output is directly executable by SDLC workers

**COMPOSITOR** → **BOOK_STORYBOARD**:
- QA_COHERENCE checks: chapter ordering, orphaned concepts, section misplacement, STRUCTURE.md consistency, dependency acyclicity, co-location
- The carved output is the input for a CONSTRUCTIVE workflow (STORYBOARD), not directly executable
- A human review step occurs between COMPOSITOR output and STORYBOARD trigger

This difference has downstream effects: TAXONOMIST's output validity is measured by SDLC-executability (can a DEV worker take T-P1.0.1 and produce the deliverable?). COMPOSITOR's output validity is measured by storyboard-readiness (can a storyboarder produce a coherent logical skeleton from these chapters?).

---

## 3. Summary table

| Dimension | TAXONOMIST (RDC) | COMPOSITOR (BOOK) |
|---|---|---|
| Role position in pipeline | SCRIBE_LOOP → COURT → [this] → QA | Identical |
| Discovery principle | phases from content, not predetermined | chapters from content, not predetermined |
| Primary input | MASTER.md | MASTER.md + BOOK_MANIFEST.json |
| Output type | PROJECT + ARCH×N + TODO×N + CLARIFICATION | chapters/CH_NN×N + STRUCTURE.md |
| Output structure | flat (PROJECT, per-phase pairs, CLARIFICATION) | one chapter file per chapter + one skeleton |
| Output count formula | 2N + 2 documents | N + 1 documents |
| ARCH/TODO split | yes (semantic split within each phase) | no (chapter is unified) |
| Dependency type | engineering artifact dependencies | reader knowledge-state dependencies |
| Section hierarchy | flat (ARCH prose, TODO tasks) | explicit (H1→H2→H3 per chapter) |
| Structural skeleton doc | PROJECT.md (scope + phase overview) | STRUCTURE.md (TOC + summaries + sections + DAG) |
| Dependency map in output | no (implicit in phase numbering) | yes (explicit DAG in STRUCTURE.md) |
| Meta-framing doc | CLARIFICATION.md | none (content IS the framing) |
| Downstream consumer | SDLC_WORKFLOW (execution) | BOOK_STORYBOARD (constructive planning) |
| Granularity calibration | SDLC-feasibility (weeks-months of work) | reader-cognitive (manuscript arc unit) |
| QA_COHERENCE primary checks | TODO/ARCH refs, cross-refs, scope separation | ordering, orphans, section placement, DAG |

---

## 4. Implications for refactoring (dissertation §16.2)

### 4.1 What is genuinely shared

The following could plausibly be extracted into a shared base role:

- MASTER-scanning algorithm structure (read → cluster → order → carve → verify)
- Discovery-over-prescription principle (not predetermined)
- Faithful-carve discipline (no new content, no silent dropping)
- Coverage self-check before reporting
- Report structure (files produced, discovery narrative, confidence level, outstanding)
- Court back-reference preservation
- Blocked handling patterns (MASTER too incoherent → BLOCKED; ambiguous boundaries → flag)

### 4.2 What is genuinely different (not factored)

The following differences are deep enough that a shared base would require heavy parameterization:

- Output shape is fundamentally different (ARCH/TODO pairs vs. chapter files)
- Section hierarchy spec is novel to COMPOSITOR — no TAXONOMIST equivalent
- STRUCTURE.md format is COMPOSITOR-specific and complex
- CLARIFICATION.md is TAXONOMIST-specific
- Dependency semantics are different (engineering artifacts vs. knowledge state)
- Granularity calibration criteria are different
- BOOK_MANIFEST.json input is COMPOSITOR-specific

A parameterized base role (`output_mode: "chapters" | "phases"`) would need to branch on nearly every concrete decision. The parameterization surface would be approximately as large as the two roles themselves.

### 4.3 Proposed disposition

**Defer refactoring until after the first real BOOK project completes.** Rationale:

1. COMPOSITOR has not been tested on real material yet. Its algorithm may need adjustment. Premature abstraction before empirical validation is risky.
2. The shared surface is real but smaller than it appears — the high-level algorithm is similar but the low-level specifics diverge substantially.
3. The cost of maintaining two role docs is low (they're narrative documents, not code). The cost of a premature abstraction that turns out to be wrong is higher.
4. After the first BOOK project, concrete empirical experience will inform which parts of COMPOSITOR's algorithm were stable and which needed adjustment — making the shared-vs-different boundary clearer.

This disposition is consistent with the dissertation's recommendation in §16.2.

---

*End of COMPOSITOR_VS_TAXONOMIST comparison doc.*
