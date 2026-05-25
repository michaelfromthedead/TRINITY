# BOOK — The Book Writing Cycle

## A Workflow System for Manuscript Consolidation, Editorial Review, and Print Production

**Version:** 1.0.0
**Author:** Michael (owner) + Claude (co-architect)
**Status:** IMPLEMENTED (as of 2026-04-18). All 6 workflow JSONs, 18 worker docs, and core infrastructure artifacts are built. This document is the authoritative design reference and implementation record for the BOOK workflow family.

---

## 1. Purpose and Scope

BOOK is a family of AI swarm workflows that transform raw written material into print-ready manuscripts. It operates on the same swarm architecture as the existing SDLC and RDC workflows — QUEEN orchestrator, role-specialized workers, adversarial QA, loop limits with auto-escalation, INPROGRESS.md as the unified progress log — but targets a fundamentally different domain: authored prose rather than source code or engineering documentation.

The system handles a wide range of written works: academic meta-studies, exploratory science books, mathematical exposition, hard science fiction, theoretical physics, and cognitive science texts. The author (Michael) writes across all of these. The system must accommodate different genres, voice registers, pedagogical contracts, and stylistic conventions without being hard-coded to any single one.

BOOK is designed as a pipeline of independent workflows with manual handoff between stages. Each workflow is self-contained and independently triggerable. A manuscript can enter the pipeline at any stage depending on its current state of completion.

---

## 2. Pipeline Overview

The BOOK pipeline consists of one pre-check operation and four workflows, executed in sequence with human review between each stage:

```
BOOK_TRIAGE (pre-check — classifies input, produces manifest)
    ↓ human reviews manifest, sets template declarations, confirms routing
BOOK_CONSOLIDATION (chaotic folder → structured manuscript)
    ↓ human reviews structured output
BOOK_STORYBOARD (structured manuscript → storyboarded manuscript)
    ↓ human reviews storyboard
BOOK_EDITORIAL (storyboarded manuscript → polished manuscript)
    ↓ human reviews polished output
BOOK_PRODUCTION (polished manuscript → automation-ready intermediate)
    ↓ mechanical automation (not AI)
[LULU_PIPELINE] → print-ready PDF
```

Entry points are flexible. Material that is already structured can skip BOOK_CONSOLIDATION. Material that is already storyboarded can skip BOOK_STORYBOARD. Material that is already polished can skip BOOK_EDITORIAL. BOOK_TRIAGE determines the appropriate entry point by assessing the input folder's state.

All handoffs are manual. The human reviews each stage's output before triggering the next workflow. This matches the existing RDC→SDLC handoff pattern and ensures the author retains full creative control at every transition.

---

## 3. Architecture: Two Domains

The BOOK system has two distinct domains of artifacts:

### 3.1. Infrastructure (lives outside any individual book project)

Infrastructure artifacts are reusable across all book projects. They define *how* the workflows evaluate and process text. They are developed and maintained independently of any specific manuscript.

| Artifact | Description | Status |
|---|---|---|
| **TEMPLATE_STANDARD** | The schema defining what a voice/persona/style/prose template *is*. All individual templates conform to this schema. | Needs design |
| **Atomic templates** | Individual template files, one per axis (VOICE, PERSONA, STYLE, PROSE). Each defines one dimension of how a manuscript should read. | Needs identification + writing |
| **Bundle docs** | Hand-composed combinations of 4 atomics with synergy notes, interaction rules, and overrides specific to the combination. | Hand-composed by author |
| **Compatibility matrix** | Declares which atomic templates compose cleanly, which conflict, and which require a bundle to mediate. | Needs design |
| **LULU_SPEC** | Lulu.com's print requirements captured as a reference document (trim sizes, margins, bleed, gutter, spine calculations, PDF/X compliance, cover template dimensions). | Needs research |
| **LULU_PIPELINE** | Mechanical automation (programming, not AI) that consumes workflow output + LULU_SPEC and produces print-ready PDF. | Needs programming |

### 3.2. Workflows (the BOOK family)

Workflow artifacts are the state machines, worker role documents, and shared protocols that define how each stage operates. They reference infrastructure artifacts (templates, specs) but do not contain them.

| Workflow | Trigger phrase | Purpose |
|---|---|---|
| **BOOK_TRIAGE** | `BOOK_TRIAGE` | Classify input folder, produce BOOK_MANIFEST.json, recommend entry workflow |
| **BOOK_CONSOLIDATION** | `BOOK_CONSOLIDATION` | Consolidate chaotic folder of docs into structured manuscript |
| **BOOK_STORYBOARD** | `BOOK_STORYBOARD` | Produce pedagogical/logical skeleton for structured manuscript |
| **BOOK_EDITORIAL** | `BOOK_EDITORIAL` | Audit and revise manuscript against declared templates |
| **BOOK_PRODUCTION** | `BOOK_PRODUCTION` | Produce automation-ready intermediate for mechanical PDF generation |

---

## 4. Template Architecture

Templates are the central mechanism by which the BOOK system adapts to different genres, voices, and styles. They are not part of any single workflow — they are infrastructure that all editorial workflows reference.

### 4.1. The Four Axes

Every manuscript's written character is described along four independent-but-interacting axes:

**VOICE** — How the author relates to the reader. The pedagogical or narrative contract. Voice determines the fundamental posture: guide-alongside (Socratic), peer-explanation (Feynman), authoritative declaration (textbook), exploratory-companion (essay), etc.

Examples of what VOICE governs:
- Question frequency and placement (Socratic voice asks before answering; textbook voice states then illustrates)
- Use of "we" vs. "I" vs. passive construction
- How uncertainty is expressed (Socratic: "what might we expect?"; Feynman: "nobody knows, and here's why that's exciting"; textbook: "this remains an open question")
- Whether the reader is addressed directly
- How much the author "shows their work" vs. presenting conclusions
- The pedagogical contract: what the author promises the reader about how ideas will be presented

**PERSONA** — Who the author is performing as. Distinct from voice because the same persona can deploy different voices. Persona defines domain authority level, relationship to the material (discoverer vs. synthesizer vs. teacher vs. narrator), and what the author is allowed to "not know" on the page.

Examples of what PERSONA governs:
- Domain authority: does the author speak as an expert, a fellow learner, or a synthesizer of others' expertise?
- Relationship to uncertainty: does the persona have opinions, or does it present all sides?
- Emotional range: is the persona allowed to express excitement, frustration, wonder?
- Self-reference: does the persona share personal experience, or remain impersonal?

**STYLE** — Genre-level conventions. The structural and formal expectations that a reader of this type of book would hold. Style is the genre contract.

Examples of what STYLE governs:
- Citation conventions (inline, footnoted, endnoted, informal reference, none)
- Argument structure norms (thesis-evidence-conclusion, exploration-observation-synthesis, narrative, dialectic)
- Expected section patterns (chapters with abstracts, chapters with summaries, continuous narrative)
- Prose density expectations (how much white space, how long are paragraphs, how dense is information per page)
- Jargon calibration (defined on first use, assumed known, glossaried)
- Mathematical notation conventions (inline vs. display, proof style, equation numbering)

**PROSE** — Sentence-level craft rules. The mechanical texture of the writing at the paragraph and sentence level.

Examples of what PROSE governs:
- Sentence complexity range (simple declarative to multi-clause)
- Paragraph length norms (short punchy vs. sustained development)
- Vocabulary register (conversational, semi-formal, formal, technical)
- Use of metaphor and analogy (frequent, occasional, avoided)
- How technical terms are introduced (defined inline, defined in context, assumed)
- Parenthetical asides (permitted, frequent, avoided)
- Humor (permitted, frequent, avoided, genre-dependent)
- Rhythm and cadence preferences

### 4.2. Template Document Structure

Each atomic template conforms to TEMPLATE_STANDARD and has the following structure:

```
TEMPLATE: <name>
AXIS: VOICE | PERSONA | STYLE | PROSE
VERSION: <semver>
APPLIES_TO: <what genre/format/context this template is for>

## Contract
<3-5 sentences defining the core commitment of this template.
 This is the "if you follow nothing else, follow this" statement.>

## Characteristic Patterns
<What writing that follows this template looks like.
 Positive examples with brief annotation explaining
 why each example demonstrates the template.>

## Anti-Patterns
<What this template explicitly forbids.
 Negative examples with brief annotation explaining
 why each example violates the template.>

## Audit Checklist
<Specific, mechanically-checkable things an editorial worker
 verifies when auditing against this template.
 Each item should be answerable as pass/flag/not-applicable.>

## Interaction Notes
<How this template interacts with other template axes.
 E.g., "Socratic voice requires prose templates that permit
 questions in body text" or "this persona conflicts with
 formal-academic style because it demands self-reference.">
```

### 4.3. Layers

**Layer 1 — Atomic templates.** One file per axis, always present. Named `VOICE_<name>.md`, `PERSONA_<name>.md`, `STYLE_<name>.md`, `PROSE_<name>.md`. Each atomic template is independent and self-contained.

**Layer 2 — Bundle docs.** Optional. Hand-composed by the author. Named `BUNDLE_<name>.md`. A bundle declares a specific combination of 4 atomics plus any synergy notes, interaction rules, or overrides that exist only in this specific combination. Bundles are authored artifacts, not generated — the author composes them because certain combinations produce emergent properties that no individual atomic captures.

**Layer 3 — Compatibility matrix.** A single file (`TEMPLATE_COMPATIBILITY.md` or `.json`) that declares which atomic templates compose cleanly, which conflict, and which require a bundle doc to mediate the interaction. Used by BOOK_TRIAGE and BOOK_EDITORIAL to validate that the BOOK_MANIFEST's template declarations are coherent.

### 4.4. Template Resolution

BOOK_MANIFEST.json declares templates in one of two modes:

**Bundle mode:** `"template_bundle": "BUNDLE_SOCRATIC_ACADEMIC"` — the bundle file resolves all four axes plus interaction rules.

**Composition mode:** `"template_voice": "VOICE_SOCRATIC"`, `"template_persona": "PERSONA_PHYSICIST_TEACHER"`, `"template_style": "STYLE_ACADEMIC_EXPLORATORY"`, `"template_prose": "PROSE_MEDIUM_ACCESSIBLE"` — four atomics declared individually. BOOK_EDITORIAL checks this combination against the compatibility matrix before proceeding.

Workers load whatever the manifest points to. If a bundle is declared, workers load the bundle (which internally references its constituent atomics). If atomics are declared, workers load all four.

### 4.5. Initial Templates to Build

Based on Michael's existing body of work, the following templates are the minimum initial set:

**VOICE templates:**
- `VOICE_SOCRATIC` — the "show, compare, ask" discovery voice. Every chapter starts with observation, builds through exploration, arrives at insight. Never declares truth top-down. The reader is a bright student being guided to discover.
- `VOICE_FEYNMAN` — peer-explanation, concrete-before-abstract, "let me show you how I think about this." Humor permitted. Wonder encouraged. Complexity acknowledged but never used as a gatekeeping mechanism.
- Others TBD as works are catalogued.

**PERSONA templates:**
- `PERSONA_PHYSICIST_TEACHER` — domain authority as theoretical physicist. Speaks as a discoverer who has thought deeply and wants to share the thinking process, not just the conclusions.
- Others TBD.

**STYLE templates:**
- `STYLE_ACADEMIC_EXPLORATORY` — academic in rigor, exploratory in structure. Citations present but not dominant. Argument structure is dialectic (thesis-antithesis-synthesis) or inductive (observations-pattern-principle). Chapter structure follows natural conceptual boundaries.
- `STYLE_ACADEMIC_METASTUDY` — synthesizes across multiple domains. Heavy citation. Systematic coverage. Comparative analysis structure.
- Others TBD.

**PROSE templates:**
- `PROSE_MEDIUM_ACCESSIBLE` — sentences range from simple to moderately complex. Paragraphs develop a single idea. Technical terms defined in context. Metaphor and analogy used regularly. Parenthetical asides permitted sparingly.
- Others TBD.

**Bundle templates:**
- `BUNDLE_SPIN_OF_GRAVITY` — Socratic voice + physicist-teacher persona + academic-exploratory style + medium-accessible prose. Synergy note: the "we tell nothing — we show, compare, ask" rule is the emergent property of this specific combination.
- Others TBD.

---

## 5. BOOK_MANIFEST.json

The manifest is the per-project configuration file that travels with the manuscript through all workflow stages. It is produced by BOOK_TRIAGE, reviewed and edited by the human, and consumed by all downstream workflows.

### 5.1. Schema

```json
{
  "version": "1.0.0",
  "title": "<book title>",
  "author": "<author name>",
  "project_root": "<path to project root>",

  "triage": {
    "performed_at": "<ISO timestamp>",
    "file_count": "<N>",
    "quality_assessment": "<rough | structured | storyboarded | polished>",
    "recommended_entry": "<BOOK_CONSOLIDATION | BOOK_STORYBOARD | BOOK_EDITORIAL | BOOK_PRODUCTION>",
    "notes": "<triage observations>"
  },

  "templates": {
    "mode": "bundle | composition",
    "bundle": "<BUNDLE_<name> if mode=bundle>",
    "voice": "<VOICE_<name> if mode=composition>",
    "persona": "<PERSONA_<name> if mode=composition>",
    "style": "<STYLE_<name> if mode=composition>",
    "prose": "<PROSE_<name> if mode=composition>"
  },

  "structure": {
    "genre": "<academic-exploratory | academic-metastudy | hard-sci-fi | math-exposition | science-theory | etc.>",
    "target_audience": "<description of intended reader>",
    "estimated_chapters": "<N or null if unknown>",
    "front_matter": ["title_page", "copyright", "dedication", "toc", "preface"],
    "back_matter": ["bibliography", "index", "appendices", "glossary"],
    "special_elements": ["equations", "figures", "tables", "code_listings", "proofs"]
  },

  "production": {
    "target_format": "lulu | other",
    "trim_size": "<6x9 | 5.5x8.5 | etc.>",
    "color_interior": "<true | false>",
    "template_id": "<LULU template identifier if known>"
  }
}
```

### 5.2. Lifecycle

1. BOOK_TRIAGE produces the initial manifest with `triage` section filled and `templates`/`structure`/`production` sections partially filled or left as placeholders.
2. Human reviews the manifest, fills in template declarations, confirms or overrides triage recommendations, specifies production parameters.
3. Each downstream workflow reads the manifest at engagement and uses it to configure worker behavior.
4. The manifest is not modified by workflows after triage — it is a human-owned configuration document. Workflows read it; they do not write to it. (Exception: BOOK_PRODUCTION may fill in calculated production fields like spine width.)

---

## 6. BOOK_TRIAGE

### 6.1. Nature

BOOK_TRIAGE is not a full workflow with worker spawning and QA loops. It is a single-pass QUEEN-level operation that classifies an input folder and produces a BOOK_MANIFEST.json.

### 6.2. Trigger

`BOOK_TRIAGE`

### 6.3. Engagement

1. QUEEN reads the BOOK_TRIAGE workflow doc.
2. QUEEN reports: `"BOOK_TRIAGE mode engaged. Ready. Point me to the source folder."`
3. QUEEN waits for human to specify the folder path.

### 6.4. Execution

QUEEN performs the following:

**Step 1 — Inventory.** List all files in the source folder. Count files, identify types (`.md`, `.txt`, `.tex`, `.docx`, etc.), note any existing structural files (TOC, README, chapter ordering).

**Step 2 — Sampling.** Read the first ~100 lines and last ~50 lines of each file. Identify: file length, presence of headers/structure, prose quality (rough notes vs. drafted prose vs. polished writing), presence of TOC or chapter markers, presence of citations, presence of equations/figures/code.

**Step 3 — Classification.** Assess each file individually:
- **Fragment** — short, unstructured, note-like, incomplete thoughts
- **Draft** — has structure (headers, sections) but prose is rough or incomplete
- **Chapter** — structured, substantial prose, appears to be a complete chapter or major section
- **Polished** — structured, complete, prose is publication-quality or near it

**Step 4 — Aggregate assessment.** Determine the overall folder's state:
- **Rough** — majority fragments and drafts, no coherent chapter structure → recommend BOOK_CONSOLIDATION
- **Structured** — majority chapters, some coherent ordering exists → recommend BOOK_STORYBOARD
- **Storyboarded** — chapters exist with clear logical progression and pedagogical structure → recommend BOOK_EDITORIAL
- **Polished** — prose is complete and high quality → recommend BOOK_PRODUCTION

**Step 5 — Genre/voice detection.** Attempt to identify from content: what genre does this appear to be? What voice patterns are visible? What audience does it seem targeted at? These are suggestions — the human fills in the definitive values.

**Step 6 — Produce BOOK_MANIFEST.json.** Write the manifest to project root with triage section filled. Template and production sections contain best-guess suggestions or explicit placeholders for human review.

**Step 7 — Report.** Present the triage findings to the human: file count, per-file classification, aggregate assessment, recommended entry workflow, template suggestions. Wait for human to review, edit the manifest, and trigger the appropriate workflow.

### 6.5. Output

- `BOOK_MANIFEST.json` at project root

### 6.6. No QA Loop

TRIAGE has no QA loop, no verdicts, no loop limits. It is a one-shot classification. If the human disagrees with the assessment, they edit the manifest directly. There is no "re-triage" — just edit and proceed.

---

## 7. BOOK_CONSOLIDATION

### 7.1. Nature

BOOK_CONSOLIDATION is a full swarm workflow, forked from RDC_WORKFLOW. It transforms a chaotic folder of documents — fragments, drafts, notes, partial chapters of varying quality and epoch — into a structured manuscript with chapter organization and section hierarchy.

### 7.2. Relationship to RDC

BOOK_CONSOLIDATION uses the same core mechanics as RDC:
- Sequential temporal-order SCRIBE upsert into MASTER.md
- COURT mechanism for conflict resolution (4 advocates, 2 per side, QUEEN as judge)
- PEDAGOGY.md as archaeological record of concept evolution
- EVALUATIONS.md as per-document processing record
- INVENTORY.md as temporal manifest
- INPROGRESS.md as unified progress log

The critical difference is in the output shape. Where RDC's TAXONOMIST carves MASTER into engineering documents (PROJECT/ARCH/TODO), BOOK_CONSOLIDATION's **COMPOSITOR** carves MASTER into manuscript documents (chapters with section hierarchy).

### 7.3. Trigger

`BOOK_CONSOLIDATION`

### 7.4. Engagement

1. QUEEN reads the BOOK_CONSOLIDATION workflow JSON in full.
2. QUEEN reads `workflows/SHARED/WORKER_QUEEN.md`.
3. QUEEN reads `workflows/SHARED/WORKER_PROTOCOL.md`.
4. QUEEN reads `workflows/SHARED/WORKER.md`.
5. QUEEN reads `BOOK_MANIFEST.json` from project root.
6. QUEEN reports: `"BOOK_CONSOLIDATION mode engaged. Ready. Point me to the source directory."`
7. QUEEN waits for explicit human direction specifying the source directory path.

### 7.5. Roles

**QUEEN** — Orchestrator. Same responsibilities as RDC QUEEN: inventory, SCRIBE_LOOP orchestration, COURT adjudication, progress tracking, verdict execution, escalation. Not a spawned worker.

**SCRIBE** — Core upsert worker. Identical to RDC SCRIBE. Spawned once per source document in temporal order. Receives current MASTER + one source doc. Reads the source doc, upserts all concepts into MASTER. Logs changes to PEDAGOGY.md. Produces evaluation entry for EVALUATIONS.md. Flags contradictions for COURT.

Upsert rules (same as RDC):
- New concept not in MASTER → INSERT into structurally appropriate location
- Revised concept already in MASTER → OVERWRITE in place, log prior value to PEDAGOGY
- Identical concept → NO-OP, note as unchanged in evaluation
- Contradicting concept without clear temporal supersession → FLAG for COURT, insert both versions with conflict marker
- Never delete from MASTER, only overwrite. Deprecated concepts are marked deprecated, not removed.

**ADVOCATE** — Court advocate. Identical to RDC ADVOCATE. Given a conflicting concept and an assigned side, produces a brief arguing for that side. Spawned 4 per court session (2 per side). Advocates assigned side regardless of personal assessment.

**COMPOSITOR** — Replaces RDC's TAXONOMIST. This is the key role that differs from RDC. COMPOSITOR takes the final MASTER.md (after all SCRIBE passes and COURT resolutions) and carves it into a structured manuscript.

COMPOSITOR responsibilities:
- Discover chapter structure from MASTER content (chapters are discovered, not predetermined — same philosophy as RDC's phase discovery)
- Identify natural chapter boundaries based on conceptual coherence, scope, and logical grouping
- Produce ordered chapter files with section hierarchy within each chapter
- Produce `STRUCTURE.md` — the structural skeleton containing: table of contents, chapter summaries, section listings, and inter-chapter dependency map
- Ensure every concept from MASTER lands in exactly one chapter (no duplication, no loss)
- Preserve MASTER's concept relationships in the chapter assignments (related concepts should be in the same or adjacent chapters)

COMPOSITOR outputs:
- `chapters/CH_<NN>_<TITLE>.md` — one file per discovered chapter, with section structure
- `STRUCTURE.md` — the manuscript skeleton (TOC + chapter summaries + dependency map)

**QA_COMPLETENESS** — Concept-loss hunter. Identical in spirit to RDC's QA_COMPLETENESS. Reviews ALL source documents against the carved chapter files. Every concept that appeared in any source doc must be present in either (a) the chapter files, (b) PEDAGOGY.md as a superseded/deprecated entry, or (c) INPROGRESS.md as a court-resolved entry. Missing concepts are flagged.

**QA_COHERENCE** — Structural integrity auditor. Adapted from RDC's QA_COHERENCE for manuscript structure. Checks: do chapters follow a logical ordering? Are there orphaned concepts? Do sections belong in their chapters? Is STRUCTURE.md consistent with the actual chapter files? Is the manuscript structurally ready for storyboarding?

### 7.6. Pipeline

```
INVENTORY (QUEEN scans source directory, temporal ordering, human confirms)
    ↓
SCRIBE_LOOP (one SCRIBE per source doc, temporal order, sequential)
    ↓
COURT (conditional — only if conflicts flagged during SCRIBE_LOOP)
    4 ADVOCATEs per session (2 per side, parallel)
    QUEEN rules: SIDE_A | SIDE_B | SYNTHESIS | DEFER | REJECT_BOTH | NO_DECISION
    Recorded to INPROGRESS.md
    ↓
COMPOSITOR (carve MASTER → chapter files + STRUCTURE.md)
    ↓
QA_UNIT: QA_COMPLETENESS → QA_COHERENCE (sequential)
    ↓
QUEEN executes verdict
```

### 7.7. Verdicts

**GREEN_LIGHT** — QA_COMPLETENESS finds zero missing concepts AND QA_COHERENCE finds zero structural issues. QUEEN commits chapter files + STRUCTURE.md + PEDAGOGY.md + MASTER.md to project root. Reports: "BOOK_CONSOLIDATION complete. Structured manuscript ready for BOOK_STORYBOARD."

**SCRIBE_REVISIT** — QA_COMPLETENESS finds missing concepts. QUEEN re-spawns SCRIBE with current MASTER + flagged source docs + QA findings as focus directive. Targeted upsert. Then re-COMPOSITOR, then full re-QA. qa_cycle_counter++.

**RECOMPOSE** — QA_COMPLETENESS passes (no missing concepts) but QA_COHERENCE finds structural issues (wrong chapter assignments, ordering problems, dependency violations). QUEEN re-spawns COMPOSITOR with current MASTER + QA_COHERENCE findings as correction directive. Then full re-QA. qa_cycle_counter++.

**ESCALATE** — qa_cycle_counter >= 3 OR court session rules NO_DECISION/REJECT_BOTH/DEFER-to-human OR unresolvable conflict. QUEEN pauses, reports to human, waits.

### 7.8. Loop Limits

- qa_cycles: 3 (shared across SCRIBE_REVISIT and RECOMPOSE — 3 total attempts, not 3 per type)
- Court sessions per concept: no hard limit, but escalation on NO_DECISION/REJECT_BOTH/DEFER blocks progress

### 7.9. Files Produced

| File | Semantics |
|---|---|
| `chapters/CH_<NN>_<TITLE>.md` | One per discovered chapter. Ordered. Section hierarchy within each. |
| `STRUCTURE.md` | TOC + chapter summaries + dependency map. The manuscript skeleton. |
| `MASTER.md` | Consolidated single-file representation. Preserved for reference. |
| `PEDAGOGY.md` | Archaeological record of concept evolution during consolidation. |
| `EVALUATIONS.md` | Per-source-document processing record. |
| `INVENTORY.md` | Temporal manifest of source documents. |
| `INPROGRESS.md` | Progress log including COURT transcripts. |

### 7.10. Hard Rules

All RDC hard rules apply, plus:

1. COMPOSITOR discovers chapter structure from content — never predetermined
2. Every concept in MASTER must land in exactly one chapter — no duplication, no loss
3. STRUCTURE.md must be consistent with actual chapter files — no phantom entries, no missing chapters
4. Chapter files are the output, MASTER.md is preserved as reference but not consumed downstream
5. Chapter ordering in STRUCTURE.md reflects conceptual dependency, not source-doc ordering

---

## 8. BOOK_STORYBOARD

### 8.1. Nature

BOOK_STORYBOARD is a lighter workflow than BOOK_CONSOLIDATION. It takes the structured manuscript (ordered chapters with section hierarchy) and produces a pedagogical/logical skeleton — the storyboard. The storyboard is a planning document that describes what each chapter does, how chapters connect, what the reader knows at each chapter boundary, and what the argument arc of the full work looks like.

The storyboard is voice-neutral. It is written in direct academic descriptive prose — it describes what happens in the manuscript, not how it's said. Voice is applied later in BOOK_EDITORIAL. The storyboard is the structure on top of which voice is layered.

### 8.2. Why a Separate Workflow

The storyboard is constructive — it builds a new artifact (the logical skeleton). BOOK_EDITORIAL is evaluative and corrective — it audits and fixes. Mixing construction and evaluation muddies the QA semantics. The storyboard also serves as a reference document for BOOK_EDITORIAL's workers — they check the voiced manuscript against the storyboard to ensure voice application didn't break the logical structure.

Additionally, the storyboard is a natural human review point. The author should see and approve the logical structure before voice and editorial polish are applied. If the structure is wrong, fixing it after voice application wastes work.

### 8.3. Trigger

`BOOK_STORYBOARD`

### 8.4. Engagement

1. QUEEN reads the BOOK_STORYBOARD workflow JSON.
2. QUEEN reads shared worker docs.
3. QUEEN reads BOOK_MANIFEST.json.
4. QUEEN reads STRUCTURE.md (from BOOK_CONSOLIDATION output or human-authored).
5. QUEEN reports: `"BOOK_STORYBOARD mode engaged. Ready."`
6. QUEEN proceeds (no directory prompt needed — QUEEN reads from project root where chapter files and STRUCTURE.md live).

### 8.5. Roles

**QUEEN** — Orchestrator. Spawns STORYBOARDER, manages QA, executes verdicts.

**STORYBOARDER** — The constructive worker. Reads the entire manuscript (all chapter files + STRUCTURE.md) and produces the storyboard. This is a single worker, not a per-chapter loop, because the storyboard must capture inter-chapter relationships and the full-work arc.

STORYBOARDER produces for each chapter:
- **Opening state:** what the reader knows/believes entering this chapter
- **Key moves:** the 3-7 major conceptual steps the chapter takes (in academic-descriptive voice, not the manuscript's voice)
- **Closing state:** what the reader knows/believes exiting this chapter
- **Concepts introduced:** new terms, ideas, frameworks that appear for the first time
- **Concepts required:** prerequisites — things the reader must already understand
- **Chapter function:** what role this chapter plays in the full work's argument/exploration

STORYBOARDER produces for the full work:
- **Arc map:** how chapters build on each other, where the work's argument/exploration peaks, what the overall trajectory is
- **Prerequisite chain:** a directed graph of concept dependencies across chapters (concept X in chapter 5 requires concept Y from chapter 2)
- **Reader journey:** a narrative description of what the reader experiences across the full work — not what they read, but what they *understand* at each stage

STORYBOARDER output: `STORYBOARD.md`

**QA_STORYBOARD** — Single QA worker (not a multi-pass unit — the storyboard is a planning document, not a manuscript, so lighter QA is appropriate).

QA_STORYBOARD checks:
- **Prerequisite satisfaction:** for every concept marked as "required" in a chapter, is that concept "introduced" in an earlier chapter? No forward dependencies.
- **Progressive arc:** does the work build progressively? No chapter is an island — every chapter connects to what came before and what comes after.
- **Completeness:** does the storyboard cover every chapter? Does every section in STRUCTURE.md have representation in the storyboard?
- **Consistency:** does the storyboard match the actual chapter content? (Spot-check: STORYBOARDER's description of what chapter N does should match what chapter N actually contains.)
- **Genre/style alignment:** does the storyboard's described structure match what the BOOK_MANIFEST's declared genre/style expects? (An academic-exploratory book should have a discovery arc, not a textbook-declarative structure.)

### 8.6. Pipeline

```
QUEEN reads chapter files + STRUCTURE.md + BOOK_MANIFEST.json
    ↓
STORYBOARDER (single worker, reads full manuscript, produces STORYBOARD.md)
    ↓
QA_STORYBOARD (single QA worker)
    ↓
Verdict: GREEN_LIGHT / REVISE / ESCALATE
```

### 8.7. Verdicts

**GREEN_LIGHT** — QA_STORYBOARD finds no issues. QUEEN commits STORYBOARD.md. Reports: "BOOK_STORYBOARD complete. Storyboard ready for human review. After review, trigger BOOK_EDITORIAL."

**REVISE** — QA_STORYBOARD finds issues (broken prerequisites, missing coverage, structural problems). QUEEN re-spawns STORYBOARDER with QA findings as correction directive. Then re-QA. revision_counter++.

**ESCALATE** — revision_counter >= 3 OR QA finds issues that suggest the chapter structure itself is wrong (not just the storyboard's description of it — the actual chapters need restructuring, which means going back to BOOK_CONSOLIDATION). QUEEN pauses, reports to human, waits.

### 8.8. Loop Limits

- revision_cycles: 3

### 8.9. Files Produced

| File | Semantics |
|---|---|
| `STORYBOARD.md` | Full logical skeleton: per-chapter storyboard entries + full-work arc map + prerequisite chain + reader journey |
| `INPROGRESS.md` | Updated with storyboard progress entries |

### 8.10. Hard Rules

1. STORYBOARD.md is voice-neutral — written in direct academic descriptive prose, never in the manuscript's declared voice
2. Every chapter in STRUCTURE.md must have a corresponding storyboard entry
3. No forward dependencies in the prerequisite chain (concept must be introduced before it is required)
4. STORYBOARDER reads the full manuscript, not summaries — the storyboard must accurately describe what the chapters actually contain
5. The storyboard is a planning document, not prose — it describes structure, not style

---

## 9. BOOK_EDITORIAL

### 9.1. Nature

BOOK_EDITORIAL is the most complex workflow in the BOOK family. It takes a storyboarded manuscript and audits it against the declared templates (voice, persona, style, prose) for consistency, correctness, and quality. When issues are found, a REVISION worker patches the flagged passages. The result is a polished manuscript ready for production.

This is where voice is applied, audited, and corrected. The storyboard (voice-neutral structural skeleton) serves as the reference for whether the voiced manuscript maintains its logical structure. The templates serve as the reference for whether the voice was applied correctly and consistently.

### 9.2. Template Loading

At engagement, QUEEN reads BOOK_MANIFEST.json and resolves the template declaration:

- If `templates.mode == "bundle"`: QUEEN loads the named bundle doc, which internally references its constituent atomics.
- If `templates.mode == "composition"`: QUEEN loads all four atomic templates and checks the combination against the compatibility matrix. If an incompatibility is detected, QUEEN reports the conflict and waits for human resolution before proceeding.

All editorial workers receive the resolved template set as part of their context packet.

### 9.3. Trigger

`BOOK_EDITORIAL`

### 9.4. Engagement

1. QUEEN reads the BOOK_EDITORIAL workflow JSON.
2. QUEEN reads shared worker docs.
3. QUEEN reads BOOK_MANIFEST.json and resolves templates.
4. QUEEN reads STORYBOARD.md.
5. QUEEN validates template compatibility (if composition mode).
6. QUEEN reports: `"BOOK_EDITORIAL mode engaged. Templates loaded: [list]. Ready."`
7. QUEEN proceeds to spawn editorial workers.

### 9.5. Roles

**QUEEN** — Orchestrator. Spawns all editorial workers, manages the junior→synthesis→senior pipeline, spawns REVISION worker on REVISE verdict, manages QA loop, executes verdicts.

**JUNIOR_VOICE** — Audits the manuscript against the declared VOICE template. Checks every chapter for adherence to the voice contract. Stance: hypercritical, adversarial, high-recall — bias strongly toward flagging. It is better to flag a passage that turns out to be fine than to miss a genuine voice break.

JUNIOR_VOICE checks:
- Pedagogical contract adherence (e.g., for Socratic voice: does the text show before telling? ask before answering?)
- Narrator posture consistency (does the voice maintain its relationship to the reader throughout?)
- "We"/"I"/passive usage patterns match the voice template
- Uncertainty expression matches the voice template
- Question frequency and placement match the voice template
- No code-switching into a different voice mid-chapter without structural justification

Output: findings list with severity labels (Critical/High/Medium/Low) per the voice template's audit checklist, with specific passage references (chapter, section, paragraph).

**JUNIOR_CONCEPT** — Audits concept consistency across the entire manuscript. Not tied to any single template — this is a content-level check that applies regardless of voice/style.

JUNIOR_CONCEPT checks:
- Term consistency: is the same concept called the same thing throughout? (No "spin-drag" in chapter 3 becoming "rotational friction" in chapter 7 without explanation)
- Definition-before-use: every technical term is introduced/defined before it is used as assumed knowledge
- No contradictions: concept descriptions in chapter N don't contradict descriptions in chapter M
- Concept completeness: concepts referenced in later chapters were actually established in earlier chapters (cross-reference against STORYBOARD.md prerequisite chain)
- Notation consistency: mathematical notation, variable naming, unit conventions stable throughout

Output: findings list with passage references, keyed to specific concepts.

**JUNIOR_STYLE** — Audits the manuscript against the declared STYLE template. Checks genre-level conventions.

JUNIOR_STYLE checks:
- Citation conventions match the style template
- Argument structure matches the style template's expected patterns
- Section/chapter structure follows genre norms
- Prose density is within the style template's expected range
- Jargon handling matches the style template
- Mathematical/technical presentation follows the style template's conventions

Output: findings list with passage references.

**JUNIOR_FLOW** — Audits logical progression, transitions, and arc adherence. Cross-references against STORYBOARD.md.

JUNIOR_FLOW checks:
- Chapter transitions: does each chapter's opening connect to the previous chapter's closing?
- Argument arc: does the manuscript follow the arc described in STORYBOARD.md?
- Redundancy: are any concepts explained multiple times without justification?
- Pacing: are any chapters disproportionately dense or sparse relative to their role in the arc?
- Reader journey: at each chapter boundary, does the reader have what they need for the next chapter?

Output: findings list with passage and storyboard references.

**EDITORIAL_SYNTHESIS** — Integration worker. Reads all 4 junior reports and finds cross-axis interactions that no individual junior could detect.

EDITORIAL_SYNTHESIS looks for:
- Voice-style conflicts: voice is consistent but doesn't match the genre's style expectations
- Concept-flow conflicts: concepts are individually consistent but introduced in an order that breaks the pedagogical flow
- Prose-voice conflicts: prose register shifts in places where the voice doesn't call for it
- Style-flow conflicts: genre conventions are followed locally but the overall arc doesn't match genre expectations
- Emergent patterns: issues that appear minor on one axis but compound across axes into a significant problem

Output: integrated findings report. Each finding is tagged with which axes interact and why no individual junior would catch it. Includes all junior findings (passed through) plus new cross-axis findings.

**SENIOR_SANITY** — Precision filter. Reads the integrated findings report. For each finding: marks it `real` (valid issue, must address) or `overzealous` (false positive, drop) with rationale. Does not produce new findings — only rules on existing ones. Same role as SDLC's SENIOR_QA_SANITY.

Output: filtered findings list, each marked real|overzealous with 1-line rationale.

**SENIOR_FINAL** — Last word. Performs an independent pass over the full manuscript plus all prior reports. May surface new findings that the entire junior+synthesis pipeline missed. Emits the binding verdict. Same authority model as SDLC's SENIOR_QA_FINAL — QUEEN does not override this verdict.

Output:
- Final verdict: GREEN_LIGHT | REVISE | ESCALATE
- Verdict rationale
- If REVISE: consolidated actionable findings for REVISION worker (filtered to only `real` findings, with specific passage references and correction guidance)
- If ESCALATE: blocker description for human

**REVISION** — The prose writer. Receives the filtered findings + full manuscript + all templates + STORYBOARD.md. Rewrites only the flagged passages. Does not touch unflagged text. Scope discipline is critical — REVISION is surgical, not wholesale.

REVISION responsibilities:
- For each flagged passage: rewrite to address the finding while maintaining consistency with unflagged surrounding text
- Maintain template adherence in all rewrites (voice, persona, style, prose)
- Maintain storyboard adherence in all rewrites (don't break the logical structure while fixing voice/style)
- Maintain concept consistency (don't introduce new terminology or contradict existing definitions while rewriting)
- Report what was changed and why for each passage

REVISION is the highest-skill worker in the BOOK family. It must simultaneously satisfy template constraints, storyboard constraints, concept constraints, and local prose quality — all while making surgical edits that don't destabilize surrounding text.

Output: revised chapter files (only the files containing flagged passages), with a revision report listing every change made.

### 9.6. Pipeline

```
QUEEN loads templates from manifest, validates compatibility
    ↓
JUNIOR_EDITORIAL (4 parallel sub-workers):
    JUNIOR_VOICE     — audits against VOICE template
    JUNIOR_CONCEPT   — audits concept consistency
    JUNIOR_STYLE     — audits against STYLE template
    JUNIOR_FLOW      — audits logical flow against STORYBOARD.md
    ↓
EDITORIAL_SYNTHESIS (cross-axis interaction detection)
    ↓
SENIOR_SANITY (filter: real vs. overzealous)
    ↓
SENIOR_FINAL (independent pass + binding verdict)
    ↓
Verdict:
    GREEN_LIGHT → polished manuscript committed
    REVISE      → REVISION worker patches flagged passages
                   → full re-QA from JUNIOR through SENIOR_FINAL
                   → qa_cycle_counter++
    ESCALATE    → pause, report to human
```

### 9.7. Verdicts

**GREEN_LIGHT** — SENIOR_FINAL finds no blocking issues. QUEEN commits polished chapter files. Reports: "BOOK_EDITORIAL complete. Polished manuscript ready for BOOK_PRODUCTION."

**REVISE** — SENIOR_FINAL identifies fixable issues. REVISION worker patches flagged passages. Full re-QA runs (all 4 juniors + synthesis + sanity + final). qa_cycle_counter++. If qa_cycle_counter == 3 and not GREEN_LIGHT: auto-ESCALATE.

**ESCALATE** — qa_cycle_counter >= 3 OR SENIOR_FINAL determines issues require human judgment (e.g., a voice break that might be intentional authorial choice, or a concept inconsistency that reflects genuine ambiguity in the underlying material). QUEEN pauses, reports to human with full findings history, waits.

### 9.8. Loop Limits

- qa_cycles: 3 (each REVISE → re-QA cycle increments the counter)

### 9.9. Files Produced/Modified

| File | Semantics |
|---|---|
| `chapters/CH_<NN>_<TITLE>.md` | Modified in place by REVISION worker. Only flagged files change. |
| `INPROGRESS.md` | Updated with editorial progress: junior reports, synthesis report, senior verdicts, revision reports |

### 9.10. Hard Rules

1. JUNIOR workers are parallel — they do not see each other's findings
2. EDITORIAL_SYNTHESIS reads all junior findings — it is the only worker that sees cross-axis interactions
3. SENIOR_SANITY does not produce new findings — it only rules on integrated findings
4. SENIOR_FINAL's verdict is binding — QUEEN does not override
5. REVISION touches only flagged passages — unflagged text is immutable
6. REVISION must maintain template + storyboard + concept consistency simultaneously
7. Every REVISE loop re-enters the full editorial pipeline from JUNIOR — no shortcuts
8. Template compatibility is validated before any workers spawn — incompatible compositions block engagement
9. No GREEN_LIGHT without full editorial pipeline (JUNIOR → SYNTHESIS → SANITY → FINAL)
10. If template resolution fails (missing template file, unknown template name), QUEEN escalates immediately — does not proceed with partial templates

---

## 10. BOOK_PRODUCTION

### 10.1. Nature

BOOK_PRODUCTION is the most mechanical workflow. It takes a polished manuscript and produces an automation-ready intermediate — a structured set of files and a configuration JSON that a mechanical pipeline (not AI) consumes to produce the final print-ready PDF.

The AI's job in this workflow is NOT to produce PDFs. It is to:
- Validate the manuscript is complete and well-formed
- Generate or validate front matter and back matter
- Produce a BOOK_SPEC.json that fully specifies the physical book's parameters
- Ensure the output set is ready for consumption by LULU_PIPELINE (or another mechanical pipeline)

### 10.2. Trigger

`BOOK_PRODUCTION`

### 10.3. Engagement

1. QUEEN reads the BOOK_PRODUCTION workflow JSON.
2. QUEEN reads shared worker docs.
3. QUEEN reads BOOK_MANIFEST.json (especially the `production` section).
4. QUEEN reads LULU_SPEC (if target is Lulu).
5. QUEEN reports: `"BOOK_PRODUCTION mode engaged. Target: [format]. Ready."`
6. QUEEN proceeds.

### 10.4. Roles

**QUEEN** — Orchestrator.

**FORMATTER** — The constructive worker. Reads the polished manuscript and produces the automation-ready intermediate.

FORMATTER responsibilities:
- Validate manuscript completeness: all chapters present, all sections populated, no placeholder text, no TODO markers
- Generate front matter files (or validate existing ones): title page, copyright page, dedication, table of contents, preface, acknowledgments — as declared in BOOK_MANIFEST
- Generate back matter files (or validate existing ones): bibliography, index, appendices, glossary — as declared in BOOK_MANIFEST
- Produce `BOOK_SPEC.json`: the complete physical specification of the book

BOOK_SPEC.json includes:
```json
{
  "title": "<title>",
  "author": "<author>",
  "trim_size": { "width_inches": 6, "height_inches": 9 },
  "margins": {
    "top_inches": 0.75,
    "bottom_inches": 0.75,
    "inside_inches": 0.875,
    "outside_inches": 0.625
  },
  "gutter": "<calculated from trim size>",
  "bleed": "<per LULU_SPEC>",
  "spine_width_inches": "<calculated from page count>",
  "fonts": {
    "body": "<font family + size>",
    "heading": "<font family + size>",
    "caption": "<font family + size>"
  },
  "page_numbering": {
    "front_matter": "roman_lowercase",
    "body": "arabic",
    "start_page": "<recto/verso>"
  },
  "chapter_breaks": "recto | next_page",
  "header_footer_spec": { "...": "..." },
  "color_interior": false,
  "pdf_standard": "PDF/X-1a:2001",
  "template_id": "<reference to production template if applicable>",
  "file_manifest": [
    { "type": "front_matter", "file": "front/title.md", "order": 1 },
    { "type": "front_matter", "file": "front/copyright.md", "order": 2 },
    { "type": "front_matter", "file": "front/toc.md", "order": 3 },
    { "type": "chapter", "file": "chapters/CH_01_TITLE.md", "order": 4 },
    "..."
  ]
}
```

**QA_PRODUCTION** — Single QA worker.

QA_PRODUCTION checks:
- All files in BOOK_SPEC.json file_manifest actually exist
- BOOK_SPEC.json values are valid against LULU_SPEC (trim size supported, margins within bounds, spine width correctly calculated)
- Front matter is complete per BOOK_MANIFEST declarations
- Back matter is complete per BOOK_MANIFEST declarations
- No placeholder text, no TODO markers, no incomplete sections in any file
- Chapter files are well-formed (consistent heading levels, no broken markup)
- File ordering in manifest matches STRUCTURE.md

### 10.5. Pipeline

```
QUEEN reads polished manuscript + BOOK_MANIFEST + LULU_SPEC
    ↓
FORMATTER (validates manuscript, produces front/back matter, produces BOOK_SPEC.json)
    ↓
QA_PRODUCTION (validates output against LULU_SPEC and completeness)
    ↓
Verdict: GREEN_LIGHT / FIX / ESCALATE
```

### 10.6. Verdicts

**GREEN_LIGHT** — QA_PRODUCTION finds no issues. QUEEN commits all output files. Reports: "BOOK_PRODUCTION complete. Automation-ready output in [directory]. Run LULU_PIPELINE to produce print-ready PDF."

**FIX** — QA_PRODUCTION finds fixable issues (missing front matter file, incorrect spine width calculation, etc.). QUEEN re-spawns FORMATTER with QA findings. Then re-QA. fix_counter++. If fix_counter == 3 and not GREEN_LIGHT: auto-ESCALATE.

**ESCALATE** — fix_counter >= 3 OR QA finds issues that require human judgment (e.g., LULU_SPEC has changed and BOOK_SPEC values no longer valid, or the manuscript structure doesn't fit any supported trim size). QUEEN pauses, reports to human.

### 10.7. Loop Limits

- fix_cycles: 3

### 10.8. Files Produced

| File | Semantics |
|---|---|
| `BOOK_SPEC.json` | Complete physical specification for mechanical pipeline |
| `front/` | Front matter files (title page, copyright, TOC, etc.) |
| `back/` | Back matter files (bibliography, index, etc.) |
| `chapters/` | Chapter files (passed through from BOOK_EDITORIAL, validated) |
| `INPROGRESS.md` | Updated with production progress |

### 10.9. Hard Rules

1. FORMATTER does not modify chapter prose — it only validates and produces supporting files
2. BOOK_SPEC.json must be valid against LULU_SPEC (or the declared target spec) — no values outside supported ranges
3. Every file referenced in BOOK_SPEC.json file_manifest must exist
4. Spine width is calculated, not estimated — formula is page_count × paper_thickness_per_page (from LULU_SPEC)
5. PDF standard compliance is declared, not verified — verification is LULU_PIPELINE's job
6. No chapter content modification at this stage — if prose issues are found, ESCALATE back to BOOK_EDITORIAL

---

## 11. Cross-Workflow Integration

### 11.1. Pipeline Flow

```
BOOK_TRIAGE
    ↓ produces: BOOK_MANIFEST.json
    ↓ human reviews, fills template declarations
    ↓
BOOK_CONSOLIDATION (if needed)
    ↓ produces: chapters/, STRUCTURE.md, MASTER.md, PEDAGOGY.md
    ↓ human reviews chapter structure
    ↓
BOOK_STORYBOARD (if needed)
    ↓ produces: STORYBOARD.md
    ↓ human reviews storyboard
    ↓
BOOK_EDITORIAL
    ↓ modifies: chapters/ (surgical revisions only)
    ↓ human reviews polished manuscript
    ↓
BOOK_PRODUCTION
    ↓ produces: BOOK_SPEC.json, front/, back/
    ↓ output is automation-ready
    ↓
[LULU_PIPELINE] (mechanical, not AI)
    ↓ produces: print-ready PDF
```

### 11.2. Handoff Protocol

Every handoff between workflows is manual. The human reviews the output of one workflow before triggering the next. This ensures:
- The author retains full creative control at every stage
- Workflow errors are caught at boundaries, not compounded across stages
- The author can intervene (edit chapters, adjust the storyboard, change template declarations) between stages

No workflow auto-triggers another workflow. Typing `BOOK_CONSOLIDATION` after `BOOK_TRIAGE` is an explicit human action.

### 11.3. Shared Artifacts

| Artifact | Produced by | Consumed by |
|---|---|---|
| `BOOK_MANIFEST.json` | BOOK_TRIAGE | All downstream workflows |
| `chapters/` | BOOK_CONSOLIDATION | BOOK_STORYBOARD, BOOK_EDITORIAL, BOOK_PRODUCTION |
| `STRUCTURE.md` | BOOK_CONSOLIDATION | BOOK_STORYBOARD, BOOK_EDITORIAL |
| `STORYBOARD.md` | BOOK_STORYBOARD | BOOK_EDITORIAL (JUNIOR_FLOW, REVISION) |
| `MASTER.md` | BOOK_CONSOLIDATION | Reference only (not consumed by downstream workflows) |
| `PEDAGOGY.md` | BOOK_CONSOLIDATION | Reference only |
| `BOOK_SPEC.json` | BOOK_PRODUCTION | LULU_PIPELINE |
| `INPROGRESS.md` | All workflows | All workflows (append-only shared log) |
| Template files | Infrastructure | BOOK_EDITORIAL |
| LULU_SPEC | Infrastructure | BOOK_PRODUCTION |

### 11.4. Entry Point Flexibility

Not every manuscript needs every stage:

| Input state | Entry point | Skips |
|---|---|---|
| Chaotic folder of fragments | BOOK_CONSOLIDATION | — |
| Already-structured chapters | BOOK_STORYBOARD | Consolidation |
| Already-storyboarded manuscript | BOOK_EDITORIAL | Consolidation, Storyboard |
| Already-polished manuscript | BOOK_PRODUCTION | Consolidation, Storyboard, Editorial |

BOOK_TRIAGE determines the appropriate entry point. The human can override.

When entering at a later stage, the human must ensure prerequisite artifacts exist:
- BOOK_STORYBOARD requires: chapter files + STRUCTURE.md
- BOOK_EDITORIAL requires: chapter files + STRUCTURE.md + STORYBOARD.md + template declarations in BOOK_MANIFEST
- BOOK_PRODUCTION requires: polished chapter files + BOOK_MANIFEST with production section filled

---

## 12. Relationship to Existing Workflows

### 12.1. Shared Infrastructure

BOOK workflows share the same swarm infrastructure as SDLC and RDC:
- `workflows/SHARED/WORKER_QUEEN.md` — QUEEN orchestrator role (workflow-agnostic)
- `workflows/SHARED/WORKER_PROTOCOL.md` — non-negotiable worker contract
- `workflows/SHARED/WORKER.md` — master role index
- `INPROGRESS.md` — unified progress log (shared across all active workflows)

### 12.2. What's Forked from RDC

BOOK_CONSOLIDATION is a direct fork of RDC_WORKFLOW with the following changes:
- COMPOSITOR replaces TAXONOMIST (different output shape — chapters instead of PHASE/ARCH/TODO)
- QA_COHERENCE checks manuscript structure instead of engineering structure
- Output files are chapter files + STRUCTURE.md instead of PROJECT/PHASE_ARCH/PHASE_TODO
- RECOMPOSE verdict replaces RETAXONOMIZE

Everything else — SCRIBE, COURT, ADVOCATE, PEDAGOGY, EVALUATIONS, INVENTORY, upsert rules, hard rules — ports directly.

### 12.3. What's Novel

- Template architecture (4 axes + bundles + compatibility matrix)
- BOOK_MANIFEST.json as per-project configuration
- BOOK_TRIAGE as pre-check routing
- BOOK_STORYBOARD as constructive planning workflow
- BOOK_EDITORIAL's junior-parallel → synthesis → senior pipeline
- EDITORIAL_SYNTHESIS as cross-axis interaction detector
- REVISION worker as prose-writing worker under template constraints
- BOOK_PRODUCTION's separation of AI validation from mechanical PDF generation
- BOOK_SPEC.json as automation-ready intermediate format

### 12.4. No Cross-Contamination

BOOK workflows and SDLC/RDC workflows do not mix. If BOOK_CONSOLIDATION is active, SDLC operations don't run, and vice versa. They share infrastructure and INPROGRESS.md but operate independently.

---

## 13. Hard Rules (Cross-Workflow)

These rules apply whenever any BOOK workflow is active. They override any conflicting defaults.

1. **Never fabricate text.** Every passage in a report must reference actual manuscript content. Quote the relevant passage.
2. **Never skip editorial QA.** No GREEN_LIGHT without full JUNIOR → SYNTHESIS → SANITY → FINAL pipeline in BOOK_EDITORIAL.
3. **Never override SENIOR_FINAL.** QUEEN trusts the binding verdict. SENIOR_FINAL is the authority.
4. **REVISION is surgical.** Only flagged passages are modified. Unflagged text is immutable during revision.
5. **Templates are authoritative.** If the manifest declares a template, workers enforce it. Workers do not substitute their own aesthetic preferences for template specifications.
6. **STORYBOARD.md is voice-neutral.** Written in direct academic descriptive prose. Never in the manuscript's declared voice.
7. **INPROGRESS.md is prepend-only except checkbox toggles.** Same rule as SDLC/RDC.
8. **BOOK_MANIFEST.json is human-owned.** Workflows read it; they do not modify it (with the narrow exception of BOOK_PRODUCTION filling calculated physical parameters).
9. **Workers have no conversation history.** Context packets must be self-contained.
10. **Scope is authoritative.** Each worker has a defined scope. Stay in it.
11. **Stop and raise on loop limit.** Don't silently continue past 3 cycles.
12. **Manual handoff between workflows.** No workflow auto-triggers another.
13. **Chapter structure is discovered, not predetermined** (in BOOK_CONSOLIDATION — same as RDC's phase discovery principle).
14. **Template compatibility must be validated before editorial workers spawn.** Incompatible compositions are a blocking error.

---

## 14. File Layout

After all workflows have run, a completed book project's root looks like:

```
project-root/
├── BOOK_MANIFEST.json          ← per-project config (TRIAGE-generated, human-reviewed)
├── BOOK_SPEC.json              ← physical book spec (PRODUCTION output)
├── STRUCTURE.md                ← manuscript skeleton (CONSOLIDATION output)
├── STORYBOARD.md               ← logical skeleton (STORYBOARD output)
├── MASTER.md                   ← consolidated single-file reference (CONSOLIDATION output, preserved)
├── PEDAGOGY.md                 ← concept evolution record (CONSOLIDATION output)
├── EVALUATIONS.md              ← per-source-doc processing record (CONSOLIDATION output)
├── INVENTORY.md                ← temporal manifest of source docs (CONSOLIDATION output)
├── INPROGRESS.md               ← unified progress log (all workflows)
│
├── source/                     ← original source documents (input to CONSOLIDATION)
│   ├── rough_notes_2024.md
│   ├── chapter_draft_v2.md
│   └── ...
│
├── chapters/                   ← manuscript chapters (CONSOLIDATION→EDITORIAL→PRODUCTION)
│   ├── CH_01_<TITLE>.md
│   ├── CH_02_<TITLE>.md
│   └── ...
│
├── front/                      ← front matter (PRODUCTION output)
│   ├── title.md
│   ├── copyright.md
│   ├── dedication.md
│   ├── toc.md
│   └── preface.md
│
├── back/                       ← back matter (PRODUCTION output)
│   ├── bibliography.md
│   ├── index.md
│   ├── glossary.md
│   └── appendix_A.md
│
└── output/                     ← mechanical pipeline output (LULU_PIPELINE)
    └── <title>_print_ready.pdf
```

---

## 15. Implementation Roadmap

### Phase 1 — Foundation
1. Design TEMPLATE_STANDARD (the schema all templates conform to)
2. Write first atomic templates: VOICE_SOCRATIC, VOICE_FEYNMAN, PERSONA_PHYSICIST_TEACHER, STYLE_ACADEMIC_EXPLORATORY, PROSE_MEDIUM_ACCESSIBLE
3. Write first bundle: BUNDLE_SPIN_OF_GRAVITY
4. Write BOOK_TRIAGE workflow JSON + worker docs
5. Write BOOK_MANIFEST.json schema

### Phase 2 — Consolidation + Storyboard
6. Fork RDC_WORKFLOW.json → BOOK_CONSOLIDATION workflow JSON
7. Write COMPOSITOR worker doc (replaces TAXONOMIST)
8. Adapt QA_COMPLETENESS and QA_COHERENCE for manuscript context
9. Write BOOK_STORYBOARD workflow JSON
10. Write STORYBOARDER and QA_STORYBOARD worker docs

### Phase 3 — Editorial
11. Write BOOK_EDITORIAL workflow JSON
12. Write JUNIOR_VOICE, JUNIOR_CONCEPT, JUNIOR_STYLE, JUNIOR_FLOW worker docs
13. Write EDITORIAL_SYNTHESIS worker doc
14. Write SENIOR_SANITY and SENIOR_FINAL worker docs (adapted from SDLC equivalents)
15. Write REVISION worker doc
16. Write compatibility matrix

### Phase 4 — Production
17. Research LULU_SPEC (capture Lulu.com requirements)
18. Write BOOK_PRODUCTION workflow JSON
19. Write FORMATTER and QA_PRODUCTION worker docs
20. Design BOOK_SPEC.json schema
21. Build LULU_PIPELINE (mechanical automation — programming, not workflow)

### Phase 5 — Extension
22. Additional atomic templates as new works are catalogued
23. Additional bundles for established combinations
24. Additional production targets beyond Lulu (if needed)
25. Refactoring pass: compare BOOK_CONSOLIDATION with RDC for potential shared-code extraction

---

## 16. Open Questions — Resolved

These questions were deferred at v0.1.0-DRAFT. All are now resolved.

1. **CI/CD hooks for BOOK workflows?** — **RESOLVED in T9.11.** Decision: add `workflows/ci-prose.sh` with markdown lint + spell check + template-conformance spot check. Hook fires on commits touching `chapters/`, `front/`, `back/`, `templates/`. Prose-level CI fires at pre-commit (quick mode: staged files only) and pre-push (gate mode: full pipeline). See `workflows/BOOK/OPEN_QUESTIONS_RESOLVED.md` for details (produced in Part 9-C).

2. **COMPOSITOR vs. TAXONOMIST refactoring.** — **RESOLVED in T9.12 (deferred).** Decision: defer until after first real BOOK project completes. Empirical comparison (`workflows/BOOK/COMPOSITOR_VS_TAXONOMIST.md`) documents structural parallels and divergences. The two roles share enough mechanics to warrant eventual shared-base extraction, but premature refactoring would couple BOOK_CONSOLIDATION's stability to RDC's. Revisit post-first-book.

3. **REVISION worker capability boundaries.** — **RESOLVED in T7.4 / WORKER_REVISION.md.** Decision: soft cap at 20 passages per REVISE cycle. If findings exceed 20, REVISION addresses highest-severity first and ESCALATEs rather than exceeding the cap. Rationale: surgical discipline degrades beyond ~20 simultaneous edits; bulk rewriting is not REVISION's role. For drafter-origin chapters, passage-scale edits are permitted (vs. sentence-scale for normal revision) — this is an explicit documented exception, not an override of the surgical principle.

4. **Multi-language manuscripts.** — **RESOLVED in T9.14.** Decision: language is handled as a sub-axis within PROSE templates (e.g., `PROSE_MEDIUM_ACCESSIBLE_KR.md` for Korean-language prose). No 5th axis is introduced. The PROSE template carries language register, script conventions, and any language-specific prose rules. BOOK_MANIFEST references the appropriate PROSE template; all editorial workers load it with the rest of the template set.

5. **Figure/equation/code integration.** — **RESOLVED in T9.15.** Decision: inline in markdown using standard syntax — LaTeX delimiters for equations (`$...$` inline, `$$...$$` display), fenced code blocks for listings, markdown image syntax for figures (`![caption](path)`). `BOOK_SPEC.json` carries a `special_elements` flag array that drives typesetting engine behavior in LULU_PIPELINE. No external-file reference scheme is introduced at this stage.

6. **Index generation.** — **RESOLVED in T9.16.** Decision: FORMATTER produces a preliminary index by key-term extraction from chapter files. The preliminary index is a human-review artifact — the author reviews and refines before BOOK_PRODUCTION GREEN_LIGHT. The mechanical LULU_PIPELINE does not modify the index. A dedicated INDEX worker is deferred (Part 5 extension).

7. **Versioning strategy for templates.** — **RESOLVED in T1.5 / TEMPLATE_STANDARD.md.** Decision: SemVer per template file (MAJOR.MINOR.PATCH in frontmatter). MAJOR changes require manifest pinning (BOOK_MANIFEST.json records the version used). MINOR and PATCH are compatible within a manifest. Templates are resolved at workflow engagement from the declared name; version pinning is optional but recommended for published projects.

---

## 17. Implementation Learnings

Observations from the buildout worth capturing for future workflow family design:

1. **Stance 3 with safeguards worked — drafter-origin flagging is load-bearing.** The decision to have DRAFTER produce full prose under template constraint (Stance 3) rather than skeletal output (Stance 2) was correct: it produces immediately useful chapter content. The critical safeguard is the `drafter_origin: true` frontmatter flag — without it, EDITORIAL workers have no mechanism to apply appropriate extra scrutiny to AI-authored prose vs. human-authored prose. The flag is not cosmetic; it is a data contract that flows through STORYBOARD, EDITORIAL, and REVISION.

2. **Compatibility matrix surfaced a real conflict that templates alone couldn't predict.** During buildout, the STYLE_ACADEMIC_EXPLORATORY × STYLE_ACADEMIC_METASTUDY pair interaction was identified as conflicting (two STYLE templates cannot both apply simultaneously — the axes are not additive within the same axis). The compatibility matrix mechanism (`TEMPLATE_COMPATIBILITY.md`) proved essential for catching this at engagement rather than mid-editorial. Template validation at BOOK_EDITORIAL engagement is not optional overhead — it is the gate that prevents incompatible compositions from producing incoherent editorial findings downstream.

3. **FORMATTER's non-responsibilities list was critical to maintain the AI/mechanical pipeline boundary.** FORMATTER explicitly does not produce PDFs, does not apply typesetting, and does not make aesthetic decisions. Documenting what FORMATTER does NOT do (in WORKER_FORMATTER.md) was as important as documenting what it does. The AI/mechanical boundary — FORMATTER hands off to LULU_PIPELINE at BOOK_SPEC.json — is only clean if FORMATTER resists scope creep into typesetting territory.

4. **BOOK_COMPLETION as a meta-orchestrator required the chapter-subsetting protocol to exist first.** The subsetting parameter (`chapter_subset` in workflow invocations) had to be designed as a cross-workflow protocol before BOOK_COMPLETION could be specified. Any attempt to add it workflow-by-workflow after the fact would produce inconsistent parameter semantics. The lesson: meta-orchestrators expose gaps in per-workflow API consistency that would otherwise remain latent.

5. **The 6-workflow structure matched the natural cognitive handoff points.** The manual-handoff-between-stages design was validated by the buildout: each stage produces a distinct artifact that a human author can meaningfully review (BOOK_MANIFEST.json after TRIAGE; chapter files + STRUCTURE.md after CONSOLIDATION; STORYBOARD.md after STORYBOARD; polished chapters after EDITORIAL; BOOK_SPEC.json + front/back after PRODUCTION). The stages are not arbitrary — they correspond to the natural decision points where an author needs to exercise judgment before proceeding.

---

*End of BOOK Workflow Dissertation.*
