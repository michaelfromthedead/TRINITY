# BOOK Workflow Family — Build-Out TODO

**Date:** 2026-04-18
**Author:** Michael (owner) + Claude (co-architect)
**Status:** Draft for review
**Workflow being built:** BOOK (five sub-workflows + template infrastructure + mechanical pipeline)
**Dissertation reference:** `workflows/BOOK/BOOK_WORKFLOW_DISSERTATION.md` v0.1.0-DRAFT
**This file's role:** The SDLC-consumable TODO for the BOOK workflow buildout itself. Structured as 9 parts, each with numbered tasks + acceptance criteria + dependencies.

**Status legend:** `[x]` Done · `[ ]` Not started · `[~]` In progress · `[!]` Blocked · `[-]` Cut/deferred

---

## Dependency map between parts

```
Part 1 — Template Standard
   ↓
Part 2 — Initial Atomic Templates        ── uses Part 1
   ↓
Part 3 — Bundles + Compatibility Matrix  ── uses Part 2
   ↓
Part 6 — Editorial Juniors + Synthesis   ── needs Parts 1–3 (templates)
   ↓
Part 7 — Editorial Seniors + Revision    ── needs Part 6

Part 4 — Consolidation Workers           ── RDC fork, mostly independent
   ↓
Part 5 — Storyboard Workers              ── needs Part 4 (consumes its output)

Part 4.5 — BOOK_COMPLETION (mixed-state orchestrator + DRAFTER worker)
   ├── needs: scope declaration in manifest (T4.5.2) + taxonomy extension (T4.5.3)
   ├── uses: templates from Parts 1–3 (DRAFTER writes under template constraint)
   ├── invokes: existing workflows on chapter subsets (requires T4.8, T5.5, T6.8, T7.7)
   └── feeds: Part 5 (DRAFTER output → STORYBOARD, not CONSOLIDATION)

Part 8 — LULU_SPEC + Production Workers  ── needs Part 7 (consumes polished output)
   ↓
Part 9 — LULU_PIPELINE + Integration     ── needs Part 8; also rolls up everything else
```

**Critical path:** 1 → 2 → 3 → 6 → 7 → 8 → 9 (templates are blocking for editorial; editorial is blocking for production).
**Parallelizable:** Part 4 can be worked on any time after dissertation is stable; Part 5 needs Part 4 complete. Part 4.5 can start any time after Part 3 (needs templates) but is most valuable after Parts 4–7 exist (it orchestrates them).
**Mixed-state support:** Part 4.5 makes the system capable of handling Case A (rough + incomplete) and Case B (polished skeleton, unfinished body). Without Part 4.5, the pipeline assumes uniform per-corpus maturity.

---

# PART 1: TEMPLATE STANDARD & INFRASTRUCTURE FOUNDATION

**Scope:** Design the schema and directory layout that all templates conform to. This is the novel architectural contribution — nothing editorial works until this is done.
**Blocks:** Parts 2, 3, 6, 7.
**Estimated pessimistic:** 2 focused sessions.
**Artifact location:** new top-level `templates/` directory at project root (peer of `workflows/`) — or nested under `workflows/BOOK/templates/`. Decide in T1.1.

### Tasks

- [ ] **T1.1** — Decide templates directory location (root `templates/` vs `workflows/BOOK/templates/`). Document rationale.
  - **Acceptance:** decision recorded; directory created.

- [ ] **T1.2** — Write `TEMPLATE_STANDARD.md` — the schema doc specifying what every template file must contain.
  - **Acceptance:** doc lists required sections (Contract / Characteristic Patterns / Anti-Patterns / Audit Checklist / Interaction Notes), YAML frontmatter format (axis, version, applies_to, semver), naming convention (`VOICE_<name>.md`, `PERSONA_<name>.md`, `STYLE_<name>.md`, `PROSE_<name>.md`, `BUNDLE_<name>.md`), and references BOOK_EDITORIAL JSON's `template_architecture` section.

- [ ] **T1.3** — Design the Audit Checklist format — how editorial workers mechanically use it.
  - **Acceptance:** `TEMPLATE_STANDARD.md` includes spec: each checklist item must be answerable as PASS / FLAG / N/A, and writable as a machine-readable bullet (`- [AXIS:key] text description`). Document how JUNIOR workers cite audit items in their findings.

- [ ] **T1.4** — Write `TEMPLATE_STANDARD_EXAMPLE.md` — a tiny dummy template (e.g., `VOICE_EXAMPLE.md`) demonstrating every section with placeholder content.
  - **Acceptance:** exists; follows TEMPLATE_STANDARD exactly; can be used as a template-for-templates (`cp VOICE_EXAMPLE.md VOICE_NEWONE.md` + edit).

- [ ] **T1.5** — Design template versioning policy.
  - **Acceptance:** decision made + documented in TEMPLATE_STANDARD: SemVer per template file; MAJOR changes require manifest pinning; MINOR/PATCH compatible within a manifest.

- [ ] **T1.6** — Decide template storage: per-book-project local vs central library.
  - **Acceptance:** rationale documented. Current leaning (to validate): central library at project root `templates/`; individual books reference by name in BOOK_MANIFEST.json.

- [ ] **T1.7** — Spec BOOK_MANIFEST.json template reference resolution behavior (what happens when a referenced template is missing / renamed / versioned).
  - **Acceptance:** spec added to BOOK_EDITORIAL JSON's `template_resolution` section, OR to a new `TEMPLATE_RESOLUTION.md`. Error paths specified (blocking vs non-blocking).

- [ ] **T1.8** — Write `TEMPLATE_COMPATIBILITY.md` skeleton — the matrix spec.
  - **Acceptance:** empty matrix structure in place; `TEMPLATE_COMPATIBILITY.md` exists with spec for how to declare compatibilities. The matrix itself is populated in Part 3.

**Part 1 verification:** A new collaborator can read `TEMPLATE_STANDARD.md` + `TEMPLATE_STANDARD_EXAMPLE.md` and write a conformant atomic template without further guidance.

---

# PART 2: INITIAL ATOMIC TEMPLATE SET

**Scope:** Write the 6 atomic templates named in dissertation §4.5. These seed the system; more templates are added as Michael catalogs future works.
**Depends on:** Part 1 complete.
**Blocks:** Part 3 (bundles), Part 6 (editorial juniors need at least one full template set to audit against).
**Estimated pessimistic:** 4–6 focused sessions (writing the templates is slow; each is a careful distillation of Michael's established voice/style).

### Voice templates

- [ ] **T2.1** — Write `VOICE_SOCRATIC.md`.
  - **Contract (must appear):** "We tell nothing — we show, compare, ask." Every chapter starts with observation, builds through exploration, arrives at insight.
  - **Acceptance:** conforms to TEMPLATE_STANDARD; Audit Checklist has ≥10 machine-checkable items; Characteristic Patterns has ≥5 positive examples with annotation; Anti-Patterns has ≥5 negative examples; Interaction Notes addresses how this voice interacts with PERSONA / STYLE / PROSE axes.

- [ ] **T2.2** — Write `VOICE_FEYNMAN.md`.
  - **Contract:** Peer-explanation, concrete-before-abstract, "let me show you how I think about this." Humor permitted. Wonder encouraged. Complexity never used as gatekeeping.
  - **Acceptance:** same criteria as T2.1.

### Persona templates

- [ ] **T2.3** — Write `PERSONA_PHYSICIST_TEACHER.md`.
  - **Contract:** Domain authority as theoretical physicist who has thought deeply and wants to share thinking process, not just conclusions.
  - **Acceptance:** same criteria as T2.1, plus Interaction Notes on which voices this persona composes with (Socratic: yes; authoritative-textbook: tension).

### Style templates

- [ ] **T2.4** — Write `STYLE_ACADEMIC_EXPLORATORY.md`.
  - **Contract:** Academic rigor, exploratory structure. Citations present but not dominant. Argument structure dialectic or inductive. Chapter structure follows natural conceptual boundaries.
  - **Acceptance:** same criteria as T2.1, plus explicit citation convention spec (inline? footnote? parenthetical?), prose density range (words per page approx), heading level rules.

- [ ] **T2.5** — Write `STYLE_ACADEMIC_METASTUDY.md`.
  - **Contract:** Synthesizes across multiple domains. Heavy citation. Systematic coverage. Comparative analysis structure.
  - **Acceptance:** same criteria as T2.4.

### Prose templates

- [ ] **T2.6** — Write `PROSE_MEDIUM_ACCESSIBLE.md`.
  - **Contract:** Sentences range simple→moderately complex. Paragraphs develop single idea. Technical terms defined in context. Metaphor/analogy used regularly. Parentheticals sparingly.
  - **Acceptance:** same criteria as T2.1, plus explicit ranges: sentence complexity bounds, paragraph length bounds, vocabulary register tier (B2/C1/C2 CEFR equivalents), parenthetical frequency cap.

### Meta

- [ ] **T2.7** — Cross-review: each atomic read by "another axis's template" author to check for contradiction (e.g., does VOICE_SOCRATIC's question-frequency rule conflict with PROSE_MEDIUM_ACCESSIBLE's sentence-complexity ceiling?). Document conflicts or certify clean.
  - **Acceptance:** cross-review report appended to each atomic as `Interaction Notes` addenda, OR the atomic is already self-consistent.

- [ ] **T2.8** — Smoke-test each atomic by running it (mentally) against a 1-page sample of Michael's existing writing. Do the Audit Checklist items actually fire on expected passages?
  - **Acceptance:** each atomic has a smoke-test note in its frontmatter confirming it was tested against real prose.

**Part 2 verification:** 6 atomic templates all conform to TEMPLATE_STANDARD; cross-reviewed; smoke-tested; ready to reference from bundles and manifests.

---

# PART 3: BUNDLES & COMPATIBILITY MATRIX

**Scope:** Hand-compose the first bundle (BUNDLE_SPIN_OF_GRAVITY) that combines 4 atomics with synergy notes. Populate the compatibility matrix across the 6 atomics from Part 2.
**Depends on:** Parts 1, 2 complete.
**Blocks:** Part 6 (editorial needs compatibility matrix to validate manifest composition mode at engagement).
**Estimated pessimistic:** 2 focused sessions.

### Tasks

- [ ] **T3.1** — Write `BUNDLE_SPIN_OF_GRAVITY.md`.
  - **Components:** VOICE_SOCRATIC + PERSONA_PHYSICIST_TEACHER + STYLE_ACADEMIC_EXPLORATORY + PROSE_MEDIUM_ACCESSIBLE.
  - **Synergy notes:** the emergent "we tell nothing — we show, compare, ask" rule that is the product of this combination (not present in any atomic alone).
  - **Overrides (if any):** places where the bundle's combined behavior differs from what atomic interaction notes predict.
  - **Acceptance:** conforms to TEMPLATE_STANDARD's bundle subsection; lists all 4 atomic references; synergy section has ≥3 emergent rules; can be referenced in BOOK_MANIFEST.json via `templates.mode = "bundle"`.

- [ ] **T3.2** — Populate `TEMPLATE_COMPATIBILITY.md` with the 6-atomic matrix.
  - **Format:** table or structured JSON. Each cell indicates: `compatible | conflicts | requires-bundle-mediation | not-tested`.
  - **Axes pair:** VOICE×PERSONA, VOICE×STYLE, VOICE×PROSE, PERSONA×STYLE, PERSONA×PROSE, STYLE×PROSE.
  - **Acceptance:** all pairs among the 6 initial atomics classified; rationale notes for `conflicts` and `requires-bundle-mediation` entries; `not-tested` rows explicitly marked.

- [ ] **T3.3** — Write `TEMPLATE_COMPATIBILITY.md` usage spec — how editorial QUEEN validates a composition-mode manifest against the matrix at engagement.
  - **Acceptance:** algorithm documented (for each pair of declared atomics, look up matrix entry; if `conflicts` → block; if `requires-bundle-mediation` → warn and recommend bundle; if `compatible` → proceed; if `not-tested` → proceed with warning).

- [ ] **T3.4** — Identify and document at least 2 additional bundles worth writing beyond BUNDLE_SPIN_OF_GRAVITY (based on Michael's other works).
  - **Acceptance:** names + rationale in a `BUNDLES_ROADMAP.md` or section inside TEMPLATE_STANDARD. Actual writing of these deferred to Part 2 extension later.

- [ ] **T3.5** — Add bundle-vs-composition mode resolution logic spec to BOOK_EDITORIAL JSON (or reference TEMPLATE_RESOLUTION.md from T1.7).
  - **Acceptance:** JSON's `trigger.template_resolution` section is complete; error paths explicit; pointer to compatibility matrix.

**Part 3 verification:** BOOK_EDITORIAL can resolve a manifest's template declaration (bundle or composition), validate compatibility, and load the resolved template set for workers.

---

# PART 4: BOOK_CONSOLIDATION WORKERS (RDC FORK)

**Scope:** Write the 5 worker role docs for BOOK_CONSOLIDATION. 4 of them (SCRIBE, ADVOCATE, QA_COMPLETENESS, QA_COHERENCE) are adapted from RDC's existing role docs — minimal novel work. COMPOSITOR is new (replaces RDC's TAXONOMIST).
**Depends on:** Dissertation is stable. Does NOT depend on Parts 1–3 (BOOK_CONSOLIDATION operates on raw source chaos, not template-driven editorial).
**Blocks:** Part 5 (BOOK_STORYBOARD consumes BOOK_CONSOLIDATION output).
**Estimated pessimistic:** 3 focused sessions.

### Tasks

- [ ] **T4.1** — Copy-adapt `workflows/RDC/WORKER_SCRIBE.md` → `workflows/BOOK/WORKER_SCRIBE.md`.
  - **Changes:** replace "engineering concepts" framing with "manuscript concepts"; replace RDC doc references with BOOK doc references; otherwise mechanics identical.
  - **Acceptance:** exists; all doc references resolve; upsert rules preserved verbatim from RDC; forbidden-files / anti-patterns preserved.

- [ ] **T4.2** — Copy-adapt `workflows/RDC/WORKER_ADVOCATE.md` → `workflows/BOOK/WORKER_ADVOCATE.md`.
  - **Changes:** reframe examples for manuscript conflicts instead of architecture conflicts.
  - **Acceptance:** exists; court discipline preserved; references resolve.

- [ ] **T4.3** — Copy-adapt `workflows/RDC/WORKER_QA_COMPLETENESS.md` → `workflows/BOOK/WORKER_QA_COMPLETENESS.md`.
  - **Changes:** "concept-loss against engineering output docs" → "concept-loss against chapter files"; re-frame status categories for manuscript context.
  - **Acceptance:** exists; adversarial stance preserved; new inputs list matches BOOK_CONSOLIDATION JSON.

- [ ] **T4.4** — Copy-adapt `workflows/RDC/WORKER_QA_COHERENCE.md` → `workflows/BOOK/WORKER_QA_COHERENCE.md`.
  - **Changes:** engineering structure checks → manuscript structure checks (chapter ordering, section placement, STRUCTURE.md consistency); failure categories rewritten per BOOK_CONSOLIDATION JSON's `categories` list.
  - **Acceptance:** exists; all 7 checks from JSON's `roles.QA_COHERENCE.checks` are covered.

- [ ] **T4.5** — Write `workflows/BOOK/WORKER_COMPOSITOR.md` **from scratch**. This is the novel worker.
  - **Contract:** takes final MASTER.md + PEDAGOGY.md + EVALUATIONS.md + INPROGRESS court entries + BOOK_MANIFEST.json; produces `chapters/CH_<NN>_<TITLE>.md` files + `STRUCTURE.md`.
  - **Chapter discovery algorithm:** documented explicitly — how does COMPOSITOR identify chapter boundaries from MASTER content? (Conceptual coherence heuristics, dependency ordering, section-to-chapter mapping rules.)
  - **STRUCTURE.md format spec:** table of contents + chapter summaries + per-chapter section listings + inter-chapter dependency map (acyclic directed graph).
  - **Acceptance:** doc is self-contained per WORKER_PROTOCOL; chapter discovery rules are specific enough that two runs on the same MASTER produce similar chapter structure; STRUCTURE.md format spec is formal enough for QA_COHERENCE to validate.

- [ ] **T4.6** — Write `workflows/BOOK/COMPOSITOR_VS_TAXONOMIST.md` — deliberate comparison doc.
  - **Acceptance:** comparison of what's identical (scanning MASTER, discovery-over-prescription, output-set-carving) vs what's different (chapters vs phases, section hierarchy vs ARCH/TODO split, manuscript dependency vs engineering dependency). Informs the open question about future refactoring (dissertation §16.2).

- [ ] **T4.7** — Smoke-test: run mental walkthrough of BOOK_CONSOLIDATION end-to-end with worker docs in place. Does the flow work?
  - **Acceptance:** walkthrough document at `workflows/BOOK/CONSOLIDATION_WALKTHROUGH.md` using a hypothetical manuscript input; all worker interactions traced; any gaps identified.

- [ ] **T4.8** — Add chapter subsetting capability to `BOOK_CONSOLIDATION.json`.
  - **Scope:** engagement accepts optional `chapter_subset: [names or indices]` parameter (from manifest or invocation). When present, CONSOLIDATION's SCRIBE_LOOP only processes source docs relevant to those chapters; COMPOSITOR carves only those chapters; QA scopes to the subset.
  - **Acceptance:** JSON updated with subset param; worker docs note subset-awareness in inputs section; walkthrough (T4.7) covers subset invocation path.

**Part 4 verification:** BOOK_CONSOLIDATION can be triggered end-to-end. All referenced worker docs exist. RDC fork is load-bearing and self-consistent. Subset invocation works.

---

# PART 4.5: BOOK_COMPLETION — Mixed-State Handling & Drafting

**Scope:** Handles Case A (rough + incomplete) and Case B (polished skeleton, unfinished body). This is the fix for the structural gap the other parts inherit: the existing pipeline is transformation-oriented (reshapes existing content) but real manuscripts often have gaps (chapters that don't exist yet, scope intended but not realized). BOOK_COMPLETION is a meta-orchestrator that reads per-chapter state vs intent-scope and invokes existing workflows on per-chapter subsets. It owns the new DRAFTER worker, which authors prose from notes + scope + templates for chapters that don't yet exist.
**Depends on:** Parts 1–3 (templates — DRAFTER writes under template constraint) + Parts 4, 5, 6, 7 for workflows to orchestrate + T4.8 / T5.5 / T6.8 / T7.7 (chapter subsetting support).
**Blocks:** nothing critical-path; but enables mixed-state corpus handling which is a major capability.
**Estimated pessimistic:** 5–7 focused sessions. DRAFTER is substantial; COMPLETION orchestrator logic is novel for this project (first meta-workflow).

### Manifest + taxonomy foundation

- [ ] **T4.5.1** — Decide DRAFTER authorship stance: Stance 2 (skeletal, marked-for-revoicing) vs Stance 3 (full prose under template constraint). Document decision + rationale.
  - **Acceptance:** decision recorded at top of `workflows/BOOK/DRAFTER_AUTHORSHIP_STANCE.md` with consequences laid out; everything downstream in Part 4.5 assumes this stance.
  - **Proposal:** Stance 3 with safeguards — DRAFTER output MARKED in manifest as `drafter_origin: true` per chapter; EDITORIAL applies extra scrutiny to drafter-origin chapters; human review of drafter output is mandatory before EDITORIAL engagement.

- [ ] **T4.5.2** — Extend `BOOK_MANIFEST.json` schema with `scope` section.
  - **Scope fields:**
    - `intended_chapters: [{ index, title, target_topic, target_length_words, rationale, status }]` — author's up-front statement of book shape
    - `scope_declared_at: <ISO timestamp>` — when author set this
    - `scope_revision_log: [...]` — if scope was revised during the project
  - **Acceptance:** schema updated in `BOOK_TRIAGE.json` §`manifest_schema.fields` AND documented in a new `BOOK_MANIFEST_SCOPE.md`. Backward-compatible — manifests without scope are still valid (degrade to current behavior).

- [ ] **T4.5.3** — Extend per-chapter state taxonomy beyond the TRIAGE v1 labels.
  - **New labels to add:**
    - `MISSING` — no material for this intended chapter
    - `OUTLINE_ONLY` — TOC entry + 1-line description exists, no material
    - `NOTES_ONLY` — research notes/fragments exist, no prose structure
    - `PARTIALLY_DRAFTED` — prose exists but incomplete (gaps in sections)
    - `DRAFT` — prose complete but rough (existing label)
    - `CHAPTER` — structured, substantial (existing label — for mid-maturity)
    - `POLISHED` — publication-ready (existing label)
    - `DRAFTER_ORIGIN` — flag (not state): chapter was authored by DRAFTER; EDITORIAL should apply extra scrutiny
  - **Acceptance:** taxonomy documented in `workflows/BOOK/CHAPTER_STATE_TAXONOMY.md`; includes transition rules (MISSING → NOTES_ONLY is possible via human notes addition; NOTES_ONLY → DRAFT is possible via DRAFTER; etc.).

- [ ] **T4.5.4** — Extend BOOK_TRIAGE to produce per-chapter state table.
  - **Change:** TRIAGE currently outputs aggregate folder state. Extend to: match folder material against manifest's `intended_chapters`, produce per-chapter state classification using T4.5.3 taxonomy.
  - **New output field in BOOK_MANIFEST.json:** `triage.per_chapter_state: [{ chapter_index, chapter_title, detected_state, confidence, source_files, notes }]`
  - **Acceptance:** BOOK_TRIAGE.json updated with v1.1.0 behavior (backward-compatible — if no manifest scope, falls back to v1.0.0 aggregate-only). WORKER_TRIAGE doc (if we make one) or TRIAGE's JSON execution steps updated.

### BOOK_COMPLETION workflow

- [ ] **T4.5.5** — Write `workflows/BOOK/BOOK_COMPLETION.json` — the meta-orchestrator workflow spec.
  - **Trigger:** `BOOK_COMPLETION`
  - **Engagement:** QUEEN reads manifest (including scope + per-chapter state), reads BOOK_COMPLETION.json, reads WORKER_DRAFTER.md, reads BOOK_COMPLETION_ROUTING.md.
  - **Flow:** read per-chapter state → produce routing plan (which workflow per chapter) → invoke per-chapter-subset workflow calls (may spawn DRAFTER, may invoke BOOK_CONSOLIDATION on subset, may invoke BOOK_STORYBOARD on subset, may invoke BOOK_EDITORIAL on subset) → collect outputs → report completion state.
  - **Verdicts:** `ALL_CHAPTERS_COMPLETE | PARTIAL_COMPLETION (some chapters still blocked) | ESCALATE`.
  - **Acceptance:** JSON spec exists matching convention of other BOOK workflow JSONs; references WORKER_DRAFTER and routing doc; specifies that COMPLETION can be re-invoked iteratively as chapters progress.

- [ ] **T4.5.6** — Write `workflows/BOOK/BOOK_COMPLETION_ROUTING.md` — the state → workflow mapping table.
  - **Content:** for each state from T4.5.3, specify which workflow(s) apply:
    - `MISSING` → DRAFTER (from scope only, no notes) or human-authoring-blocker if Stance 1
    - `OUTLINE_ONLY` → DRAFTER (from scope + outline)
    - `NOTES_ONLY` → DRAFTER (from scope + notes)
    - `PARTIALLY_DRAFTED` → DRAFTER (fill gaps) + then normal CONSOLIDATION
    - `DRAFT` → CONSOLIDATION (as normal)
    - `CHAPTER` → STORYBOARD (if not yet storyboarded) or EDITORIAL (if storyboarded)
    - `POLISHED` → EDITORIAL (audit-only, no revision likely needed) or PRODUCTION (if already editorially-green-lit)
  - **Acceptance:** routing table complete; edge cases documented (e.g., what if DRAFTER produces from NOTES_ONLY — does output go straight to STORYBOARD or through CONSOLIDATION first?); per-chapter invocation protocol specified.

### DRAFTER worker

- [ ] **T4.5.7** — Write `workflows/BOOK/WORKER_DRAFTER.md` — the prose-generation worker.
  - **Role:** authors prose for chapters in `MISSING / OUTLINE_ONLY / NOTES_ONLY / PARTIALLY_DRAFTED` states under simultaneous constraints.
  - **Inputs:** manifest scope for the target chapter(s); any existing notes/outlines/partial prose; resolved templates (VOICE + PERSONA + STYLE + PROSE or bundle); STORYBOARD.md if exists (for prerequisite chain awareness); BOOK_MANIFEST.json for genre.
  - **Output:** `chapters/CH_<NN>_<TITLE>.md` — chapter-shaped markdown file matching CONSOLIDATION's output format, with frontmatter flag `drafter_origin: true` (per T4.5.1 safeguard).
  - **Constraints (same pattern as REVISION, applied to whole-chapter authoring):**
    - template adherence (all 4 axes)
    - scope adherence (output covers the intended_chapter.target_topic)
    - length target (hit target_length_words within ±20%)
    - prerequisite respect (don't introduce concepts the storyboard hasn't established yet, if storyboard exists)
    - consistency with existing drafter output (terminology stable across DRAFTER-produced chapters)
  - **Non-responsibilities:**
    - DRAFTER does not invent facts beyond what notes/scope provide — if material is insufficient to hit target length, DRAFTER produces a shorter draft and flags the gap
    - DRAFTER does not modify unrelated existing chapters
  - **Acceptance:** self-contained per WORKER_PROTOCOL; all 5 constraints have specific procedure; gap-flagging format specified; drafter_origin frontmatter flag documented.

- [ ] **T4.5.8** — Design DRAFTER's "insufficient material" handling.
  - **Question:** what does DRAFTER do when notes are thin or scope is vague?
  - **Acceptance:** policy documented. Proposal: produce a skeletal draft with clearly-marked `[DRAFTER_GAP: reason]` placeholders indicating where human authorial input is needed; DRAFTER does not hallucinate content beyond what notes support. Gap markers are first-class flags for BOOK_EDITORIAL or human-review.

- [ ] **T4.5.9** — Design DRAFTER output format integration with existing CONSOLIDATION output.
  - **Question:** if BOOK_CONSOLIDATION already ran on some chapters (`DRAFT` state material) and DRAFTER produces new chapters (`MISSING` or `NOTES_ONLY`), how do they merge into a unified `chapters/` + `STRUCTURE.md`?
  - **Acceptance:** protocol documented. Proposal: DRAFTER produces chapter files directly into `chapters/` with correct CH_<NN>_<TITLE>.md naming; STRUCTURE.md gets appended/edited to include DRAFTER chapters in the right position; if STRUCTURE.md doesn't exist yet, DRAFTER also produces structure entries. COMPLETION orchestrator is responsible for merging.

- [ ] **T4.5.10** — Design DRAFTER → STORYBOARD handoff (per user direction: DRAFTER output flows to STORYBOARD, not CONSOLIDATION).
  - **Rationale:** DRAFTER produces chapter-shaped output directly — doesn't need consolidation (which is about collapsing chaotic source material). Goes straight to STORYBOARD where logical structure is assessed.
  - **Acceptance:** documented in BOOK_COMPLETION_ROUTING.md; STORYBOARD is aware of `drafter_origin: true` flag for extra scrutiny in QA.

### Chapter subsetting (cross-cutting)

- [ ] **T4.5.11** — Design the chapter_subset parameter spec for workflow invocation.
  - **Scope:** a single spec that BOOK_CONSOLIDATION (T4.8), BOOK_STORYBOARD (T5.5), BOOK_EDITORIAL (T6.8, T7.7) all adopt.
  - **Spec:** `chapter_subset: null | ["CH_01", "CH_03", "CH_05"] | {"from": "CH_03", "to": "CH_08"}`. When null/absent = full-manuscript pass. When specified = only operate on listed chapters.
  - **Source:** parameter can come from manifest (persistent setting) or invocation (per-run override).
  - **Acceptance:** `workflows/BOOK/CHAPTER_SUBSET_PROTOCOL.md` documents format, precedence (invocation > manifest), and how each workflow interprets (e.g., CONSOLIDATION scopes SCRIBE_LOOP by source-doc → intended-chapter mapping; EDITORIAL scopes all 4 juniors to subset files).

### Walkthroughs + testing

- [ ] **T4.5.12** — Write Case A walkthrough: `workflows/BOOK/COMPLETION_CASE_A_WALKTHROUGH.md`.
  - **Scenario:** 12-chapter book intended; folder contains rough notes for chapters 1, 2, 4, 5, 7 only; chapters 3, 6, 8–12 entirely absent.
  - **Trace:** manifest scope declaration → TRIAGE v1.1 classifies per-chapter state → COMPLETION invokes DRAFTER on chapters 3, 6, 8–12 → invokes CONSOLIDATION on chapters 1, 2, 4, 5, 7 subset → STORYBOARD on merged result → EDITORIAL on full set → PRODUCTION.
  - **Acceptance:** walkthrough traces each workflow invocation with expected inputs/outputs; gap-flags from DRAFTER are highlighted as items needing human follow-up.

- [ ] **T4.5.13** — Write Case B walkthrough: `workflows/BOOK/COMPLETION_CASE_B_WALKTHROUGH.md`.
  - **Scenario:** 12-chapter book intended; chapters 1–4 POLISHED, chapter 5 PARTIALLY_DRAFTED, chapters 6–8 NOTES_ONLY with detailed outlines, chapters 9–12 OUTLINE_ONLY (TOC entry + 1-line description).
  - **Trace:** TRIAGE v1.1 per-chapter classification → COMPLETION routing: chapters 1–4 → EDITORIAL audit-only; chapter 5 → DRAFTER fills gaps then CONSOLIDATION then STORYBOARD; chapters 6–8 → DRAFTER from notes; chapters 9–12 → DRAFTER from outline (expected to produce more gap-flags); all chapters merge → STORYBOARD on full set → EDITORIAL on full set → PRODUCTION.
  - **Acceptance:** walkthrough traces each invocation; highlights where human review is expected (drafter output review gates).

- [ ] **T4.5.14** — Smoke-test Case A (manual walkthrough with hypothetical inputs).
  - **Acceptance:** no logical contradictions in the flow; all worker hand-offs are coherent.

- [ ] **T4.5.15** — Smoke-test Case B.
  - **Acceptance:** same as T4.5.14; mixed-state routing produces a coherent execution plan.

**Part 4.5 verification:** BOOK_COMPLETION can orchestrate mixed-state corpora. DRAFTER produces template-constrained drafts for missing/outline/notes chapters. Chapter subsetting works across existing workflows. Cases A and B both have traced walkthroughs.

---

# PART 5: BOOK_STORYBOARD WORKERS

**Scope:** Write STORYBOARDER (constructive) + QA_STORYBOARD (auditing) role docs.
**Depends on:** Part 4 (BOOK_STORYBOARD consumes `chapters/` + `STRUCTURE.md` produced by BOOK_CONSOLIDATION).
**Blocks:** Part 6 (BOOK_EDITORIAL's JUNIOR_FLOW references STORYBOARD.md).
**Estimated pessimistic:** 2 focused sessions.

### Tasks

- [ ] **T5.1** — Write `workflows/BOOK/WORKER_STORYBOARDER.md`.
  - **Contract:** voice-neutral, reads full manuscript, produces `STORYBOARD.md` with per-chapter entries (opening state / key moves / closing state / concepts introduced / concepts required / chapter function) + full-work output (arc map / prerequisite chain / reader journey).
  - **Voice-neutrality enforcement:** doc includes explicit anti-examples — phrases that STORYBOARDER must NOT use because they would leak manuscript voice into the skeleton.
  - **Acceptance:** self-contained per WORKER_PROTOCOL; per-chapter output format is a template STORYBOARDER can fill mechanically; full-work output format specified.

- [ ] **T5.2** — Write `workflows/BOOK/WORKER_QA_STORYBOARD.md`.
  - **Checks (from BOOK_STORYBOARD JSON §roles.QA_STORYBOARD.checks):** prerequisite satisfaction, progressive arc, completeness, accuracy (spot-check), genre alignment, dependency acyclicity, reader journey coherence.
  - **Acceptance:** each check has specific verification procedure; spot-check methodology for accuracy (how many chapters to sample, how to compare); DAG verification algorithm for prerequisite chain.

- [ ] **T5.3** — Define `STORYBOARD.md` format spec more formally — specific enough that QA_STORYBOARD can validate structurally.
  - **Acceptance:** format spec document exists either as section in WORKER_STORYBOARDER or as separate `STORYBOARD_FORMAT.md`; includes heading hierarchy, required fields per chapter entry, prerequisite chain syntax.

- [ ] **T5.4** — Smoke-test walkthrough.
  - **Acceptance:** `workflows/BOOK/STORYBOARD_WALKTHROUGH.md` with hypothetical chapters-input through to STORYBOARD.md output; flow verified.

- [ ] **T5.5** — Add chapter subsetting + DRAFTER-origin awareness to `BOOK_STORYBOARD.json`.
  - **Subset scope:** same pattern as T4.8 — accept `chapter_subset` parameter at engagement.
  - **DRAFTER-origin awareness:** STORYBOARD's engagement pre-step reads chapter frontmatter; chapters with `drafter_origin: true` get additional attention in QA_STORYBOARD's accuracy spot-check (verify DRAFTER's prose actually matches storyboard's description of what the chapter does).
  - **Acceptance:** JSON updated; WORKER_STORYBOARDER (T5.1) and WORKER_QA_STORYBOARD (T5.2) updated with DRAFTER-origin handling; walkthrough (T5.4) covers subset + drafter-origin invocation paths.

**Part 5 verification:** BOOK_STORYBOARD can be triggered against BOOK_CONSOLIDATION or DRAFTER output and produces a valid STORYBOARD.md that editorial can consume. Subset invocation works; drafter-origin chapters get extra scrutiny.

---

# PART 6: BOOK_EDITORIAL JUNIORS + SYNTHESIS (Part 1 of Editorial)

**Scope:** Write the 4 parallel junior workers + the cross-axis synthesis worker. These are the "front end" of the editorial pipeline.
**Depends on:** Parts 1, 2, 3 complete (templates must exist); Part 5 complete (STORYBOARD.md exists as reference for JUNIOR_FLOW).
**Blocks:** Part 7 (seniors consume junior output).
**Estimated pessimistic:** 4 focused sessions (each junior is substantial; SYNTHESIS is complex).

### Tasks

- [ ] **T6.1** — Write `workflows/BOOK/WORKER_JUNIOR_VOICE.md`.
  - **Stance:** hypercritical, adversarial, high-recall. Over-flagging is by design.
  - **Reference:** the manifest-declared VOICE template (atomic or from bundle). Uses its Audit Checklist as primary evaluation framework.
  - **Output:** findings list with severity labels (Critical/High/Medium/Low), each finding citing: chapter / section / paragraph / specific passage / violation description / which audit checklist item was violated.
  - **Isolation rule:** explicit note that JUNIOR_VOICE cannot see other juniors' findings.
  - **Acceptance:** self-contained; Audit Checklist iteration procedure documented; severity taxonomy defined.

- [ ] **T6.2** — Write `workflows/BOOK/WORKER_JUNIOR_CONCEPT.md`.
  - **Stance:** same as T6.1.
  - **Reference:** not tied to any template — content-level check. Uses STORYBOARD.md's prerequisite chain as structural reference.
  - **Checks:** term consistency, definition-before-use, no contradictions, concept completeness (cross-ref prerequisite chain), notation consistency, cross-reference validity.
  - **Acceptance:** each check has specific detection procedure; findings format matches T6.1 structure.

- [ ] **T6.3** — Write `workflows/BOOK/WORKER_JUNIOR_STYLE.md`.
  - **Reference:** manifest-declared STYLE template + PROSE template.
  - **Checks:** citation conventions, argument structure, section/chapter structure per genre norms, prose density, jargon handling, math/technical presentation.
  - **Acceptance:** each check references specific TEMPLATE_STANDARD audit items; findings format matches T6.1.

- [ ] **T6.4** — Write `workflows/BOOK/WORKER_JUNIOR_FLOW.md`.
  - **Reference:** primarily STORYBOARD.md; secondarily STRUCTURE.md.
  - **Checks:** chapter transitions, argument arc, redundancy, pacing, reader journey, storyboard fidelity.
  - **Acceptance:** each check has specific verification procedure against STORYBOARD.md; findings format matches T6.1.

- [ ] **T6.5** — Write `workflows/BOOK/WORKER_EDITORIAL_SYNTHESIS.md`.
  - **Role:** the ONLY worker that sees all 4 junior reports simultaneously. Passes through all junior findings (axis-tagged) + surfaces NEW cross-axis findings.
  - **Cross-axis detection targets (from BOOK_EDITORIAL JSON):** voice-style conflicts, concept-flow conflicts, prose-voice conflicts, style-flow conflicts, voice-concept conflicts, compound issues.
  - **Acceptance:** each cross-axis interaction type has a detection procedure; pass-through format preserves junior findings with axis tags; new-findings section uses distinct severity framing.

- [ ] **T6.6** — Design unified "finding" data format used by all juniors + synthesis + seniors.
  - **Acceptance:** `FINDING_FORMAT.md` specifies schema: { id, axis, severity, chapter, section, paragraph_range, passage_quote, violation_description, audit_checklist_item_ref, correction_guidance (optional) }. Used consistently across T6.1–T6.5.

- [ ] **T6.7** — Smoke-test: walk a hypothetical storyboarded manuscript through the 4 juniors + synthesis. Does the data format hold?
  - **Acceptance:** `workflows/BOOK/EDITORIAL_FRONT_WALKTHROUGH.md` traces a 3-chapter mock through; finding counts and cross-axis examples reasonable.

- [ ] **T6.8** — Add chapter subsetting + DRAFTER-origin awareness to `BOOK_EDITORIAL.json` (junior/synthesis stage).
  - **Subset scope:** same pattern as T4.8 — accept `chapter_subset` parameter; all 4 juniors and SYNTHESIS scope to subset.
  - **DRAFTER-origin awareness:** when a chapter has `drafter_origin: true` in frontmatter, juniors apply extra scrutiny and JUNIOR_CONCEPT specifically checks for DRAFTER_GAP placeholder markers (see T4.5.8) — any such marker is a blocking Critical finding unless resolved.
  - **Acceptance:** JSON updated; junior worker docs (T6.1–T6.4) note DRAFTER-origin handling + GAP-marker detection; SYNTHESIS considers axis interactions specific to freshly-drafted content.

**Part 6 verification:** The front half of BOOK_EDITORIAL produces an integrated findings report. Seniors can consume it. Subset works; DRAFTER-origin chapters get appropriate scrutiny.

---

# PART 7: BOOK_EDITORIAL SENIORS + REVISION (Part 2 of Editorial)

**Scope:** Write the two senior roles (SANITY, FINAL) + the REVISION worker. REVISION is the highest-skill role in the entire BOOK family — it performs actual prose writing under simultaneous template + storyboard + concept + context constraints.
**Depends on:** Part 6 complete (consumes synthesis output).
**Blocks:** Part 8 (BOOK_PRODUCTION consumes polished manuscript from editorial).
**Estimated pessimistic:** 3 focused sessions (REVISION alone is substantial).

### Tasks

- [ ] **T7.1** — Write `workflows/BOOK/WORKER_SENIOR_SANITY.md`.
  - **Role:** precision filter. Reads integrated findings. For each finding: marks `real` (valid) or `overzealous` (false positive) with 1-line rationale. Does NOT produce new findings.
  - **Hard constraint:** "does not produce new findings" is explicit and repeated in the doc.
  - **Acceptance:** procedure for evaluating each finding; rationale format spec; isolation rule stated (may not propose corrections — only rules on findings).

- [ ] **T7.2** — Write `workflows/BOOK/WORKER_SENIOR_FINAL.md`.
  - **Role:** independent full-manuscript pass + binding verdict emission. May surface NEW findings the entire pipeline missed (especially emergent / holistic issues). Emits verdict: GREEN_LIGHT | REVISE | ESCALATE.
  - **Authority:** QUEEN does NOT override. This is stated.
  - **REVISE verdict output:** consolidated actionable findings for REVISION worker — only `real` findings from sanity pass + any new findings from this pass, with specific passage references and correction guidance.
  - **ESCALATE verdict output:** blocker description; examples documented (intentional voice shift? genuine material ambiguity?).
  - **Acceptance:** verdict-emission rules explicit (what combinations of findings → which verdict); correction-guidance format spec; escalation examples catalog.

- [ ] **T7.3** — Write `workflows/BOOK/WORKER_REVISION.md` — **the hardest worker doc to write**.
  - **Role:** surgical rewrite. ONLY flagged passages are modified. All other text immutable.
  - **Constraints to satisfy simultaneously:**
    - template adherence (voice, persona, style, prose) per declared manifest templates
    - storyboard adherence (don't break logical structure described in STORYBOARD.md)
    - concept consistency (don't introduce new terminology, don't contradict existing definitions)
    - local context (surrounding unflagged text must still flow naturally after the edit)
    - minimality (prefer smallest edit that addresses the finding)
  - **Conflict handling:** if REVISION cannot satisfy all constraints simultaneously, it must flag the specific conflict in its report rather than silently compromising one constraint.
  - **Output:** revised chapter files (only touched files) + per-passage revision log: { finding_id, original_passage, revised_passage, change_rationale }.
  - **Acceptance:** doc specifies all 5 constraints with examples of each; conflict-flagging procedure documented; minimality heuristics given; cross-references all templates and STORYBOARD.md.

- [ ] **T7.4** — Define "revision budget" — open question from dissertation §16.3.
  - **Question:** max passages per cycle to prevent wholesale rewriting?
  - **Acceptance:** decision documented. Proposal: soft cap at 20 passages per REVISE cycle; if findings exceed, REVISION addresses highest-severity first and ESCALATEs rather than exceeding cap. Rationale: surgical discipline degrades beyond ~20 simultaneous edits.

- [ ] **T7.5** — Write `workflows/BOOK/REVISION_CONSTRAINT_MATRIX.md` — a reference doc mapping finding types to which constraints they most-likely-interact-with.
  - **Acceptance:** table shows common finding types (voice break / concept mismatch / flow break / style violation) × each constraint axis (template / storyboard / concept / context) with expected interaction hints for REVISION to use.

- [ ] **T7.6** — Smoke-test: walk SENIOR_SANITY + SENIOR_FINAL + REVISION with a mock findings input. Does the REVISE loop converge?
  - **Acceptance:** `workflows/BOOK/EDITORIAL_BACK_WALKTHROUGH.md` traces a mock pipeline through 2 REVISE cycles; verdict progression sensible.

- [ ] **T7.7** — Add chapter subsetting + DRAFTER-origin discipline to `BOOK_EDITORIAL.json` (senior/revision stage) and WORKER_REVISION.md.
  - **Subset scope:** SENIOR_SANITY, SENIOR_FINAL, REVISION all scope to subset.
  - **DRAFTER-origin in REVISION:** revision discipline is slightly different for drafter-origin chapters — prose is newer and less "owned" by author, so REVISION may make larger edits than normal surgical scope; but still within template + storyboard + concept constraints. Document the looser scope explicitly to avoid conflict with the "REVISION is surgical" rule.
  - **Acceptance:** JSON updated; WORKER_REVISION.md (T7.3) has a section on DRAFTER-origin revision discipline; conflict with surgical-scope rule addressed explicitly (DRAFTER-origin chapters permit "passage-scale" edits where normal revision permits only "sentence-scale" edits).

**Part 7 verification:** BOOK_EDITORIAL end-to-end is complete. 8 worker roles exist and are self-consistent. Pipeline can be triggered on a storyboarded manuscript, including subset invocation and drafter-origin handling.

---

# PART 8: LULU_SPEC + BOOK_PRODUCTION WORKERS

**Scope:** Research and capture Lulu.com print requirements. Write FORMATTER (constructive) and QA_PRODUCTION (validation) role docs.
**Depends on:** Part 7 complete (FORMATTER validates editorial's polished output).
**Blocks:** Part 9 (LULU_PIPELINE consumes BOOK_SPEC.json produced by FORMATTER).
**Estimated pessimistic:** 3 focused sessions.

### Tasks

- [ ] **T8.1** — Research Lulu.com's current print specifications.
  - **Scope:** trim sizes (6x9, 5.5x8.5, 5x8, 8.5x11, others); margin/bleed/gutter rules per trim size; spine width formula (page count × paper thickness by paper type); PDF/X standards supported; cover template dimensions; color-vs-B&W rules; minimum page count; maximum page count; paper types available; binding types supported.
  - **Acceptance:** `LULU_SPEC.md` at project root (or `infrastructure/LULU_SPEC.md`) contains: all trim sizes as a table with margin rules; spine formula with paper-type coefficients; PDF standard requirements; cover template dims; special-case rules (e.g., color interior requires different paper). Every fact cited to a Lulu.com documentation URL.

- [ ] **T8.2** — Design LULU_SPEC update policy.
  - **Acceptance:** policy doc section — how often to re-verify against Lulu.com; how to handle spec version drift; LULU_SPEC's own version field.

- [ ] **T8.3** — Write `workflows/BOOK/WORKER_FORMATTER.md`.
  - **Responsibilities (from BOOK_PRODUCTION JSON §roles.FORMATTER):**
    - manuscript validation (all chapters, no placeholders, consistent heading levels, well-formed markup)
    - front matter generation (title page, copyright, TOC always; dedication/preface/foreword/acknowledgments/epigraph conditional on manifest)
    - back matter generation (bibliography/index/glossary/appendices/about-author/colophon conditional on manifest)
    - BOOK_SPEC.json generation with calculated fields (spine_width, gutter, page_count_estimate)
  - **Non-responsibilities (explicit):** no chapter prose modification; no PDF production; no aesthetic decisions beyond manifest/templates.
  - **Acceptance:** each responsibility has specific procedure; generation-vs-validation branch clearly specified (if file exists in `front/`, validate; else generate); calculated-field formulas cited to LULU_SPEC.

- [ ] **T8.4** — Write `workflows/BOOK/WORKER_QA_PRODUCTION.md`.
  - **Checks (from BOOK_PRODUCTION JSON §roles.QA_PRODUCTION.checks):**
    - file integrity: all manifest files exist; manifest order matches STRUCTURE.md; no extras
    - spec compliance: trim_size valid, margins in bounds, spine width correct, bleed correct, PDF standard declaration valid
    - content completeness: declared front/back matter files exist, no placeholder text, TOC matches structure
    - structural consistency: heading levels, well-formed markup, cross-refs, figure/table/equation numbering
  - **Acceptance:** each check has specific verification procedure; spec-compliance checks cite LULU_SPEC sections for validation values.

- [ ] **T8.5** — Finalize BOOK_SPEC.json schema.
  - **Scope:** take the schema in BOOK_PRODUCTION JSON §book_spec_schema and elevate to a standalone `BOOK_SPEC_SCHEMA.json` (JSON Schema format) for validation tooling.
  - **Acceptance:** schema file exists; validates against a sample BOOK_SPEC.json; validation command documented.

- [ ] **T8.6** — Design front matter generation templates.
  - **Scope:** per-type boilerplate that FORMATTER fills from BOOK_MANIFEST.json (title template, copyright template, TOC-from-STRUCTURE template).
  - **Acceptance:** `workflows/BOOK/front_matter_templates/` directory with starter templates for each front-matter type.

- [ ] **T8.7** — Design back matter generation templates.
  - **Scope:** bibliography extraction algorithm (collect citations from chapters); preliminary index generation algorithm (extract key terms); glossary extraction (defined-terms harvesting).
  - **Acceptance:** `workflows/BOOK/back_matter_templates/` directory; algorithm docs for each generator.

- [ ] **T8.8** — Smoke-test: walk a polished manuscript through FORMATTER + QA_PRODUCTION. Does BOOK_SPEC.json validate against LULU_SPEC?
  - **Acceptance:** `workflows/BOOK/PRODUCTION_WALKTHROUGH.md` traces mock polished input to BOOK_SPEC.json + front/ + back/ output; all checks pass.

**Part 8 verification:** BOOK_PRODUCTION can be triggered against a polished manuscript; produces validated, automation-ready output set.

---

# PART 9: LULU_PIPELINE + INTEGRATION + OPEN-QUESTION RESOLUTION

**Scope:** The mechanical (non-AI) build automation; registry integration with CLAUDE_APPENDIX and WORKER.md index; resolve dissertation §16 open questions; end-to-end testing; documentation.
**Depends on:** Part 8 complete.
**Blocks:** nothing — this is the terminal part.
**Estimated pessimistic:** 4–6 focused sessions (LULU_PIPELINE programming alone is substantial; choice of typesetting engine affects scope).

### LULU_PIPELINE (mechanical, non-AI)

- [ ] **T9.1** — Choose typesetting engine: LaTeX (pdflatex/xelatex/lualatex) vs Typst vs WeasyPrint/Paged.js vs Pandoc+template.
  - **Acceptance:** decision documented with rationale in `workflows/BOOK/LULU_PIPELINE_DESIGN.md`; criteria include: markdown-to-typeset fidelity, mathematical typesetting quality, reproducibility across systems, learning curve, ecosystem maturity.

- [ ] **T9.2** — Design LULU_PIPELINE directory structure.
  - **Acceptance:** layout documented (e.g., `lulu_pipeline/src/`, `lulu_pipeline/templates/`, `lulu_pipeline/tests/`).

- [ ] **T9.3** — Implement LULU_PIPELINE reader: BOOK_SPEC.json + file_manifest → internal IR.
  - **Acceptance:** script reads a valid BOOK_SPEC.json and produces an internal representation suitable for typesetting engine.

- [ ] **T9.4** — Implement typesetting engine invocation with BOOK_SPEC-derived parameters (trim, margins, fonts, page numbering, header/footer, chapter break mode).
  - **Acceptance:** given a valid BOOK_SPEC.json + file_manifest, pipeline produces a PDF.

- [ ] **T9.5** — Implement PDF/X compliance step (post-process or engine flag).
  - **Acceptance:** output PDF passes PDF/X validation (use Ghostscript, verapdf, or similar).

- [ ] **T9.6** — Write LULU_PIPELINE test suite with a reference manuscript.
  - **Acceptance:** reference test manuscript exists; pipeline produces byte-identical PDF on repeat runs; test case for each supported trim size.

- [ ] **T9.7** — Document LULU_PIPELINE invocation.
  - **Acceptance:** `lulu_pipeline/README.md` with installation, invocation, troubleshooting sections.

### Integration

- [ ] **T9.8** — Update `workflows/CLAUDE_APPENDIX.md` to list BOOK workflows.
  - **Acceptance:** BOOK family appears in the workflow table (as 5 entries or as one rollup); engagement behavior section covers BOOK triggers; exit behavior covered; file artifacts table updated with BOOK-family outputs; references list updated.

- [ ] **T9.9** — Update `workflows/SHARED/WORKER.md` master index with BOOK workflows + all new role docs.
  - **Acceptance:** master index lists BOOK as 4th workflow family; all worker roles listed with doc paths; composite units documented (JUNIOR_EDITORIAL, EDITORIAL_PIPELINE, COURT_UNIT, QA_UNIT).

- [ ] **T9.10** — Re-run `bash workflows/install.sh appendix` to regenerate project CLAUDE.md.
  - **Acceptance:** project CLAUDE.md shows BOOK workflows; install pipeline idempotent.

### Open-question resolution (dissertation §16)

- [ ] **T9.11** — Resolve: CI/CD hooks for BOOK workflows (dissertation §16.1).
  - **Acceptance:** decision documented. Proposal: add `workflows/ci-prose.sh` with markdown lint + spell check + template-conformance spot check; hook fires on commits touching `chapters/`, `front/`, `back/`, `templates/`.

- [ ] **T9.12** — Resolve: COMPOSITOR vs TAXONOMIST refactoring (dissertation §16.2).
  - **Acceptance:** decision based on results of T4.6 comparison doc + observed duplication. Proposal: defer until after first real BOOK project completes, then assess empirically.

- [ ] **T9.13** — Resolve: REVISION capability boundaries (dissertation §16.3).
  - **Acceptance:** resolved in T7.4; document decision in dissertation open-questions section as closed.

- [ ] **T9.14** — Resolve: multi-language manuscripts (dissertation §16.4).
  - **Acceptance:** decision documented. Proposal: language handled within PROSE template as a sub-axis (e.g., `PROSE_MEDIUM_ACCESSIBLE_KR.md`) rather than introducing a 5th axis.

- [ ] **T9.15** — Resolve: figure/equation/code integration (dissertation §16.5).
  - **Acceptance:** decision documented. Proposal: inline markdown for equations (LaTeX delimiters), inline code blocks for listings, figures as file references with markdown image syntax; BOOK_SPEC.json `special_elements` flags drive typesetting.

- [ ] **T9.16** — Resolve: index generation ownership (dissertation §16.6).
  - **Acceptance:** decision documented. Proposal: FORMATTER produces preliminary index (key-term extraction); human reviews/refines during BOOK_PRODUCTION review; mechanical pipeline does not modify.

- [ ] **T9.17** — Resolve: template versioning strategy (dissertation §16.7).
  - **Acceptance:** resolved in T1.5; document decision in dissertation open-questions section as closed.

### End-to-end testing

- [ ] **T9.18** — Create a test manuscript project (small but real).
  - **Acceptance:** a mini manuscript exists (maybe 3-5 chapters on a self-contained topic) to serve as the regression-test input for all BOOK workflows.

- [ ] **T9.19** — Run the full BOOK pipeline against the test manuscript (uniform-maturity case).
  - **Acceptance:** TRIAGE → CONSOLIDATION → STORYBOARD → EDITORIAL → PRODUCTION → LULU_PIPELINE all execute to GREEN_LIGHT on the test manuscript; print-ready PDF produced; visual inspection confirms reasonable output.

- [ ] **T9.19.5** — Run the mixed-state BOOK pipeline against a Case A test manuscript (rough + incomplete).
  - **Acceptance:** TRIAGE v1.1 → COMPLETION (with DRAFTER invocations + CONSOLIDATION/STORYBOARD/EDITORIAL subset invocations) → PRODUCTION → LULU_PIPELINE produces a coherent PDF. Human-review gates at DRAFTER-output boundaries are observed.

- [ ] **T9.19.6** — Run the mixed-state BOOK pipeline against a Case B test manuscript (polished skeleton, unfinished body).
  - **Acceptance:** TRIAGE v1.1 → COMPLETION orchestrates per-chapter routing (some chapters to EDITORIAL audit-only, some to DRAFTER, some to CONSOLIDATION) → PRODUCTION → LULU_PIPELINE. Subset invocation protocol verified. Drafter-origin chapters successfully marked and reviewed.

- [ ] **T9.20** — Document known limitations / edge cases discovered during testing.
  - **Acceptance:** `workflows/BOOK/KNOWN_LIMITATIONS.md` lists any issues found during E2E test and their proposed follow-ups.

### Documentation

- [ ] **T9.21** — Update dissertation to v1.0.0 (post-build state).
  - **Acceptance:** version bumped; open-questions section updated with resolutions; any architectural learnings from actual implementation folded in; status changed from DRAFT to IMPLEMENTED.

- [ ] **T9.22** — Write `workflows/BOOK/README.md` — entry point doc for BOOK family.
  - **Acceptance:** README exists; links to dissertation + each workflow JSON + each worker doc; quickstart section for triggering a BOOK project.

**Part 9 verification:** BOOK workflow family is production-ready. E2E pipeline works. Registry updated. Open questions resolved or intentionally deferred. Documentation current.

---

# SUMMARY STATS

| Part | Name | Tasks | Critical Path? | Blocks |
|---|---|---|---|---|
| 1 | Template Standard | 8 | YES | 2, 3, 6, 7 |
| 2 | Initial Atomic Templates | 8 | YES | 3, 6 |
| 3 | Bundles + Compatibility Matrix | 5 | YES | 6 |
| 4 | Consolidation Workers (+ subset cap) | 8 | no (parallel) | 5, 4.5 |
| 4.5 | BOOK_COMPLETION (meta + DRAFTER) | 15 | no (parallel) | enables mixed-state |
| 5 | Storyboard Workers (+ subset + drafter-origin) | 5 | no | 6 |
| 6 | Editorial Juniors + Synthesis (+ subset) | 8 | YES | 7 |
| 7 | Editorial Seniors + Revision (+ subset + drafter-origin) | 7 | YES | 8 |
| 8 | LULU_SPEC + Production Workers | 8 | YES | 9 |
| 9 | LULU_PIPELINE + Integration + E2E (uniform + Case A + Case B) | 24 | YES | (terminal) |
| **Total** | **10 parts** | **96 tasks** | | |

**Estimated pessimistic effort (sum of per-part pessimistic):** ~32–42 focused sessions (Part 4.5 adds ~5–7 sessions for COMPLETION orchestrator + DRAFTER + cross-cutting subset capability).

---

# ACCEPTANCE CRITERIA FOR "BOOK WORKFLOW FAMILY COMPLETE"

1. All 96 tasks marked `[x]` or explicitly `[-]` (cut/deferred with rationale).
2. `workflows/BOOK/BOOK_WORKFLOW_DISSERTATION.md` bumped to v1.0.0 status IMPLEMENTED; includes dissertation update documenting BOOK_COMPLETION + DRAFTER + mixed-state handling.
3. `workflows/CLAUDE_APPENDIX.md` lists BOOK family (6 workflows: TRIAGE, CONSOLIDATION, COMPLETION, STORYBOARD, EDITORIAL, PRODUCTION); install.sh re-run; project CLAUDE.md reflects.
4. `workflows/SHARED/WORKER.md` master index includes all BOOK worker roles (including DRAFTER).
5. Three end-to-end tests pass:
   - **Uniform-maturity test** (T9.19): standard pipeline on a test manuscript
   - **Case A test** (T9.19.5): rough + incomplete corpus routed through COMPLETION
   - **Case B test** (T9.19.6): polished skeleton with unfinished body, mixed-state routing
6. All dissertation §16 open questions resolved or explicitly deferred with rationale.
7. Authorship stance (T4.5.1) documented and consistently applied across DRAFTER + REVISION disciplines.
8. `workflows/BOOK/README.md` exists and accurately reflects the implemented system.

---

*End of BOOK Buildout TODO.*
