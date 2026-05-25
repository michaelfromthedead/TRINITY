# COMPLETION_CASE_A_WALKTHROUGH

**Version:** 1.0.0
**Date:** 2026-04-18
**Tasks:** T4.5.12, T4.5.14 (walkthrough + smoke-test)
**Scenario:** Rough + incomplete corpus — notes for some chapters, entirely absent chapters

---

## Scenario definition

**Book:** 12-chapter academic-exploratory book. Templates declared: `BUNDLE_SPIN_OF_GRAVITY`. Genre: `academic-exploratory`. Target audience: graduate physics students.

**Folder state at TRIAGE:**
- CH_01: `source/intro_notes.md` — research notes, no prose structure → NOTES_ONLY
- CH_02: `source/rotation_draft.md` — research notes + some structured paragraphs → NOTES_ONLY
- CH_03: no source material → MISSING
- CH_04: `source/su2_notes.md` — detailed research notes → NOTES_ONLY
- CH_05: `source/spin_formalism_v1.md` — research notes + rough prose → NOTES_ONLY
- CH_06: no source material → MISSING
- CH_07: `source/representations.md` — structural notes with some developed prose → PARTIALLY_DRAFTED
- CH_08 through CH_12: no source material → MISSING (5 chapters)

**Total: 7 source files; 7 chapters with material; 5 chapters MISSING; 1 PARTIALLY_DRAFTED**

---

## Step 1: Manifest setup

**Human action:** Runs BOOK_TRIAGE, reviews results, adds scope section to BOOK_MANIFEST.json.

```json
{
  "version": "1.1.0",
  "title": "The Geometry of Spin",
  "author": "Michael",
  "scope": {
    "scope_declared_at": "2026-04-18T09:00:00Z",
    "scope_revision_log": [],
    "intended_chapters": [
      {"index": 1, "title": "The Problem of Rotation", "slug": "CH_01_PROBLEM_OF_ROTATION", "target_topic": "Establish why classical rotation is insufficient. Motivate mathematical upgrade.", "target_length_words": 5000, "rationale": "Opens manuscript; no prerequisites.", "status": "NOTES_ONLY"},
      {"index": 2, "title": "SU(2) and Rotation Groups", "slug": "CH_02_SU2_ROTATION", "target_topic": "Introduce SU(2) group structure and its relationship to SO(3).", "target_length_words": 6000, "rationale": "Requires CH_01. Establishes SU(2) language used throughout.", "status": "NOTES_ONLY"},
      {"index": 3, "title": "Spinors: The Mathematical Object", "slug": "CH_03_SPINORS_MATH", "target_topic": "Define spinors, spinor space, transformation rules under SU(2).", "target_length_words": 6000, "rationale": "Requires CH_02. Establishes spinor notation for CH_04–CH_12.", "status": "MISSING"},
      {"index": 4, "title": "Spin and Angular Momentum", "slug": "CH_04_SPIN_ANGULAR_MOMENTUM", "target_topic": "Connect mathematical spinor formalism to physical spin observable. Introduce spin-1/2 system.", "target_length_words": 5500, "rationale": "Requires CH_03. Establishes physical spin for CH_05+.", "status": "NOTES_ONLY"},
      {"index": 5, "title": "Pauli Matrices and Their Algebra", "slug": "CH_05_PAULI_MATRICES", "target_topic": "Develop Pauli matrix algebra, eigenvalues, commutation relations, physical interpretation.", "target_length_words": 5000, "rationale": "Requires CH_04. Establishes formalism for CH_06+.", "status": "NOTES_ONLY"},
      {"index": 6, "title": "Spin in Magnetic Fields", "slug": "CH_06_SPIN_MAGNETIC", "target_topic": "Larmor precession, Zeeman effect, spin resonance. Connect formalism to experiment.", "target_length_words": 5000, "rationale": "Requires CH_05. First experimental chapter.", "status": "MISSING"},
      {"index": 7, "title": "Higher Spin Representations", "slug": "CH_07_HIGHER_SPIN", "target_topic": "Extend from spin-1/2 to general spin-j. Clebsch-Gordan coefficients. Addition of angular momenta.", "target_length_words": 7000, "rationale": "Requires CH_04 and CH_05.", "status": "PARTIALLY_DRAFTED"},
      {"index": 8, "title": "The Dirac Equation", "slug": "CH_08_DIRAC", "target_topic": "Derive Dirac equation from relativistic requirements. Dirac spinors. Interpretation of spin from Dirac's perspective.", "target_length_words": 7000, "rationale": "Requires CH_03, CH_05. Major chapter.", "status": "MISSING"},
      {"index": 9, "title": "Spin-Statistics Theorem", "slug": "CH_09_SPIN_STATISTICS", "target_topic": "Establish spin-statistics connection. Fermions vs. bosons. Pauli exclusion principle.", "target_length_words": 5000, "rationale": "Requires CH_08.", "status": "MISSING"},
      {"index": 10, "title": "Spin in Curved Spacetime", "slug": "CH_10_CURVED_SPACETIME", "target_topic": "Spinors on curved manifolds. Vierbein formalism. Hawking radiation and spin.", "target_length_words": 6000, "rationale": "Requires CH_03, CH_08. Advanced chapter.", "status": "MISSING"},
      {"index": 11, "title": "Topological Aspects of Spin", "slug": "CH_11_TOPOLOGY", "target_topic": "Topological aspects: Berry phase, spin holonomy, Aharonov-Bohm. Fiber bundle perspective.", "target_length_words": 6000, "rationale": "Requires CH_03, CH_06. Advanced chapter.", "status": "MISSING"},
      {"index": 12, "title": "Synthesis and Open Questions", "slug": "CH_12_SYNTHESIS", "target_topic": "Synthesize the journey. Connect all threads. Identify open research questions.", "target_length_words": 4000, "rationale": "Requires all prior chapters. Closes manuscript.", "status": "MISSING"}
    ]
  }
}
```

**TRIAGE v1.1 runs Step 3.5:** Per-chapter classification confirms the above states. Aggregate assessment: ROUGH + MIXED → recommends BOOK_COMPLETION as entry workflow.

---

## Step 2: BOOK_COMPLETION engagement

Human types: `BOOK_COMPLETION`

**QUEEN reads:** BOOK_COMPLETION.json, BOOK_COMPLETION_ROUTING.md, WORKER_DRAFTER.md, manifest.

**QUEEN produces routing plan:**

| Chapter | State | Routing |
|---|---|---|
| CH_01 | NOTES_ONLY | DRAFTER → human review → STORYBOARD → EDITORIAL |
| CH_02 | NOTES_ONLY | DRAFTER → human review → STORYBOARD → EDITORIAL |
| CH_03 | MISSING | DRAFTER (scope only) → human review → STORYBOARD → EDITORIAL |
| CH_04 | NOTES_ONLY | DRAFTER → human review → STORYBOARD → EDITORIAL |
| CH_05 | NOTES_ONLY | DRAFTER → human review → STORYBOARD → EDITORIAL |
| CH_06 | MISSING | DRAFTER (scope only) → human review → STORYBOARD → EDITORIAL |
| CH_07 | PARTIALLY_DRAFTED | DRAFTER (fill gaps) → CONSOLIDATION → human review → STORYBOARD → EDITORIAL |
| CH_08 | MISSING | DRAFTER (scope only) → human review → STORYBOARD → EDITORIAL |
| CH_09 | MISSING | DRAFTER (scope only) → human review → STORYBOARD → EDITORIAL |
| CH_10 | MISSING | DRAFTER (scope only) → human review → STORYBOARD → EDITORIAL |
| CH_11 | MISSING | DRAFTER (scope only) → human review → STORYBOARD → EDITORIAL |
| CH_12 | MISSING | DRAFTER (scope only) → human review → STORYBOARD → EDITORIAL |

QUEEN reports: "12 chapters. 7 require DRAFTER (NOTES_ONLY); 5 require DRAFTER (MISSING); 1 requires DRAFTER gap-fill + CONSOLIDATION (PARTIALLY_DRAFTED). No chapters in DRAFT, CHAPTER, or POLISHED state. Initiating DRAFTER phase."

---

## Step 3: DRAFTER phase

QUEEN spawns DRAFTER sequentially for each chapter requiring DRAFTER.

**Invocation order:** CH_01 through CH_12 in index order. Each later chapter has the preceding DRAFTER-produced chapters in context for terminology consistency.

### DRAFTER — CH_01_PROBLEM_OF_ROTATION (NOTES_ONLY)

**Inputs:**
- scope entry: "Establish why classical rotation is insufficient. Motivate mathematical upgrade."
- source file: `source/intro_notes.md` + `source/rotation_draft.md`
- Templates: BUNDLE_SPIN_OF_GRAVITY (Socratic voice, physicist-teacher persona, academic-exploratory style, medium-accessible prose)
- No prior DRAFTER chapters yet

**DRAFTER process:**
1. Reads templates — internalizes Socratic voice (show-before-tell, questions before answers), physicist-teacher persona (domain authority, sharing thinking process), academic-exploratory style (inductive structure, citations as support), medium-accessible prose (moderately complex sentences, regular metaphor)
2. Reads source files — extracts: rotation intuition notes, reference to SO(3) vs SU(2) gap, draft paragraph about the "double cover" concept
3. Plans section structure: (1) The Intuition of Rotation | (2) Where Classical Description Fails | (3) Toward a Better Language | (4) What This Book Will Build
4. Drafts 5,100 words. Applies Socratic voice throughout (chapter opens with an observation about everyday rotation, not a declaration)
5. Self-check: template adherence PASS; scope adherence PASS; length 5,100 (within ±20% of 5,000 target: PASS); prerequisites PASS (no prerequisites for CH_01); consistency N/A (first chapter)
6. Gap-flags: 1 gap in §3 where notes mention a specific historical reference but don't provide bibliographic details

**Output:**
```
chapters/CH_01_PROBLEM_OF_ROTATION.md
  drafter_origin: true
  drafter_gaps: ["§3: historical reference to Cartan's original treatment needed — notes mention it but don't cite"]
  actual_length_words: 5100
```

QUEEN logs to INPROGRESS.md.

### DRAFTER — CH_02 through CH_06 (sequential)

Same pattern. Each invocation includes all prior DRAFTER-produced chapters for terminology consistency.

**Notable gap-flag density by state:**
- CH_01, CH_02, CH_04, CH_05 (NOTES_ONLY): 1–3 gap-flags each (notes provide substantial material)
- CH_03, CH_06 (MISSING): 4–7 gap-flags each (only scope available — more structural placeholders)

### DRAFTER — CH_07_HIGHER_SPIN (PARTIALLY_DRAFTED)

**Inputs:**
- scope entry for CH_07
- source file: `source/representations.md` (existing partial prose — has §§1–3 written, §4 is empty, §5 is half-written)
- Prior DRAFTER chapters CH_01–CH_06 for terminology consistency
- Templates

**DRAFTER process:**
1. Reads existing partial prose — §§1–3 are well-formed; §4 header "Clebsch-Gordan Coefficients" has no content; §5 "Physical Applications" is half-complete
2. Gap-fills §4 (~1,400 words on Clebsch-Gordan) and §5 remainder (~800 words)
3. Marks additions with `<!-- DRAFTER-ADDED -->` and `<!-- END DRAFTER-ADDED -->`
4. Sets `drafter_pass: "gap-fill"` in frontmatter
5. Sets `drafter_origin: true` (chapter contains DRAFTER-authored content)

**Output:**
```
chapters/CH_07_HIGHER_SPIN.md
  drafter_origin: true
  drafter_pass: "gap-fill"
  drafter_gaps: ["§4.3: normalization convention for CG coefficients — author should confirm sign convention used throughout"]
  actual_length_words: 6800 (2200 original + 2200 DRAFTER-added; target 7000, within ±20%)
```

### DRAFTER — CH_08 through CH_12 (MISSING)

**CH_08 (Dirac Equation):** Most technically challenging MISSING chapter. DRAFTER produces 7,200 words from scope alone. Gap-flags: 6 (3 requiring specific derivations; 3 requesting the author's interpretive perspective on contested points). DRAFTER uses terminology established in CH_03 and CH_05 for spinor notation and Pauli matrices.

**CH_09–CH_12:** Each produces 4,800–6,200 words with 3–8 gap-flags. CH_12 (synthesis) has the fewest gap-flags (3) because synthesis chapters draw on concepts already established.

### DRAFTER phase complete

QUEEN reports: "DRAFTER phase complete. 12 chapter files produced in `chapters/`. Total gap-flags: 47 across all chapters (range: 1–8 per chapter). STRUCTURE.md produced. Human review gate required."

**STRUCTURE.md state after DRAFTER phase:**
All 12 chapters have STRUCTURE.md entries with `drafter_origin: true` (all are drafter-origin since the NOTES_ONLY chapters were drafted by DRAFTER, not CONSOLIDATION).

---

## Step 4: Human review gate

QUEEN presents:
```
HUMAN REVIEW REQUIRED

12 drafter-origin chapters await review. Per DRAFTER_AUTHORSHIP_STANCE.md Safeguard 2, 
human review is mandatory before STORYBOARD engagement.

Total gap-flags: 47
  - 23 informational (suggest elaboration)
  - 24 critical (content needed to meet scope requirements)

Chapters with highest gap-flag counts:
  - CH_08_DIRAC: 6 gaps
  - CH_03_SPINORS_MATH: 7 gaps
  - CH_06_SPIN_MAGNETIC: 5 gaps

Please:
1. Review each chapter file in chapters/
2. Resolve [DRAFTER_GAP: ...] markers where you can (add material, add citations, clarify)
3. For gaps you cannot resolve now, leave the marker — EDITORIAL will treat them as Critical findings
4. When ready to proceed, trigger BOOK_CONSOLIDATION for CH_07 subset, then trigger BOOK_STORYBOARD

Invocation when ready:
  BOOK_CONSOLIDATION (for CH_07 normalization)
  chapter_subset: ["CH_07_HIGHER_SPIN"]
  source_directory: chapters/  [treating the gap-filled CH_07 as a source doc]
```

**Human action:** Reviews all 12 chapters. Resolves 15 gap-flags (adds material, citations, clarifications). Leaves 32 gap-flags as Critical findings for EDITORIAL to surface. Human triggers BOOK_CONSOLIDATION for CH_07.

---

## Step 5: BOOK_CONSOLIDATION on CH_07 subset

**Trigger:** BOOK_CONSOLIDATION with `chapter_subset: ["CH_07_HIGHER_SPIN"]`

QUEEN notes this is a targeted CONSOLIDATION run to normalize the DRAFTER gap-fill in CH_07. The SCRIBE_LOOP runs with the gap-filled CH_07 file as the sole source doc. COMPOSITOR produces a normalized chapter file. QA_COMPLETENESS and QA_COHERENCE verify no content was lost.

**Output:** Clean `chapters/CH_07_HIGHER_SPIN.md` with DRAFTER-added sections integrated. `drafter_origin: true` preserved in frontmatter (COMPOSITOR preserves frontmatter flags).

**CONSOLIDATION verdict: GREEN_LIGHT** for CH_07 subset.

---

## Step 6: BOOK_STORYBOARD

**Trigger:** BOOK_STORYBOARD with `chapter_subset: null` (full 12-chapter set)

**Inputs to STORYBOARDER:** All 12 chapter files in `chapters/`, STRUCTURE.md, BOOK_MANIFEST.json.

**STORYBOARDER runs:** Reads all 12 chapters. All are drafter-origin (or drafter-contributed for CH_07). STORYBOARDER produces STORYBOARD.md with per-chapter storyboard entries + full arc map + prerequisite chain + reader journey.

**QA_STORYBOARD:** Performs all standard checks plus drafter-origin accuracy spot-checks for all 12 chapters (since all are `drafter_origin: true`). Verifies:
- Prerequisite chain is acyclic (CH_01 → CH_02 → CH_03 etc.) — PASS
- Progressive arc builds toward synthesis — PASS
- Spot-check CH_03, CH_08, CH_12 accuracy against STORYBOARD entries — CH_08 flagged: STORYBOARD says chapter "introduces Dirac matrices" but actual prose introduces Dirac equation without formalizing the matrix notation — REVISE issued

**STORYBOARD REVISE cycle:** STORYBOARDER corrects storyboard entry for CH_08. QA_STORYBOARD re-checks — PASS.

**STORYBOARD verdict: GREEN_LIGHT**. STORYBOARD.md committed.

Human reviews STORYBOARD.md (manual handoff). Approves.

---

## Step 7: BOOK_EDITORIAL

**Trigger:** BOOK_EDITORIAL with `chapter_subset: null` (full 12-chapter set)

**Templates:** BUNDLE_SPIN_OF_GRAVITY (Socratic + physicist-teacher + academic-exploratory + medium-accessible).

**JUNIOR_EDITORIAL (4 parallel workers):**

All 4 juniors apply full audit checklists to all 12 chapters.

For all 12 chapters (`drafter_origin: true`):
- JUNIOR_CONCEPT checks for `[DRAFTER_GAP: ...]` markers — 32 remaining markers are flagged as Critical findings
- All other JUNIOR workers apply standard checklists

**Aggregate findings after JUNIOR_EDITORIAL:**
- JUNIOR_VOICE: 28 findings (High/Medium) — primarily in MISSING-state chapters where DRAFTER maintained Socratic voice but occasionally shifted into declarative mode
- JUNIOR_CONCEPT: 52 findings — 32 Critical (unresolved gap-flags) + 20 High/Medium (terminology drift between chapters, primarily CH_08–CH_12 which had no notes to anchor terminology)
- JUNIOR_STYLE: 14 findings — citation convention inconsistencies in DRAFTER-produced chapters
- JUNIOR_FLOW: 19 findings — chapter transitions at CH_06/CH_07 boundary and CH_09/CH_10 boundary are abrupt (DRAFTER chapters don't reference each other's content)
- Total: 113 findings

**EDITORIAL_SYNTHESIS:** Identifies 8 cross-axis findings — primarily voice-concept conflicts where DRAFTER's Socratic questions reference concepts not yet introduced.

**SENIOR_SANITY:** Reviews 121 integrated findings. Marks 31 as overzealous (DRAFTER-origin voice variations that are valid within the template's range). Passes 90 as real.

**SENIOR_FINAL:** Independent pass. Emits **REVISE** verdict. Notes that 32 Critical gap-flag findings must be resolved before GREEN_LIGHT. Notes that the holistic MISSING chapters (CH_08–CH_12) have higher finding density than expected — recommends human attention to CH_08 and CH_09 in particular.

**REVISION cycle 1:** REVISION addresses 58 high-priority findings (all Critical that can be resolved by rewriting prose, plus High findings). For CH_03, CH_06, CH_08 (drafter-origin, MISSING state): operates at passage-scale per DRAFTER_AUTHORSHIP_STANCE.md Safeguard 4. For CH_07 (drafter + human-authored): sentence-scale for human-authored sections, passage-scale for DRAFTER-added sections.

The 32 Critical gap-flag findings cannot be resolved by REVISION — they require human-supplied content. These are passed back to SENIOR_FINAL.

**SENIOR_FINAL round 2:** Remaining findings: 32 Critical (gap-flags awaiting human content) + 22 lower-priority. SENIOR_FINAL emits **ESCALATE** with structured report:

```
ESCALATE: 32 DRAFTER_GAP markers remain unresolved. These represent content 
that neither DRAFTER nor REVISION can supply without fabricating facts. 
Human authorial input required for:
  - CH_03: 7 gaps (spinor derivation details)
  - CH_06: 5 gaps (experimental citations)
  - CH_08: 6 gaps (Dirac matrix formalization, interpretation)
  ...
```

**Human action:** Reviews ESCALATE report. Addresses 24 of 32 gaps (supplies prose, citations, derivation sketches). 8 gaps remain (truly open research questions or material the author will supply in a future writing session).

**BOOK_COMPLETION re-invocation:** Human updates the 8 unresolvable gaps to be marked `[DRAFTER_GAP: ACKNOWLEDGED — open research question]` rather than plain gap-flags. JUNIOR_CONCEPT now treats these as Low findings (acknowledged gaps) rather than Critical. BOOK_EDITORIAL runs again (second invocation).

**EDITORIAL cycle 2:** With 24 gaps resolved and 8 acknowledged, finding count drops substantially. SENIOR_FINAL emits **REVISE** (not ESCALATE). REVISION cycle 2 addresses remaining High/Medium findings. SENIOR_FINAL cycle 3 emits **GREEN_LIGHT**.

**EDITORIAL verdict: GREEN_LIGHT**. Polished chapters committed.

---

## Step 8: BOOK_PRODUCTION

**Trigger:** BOOK_PRODUCTION

Follows standard pipeline. All 12 chapters are polished. FORMATTER validates, generates front matter (title page, copyright, preface, TOC from STRUCTURE.md) and back matter (bibliography, index). QA_PRODUCTION verifies. BOOK_SPEC.json produced. **GREEN_LIGHT**.

---

## Step 9: COMPLETION check

QUEEN re-reads manifest. All 12 chapters at POLISHED state. Issues verdict:

**VERDICT: ALL_CHAPTERS_COMPLETE**

---

## Smoke-test results (T4.5.14)

**Logical coherence check:**

| Check | Result |
|---|---|
| DRAFTER produces before STORYBOARD | PASS — DRAFTER phase precedes STORYBOARD phase |
| DRAFTER output bypasses CONSOLIDATION for NOTES_ONLY | PASS — CH_01–CH_06 go directly to STORYBOARD |
| PARTIALLY_DRAFTED chapter routes through CONSOLIDATION | PASS — CH_07 goes DRAFTER → CONSOLIDATION → STORYBOARD |
| Human review gate fires before STORYBOARD | PASS — explicit gate in Step 4 |
| `drafter_origin: true` propagates through all workflow stages | PASS — preserved by COMPOSITOR in CONSOLIDATION and flagged at EDITORIAL |
| Gap-flags treated as Critical by JUNIOR_CONCEPT | PASS — 32 Critical findings generated |
| REVISION operates at passage-scale on drafter-origin chapters | PASS — DRAFTER_AUTHORSHIP_STANCE.md Safeguard 4 invoked |
| PRODUCTION runs on full manuscript after EDITORIAL GREEN_LIGHT | PASS |
| COMPLETION verdict ALL_CHAPTERS_COMPLETE after all chapters POLISHED | PASS |
| DRAFTER sequential (not parallel) to maintain terminology consistency | PASS — CH_01 through CH_12 in order, each using prior chapters as context |

**Potential issue flagged:** ESCALATE after EDITORIAL round 1 (32 unresolved gaps) may be unexpected for an author who assumes DRAFTER handles all missing content. The ESCALATE is correct behavior — DRAFTER explicitly does not invent facts — but the author should be informed at engagement that MISSING-state chapters will likely require one human authoring session after DRAFTER runs.

**Recommendation:** BOOK_COMPLETION engagement message should explicitly warn: "N chapters are in MISSING or OUTLINE_ONLY state. DRAFTER will produce prose from scope alone. Expect a human-review session after DRAFTER completes where gap-flags are resolved."

**Smoke-test verdict: PASS** (with noted user-expectation advisory).

---

*End of COMPLETION_CASE_A_WALKTHROUGH.md*
