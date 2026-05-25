# COMPLETION_CASE_B_WALKTHROUGH

**Version:** 1.0.0
**Date:** 2026-04-18
**Tasks:** T4.5.13, T4.5.15 (walkthrough + smoke-test)
**Scenario:** Polished skeleton with unfinished body — highly mixed states

---

## Scenario definition

**Book:** 12-chapter academic-exploratory book (same title and template setup as Case A for comparability). BUNDLE_SPIN_OF_GRAVITY declared. Genre: `academic-exploratory`.

**Folder state at TRIAGE:**
- CH_01: publication-quality prose → POLISHED
- CH_02: publication-quality prose → POLISHED
- CH_03: publication-quality prose → POLISHED
- CH_04: publication-quality prose → POLISHED
- CH_05: has structured sections; §§1–3 complete, §4 half-written, §5 missing → PARTIALLY_DRAFTED
- CH_06: detailed notes and outlines → NOTES_ONLY
- CH_07: detailed notes and outlines → NOTES_ONLY
- CH_08: detailed notes and outlines → NOTES_ONLY
- CH_09: TOC entry + 1-line description only → OUTLINE_ONLY
- CH_10: TOC entry + 1-line description only → OUTLINE_ONLY
- CH_11: TOC entry + 1-line description only → OUTLINE_ONLY
- CH_12: TOC entry + 1-line description only → OUTLINE_ONLY

**Summary:** 4 POLISHED, 1 PARTIALLY_DRAFTED, 3 NOTES_ONLY, 4 OUTLINE_ONLY.

---

## Step 1: Manifest setup

**Human action:** Runs BOOK_TRIAGE v1.1. TRIAGE produces per-chapter state classifications. Human adds scope section to manifest. Existing POLISHED chapters already have structured chapter files in `chapters/` from an earlier CONSOLIDATION run (or hand-authored).

```json
"scope": {
  "scope_declared_at": "2026-04-18T09:30:00Z",
  "scope_revision_log": [],
  "intended_chapters": [
    {"index": 1, ..., "status": "POLISHED"},
    {"index": 2, ..., "status": "POLISHED"},
    {"index": 3, ..., "status": "POLISHED"},
    {"index": 4, ..., "status": "POLISHED"},
    {"index": 5, ..., "status": "PARTIALLY_DRAFTED"},
    {"index": 6, ..., "status": "NOTES_ONLY"},
    {"index": 7, ..., "status": "NOTES_ONLY"},
    {"index": 8, ..., "status": "NOTES_ONLY"},
    {"index": 9, ..., "status": "OUTLINE_ONLY"},
    {"index": 10, ..., "status": "OUTLINE_ONLY"},
    {"index": 11, ..., "status": "OUTLINE_ONLY"},
    {"index": 12, ..., "status": "OUTLINE_ONLY"}
  ]
}
```

---

## Step 2: BOOK_COMPLETION engagement

Human types: `BOOK_COMPLETION`

**QUEEN reads all required docs. Produces routing plan:**

| Chapter | State | Routing |
|---|---|---|
| CH_01 | POLISHED | EDITORIAL (audit-only) |
| CH_02 | POLISHED | EDITORIAL (audit-only) |
| CH_03 | POLISHED | EDITORIAL (audit-only) |
| CH_04 | POLISHED | EDITORIAL (audit-only) |
| CH_05 | PARTIALLY_DRAFTED | DRAFTER (fill gaps) → CONSOLIDATION → human review → STORYBOARD → EDITORIAL |
| CH_06 | NOTES_ONLY | DRAFTER → human review → STORYBOARD → EDITORIAL |
| CH_07 | NOTES_ONLY | DRAFTER → human review → STORYBOARD → EDITORIAL |
| CH_08 | NOTES_ONLY | DRAFTER → human review → STORYBOARD → EDITORIAL |
| CH_09 | OUTLINE_ONLY | DRAFTER → human review → STORYBOARD → EDITORIAL |
| CH_10 | OUTLINE_ONLY | DRAFTER → human review → STORYBOARD → EDITORIAL |
| CH_11 | OUTLINE_ONLY | DRAFTER → human review → STORYBOARD → EDITORIAL |
| CH_12 | OUTLINE_ONLY | DRAFTER → human review → STORYBOARD → EDITORIAL |

QUEEN reports: "12 chapters. 4 POLISHED (proceed directly to EDITORIAL in final pass). 1 PARTIALLY_DRAFTED (DRAFTER gap-fill + CONSOLIDATION). 3 NOTES_ONLY (DRAFTER). 4 OUTLINE_ONLY (DRAFTER, elevated gap-flag risk). Initiating DRAFTER phase for chapters 5–12."

---

## Step 3: DRAFTER phase

### DRAFTER — CH_05_PAULI_MATRICES (PARTIALLY_DRAFTED)

**Inputs:**
- scope entry for CH_05
- Existing partial prose: `source/pauli_draft.md` — §§1–3 complete; §4 "Commutation Relations" has header + 1 paragraph + trail-off; §5 "Physical Interpretation" is absent entirely
- Notes file: `source/pauli_notes.md` — has research notes on commutation algebra and Bloch sphere
- Prior DRAFTER chapters: none yet (CH_05 is first in the DRAFTER sequence; CH_01–CH_04 are human-authored and in context)
- Templates: BUNDLE_SPIN_OF_GRAVITY

**DRAFTER gap-fill:**
1. Reads existing prose. Maps: §§1–3 are complete (4,200 words). §4 is truncated at 180 words. §5 is absent.
2. Reads pauli_notes.md for §4 material (commutation details) and §5 material (Bloch sphere, spin-1/2 visualization).
3. Gap-fills §4 continuation (600 words) and §5 entirely (1,100 words).
4. Marks additions with `<!-- DRAFTER-ADDED -->` markers.
5. Total chapter: 6,080 words. Target: 5,000 (+21.6% — marginally outside ±20%). DRAFTER notes this in report and recommends author confirm that the additional length is justified by content.

**Output:**
```
chapters/CH_05_PAULI_MATRICES.md
  drafter_origin: true
  drafter_pass: "gap-fill"
  drafter_gaps: ["§4.3: author should confirm sign convention for commutator — notes show two conflicting notations"]
  actual_length_words: 6080
  target_length_words: 5000
  length_deviation: "+21.6% (outside ±20% — author should review)"
```

### DRAFTER — CH_06, CH_07, CH_08 (NOTES_ONLY)

Detailed notes available for all three. DRAFTER produces substantial first drafts.

**CH_06 (NOTES_ONLY, detailed notes):**
- Source: `source/ch06_notes.md` — extensive notes on Larmor precession, Zeeman effect, spin resonance, multiple experiments cited
- DRAFTER produces 5,200 words. 2 gap-flags (1 historical date needed, 1 derivation step needing author elaboration)
- `drafter_origin: true`, `drafter_gaps: [2 items]`

**CH_07 (NOTES_ONLY, detailed notes):**
- Source: `source/ch07_outline_notes.md` — structured outline + notes on Clebsch-Gordan
- DRAFTER produces 6,700 words. 3 gap-flags (2 notation clarifications, 1 specific example the notes mention but don't develop)
- `drafter_origin: true`, `drafter_gaps: [3 items]`

**CH_08 (NOTES_ONLY, detailed notes):**
- Source: `source/ch08_dirac_notes.md` — notes on Dirac equation derivation, comments on physical interpretation
- DRAFTER produces 7,100 words. 4 gap-flags (2 derivation steps needing expansion, 1 author's interpretive position needed, 1 connection to curved spacetime that the notes hint at)
- `drafter_origin: true`, `drafter_gaps: [4 items]`

### DRAFTER — CH_09, CH_10, CH_11, CH_12 (OUTLINE_ONLY)

Each of these chapters has only: a TOC entry title + 1-line description. DRAFTER must work from scope alone.

**CH_09 (OUTLINE_ONLY):**
- Scope entry: "Establish spin-statistics connection. Fermions vs. bosons. Pauli exclusion principle."
- TOC entry: "Chapter 9: Why Half-Integer Spins Require Antisymmetry — One sentence description"
- No source files
- DRAFTER produces 4,900 words (target 5,000, within ±20%)
- Gap-flags: 7 — OUTLINE_ONLY chapters have high gap-flag density because scope + outline alone cannot support technical derivations
- Key gaps: 2 require specific mathematical arguments (the Pauli-Lüders theorem derivation), 3 require specific physical examples, 2 request author's interpretive perspective on why the theorem is "deep"
- `drafter_origin: true`, `drafter_gaps: [7 items]`

**CH_10 (OUTLINE_ONLY):**
- Scope: "Spinors on curved manifolds. Vierbein formalism. Hawking radiation and spin."
- Target: 6,000 words. DRAFTER produces 5,800 words.
- Gap-flags: 8 — vierbein formalism requires specific notation decisions the scope doesn't specify; Hawking radiation connection requires the author's physics perspective
- `drafter_origin: true`, `drafter_gaps: [8 items]`

**CH_11 (OUTLINE_ONLY):**
- Scope: "Berry phase, spin holonomy, Aharonov-Bohm. Fiber bundle perspective."
- Target: 6,000 words. DRAFTER produces 5,600 words.
- Gap-flags: 6 — fiber bundle exposition requires the author's pedagogical choice of level (how much differential geometry to assume)
- `drafter_origin: true`, `drafter_gaps: [6 items]`

**CH_12 (OUTLINE_ONLY):**
- Scope: "Synthesize the journey. Connect all threads. Identify open research questions."
- Target: 4,000 words. DRAFTER produces 4,100 words.
- Gap-flags: 3 — synthesis chapter is more tractable from scope because DRAFTER can reference terminology from CH_01–CH_11; gaps are in identifying specific open research questions (which require the author's research perspective)
- `drafter_origin: true`, `drafter_gaps: [3 items]`

### DRAFTER phase complete

QUEEN reports: "DRAFTER phase complete. 8 chapters produced. CH_01–CH_04 untouched (POLISHED). Total gap-flags across drafter-produced chapters: 34 (1 in CH_05, 2 in CH_06, 3 in CH_07, 4 in CH_08, 7 in CH_09, 8 in CH_10, 6 in CH_11, 3 in CH_12)."

---

## Step 4: CONSOLIDATION for CH_05

**Trigger:** BOOK_CONSOLIDATION with `chapter_subset: ["CH_05_PAULI_MATRICES"]`

QUEEN routes the gap-filled CH_05 through CONSOLIDATION to normalize the merged human + DRAFTER content. SCRIBE processes the gap-filled file. COMPOSITOR produces normalized chapter. QA_COMPLETENESS verifies original §§1–3 prose preserved. **GREEN_LIGHT**.

---

## Step 5: Human review gate

QUEEN presents review requirements for CH_05–CH_12.

**Human action:** Reviews 8 drafter-origin chapters. Focus on high-gap chapters (CH_09, CH_10 with 7–8 gaps). Resolves:
- All 2 gaps in CH_06 (adds historical dates and derivation elaboration)
- 2 of 3 gaps in CH_07 (clarifies notation; defers 1 to open-question status)
- 2 of 4 gaps in CH_08 (supplies derivation steps; marks 2 as interpretation questions for later)
- 4 of 7 gaps in CH_09 (fills theorem sketch; leaves 3 as open-question markers)
- 3 of 8 gaps in CH_10 (confirms vierbein notation; leaves 5 for future authoring session)
- 3 of 6 gaps in CH_11 (fills fiber bundle intro; leaves 3)
- All 3 gaps in CH_12 (author writes brief open-questions section)
- 0 gaps in CH_05 (the 1 gap — sign convention — requires author to choose; author adds a notation clarification sentence)

Remaining unresolved gaps: 13 (across CH_08, CH_09, CH_10, CH_11). Human marks these as `[DRAFTER_GAP: ACKNOWLEDGED]`.

---

## Step 6: BOOK_STORYBOARD

**Trigger:** BOOK_STORYBOARD with `chapter_subset: null` (full 12-chapter set)

**Key challenge for STORYBOARDER:** Chapters 1–4 are POLISHED (human-authored, sophisticated prose). Chapters 5–12 are drafter-origin (varying quality). STORYBOARDER treats all 12 as input chapter files — voice-neutral description regardless of prose quality.

**STORYBOARDER produces:** STORYBOARD.md with 12 chapter entries + full arc map.

**QA_STORYBOARD:**
- Standard checks: PASS (prerequisite chain acyclic; progressive arc detected)
- DRAFTER-origin accuracy spot-checks (for CH_05–CH_12): checks 3 randomly sampled chapters (CH_07, CH_09, CH_12)
  - CH_07: storyboard entry matches chapter content — PASS
  - CH_09: storyboard entry says chapter "proves spin-statistics" — but chapter prose only sketches the argument without a full proof. QA_STORYBOARD flags: accuracy mismatch — REVISE
  - CH_12: storyboard entry accurate — PASS
- Storyboard REVISE for CH_09: STORYBOARDER corrects entry to say "introduces and motivates spin-statistics connection" rather than "proves." QA_STORYBOARD re-checks — PASS.

**STORYBOARD verdict: GREEN_LIGHT**. Human reviews. Approves.

---

## Step 7: BOOK_EDITORIAL

**Trigger:** BOOK_EDITORIAL with `chapter_subset: null` (full 12-chapter set, all in one pass)

**Template loading:** BUNDLE_SPIN_OF_GRAVITY loaded. Compatibility: bundle mode, no matrix check needed.

**JUNIOR_EDITORIAL (4 parallel workers on all 12 chapters):**

**Distinct behavior for CH_01–CH_04 (POLISHED, human-authored):**
- Full audit checklist applies
- `drafter_origin` flag absent → sentence-scale REVISION if needed
- JUNIOR_CONCEPT checks for gap markers: none present → no Critical findings
- Expected findings: Low/Medium (POLISHED chapters already meet most template requirements)

**Distinct behavior for CH_05–CH_12 (drafter-origin):**
- Full audit checklist applies
- JUNIOR_CONCEPT checks for `[DRAFTER_GAP: ACKNOWLEDGED]` markers → treated as Low findings (not Critical since acknowledged)
- Enhanced scrutiny applied per DRAFTER_AUTHORSHIP_STANCE.md Safeguard 3
- EDITORIAL_SYNTHESIS notes higher cross-axis risk for drafter-origin chapters

**Aggregate JUNIOR findings:**
- CH_01–CH_04: 12 findings total (mostly Low/Medium — these are POLISHED chapters already reviewed historically)
- CH_05–CH_08: 44 findings (NOTES_ONLY/PARTIALLY_DRAFTED with good source material — voice is mostly right but concept consistency and flow issues)
- CH_09–CH_12: 67 findings (OUTLINE_ONLY chapters show more voice breaks, more concept-flow issues — DRAFTER had least material to work with)
- Total: 123 findings

**EDITORIAL_SYNTHESIS:** 11 cross-axis findings, notably:
- Voice-concept conflict across CH_04/CH_05 boundary: CH_04 (POLISHED) introduces a concept using one voice pattern; CH_05 (drafter-origin) assumes the concept without the voice pattern that established it → the transition loses the Socratic arc
- Flow inconsistency between CH_08 and CH_09: CH_08 (NOTES_ONLY) ends with open questions that CH_09 (OUTLINE_ONLY) doesn't acknowledge → reader journey disrupted

**SENIOR_SANITY:** Reviews 134 findings. Marks 22 overzealous. Passes 112 as real.

**SENIOR_FINAL:** Independent pass. Notes that CH_01–CH_04 are high quality; CH_05–CH_12 have varying quality. Emits **REVISE**.

**REVISION cycle 1 (passage-scale for CH_05–CH_12, sentence-scale for CH_01–CH_04):**
- Addresses 80 findings (all Critical/High; some Medium)
- For CH_01–CH_04: minimal edits (these are POLISHED — REVISION is conservative)
- For CH_09–CH_12: passage-scale revisions where DRAFTER's voice broke; REVISION rewrites paragraph blocks to restore Socratic arc
- 32 Low findings deferred (ACKNOWLEDGED gap markers)
- CH_04/CH_05 boundary: REVISION adds bridge paragraph to CH_05 opening that restores voice continuity from CH_04

**SENIOR_FINAL cycle 2:** Reviews revised manuscript. 32 acknowledged gap markers at Low severity. 14 remaining Medium findings. 8 Low findings. Emits **REVISE** (residual Medium findings must be addressed).

**REVISION cycle 2:** Addresses 14 Medium findings. Passes on 8 Low + 32 ACKNOWLEDGED.

**SENIOR_FINAL cycle 3:** 32 ACKNOWLEDGED gap markers (Low), 8 Low findings. Makes judgment call: these are below the threshold for blocking GREEN_LIGHT. Emits **GREEN_LIGHT**.

**EDITORIAL verdict: GREEN_LIGHT**. Polished chapters committed.

---

## Step 8: BOOK_PRODUCTION

Standard pipeline. All 12 chapters polished (4 POLISHED originally + 8 drafter-origin now editorially-greenlit). BOOK_SPEC.json produced. **GREEN_LIGHT**.

---

## Step 9: COMPLETION check

All 12 chapters at POLISHED state. **VERDICT: ALL_CHAPTERS_COMPLETE.**

---

## Per-chapter summary table

| Chapter | Initial State | Workflows Run | drafter_origin | Gap-flags Initial | Gap-flags Resolved | EDITORIAL cycles |
|---|---|---|---|---|---|---|
| CH_01 | POLISHED | EDITORIAL | false | 0 | — | 1 (GREEN_LIGHT) |
| CH_02 | POLISHED | EDITORIAL | false | 0 | — | 1 |
| CH_03 | POLISHED | EDITORIAL | false | 0 | — | 1 |
| CH_04 | POLISHED | EDITORIAL | false | 0 | — | 1 |
| CH_05 | PARTIALLY_DRAFTED | DRAFTER → CONSOLIDATION → STORYBOARD → EDITORIAL | true | 1 | 1 | 3 |
| CH_06 | NOTES_ONLY | DRAFTER → STORYBOARD → EDITORIAL | true | 2 | 2 | 3 |
| CH_07 | NOTES_ONLY | DRAFTER → STORYBOARD → EDITORIAL | true | 3 | 2 | 3 |
| CH_08 | NOTES_ONLY | DRAFTER → STORYBOARD → EDITORIAL | true | 4 | 2 | 3 |
| CH_09 | OUTLINE_ONLY | DRAFTER → STORYBOARD → EDITORIAL | true | 7 | 4 | 3 |
| CH_10 | OUTLINE_ONLY | DRAFTER → STORYBOARD → EDITORIAL | true | 8 | 3 | 3 |
| CH_11 | OUTLINE_ONLY | DRAFTER → STORYBOARD → EDITORIAL | true | 6 | 3 | 3 |
| CH_12 | OUTLINE_ONLY | DRAFTER → STORYBOARD → EDITORIAL | true | 3 | 3 | 3 |

---

## Key observations — Case B specific

### Observation 1: POLISHED chapters don't slow DRAFTER phase

CH_01–CH_04 are ready when BOOK_COMPLETION engages. DRAFTER is not invoked for them. The only overhead for POLISHED chapters is the EDITORIAL audit-only pass — which is fast (few findings expected).

### Observation 2: NOTES_ONLY vs OUTLINE_ONLY gap-flag difference is material

CH_06–CH_08 (NOTES_ONLY with detailed outlines): 2–4 gap-flags each.
CH_09–CH_12 (OUTLINE_ONLY): 3–8 gap-flags each.

The difference is real and expected. Notes give DRAFTER factual content to draw from. An outline entry gives only a 1-line description. OUTLINE_ONLY chapters are inherently more reliant on the human for gap resolution.

### Observation 3: The CH_04/CH_05 boundary is the highest-risk handoff

This boundary separates POLISHED human-authored prose (CH_04) from drafter-origin prose (CH_05). DRAFTER must match the voice and prose density of CH_04 when opening CH_05. This is Case B's structurally hardest REVISION target — the reader is accustomed to polished Socratic voice, then encounters DRAFTER's approximation of it.

REVISION's passage-scale authorization for CH_05 (drafter-origin) is critical here. Sentence-scale REVISION would not be sufficient to restore the voice continuity.

### Observation 4: EDITORIAL subsetting could be used for efficiency

An advanced Case B optimization: run EDITORIAL on CH_01–CH_04 first (quick audit-only, likely 1 cycle), then run EDITORIAL on CH_05–CH_12 (full audit, 3 cycles). Merge results. This avoids the POLISHED chapters slowing down the OUTLINE_ONLY chapter editorial cycles.

However, full-manuscript EDITORIAL is recommended for the final pass to detect cross-boundary issues (like the CH_04/CH_05 voice discontinuity, which requires both chapters in context to detect). Use subset EDITORIAL for iterative passes, null subset for the final pass.

---

## Smoke-test results (T4.5.15)

| Check | Result |
|---|---|
| POLISHED chapters route to EDITORIAL (audit-only), not DRAFTER | PASS |
| PARTIALLY_DRAFTED routes through DRAFTER → CONSOLIDATION | PASS |
| NOTES_ONLY routes through DRAFTER → STORYBOARD (bypass CONSOLIDATION) | PASS |
| OUTLINE_ONLY routes same as NOTES_ONLY | PASS |
| Human review gate fires between DRAFTER phase and STORYBOARD | PASS |
| QA_STORYBOARD applies accuracy spot-check on drafter-origin chapters | PASS (CH_09 storyboard mismatch caught) |
| EDITORIAL applies enhanced scrutiny to CH_05–CH_12 | PASS |
| EDITORIAL applies standard scrutiny to CH_01–CH_04 | PASS |
| REVISION uses passage-scale for drafter-origin chapters | PASS (CH_09–CH_12) |
| REVISION uses sentence-scale for POLISHED chapters | PASS (CH_01–CH_04) |
| ACKNOWLEDGED gap markers treated as Low (not Critical) | PASS |
| CH_04/CH_05 boundary voice discontinuity detected by EDITORIAL_SYNTHESIS | PASS |
| ALL_CHAPTERS_COMPLETE verdict issued after EDITORIAL GREEN_LIGHT | PASS |
| Case B produces coherent execution plan with no workflow contradictions | PASS |

**Potential issue flagged:** CH_05's word count (+21.6% over target) is marginally outside the ±20% constraint. DRAFTER correctly reports this and flags it for author review. CONSOLIDATION does not trim the chapter (COMPOSITOR preserves content). EDITORIAL does not shorten it (REVISION's job is voice/concept/style, not length). The author must decide whether to trim or revise the target_length_words. This is expected behavior — a flag, not a failure.

**Smoke-test verdict: PASS** (with noted length-overage advisory for CH_05).

---

*End of COMPLETION_CASE_B_WALKTHROUGH.md*
