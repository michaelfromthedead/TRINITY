# CHAPTER_STATE_TAXONOMY

**Version:** 1.0.0
**Date:** 2026-04-18
**Task:** T4.5.3
**Covers:** Per-chapter state label vocabulary for BOOK_TRIAGE v1.1 and BOOK_COMPLETION routing

---

## Purpose

This taxonomy defines the vocabulary of per-chapter states used throughout the BOOK workflow family when operating in mixed-state mode (TRIAGE v1.1, BOOK_COMPLETION). Each state describes how much usable material exists for a given intended chapter relative to the finished-chapter standard.

States are assigned by BOOK_TRIAGE v1.1 based on source file analysis matched against the manifest `scope.intended_chapters` list. States may be manually overridden by the human in `BOOK_MANIFEST.json`. BOOK_COMPLETION reads states and routes each chapter to the appropriate workflow(s).

---

## 1. State Definitions

### 1.1 `MISSING`

**Definition:** No source material exists for this intended chapter. The chapter appears in `scope.intended_chapters` but there are zero source files in the project that correspond to its topic.

**Distinguishing criteria:**
- No source file has content matching the chapter's `target_topic`
- No file is named or titled in a way that suggests association with this chapter
- No excerpts, notes, or references to this chapter's subject matter found in any source doc

**Implications:**
- DRAFTER must author from scope alone (the `target_topic` and `rationale` fields are the sole inputs)
- DRAFTER will produce the most gap-flag-dense output of any state
- Human review after DRAFTER is especially critical for MISSING-state chapters

**Routing:** DRAFTER → human review gate → STORYBOARD → EDITORIAL → PRODUCTION

---

### 1.2 `OUTLINE_ONLY`

**Definition:** A TOC entry or chapter listing exists with at most one paragraph of description, but no prose material, no notes, and no research content.

**Distinguishing criteria:**
- Chapter appears in an existing TOC, table of contents draft, or structural document
- The entry has a title and possibly a 1–2 sentence description
- No elaboration beyond that exists in any source file

**Distinguishing from MISSING:** OUTLINE_ONLY has explicit authorial acknowledgment that this chapter exists in the intended structure. MISSING has no such acknowledgment — COMPLETION infers the chapter only from the manifest `scope` declaration.

**Implications:**
- DRAFTER inputs: `target_topic` + `rationale` from scope, plus the outline entry (title + description)
- Slightly more context than MISSING; the outline entry may constrain the chapter's arc
- Still expected to produce significant gap-flags

**Routing:** DRAFTER → human review gate → STORYBOARD → EDITORIAL → PRODUCTION

---

### 1.3 `NOTES_ONLY`

**Definition:** Research notes, fragments, or rough material related to this chapter exist, but no structured prose. The notes may be substantial in volume but have no chapter-shaped organization.

**Distinguishing criteria:**
- Source files contain content clearly associated with the chapter's topic
- Content is note-like: bullet points, raw observations, research excerpts, draft sentences, reference citations, equations without prose context
- No coherent section structure or narrative thread exists within the material

**Distinguishing from PARTIALLY_DRAFTED:** NOTES_ONLY has no prose structure at all — just raw material. PARTIALLY_DRAFTED has prose that is recognizably structured but incomplete (sections exist but some are empty or cut off).

**Implications:**
- DRAFTER inputs: `target_topic` + `rationale` + all associated notes files
- This is DRAFTER's most common and most productive input state
- Notes give DRAFTER factual content and conceptual vocabulary to draw on
- DRAFTER still determines prose structure from scratch; the notes are raw material

**Routing:** DRAFTER → human review gate → STORYBOARD (not CONSOLIDATION — DRAFTER output is chapter-shaped) → EDITORIAL → PRODUCTION

---

### 1.4 `PARTIALLY_DRAFTED`

**Definition:** Prose exists for this chapter but is incomplete. Structured sections are present; some sections have content, others are empty or cut off mid-thought. The chapter has recognizable prose shape but does not reach a natural conclusion.

**Distinguishing criteria:**
- Markdown headers or section structure exists
- At least some sections have written prose (not just notes)
- Other sections have headers with no content, `[TODO]` markers, or obviously unfinished paragraphs
- The word count is substantially below the `target_length_words`

**Implications:**
- DRAFTER fills the gaps in existing prose
- DRAFTER must preserve and integrate with existing prose rather than authoring from scratch
- DRAFTER-filled gaps must be tagged with `[DRAFTER_GAP: ...]` markers (§2 below)
- The existing prose portions are NOT drafter-origin; only the DRAFTER-filled portions are
- The chapter as a whole receives `drafter_origin: true` flag because it contains DRAFTER-authored content
- CONSOLIDATION runs after DRAFTER to normalize the merged document before STORYBOARD

**Routing:** DRAFTER (fill gaps) → CONSOLIDATION (normalize merged chapter) → human review gate → STORYBOARD → EDITORIAL → PRODUCTION

---

### 1.5 `DRAFT`

**Definition:** Prose is complete but rough — all sections have content, the chapter reads as a coherent unit, but prose quality is below publication standard. This is the v1.0.0 TRIAGE label, preserved here for continuity.

**Distinguishing criteria:**
- All sections have content; no empty headers
- The chapter is readable end-to-end
- Prose is rough: unclear passages, inconsistent terminology, voice not yet applied, structural awkwardness
- Word count is near the target range

**Implications:**
- No DRAFTER involvement — CONSOLIDATION processes this chapter as source material
- QA_COMPLETENESS + QA_COHERENCE apply
- Standard pipeline: CONSOLIDATION → STORYBOARD → EDITORIAL → PRODUCTION

**Routing:** CONSOLIDATION → STORYBOARD → EDITORIAL → PRODUCTION

---

### 1.6 `CHAPTER`

**Definition:** Structured and substantial — the chapter has coherent prose, appropriate length, and logical section organization. May have voice/style inconsistencies but the intellectual content is solid. This is the mid-maturity v1.0.0 TRIAGE label.

**Distinguishing criteria:**
- Chapter has a clear argument or explanatory arc
- Sections are well-organized and substantiated
- Prose is clear even if not polished
- Word count meets or approaches the target

**Implications:**
- CONSOLIDATION passes may still be useful to extract best-of phrasing if multiple versions exist
- If source material has a single CHAPTER-state file: CONSOLIDATION may be skipped and STORYBOARD entered directly
- EDITORIAL applies full review

**Routing:** CONSOLIDATION (if multiple source versions exist) or STORYBOARD (if single source) → EDITORIAL → PRODUCTION

---

### 1.7 `POLISHED`

**Definition:** Publication-ready or near publication-ready. Prose is complete, voice is consistent, structure is sound. This is the highest v1.0.0 TRIAGE label.

**Distinguishing criteria:**
- Chapter reads as if it could appear in a published book
- Voice is applied consistently
- No structural issues, no placeholder text, no obvious gaps

**Implications:**
- EDITORIAL in audit-only mode — likely to GREEN_LIGHT without REVISE loops
- May proceed directly to PRODUCTION after EDITORIAL
- No CONSOLIDATION or STORYBOARD needed if the chapter is independently polished (though the full-book storyboard still runs to verify arc coherence)

**Routing:** EDITORIAL (audit-only, no revision expected) → PRODUCTION

---

## 2. DRAFTER_ORIGIN Flag

**`DRAFTER_ORIGIN` is a flag, not a state.** It is orthogonal to the state taxonomy above. A chapter has the DRAFTER_ORIGIN flag set when any portion of its prose was authored by DRAFTER.

**Where stored:** In the chapter file's YAML frontmatter: `drafter_origin: true`

**Which states can produce drafter-origin chapters:**
- `MISSING` — entire chapter authored by DRAFTER → drafter_origin: true
- `OUTLINE_ONLY` — entire chapter authored by DRAFTER → drafter_origin: true
- `NOTES_ONLY` — entire prose structure authored by DRAFTER (notes are inputs) → drafter_origin: true
- `PARTIALLY_DRAFTED` — gap-filled by DRAFTER → drafter_origin: true (even though some prose is human-authored)

**Which states do NOT produce drafter-origin chapters:**
- `DRAFT` — human-authored, CONSOLIDATION processed → drafter_origin: false (or absent)
- `CHAPTER` — human-authored → drafter_origin: false (or absent)
- `POLISHED` — human-authored → drafter_origin: false (or absent)

**Editorial consequences:** See `DRAFTER_AUTHORSHIP_STANCE.md` §"Safeguard 3 — Enhanced editorial scrutiny" and §"Safeguard 4 — Revision looser scope."

---

## 3. Transition Rules

The following transitions are valid — i.e., a chapter can move between states as work progresses. Transitions move forward (toward higher maturity); backward transitions are only possible via explicit rollback (human removes material).

```
MISSING
  ↓ (human adds notes or outline)
OUTLINE_ONLY   or   NOTES_ONLY
  ↓ (DRAFTER runs)
DRAFT or PARTIALLY_DRAFTED → DRAFT (after DRAFTER fills gaps)
  ↓ (CONSOLIDATION runs)
CHAPTER
  ↓ (EDITORIAL runs and GREEN_LIGHTs)
POLISHED
```

Specific valid transitions:

| From | To | Trigger |
|---|---|---|
| MISSING | OUTLINE_ONLY | Human adds a TOC entry or chapter description |
| MISSING | NOTES_ONLY | Human adds research notes |
| MISSING | DRAFT | DRAFTER produces chapter (enters pipeline as DRAFT after DRAFTER) |
| OUTLINE_ONLY | NOTES_ONLY | Human adds substantive notes beyond the outline |
| OUTLINE_ONLY | DRAFT | DRAFTER produces chapter |
| NOTES_ONLY | DRAFT | DRAFTER produces chapter |
| NOTES_ONLY | PARTIALLY_DRAFTED | Human writes some sections; others remain notes |
| PARTIALLY_DRAFTED | DRAFT | DRAFTER fills gaps OR human completes remaining sections |
| DRAFT | CHAPTER | CONSOLIDATION processes and carves |
| CHAPTER | POLISHED | EDITORIAL GREEN_LIGHTs the chapter |
| Any | DRAFTER_ORIGIN flag | DRAFTER produces or contributes prose |

---

## 4. State detection heuristics for TRIAGE v1.1

When BOOK_TRIAGE v1.1 runs per-chapter classification (Step 3.5), it uses these heuristics:

| Indicator | Likely state |
|---|---|
| No source files match chapter topic | MISSING |
| Only a TOC entry or 1-sentence description exists | OUTLINE_ONLY |
| Bullet points, raw notes, research fragments; no structure | NOTES_ONLY |
| Markdown headers + some content + obvious gaps | PARTIALLY_DRAFTED |
| All sections have prose; rough quality | DRAFT |
| Well-organized, coherent argument, decent prose | CHAPTER |
| Publication-quality, voice-consistent, complete | POLISHED |

TRIAGE assigns a `confidence` field (high / medium / low) to each classification. Low-confidence assignments warrant human review before COMPLETION routing begins.

---

## 5. Interaction with BOOK_COMPLETION routing

BOOK_COMPLETION reads the `detected_state` from `triage.per_chapter_state` for each intended chapter and routes accordingly. The full routing table is in `BOOK_COMPLETION_ROUTING.md`.

Summary:

| State | Primary next workflow |
|---|---|
| MISSING | DRAFTER |
| OUTLINE_ONLY | DRAFTER |
| NOTES_ONLY | DRAFTER → STORYBOARD |
| PARTIALLY_DRAFTED | DRAFTER → CONSOLIDATION → STORYBOARD |
| DRAFT | CONSOLIDATION |
| CHAPTER | STORYBOARD or EDITORIAL |
| POLISHED | EDITORIAL (audit-only) |

---

*End of CHAPTER_STATE_TAXONOMY.md*
