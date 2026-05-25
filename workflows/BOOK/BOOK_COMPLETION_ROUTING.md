# BOOK_COMPLETION_ROUTING

**Version:** 1.0.0
**Date:** 2026-04-18
**Tasks:** T4.5.6, T4.5.10
**Consumed by:** BOOK_COMPLETION.json (QUEEN at engagement)

---

## Purpose

This document is the authoritative state-to-workflow routing table for BOOK_COMPLETION. For each chapter state in `CHAPTER_STATE_TAXONOMY.md`, it specifies which workflow(s) apply, in what order, with what inputs, and what output the chapter produces.

QUEEN reads this document at BOOK_COMPLETION engagement and uses it to produce the per-chapter routing plan.

---

## 1. Routing Table

| Chapter State | Workflow Sequence | Notes |
|---|---|---|
| `MISSING` | DRAFTER → human review gate → STORYBOARD → EDITORIAL | Scope alone. Most gap-flags expected. |
| `OUTLINE_ONLY` | DRAFTER → human review gate → STORYBOARD → EDITORIAL | Scope + outline entry as inputs. |
| `NOTES_ONLY` | DRAFTER → human review gate → STORYBOARD | Skip CONSOLIDATION — see §3. |
| `PARTIALLY_DRAFTED` | DRAFTER (fill gaps) → CONSOLIDATION → human review gate → STORYBOARD → EDITORIAL | CONSOLIDATION normalizes merged doc. |
| `DRAFT` | CONSOLIDATION → STORYBOARD → EDITORIAL | Standard pipeline, no DRAFTER. |
| `CHAPTER` | STORYBOARD (if not yet storyboarded) or EDITORIAL (if storyboarded) | Human judgment on entry point. |
| `POLISHED` | EDITORIAL (audit-only) | No revision expected; may GREEN_LIGHT immediately. |

---

## 2. Detailed Routing Per State

### 2.1 `MISSING` → DRAFTER → human review → STORYBOARD → EDITORIAL

**Trigger condition:** `triage.per_chapter_state[i].detected_state == "MISSING"`

**DRAFTER inputs:**
- `scope.intended_chapters[i]` — `target_topic`, `rationale`, `target_length_words`, `slug`
- No source files (none associated)
- Resolved templates from BOOK_MANIFEST.json
- STORYBOARD.md if it exists (for prerequisite-chain awareness when drafting later chapters)

**DRAFTER task:** Author a complete chapter from scope alone. Produce `chapters/CH_<NN>_<TITLE>.md` with `drafter_origin: true` and `drafter_gaps: [...]`.

**Expected gap-flag density:** High. DRAFTER will produce more `[DRAFTER_GAP: ...]` markers for MISSING-state chapters than for any other state, because it has no source material to draw on for specifics.

**Human review gate:** Mandatory (Safeguard 2 in DRAFTER_AUTHORSHIP_STANCE.md). Human inspects chapter, resolves gap-flags where possible, may add material before STORYBOARD.

**STORYBOARD invocation:** `chapter_subset: ["CH_<NN>_<TITLE>"]` — subset invocation if only this chapter is new; full invocation if this is part of a batch DRAFTER run.

**EDITORIAL invocation:** All drafter-origin chapters receive full junior audit + enhanced scrutiny. No subset invocation at EDITORIAL unless the human specifically requests a partial pass.

**Output frontmatter:**
```yaml
drafter_origin: true
drafter_gaps: ["gap description 1", "gap description 2"]
target_topic: "..."
target_length_words: 5000
```

---

### 2.2 `OUTLINE_ONLY` → DRAFTER → human review → STORYBOARD → EDITORIAL

**Trigger condition:** `triage.per_chapter_state[i].detected_state == "OUTLINE_ONLY"`

**DRAFTER inputs:**
- `scope.intended_chapters[i]`
- The outline entry source file (TOC doc or structural doc containing the 1-2 sentence description)
- Resolved templates
- STORYBOARD.md if exists

**DRAFTER task:** Author a complete chapter from scope + the outline's description. The outline may constrain the chapter's framing but not its depth.

**Expected gap-flag density:** High (slightly lower than MISSING due to outline entry providing framing context).

**Otherwise identical to MISSING routing.**

---

### 2.3 `NOTES_ONLY` → DRAFTER → human review → STORYBOARD (skip CONSOLIDATION)

**Trigger condition:** `triage.per_chapter_state[i].detected_state == "NOTES_ONLY"`

**DRAFTER inputs:**
- `scope.intended_chapters[i]`
- All associated notes source files from `triage.per_chapter_state[i].source_files`
- Resolved templates
- STORYBOARD.md if exists

**DRAFTER task:** Author a complete chapter from scope + notes. Notes provide factual content and conceptual vocabulary. DRAFTER determines prose structure from scratch.

**Expected gap-flag density:** Medium. Notes give DRAFTER substance to draw on; gap-flags occur where notes are thin on a specific sub-topic.

**Why skip CONSOLIDATION (T4.5.10 resolution):**

DRAFTER produces chapter-shaped output directly — a well-formed `chapters/CH_<NN>_<TITLE>.md` matching the COMPOSITOR output format. CONSOLIDATION's purpose is to collapse chaotic source material into MASTER and then carve it into chapters. Since DRAFTER has already done the equivalent of CONSOLIDATION + carving in a single pass (notes in, chapter out), running CONSOLIDATION on the result would:
1. Treat a chapter file as a source document (wrong — it is already carved output)
2. Re-consolidate prose that is already structured (wasteful and potentially corrupting)
3. Run QA_COMPLETENESS against the chapter's own content (circular)

**Therefore: NOTES_ONLY chapters produced by DRAFTER go directly to STORYBOARD, not CONSOLIDATION.** This is a hard rule (see BOOK_COMPLETION.json `hard_rules.drafter_output_bypasses_consolidation`).

**STORYBOARD awareness of drafter-origin:** STORYBOARD's QA_STORYBOARD checks `drafter_origin: true` in frontmatter and applies extra accuracy spot-checks — verifying that DRAFTER's prose actually delivers what the storyboard entry says the chapter does. This catches drift between DRAFTER's authorial choices and the chapter's intended arc role.

---

### 2.4 `PARTIALLY_DRAFTED` → DRAFTER (fill gaps) → CONSOLIDATION → human review → STORYBOARD → EDITORIAL

**Trigger condition:** `triage.per_chapter_state[i].detected_state == "PARTIALLY_DRAFTED"`

**Phase 1 — DRAFTER (gap-filling):**

DRAFTER inputs:
- `scope.intended_chapters[i]`
- The existing partial prose file(s)
- All associated notes source files
- Resolved templates

DRAFTER task: Read the existing partial prose. Fill the gaps (empty sections, cut-off paragraphs, `[TODO]` markers). Produce a complete chapter draft by merging existing prose with DRAFTER-authored fill material.

DRAFTER marks its added sections with inline `[DRAFTER_GAP: reason]` markers where material is thin. DRAFTER does NOT rewrite existing prose — only fills gaps.

Output: updated chapter file with `drafter_origin: true` (because it contains DRAFTER-authored content, even though some sections are human-authored).

**Why CONSOLIDATION runs here (unlike NOTES_ONLY):**

PARTIALLY_DRAFTED chapters exist as partially structured files. DRAFTER's gap-filling produces a hybrid document: some sections are human-authored prose, some are DRAFTER-authored fills. CONSOLIDATION normalizes this hybrid — it treats the gap-filled chapter as a source doc and produces a clean structured output (going through SCRIBE → COMPOSITOR). This ensures the hybrid content is coherent and that QA_COMPLETENESS checks that no existing content was lost during the gap-fill.

This is the key distinction:
- NOTES_ONLY: DRAFTER structures everything → output is clean → CONSOLIDATION not needed
- PARTIALLY_DRAFTED: DRAFTER merges with existing structure → output may need normalization → CONSOLIDATION runs

**Phase 2 — CONSOLIDATION (with chapter_subset):**

CONSOLIDATION is invoked with:
```
chapter_subset: ["CH_<NN>_<TITLE>"]
```

The DRAFTER-produced gap-filled chapter file is the source document for this CONSOLIDATION run. COMPOSITOR carves it into the final chapter file. QA_COMPLETENESS verifies original partial prose was preserved.

**Human review gate:** After CONSOLIDATION, before STORYBOARD. Human reviews the normalized chapter. drafter-origin sections are still marked.

---

### 2.5 `DRAFT` → CONSOLIDATION → STORYBOARD → EDITORIAL

**Trigger condition:** `triage.per_chapter_state[i].detected_state == "DRAFT"`

**Standard pipeline.** No DRAFTER involvement.

CONSOLIDATION is invoked with:
```
chapter_subset: ["CH_<NN>_<TITLE>", ...]  (all DRAFT-state chapters as a batch)
```

DRAFT-state chapters are processed together in a single CONSOLIDATION run (SCRIBE_LOOP handles multiple source docs). COMPOSITOR carves all DRAFT chapters together, which allows it to discover optimal chapter boundaries across the set.

After CONSOLIDATION, these chapters flow to STORYBOARD and EDITORIAL as normal. No drafter-origin flag. No enhanced editorial scrutiny beyond standard.

---

### 2.6 `CHAPTER` → STORYBOARD or EDITORIAL

**Trigger condition:** `triage.per_chapter_state[i].detected_state == "CHAPTER"`

**Entry point depends on whether STORYBOARD.md already exists:**

- If no STORYBOARD.md exists: route to STORYBOARD (with `chapter_subset` if other chapters are at different stages)
- If STORYBOARD.md exists for this chapter: route directly to EDITORIAL

QUEEN presents this choice in the routing plan. Human may override.

No DRAFTER involvement. No CONSOLIDATION if the chapter is already well-structured as a single chapter file.

---

### 2.7 `POLISHED` → EDITORIAL (audit-only)

**Trigger condition:** `triage.per_chapter_state[i].detected_state == "POLISHED"`

**Lightest path.** EDITORIAL runs in audit-only mode on these chapters — all four JUNIOR workers run their checklists, but SENIOR_FINAL is expected to GREEN_LIGHT without requiring a REVISE cycle.

If POLISHED chapters are mixed with non-POLISHED chapters in the same EDITORIAL invocation:
- Juniors apply full audit to all chapters (including POLISHED ones — no exemptions)
- SENIOR_FINAL may note that findings in POLISHED chapters are typically lower-severity
- REVISION, if triggered, handles POLISHED chapters at standard sentence-scale (not passage-scale — these are human-authored)

---

## 3. DRAFTER → STORYBOARD Direct Handoff (T4.5.10)

This is the canonical route for NOTES_ONLY and MISSING-state chapters after DRAFTER runs:

```
DRAFTER produces: chapters/CH_<NN>_<TITLE>.md (drafter_origin: true)
         ↓
         [human review gate]
         ↓
BOOK_STORYBOARD reads: chapters/ (including drafter-origin chapters)
QA_STORYBOARD checks: drafter_origin: true → applies accuracy spot-check
         ↓
BOOK_EDITORIAL (all drafter-origin chapters: enhanced scrutiny)
```

**STORYBOARD receives drafter-origin chapter files as if they were COMPOSITOR-produced chapters** — same file format, same directory, same STRUCTURE.md registration. From STORYBOARD's perspective, there is no structural difference. The `drafter_origin: true` flag tells QA_STORYBOARD to spot-check this chapter's content against the storyboard entry it produces.

**STORYBOARD's QA_STORYBOARD additional check for drafter-origin chapters:** After producing the storyboard entry for a drafter-origin chapter, QA_STORYBOARD verifies that the chapter's actual prose delivers what the storyboard entry claims the chapter does. This catches cases where DRAFTER's content drifted from the `target_topic` or `rationale` the storyboard is based on.

---

## 4. Batch vs. Per-Chapter Invocation

QUEEN makes a practical decision about whether to batch chapters:

**Batch DRAFTER invocations sequentially per chapter.** DRAFTER cannot run chapters in parallel because later chapters may reference terminology established in earlier chapters. Each DRAFTER run for a later chapter should be provided with the chapters already produced by earlier DRAFTER runs (as additional context for terminology consistency).

**Batch CONSOLIDATION as a single invocation.** All DRAFT-state chapters can be processed in a single BOOK_CONSOLIDATION run with a chapter_subset that covers all of them. This is more efficient and allows COMPOSITOR to discover optimal inter-chapter structure within the subset.

**STORYBOARD and EDITORIAL run once on the full chapter set** (or on a subset if specifically requested). Running these workflows on subsets and then re-running on the full set would be wasteful and could produce incoherent storyboard entries (storyboard needs to see all chapters to establish the full arc).

---

## 5. Edge Cases

### 5.1 Chapter changes state between COMPLETION invocations

If a chapter was classified `NOTES_ONLY` at first TRIAGE, then the human adds a partial prose draft before re-running TRIAGE:

- TRIAGE v1.1 re-runs per-chapter classification → updates `detected_state` to `PARTIALLY_DRAFTED`
- BOOK_COMPLETION re-invocation reads the updated state → routes chapter to DRAFTER (fill gaps) → CONSOLIDATION
- Prior DRAFTER run (if it happened under NOTES_ONLY) is discarded; new DRAFTER run uses updated source material

COMPLETION is designed to handle state evolution. Re-read manifest at each invocation.

### 5.2 Human resolves all gap-flags before STORYBOARD

If the human reviews drafter-origin chapters and manually fills all `[DRAFTER_GAP: ...]` markers with prose of their own:
- The chapter still retains `drafter_origin: true` in frontmatter (mixed authorship)
- EDITORIAL still applies enhanced scrutiny (some DRAFTER-authored sections remain even if gaps are filled)
- JUNIOR_CONCEPT does not flag gap markers as Critical because they no longer exist

### 5.3 All chapters are POLISHED at COMPLETION engagement

If TRIAGE reports all chapters as `POLISHED`:
- BOOK_COMPLETION skips DRAFTER, CONSOLIDATION, and STORYBOARD phases
- QUEEN routes directly to: EDITORIAL (audit-only on full set) → PRODUCTION
- Verdict: ALL_CHAPTERS_COMPLETE upon EDITORIAL GREEN_LIGHT

### 5.4 Mixed DRAFT and NOTES_ONLY for the same chapter

TRIAGE may be uncertain whether a chapter is DRAFT or NOTES_ONLY when source material is borderline. In this case:
- TRIAGE assigns confidence: low and notes the ambiguity
- QUEEN presents the ambiguity in the routing plan
- Human resolves: if the chapter has structured prose → DRAFT (CONSOLIDATION path); if unstructured → NOTES_ONLY (DRAFTER path)
- Human updates `scope.intended_chapters[i].status` manually before proceeding

---

## 6. Per-Chapter Invocation Protocol

When QUEEN invokes a downstream workflow on a chapter subset, the following format is used:

**BOOK_CONSOLIDATION:**
```
Trigger: BOOK_CONSOLIDATION
chapter_subset: ["CH_01_TITLE", "CH_02_TITLE", "CH_05_TITLE"]
source_directory: <path to source docs for these chapters>
```

**BOOK_STORYBOARD:**
```
Trigger: BOOK_STORYBOARD
chapter_subset: null  (run on full chapter set — recommended)
-- or --
chapter_subset: ["CH_03_NEW_CHAPTER", "CH_06_NEW_CHAPTER"]  (if only new chapters need storyboarding)
```

**BOOK_EDITORIAL:**
```
Trigger: BOOK_EDITORIAL
chapter_subset: null  (always run on full set for final editorial pass)
```

See `CHAPTER_SUBSET_PROTOCOL.md` for the full chapter_subset parameter specification.

---

*End of BOOK_COMPLETION_ROUTING.md*
