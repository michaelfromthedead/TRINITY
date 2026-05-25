# CHAPTER_SUBSET_PROTOCOL

**Version:** 1.0.0
**Date:** 2026-04-18
**Task:** T4.5.11
**Consumed by:** BOOK_CONSOLIDATION (T4.8), BOOK_STORYBOARD (T5.5), BOOK_EDITORIAL (T6.8, T7.7), BOOK_COMPLETION routing

---

## Purpose

This document specifies the `chapter_subset` parameter — the cross-cutting mechanism by which BOOK workflows operate on a subset of chapters rather than the full manuscript. All downstream BOOK workflows that accept chapter subsets follow this protocol.

---

## 1. Parameter format

The `chapter_subset` parameter accepts three forms:

### Form 1 — Null / absent (full-manuscript pass)

```json
"chapter_subset": null
```

Or simply: omitting the parameter entirely.

**Behavior:** The workflow operates on the complete manuscript. All chapters in `chapters/` (for STORYBOARD and EDITORIAL) or all source documents for the intended chapters (for CONSOLIDATION) are included.

**When to use:** Whenever you want the full pipeline to run. STORYBOARD and EDITORIAL should almost always use null/absent for their final passes — the full arc must be assessed to produce a coherent storyboard and consistent editorial review.

### Form 2 — Array of chapter slugs

```json
"chapter_subset": ["CH_01_PROBLEM_OF_ROTATION", "CH_03_SPINOR_FIELDS", "CH_06_GAUGE_THEORY"]
```

**Behavior:** The workflow operates only on the listed chapters. All other chapters are excluded from processing.

**Slug format:** Slugs must match exactly the `slug` values in `scope.intended_chapters` and the actual filenames in `chapters/` (without `.md` extension). Example: `CH_01_PROBLEM_OF_ROTATION` not `chapters/CH_01_PROBLEM_OF_ROTATION.md` and not `CH_1`.

**Validation:** QUEEN verifies that every slug in the array exists as a chapter file before invoking any worker. If a slug does not exist (file not yet created), QUEEN reports the missing file and halts the workflow.

### Form 3 — Range specification

```json
"chapter_subset": {"from": "CH_03_SPINOR_FIELDS", "to": "CH_08_GAUGE_INVARIANCE"}
```

**Behavior:** The workflow operates on all chapters from `CH_03` through `CH_08` inclusive, as ordered in STRUCTURE.md (not by slug alphabetical order — by position in the chapter sequence).

**Boundary chapter handling:** Both `from` and `to` are inclusive. A range of `{"from": "CH_05", "to": "CH_05"}` processes only CH_05.

**Validation:** QUEEN verifies that both boundary slugs exist and that `from` precedes `to` in the chapter sequence. If `to` precedes `from`, QUEEN reports an invalid range and halts.

---

## 2. Parameter source and precedence

The `chapter_subset` parameter can come from two sources:

### 2.1 Invocation parameter (per-run override)

Specified at the time of triggering the workflow:
```
Trigger: BOOK_EDITORIAL
chapter_subset: ["CH_03_SPINOR_FIELDS", "CH_06_GAUGE_THEORY"]
```

**This is the most common form.** BOOK_COMPLETION's QUEEN constructs the subset parameter and provides it to the human as the invocation instruction.

### 2.2 Manifest setting (persistent)

Set in BOOK_MANIFEST.json as a persistent workflow setting:
```json
"workflow_settings": {
  "book_editorial_default_subset": ["CH_01", "CH_02", "CH_03"]
}
```

**Rarely used.** Manifest settings are appropriate when a multi-session project has a persistent subset focus (e.g., only chapters 1–3 are ready for editorial review and will remain so across multiple sessions).

### 2.3 Precedence rule

**Invocation parameter overrides manifest setting.** If both are present:
- Invocation parameter is used
- QUEEN logs: "chapter_subset from invocation parameter overrides manifest setting"

If neither is present: full-manuscript pass (null behavior).

---

## 3. How each workflow interprets chapter_subset

### 3.1 BOOK_CONSOLIDATION (T4.8)

**Scope: SCRIBE_LOOP and COMPOSITOR**

When `chapter_subset` is specified:

1. **INVENTORY phase:** QUEEN scans source files for all intended chapters but identifies only those source files associated with the subset chapters (from `triage.per_chapter_state[i].source_files` for each chapter in the subset).
2. **SCRIBE_LOOP:** Only source files associated with subset chapters are processed. SCRIBE runs only on these files. MASTER.md may receive contributions from other chapters' source files if those files also contain relevant content for subset chapters — this is handled by SCRIBE's normal upsert logic.
3. **COMPOSITOR:** Carves only the subset chapters from MASTER. Produces chapter files only for the subset. STRUCTURE.md entry additions are limited to the subset chapters (existing STRUCTURE.md entries for other chapters are preserved).
4. **QA_COMPLETENESS + QA_COHERENCE:** Scope to subset chapters and their associated source files only.

**Full-manuscript note:** CONSOLIDATION should ideally run on the full corpus or on natural groupings of related chapters. Subsetting CONSOLIDATION too aggressively may cause COMPOSITOR to miss inter-chapter structural relationships. Prefer running CONSOLIDATION on all DRAFT-state chapters at once rather than chapter-by-chapter.

### 3.2 BOOK_STORYBOARD (T5.5)

**Scope: STORYBOARDER and QA_STORYBOARD**

When `chapter_subset` is specified:

1. **STORYBOARDER inputs:** Only chapter files in the subset are passed to STORYBOARDER. However, if STORYBOARD.md already exists from a prior pass, STORYBOARDER also reads the existing storyboard entries for non-subset chapters (as context for prerequisite chain continuity).
2. **STORYBOARDER output:** Produces storyboard entries only for subset chapters. Appends to existing STORYBOARD.md rather than regenerating the entire storyboard.
3. **QA_STORYBOARD:** Checks only subset entries' prerequisite satisfaction and arc coherence. May also check consistency of new entries against existing entries for non-subset chapters.

**Warning:** Subsetting STORYBOARD should only be done when the subset represents genuinely new chapters (e.g., DRAFTER-produced chapters being storyboarded for the first time while the rest of the storyboard is already stable). Running STORYBOARD on a subset for iterative refinement risks producing an inconsistent storyboard (new entries may conflict with existing entries). Prefer full-manuscript STORYBOARD runs.

**DRAFTER-origin awareness (T5.5):** When STORYBOARD is invoked after DRAFTER produces chapters, QA_STORYBOARD reads frontmatter for each chapter. Chapters with `drafter_origin: true` receive an additional accuracy spot-check — QA_STORYBOARD verifies that the chapter's prose content matches the storyboard entry it produced for that chapter. This check is in addition to all standard QA_STORYBOARD checks.

### 3.3 BOOK_EDITORIAL — Junior/Synthesis stage (T6.8)

**Scope: JUNIOR_EDITORIAL unit (4 parallel workers) and EDITORIAL_SYNTHESIS**

When `chapter_subset` is specified:

1. **JUNIOR_EDITORIAL workers (all 4):** Each junior worker receives only the subset chapter files. They do not review non-subset chapters. Their findings lists contain findings only from subset chapters.
2. **EDITORIAL_SYNTHESIS:** Integrates findings from the four juniors, all scoped to the subset. Cross-axis findings are identified within the subset only.
3. **Context:** Juniors and SYNTHESIS still have access to STORYBOARD.md and STRUCTURE.md in full — these are reference documents for the full manuscript. The subset restriction applies to the chapter files being audited, not to reference documents.

**DRAFTER-origin awareness (T6.8):** Each JUNIOR worker reads frontmatter for each chapter in the subset. When `drafter_origin: true` is present:
- All four juniors apply their full audit checklist (no exemptions for DRAFTER-origin chapters)
- JUNIOR_CONCEPT additionally checks for `[DRAFTER_GAP: ...]` markers; any such marker is a blocking Critical finding unless it has been resolved by human annotation
- EDITORIAL_SYNTHESIS is aware that drafter-origin content is higher risk for voice-style conflicts; flags compound issues at Medium severity (vs. Low for human-authored chapters where the same issue would be lower-priority)

### 3.4 BOOK_EDITORIAL — Senior/Revision stage (T7.7)

**Scope: SENIOR_SANITY, SENIOR_FINAL, REVISION**

When `chapter_subset` is specified:

1. **SENIOR_SANITY:** Reviews only findings from the junior/synthesis stage that reference subset chapters. Does not review findings from (hypothetical) previous full-manuscript passes for non-subset chapters.
2. **SENIOR_FINAL:** Performs independent pass on subset chapters only. May still reference STORYBOARD.md and STRUCTURE.md in full. Issues verdict for the subset: GREEN_LIGHT for the subset / REVISE subset chapters / ESCALATE.
3. **REVISION:** Makes surgical edits only in subset chapter files. Does not touch non-subset chapters even if SENIOR_FINAL notes holistic issues that span subset and non-subset chapters (those holistic issues become ESCALATE candidates).

**DRAFTER-origin in REVISION (T7.7):** When REVISION is applied to a chapter with `drafter_origin: true`:
- REVISION may operate at passage-scale (up to a full paragraph or section block per finding)
- Normal author-owned chapters: REVISION operates at sentence-scale minimum-edit
- This distinction is documented in `DRAFTER_AUTHORSHIP_STANCE.md` §"Safeguard 4 — Revision looser scope for drafter-origin chapters"
- REVISION still respects all five constraints: template adherence, storyboard adherence, concept consistency, local context, and minimality (at the passage level, not sentence level)

**Subset verdict semantics:** When SENIOR_FINAL emits a verdict for a subset invocation, the verdict applies to the subset:
- GREEN_LIGHT on subset ≠ GREEN_LIGHT on full manuscript
- After all subsets reach GREEN_LIGHT (or the full manuscript is run), BOOK_COMPLETION's QUEEN issues the aggregate verdict

### 3.5 BOOK_PRODUCTION (not subset-able)

**BOOK_PRODUCTION does not support chapter_subset.** Production operates on the whole final manuscript — the complete set of chapters, front matter, and back matter. Partial production makes no sense because BOOK_SPEC.json must describe the entire physical book, and FORMATTER validates the complete file manifest.

If chapter_subset is provided to BOOK_PRODUCTION invocation, QUEEN should warn: "chapter_subset parameter is not applicable to BOOK_PRODUCTION. This workflow always operates on the full manuscript. Proceeding with full manuscript."

---

## 4. QUEEN's subset verification protocol

Before invoking any worker with a chapter_subset parameter, QUEEN performs:

1. **Slug validation:** Every slug in the subset exists in `chapters/` as `CH_<slug>.md`. Report any missing files before proceeding.
2. **STRUCTURE.md consistency:** Every slug in the subset has a corresponding entry in STRUCTURE.md. If not, QUEEN notes the inconsistency and resolves it before invoking workers.
3. **Range validation (Form 3):** `from` slug appears before `to` slug in STRUCTURE.md chapter ordering. If not, QUEEN reports invalid range.
4. **Minimum subset size:** Subset must contain at least 1 chapter. Empty subset is an error.
5. **Full-manuscript safety check (STORYBOARD and EDITORIAL):** QUEEN warns if the subset is significantly smaller than the full chapter count and the workflow is STORYBOARD or EDITORIAL — these workflows are most useful on the full manuscript. Warning does not block.

---

## 5. Interaction with INPROGRESS.md

Subset invocations are logged to INPROGRESS.md with the chapter_subset explicitly noted:

```markdown
---
DATE: 2026-04-18
WORKFLOW: BOOK_EDITORIAL
CHAPTER_SUBSET: ["CH_03_SPINOR_FIELDS", "CH_06_GAUGE_THEORY"]
STATUS: IN_PROGRESS
NOTES: EDITORIAL audit of DRAFTER-produced chapters after human review gate.
---
```

Full-manuscript passes are noted with `CHAPTER_SUBSET: null`.

---

## 6. Examples

### Example 1 — COMPLETION routes DRAFTER chapters to STORYBOARD

12-chapter book. DRAFTER produced CH_03, CH_06, CH_08, CH_09, CH_10, CH_11, CH_12.
CONSOLIDATION already ran on CH_01, CH_02, CH_04, CH_05, CH_07.

QUEEN instructs human to trigger STORYBOARD on full manuscript (null subset) to produce a unified arc-aware storyboard:
```
Trigger: BOOK_STORYBOARD
chapter_subset: null
```

Rationale: the storyboard must see all 12 chapters to establish the prerequisite chain and arc. Subset storyboarding would produce an incomplete arc description.

### Example 2 — EDITORIAL on DRAFTER-only chapters

After human review gate, 6 drafter-origin chapters (CH_03, CH_06, CH_08, CH_09, CH_10, CH_11, CH_12) need editorial review. CH_01, CH_02, CH_04, CH_05, CH_07 were already editorially reviewed in an earlier session.

QUEEN constructs:
```
Trigger: BOOK_EDITORIAL
chapter_subset: ["CH_03_SPINOR_FIELDS", "CH_06_GAUGE_THEORY", "CH_08_RENORMALIZATION", "CH_09_SYMMETRY_BREAKING", "CH_10_STANDARD_MODEL", "CH_11_OPEN_QUESTIONS", "CH_12_CONCLUSION"]
```

All 4 JUNIOR workers audit only these 7 chapters. SENIOR_FINAL emits a verdict for this subset. After GREEN_LIGHT, a final full-manuscript EDITORIAL pass (null subset) may be run to check holistic arc and cross-chapter coherence.

### Example 3 — Range subset for CONSOLIDATION

Chapters 1–5 are in DRAFT state. A single CONSOLIDATION run processes all of them:
```
Trigger: BOOK_CONSOLIDATION
chapter_subset: {"from": "CH_01_PROBLEM_OF_ROTATION", "to": "CH_05_REPRESENTATIONS"}
source_directory: source/
```

QUEEN identifies which source files map to chapters 1–5 from `triage.per_chapter_state` and constructs the SCRIBE_LOOP input accordingly.

---

*End of CHAPTER_SUBSET_PROTOCOL.md*
