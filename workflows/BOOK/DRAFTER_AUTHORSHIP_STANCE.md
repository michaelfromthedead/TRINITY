# DRAFTER_AUTHORSHIP_STANCE

**Document type:** Decision record
**Date:** 2026-04-18
**Task:** T4.5.1
**Covers:** Authorship stance for DRAFTER worker across all of Part 4.5

---

## Decision

**Adopted stance: Stance 3 — Full prose under template constraint, with safeguards.**

DRAFTER produces complete, well-formed prose (not skeletal placeholders) for chapters in `MISSING`, `OUTLINE_ONLY`, `NOTES_ONLY`, and `PARTIALLY_DRAFTED` states. Output conforms to the manuscript's declared templates (VOICE, PERSONA, STYLE, PROSE or BUNDLE) to the best degree possible given available material. Output is marked `drafter_origin: true` in chapter YAML frontmatter and is subject to mandatory human review and enhanced editorial scrutiny before proceeding through the pipeline.

---

## Rationale

### Why not Stance 1 (human-authored only)?

Stance 1 blocks the pipeline whenever a chapter is missing. In Case A scenarios (chapters 3, 6, 8–12 entirely absent), COMPLETION cannot proceed at all. For a workflow family whose purpose is to bring manuscripts to completion, a stance that blocks on missing chapters is operationally untenable.

### Why not Stance 2 (skeletal drafts marked for revoicing)?

Stance 2 produces structurally useful but prose-inferior output — outlines, bullet points, placeholder paragraphs. This forces EDITORIAL and REVISION to apply voice and prose afterward, effectively doubling the editorial work on drafter-origin chapters. Template adherence at the skeletal level is partial at best, which means JUNIOR_VOICE and JUNIOR_STYLE will produce high finding rates that must be worked through. Stance 2 yields the problems of Stance 3 (AI-authored content exists) without the benefit (well-formed prose that editorial can verify against templates).

### Why Stance 3?

Full prose output under template constraint gives EDITORIAL a complete, auditable target. The juniors can run their standard checklist against the chapter. REVISION can make surgical corrections. The gap-flagging system (`[DRAFTER_GAP: reason]`) handles the cases where material is genuinely insufficient — producing a partial draft rather than hallucinated content. Stance 3 maximizes DRAFTER's usefulness while the safeguards contain the authorship risks.

**Dissertation basis:** §9 (BOOK_EDITORIAL's REVISION is the highest-skill worker precisely because it rewrites prose under simultaneous template + storyboard + concept constraints). A similar constraint set applied to DRAFTER's initial authorship produces first drafts that REVISION can correct — a tractable problem. Skeletal stubs would require wholesale authoring by REVISION, which violates its "surgical" discipline and would likely trigger an ESCALATE.

---

## Safeguards

The following safeguards are **non-negotiable** whenever Stance 3 is active:

### Safeguard 1 — Frontmatter flag

Every chapter file produced by DRAFTER must include in its YAML frontmatter:

```yaml
drafter_origin: true
drafter_gaps: []  # populated if any [DRAFTER_GAP] markers were placed
```

This flag travels with the chapter file through all downstream workflows. No workflow downstream of DRAFTER may strip or alter this flag.

### Safeguard 2 — Mandatory human review gate

**Before BOOK_EDITORIAL may be engaged on a drafter-origin chapter, the human must review that chapter.** This is not automated. BOOK_COMPLETION's routing plan explicitly gates drafter-origin chapters at the human review boundary after STORYBOARD and before EDITORIAL. The QUEEN reports this gate and waits.

Rationale: AI-authored prose, however template-constrained, can introduce subtle voice-breaks, invented claims, or conceptual drift that only the author recognizes as wrong. Human review at this gate is the author's creative control checkpoint.

### Safeguard 3 — Enhanced editorial scrutiny

In BOOK_EDITORIAL, chapters with `drafter_origin: true` in frontmatter receive:
- All four JUNIOR workers apply their full audit checklist (no reduced scope)
- JUNIOR_CONCEPT additionally checks every `[DRAFTER_GAP: ...]` marker as a **Critical blocking finding** until the gap is resolved by human input or additional source material
- EDITORIAL_SYNTHESIS notes that drafter-origin content is more likely to produce voice-style conflicts than human-authored content, and flags compound issues at Medium severity where it would normally flag Low
- SENIOR_FINAL is explicitly aware that a drafter-origin chapter's GREEN_LIGHT represents AI-authored prose passing editorial review — the verdict is still binding, but SENIOR_FINAL's rationale should explicitly note this

**Basis:** BOOK_EDITORIAL JSON §roles.JUNIOR_CONCEPT; BOOK_EDITORIAL JSON §hard_rules.no_fabricated_findings.

### Safeguard 4 — Revision looser scope for drafter-origin chapters

The BOOK_EDITORIAL JSON specifies REVISION as "surgical" — modifying only flagged passages. Dissertation §9.5 specifies this constraint explicitly.

For drafter-origin chapters, REVISION may operate at **passage-scale** rather than sentence-scale. This means:
- Normal author-owned chapters: REVISION rewrites the flagged sentence(s) and adjacent connector text, minimum edit
- Drafter-origin chapters: REVISION may rewrite an entire paragraph or section block if the finding warrants it, provided the rewrite stays within the storyboard's description of what that section does

This looser scope is explicitly documented to avoid conflict with the "surgical" rule. The distinction is: drafter-origin prose was not written by the author, carries no authorial intent, and therefore REVISION is not violating any authorial decision when it edits at passage scale. For author-owned prose, sentence-scale minimality protects authorial intent.

**Explicit rule:** REVISION does NOT get blanket license to wholesale-rewrite drafter-origin chapters. Passage-scale applies per-finding, not per-chapter.

### Safeguard 5 — No fact invention

DRAFTER does not invent facts, claims, experiments, citations, or numerical results not supported by the manifest scope or provided notes. If material is insufficient to support a claim, DRAFTER places a `[DRAFTER_GAP: reason]` marker rather than fabricating supporting content.

This is the strongest safeguard and the hardest to enforce automatically. JUNIOR_CONCEPT and SENIOR_FINAL both check for conceptual coherence; invented facts will typically show up as consistency violations. Human review (Safeguard 2) is the ultimate backstop.

---

## Consequences for DRAFTER Design

1. DRAFTER is a prose-writing worker, not an outliner. It must produce complete sentences and paragraphs.
2. DRAFTER applies all template axes simultaneously — the same constraint set as REVISION, applied at initial authoring time.
3. DRAFTER reports its gap-flags explicitly so QUEEN can surface them at the human review gate.
4. DRAFTER does not pretend gaps are filled. Partial drafts with clear gap-flags are better than complete drafts that hallucinate content.
5. DRAFTER's output goes to STORYBOARD (not CONSOLIDATION) because the output is already chapter-shaped. CONSOLIDATION is for collapsing chaotic source material; DRAFTER's output bypasses that stage.

## Consequences for EDITORIAL Discipline

1. JUNIOR workers always run their full checklist on drafter-origin chapters — no reduced scope.
2. JUNIOR_CONCEPT treats `[DRAFTER_GAP: ...]` markers as Critical findings. They block GREEN_LIGHT.
3. REVISION may operate at passage-scale on drafter-origin chapters, but not wholesale.
4. SENIOR_FINAL documents that drafter-origin chapters were reviewed in its rationale.

## Consequences for REVISION Discipline

REVISION's "surgical" rule is modified as follows for drafter-origin chapters:

> **For author-owned chapters:** REVISION modifies only flagged passages at sentence scale. Minimum edit.
> **For drafter-origin chapters:** REVISION may modify flagged passages at passage scale (up to a full paragraph or section block per finding). Still bounded by template + storyboard + concept constraints. Does not touch unflagged passages.

This distinction must be maintained explicitly to prevent confusion between the two modes.

---

*End of DRAFTER_AUTHORSHIP_STANCE.md*
