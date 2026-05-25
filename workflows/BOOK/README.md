# BOOK Workflow Family — Entry Point

The BOOK family is a pipeline of six AI swarm workflows that transform raw manuscript material into print-ready books. It operates on the same swarm infrastructure as SDLC and RDC (QUEEN orchestrator, specialized workers, adversarial QA, loop limits with auto-escalation) but targets authored prose rather than source code. The system accommodates different genres, voice registers, pedagogical contracts, and stylistic conventions through a template architecture — each manuscript declares its own voice, persona, style, and prose templates, and all editorial workers enforce them.

---

## The 6-Workflow Pipeline

```
BOOK_TRIAGE         — classify folder, produce BOOK_MANIFEST.json, recommend entry point
    ↓ [manual review]
BOOK_CONSOLIDATION  — chaotic folder → structured chapters + STRUCTURE.md
    ↓ [manual review]
BOOK_STORYBOARD     — chapters → logical skeleton (STORYBOARD.md), voice-neutral
    ↓ [manual review]
BOOK_EDITORIAL      — audit + revise against declared templates; surgical REVISION
    ↓ [manual review]
BOOK_PRODUCTION     — validate, generate front/back matter, produce BOOK_SPEC.json
    ↓ [human decision]
[LULU_PIPELINE]     — mechanical (not AI) → print-ready PDF
```

Every arrow is a **manual handoff**. No workflow auto-triggers the next. The author reviews each stage's output and explicitly types the next trigger phrase to continue.

**If your manuscript is incomplete or mixed-state**, use `BOOK_COMPLETION` instead of entering the pipeline directly at BOOK_CONSOLIDATION. BOOK_COMPLETION reads the per-chapter state map from BOOK_MANIFEST.json, routes complete chapters to the appropriate workflow stage, and invokes the DRAFTER worker for chapters with insufficient existing material (missing, outline-only, or notes-only state).

---

## Quickstart: "I want to turn my folder of notes into a book"

**Step 1 — Triage your folder.**
Type `BOOK_TRIAGE`. QUEEN will ask for your folder path, scan all files, classify each one, and produce `BOOK_MANIFEST.json` at your project root. Review the manifest; fill in the template declarations (which VOICE, PERSONA, STYLE, PROSE templates apply — or which BUNDLE).

**Step 2 — Review BOOK_MANIFEST.json; fill template declarations.**
Open the manifest. Confirm or override the triage recommendation. Set `templates.mode` to `"bundle"` or `"composition"` and fill in the template names. Set `structure.genre`, `target_audience`, and `production` fields. The manifest is the human-owned config document that travels through all downstream workflows.

**Step 3 — Decide entry workflow based on triage recommendation.**

| Triage recommendation | What to type next |
|---|---|
| BOOK_CONSOLIDATION (rough / chaotic) | `BOOK_CONSOLIDATION` |
| Mixed-state or incomplete | `BOOK_COMPLETION` |
| BOOK_STORYBOARD (already structured) | `BOOK_STORYBOARD` |
| BOOK_EDITORIAL (already storyboarded) | `BOOK_EDITORIAL` |
| BOOK_PRODUCTION (already polished) | `BOOK_PRODUCTION` |

**Step 4 — Proceed through the pipeline with manual review between stages.**
After each workflow reaches GREEN_LIGHT, review the output, make any direct edits you want, and then type the next trigger phrase. BOOK_STORYBOARD requires `chapters/` + `STRUCTURE.md`. BOOK_EDITORIAL requires those plus `STORYBOARD.md` and filled template declarations. BOOK_PRODUCTION requires polished chapters and a filled `production` section in the manifest.

---

## Template Architecture

Templates are the mechanism by which the BOOK system adapts to different genres, voices, and styles. They are infrastructure artifacts (not workflow-specific) that all editorial workers reference. Every manuscript's written character is described along four independent-but-interacting axes: **VOICE** (pedagogical contract with the reader), **PERSONA** (who the author performs as), **STYLE** (genre-level conventions), and **PROSE** (sentence-level craft rules). Each axis has one or more atomic template files; hand-composed **bundle docs** capture emergent properties of specific four-axis combinations.

For the template schema, naming conventions, audit checklist format, and versioning policy, see `TEMPLATE_STANDARD.md` in the templates directory. For the compatibility matrix (which atomics compose cleanly, which conflict), see `TEMPLATE_COMPATIBILITY.md`.

---

## Key Documents

| Document | Purpose |
|---|---|
| `workflows/BOOK/BOOK_WORKFLOW_DISSERTATION.md` | Authoritative architectural reference (v1.0.0 IMPLEMENTED). Covers all 6 workflows, template architecture, cross-workflow integration, hard rules, file layout, open questions resolved, and implementation learnings. Read this first for deep understanding. |
| `workflows/BOOK/BOOK_BUILDOUT_TODO.md` | 96-task buildout TODO (Parts 1–9). Implementation status of all BOOK artifacts. |
| `workflows/BOOK/BOOK_TRIAGE.json` | BOOK_TRIAGE state machine |
| `workflows/BOOK/BOOK_CONSOLIDATION.json` | BOOK_CONSOLIDATION state machine |
| `workflows/BOOK/BOOK_COMPLETION.json` | BOOK_COMPLETION state machine (mixed-state meta-orchestrator) |
| `workflows/BOOK/BOOK_STORYBOARD.json` | BOOK_STORYBOARD state machine |
| `workflows/BOOK/BOOK_EDITORIAL.json` | BOOK_EDITORIAL state machine |
| `workflows/BOOK/BOOK_PRODUCTION.json` | BOOK_PRODUCTION state machine |

### Worker docs by workflow

**BOOK_CONSOLIDATION workers:**
- `workflows/BOOK/WORKER_SCRIBE.md` — temporal upsert into MASTER.md
- `workflows/BOOK/WORKER_ADVOCATE.md` — court advocacy
- `workflows/BOOK/WORKER_COMPOSITOR.md` — carves MASTER into chapter files + STRUCTURE.md

**BOOK_COMPLETION workers:**
- `workflows/BOOK/WORKER_DRAFTER.md` — authors prose for missing/incomplete chapters under template constraint; output marked `drafter_origin: true`

**BOOK_STORYBOARD workers:**
- `workflows/BOOK/WORKER_STORYBOARDER.md` — produces STORYBOARD.md (voice-neutral)
- `workflows/BOOK/WORKER_QA_STORYBOARD.md` — audits prerequisite chain, arc, completeness, genre alignment

**BOOK_EDITORIAL workers:**
- `workflows/BOOK/WORKER_JUNIOR_VOICE.md` — audits against VOICE template (hypercritical, high-recall)
- `workflows/BOOK/WORKER_JUNIOR_CONCEPT.md` — audits concept consistency (not template-bound)
- `workflows/BOOK/WORKER_JUNIOR_STYLE.md` — audits against STYLE + PROSE templates
- `workflows/BOOK/WORKER_JUNIOR_FLOW.md` — audits logical flow against STORYBOARD.md
- `workflows/BOOK/WORKER_EDITORIAL_SYNTHESIS.md` — cross-axis interaction detection; only worker that sees all 4 junior reports
- `workflows/BOOK/WORKER_SENIOR_SANITY.md` — precision filter (real vs. overzealous; no new findings)
- `workflows/BOOK/WORKER_SENIOR_FINAL.md` — independent pass + binding verdict; QUEEN does not override
- `workflows/BOOK/WORKER_REVISION.md` — surgical prose rewriter; touches only flagged passages

**BOOK_PRODUCTION workers:**
- `workflows/BOOK/WORKER_FORMATTER.md` — validates manuscript, generates front/back matter, produces BOOK_SPEC.json
- `workflows/BOOK/WORKER_QA_PRODUCTION.md` — validates output against LULU_SPEC

**QA workers shared with BOOK_CONSOLIDATION:**
- `workflows/BOOK/WORKER_QA_COMPLETENESS.md` — concept-loss hunter
- `workflows/BOOK/WORKER_QA_COHERENCE.md` — structural integrity auditor

---

## Mixed-State Handling

If your manuscript is incomplete — chapters exist at different levels of maturity, or some chapters don't exist yet — engage `BOOK_COMPLETION` rather than BOOK_CONSOLIDATION.

BOOK_COMPLETION reads the per-chapter state classification produced by BOOK_TRIAGE v1.1 (requires `scope` section in BOOK_MANIFEST.json with `intended_chapters`) and routes each chapter to the appropriate workflow:

| Chapter state | Routed to |
|---|---|
| `MISSING` / `OUTLINE_ONLY` / `NOTES_ONLY` | DRAFTER (produces prose from scope + notes) |
| `PARTIALLY_DRAFTED` | DRAFTER (fills gaps) then CONSOLIDATION |
| `DRAFT` | CONSOLIDATION |
| `CHAPTER` | STORYBOARD or EDITORIAL (depending on storyboard status) |
| `POLISHED` | EDITORIAL (audit-only) or PRODUCTION |

DRAFTER output carries `drafter_origin: true` in chapter frontmatter. Human review of DRAFTER output is mandatory before BOOK_EDITORIAL engagement. EDITORIAL workers apply extra scrutiny to drafter-origin chapters; any `[DRAFTER_GAP: ...]` placeholder is a blocking Critical finding.

---

## Shared Infrastructure

BOOK workflows share the project-wide swarm infrastructure:

- `workflows/SHARED/WORKER_QUEEN.md` — QUEEN orchestrator role (workflow-agnostic)
- `workflows/SHARED/WORKER_PROTOCOL.md` — non-negotiable worker contract
- `workflows/SHARED/WORKER.md` — master role index (includes all BOOK roles)
- `INPROGRESS.md` — unified progress log shared across all active workflows
