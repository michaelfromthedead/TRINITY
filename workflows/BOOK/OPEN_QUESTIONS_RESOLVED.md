# BOOK Workflow — Open Questions Resolved

**Document version:** 1.0.0
**Date:** 2026-04-18
**Author:** BUILD_WORKER (Part 9-C execution)
**Source:** Dissertation §16 open questions (all 7); resolutions grounded in existing worker docs and buildout task record.
**Dissertation reference:** `workflows/BOOK/BOOK_WORKFLOW_DISSERTATION.md` §16
**Buildout TODO reference:** `workflows/BOOK/BOOK_BUILDOUT_TODO.md` Part 9, T9.11–T9.17

---

## §16.1 — CI/CD Hooks for BOOK Workflows

**Question verbatim from dissertation §16.1:**
> CI/CD hooks for BOOK workflows? SDLC has language-level CI. BOOK workflows operate on prose, not code. Is there a meaningful "pre-commit check" for manuscript files? (Markdown linting? Spell check? Template-conformance spot check?)

**Resolution:**
Add `workflows/ci-prose.sh`, modeled on the existing `workflows/ci-python.sh` and `workflows/ci-rust.sh` scripts (same mode structure: usage block, `quick` mode for pre-commit, `gate` mode for pre-push, per-stage sub-commands). The hook fires on commits touching any of the following paths:

- `chapters/`
- `front/`
- `back/`
- `templates/`
- `BOOK_MANIFEST.json`
- `BOOK_SPEC_SCHEMA.json`
- `STRUCTURE.md`
- `STORYBOARD.md`

**Checks (ci-prose.sh stage map):**

| Mode | Stages |
|---|---|
| `quick` (pre-commit) | markdown-lint + placeholder-scan |
| `gate` (pre-push) | markdown-lint + spell-check + json-validate + template-conformance-spot + placeholder-scan |

**Stage definitions:**

1. **markdown-lint** — run `markdownlint-cli` (or `mdl`) against all modified markdown files in the above paths. Catches structural invalidity: unclosed fences, heading level skips, broken link syntax, malformed table syntax.

2. **placeholder-scan** — scan modified chapter/front/back files for literal strings: `TODO`, `FIXME`, `TBD`, `[INSERT`, `[DRAFTER_GAP:`, `PLACEHOLDER` (any case). Any match in a committed chapter, front, or back file is a hard failure. This is the mechanical equivalent of FORMATTER Check 2 (§3.2 of `WORKER_FORMATTER.md`) applied pre-commit rather than at production time.

3. **spell-check** — run `aspell` or `hunspell` in batch mode on modified chapter/front/back files. Use a custom dictionary per project (seeded with technical vocabulary from `BOOK_MANIFEST.json` author-declared domain terms). Flag unrecognized words as warnings (not hard failures, to avoid blocking on legitimate technical neologisms). Human reviews spell-check output as a soft gate.

4. **json-validate** — for modified `BOOK_MANIFEST.json` and `BOOK_SPEC_SCHEMA.json`: validate JSON syntax (e.g., `python3 -m json.tool` or `jq .`). Invalid JSON is a hard failure.

5. **template-conformance-spot** — for modified template files in `templates/`: verify required sections are present (Contract, Characteristic Patterns, Anti-Patterns, Audit Checklist, Interaction Notes) and that Audit Checklist items use the `[AXIS:key]` format. This is a structural check only — it does not evaluate template quality.

**Rationale:**
BOOK operates on prose, but prose validity has a mechanical baseline (markup structure, placeholder text, spelling, JSON syntax) that CI can catch before human editorial review. These checks save editorial cycles on easily-caught issues. The two-layer enforcement model from SDLC/RDC (language-level CI in Layer 1, semantic QA in Layer 2) is directly applicable to BOOK: ci-prose.sh is Layer 1, the BOOK editorial pipeline is Layer 2. The stage map mirrors the ci-python.sh pattern (quick/gate modes, exit 0/1, verbatim output on failure).

**Reference / cross-link:**
Pattern source: `workflows/ci-python.sh`, `workflows/ci-rust.sh`.
Placeholder check source: `workflows/BOOK/WORKER_FORMATTER.md` §3.2.
Install mechanism: `workflows/install-hooks.sh` (extend to detect prose-path touches).

**Follow-up flagged:**
**T9.11-FOLLOWUP** — Actual implementation of `workflows/ci-prose.sh` (the shell script itself) is future work. This resolution documents the decision and structural design only. The script must be implemented, integrated with `install-hooks.sh`, and tested against a manuscript project before marking T9.11 fully complete.

**Status:** RESOLVED (decision + structure documented; script implementation is T9.11-FOLLOWUP)

---

## §16.2 — COMPOSITOR vs. TAXONOMIST Refactoring

**Question verbatim from dissertation §16.2:**
> COMPOSITOR vs. TAXONOMIST refactoring. After BOOK_CONSOLIDATION is built and tested, compare with RDC's TAXONOMIST to assess whether a shared base role with output-mode parameterization is cleaner than two separate role docs.

**Resolution:**
Defer refactoring until after the first real BOOK project completes. Do not extract a shared base role at this time.

**Rationale:**
The comparison was performed in T4.6 and documented in `workflows/BOOK/COMPOSITOR_VS_TAXONOMIST.md`. The comparison identifies what is genuinely shared (§1 of that doc: MASTER-scanning algorithm structure, discovery-over-prescription principle, faithful-carve discipline, coverage self-check, report structure, court back-reference preservation) and what is genuinely different (§2: output shape is fundamentally different — ARCH/TODO pairs vs. chapter files; section hierarchy is novel to COMPOSITOR with no TAXONOMIST equivalent; STRUCTURE.md format is COMPOSITOR-specific and elaborate; CLARIFICATION.md is TAXONOMIST-specific; dependency semantics are different; granularity calibration criteria differ; BOOK_MANIFEST.json is a COMPOSITOR-specific input).

The comparison doc's §4.2 conclusion is authoritative: "A parameterized base role (`output_mode: 'chapters' | 'phases'`) would need to branch on nearly every concrete decision. The parameterization surface would be approximately as large as the two roles themselves." Additionally, COMPOSITOR has not been tested on real material — its algorithm may require adjustment. Premature abstraction before empirical validation risks abstracting the wrong interface. The cost of maintaining two narrative role docs is low. The cost of a premature abstraction that turns out to be wrong is higher. Empirical experience from a first real BOOK project will show which parts of COMPOSITOR's algorithm were stable, making the shared-vs.-different boundary clearer before any refactoring attempt.

**Reference / cross-link:**
`workflows/BOOK/COMPOSITOR_VS_TAXONOMIST.md` §4 (T4.6).

**Status:** DEFERRED — pending empirical evidence from first real BOOK project

---

## §16.3 — REVISION Worker Capability Boundaries

**Question verbatim from dissertation §16.3:**
> REVISION worker capability boundaries. How much prose rewriting is the REVISION worker permitted to do in a single pass? Should there be a maximum-changed-passages-per-cycle limit to prevent wholesale rewriting that undermines the "surgical" principle?

**Resolution:**
Already resolved in T7.4. Soft cap of 20 passages per REVISE cycle. When findings exceed the cap, REVISION processes Critical findings first, then High, then Medium, then Low, stopping at 20 passages and flagging remaining findings as deferred. SENIOR_FINAL can raise the cap via explicit `revision_budget_override: N` in the REVISE list header. The cap is a discipline rule rooted in the empirical observation that surgical constraint tracking (five simultaneous constraints across multiple locations) degrades in reliability above approximately 20 concurrent edits.

The `REVISION_BUDGET` section in `WORKER_REVISION.md` also establishes the distinction between author-owned chapters (sentence-scale minimum edit) and drafter-origin chapters (`drafter_origin: true` in frontmatter, passage/paragraph-scale edit permitted). This distinction resolves the apparent tension between "REVISION is surgical" and the reality that drafter-origin prose carries no authorial intent and does not require sentence-scale protection.

**Reference / cross-link:**
`workflows/BOOK/WORKER_REVISION.md` §REVISION_BUDGET (T7.4); `workflows/BOOK/WORKER_REVISION.md` §7 (drafter-origin discipline, T7.7).

**Status:** RESOLVED_ELSEWHERE (T7.4)

---

## §16.4 — Multi-Language Manuscripts

**Question verbatim from dissertation §16.4:**
> Multi-language manuscripts. Some of Michael's works may include Korean text. Do templates need a language dimension, or is that handled by prose templates?

**Resolution:**
Handle within the PROSE template axis as a language-aware sub-variant. No new fifth axis is introduced.

Template naming convention for language variants:
`PROSE_<style>_<language_code>.md`

Examples:
- `PROSE_MEDIUM_ACCESSIBLE_EN.md` (default English)
- `PROSE_MEDIUM_ACCESSIBLE_KR.md` (Korean variant)

`BOOK_MANIFEST.json` is extended to declare:

```json
"language": {
  "primary": "en",
  "additional": ["kr"]
}
```

The PROSE template's Audit Checklist items are language-specific where needed. For English PROSE, items reference CEFR level (B2/C1/C2 vocabulary register). For Korean PROSE variants, equivalent items reference 한국어 성인 읽기 수준 (Korean adult reading level) and character density norms appropriate to Korean prose. The Contract section and structural sections (Characteristic Patterns, Anti-Patterns, Interaction Notes) of a Korean PROSE variant may be written in either English or Korean at the template author's discretion.

For mixed-language manuscripts (e.g., English body with Korean section introductions or captions):
- `BOOK_MANIFEST.json` declares a primary language and one or more additional languages.
- REVISION respects the appropriate PROSE template per chapter or per section, as declared in the chapter frontmatter or manifest scope.
- JUNIOR_PROSE (under BOOK_EDITORIAL) applies the correct PROSE variant for each chapter based on its declared language.

**Rationale:**
Introducing a fifth template axis for language would require updating the entire axis system (TEMPLATE_STANDARD, TEMPLATE_COMPATIBILITY, BUNDLE format, BOOK_MANIFEST resolution logic, and all four juniors' worker docs). For a use case that is fundamentally "the same prose register expressed in a different language," this is over-engineering. The PROSE template already governs sentence-level craft rules — vocabulary register, sentence complexity, and density — which are precisely the dimensions that differ across languages. A language sub-variant of an existing PROSE template inherits the axis's structural contract and adds language-specific calibration without multiplying the axis count. Template authors for Korean PROSE variants reference the English variant as a structural template (`cp PROSE_MEDIUM_ACCESSIBLE_EN.md PROSE_MEDIUM_ACCESSIBLE_KR.md` + language-specific edits), which follows the same template-for-templates pattern established in T1.4.

**Reference / cross-link:**
`workflows/BOOK/BOOK_BUILDOUT_TODO.md` T9.14; `TEMPLATE_STANDARD.md` (once written, T1.2) — the naming convention adds `_<language_code>` suffix to PROSE template names.

**Status:** RESOLVED

---

## §16.5 — Figure, Equation, and Code Listing Integration

**Question verbatim from dissertation §16.5:**
> Figure/equation/code integration. The current design treats chapters as text. How are figures, equations, and code listings handled? Are they inline in the markdown, or external files referenced by the chapter? This affects QA_PRODUCTION and LULU_PIPELINE.

**Resolution:**
Inline-first policy. Each element type is handled as follows:

**Equations:**
Inline LaTeX delimiters. Inline equations use `$...$`. Display equations (block-level, numbered or unnumbered) use `$$...$$`. BOOK_SPEC.json field `special_elements.equations: true` signals LULU_PIPELINE to enable math rendering (e.g., MathJax, KaTeX, or LaTeX engine depending on T9.1 typesetting engine choice). FORMATTER Check 5 (markup validation) scans for unclosed LaTeX delimiters.

**Code listings:**
Inline fenced code blocks with language hints (` ```python `, ` ```rust `, etc.). BOOK_SPEC.json field `special_elements.code_listings: true`. Chapter files are self-contained; no external code files referenced unless the author explicitly declares external code as an appendix.

**Figures:**
File references via standard markdown image syntax: `![caption](figures/figure_NN_description.ext)`. Figures live in a `figures/` directory at project root (peer of `chapters/`, `front/`, `back/`). BOOK_SPEC.json fields: `special_elements.figures: true` and `figures_directory: "figures/"`. FORMATTER Check 1 extension: verify that every figure reference in chapter files resolves to an existing file in `figures/`. QA_PRODUCTION checks figure references as part of file integrity validation.

**Tables:**
Inline markdown table syntax for simple cases. Pandoc-extended table syntax for complex multi-line cells or merged cells. BOOK_SPEC.json field `special_elements.tables: true`. FORMATTER Check 5 validates table row consistency.

**LULU_PIPELINE behavior:**
The typesetting engine processes inline elements (equations, code, tables) natively from markdown. Figures are resolved from file references in `figures/`. No element type requires a dedicated external reference system beyond the file reference convention for figures.

**Rationale:**
Inline-first keeps each chapter file self-contained and readable as plain markdown prose. An author reviewing `chapters/CH_03_ANGULAR_MOMENTUM.md` can read the equations and code listings in context without dereferencing external files. External references are used only where pandoc/markdown's inline form is inadequate — specifically, images (which require binary files) and explicitly-declared code appendices. The BOOK_SPEC.json `special_elements` flags allow LULU_PIPELINE to configure the appropriate render chain for each element type without requiring hard-coded logic. FORMATTER's existing Check 5 (markup validation) already scans for unclosed LaTeX delimiters and broken fenced blocks, which covers equation and code integrity at the pre-production stage.

**Reference / cross-link:**
`workflows/BOOK/WORKER_FORMATTER.md` §3.5 (markup validation), §5.2 (index generator uses figure reference detection), §6.6 (special elements detection); `workflows/BOOK/BOOK_WORKFLOW_DISSERTATION.md` §5 (BOOK_MANIFEST.json schema includes `special_elements`).

**Status:** RESOLVED

---

## §16.6 — Index Generation Ownership

**Question verbatim from dissertation §16.6:**
> Index generation. Is the index (back matter) generated by FORMATTER, by a dedicated INDEX worker, or by the mechanical pipeline? Index generation is a significant task for non-fiction.

**Resolution:**
Three-phase ownership:

1. **FORMATTER produces a preliminary index** (`back/index.md`) via automated key-term extraction. The algorithm runs in T8.7 / BOOK_PRODUCTION and is documented in `workflows/BOOK/back_matter_templates/index_generator.md`. Candidate terms are collected from four sources in priority order: (a) concepts listed in STORYBOARD.md `Concepts Introduced` sections, (b) typographically emphasized terms (bold/italic), (c) explicitly defined terms (definition-marker patterns), and (d) high-frequency terms (≥3 chapter files or ≥8 occurrences in a single chapter). Each term is recorded with chapter:section locators. The preliminary index is clearly marked at its head: "NOTE: This is a preliminary index generated by FORMATTER from automated key-term extraction. Human review and refinement is required before publication."

2. **Human reviews and refines** the preliminary index during the BOOK_PRODUCTION human review gate — before LULU_PIPELINE runs. The human makes all judgments that require authorial knowledge: which candidate terms merit entries, appropriate granularity, subentries, cross-references ("see also"), variant forms, and mathematical symbol entries (the algorithm handles text terms only — mathematical symbols require manual addition).

3. **LULU_PIPELINE does NOT modify the index.** The mechanical pipeline typesets whatever is in `back/index.md`. It converts section-level locators to actual page numbers during pagination. It does not add or remove index entries.

No dedicated INDEX worker is introduced. The index task is split between a mechanical pre-pass (FORMATTER) and human editorial judgment. Adding a dedicated INDEX worker would add a role for a task that is, at its core, a human authorial decision about what deserves to be indexed — which no worker can substitute for.

**Rationale:**
Full index generation requires authorial judgment. The preliminary-index pattern (mechanical extraction → human review → mechanical typesetting) matches FORMATTER's defined non-responsibility boundary: "no aesthetic decisions" and "no chapter prose modification." FORMATTER's role is pre-work. The index_generator.md algorithm (Steps 1–8 documented therein) is comprehensive enough to produce a useful working draft that substantially reduces the human's indexing work without removing the judgment that only the author can supply. This also matches the established pattern for other back matter types (bibliography, glossary) where FORMATTER produces a generated draft and the human reviews before production.

**Reference / cross-link:**
`workflows/BOOK/back_matter_templates/index_generator.md` (algorithm, Steps 1–8);
`workflows/BOOK/WORKER_FORMATTER.md` §5.2 (index generation procedure) and §7 (non-responsibilities: no aesthetic decisions);
`workflows/BOOK/BOOK_BUILDOUT_TODO.md` T8.7.

**Status:** RESOLVED

---

## §16.7 — Template Versioning Strategy

**Question verbatim from dissertation §16.7:**
> Versioning strategy for templates. Templates will evolve. How are versions tracked? SemVer per template file? Does BOOK_MANIFEST pin template versions?

**Resolution:**
Already resolved in T1.5. SemVer per template file, declared in the template's YAML frontmatter (`VERSION: <semver>`). Version semantics:

- **MAJOR** — breaking change (the template's Contract has changed, or Audit Checklist items have been removed or fundamentally redefined). BOOK_MANIFEST must pin the major version when referencing a template that has undergone a MAJOR change; unpinned manifests will receive a warning at BOOK_EDITORIAL engagement if the loaded template's major version differs from the version present when the manifest was created.
- **MINOR** — additive change (new Audit Checklist items added, new Characteristic Patterns documented, Interaction Notes expanded). Compatible within a manifest — existing manifests do not need to pin minor versions.
- **PATCH** — clarification, wording fix, example improvement with no semantic change. Compatible within a manifest.

Unpinned manifests load the latest active version of the referenced template. A manifest that pins a specific version (`"prose": "PROSE_MEDIUM_ACCESSIBLE@1.2.0"`) loads exactly that version.

**Reference / cross-link:**
`workflows/BOOK/BOOK_BUILDOUT_TODO.md` T1.5; `TEMPLATE_STANDARD.md` §versioning (once written, T1.2) — the full policy lives there.

**Status:** RESOLVED_ELSEWHERE (T1.5)

---

## Summary Table

| § | Question | Status | Resolved in | Follow-ups |
|---|---|---|---|---|
| 16.1 | CI/CD hooks for BOOK workflows | RESOLVED | This doc | T9.11-FOLLOWUP: implement ci-prose.sh |
| 16.2 | COMPOSITOR vs. TAXONOMIST refactoring | DEFERRED | This doc (citing T4.6 comparison) | Revisit after first real BOOK project |
| 16.3 | REVISION capability boundaries | RESOLVED_ELSEWHERE | T7.4 / WORKER_REVISION.md §REVISION_BUDGET | — |
| 16.4 | Multi-language manuscripts | RESOLVED | This doc | Template authors write KR PROSE variants when needed |
| 16.5 | Figure/equation/code integration | RESOLVED | This doc | T9.1 engine choice may refine equation/code rendering details |
| 16.6 | Index generation ownership | RESOLVED | This doc (citing WORKER_FORMATTER.md §5.2 + index_generator.md) | — |
| 16.7 | Template versioning strategy | RESOLVED_ELSEWHERE | T1.5 / TEMPLATE_STANDARD.md | — |

---

```
==== PART 9-C BUILD_WORKER REPORT ====
Tasks completed: 7 / 7 open questions

Resolution summary:
  T9.11 (CI hooks): RESOLVED (decision documented; ci-prose.sh implementation flagged as T9.11-FOLLOWUP)
  T9.12 (COMPOSITOR refactoring): DEFERRED (post-first-project empirical evidence; cites COMPOSITOR_VS_TAXONOMIST.md §4)
  T9.13 (REVISION budget): RESOLVED_ELSEWHERE (T7.4 / WORKER_REVISION.md §REVISION_BUDGET)
  T9.14 (multi-language): RESOLVED (PROSE sub-variant convention; PROSE_<style>_<lang>.md naming)
  T9.15 (figure/eq/code): RESOLVED (inline-first policy; figures/ dir for images; BOOK_SPEC flags)
  T9.16 (index generation): RESOLVED (preliminary by FORMATTER; human review; LULU_PIPELINE typesets only)
  T9.17 (template versioning): RESOLVED_ELSEWHERE (T1.5 / TEMPLATE_STANDARD.md §versioning)

Files produced:
  - workflows/BOOK/OPEN_QUESTIONS_RESOLVED.md (298 lines)

Follow-ups flagged:
  - T9.11-FOLLOWUP: ci-prose.sh implementation (future work — script not yet written)

Fabrication audit: zero
  Every reference to a worker doc, section, or algorithm cites a specific existing file
  verified during research (WORKER_REVISION.md, WORKER_FORMATTER.md, COMPOSITOR_VS_TAXONOMIST.md,
  index_generator.md, ci-python.sh, BOOK_BUILDOUT_TODO.md, BOOK_WORKFLOW_DISSERTATION.md).
  No policy, section number, or file reference was invented.
```
