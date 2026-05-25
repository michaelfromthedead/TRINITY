# ORGANIZE Template Library — Roadmap

**Task:** T2.7
**Date:** 2026-04-18
**Workflow:** ORGANIZE_WORKFLOW v0.1.0-DRAFT
**Location:** `workflows/ORGANIZE/templates/`

---

## Purpose

This document records templates that were identified but deferred from the initial v1.0.0 library release. Each entry explains why the template was deferred, what it would cover, and what conditions would trigger its promotion to the library.

The initial library (5 templates) covers the project shapes that are currently in active use. Templates outside this set are deferred for one of three reasons:

1. **No current project uses the shape** — no real-world validation
2. **Sufficient overlap with an existing template** — users can extend the closest template with custom rules
3. **Too speculative** — the shape is not yet well-defined enough to write canonical rules without over-specifying

---

## Promotion criteria

A deferred template should be promoted when:

1. A real project cannot be adequately served by any existing canonical template or by `custom` mode.
2. The project's shape is stable enough that a set of canonical rules can be written without heavy speculation.
3. The rules have been validated against at least one actual project (not just a hypothetical one).

**Process for promotion:**
1. Identify the candidate project and run BOOTSTRAP in `custom` mode.
2. After 2–3 MAINTENANCE circuits, inspect the `.organize.json` rules that emerged.
3. Extract the stable rules into a new template file.
4. Write the template, add to `ORGANIZE_CONFIG_SCHEMA.json` enum, update this roadmap, update `TEMPLATES_INDEX.md`.

---

## Deferred templates (prioritized)

### Priority 1 — `python-app`

**File (future):** `python-app.json`
**Applies to:** Python application project with a clear entrypoint (`src/app/main.py` or `app.py` at root), configuration directory (`config/`), and scripts directory. Differs from `python-lib` by having an application entry point and a `config/` directory for environment-specific settings.

**Why deferred:** The `python-app` shape has significant variation — Django apps, FastAPI services, CLI tools, and data pipelines all have Python at root but differ substantially in layout. Premature canonicalization would produce a template that fits no real project well. Wait for a concrete project to validate against.

**Key rules that would differ from `python-lib`:**
- `config/**/*` → IN_PLACE (application configuration is load-bearing)
- `*.env.example` → IN_PLACE at root (environment template files stay at root)
- `app.py` / `main.py` at root → IN_PLACE (application entrypoints stay at root)
- `**/*.cfg`, `**/*.ini`, `**/*.toml` in `config/` → IN_PLACE

**Promotion trigger:** A Python web service or CLI application project that needs ORGANIZE and cannot be adequately served by `python-lib`.

---

### Priority 2 — `rust-app`

**File (future):** `rust-app.json`
**Applies to:** Rust binary application with `src/main.rs` (or `src/bin/<name>.rs`) as the entry point. Differs from `rust-crate` by the absence of `src/lib.rs` and the presence of `src/main.rs` as the binary entry point.

**Why deferred:** Rust binary applications and library crates share most of the same project structure (both use `src/`, `tests/`, `benches/`, `Cargo.toml`). The difference is `src/lib.rs` vs `src/main.rs` — a single signal. A user with a Rust binary project can use `rust-crate` with no functional difference for ORGANIZE's purposes, since the rule patterns are nearly identical.

**Key rules that would differ from `rust-crate`:**
- `src/main.rs` → IN_PLACE (binary entry point; rule t1 in rust-crate already covers `src/**/*.rs` so this may not require a separate rule)
- `src/bin/**/*.rs` → IN_PLACE (multiple binary targets)

**Promotion trigger:** A Rust binary application project that needs different ORGANIZE behavior from `rust-crate`. Currently unlikely — the templates would be nearly identical.

---

### Priority 3 — `polyglot`

**File (future):** `polyglot.json`
**Applies to:** Multi-language projects where different directories follow different per-language conventions. Example: a repository with `backend/` (Rust or Python), `frontend/` (TypeScript/React), `infra/` (Terraform), and `docs/` (markdown). Each subdirectory follows its own language convention internally.

**Why deferred:** Polyglot projects have too much variation in structure to canonicalize safely. The correct approach is per-subdirectory rules that match whichever language convention applies. Writing these rules speculatively would produce incorrect classifications for most real polyglot projects.

**Key design challenge:** Rules in a polyglot template would need to be parameterized by subdirectory name (e.g., `backend/src/**/*.py` vs `services/src/**/*.go`). The current rule schema does not support parameterization. Future enhancement: template variables that INSPECTOR fills in during BOOTSTRAP based on detected per-directory languages.

**Promotion trigger:** A real polyglot project reaches BOOTSTRAP and needs ORGANIZE. At that point, run `custom` mode and observe which rules emerge. If a stable pattern appears across 2+ polyglot projects, write the template.

---

### Priority 4 — `book-print`

**File (future):** `book-print.json`
**Applies to:** Typesetting-heavy book or document projects using LaTeX (`.tex`) or Typst (`.typ`) as the source format. Distinct from `book-markdown` by being production-focused: the source is typesetting markup, and the workflow involves compiling to PDF. Common structure: `chapters/` (`.tex` or `.typ` files), `figures/` (images, vector graphics), `bibliography/` (`.bib` or citation files), `build/` or `output/` (compiled PDFs — ignored by ORGANIZE).

**Why deferred:** No current project uses this shape. The BOOK workflow family currently targets markdown-first manuscripts. LaTeX/Typst projects have a significantly different structure and different canonical file placement conventions.

**Key rules that would differ from `book-markdown`:**
- `chapters/**/*.tex` (or `.typ`) → IN_PLACE
- `**/*.bib` → IN_PLACE in `bibliography/` or root (convention varies)
- `**/*.sty`, `**/*.cls` → IN_PLACE at root or `styles/` (LaTeX style files)
- `build/**`, `output/**` → ignore entirely (compiled artefacts)
- `figures/**/*.{pdf,eps,svg,png}` → IN_PLACE

**Promotion trigger:** A LaTeX or Typst manuscript project that would benefit from ORGANIZE's structural enforcement.

---

## The `custom` template

`custom` is a special template that has no corresponding file in the library. It is the fallback when INSPECTOR cannot match the project to any canonical template (fewer than 2 signals matched). In `custom` mode, the user builds rules from scratch during the wizard dialog, guided by INSPECTOR's Phase 4 seed rule proposals.

`custom` mode is not a failure — it is the correct behavior for projects with unusual or unique structure. The FLAG_NEW_RULE mechanism allows custom projects to develop their own canonical ruleset over time through MAINTENANCE circuits.

---

## Template versioning

All templates in the v1.0.0 library are at version `1.0.0`. Template versions follow SemVer independently of the ORGANIZE config schema version. See `ORGANIZE_TEMPLATE_FORMAT.md §3.2` and `ORGANIZE_SCHEMA_VERSIONING.md` for versioning policy.

When a template is promoted from this roadmap, it ships at `1.0.0`. Subsequent changes follow:
- **PATCH** (`1.0.1`): Non-breaking rule changes (note field updates, minor pattern corrections)
- **MINOR** (`1.1.0`): New rules added that do not change existing rule behavior
- **MAJOR** (`2.0.0`): Rules removed or fundamentally changed; projects seeded from the prior template version may behave differently

---

*End of TEMPLATES_ROADMAP.md.*
