# ORGANIZE Template Library — Index

**Task:** T2.1
**Date:** 2026-04-18
**Workflow:** ORGANIZE_WORKFLOW v0.1.0-DRAFT
**Location:** `workflows/ORGANIZE/templates/`

---

## What this index is

This index lists all canonical template files in the ORGANIZE template library. Templates are pre-composed rulesets for common project shapes. During BOOTSTRAP, INSPECTOR proposes the best-matching template based on detected project signals; QUEEN loads the template file and seeds the project's `.organize.json` with its `default_rules` and `default_ignore_paths`.

For template format specification, see `workflows/ORGANIZE/ORGANIZE_TEMPLATE_FORMAT.md`.
For deferred templates and promotion criteria, see `TEMPLATES_ROADMAP.md` (this directory).

---

## Template library (v1.0.0 — initial release)

| Template | File | Applies to | Signal match |
|---|---|---|---|
| `python-lib` | `python-lib.json` | Python library project with `src/` + `tests/` layout and `pyproject.toml` at root | `lang:python` + `kind:library` + `manifest:pyproject.toml` |
| `rust-crate` | `rust-crate.json` | Rust library crate with `src/lib.rs`, `tests/`, `benches/`, `examples/` | `lang:rust` + `kind:library` + `manifest:Cargo.toml` |
| `book-markdown` | `book-markdown.json` | Manuscript project processed by the BOOK workflow family; chapters in `chapters/CH_*.md` | `kind:book-markdown` + `workflow_artefacts:BOOK_MANIFEST.json` |
| `mixed-research` | `mixed-research.json` | Applied research project with code (`src/`), prose (`docs/`), data (`data/`), and notebooks (`notebooks/`) coexisting | `kind:mixed-research` + `lang:python` + `structure:src_docs_data` |
| `knowledge-base` | `knowledge-base.json` | Topic-organized markdown wiki or knowledge base; no code expected; topic directories each have a `README.md` | `kind:knowledge-base` + `structure:topic_dirs_with_readme` |

---

## Template descriptions

### `python-lib`

A Python library intended for distribution (PyPI) or internal reuse. Assumes `src/` layout (`src/<pkg>/__init__.py`), test files in `tests/`, documentation in `docs/`, and utility scripts in `scripts/`. Shell scripts outside `scripts/` are moved there. Loose markdown at root (excluding README, CHANGELOG, LICENSE) is routed to `docs/`. Numbered note files are archived. Scratch/tmp files are quarantined to `.delete`.

**Rules:** 10 rules. **Ignore paths:** 15 entries.

---

### `rust-crate`

A Rust library crate managed by Cargo. Key invariant: `Cargo.toml` and `Cargo.lock` are load-bearing at root and must not be moved. Rust source files under `src/` stay in place. Integration tests go in `tests/`, benchmarks in `benches/`, usage examples in `examples/`. Stray `test_*.rs` files outside `tests/` are proposed for relocation. Backup and `old_*` files are candidates for `.archive`.

**Rules:** 10 rules. **Ignore paths:** 10 entries (including `target/**` and `Cargo.lock`).

---

### `book-markdown`

A manuscript project processed by the BOOK workflow family. Chapter files under `chapters/CH_*.md`, front matter under `front/`, back matter under `back/`, drafts under `drafts/`, RDC source under `source/`. BOOK and RDC workflow artefacts (`BOOK_MANIFEST.json`, `STRUCTURE.md`, `MASTER.md`, `PEDAGOGY.md`, `EVALUATIONS.md`, `INVENTORY.md`, `INPROGRESS.md`, `PROJECT.md`) are load-bearing and must not be moved. Loose markdown at root always triggers ASK_USER — authorial judgment required. LULU_PIPELINE output in `output/` is ignored.

**Rules:** 11 rules. **Ignore paths:** 8 entries.

Cross-reference: BOOK_WORKFLOW_DISSERTATION.md §14 for canonical project layout.

---

### `mixed-research`

Applied research projects where code, prose, and data coexist. This template is intentionally ASK_USER-heavy because mixed projects have high per-file judgment requirements — the boundary between active experiment and orphaned scratch work is rarely clear from file path alone. Loose Python files at root are proposed for `src/` with ASK_USER confirmation. Loose markdown always requires ASK_USER. Loose data files (CSV, JSON, Parquet) always require ASK_USER. Expect FLAG_NEW_RULE to surface project-specific patterns over time.

**Rules:** 11 rules. **Ignore paths:** 14 entries (Python + Jupyter patterns).

---

### `knowledge-base`

Topic-organized markdown wiki or knowledge base with no code. Topic placement is authorial — ORGANIZE cannot reliably determine which topic directory a loose markdown file belongs in. This template is ASK_USER-heavy by design. Per-topic `README.md` files and existing topic subtree files are kept IN_PLACE. Loose root-level markdown always triggers ASK_USER. Use FLAG_NEW_RULE to promote recurring topic-placement patterns to explicit rules over time.

**Rules:** 7 rules. **Ignore paths:** 9 entries (including `.obsidian/**` for Obsidian vaults).

---

## How INSPECTOR selects a template

INSPECTOR (the BOOTSTRAP worker) compares its detected signals against each template's `applies_to.signals` array. The template with the most matching signals is proposed. If the top template scores fewer than 2 signal matches, INSPECTOR proposes `custom` — the user builds rules from scratch during the wizard.

Signal confidence levels:
- **HIGH:** 4+ signals matched
- **MEDIUM:** 2–3 signals matched
- **LOW:** fewer than 2 signals matched → propose `custom`

---

## Adding a new template

1. Write the template file at `workflows/ORGANIZE/templates/<name>.json`
2. Ensure `name` matches the filename (without `.json`)
3. Add the `name` to the `template` enum in `ORGANIZE_CONFIG_SCHEMA.json`
4. Add an entry to this index
5. Remove from `TEMPLATES_ROADMAP.md` if it was a deferred template

Validate with:

```bash
python3 -c "import json; json.load(open('workflows/ORGANIZE/templates/<name>.json')); print('JSON valid')"
```

---

## Future templates

See `TEMPLATES_ROADMAP.md` for deferred templates and their promotion criteria.

---

*End of TEMPLATES_INDEX.md.*
