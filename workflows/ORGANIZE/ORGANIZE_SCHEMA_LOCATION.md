# ORGANIZE Schema Location Decision

**Task:** T1.1
**Date:** 2026-04-18
**Status:** DECIDED

---

## Decision

Schema files for the ORGANIZE workflow live **flat in `workflows/ORGANIZE/`**, not in a subdirectory.

**Decided location:**

```
workflows/ORGANIZE/
  ORGANIZE_CONFIG_SCHEMA.json      (T1.2)
  ORGANIZE_RULE_FORMAT.md          (T1.3)
  ORGANIZE_TEMPLATE_FORMAT.md      (T1.4)
  ORGANIZE_CONFIG_EXAMPLE.json     (T1.5)
  ORGANIZE_SCHEMA_VERSIONING.md    (T1.6)
```

---

## Rationale

### 1. Matches BOOK convention

The BOOK workflow family keeps its schema flat in `workflows/BOOK/`:

```
workflows/BOOK/BOOK_SPEC_SCHEMA.json
```

Consistency across workflow families reduces navigation friction. Anyone who learned where BOOK keeps its schema will find ORGANIZE's schema in the same relative position.

### 2. Small file count does not justify a subdirectory

A `schemas/` subdirectory adds hierarchy without benefit at this scale. ORGANIZE's schema artifacts total five files. Subdirectories are warranted when a directory would otherwise exceed ~15–20 files or when grouping reduces ambiguity — neither condition applies here.

### 3. Co-location with the workers that consume the schema

WORKER_INSPECTOR.md and WORKER_TRIAGE.md live in `workflows/ORGANIZE/`. Keeping schema files alongside their consumers means a reader navigating the ORGANIZE directory immediately sees the full picture: workflow spec, worker docs, schema, and example config in one place.

### 4. Rejected alternative: `workflows/ORGANIZE/schemas/`

**Why rejected:** Adds a navigation hop with no benefit. Schema discovery is simpler when the schema is adjacent to the workflow JSON. The BOOK parallel argues against it. If the ORGANIZE family grows to warrant subdirectories (e.g., a `templates/` subdir for Part 2's template library, or a `migrations/` subdir for versioning scripts), those specific subdirectories will be created for that specific purpose — schemas do not need their own bucket.

---

## Template library exception

Template files (Part 2) **do** use a subdirectory:

```
workflows/ORGANIZE/templates/
  python-lib.json
  rust-crate.json
  ...
```

This is because the template library is a collection of peer files that share a common purpose and will grow over time. The subdirectory grouping is justified there. Schema files are not a growing collection — they are a small, fixed set of infrastructure documents.

---

## Impact on other parts

- Part 2 workers read `ORGANIZE_CONFIG_SCHEMA.json` from `workflows/ORGANIZE/` directly.
- Part 4 walkthroughs reference all schema artifacts by their flat paths.
- Migration scripts (T1.6, future) will live in `workflows/ORGANIZE/migrations/` — a separate subdirectory specific to migration artifacts.

---

*End of T1.1 decision record.*
