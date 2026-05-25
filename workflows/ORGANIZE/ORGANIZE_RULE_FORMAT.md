# ORGANIZE Rule Format — Deep Dive

**Task:** T1.3
**Date:** 2026-04-18
**Workflow:** ORGANIZE_WORKFLOW v0.1.0-DRAFT
**Authoritative source:** ORGANIZE_WORKFLOW.json §`organize_json_schema.fields.rules` + WORKER_TRIAGE.md §5

---

## 1. Overview

A rule is the unit of classification logic in `.organize.json`. Every file that reaches TRIAGE is evaluated against the active rule set in priority order; the first matching rule determines the verdict. Rules are declared during BOOTSTRAP (by INSPECTOR's Phase 4 proposal + user ratification) and accumulate during MAINTENANCE (via FLAG_NEW_RULE → ratification). Rules are **append-only** — no rule is ever mutated or deleted.

### Rule object structure

```json
{
  "id":          "r7",
  "created":     "2026-04-18T14:30:00Z",
  "kind":        "glob",
  "pattern":     "**/test_*.py",
  "destination": "tests/",
  "priority":    90,
  "active":      true,
  "note":        "Python test files not already under tests/ should be moved there"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | yes | Unique; alphanumeric + hyphens; never reused |
| `created` | string (ISO 8601) | yes | When the rule was created or ratified |
| `kind` | `"glob"` \| `"regex"` \| `"hint"` | yes | How `pattern` is interpreted |
| `pattern` | string | yes | The match expression (see per-kind semantics below) |
| `destination` | string | yes | Where a matching file goes (see §5) |
| `priority` | integer ≥ 0 | yes | Evaluation order; higher = first |
| `active` | boolean | yes | Whether this rule is enforced |
| `note` | string | recommended | One-line rationale; required by convention |

---

## 2. Rule kind: `glob`

### 2.1 Syntax

Glob patterns follow standard Unix glob conventions as used by Python's `pathlib.Path.match` / `fnmatch` family:

| Token | Meaning |
|---|---|
| `*` | Matches any sequence of characters **within a single path segment** (does not cross `/`) |
| `**` | Matches **zero or more path segments** (crosses directory boundaries) |
| `?` | Matches any **single character** within a segment |
| `[abc]` | Character class — matches any of the listed characters |
| `[!abc]` | Negated class — matches any character not listed |

### 2.2 Anchoring behavior

- Patterns with a leading `**/` or no leading `/` are matched against the file's **path relative to `target_dir`**, not the absolute path.
- A pattern with no directory separator (e.g., `scratch*`) matches **any file at any depth** whose basename matches.
- A pattern with explicit directory components (e.g., `src/**/*.py`) is anchored to that directory structure.

### 2.3 Evaluation by TRIAGE

TRIAGE applies the glob pattern to the file's relative path (relative to `target_dir`). If it matches, the rule fires and the verdict is determined by the rule's `destination`.

### 2.4 Worked examples

**Example G1 — Source file placement (IN_PLACE)**

```json
{
  "id": "r1",
  "kind": "glob",
  "pattern": "src/**/*.py",
  "destination": "IN_PLACE",
  "priority": 100,
  "active": true,
  "note": "All Python source files under src/ are canonically placed; no action needed"
}
```

Matches: `src/main.py`, `src/utils/helpers.py`, `src/models/user.py`
Does not match: `main.py` (root), `tests/test_main.py` (different prefix)

Verdict when matched: `KEEP_IN_PLACE`

---

**Example G2 — Test file relocation**

```json
{
  "id": "r2",
  "kind": "glob",
  "pattern": "**/test_*.py",
  "destination": "tests/",
  "priority": 90,
  "active": true,
  "note": "Python test files not already under tests/ should be moved there"
}
```

Matches: `test_auth.py` (root), `scratch/test_edge.py`, `lib/test_db.py`
Does not match: `tests/test_main.py` (TRIAGE checks current location vs destination; already there → `KEEP_IN_PLACE`)

Verdict when matched and not already in `tests/`: `MOVE_TO:tests/`

Note: TRIAGE compares the file's current directory against the rule's destination. If they match (file is already where the rule would send it), TRIAGE emits `KEEP_IN_PLACE`.

---

**Example G3 — Scratch file quarantine**

```json
{
  "id": "r5",
  "kind": "glob",
  "pattern": "scratch*",
  "destination": ".delete",
  "priority": 30,
  "active": true,
  "note": "Any file whose name starts with 'scratch' is a temporary artefact; quarantine to .delete"
}
```

Matches: `scratch.py`, `scratch_ideas.md`, `scratchpad.txt` (at any depth, since no `/` in pattern)
Does not match: `src/scratch_module.py` would match `scratch*` on basename — TRIAGE evaluates basename against non-directory-path patterns

Verdict when matched: `QUARANTINE:.delete`

---

**Example G4 — Shell script placement**

```json
{
  "id": "r6",
  "kind": "glob",
  "pattern": "**/*.sh",
  "destination": "scripts/",
  "priority": 80,
  "active": true,
  "note": "Shell scripts not already in scripts/ should be moved there"
}
```

Matches: `deploy.sh`, `tools/build.sh`, `ci-python.sh`
Does not match: `scripts/install.sh` (already in destination)

---

**Example G5 — Loose markdown at root**

```json
{
  "id": "r8",
  "kind": "glob",
  "pattern": "*.md",
  "destination": "docs/",
  "priority": 50,
  "active": true,
  "note": "Loose markdown at project root (except README, CHANGELOG, LICENSE) should move to docs/"
}
```

Note: This rule is deliberately broad. If the project has canonical root-level `.md` files (like `README.md`, `CHANGELOG.md`), they should be handled by a higher-priority IN_PLACE rule:

```json
{
  "id": "r7",
  "kind": "glob",
  "pattern": "{README,CHANGELOG,LICENSE,CONTRIBUTING}.md",
  "destination": "IN_PLACE",
  "priority": 95,
  "active": true,
  "note": "Standard root-level documentation files are canonically at root"
}
```

Since `r7` has priority 95 > `r8`'s priority 50, `r7` evaluates first and `README.md` gets `KEEP_IN_PLACE` before `r8` can fire.

---

## 3. Rule kind: `regex`

### 3.1 Syntax

Standard regular expression syntax. TRIAGE applies the pattern to the file's path relative to `target_dir`. The regex is applied as-written — no automatic anchoring is added. If you want to anchor to the start or end, use `^` and `$` explicitly.

**Flags supported:** None in v0.1.0 — patterns are case-sensitive by default. Future flag support (case-insensitive, multiline) is a potential enhancement but not implemented.

### 3.2 Anchoring

| Pattern form | Behavior |
|---|---|
| `^tests/.*\\.py$` | Only matches Python files directly under `tests/` at root |
| `\\.pyc$` | Matches any `.pyc` file at any depth (no `^` anchor) |
| `^(src\|lib)/.*\\.rs$` | Matches Rust files under either `src/` or `lib/` at root |
| `notes\\d+\\.md` | Matches `notes` followed by one or more digits, then `.md`, anywhere in path |

Note: In JSON, backslashes in regex patterns must be escaped. `\.py$` is written as `\\.py$` in JSON.

### 3.3 When to use regex over glob

Use `regex` when:
- You need alternation: `^(src|lib)/` (glob can't do this cleanly)
- You need character-class precision: `notes\\d+\\.md` (one or more digits, not arbitrary suffix)
- You need negative lookaheads or other advanced constructs

Prefer `glob` when both work — globs are easier to read and audit.

### 3.4 Worked examples

**Example RE1 — Numbered note files to archive**

```json
{
  "id": "r10",
  "kind": "regex",
  "pattern": "notes\\d+\\.md",
  "destination": ".archive",
  "priority": 25,
  "active": true,
  "note": "Numbered note files (notes1.md, notes22.md) are historical; archive them"
}
```

Matches: `notes1.md`, `notes22.md`, `archive/notes5.md` (anywhere in path)
Does not match: `notes.md` (no digits), `notes_v2.md` (underscore, not bare digits)

---

**Example RE2 — Rust binary targets are in-place**

```json
{
  "id": "r3",
  "kind": "regex",
  "pattern": "^src/bin/[^/]+\\.rs$",
  "destination": "IN_PLACE",
  "priority": 100,
  "active": true,
  "note": "Rust binary entry points under src/bin/ are canonically placed"
}
```

Matches: `src/bin/main.rs`, `src/bin/cli.rs`
Does not match: `src/main.rs` (not under `src/bin/`), `src/bin/subfolder/tool.rs` (subdirectory inside bin)

---

**Example RE3 — Python cache files quarantine**

```json
{
  "id": "r4",
  "kind": "regex",
  "pattern": "\\.pyc$",
  "destination": ".delete",
  "priority": 20,
  "active": true,
  "note": "Compiled Python bytecode files are generated artifacts; quarantine to .delete"
}
```

Note: `.pyc` files are also typically covered by `ignore_paths` (e.g., `**/*.pyc`). If they appear in a TRIAGE batch despite ignore_paths (QUEEN belt-and-suspenders), this rule catches them.

---

**Example RE4 — Deactivated rule (append-only example)**

```json
{
  "id": "r9",
  "kind": "regex",
  "pattern": "^data/raw/.*\\.csv$",
  "destination": "data/",
  "priority": 70,
  "active": false,
  "note": "DEACTIVATED 2026-05-12: project removed data/ directory; rule no longer applies"
}
```

This rule was once valid, is now inactive, and is preserved for history. TRIAGE skips rules with `active: false`.

---

## 4. Rule kind: `hint`

### 4.1 What a hint rule is

A `hint` rule is a natural-language instruction to TRIAGE. The `pattern` field contains free-form prose describing the classification intent. TRIAGE interprets the rule using:

1. The file's **full content** (per the full-read-or-skip discipline)
2. The file's **path** and **name**
3. The **project template** context (from `.organize.json`'s `template` field)
4. The **other active rules** in the ruleset (for disambiguation)

A hint rule fires when TRIAGE's reading of the file content + path satisfies the condition described in the natural-language instruction.

### 4.2 When to use hint rules

Use `hint` when pattern-matching on path alone cannot capture the intent:

- Intent depends on **file content** (e.g., "Python files without pytest imports are not tests")
- Intent depends on **semantic meaning** (e.g., "markdown files that describe past decisions are archival; markdown files that describe current state are not")
- Intent involves **relationship to other files** (e.g., "any markdown file that is not linked from any other file in the project is orphaned")
- The pattern is too complex to express cleanly in glob or regex without becoming unreadable

### 4.3 TRIAGE interpretation procedure for hint rules

TRIAGE follows this procedure when evaluating a `kind: hint` rule:

1. **Read the file fully** (or emit `ASK_USER` if too large — the full-read-or-skip discipline applies without exception to hint rules, since content is the evidence base).
2. **Parse the hint's condition** from the natural-language `pattern` field.
3. **Gather evidence** from the file: content, imports, references, dates, markers, tone.
4. **Evaluate the condition** against the evidence. Interpret conservatively — if the condition's application is ambiguous given the evidence, **do not apply the hint rule**; fall through to the next rule or emit `ASK_USER`.
5. **If condition is satisfied with HIGH confidence:** emit the verdict prescribed by the rule's `destination`.
6. **If confidence is MEDIUM or LOW:** emit `ASK_USER` with the hint rule's text included in the rationale, so the user understands what the rule attempted to match.
7. **Never fabricate evidence.** The verdict must cite a specific observation from the file content (e.g., "file contains 'DEPRECATED:' at line 3" or "file has no `import pytest` or `test_` prefix in function names").

### 4.4 Worked examples

**Example H1 — Python files without pytest are not tests**

```json
{
  "id": "r11",
  "kind": "hint",
  "pattern": "Python files that do not import pytest and whose filename does not begin with 'test_' are not test files — do not move them to tests/ even if they happen to be adjacent to test files.",
  "destination": "IN_PLACE",
  "priority": 85,
  "active": true,
  "note": "Guards against accidentally treating utility modules near tests/ as test files"
}
```

TRIAGE evaluation procedure for a file like `tests/helpers.py`:
- Read the file fully.
- Check: does filename start with `test_`? No (`helpers.py`).
- Check: does file contain `import pytest` or `from pytest`? (Read imports section of file.)
- If no pytest import found: hint condition satisfied → `KEEP_IN_PLACE` (file is not a test; it's a helper that legitimately lives in tests/ directory).
- Evidence cited: `"filename does not start with test_; no 'import pytest' found in file"`

---

**Example H2 — Superseded design documents go to archive**

```json
{
  "id": "r12",
  "kind": "hint",
  "pattern": "Markdown files that contain a 'SUPERSEDED', 'DEPRECATED', or 'OBSOLETE' marker at the top of the document (within the first 10 lines) are historical — they describe something that has been replaced. Quarantine to .archive.",
  "destination": ".archive",
  "priority": 60,
  "active": true,
  "note": "Prose docs that explicitly self-identify as superseded should be archived"
}
```

TRIAGE evaluation:
- Read the file fully.
- Check lines 1–10 for the strings `SUPERSEDED`, `DEPRECATED`, `OBSOLETE` (case-insensitive per conservative interpretation; hint rules do not specify case sensitivity explicitly, so TRIAGE uses broad matching).
- If found: emit `QUARANTINE:.archive`. Evidence: `"file contains 'SUPERSEDED' at line 2: 'SUPERSEDED by ARCH_V2.md'"`
- If not found: hint does not fire; fall to next rule.

---

**Example H3 — Jupyter notebook organization**

```json
{
  "id": "r13",
  "kind": "hint",
  "pattern": "Jupyter notebooks (.ipynb files) that contain only markdown cells and no code cells are documentation notebooks — they should go to docs/notebooks/. Notebooks with code cells are analysis artifacts and should go to notebooks/ at the project root.",
  "destination": "notebooks/",
  "priority": 75,
  "active": true,
  "note": "Split notebooks by content type: doc-only to docs/notebooks/, code-bearing to notebooks/"
}
```

TRIAGE evaluation (for a file like `experiment_2026_01.ipynb`):
- Read the file fully (JSON structure inside `.ipynb`).
- Parse the `cells` array. Check each cell's `cell_type`: if any cell has `"cell_type": "code"` with non-empty source → code-bearing notebook.
- If code cells found: destination is `notebooks/`. If not in `notebooks/`: emit `MOVE_TO:notebooks/`.
- If only `"cell_type": "markdown"` cells: destination is `docs/notebooks/`. Emit accordingly.
- Evidence: `"notebook contains 3 code cells with non-empty source at cells[1], cells[3], cells[5]"`

Note: If the `.ipynb` is too large to read fully, TRIAGE emits `ASK_USER` per the full-read-or-skip rule. Hint rules cannot be partially evaluated.

---

**Example H4 — Orphaned reference docs (content-relationship check)**

```json
{
  "id": "r14",
  "kind": "hint",
  "pattern": "Markdown files at the project root that are not linked from README.md and are not listed in any BOOK_MANIFEST.json, SDLC TODO file, or INPROGRESS.md are likely orphaned design notes. Flag as ASK_USER with the observation that no referencing file was found.",
  "destination": "IN_PLACE",
  "priority": 45,
  "active": true,
  "note": "Orphan detection: root .md files not referenced by any project index are candidates for archival or deletion"
}
```

Note: This hint rule has `destination: IN_PLACE` but instructs TRIAGE to emit `ASK_USER` when it fires (not a direct move). This is an unusual pattern — a hint rule may express conditional logic that overrides the simple destination field. TRIAGE respects the hint's instruction and emits `ASK_USER` with the orphan observation.

This example demonstrates that `hint` rules are the most powerful and most complex kind — they require careful drafting and TRIAGE uses maximum caution when evaluating them.

---

## 5. Destinations

| Destination value | TRIAGE verdict emitted | Semantics |
|---|---|---|
| `IN_PLACE` | `KEEP_IN_PLACE` | File is at its canonical location. No move. |
| `<relative_path>` | `MOVE_TO:<path>` | File belongs at the specified path. Execute via `git mv`. |
| `.delete` | `QUARANTINE:.delete` | File is garbage-shaped. Move to `<target_dir>/.delete/<original_relative_path>`. Reversible. |
| `.archive` | `QUARANTINE:.archive` | File has historical/intellectual value but is no longer active. Move to `<target_dir>/.archive/<original_relative_path>`. |

**Nothing is ever deleted.** The destination `.delete` is a move, not an `rm`. Files in `.delete/` are user-managed after the workflow exits.

**Path mirroring in quarantine:** A file at `src/scratch.py` quarantined to `.delete` becomes `.delete/src/scratch.py` — the original relative path is mirrored under the quarantine root so provenance is retained.

---

## 6. Priority

### 6.1 Semantics

- Rules are sorted **descending by priority** before evaluation.
- **First match wins.** Once a rule fires, TRIAGE emits the verdict and stops evaluating remaining rules for that file.
- A file with no matching rule falls to the fallback procedure (see WORKER_TRIAGE.md §5, Step 2d).

### 6.2 Conventional priority ranges

| Range | Purpose |
|---|---|
| 100–110 | Highly specific canonical rules: exact source tree layout (`src/**/*.py`, `src/lib.rs`) |
| 80–99 | Standard structural rules: test placement, doc placement, script placement |
| 50–79 | Pattern rules: loose markdown routing, notebook routing |
| 20–49 | Cruft / archival rules: scratch* patterns, numbered notes |
| 1–19 | Last-resort catch-all rules (rare; usually a single broad rule with very low priority) |

### 6.3 Priority collision

If two rules have the **same priority**, evaluation order is determined by their **position in the rules array** (lower index evaluated first — earlier appended rules win ties). By convention, avoid same-priority rules that could match the same files. Use different priorities to express intent explicitly.

### 6.4 Future: `continue-matching` flag

The v0.1.0-DRAFT spec documents `continue-matching: true` as a future rule flag. When set, TRIAGE would not stop at the first match — it would apply the rule's action and continue evaluating lower-priority rules. **This flag is NOT implemented in v0.1.0** — first-match-wins is the only behavior. Documented here so implementors know the design intent.

---

## 7. Rule lifecycle

### 7.1 Creation paths

Rules enter the `.organize.json` rules array via two paths:

1. **BOOTSTRAP:** INSPECTOR proposes seed rules in Phase 4; user ratifies/edits via WIZARD_LOOP; QUEEN writes them to the rules array with the BOOTSTRAP timestamp.
2. **MAINTENANCE — FLAG_NEW_RULE:** TRIAGE observes a recurring uncovered pattern (3+ files matching no rule); emits `FLAG_NEW_RULE` with a `suggested_rule` draft; user ratifies; QUEEN appends the ratified rule to the rules array.

### 7.2 Active / inactive

- `active: true` — rule is evaluated by TRIAGE during MAINTENANCE.
- `active: false` — rule is present in the array but skipped by TRIAGE. Set when a rule becomes inapplicable (e.g., the directory it references was removed, or the rule was superseded by a more precise rule).

When deactivating a rule, always update the `note` field to record why:

```json
{
  "id": "r9",
  "kind": "glob",
  "pattern": "data/raw/*.csv",
  "destination": "data/",
  "priority": 70,
  "active": false,
  "note": "DEACTIVATED 2026-05-12: project migrated away from local data/ dir; data now in S3"
}
```

### 7.3 Why append-only matters

The rules array is the project's accumulated structural memory. Deleting a rule erases the record of why it was there and what happened to files it once governed. An inactive rule is transparent — a future reader can see the full history of structural decisions. This is especially important for `quarantine_log` cross-reference: a quarantine event references the run number; the rules active in that run are preserved in the array.

### 7.4 Adding rules in order

New rules are always **appended to the end** of the array. Their evaluation order is determined by their `priority` field, not their position. This means the rules array is a chronological record (append = history) while the priority field provides the logical evaluation order.

---

## 8. Quick reference — rule drafting checklist

When drafting a rule (INSPECTOR Phase 4 or TRIAGE FLAG_NEW_RULE):

- [ ] `id` is unique in this file and follows `r<N>` convention
- [ ] `created` timestamp is accurate
- [ ] `kind` is the simplest kind that captures the intent (prefer `glob` > `regex` > `hint`)
- [ ] `pattern` is tested against expected matches and non-matches
- [ ] `destination` is one of: `IN_PLACE`, relative path, `.delete`, `.archive`
- [ ] `priority` does not collide with existing rules for the same file patterns
- [ ] `active` is `true` for new rules
- [ ] `note` explains why this rule exists (not just what it does)
- [ ] For `hint` rules: the natural-language pattern is unambiguous and cites the evidence TRIAGE should look for

---

*End of ORGANIZE_RULE_FORMAT.md.*
