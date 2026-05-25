# ORGANIZE Template Format

**Task:** T1.4
**Date:** 2026-04-18
**Workflow:** ORGANIZE_WORKFLOW v0.1.0-DRAFT
**Relationship:** Defines the schema for files in `workflows/ORGANIZE/templates/<name>.json` (Part 2 produces the actual template library).

---

## 1. What a template is

A template is a **pre-composed ruleset for a common project shape**. Instead of INSPECTOR drafting rules from scratch during every BOOTSTRAP, the template library provides a starting point: a set of rules, ignore paths, and configuration that fit well-known project structures (Python library, Rust crate, book manuscript, etc.).

Templates are stored as JSON files under `workflows/ORGANIZE/templates/`. Each template file has a well-defined schema (this document) and is referenced by the `template` field in `.organize.json`.

The relationship between templates and project configs:

```
workflows/ORGANIZE/templates/python-lib.json
        ↓  (loaded by QUEEN during BOOTSTRAP)
<target_dir>/.organize.json
        "template": "python-lib"
        "rules": [ ...seeded from template, then user-edited... ]
```

Once a project's `.organize.json` is written, the template file is no longer consulted at runtime. The project's own `rules` array is the authoritative source of truth. Templates are only used during BOOTSTRAP (initial seeding) or when re-running BOOTSTRAP with `mode_override`.

---

## 2. Template file schema

A template file is stored at `workflows/ORGANIZE/templates/<name>.json`.

### 2.1 Required fields

| Field | Type | Description |
|---|---|---|
| `name` | string | Template identifier. Must match the filename without `.json` extension. Must be one of the enum values from ORGANIZE_CONFIG_SCHEMA.json's `template` property. |
| `version` | string | SemVer version of this template file. Template versioning is independent of config schema versioning. |
| `applies_to` | object | Description of what kind of project this template fits. |
| `default_rules` | array | Array of rule objects (same schema as ORGANIZE_CONFIG_SCHEMA.json `definitions/rule`). Seeded into the project's `.organize.json` `rules` array during BOOTSTRAP. |
| `default_ignore_paths` | array | Array of glob pattern strings. Seeded into `ignore_paths` during BOOTSTRAP. |

### 2.2 Optional fields

| Field | Type | Description |
|---|---|---|
| `notes` | string | Free-form notes about the template: conventions, rationale, known limitations. |
| `inherits_from` | string | Name of a parent template this one extends. See §4 for inheritance semantics. |

### 2.3 Example template file structure

```json
{
  "name": "python-lib",
  "version": "1.0.0",
  "applies_to": {
    "description": "Python library project with src/ layout, tests/ mirror, and pyproject.toml. Suitable for packages intended for distribution via PyPI.",
    "signals": [
      "lang:python",
      "kind:library",
      "manifest:pyproject.toml"
    ]
  },
  "default_rules": [
    {
      "id": "r1",
      "created": "2026-04-18T00:00:00Z",
      "kind": "glob",
      "pattern": "src/**/*.py",
      "destination": "IN_PLACE",
      "priority": 100,
      "active": true,
      "note": "Python source files under src/ are canonically placed"
    },
    {
      "id": "r2",
      "created": "2026-04-18T00:00:00Z",
      "kind": "glob",
      "pattern": "**/test_*.py",
      "destination": "tests/",
      "priority": 90,
      "active": true,
      "note": "Python test files not already under tests/ should be moved there"
    }
  ],
  "default_ignore_paths": [
    ".git/**",
    ".venv/**",
    "**/__pycache__/**",
    "**/*.pyc",
    ".pytest_cache/**",
    ".mypy_cache/**",
    "**/*.egg-info/**",
    "dist/**",
    "build/**",
    ".claude/**",
    ".claude-flow/**",
    ".hive-mind/**",
    ".mcp.json",
    ".delete/**",
    ".archive/**"
  ],
  "notes": "Standard python-lib template. Assumes src/ layout (src/mypkg/__init__.py). If project uses flat layout (mypkg/ at root), adjust r1 pattern.",
  "inherits_from": null
}
```

---

## 3. Field specifications

### 3.1 `name`

- Must exactly match the filename (without `.json`).
- Must be a valid `template` enum value from `ORGANIZE_CONFIG_SCHEMA.json`.
- Current valid names: `python-lib`, `python-app`, `rust-crate`, `rust-app`, `book-markdown`, `book-print`, `mixed-research`, `knowledge-base`, `polyglot`, `custom`.
- `custom` is a special case — it has no template file in the library (users build rules from scratch in the wizard).

### 3.2 `version`

- SemVer format (`MAJOR.MINOR.PATCH`).
- Template versioning is **independent** of `.organize.json` config schema versioning.
- A project's `.organize.json` is seeded from a specific template version at BOOTSTRAP time. Subsequent template version bumps do not retroactively change existing project configs.
- Future enhancement (not v0.1.0): a `template_version_at_bootstrap` field in `.organize.json` could record which template version was used, enabling QUEEN to detect when a newer template is available and offer to merge new default rules.

### 3.3 `applies_to`

An object describing what project shape this template targets:

```json
{
  "applies_to": {
    "description": "Human-readable description of the project shape",
    "signals": ["signal1", "signal2"]
  }
}
```

- `description` (string, required): Free text explaining the project type. Used by INSPECTOR when proposing a template to the user in the wizard.
- `signals` (array of strings, optional): Structured signals from INSPECTOR's Phase 2 that should match for this template to be proposed. Follows the signal vocabulary from WORKER_INSPECTOR.md §4 Phase 2 (`lang:python`, `kind:library`, `manifest:pyproject.toml`, etc.).

INSPECTOR matches observed signals against template `applies_to.signals` to score template candidates. The highest-scoring match is proposed; ties are broken by specificity (more signals matched = more specific fit).

### 3.4 `default_rules`

Array of rule objects. Each rule must conform to the rule schema from `ORGANIZE_CONFIG_SCHEMA.json §definitions/rule`. All rules in a template file should have `active: true` — they are starting-point rules, not historical inactive records.

**Rule IDs in templates:** Template rule IDs use a `t` prefix (e.g., `t1`, `t2`) to distinguish them from project-specific rules which use `r` prefix (e.g., `r1`, `r2`). When QUEEN seeds a project's `.organize.json` from a template, it rewrites the IDs to `r1`, `r2`, ... in append order. This prevents ID collisions between template origins.

**Rule ordering in templates:** Template rules are ordered by priority (descending) in the file for readability. TRIAGE sorts by priority at runtime regardless, but the template file ordering makes the intent clear.

### 3.5 `default_ignore_paths`

Array of glob pattern strings. These are merged into the project's `ignore_paths` array during BOOTSTRAP. The root invariant paths (`.claude/**`, `.claude-flow/**`, `.hive-mind/**`, `.mcp.json`) should be present in every template's `default_ignore_paths` as documentation, even though QUEEN enforces them implicitly.

### 3.6 `notes`

Free-form string. Should explain:
- Known limitations of the template's rules
- Common customization points (what the user will likely want to change)
- Project convention assumptions baked into the rules
- Relationship to other templates if relevant

### 3.7 `inherits_from`

Optional string. Name of the parent template this template extends. See §4.

---

## 4. Template inheritance

### 4.1 Model

A template may declare `inherits_from: "<parent_name>"`. This creates a parent-child relationship:

```
base-python  (hypothetical base)
  └── python-lib  (inherits_from: "base-python")
  └── python-app  (inherits_from: "base-python")
```

Inheritance semantics:
- **Rules:** Child's rules take precedence. On ID collision between parent and child rules, the child's rule replaces the parent's. Rules with unique IDs from both parent and child are merged.
- **Ignore paths:** Union of parent and child `default_ignore_paths`. No deduplication needed (QUEEN deduplicates at application time).
- **`applies_to.signals`:** Child's signals replace parent's entirely. Child templates are more specific and their signal matching is defined independently.
- **`notes`:** Child's notes are standalone (not concatenated with parent's). If child wants to reference parent context, it does so explicitly in its `notes` text.

### 4.2 Resolution procedure (QUEEN during BOOTSTRAP)

When QUEEN loads a template with `inherits_from` set:

1. Load the parent template (recursively, up the chain).
2. Check for cycles: if any template in the chain has already been visited, abort with `ESCALATE` — circular inheritance is a workflow bug.
3. Apply child rules over parent rules: merge arrays; child rules with matching IDs override parent rules with the same ID.
4. Union ignore paths from all ancestors.
5. Present the resolved (merged) template to the user in the wizard.

### 4.3 Cycle detection

Cycles are detected by tracking visited template names during resolution:

```
visited = []
current = <child>
while current.inherits_from:
  if current.inherits_from in visited → CYCLE DETECTED → ESCALATE
  visited.append(current.name)
  current = load(current.inherits_from)
```

A cycle produces an `ESCALATE` verdict to the user with the cycle path listed.

### 4.4 v0.1.0 inheritance status

In the v0.1.0 template library (Part 2), no templates use inheritance — each template is standalone. The `inherits_from` field is present in the schema for future use. The cycle detection and merge logic is specified here so implementors building future templates have a defined contract.

---

## 5. How INSPECTOR uses templates during BOOTSTRAP Phase 3

INSPECTOR's Phase 3 (Template Proposal) operates as follows:

1. **Collect observed signals** from Phase 2 (language, kind, structure quality, cruft patterns, etc.).
2. **Score each template** in the library against observed signals: count how many of the template's `applies_to.signals` match the observed signals.
3. **Propose the highest-scoring match** as the recommended template. If tie, prefer more specific (more signals matched).
4. **If no template scores ≥ 2 signal matches:** propose `custom` template with a note that the project shape did not match any canonical template.
5. **Record confidence:** if the top template scores ≥ 4 matched signals → `template_confidence: HIGH`. 2–3 matched signals → `MEDIUM`. < 2 → `LOW`.

INSPECTOR presents this in its proposal block:

```
## proposed_template
name: python-lib
rationale: Detected lang:python + kind:library + manifest:pyproject.toml + structure:src_tests. All four signals match python-lib template.
```

### 5.1 QUEEN's role after INSPECTOR proposes

1. QUEEN reads the proposed template name.
2. QUEEN loads `workflows/ORGANIZE/templates/<name>.json` if it exists.
3. QUEEN resolves inheritance (if any).
4. QUEEN presents the resolved `default_rules` and `default_ignore_paths` to the user in the wizard's `propose_seed_rules` and `propose_ignore_paths` stages.
5. User accepts, edits, or overrides.
6. QUEEN writes the ratified rules to `.organize.json`.

If the template file does not exist (e.g., a future template name not yet in the library), QUEEN falls back to INSPECTOR's inline proposed rules from the Phase 4 proposal block. QUEEN notes the missing template file in the wizard dialog so the user is aware.

---

## 6. Template file location and naming

```
workflows/ORGANIZE/templates/
  python-lib.json
  python-app.json
  rust-crate.json
  book-markdown.json
  mixed-research.json
  knowledge-base.json
  TEMPLATES_INDEX.md      (created in Part 2 — T2.1)
  TEMPLATES_ROADMAP.md    (created in Part 2 — T2.7)
```

- Template filename must be `<name>.json` where `<name>` matches the `name` field inside the file.
- Template file must be valid JSON (no comments, no trailing commas).
- Template rules must validate against the rule schema defined in `ORGANIZE_CONFIG_SCHEMA.json §definitions/rule`.

---

## 7. Template validation procedure

A template file can be validated with:

```bash
python3 -c "
import json, jsonschema

# Load the config schema to extract the rule definition
with open('workflows/ORGANIZE/ORGANIZE_CONFIG_SCHEMA.json') as f:
    config_schema = json.load(f)

# Load the template
with open('workflows/ORGANIZE/templates/python-lib.json') as f:
    template = json.load(f)

# Validate each rule against the rule schema
rule_schema = {
    '\$schema': 'http://json-schema.org/draft-07/schema#',
    **config_schema['definitions']['rule'],
    'definitions': config_schema['definitions']
}

for i, rule in enumerate(template['default_rules']):
    jsonschema.validate(rule, rule_schema)
    print(f'Rule {i} ({rule[\"id\"]}): OK')

print('Template valid.')
"
```

In Part 4 walkthroughs, this validation is run against all 5 canonical templates to confirm they conform.

---

## 8. Relationship to INSPECTOR's Phase 4 seed rules

There is an intentional overlap between:

- **Template's `default_rules`** — the canonical starting-point rules from the library file
- **INSPECTOR's Phase 4 proposed rules** — rules derived from direct observation of the specific project

During BOOTSTRAP, QUEEN **merges** both sources:

1. Load template's `default_rules` as the base.
2. Overlay INSPECTOR's project-specific proposed rules (INSPECTOR may propose fewer rules but they are grounded in what the project actually has).
3. Present the merged set to the user in the wizard.
4. User ratifies, edits, adds, or removes from the merged set.
5. QUEEN writes the final ratified set to `.organize.json`.

INSPECTOR's observations take precedence over template defaults for any rule where they conflict — the specific beats the general.

---

*End of ORGANIZE_TEMPLATE_FORMAT.md.*
