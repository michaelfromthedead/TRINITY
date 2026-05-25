# BOOK_MANIFEST_SCOPE

**Version:** 1.0.0
**Date:** 2026-04-18
**Task:** T4.5.2
**Covers:** Schema extension — `scope` section for BOOK_MANIFEST.json

---

## Purpose

This document specifies the `scope` section added to BOOK_MANIFEST.json in TRIAGE v1.1.0. The scope section enables authors to declare, at the start of a book project, the intended chapter structure — the shape the completed book should take, independent of what source material currently exists.

The scope section is the input that BOOK_TRIAGE v1.1 uses to perform per-chapter state classification (T4.5.4), and that BOOK_COMPLETION uses to route each chapter to the appropriate workflow.

**Backward compatibility:** Manifests without a `scope` section are still valid. BOOK_TRIAGE v1.1 degrades gracefully to v1.0.0 aggregate-only behavior when no scope is present.

---

## 1. Schema Definition

### 1.1 Top-level addition to BOOK_MANIFEST.json

```json
{
  "version": "1.1.0",
  "title": "...",
  "author": "...",
  "...": "...",

  "scope": {
    "scope_declared_at": "<ISO 8601 timestamp>",
    "scope_revision_log": [],
    "intended_chapters": [
      {
        "index": 1,
        "title": "<chapter title>",
        "slug": "CH_01_<TITLE_SLUG>",
        "target_topic": "<1-3 sentence description of what this chapter covers>",
        "target_length_words": 4000,
        "rationale": "<why this chapter exists in the book — its function in the arc>",
        "status": "MISSING"
      }
    ]
  }
}
```

### 1.2 Field definitions

#### `scope` object

| Field | Type | Required | Description |
|---|---|---|---|
| `scope_declared_at` | ISO 8601 string | yes | When the author first declared this scope. Human fills at manifest creation time or after TRIAGE runs. |
| `scope_revision_log` | array of revision entries | yes (may be empty) | Ordered list of scope revisions (see §1.4). Empty array `[]` if scope has never been revised. |
| `intended_chapters` | array of chapter intent entries | yes | The author's authoritative statement of the book's intended chapter structure. |

#### `intended_chapters` entry

| Field | Type | Required | Description |
|---|---|---|---|
| `index` | integer (1-based) | yes | Chapter position in the finished book. Determines ordering in STRUCTURE.md and `chapters/` directory. |
| `title` | string | yes | Human-readable chapter title. Will become the `<Full Chapter Title>` in chapter files. |
| `slug` | string | yes | File-system slug following `CH_<NN>_<TITLE>` convention. Zero-padded to 2 digits minimum. Author sets; TRIAGE may suggest. |
| `target_topic` | string | yes | 1–3 sentence description of what this chapter covers. Used by DRAFTER as primary scope input when notes are absent. Must be specific enough that DRAFTER can produce a chapter outline from it alone. |
| `target_length_words` | integer | yes | Author's target word count for this chapter. DRAFTER must hit this within ±20%. If no target is set, DRAFTER uses a genre-appropriate default from BOOK_MANIFEST.json `structure.genre`. |
| `rationale` | string | yes | Why this chapter exists — its role in the book's arc, what it establishes for later chapters, and what it expects readers to already know. Used by DRAFTER for prerequisite-chain awareness. |
| `status` | string | TRIAGE fills | The chapter's current state, set by TRIAGE v1.1 per-chapter classification. One of the taxonomy values from `CHAPTER_STATE_TAXONOMY.md`. Human may override. |

#### `scope_revision_log` entry

```json
{
  "revised_at": "<ISO 8601 timestamp>",
  "revised_by": "human | <workflow name>",
  "changes": "<description of what changed in the scope>",
  "chapters_added": [],
  "chapters_removed": [],
  "chapters_modified": []
}
```

Scope revision log is append-only. Entries are never deleted. If scope changes significantly (chapters added/removed), a new entry is appended. The log exists because COMPLETION may have already partially processed the book when scope changes; the log provides audit trail for which chapters were processed under which scope.

---

## 2. Lifecycle

### 2.1 Who sets the scope

**The human sets the scope.** TRIAGE v1.1 does not generate scope from source material. TRIAGE v1.0 aggregate assessment gives the human enough information to write the `intended_chapters` list, but the list itself is an authorial decision.

Typical workflow:
1. Human runs BOOK_TRIAGE. Gets back BOOK_MANIFEST.json with aggregate assessment + per-file classifications.
2. Human opens BOOK_MANIFEST.json and adds a `scope` section declaring the intended chapters.
3. Human saves and re-runs BOOK_TRIAGE with `scope` present — or QUEEN re-runs the per-chapter classification step (Step 3.5 in TRIAGE v1.1 execution) against the declared scope.
4. TRIAGE v1.1 fills in `status` on each `intended_chapters` entry based on source material found.

### 2.2 Who reads the scope

- **BOOK_TRIAGE v1.1:** reads `intended_chapters` to perform per-chapter state classification.
- **BOOK_COMPLETION:** reads `intended_chapters` + `triage.per_chapter_state` to build the routing plan.
- **DRAFTER:** reads the specific `intended_chapters[i]` entry for its target chapter as primary scope input.
- **BOOK_STORYBOARD:** reads `intended_chapters` for expected chapter count and rationale (helps QA_STORYBOARD verify coverage).
- **BOOK_EDITORIAL:** reads `intended_chapters` indirectly through chapter frontmatter `target_topic` field.

### 2.3 Who modifies the scope

**Only the human modifies scope.** Workflows read it; they do not overwrite `intended_chapters` entries. BOOK_TRIAGE v1.1 fills in `status` fields on each entry (this is its sole write to the scope section). Workflows track their own per-chapter state in `triage.per_chapter_state` (§3 in BOOK_TRIAGE.json manifest_schema).

---

## 3. Sample BOOK_MANIFEST.json with scope section

```json
{
  "version": "1.1.0",
  "title": "The Geometry of Spin",
  "author": "Michael",
  "project_root": "/path/to/project",

  "triage": {
    "performed_at": "2026-04-18T10:00:00Z",
    "file_count": 23,
    "quality_assessment": "rough",
    "recommended_entry": "BOOK_COMPLETION",
    "notes": "Mix of notes files for chapters 1, 2, 4, 5, 7; chapters 3, 6, 8-12 absent.",
    "per_chapter_state": [
      {
        "chapter_index": 1,
        "chapter_title": "The Problem of Rotation",
        "detected_state": "NOTES_ONLY",
        "confidence": "high",
        "source_files": ["source/rotation_notes.md", "source/draft_intro.md"],
        "notes": "Research notes present; no structured prose"
      },
      {
        "chapter_index": 3,
        "chapter_title": "Spinor Fields",
        "detected_state": "MISSING",
        "confidence": "high",
        "source_files": [],
        "notes": "No source material found"
      }
    ]
  },

  "scope": {
    "scope_declared_at": "2026-04-18T09:00:00Z",
    "scope_revision_log": [],
    "intended_chapters": [
      {
        "index": 1,
        "title": "The Problem of Rotation",
        "slug": "CH_01_PROBLEM_OF_ROTATION",
        "target_topic": "Establishes why classical rotation descriptions are insufficient. Introduces the reader to the gap between intuitive rotation and the mathematical structure needed to describe quantum spin.",
        "target_length_words": 5000,
        "rationale": "Opens the manuscript. Must establish the reader's conceptual need before any formalism is introduced. No prerequisites within the book.",
        "status": "NOTES_ONLY"
      },
      {
        "index": 3,
        "title": "Spinor Fields",
        "slug": "CH_03_SPINOR_FIELDS",
        "target_topic": "Defines spinor fields as mathematical objects, develops the spinor space, and connects to the SU(2) structure established in chapter 2.",
        "target_length_words": 6000,
        "rationale": "Central formalism chapter. Requires chapter 2 (SU(2) introduction). Establishes spinor language used in chapters 4-8.",
        "status": "MISSING"
      }
    ]
  },

  "templates": {
    "mode": "bundle",
    "bundle": "BUNDLE_SPIN_OF_GRAVITY"
  },

  "structure": {
    "genre": "academic-exploratory",
    "target_audience": "Graduate physics students familiar with quantum mechanics",
    "estimated_chapters": 12,
    "front_matter": ["title_page", "copyright", "toc", "preface"],
    "back_matter": ["bibliography", "index"],
    "special_elements": ["equations", "figures"]
  },

  "production": {
    "target_format": "lulu",
    "trim_size": "6x9",
    "color_interior": false,
    "template_id": null
  }
}
```

---

## 4. Backward compatibility

### 4.1 Manifests without scope (v1.0.0 behavior)

When BOOK_TRIAGE v1.1 reads a manifest without a `scope` section:
- `step_3_per_file_classification` runs normally (classify each file as fragment/draft/chapter/polished)
- `step_4_aggregate_assessment` runs normally (rough/structured/storyboarded/polished)
- Step 3.5 (per-chapter classification against scope) is skipped entirely
- `triage.per_chapter_state` is not populated in the output manifest
- BOOK_COMPLETION will refuse engagement if `scope` is absent — it requires per-chapter state to route
- BOOK_CONSOLIDATION, BOOK_STORYBOARD, BOOK_EDITORIAL, BOOK_PRODUCTION all function normally without scope

### 4.2 Manifests with partial scope

If `intended_chapters` is present but some entries lack `target_topic` or `target_length_words`:
- TRIAGE fills in `status` based on what it can assess
- DRAFTER will flag missing `target_topic` as a blocking pre-condition (cannot draft without knowing the chapter's topic)
- DRAFTER will use a genre-default word count if `target_length_words` is absent

---

## 5. Genre-default word counts

When `target_length_words` is not specified and DRAFTER must estimate, the following defaults apply based on `BOOK_MANIFEST.json structure.genre`:

| Genre | Default chapter length |
|---|---|
| `academic-exploratory` | 5,000 words |
| `academic-metastudy` | 7,000 words |
| `hard-sci-fi` | 4,000 words |
| `math-exposition` | 6,000 words |
| `science-theory` | 5,000 words |
| `other` / unrecognized | 5,000 words |

These are starting-point estimates. DRAFTER always reports the actual word count of its output and flags significant deviation from the target.

---

*End of BOOK_MANIFEST_SCOPE.md*
