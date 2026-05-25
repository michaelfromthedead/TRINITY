# FINDING_FORMAT — Unified Editorial Finding Schema

**Version:** 1.0.0
**Status:** Active
**Used by:** JUNIOR_VOICE, JUNIOR_CONCEPT, JUNIOR_STYLE, JUNIOR_FLOW, EDITORIAL_SYNTHESIS, SENIOR_SANITY, SENIOR_FINAL, REVISION
**Produced by:** T6.6 (Part 6, BOOK Editorial Buildout)

This document specifies the unified schema for all editorial findings in the BOOK_EDITORIAL pipeline. Every finding, from every worker, at every stage, must conform to this schema. Consistency is load-bearing: SENIOR_SANITY, SENIOR_FINAL, and REVISION depend on structured fields to process findings programmatically and at scale.

---

## 1. Schema

Every finding is a YAML block:

```yaml
finding:
  id: <string>
  axis: <VOICE | CONCEPT | STYLE | FLOW | CROSS_AXIS>
  severity: <Critical | High | Medium | Low>
  chapter: <string>
  section: <string | null>
  paragraph_range: <string | null>
  passage_quote: <string>
  audit_checklist_item_ref: <string | null>
  violation_description: <string>
  correction_guidance: <string | null>
  axes_interacting: <list | null>
  originating_worker: <string | null>
  synthesis_id: <string | null>
```

---

## 2. Field Definitions

### 2.1 `id` — Required

**Type:** string
**Format:** `<WORKER_PREFIX>-<NNN>` where NNN is a zero-padded three-digit integer per-report.

Worker prefixes:
- `JV` — JUNIOR_VOICE findings
- `JC` — JUNIOR_CONCEPT findings
- `JS` — JUNIOR_STYLE findings
- `JF` — JUNIOR_FLOW findings
- `SYN` — EDITORIAL_SYNTHESIS cross-axis findings (new findings only)
- `SF` — SENIOR_FINAL new findings

When EDITORIAL_SYNTHESIS passes through junior findings, the original ID is preserved and a `synthesis_id` cross-reference is added (see §3.2).

Example: `JV-001`, `JC-014`, `SYN-003`

**Rule:** IDs must be unique within a single report. IDs from different workers may share a number (JV-001 and JC-001 are distinct findings). The combination of worker prefix and number is globally unique within a pipeline run.

---

### 2.2 `axis` — Required

**Type:** enum
**Values:**

| Value | Meaning |
|---|---|
| `VOICE` | Finding from or about the VOICE template axis |
| `CONCEPT` | Finding about concept consistency (template-independent) |
| `STYLE` | Finding from or about the STYLE or PROSE template axis |
| `FLOW` | Finding from or about narrative/structural flow |
| `CROSS_AXIS` | Finding that involves the interaction of two or more axes |

**Rule:** JUNIOR findings always use a single-axis value matching their role. CROSS_AXIS is exclusive to EDITORIAL_SYNTHESIS and SENIOR_FINAL.

---

### 2.3 `severity` — Required

**Type:** enum
**Values:** `Critical | High | Medium | Low`

**Severity taxonomy:**

| Severity | Definition | Common triggers |
|---|---|---|
| **Critical** | Violates the template's Contract directly. The passage fundamentally contradicts the pedagogical, genre, or craft commitment the template establishes. | Anti-Pattern present; Contract sentence explicitly violated; entire chapter opens wrong |
| **High** | Violates a Characteristic Pattern (pattern absent where expected) or triggers an Anti-Pattern. The passage is recognizably non-conformant but does not invalidate the chapter. | Missing Characteristic Pattern; Anti-Pattern confirmed; concept used before definition |
| **Medium** | Partially violates a checklist item. The passage is borderline — mostly conformant but with a notable deviation. | Borderline case on an audit item; partially present pattern |
| **Low** | Minor deviation that does not break the template's contract but would benefit from correction. | Single awkward sentence; one parenthetical too many; minor notation inconsistency |

**Cross-axis severity note:** cross-axis findings from EDITORIAL_SYNTHESIS are often High or Critical because they compound violations from two or more axes. When a finding is individually Low/Medium on each axis but the compound effect is High, severity should be set to High with the compounding noted in `violation_description`.

**Downgrade rule:** SENIOR_SANITY may downgrade severity (e.g., from High to Low) when ruling a finding `real` but minor. SENIOR_SANITY does not upgrade severity.

---

### 2.4 `chapter` — Required

**Type:** string
**Format:** chapter slug matching the file in `chapters/`, e.g., `CH_03_SPIN_GEOMETRY`

---

### 2.5 `section` — Optional

**Type:** string or null
**Format:** section identifier, e.g., `"3.2"` or `"The Rotation Group"` or `null` if finding is chapter-level (applies to the whole chapter, not a locatable section).

---

### 2.6 `paragraph_range` — Optional

**Type:** string or null
**Format:** `"N"` (single paragraph) or `"N-M"` (paragraph range), counting from 1 within the section. Null if finding is section-level or chapter-level.

Example: `"3-5"` means paragraphs 3 through 5 of the named section.

---

### 2.7 `passage_quote` — Required

**Type:** string
**Content:** Verbatim excerpt from the manuscript. Must be enough text to locate the passage unambiguously and to demonstrate the violation. Minimum: the sentence(s) that directly violate the checklist item. Maximum: a full paragraph if the violation spans it.

**Rule:** Never paraphrase. Verbatim only. If the violation spans multiple non-contiguous sentences, use ellipsis (`...`) and quote the relevant parts.

---

### 2.8 `audit_checklist_item_ref` — Conditional

**Type:** string or null
**Format:** `[AXIS:key]` exactly matching an audit checklist item in the declared template. Examples: `[VOICE:question_before_answer]`, `[STYLE:citation_inline_parenthetical]`, `[PROSE:sentence_length_ceiling]`, `[BUNDLE:show_compare_ask_complete]`

**When required:** for all findings that derive from a template-based check (all VOICE, STYLE, FLOW findings; PERSONA findings; BUNDLE findings).

**When null:** JUNIOR_CONCEPT findings are template-independent content checks. For CONCEPT axis findings, `audit_checklist_item_ref` is always null. For CROSS_AXIS findings from EDITORIAL_SYNTHESIS that do not map to a single checklist item, this is null (the interaction is described in `violation_description`).

**Conformance rule:** a finding that references a non-existent `[AXIS:key]` is a conformance error. The referenced key must appear verbatim in the loaded template's Audit Checklist section.

---

### 2.9 `violation_description` — Required

**Type:** string
**Length:** 1-3 sentences.
**Content:** Plain prose explaining why the quoted passage violates the checklist item (or the concept, flow, or cross-axis constraint). Must be specific enough that a REVISION worker can understand what is wrong without re-reading the checklist item.

---

### 2.10 `correction_guidance` — Optional

**Type:** string or null

**Who populates this:**
- **JUNIOR workers:** leave null unless the correction is obvious and unambiguous.
- **EDITORIAL_SYNTHESIS:** leave null (synthesis is not a correction worker).
- **SENIOR_FINAL:** populates this for every finding on a REVISE verdict. This is the actionable revision directive REVISION receives.
- **REVISION:** does not populate this field; instead produces a revision log.

**Content when populated:** specific, actionable guidance. What should the passage do instead? Reference the specific template pattern it should emulate. Do not write replacement prose — write the directive for REVISION.

Example: `"Rewrite the chapter opener to begin with a concrete observable phenomenon before stating any theorem. See VOICE_SOCRATIC Pattern 1 for the observation-first model."`

---

### 2.11 `axes_interacting` — Conditional

**Type:** list of axis values, or null

**When required:** present and non-null only for `CROSS_AXIS` findings.

**Format:** list of two or more axis names. All items must be from the `axis` enum values (excluding `CROSS_AXIS` itself).

Example: `axes_interacting: [VOICE, STYLE]`

**When null:** for all non-CROSS_AXIS findings.

---

### 2.12 `originating_worker` — Optional

**Type:** string or null

Populated by EDITORIAL_SYNTHESIS when passing through junior findings. Value is the junior role name: `JUNIOR_VOICE`, `JUNIOR_CONCEPT`, `JUNIOR_STYLE`, `JUNIOR_FLOW`. Null for cross-axis findings that originate in SYNTHESIS or SENIOR_FINAL.

---

### 2.13 `synthesis_id` — Optional

**Type:** string or null

When EDITORIAL_SYNTHESIS passes through a junior finding, it preserves the original `id` but adds a `synthesis_id` that keys the finding to the synthesis report's internal numbering. Format: `SYN-PASS-<NNN>`.

Example: a finding originally `JV-003` becomes, in the synthesis report, finding `JV-003` with `synthesis_id: SYN-PASS-001`. This allows traceback from the synthesis report to the junior report.

Null for findings that do not pass through SYNTHESIS (e.g., SENIOR_FINAL new findings).

---

## 3. Required vs Optional Fields Summary

| Field | Required | Optional / Conditional | Notes |
|---|---|---|---|
| `id` | yes | | unique per report |
| `axis` | yes | | |
| `severity` | yes | | |
| `chapter` | yes | | |
| `section` | | optional | null if chapter-level |
| `paragraph_range` | | optional | null if not paragraph-level |
| `passage_quote` | yes | | verbatim always |
| `audit_checklist_item_ref` | | conditional | required for template-derived findings; null for CONCEPT axis |
| `violation_description` | yes | | |
| `correction_guidance` | | optional | SENIOR_FINAL populates for REVISE verdicts |
| `axes_interacting` | | conditional | required when axis == CROSS_AXIS |
| `originating_worker` | | optional | SYNTHESIS populates for pass-through findings |
| `synthesis_id` | | optional | SYNTHESIS populates for pass-through findings |

---

## 4. Serialization Format

**Canonical format: YAML block.** Every finding is a YAML document block. Reports are YAML documents containing a list of findings under a top-level `findings:` key, followed by a `summary:` block.

```yaml
---
report:
  worker: JUNIOR_VOICE
  date: 2026-04-18
  chapter_count_reviewed: 3
  template_loaded: BUNDLE_SPIN_OF_GRAVITY (VOICE_SOCRATIC)

findings:

  - finding:
      id: JV-001
      axis: VOICE
      severity: Critical
      chapter: CH_02_SPIN_PRECESSION
      section: "2.1"
      paragraph_range: "1-2"
      passage_quote: |
        "The phenomenon of spin precession is explained by Larmor's theorem, which states
        that a magnetic dipole in an external field precesses at the Larmor frequency.
        This frequency is proportional to the field strength."
      audit_checklist_item_ref: "[VOICE:no_top_down_declaration]"
      violation_description: >
        The passage states Larmor's theorem and its consequence without any prior observation,
        question, or exploration that motivates the reader's need to know it. The Contract
        requires that no statement of principle appear without prior questioning or observation.
      correction_guidance: null
      axes_interacting: null
      originating_worker: null
      synthesis_id: null

  - finding:
      ...

summary:
  total: <N>
  by_severity:
    Critical: <N>
    High: <N>
    Medium: <N>
    Low: <N>
  by_chapter:
    CH_01: <N>
    CH_02: <N>
    CH_03: <N>
```

**Alternative for inline use:** a finding may also be rendered as a compact markdown table row for human-readable review. The YAML block is the canonical machine-readable format; the markdown table is for review display only.

Markdown display format (for human review, not for downstream processing):

| ID | Axis | Severity | Chapter | Section | Para | Checklist Item | Description |
|---|---|---|---|---|---|---|---|
| JV-001 | VOICE | Critical | CH_02 | 2.1 | 1-2 | `[VOICE:no_top_down_declaration]` | Larmor's theorem stated top-down without prior observation. |

---

## 5. How EDITORIAL_SYNTHESIS Tags Junior Findings

EDITORIAL_SYNTHESIS receives four junior reports. It produces one integrated report. The pass-through mechanism works as follows:

1. SYNTHESIS reads all four junior finding lists.
2. For each junior finding, SYNTHESIS copies the finding verbatim into the integrated report with two additions:
   - `originating_worker: JUNIOR_<AXIS>`
   - `synthesis_id: SYN-PASS-<NNN>` (sequential within the synthesis report)
3. The finding's original `id`, `axis`, `severity`, and all other fields are preserved unchanged.
4. SYNTHESIS then adds NEW cross-axis findings (axis: `CROSS_AXIS`) with fresh `SYN-<NNN>` IDs.

The integrated report has this structure:

```yaml
---
report:
  worker: EDITORIAL_SYNTHESIS
  date: <date>
  junior_reports_received: [JUNIOR_VOICE, JUNIOR_CONCEPT, JUNIOR_STYLE, JUNIOR_FLOW]
  junior_finding_count: <total from all four>
  new_cross_axis_findings: <count>
  total_integrated_findings: <total>

findings:

  # --- JUNIOR_VOICE pass-through ---
  - finding:
      id: JV-001
      originating_worker: JUNIOR_VOICE
      synthesis_id: SYN-PASS-001
      # ... all original JV-001 fields unchanged ...

  # --- JUNIOR_CONCEPT pass-through ---
  - finding:
      id: JC-001
      originating_worker: JUNIOR_CONCEPT
      synthesis_id: SYN-PASS-008
      # ... all original JC-001 fields unchanged ...

  # --- JUNIOR_STYLE pass-through ---
  # ...

  # --- JUNIOR_FLOW pass-through ---
  # ...

  # --- NEW CROSS-AXIS FINDINGS ---
  - finding:
      id: SYN-001
      axis: CROSS_AXIS
      severity: High
      axes_interacting: [VOICE, CONCEPT]
      # ...
```

**SYNTHESIS does not filter.** It does not mark findings as real or overzealous — that is SENIOR_SANITY's job. SYNTHESIS passes through everything and adds its own cross-axis findings.

---

## 6. How SENIOR_FINAL Populates correction_guidance

When SENIOR_FINAL emits a REVISE verdict, it produces a consolidated actionable findings list for REVISION. For each finding that SENIOR_SANITY marked `real`, SENIOR_FINAL populates `correction_guidance` with specific revision directives.

`correction_guidance` format (when populated):

- References the specific template pattern that should be achieved (e.g., "See VOICE_SOCRATIC Pattern 1")
- Specifies the structural change needed (not the prose — REVISION writes the prose)
- If a cross-axis finding, specifies which axis to prioritize if a tradeoff is necessary
- May reference STORYBOARD.md if the change touches logical structure

REVISION receives the findings list with `correction_guidance` populated and uses each directive as the specification for its rewrite.

---

## 7. Isolation Reminder for Junior Workers

**Each junior worker produces their findings independently. They do not see each other's findings. They do not know what the other juniors found. Integration is EDITORIAL_SYNTHESIS's job.**

Each junior's report is a standalone YAML document per this schema, delivered to QUEEN who then passes all four reports to EDITORIAL_SYNTHESIS simultaneously.

---

*End of FINDING_FORMAT.md.*
