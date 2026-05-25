# STORYBOARD.md — Formal Format Specification

**Version:** 1.0.0
**Produced by:** STORYBOARDER (per WORKER_STORYBOARDER.md)
**Validated by:** QA_STORYBOARD (per WORKER_QA_STORYBOARD.md)
**Consumed by:** BOOK_EDITORIAL (JUNIOR_FLOW, REVISION)

This document is the authoritative format specification for `STORYBOARD.md`. STORYBOARDER must produce a file that conforms to this spec. QA_STORYBOARD validates structural conformance as part of Check 3 (Completeness).

---

## 1. File-level frontmatter

STORYBOARD.md begins with a YAML-style frontmatter block. All fields are required.

```yaml
---
storyboard_version: 1.0.0
produced_by: STORYBOARDER
date: <ISO 8601 date, e.g. 2026-04-18>
book_title: <from BOOK_MANIFEST.json>
genre: <from BOOK_MANIFEST.json structure.genre>
total_chapters: <integer — must match STRUCTURE.md chapter count>
subset_run: false
subset_chapters: null
full_manuscript_chapters: null
---
```

**For subset runs**, use:

```yaml
---
storyboard_version: 1.0.0
produced_by: STORYBOARDER
date: <ISO 8601 date>
book_title: <from BOOK_MANIFEST.json>
genre: <from BOOK_MANIFEST.json structure.genre>
total_chapters: <integer — count of chapters in THIS storyboard, i.e. the subset>
subset_run: true
subset_chapters: ["CH_01", "CH_03", "CH_05"]
full_manuscript_chapters: <integer — total chapter count from STRUCTURE.md>
---
```

---

## 2. Heading hierarchy

```
# STORYBOARD — <Book Title>              ← H1: document title (used once)

## Per-Chapter Entries                   ← H2: major section

### CH_<NN>: <Full Chapter Title>        ← H3: one per chapter

#### Opening State                       ← H4: per-chapter fields
#### Key Moves
#### Closing State
#### Concepts Introduced
#### Concepts Required
#### Chapter Function
#### DRAFTER Origin Note                 ← H4: only present for drafter_origin chapters

## Full-Work Arc Map                     ← H2: major section

## Prerequisite Chain                    ← H2: major section

## Reader Journey                        ← H2: major section
```

No other heading levels are used. H5 and H6 are forbidden. H4 is used only for the per-chapter field labels listed above.

---

## 3. Per-chapter entry — required fields

Each chapter entry (H3) contains all of the following H4 subsections in this order:

### 3.1 Opening State

Prose paragraph(s). No list format. 2-5 sentences describing the reader's epistemic state upon entering the chapter.

```markdown
### CH_01: Why Non-Locality

#### Opening State

The reader arrives with general familiarity with quantum mechanics at the undergraduate level: wave
functions, measurement, superposition. The specific problem of spin has not yet been framed. The
motivating puzzle — why non-locality in spin correlations requires a treatment beyond standard QM I —
has not been introduced.
```

### 3.2 Key Moves

Numbered list. 3-7 items. Each item is one sentence in academic descriptive prose. Order reflects the sequence of the chapter's argument.

```markdown
#### Key Moves

1. Introduces the EPR thought experiment as the entry point for non-locality, framing it as a
   puzzle that standard quantum mechanics describes but does not explain.
2. Distinguishes the descriptive success of QM I from the explanatory gap that motivates a deeper
   treatment of spin.
3. States the book's central thesis: a field-theoretic treatment of spin resolves the explanatory
   gap.
4. Establishes the pedagogical contract: physical intuition precedes formal machinery throughout.
5. Orients the chapter order: spin before angular momentum, concrete before abstract.
```

### 3.3 Closing State

Prose paragraph(s). Same format as Opening State. 2-5 sentences.

```markdown
#### Closing State

The reader understands the book's motivating puzzle and its proposed resolution at a high level.
The pedagogical contract is established. The reader holds a question — why is spin non-local? —
and a promise: the field-theoretic treatment will answer it. The reader is oriented to expect
concrete cases before formal machinery.
```

### 3.4 Concepts Introduced

Labeled list. Each entry uses the format: `[CONCEPT_NAME]: description`. Concept names use ALL_CAPS_WITH_UNDERSCORES. Descriptions are 1-2 sentences — enough for QA_STORYBOARD to verify this concept is the one appearing in the chapter, and enough for downstream chapters to reference it in their `concepts_required`.

```markdown
#### Concepts Introduced

- [NON_LOCALITY_PROBLEM]: The observation that quantum entanglement produces correlations that
  cannot be explained by local hidden variable theories, as demonstrated by EPR and Bell's theorem.
- [PEDAGOGICAL_CONTRACT]: The book's commitment to presenting physical intuition before mathematical
  formalism throughout.
- [SPIN_FIRST_ORDERING]: The chapter ordering decision: spin (concrete) precedes angular momentum
  (general), reversing textbook convention.
- [FIELD_THEORETIC_RESOLUTION_THESIS]: The book's central claim that a QFT treatment of spin
  dissolves the explanatory gap in non-locality.
```

### 3.5 Concepts Required

Labeled list. Same format as Concepts Introduced, but with a back-reference to where the concept was established.

```markdown
#### Concepts Required

- [QUANTUM_MECHANICS_I_BASICS]: (reader prerequisite — assumed, not introduced in this manuscript)
  Wave functions, measurement, superposition at undergraduate level.
```

For subsequent chapters:

```markdown
#### Concepts Required

- [NON_LOCALITY_PROBLEM]: (established in CH_01) The explanatory gap in QM I that motivates the
  field-theoretic treatment.
- [SPIN_FIRST_ORDERING]: (established in CH_01) The pedagogical contract that spin (concrete)
  precedes angular momentum (general).
```

**For subset runs only**, concepts required from excluded chapters use:

```markdown
- [ANGULAR_MOMENTUM_GENERAL]: (CH_04 — excluded from subset, assumed established) The general
  rotational symmetry generator encompassing spin as a special case.
```

### 3.6 Chapter Function

Prose paragraph. 3-5 sentences. Meta-description of the chapter's structural role in the full work.

```markdown
#### Chapter Function

Chapter 1 serves as the motivational entry point for the full work. It establishes the explanatory
gap that every subsequent chapter works to fill. Without this chapter, the reader has no reason to
pursue the field-theoretic treatment — the motivation is missing. It also establishes the
pedagogical contract that constrains the chapter ordering and determines the level of abstraction
at which each topic is first introduced.
```

### 3.7 DRAFTER Origin Note (conditional — only for drafter-origin chapters)

Present only when the chapter file has `drafter_origin: true` in its frontmatter. If absent, this H4 section is omitted entirely.

```markdown
#### DRAFTER Origin Note

drafter_origin: true
drafter_origin_note: This chapter was produced by DRAFTER, not carved from author-sourced
MASTER.md. QA_STORYBOARD applies additional scrutiny to the accuracy check for this entry:
verify that DRAFTER's prose semantically matches the storyboard's description of what the chapter
does, and check for [DRAFTER_GAP: reason] placeholder markers in the chapter text.
```

---

## 4. Full-Work Arc Map section

The arc map is a prose-first section with a structured summary at the end.

```markdown
## Full-Work Arc Map

### Trajectory type

<One of: ascending-pyramid | dialectic | modular | discovery-arc | other (specify)>

<1-2 sentences characterizing why this trajectory type fits this manuscript's structure.>

### Peak

Chapter(s) that represent the climactic moment of the argument or exploration: <CH_<NN> or
"CH_<N> and CH_<M>">

<1-2 sentences describing what the peak consists of and why these chapters are the structural
climax.>

### Reader effort distribution

<Describe which chapters demand the most from the reader (densest concept load, highest
abstraction) and which chapters are more consolidative or narrative.>

### Structural distinguishing features

<Describe 1-3 features that distinguish this work's structure from genre defaults. E.g., "the
pedagogical-order reversal (spin before angular momentum) is a deliberate departure from textbook
convention." Only note features that affect structural reading by BOOK_EDITORIAL.>

### Chapter progression summary

| Chapter | Structural role |
|---|---|
| CH_01_<TITLE> | <one-line structural function> |
| CH_02_<TITLE> | <one-line structural function> |
| ... | ... |
```

---

## 5. Prerequisite Chain section

The prerequisite chain is a DAG expressed as an edge list, with acyclicity verification.

```markdown
## Prerequisite Chain

### Concept nodes

The following concepts are tracked in the prerequisite chain. Each appears in exactly one
chapter's `concepts_introduced` (its origin) and may appear in one or more subsequent chapters'
`concepts_required` (its dependencies).

| Concept | Origin chapter | Required by |
|---|---|---|
| [NON_LOCALITY_PROBLEM] | CH_01 | CH_03, CH_07 |
| [PEDAGOGICAL_CONTRACT] | CH_01 | CH_02, CH_03, CH_04, CH_05, CH_06, CH_07 |
| [SPIN_FORMALISM] | CH_03 | CH_04, CH_05, CH_06 |
| ... | ... | ... |

### Edge list

An edge A → B means: concept B requires concept A (established in an earlier chapter).

```
[NON_LOCALITY_PROBLEM] (CH_01) → [SPIN_FORMALISM] (CH_03)
[PEDAGOGICAL_CONTRACT] (CH_01) → [FEYNMAN_INTUITION_ROLE] (CH_04)
[SPIN_FORMALISM] (CH_03) → [CLEBSCH_GORDAN_COEFFICIENTS] (CH_06)
[SPIN_FORMALISM] (CH_03) → [SPIN_2_FIELD] (CH_07)
[FEYNMAN_INTUITION_ROLE] (CH_04) → [FORMAL_NOTATION] (CH_05)
[FORMAL_NOTATION] (CH_05) → [CLEBSCH_GORDAN_COEFFICIENTS] (CH_06)
[CLEBSCH_GORDAN_COEFFICIENTS] (CH_06) → [SPIN_GRAVITY_COUPLING] (CH_07)
```

### Reader prerequisite concepts (from BOOK_MANIFEST.json or assumed)

Concepts required by the manuscript but not introduced within it:

```
[QUANTUM_MECHANICS_I_BASICS] → [NON_LOCALITY_PROBLEM] (CH_01)
```

### DAG verification

```
DAG_VERIFICATION: ACYCLIC — confirmed.
  Concepts (nodes): <N>
  Prerequisite edges: <M>
  No cycles detected.
  Algorithm: topological ordering via iterative root removal.
    Roots (no incoming edges): [NON_LOCALITY_PROBLEM], [QUANTUM_MECHANICS_I_BASICS]
    All nodes removable in topological order: YES
```

or:

```
DAG_VERIFICATION: CYCLE_DETECTED.
  Cycle members: [CONCEPT_A], [CONCEPT_B], [CONCEPT_C]
  Cycle path: [CONCEPT_A] (CH_03) → [CONCEPT_B] (CH_05) → [CONCEPT_C] (CH_03)
  Note: this cycle implies a structural problem. See STORYBOARDER report.
```
```

---

## 6. Reader Journey section

The reader journey is organized into stages. Each stage covers one or more chapters and describes the reader's epistemic state — what they understand, not what they read.

```markdown
## Reader Journey

### Stage 1 — <Stage Name> (CH_<N>–CH_<M>)

<Prose description of what the reader understands at the end of this stage. 3-5 sentences. Focus
on what the reader can do, believe, and reason about — not on what they read. Epistemic, not
bibliographic.>

### Stage 2 — <Stage Name> (CH_<N>–CH_<M>)

<Prose description.>

### Stage 3 — ...

### Final reader state

<2-3 sentences describing the epistemic state of the reader who has completed the full work. What
can they now explain? What questions remain open? How does their understanding differ from what
they held at the start?>
```

**Stage naming:** stage names should be descriptive of the epistemic transformation, not just chapter ranges. E.g., "Motivation and Orientation" not "Stage 1"; "Formal Unification" not "Stage 3."

**Stage count:** 2-5 stages. A manuscript with 3-4 chapters may have 2 stages. A manuscript with 12+ chapters may have 4-5 stages. Do not create a stage per chapter.

---

## 7. Version field

The `storyboard_version` in the frontmatter follows SemVer:

- **MAJOR** increment: the storyboard was substantially restructured (new chapter order detected, arc map trajectory type changed, major prerequisite chain restructuring)
- **MINOR** increment: entries revised but structure preserved (REVISE cycle corrections, accuracy fixes, additional concepts added to chains)
- **PATCH** increment: minor wording corrections, typo fixes

STORYBOARDER increments the version on each production. Initial version is `1.0.0`.

---

## 8. QA_STORYBOARD format validation checklist

When QA_STORYBOARD validates format conformance (as part of Check 3 / Completeness):

- [ ] Frontmatter block present and complete (all required fields)
- [ ] H1 document title present (used once)
- [ ] H2 section: "Per-Chapter Entries" present
- [ ] H3 entry for every chapter in STRUCTURE.md (or every subset chapter for subset runs)
- [ ] Each H3 entry contains H4 sections in correct order: Opening State, Key Moves, Closing State, Concepts Introduced, Concepts Required, Chapter Function
- [ ] DRAFTER Origin Note H4 present for drafter-origin chapters; absent for non-drafter-origin chapters
- [ ] Key Moves are numbered lists (not prose paragraphs, not bullet lists)
- [ ] Concepts Introduced use [ALL_CAPS_WITH_UNDERSCORES] format
- [ ] Concepts Required include back-references to origin chapters
- [ ] H2 section: "Full-Work Arc Map" present with all subsections
- [ ] Chapter progression summary table present
- [ ] H2 section: "Prerequisite Chain" present with concept node table, edge list, and DAG verification block
- [ ] H2 section: "Reader Journey" present with 2-5 named stages and Final reader state

---

*End of STORYBOARD_FORMAT.md.*
