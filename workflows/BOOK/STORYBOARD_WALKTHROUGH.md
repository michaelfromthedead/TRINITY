# BOOK_STORYBOARD — End-to-End Mental Walkthrough

**Purpose:** Trace a hypothetical BOOK_STORYBOARD run from trigger to GREEN_LIGHT using the "Spin of Gravity" manuscript from CONSOLIDATION_WALKTHROUGH.md. Verify that all worker interactions are coherent and show example STORYBOARD.md content per the format spec.

**References:**
- `workflows/BOOK/BOOK_STORYBOARD.json` — workflow spec
- `workflows/BOOK/WORKER_STORYBOARDER.md` — STORYBOARDER role
- `workflows/BOOK/WORKER_QA_STORYBOARD.md` — QA_STORYBOARD role
- `workflows/BOOK/STORYBOARD_FORMAT.md` — format spec
- `workflows/BOOK/CONSOLIDATION_WALKTHROUGH.md` — source of the hypothetical manuscript

---

## Hypothetical Scenario: "Spin of Gravity" Manuscript (continuing from CONSOLIDATION_WALKTHROUGH.md)

### Input state

BOOK_CONSOLIDATION completed with GREEN_LIGHT. The following files exist at project root:

**Chapter files in `chapters/`:**
```
chapters/CH_01_WHY_NON_LOCALITY.md
chapters/CH_02_HISTORICAL_CONTEXT.md
chapters/CH_03_SPIN_CONCRETE.md
chapters/CH_04_FEYNMAN_INTUITION.md
chapters/CH_05_FORMALISM.md
chapters/CH_06_ANGULAR_MOMENTUM.md
chapters/CH_07_QFT_AND_GRAVITY_COUPLING.md
```

**STRUCTURE.md excerpt:**
```markdown
| # | Title | Slug | Summary |
|---|---|---|---|
| 01 | Why Non-Locality | CH_01_WHY_NON_LOCALITY | Establishes the motivating puzzle and pedagogical contract |
| 02 | Historical Context | CH_02_HISTORICAL_CONTEXT | Dirac, Pauli, and early spin intuition |
| 03 | Spin — The Concrete Case | CH_03_SPIN_CONCRETE | Physical spin mechanics, discrete states |
| 04 | Feynman Diagrams as Intuition | CH_04_FEYNMAN_INTUITION | Intuition-pump role, not computational |
| 05 | Mathematical Formalism | CH_05_FORMALISM | Notation, prerequisites, formalism setup |
| 06 | Angular Momentum | CH_06_ANGULAR_MOMENTUM | General case, Clebsch-Gordan, fine structure |
| 07 | QFT and Gravity Coupling | CH_07_QFT_AND_GRAVITY_COUPLING | Spin-2 field, gravity coupling, open frontier |
```

**BOOK_MANIFEST.json (relevant excerpt):**
```json
{
  "title": "Spin of Gravity",
  "genre": "academic_exploratory",
  "templates": { "mode": "bundle", "bundle": "BUNDLE_SPIN_OF_GRAVITY" }
}
```

**No drafter-origin chapters** — all 7 chapters were carved from author-sourced MASTER.md by COMPOSITOR.

---

## Phase 1: PRESTEP

**Human types:** `BOOK_STORYBOARD`

**QUEEN executes engagement sequence (from BOOK_STORYBOARD.json):**

1. Reads `BOOK_STORYBOARD.json` in full
2. Reads `workflows/SHARED/WORKER_QUEEN.md`
3. Reads `workflows/SHARED/WORKER_PROTOCOL.md`
4. Reads `workflows/SHARED/WORKER.md`
5. Reads `BOOK_MANIFEST.json` from project root — notes genre: `academic_exploratory`
6. Reads `STRUCTURE.md` — notes 7 chapters
7. Verifies all 7 chapter files exist in `chapters/`
8. Reports: **"BOOK_STORYBOARD mode engaged. 7 chapters found. Ready."**
9. Initializes INPROGRESS.md entry for BOOK_STORYBOARD engagement
10. Proceeds to spawn STORYBOARDER (no additional human prompt needed)

**QUEEN writes to INPROGRESS.md (prepend):**
```markdown
---
[2026-04-18] BOOK_STORYBOARD — ENGAGED
Chapters: 7
Genre: academic_exploratory
Status: STORYBOARDING in progress
```

---

## Phase 2: STORYBOARDING (STORYBOARDER)

**Context packet QUEEN assembles:**
- All 7 chapter files
- `STRUCTURE.md`
- `BOOK_MANIFEST.json`
- `PEDAGOGY.md` (3 entries from CONSOLIDATION run — available for context)
- `WORKER_STORYBOARDER.md`
- `WORKER_PROTOCOL.md`
- `STORYBOARD_FORMAT.md`

**QUEEN spawns STORYBOARDER.**

---

### STORYBOARDER execution

**Step 1 — Orient.** STORYBOARDER reads WORKER_PROTOCOL.md, BOOK_STORYBOARD.json, STORYBOARD_FORMAT.md.

**Step 2 — Check for drafter-origin chapters.** STORYBOARDER scans all 7 chapter file frontmatters. No `drafter_origin: true` found.

**Step 3 — Read the full manuscript.**

STORYBOARDER reads:
- STRUCTURE.md — notes the dependency map from COMPOSITOR: `CH_01 → CH_02 → CH_03 → CH_04 → CH_05 → CH_06 → CH_07` (with branching: CH_01 → CH_03 also via the motivational connection).
- BOOK_MANIFEST.json — genre: `academic_exploratory`.
- PEDAGOGY.md — notes the two key overwrite events: Feynman diagrams framing (intuition-pump, not computational) and chapter ordering (spin before angular momentum).
- CH_01 through CH_07 — reads each completely.

**Step 4 — Draft prerequisite chain sketch.**

STORYBOARDER identifies the following concepts and their dependency structure:

| Concept | Introduced | Required by |
|---|---|---|
| NON_LOCALITY_PROBLEM | CH_01 | CH_07 |
| PEDAGOGICAL_CONTRACT | CH_01 | CH_02–CH_07 (implicit) |
| SPIN_FIRST_ORDERING | CH_01 | CH_03, CH_06 |
| FIELD_THEORETIC_RESOLUTION_THESIS | CH_01 | CH_07 |
| HISTORICAL_SPIN_CONTEXT | CH_02 | CH_03 |
| DIRAC_PAULI_BACKGROUND | CH_02 | CH_03, CH_05 |
| SPIN_PHYSICAL_STATES | CH_03 | CH_04, CH_06 |
| SPIN_TWO_STATE_SYSTEM | CH_03 | CH_06 |
| FEYNMAN_INTUITION_ROLE | CH_04 | CH_05 |
| SPIN_NOTATION_SETUP | CH_05 | CH_06 |
| DIRAC_EQUATION_MECHANICS | CH_05 | CH_06, CH_07 |
| ANGULAR_MOMENTUM_GENERAL | CH_06 | CH_07 |
| CLEBSCH_GORDAN_COEFFICIENTS | CH_06 | CH_07 |
| TOTAL_ANGULAR_MOMENTUM_J | CH_06 | CH_07 |
| FINE_STRUCTURE | CH_06 | (terminal — no later requirement) |
| SPIN_2_FIELD | CH_07 | (terminal) |
| SPIN_GRAVITY_COUPLING | CH_07 | (terminal) |

**Acyclicity check (topological ordering via root removal):**

Roots (no incoming edges): `NON_LOCALITY_PROBLEM`, `PEDAGOGICAL_CONTRACT`, `SPIN_FIRST_ORDERING`, `FIELD_THEORETIC_RESOLUTION_THESIS`, `HISTORICAL_SPIN_CONTEXT`, `DIRAC_PAULI_BACKGROUND` — all from CH_01 or CH_02, which have no prerequisite concepts within the manuscript.

Removing roots reveals next layer (CH_03 concepts, then CH_04, CH_05, CH_06, CH_07). All nodes are removable in topological order. **DAG verified acyclic.**

**Step 5 — Write per-chapter entries.** (Example entries shown in §4 below.)

**Step 6 — Write full-work sections.** (Example shown in §4 below.)

**Step 7 — Consistency self-check.**

STORYBOARDER verifies:
- All 7 STRUCTURE.md chapters have entries: YES
- All concepts_required have earlier concepts_introduced sources: YES (with QUANTUM_MECHANICS_I_BASICS declared as reader prerequisite)
- Prerequisite chain edges consistent with per-chapter fields: YES
- Arc map consistent with chapter_function descriptions: YES

**STORYBOARDER produces STORYBOARD.md.**

---

### STORYBOARDER report (summary)

```
==== WORKER REPORT ====
Role: STORYBOARDER
BOOK_STORYBOARD run: 2026-04-18
Trigger: initial STORYBOARDING

Files produced:
  - STORYBOARD.md

Chapter count: 7 chapters storyboarded
  CH_01_WHY_NON_LOCALITY
  CH_02_HISTORICAL_CONTEXT
  CH_03_SPIN_CONCRETE
  CH_04_FEYNMAN_INTUITION
  CH_05_FORMALISM
  CH_06_ANGULAR_MOMENTUM
  CH_07_QFT_AND_GRAVITY_COUPLING

Drafter-origin chapters: 0 (none)

Prerequisite chain:
  Concepts tracked: 17
  Prerequisite edges: 21
  Acyclicity verified: YES

Full-work sections completed:
  - Arc map: YES
  - Prerequisite chain: YES
  - Reader journey: YES (4 stages)

Consistency self-check:
  - All STRUCTURE.md chapters covered: YES
  - All concepts_required have earlier concepts_introduced source: YES
  - Arc map consistent with chapter_function entries: YES

Outstanding:
  - CH_07 content is thinner than other chapters (flagged by COMPOSITOR; gravity coupling material
    was rough source notes). Storyboard entry reflects what is actually there. QA_STORYBOARD should
    note thinness but this is a pipeline-level gap, not a storyboard error.
```

---

## Phase 3: Example STORYBOARD.md content

The following is a representative excerpt of the STORYBOARD.md STORYBOARDER produces. Not every chapter is shown in full — CH_01, CH_04, and CH_07 are shown to illustrate the range (first chapter, middle chapter, terminal chapter).

```markdown
---
storyboard_version: 1.0.0
produced_by: STORYBOARDER
date: 2026-04-18
book_title: Spin of Gravity
genre: academic_exploratory
total_chapters: 7
subset_run: false
subset_chapters: null
full_manuscript_chapters: null
---

# STORYBOARD — Spin of Gravity

## Per-Chapter Entries

### CH_01: Why Non-Locality

#### Opening State

The reader arrives with general familiarity with quantum mechanics at the undergraduate level:
wave functions, superposition, measurement. The specific non-locality problem has not been
framed. The pedagogical contract of this book has not been established. The reader holds no
particular expectation about chapter ordering or level of abstraction.

#### Key Moves

1. Introduces the EPR thought experiment as the entry point for non-locality, framing it as
   a phenomenon quantum mechanics I describes but does not explain at the field-theoretic level.
2. Distinguishes the descriptive success of quantum mechanics I from the explanatory gap in
   understanding spin correlations across space.
3. States the book's central thesis: a field-theoretic treatment of spin dissolves the
   explanatory gap.
4. Establishes the pedagogical contract: physical intuition precedes mathematical formalism
   throughout the book.
5. States the ordering rationale: spin (the concrete case) precedes angular momentum (the
   general case), reversing textbook convention, because the reader needs the concrete case
   first.

#### Closing State

The reader understands the book's motivating puzzle: why spin correlations violate locality in
a way that demands field-theoretic explanation. The pedagogical contract is established — the
reader knows to expect intuition before formalism. The ordering rationale is stated: spin
comes before angular momentum because the concrete case enables the general case. The reader
holds a question and a thesis, not yet a resolution.

#### Concepts Introduced

- [NON_LOCALITY_PROBLEM]: The explanatory gap in quantum mechanics I regarding spin
  correlations: QM I describes but does not field-theoretically explain non-local correlations.
- [PEDAGOGICAL_CONTRACT]: The book's commitment to presenting physical intuition before
  mathematical formalism throughout all chapters.
- [SPIN_FIRST_ORDERING]: The deliberate reversal of textbook convention placing spin (concrete
  case) before angular momentum (general case), with the stated rationale that the concrete
  case enables the general.
- [FIELD_THEORETIC_RESOLUTION_THESIS]: The book's central claim: a quantum field theory
  treatment of spin provides the explanatory framework that quantum mechanics I lacks.

#### Concepts Required

- [QUANTUM_MECHANICS_I_BASICS]: (reader prerequisite — assumed, not introduced in this
  manuscript) Wave functions, superposition, measurement at the undergraduate level. Bell's
  theorem at the conceptual level.

#### Chapter Function

Chapter 1 serves as the motivational frame for the entire work. It establishes the
explanatory gap that every subsequent chapter works to address. Without this chapter, the
reader has no reason to pursue the field-theoretic treatment over quantum mechanics I — the
motivation is absent. The chapter also installs the pedagogical contract that constrains
chapter ordering and abstraction sequencing throughout: the reader's subsequent experience
of each chapter is shaped by the promise made here.

---

### CH_04: Feynman Diagrams as Intuition

#### Opening State

The reader understands spin as a physical phenomenon with two discrete states (CH_03),
holds the historical and intuitive context (CH_02), and carries the book's pedagogical
contract. Feynman diagrams have been mentioned in the historical context but their role in
this book — as intuition tools, not computational devices — has not been specified. The
reader may arrive with prior assumptions from quantum mechanics I that Feynman diagrams are
primarily computational.

#### Key Moves

1. Distinguishes two uses of Feynman diagrams: as shorthand for perturbation-theory
   computations (the standard use in QM I and QFT courses) versus as intuition pumps for
   visualizing particle interactions.
2. States explicitly that this book uses Feynman diagrams only in the intuition-pump role.
3. Demonstrates the intuition-pump use through a set of spin-interaction examples where the
   diagram illuminates what is happening physically without being used to compute a result.
4. Establishes the boundary: where intuition runs out and formalism becomes necessary.
5. Positions this boundary as the motivation for the formalism chapter that follows.

#### Closing State

The reader understands that Feynman diagrams in this book are intuition tools, not
computational devices. The reader can use this distinction to follow subsequent chapters
where diagrams appear without expecting them to lead to calculations. The reader also
understands why formalism is coming next: the intuition-pump approach reaches its limits at
the boundary this chapter maps.

#### Concepts Introduced

- [FEYNMAN_INTUITION_ROLE]: The use of Feynman diagrams as visualization tools for
  physical interaction (as opposed to computational shorthand in perturbation theory).
  Specific to this book's pedagogical contract.
- [INTUITION_FORMALISM_BOUNDARY]: The limit at which intuitive reasoning via Feynman
  diagrams fails and formal mathematical machinery is required.

#### Concepts Required

- [PEDAGOGICAL_CONTRACT]: (established in CH_01) Physical intuition precedes formalism.
- [SPIN_PHYSICAL_STATES]: (established in CH_03) Spin as a two-state physical system.
- [DIRAC_PAULI_BACKGROUND]: (established in CH_02) Historical context for why the Feynman
  diagram formalism emerged from specific theoretical needs.

#### Chapter Function

Chapter 4 functions as the bridge between the physical-intuition half of the book (CH_01–
CH_03) and the formal-machinery half (CH_05–CH_07). It fulfills the pedagogical contract's
promise that intuition comes before formalism by making the intuition explicit and mapping
its limits. This creates the reader's motivation for the formalism chapter: the reader
sees exactly what intuition cannot do and therefore wants the formal tools. Without this
chapter, the transition from concrete spin mechanics to abstract formalism would be abrupt.

---

### CH_07: QFT and Gravity Coupling

#### Opening State

The reader possesses the full formal framework: spin formalism, angular momentum as the
general case, the Clebsch-Gordan machinery, and the fine structure as a concrete payoff.
The field-theoretic resolution thesis (established in CH_01) has been promised but not yet
delivered. The reader understands spin within quantum mechanics and is prepared for the
extension to quantum field theory.

#### Key Moves

1. Introduces the quantum field theory framing for spin: spin as a label on representations
   of the Lorentz group, rather than as a discrete state of a particle.
2. Establishes why the graviton must be a spin-2 field: the argument from mediating a
   long-range attractive force between all forms of energy.
3. Presents the spin-2 field coupling to matter in linearized gravity, connecting the
   earlier spin formalism to the gravitational context.
4. Identifies the limits of the linearized treatment and situates those limits as open
   research questions at the interface of quantum gravity.

#### Closing State

The reader understands spin at the quantum field theory level and sees how the spin
formalism extends to the gravitational context. The book's central thesis — that a
field-theoretic treatment dissolves the explanatory gap in non-locality — is now visible
in outline, though the full treatment of quantum gravity remains an open problem. The
reader exits with an understanding of where the book's argument leads and where the
frontier of knowledge currently sits.

#### Concepts Introduced

- [LORENTZ_GROUP_SPIN]: Spin as a label on irreducible representations of the Lorentz
  group in quantum field theory.
- [SPIN_2_FIELD]: The requirement that the graviton be described by a spin-2 tensor field;
  argument from the mediated-force structure of gravity.
- [SPIN_GRAVITY_COUPLING]: The interaction term coupling the spin-2 field (graviton) to
  the stress-energy tensor of matter in linearized gravity.
- [LINEARIZATION_LIMITS]: The regime of validity of linearized gravity and what breaks
  down at strong-field / high-energy limits.

#### Concepts Required

- [FIELD_THEORETIC_RESOLUTION_THESIS]: (established in CH_01) The claim motivating the
  QFT treatment.
- [ANGULAR_MOMENTUM_GENERAL]: (established in CH_06) The general rotational symmetry
  framework.
- [TOTAL_ANGULAR_MOMENTUM_J]: (established in CH_06) J = L + S as a conserved quantity.
- [DIRAC_EQUATION_MECHANICS]: (established in CH_05) The formal treatment of spin in the
  Dirac equation.

#### Chapter Function

Chapter 7 delivers the book's thesis: it applies the field-theoretic framework to the
gravity coupling problem that the opening chapter posed. It functions as the payoff of
the entire arc. The chapter is deliberately positioned as an introduction to the open
frontier rather than a closed derivation — consistent with the academic-exploratory genre
and the pedagogical contract's emphasis on understanding over computation. Its relative
thinness (compared to CH_05 and CH_06) reflects the state of the source material; it
opens a research direction rather than closing an argument.

---

## Full-Work Arc Map

### Trajectory type

Discovery arc.

The work follows a discovery arc: it begins with a puzzle (why non-locality?), builds the
reader's conceptual and formal toolkit across the middle chapters, and delivers a partial
resolution at the end — partial because the frontier of knowledge is genuinely open. This
trajectory is characteristic of academic-exploratory genre: the reader discovers alongside
the argument rather than absorbing established doctrine.

### Peak

Chapters 6 and 7 together form the structural peak. CH_06 delivers the formal unification
of spin and angular momentum (the payoff of the spin-first ordering), and CH_07 applies
the resulting framework to the gravity coupling problem (the payoff of the motivating
thesis).

### Reader effort distribution

Chapters 1–2: low-to-moderate effort. Motivational and historical. Narrative-dominant.
Chapter 3: moderate effort. Concrete spin mechanics requires some mathematical engagement.
Chapter 4: low effort. Conceptually clarifying, not formally dense.
Chapter 5: high effort. The formalism chapter is the most mathematically demanding.
Chapter 6: high effort. The Clebsch-Gordan machinery and fine structure derivation are
dense. This is the technical climax.
Chapter 7: moderate effort. The QFT framing is presented at a conceptual level appropriate
to the genre; it is not a full QFT derivation.

### Structural distinguishing features

1. Spin before angular momentum: the ordering reversal (CH_03 before CH_06) is the
   structural signature of the book's pedagogical contract. BOOK_EDITORIAL workers should
   treat this ordering as intentional, not a sequencing error.
2. Intuition chapter precedes formalism chapter: CH_04 (Feynman intuition) appears before
   CH_05 (formalism), which is the reverse of most textbook presentations. This is
   consistent with the pedagogical contract.
3. Terminal chapter opens rather than closes: CH_07 positions the reader at a research
   frontier rather than summarizing established results. The chapter's thinness is
   deliberate (not an editorial gap at the chapter level, though the source material was
   sparse).

### Chapter progression summary

| Chapter | Structural role |
|---|---|
| CH_01_WHY_NON_LOCALITY | Motivation and contract; installs the question and the pedagogical promise |
| CH_02_HISTORICAL_CONTEXT | Context and intuition; prepares the reader for spin as a historical discovery |
| CH_03_SPIN_CONCRETE | Concrete machinery; the first formal engagement with spin's physical structure |
| CH_04_FEYNMAN_INTUITION | Bridge and boundary; marks the limit of intuition and motivates the formalism |
| CH_05_FORMALISM | Formal infrastructure; the most demanding chapter; prepares CH_06 |
| CH_06_ANGULAR_MOMENTUM | Synthesis and payoff; delivers the spin-first-ordering promise |
| CH_07_QFT_AND_GRAVITY_COUPLING | Thesis delivery and frontier opening; applies the arc's result to the motivating problem |

---

## Prerequisite Chain

### Concept nodes

| Concept | Origin chapter | Required by |
|---|---|---|
| [NON_LOCALITY_PROBLEM] | CH_01 | CH_07 |
| [PEDAGOGICAL_CONTRACT] | CH_01 | CH_02, CH_03, CH_04, CH_05, CH_06, CH_07 |
| [SPIN_FIRST_ORDERING] | CH_01 | CH_03, CH_06 |
| [FIELD_THEORETIC_RESOLUTION_THESIS] | CH_01 | CH_07 |
| [HISTORICAL_SPIN_CONTEXT] | CH_02 | CH_03 |
| [DIRAC_PAULI_BACKGROUND] | CH_02 | CH_03, CH_05 |
| [SPIN_PHYSICAL_STATES] | CH_03 | CH_04, CH_06 |
| [SPIN_TWO_STATE_SYSTEM] | CH_03 | CH_06 |
| [FEYNMAN_INTUITION_ROLE] | CH_04 | CH_05 |
| [INTUITION_FORMALISM_BOUNDARY] | CH_04 | CH_05 |
| [SPIN_NOTATION_SETUP] | CH_05 | CH_06 |
| [DIRAC_EQUATION_MECHANICS] | CH_05 | CH_06, CH_07 |
| [ANGULAR_MOMENTUM_GENERAL] | CH_06 | CH_07 |
| [CLEBSCH_GORDAN_COEFFICIENTS] | CH_06 | CH_07 |
| [TOTAL_ANGULAR_MOMENTUM_J] | CH_06 | CH_07 |
| [FINE_STRUCTURE] | CH_06 | (terminal) |
| [LORENTZ_GROUP_SPIN] | CH_07 | (terminal) |
| [SPIN_2_FIELD] | CH_07 | (terminal) |
| [SPIN_GRAVITY_COUPLING] | CH_07 | (terminal) |
| [LINEARIZATION_LIMITS] | CH_07 | (terminal) |

### Edge list

```
[NON_LOCALITY_PROBLEM] (CH_01) → [FIELD_THEORETIC_RESOLUTION_THESIS] (CH_01)
[PEDAGOGICAL_CONTRACT] (CH_01) → [FEYNMAN_INTUITION_ROLE] (CH_04)
[SPIN_FIRST_ORDERING] (CH_01) → [SPIN_PHYSICAL_STATES] (CH_03)
[FIELD_THEORETIC_RESOLUTION_THESIS] (CH_01) → [LORENTZ_GROUP_SPIN] (CH_07)
[DIRAC_PAULI_BACKGROUND] (CH_02) → [SPIN_PHYSICAL_STATES] (CH_03)
[HISTORICAL_SPIN_CONTEXT] (CH_02) → [SPIN_PHYSICAL_STATES] (CH_03)
[DIRAC_PAULI_BACKGROUND] (CH_02) → [DIRAC_EQUATION_MECHANICS] (CH_05)
[SPIN_PHYSICAL_STATES] (CH_03) → [FEYNMAN_INTUITION_ROLE] (CH_04)
[SPIN_PHYSICAL_STATES] (CH_03) → [ANGULAR_MOMENTUM_GENERAL] (CH_06)
[SPIN_TWO_STATE_SYSTEM] (CH_03) → [CLEBSCH_GORDAN_COEFFICIENTS] (CH_06)
[FEYNMAN_INTUITION_ROLE] (CH_04) → [SPIN_NOTATION_SETUP] (CH_05)
[INTUITION_FORMALISM_BOUNDARY] (CH_04) → [SPIN_NOTATION_SETUP] (CH_05)
[SPIN_NOTATION_SETUP] (CH_05) → [ANGULAR_MOMENTUM_GENERAL] (CH_06)
[DIRAC_EQUATION_MECHANICS] (CH_05) → [CLEBSCH_GORDAN_COEFFICIENTS] (CH_06)
[DIRAC_EQUATION_MECHANICS] (CH_05) → [LORENTZ_GROUP_SPIN] (CH_07)
[ANGULAR_MOMENTUM_GENERAL] (CH_06) → [SPIN_GRAVITY_COUPLING] (CH_07)
[CLEBSCH_GORDAN_COEFFICIENTS] (CH_06) → [SPIN_GRAVITY_COUPLING] (CH_07)
[TOTAL_ANGULAR_MOMENTUM_J] (CH_06) → [SPIN_2_FIELD] (CH_07)
```

### Reader prerequisite concepts

```
[QUANTUM_MECHANICS_I_BASICS] → [NON_LOCALITY_PROBLEM] (CH_01)
[QUANTUM_MECHANICS_I_BASICS] → [SPIN_PHYSICAL_STATES] (CH_03)
```

### DAG verification

```
DAG_VERIFICATION: ACYCLIC — confirmed.
  Concepts (nodes): 20
  Prerequisite edges: 18
  No cycles detected.
  Algorithm: iterative root removal (topological ordering).
  Roots (no incoming edges within manuscript): NON_LOCALITY_PROBLEM, PEDAGOGICAL_CONTRACT,
    SPIN_FIRST_ORDERING, HISTORICAL_SPIN_CONTEXT, DIRAC_PAULI_BACKGROUND
    (all originating in CH_01 or CH_02, which have no intra-manuscript prerequisites).
  All nodes removable in topological order: YES.
```

---

## Reader Journey

### Stage 1 — Motivation and Historical Ground (CH_01–CH_02)

The reader understands why the book exists: the explanatory gap in quantum mechanics I for spin
non-locality motivates a deeper treatment. The reader holds the pedagogical contract — intuition
before formalism, concrete before abstract — and understands the chapter ordering as deliberate.
The reader also holds historical context: spin was a discovery, not an abstraction invented from
first principles, and the names Dirac and Pauli represent the moment when spin acquired formal
mathematical representation. The reader can ask the motivating question but cannot yet answer it.

### Stage 2 — Concrete Machinery and Its Limits (CH_03–CH_04)

The reader can reason about spin as a physical phenomenon: two states, their physical meaning,
their behavior under measurement. The reader understands Feynman diagrams as intuition tools
and knows precisely where intuitive reasoning fails — at the boundary where formalism becomes
necessary. The reader can follow a diagram without misreading it as a computational prescription.
The reader now understands what the formalism chapter must deliver and why it comes here.

### Stage 3 — Formal Unification (CH_05–CH_06)

The reader possesses the formal framework: notation, the Dirac equation as a spin description,
angular momentum as the general case of which spin is the specific instance, the Clebsch-Gordan
machinery for combining angular momenta, and the fine structure of hydrogen as the concrete
payoff. The reader can now see why the spin-first ordering worked: the concrete case (spin) made
the general case (angular momentum) intelligible. The pedagogical promise made in CH_01 is
fulfilled. The reader holds the complete formal toolkit.

### Stage 4 — Field Theory and the Open Frontier (CH_07)

The reader understands spin at the quantum field theory level: as a label on Lorentz group
representations, not merely as a discrete particle state. The reader understands why the graviton
must be a spin-2 field and how the spin-gravity coupling appears in linearized gravity. The
book's central thesis — that a field-theoretic treatment addresses the explanatory gap — is now
visible in outline. The reader also understands that the full resolution (a complete quantum
gravity theory) remains open. The reader exits at a genuine research frontier.

### Final reader state

A reader who has completed "Spin of Gravity" can explain why quantum mechanics I fails to
provide a satisfying account of spin non-locality, trace the argument from concrete spin
mechanics through the formal machinery of angular momentum to the field-theoretic treatment
of spin-gravity coupling, and identify the open questions at the interface of quantum field
theory and quantum gravity. The reader holds a framework rather than a closed result, which
is appropriate to the academic-exploratory genre and the research frontier the work addresses.
```

---

## Phase 4: QA_STORYBOARD

**Context packet QUEEN assembles:**
- STORYBOARD.md (just produced by STORYBOARDER)
- All 7 chapter files
- STRUCTURE.md
- BOOK_MANIFEST.json
- WORKER_QA_STORYBOARD.md
- WORKER_PROTOCOL.md

**QUEEN spawns QA_STORYBOARD.**

---

### QA_STORYBOARD execution

**Check 1 — Prerequisite satisfaction:**

QA_STORYBOARD works through all `concepts_required` entries for all 7 chapters:
- CH_01: only `[QUANTUM_MECHANICS_I_BASICS]` — declared reader prerequisite. PASS.
- CH_02: `[NON_LOCALITY_PROBLEM]` → introduced CH_01. `[PEDAGOGICAL_CONTRACT]` → introduced CH_01. PASS.
- CH_03: `[HISTORICAL_SPIN_CONTEXT]` → CH_02. `[PEDAGOGICAL_CONTRACT]` → CH_01. `[DIRAC_PAULI_BACKGROUND]` → CH_02. PASS.
- CH_04: `[PEDAGOGICAL_CONTRACT]` → CH_01. `[SPIN_PHYSICAL_STATES]` → CH_03. `[DIRAC_PAULI_BACKGROUND]` → CH_02. PASS.
- CH_05: `[FEYNMAN_INTUITION_ROLE]` → CH_04. `[INTUITION_FORMALISM_BOUNDARY]` → CH_04. `[DIRAC_PAULI_BACKGROUND]` → CH_02. PASS.
- CH_06: `[SPIN_NOTATION_SETUP]` → CH_05. `[DIRAC_EQUATION_MECHANICS]` → CH_05. `[SPIN_PHYSICAL_STATES]` → CH_03. `[SPIN_TWO_STATE_SYSTEM]` → CH_03. PASS.
- CH_07: `[FIELD_THEORETIC_RESOLUTION_THESIS]` → CH_01. `[ANGULAR_MOMENTUM_GENERAL]` → CH_06. `[TOTAL_ANGULAR_MOMENTUM_J]` → CH_06. `[DIRAC_EQUATION_MECHANICS]` → CH_05. PASS.

**Result: PASS. No forward dependencies.**

**Check 2 — Progressive arc:**

- CH_01 closing state → CH_02 opening state: consistent (reader holds question + contract; CH_02 opens with the historical grounding of that question). PASS.
- CH_02 closing → CH_03 opening: consistent (reader holds historical context; CH_03 opens with the concrete physical treatment that history motivates). PASS.
- CH_04 closing → CH_05 opening: consistent (reader understands intuition limits; CH_05 opens at the point where formalism is needed). PASS.
- CH_06 closing → CH_07 opening: consistent (reader has full formal framework; CH_07 applies it to the motivating QFT question). PASS.

Arc map trajectory type (discovery arc) consistent with chapter_function descriptions: YES.

One flag: CH_06 introduces `[FINE_STRUCTURE]` as a terminal concept (no later requirements). QA_STORYBOARD notes this as a structural observation: FINE_STRUCTURE is introduced as a payoff/demonstration, not as a prerequisite for anything. This is legitimate for an exploratory work (not every concept needs to feed forward). Flag as informational, not blocking.

**Result: PASS.**

**Check 3 — Completeness:**

STRUCTURE.md lists 7 chapters. STORYBOARD.md has 7 entries. No missing. No phantom. **Result: PASS.**

**Check 4 — Accuracy (spot-check):**

QA_STORYBOARD selects:
- CH_01 (always check the opening chapter)
- CH_06 (peak chapter per arc map)
- CH_04 (random selection)

*CH_01 spot-check:* reads CH_01_WHY_NON_LOCALITY.md. Verifies EPR thought experiment is present (key move 1) — YES. Pedagogical contract stated explicitly — YES. Spin-first ordering rationale stated — YES. Concept `[NON_LOCALITY_PROBLEM]` introduced with definition — YES. Opening state claims reader has QM I familiarity but no spin framing — consistent with what CH_01 actually starts with. PASS.

*CH_06 spot-check:* reads CH_06_ANGULAR_MOMENTUM.md. Verifies Clebsch-Gordan coefficients are introduced (key move 2) — YES. Fine structure is derived (key move 4) — YES. `[ANGULAR_MOMENTUM_GENERAL]` introduced — YES. `[SPIN_NOTATION_SETUP]` required — YES (the notation from CH_05 is used without re-derivation). PASS.

*CH_04 spot-check:* reads CH_04_FEYNMAN_INTUITION.md. Verifies the two-use distinction (intuition vs. computation) is drawn explicitly — YES. `[FEYNMAN_INTUITION_ROLE]` introduced as a concept with the "intuition pump" framing — YES. `[INTUITION_FORMALISM_BOUNDARY]` established — YES. PASS.

**No drafter-origin chapters — Check 4a: N/A.**

**Result: PASS.**

**Check 5 — Genre alignment:**

Genre: `academic_exploratory`. Arc map trajectory type: discovery arc. Expected for `academic_exploratory`: discovery arc with argument that unfolds rather than declares. CONSISTENT. **Result: PASS.**

**Check 6 — Dependency acyclicity:**

QA_STORYBOARD applies the iterative root removal algorithm to the prerequisite chain edge list.

Roots: `NON_LOCALITY_PROBLEM`, `PEDAGOGICAL_CONTRACT`, `SPIN_FIRST_ORDERING`, `HISTORICAL_SPIN_CONTEXT`, `DIRAC_PAULI_BACKGROUND` — all CH_01/CH_02. Remove.

Next layer: `FIELD_THEORETIC_RESOLUTION_THESIS`, `SPIN_PHYSICAL_STATES`, `SPIN_TWO_STATE_SYSTEM` — sources in CH_01/CH_03, now all incoming edges removed. Remove.

Next layer: `FEYNMAN_INTUITION_ROLE`, `INTUITION_FORMALISM_BOUNDARY` — CH_04. Remove.

Next layer: `SPIN_NOTATION_SETUP`, `DIRAC_EQUATION_MECHANICS` — CH_05. Remove.

Next layer: `ANGULAR_MOMENTUM_GENERAL`, `CLEBSCH_GORDAN_COEFFICIENTS`, `TOTAL_ANGULAR_MOMENTUM_J`, `FINE_STRUCTURE` — CH_06. Remove.

Remaining: `LORENTZ_GROUP_SPIN`, `SPIN_2_FIELD`, `SPIN_GRAVITY_COUPLING`, `LINEARIZATION_LIMITS` — CH_07, terminal. Remove.

All nodes removed. **DAG_VERIFICATION: ACYCLIC — confirmed. Result: PASS.**

**Check 7 — Reader journey coherence:**

4 stages. Each stage verified:
- Stage 1 → Stage 2 transition: Stage 1 ends with "reader can ask the motivating question but cannot yet answer it." Stage 2 opens with concrete spin machinery. Transition via CH_03 — consistent. PASS.
- Stage 2 → Stage 3 transition: Stage 2 ends with "reader understands what the formalism chapter must deliver." Stage 3 opens with the reader "possessing the formal framework" — this describes the state after CH_05 and CH_06, which are both in Stage 3. Consistent. PASS.
- Stage 3 → Stage 4: Stage 3 closes with "reader holds the complete formal toolkit." Stage 4 opens with "reader understands spin at the QFT level" — this is what CH_07 delivers. Consistent. PASS.

No journey gaps. No inconsistencies with per-chapter closing states. **Result: PASS.**

---

### QA_STORYBOARD report (summary)

```
==== WORKER REPORT ====
Role: QA_STORYBOARD
BOOK_STORYBOARD run: 2026-04-18
Trigger: post-STORYBOARDER (initial)

Subset run: NO (full manuscript, 7 chapters)

CHECK 1: PREREQUISITE SATISFACTION — PASS (0 forward dependencies)
CHECK 2: PROGRESSIVE ARC — PASS (1 informational note: FINE_STRUCTURE terminal, expected)
CHECK 3: COMPLETENESS — PASS (7/7 chapters covered)
CHECK 4: ACCURACY — PASS (CH_01, CH_06, CH_04 spot-checked)
CHECK 4a: DRAFTER-ORIGIN — N/A (0 drafter-origin chapters)
CHECK 5: GENRE ALIGNMENT — PASS (discovery arc consistent with academic_exploratory)
CHECK 6: DEPENDENCY ACYCLICITY — PASS (ACYCLIC confirmed, 20 nodes, 18 edges)
CHECK 7: READER JOURNEY COHERENCE — PASS (4 stages, no gaps)

VERDICT RECOMMENDATION: GREEN_LIGHT

Blocking issues: none

Notes for QUEEN:
  - CH_07 (QFT and Gravity Coupling) is structurally thin relative to other chapters.
    This is a pipeline-level limitation (thin source material), not a storyboard error.
    BOOK_EDITORIAL juniors may flag this as a content gap; appropriate escalation path
    would be BOOK_COMPLETION / DRAFTER if more content is desired before editorial.
  - FINE_STRUCTURE in CH_06 is terminal (no forward dependencies). This is consistent
    with the academic-exploratory genre — payoff concepts need not feed forward.
```

---

## Phase 5: GREEN_LIGHT

**QUEEN evaluates QA results:** all 7 checks pass. QA_STORYBOARD recommendation: GREEN_LIGHT.

**QUEEN executes GREEN_LIGHT actions:**

1. Writes completion summary to INPROGRESS.md (prepend):
   ```
   [2026-04-18] BOOK_STORYBOARD — GREEN_LIGHT
   Chapters storyboarded: 7
   QA passes: 7/7 checks
   Drafter-origin chapters: 0
   Prerequisite chain: 20 concepts, 18 edges, ACYCLIC
   Status: STORYBOARD.md ready for human review
   ```

2. Commits STORYBOARD.md to project root.

3. Reports to human:
   **"BOOK_STORYBOARD complete. STORYBOARD.md ready for human review. After review, trigger BOOK_EDITORIAL."**

**Human reviews STORYBOARD.md** before triggering BOOK_EDITORIAL. This is the natural human review gate: the storyboard is the structural contract that editorial workers enforce.

---

## Subset + drafter-origin invocation example

The following traces what the walkthrough would look like if BOOK_COMPLETION invoked BOOK_STORYBOARD in subset mode on two DRAFTER-origin chapters (e.g., CH_06 and CH_07 were missing from the original manuscript and authored by DRAFTER).

**Invocation context:**
```
chapter_subset: ["CH_06", "CH_07"]
```

**PRESTEP changes:** QUEEN notes subset in engagement message: "BOOK_STORYBOARD mode engaged. Subset run: 2 of 7 chapters. Ready."

**STORYBOARDER changes:**
- Reads all 7 chapter files (still required to correctly describe opening states that depend on earlier chapters).
- Produces per-chapter entries only for CH_06 and CH_07.
- STORYBOARD.md header: `subset_run: true, subset_chapters: ["CH_06", "CH_07"]`.
- CH_06 and CH_07 entries include DRAFTER Origin Note sections (because their frontmatter has `drafter_origin: true`).
- Prerequisite chain reflects subset only; concepts required from CH_01–CH_05 are noted as `(excluded from subset, assumed established)`.

**QA_STORYBOARD changes:**
- Check 3 (Completeness): verifies CH_06 and CH_07 have entries — PASS if they do.
- Check 1 (Prerequisite satisfaction): concepts required from excluded chapters (CH_01–CH_05) are expected to use the "excluded from subset, assumed established" notation — not flagged as forward dependencies.
- Check 4a (Drafter-origin): CH_06 and CH_07 receive full drafter-origin scrutiny:
  - Reads each chapter file fully (not just sampled).
  - Verifies each key move in the storyboard entry against actual chapter prose.
  - Scans for `[DRAFTER_GAP: reason]` markers.
  - Verifies concepts_introduced are substantively developed, not just mentioned.
- Verdict: GREEN_LIGHT if no drafter-origin accuracy errors or gap markers; REVISE if storyboard mischaracterizes DRAFTER prose; ESCALATE if gap markers found.

---

## Gaps discovered during walkthrough

### Gap 1: CH_07 structural thinness

CH_07 (QFT and Gravity Coupling) is substantially thinner than other chapters. This is correctly noted in the storyboard (chapter_function explicitly states the chapter "opens a research direction rather than closing an argument" and notes the thinness). QA_STORYBOARD flags it informatively but does not make it blocking.

**Implication:** BOOK_EDITORIAL JUNIOR_CONCEPT or JUNIOR_FLOW may flag thin content in CH_07. The correct response is escalation to BOOK_COMPLETION / DRAFTER (Part 4.5), not a REVISE of the storyboard. The storyboard accurately describes what the chapter contains.

**Disposition:** Not a gap in the storyboard worker docs. Pipeline-level limitation as identified in CONSOLIDATION_WALKTHROUGH.md Gap 1.

### Gap 2: No REVISE cycle traced

This walkthrough achieves GREEN_LIGHT on the first QA pass and does not trace the REVISE loop. A REVISE scenario would involve:

- QA_STORYBOARD finding, e.g., `[CLEBSCH_GORDAN_COEFFICIENTS]` listed as required in CH_07 but not introduced until CH_06 — wait, CH_06 does introduce it. A realistic REVISE trigger: STORYBOARDER claims CH_03 establishes `[TOTAL_ANGULAR_MOMENTUM_J]` but that concept doesn't appear in CH_03 — it appears in CH_06. QA_STORYBOARD flags an accuracy error. STORYBOARDER re-reads CH_03, corrects the concepts_introduced list, and any downstream concepts_required entries that cited CH_03 as the source.

**Disposition:** The REVISE path is specified in BOOK_STORYBOARD.json and WORKER_STORYBOARDER.md. No gap in worker docs.

---

## Fabrication audit

All concepts, interactions, and flows in this walkthrough are derived from:
- `BOOK_STORYBOARD.json` (workflow spec — flow, roles, verdicts, hard rules)
- `WORKER_STORYBOARDER.md` (STORYBOARDER role — reading sequence, per-chapter format, full-work sections)
- `WORKER_QA_STORYBOARD.md` (7 checks — procedures, reporting, verdict logic)
- `STORYBOARD_FORMAT.md` (format spec — frontmatter, heading hierarchy, required fields)
- `CONSOLIDATION_WALKTHROUGH.md` (hypothetical manuscript — Spin of Gravity, 7 chapters, COMPOSITOR output)

The hypothetical manuscript content (EPR, Feynman diagrams as intuition-pumps, spin-first ordering) is taken directly from CONSOLIDATION_WALKTHROUGH.md. No workflow behavior was invented beyond what the worker docs specify.

---

*End of STORYBOARD_WALKTHROUGH.md.*
