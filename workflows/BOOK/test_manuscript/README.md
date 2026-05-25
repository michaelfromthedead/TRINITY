# Test Manuscript — "Spin of Gravity" (Extended)

**Purpose:** Pipeline regression-test scaffold for the BOOK workflow family.

**Version:** 1.0.0
**Date:** 2026-04-18
**Owner:** Michael (author) + Claude (co-architect)
**Used by:** T9.18, T9.19, T9.19.5, T9.19.6

---

## What this is

This directory contains a small but realistic test manuscript for the book *The Spin of Gravity:
A Field-Theoretic Introduction to Spin and Non-Locality*. It is the extended (5-chapter) version
of the 3-chapter hypothetical used in the individual workflow walkthroughs. The additional chapters
(CH_04 through CH_05 here, corresponding to the full pipeline's CH_04 through CH_07 concepts)
exercise pipeline phases that the 3-chapter mock does not fully cover.

The source documents in `source/` are the inputs for BOOK_CONSOLIDATION. They are intentionally
rough — a realistic mix of notes, partial drafts, and structural fragments, just as a real author's
working directory looks. Terminological consistency is maintained across all files even where prose
is rough.

**This test manuscript is used for:**
- T9.19 (E2E_UNIFORM_WALKTHROUGH.md): all 6 source docs, uniform-maturity scenario (ROUGH aggregate)
- T9.19.5 (E2E_CASE_A_WALKTHROUGH.md): 12-chapter scope declared, only some chapters have material
- T9.19.6 (E2E_CASE_B_WALKTHROUGH.md): 12-chapter scope, mixed chapter maturity levels

---

## Directory layout

```
test_manuscript/
├── README.md                           (this file)
├── BOOK_MANIFEST.json                  (template: BUNDLE_SPIN_OF_GRAVITY; 5-chapter scope)
├── source/                             (input for BOOK_CONSOLIDATION)
│   ├── rough_notes_preface.md          (preface-level framing notes)
│   ├── ch1_observations_draft.md       (motivation chapter — partial draft)
│   ├── ch2_frames_fragments.md         (reference frames + symmetry fragments)
│   ├── ch3_larmor_notes.md             (Larmor precession technical notes)
│   ├── ch4_angular_momentum_partial.md (partial draft on angular momentum)
│   └── ch5_synthesis_outline.md        (synthesis chapter outline)
└── (outputs populated by actual pipeline runs — empty for now)
    # After CONSOLIDATION: chapters/, STRUCTURE.md, MASTER.md, etc.
    # After STORYBOARD:    STORYBOARD.md
    # After EDITORIAL:     (revised chapter files)
    # After PRODUCTION:    BOOK_SPEC.json, front/, back/
```

---

## Intended chapter structure (declared in BOOK_MANIFEST.json)

| Index | Title | Slug | Source material |
|---|---|---|---|
| 1 | Why Non-Locality | CH_01_WHY_NON_LOCALITY | ch1_observations_draft.md |
| 2 | Reference Frames and Symmetry | CH_02_REFERENCE_FRAMES | ch2_frames_fragments.md |
| 3 | Larmor Precession and Spin | CH_03_LARMOR_SPIN | ch3_larmor_notes.md |
| 4 | Angular Momentum — The General Case | CH_04_ANGULAR_MOMENTUM | ch4_angular_momentum_partial.md |
| 5 | Synthesis: Spin, Field, and Gravity | CH_05_SYNTHESIS | ch5_synthesis_outline.md (thin) |

---

## Notes for pipeline runners

- `ch5_synthesis_outline.md` is the thinnest source document — primarily an outline. Expect
  COMPOSITOR to produce a thin CH_05 that QA_COHERENCE will note but not block. This is intentional:
  it simulates the common case where the terminal synthesis chapter is the last to be written.

- The manuscript uses BUNDLE_SPIN_OF_GRAVITY throughout. All terminology ("spin-first ordering,"
  "pedagogical contract," "show-compare-ask") should be understood in terms of that bundle's
  Characteristic Patterns and Synergy rules.

- The source docs contain internal cross-references using chapter titles (not slugs). COMPOSITOR
  should interpret these correctly.

- This is a walkthrough scaffold, not a real physics textbook. The physics content is plausible
  but not peer-review quality.

---

*End of README.md*
