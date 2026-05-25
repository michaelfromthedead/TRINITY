# Glossary Generator — Algorithm Document

**Used by:** FORMATTER (§5.3)
**Output:** `back/glossary.md`
**Source spec:** BOOK_PRODUCTION.json §roles.FORMATTER.responsibilities.back_matter_generation.glossary_note

---

## Purpose

Harvest defined terms from chapter files and produce a glossary with definitions extracted from their textual context. The result is a starting point for human review — not a publication-ready glossary.

---

## Algorithm

### Step 1 — STORYBOARD cross-reference (if available)

If `STORYBOARD.md` is available:

1. Extract all concepts from `#### Concepts Introduced` sections (the `[CONCEPT_NAME]: description` entries).
2. The STORYBOARD description is the 1-2 sentence description of the concept.
3. These become candidate glossary entries. The chapter where they are formally introduced (the `origin chapter`) is where the formal definition will be found.

This step provides the target list of concepts that SHOULD have glossary entries. Use it as a checklist: every STORYBOARD concept should either appear in the glossary or be explicitly excluded (e.g., concepts that are reader prerequisites, not new introductions).

---

### Step 2 — Scan for definition patterns in chapter files

Read each chapter file and scan for explicit definition markers:

**Pattern 1 — Explicit definition verb:**
```
"we define [TERM] as [DEFINITION_START]"
"we call [TERM] [DEFINITION_START]"
"[TERM] is defined as [DEFINITION_START]"
"[TERM] refers to [DEFINITION_START]"
"[TERM] is the term for [DEFINITION_START]"
```

Regex (case-insensitive):
```
(?:we (?:define|call)|(?:is|are) defined as|refers to|is the term for)\s+([A-Za-z][^.]{2,60}?)\s+(?:as|when|to be)?\s+([^.]+\.)
```

Extract: TERM (group before verb/preposition), DEFINITION (text following the verb up to sentence end).

**Pattern 2 — Em-dash or colon definition:**
```
"[TERM] — [DEFINITION]"
"[TERM]: [DEFINITION]"
```

Regex:
```
\*\*([A-Za-z][^*]{2,40})\*\*[—:\s]+([A-Za-z][^.]+\.)
```
(Only matches bold-formatted terms; bare em-dash and colon are too common for reliable extraction without bold marker.)

**Pattern 3 — Explicit parenthetical gloss:**
```
"[TERM] (that is, [DEFINITION])"
"[TERM] (i.e., [DEFINITION])"
"[TERM] (by which we mean [DEFINITION])"
```

Regex:
```
([A-Za-z][^(]{2,40})\s+\((?:that is|i\.e\.|by which we mean),?\s+([^)]+)\)
```

**Pattern 4 — "X, or Y" synonym definition:**
```
"[TERM], or [SYNONYM], is [DEFINITION]"
```

Regex:
```
([A-Za-z][^,]{2,40}),\s+or\s+([A-Za-z][^,]{2,40}),\s+(?:is|are)\s+([^.]+\.)
```

Extract: TERM, SYNONYM (add both to glossary, with the synonym entry pointing to the main term).

---

### Step 3 — Extract definition context

For each pattern match:

1. Extract the TERM (clean it: trim whitespace, remove markdown formatting).
2. Extract the DEFINITION starting text.
3. Also capture the following sentence (the sentence immediately after the definition sentence) to provide context.

Full glossary definition = definition sentence + context sentence.

If the definition sentence is very short (< 10 words), also include the preceding sentence for context.

---

### Step 4 — Flag ambiguous cases

If the pattern matched but the extracted definition seems incomplete or unclear:

```
GLOSSARY_AMBIGUOUS: "<TERM>" in <file> <section> —
  extracted: "<definition text>"
  issue: <Definition seems truncated | Term boundary unclear | Possible false positive>
  recommendation: Human review required for this entry.
```

---

### Step 5 — STORYBOARD completeness cross-check

If STORYBOARD.md was available (Step 1):

For each concept in STORYBOARD's Concepts Introduced list:
1. Check if the concept appears in the collected glossary entries (after Steps 2-4).
2. If YES: confirm the definition is present.
3. If NO: flag as a potential missing glossary entry:
   ```
   GLOSSARY_MISSING_CANDIDATE: "[CONCEPT_NAME]" (introduced in CH_NN per STORYBOARD.md) —
   No explicit definition pattern detected in chapter files.
   The STORYBOARD description is: "<description>"
   Recommendation: Verify whether this concept is formally defined in CH_NN. If so, add
   the definition manually. If the concept is a reader prerequisite (not defined in this
   work), exclude it from the glossary.
   ```

---

### Step 6 — Sort and produce

Sort alphabetically by term (A→Z, ignoring leading articles).

Produce `back/glossary.md`:

```markdown
# Glossary

**Angular momentum**: The general rotational quantity defined by transformation properties
under the rotation group SO(3). In this work, angular momentum is introduced as the
general framework of which spin is a special case. See Chapter 3.

**EPR thought experiment**: An idealized experimental scenario proposed by Einstein,
Podolsky, and Rosen in 1935, in which two entangled particles are separated and measured.
The correlations between measurements challenge local hidden variable theories.

**Larmor frequency**: The rate of precession of a magnetic moment in an external magnetic
field. Derived from the torque equation for a magnetic dipole: ω_L = (e/2m)B, where B is
the field strength. See Chapter 2.

**Non-locality**: The property of quantum entangled systems whereby measurements on
spatially separated particles are correlated in ways that cannot be explained by any
local hidden variable theory.

**Spin**: A quantum property of particles that behaves mathematically like angular
momentum (transforms as a representation of SU(2)) but does not correspond to any
physical rotation of an extended body.

[GLOSSARY_AMBIGUOUS: "..."] (if any flagged entries)

[GLOSSARY_MISSING_CANDIDATE: "..."] (if any missing STORYBOARD concepts)

---

*NOTE: This is a preliminary glossary generated by FORMATTER from automated definition
pattern extraction. Human review is expected before publication. Ambiguous and missing
entries are flagged above.*
```

---

## Output format conventions

Each entry:
```
**[term]**: [definition sentence] [context sentence if needed]
[Cross-reference: see also [related_term] if synonym was found]
```

Synonym entries:
```
**[synonym]**: See *[main term]*.
```

---

## Known limitations

- Mathematical symbols and their definitions (e.g., σ — the Pauli spin matrix) may not be captured if they use symbol notation rather than spelled-out terms. Flag these for manual addition.
- Multi-paragraph definitions (where the definition spans a paragraph boundary) are typically reduced to the first sentence plus context — the human reviewer should expand if needed.
- Circular definitions (Term A defined using Term B which is defined using Term A) will be faithfully reproduced — human reviewer must detect and resolve.
- Terms defined implicitly through extended discussion (not via any of the 4 explicit patterns) will be missed — these are the most important manual additions.
