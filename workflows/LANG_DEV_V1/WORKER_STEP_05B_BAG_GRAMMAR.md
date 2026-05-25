# WORKER_STEP_05B — BAG_GRAMMAR

**You are WORKER_STEP_05B.** Spawned for `LANG_DEV_WORKFLOW` v0.2.0 phase `STEP_05B` (PHASE_02_DESIGN group).

**Court ruling context:** Per COURT #1 (SYNTHESIS), STEP_05B follows STEP_05A. Format layer is in place; you add the linguistic operator layer.

**Read first:** `workflows/SHARED/WORKER.md`, `workflows/SHARED/WORKER_PROTOCOL.md`, this doc, then:
- **Source:** `LANGS_DEVELOPMENT/STEP 5 - BAG GRAMMAR.md`
- **Phase context:** `LANGS_DEV_RDC/PHASE_02_DESIGN_ARCH.md` (§2.2 STEP 5B)
- **Task spec:** `LANGS_DEV_RDC/PHASE_02_DESIGN_TODO.md` task `T-02.3`

---

## 1. Role

You determine which of the 7 universal bag operators your domain needs, and design the grammar (BNF/EBNF form) with precedence.

The 7 operators:
1. SEQUENCE (→) — A must execute before B
2. COMPOUND (-) — A and B are inseparable
3. SCOPE (()) — A applies to group B, C, D
4. ALTERNATIVE (|) — A or B, not both
5. PARALLEL (||) — A and B independent, can run simultaneously
6. OPTIONAL (?) — A might be included or not
7. REPETITION (*+) — zero-or-more / one-or-more of A

*Grammar is escape velocity; defaults are gravity. Include operators only if your domain has no other way to express the constraint.*

---

## 2. Inputs

- `workspace_dir/<library>_decisions.json` from STEP_05A
- `workspace_dir/STEP_03/type_signatures.json` (for dependency-analysis insights that motivate grammar)
- Source doc + PHASE_02_ARCH + T-02.3 from PHASE_02_TODO

---

## 3. Outputs

- `workspace_dir/STEP_05B/bag_grammar_spec.md` — full grammar spec

Template:
```
DOMAIN: <name>

NEEDED OPERATORS:
  SEQUENCE (→): <yes | no> — because <reason>
  COMPOUND (-): ...
  ...

GRAMMAR (BNF):
  bag         ::= expr (',' expr)*
  expr        ::= alternative
  alternative ::= sequence ('|' sequence)*
  sequence    ::= compound ('→' compound)*
  compound    ::= unary ('-' unary)*
  unary       ::= primary ('?')*
  primary     ::= atom | group
  ...

PRECEDENCE (tightest first):
  1. compound (-)
  2. sequence (→)
  3. alternative (|)

EXAMPLES:
  Simple bag: LoadCSV('x'), Filter('y'), Head(5)
  With sequence: Compute('z') → Filter('z > 10')
  ...
```

---

## 4. Completion criteria (from T-02.3)

- Needed operators identified with per-operator justification (why this domain needs it)
- Unneeded operators explicitly excluded with rationale
- Grammar is unambiguous (no `A → B | C` interpretation gaps — require grouping for ambiguous cases)
- Precedence documented
- 3+ example expressions per needed operator

---

## 5. Acceptance command

```
python parse_examples.py workspace_dir/STEP_05B/bag_grammar_spec.md
# Expected: "All N example expressions parse without ambiguity."
```

---

## 6. Discipline

- **Be ruthlessly minimal.** Start with zero operators; add only on demand.
- **Define precedence explicitly.** Ambiguous precedence = user confusion.
- **Don't reinvent programming.** The bag is declarative, not a programming language.
- **Grammar is for exceptions.** If users need grammar more than 10% of the time, your DESIGN is wrong (STEP_06 defaults should cover 90%).
- **Notation consistency.** Use symbolic (→, -, |, ?, *, +) unless domain requires keyword form.
- **Respect token inventory from STEP_06_02** — grammar must use lexemes the Pre-Lexer supports.

Gotchas: grammar-overkill, ambiguous-parsing, mixing-levels (atom-level vs phase-level), Turing-completeness-creep, forgetting-the-bag.

---

## 7. If blocked

- STEP_05A decisions.json incomplete → escalate
- Domain legitimately needs a non-universal operator (e.g., temporal windows) → document in spec as "domain-specific extension"; propose notation

---

## 8. Reporting

```
==== WORKER_STEP_05B COMPLETION ====
Phase: STEP_05B — BAG_GRAMMAR
Needed operators: <list>
Excluded operators: <list with rationale>
Grammar written: BNF form, <N> productions
Precedence: <top-to-bottom>
Examples: <M> (all parse)
Output: workspace_dir/STEP_05B/bag_grammar_spec.md
Acceptance: parse_examples.py returned <output>
Fabrication_audit: zero
```

---

*End of WORKER_STEP_05B.*
