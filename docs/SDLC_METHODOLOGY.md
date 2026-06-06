# SDLC Methodology for TRINITY

**Purpose:** Document the software development lifecycle approach used for TRINITY infrastructure development.

---

## Overview

TRINITY uses a **phased, test-driven development approach** with strict quality gates. Every feature progresses through a standardized pipeline before integration.

---

## Pipeline: DEV → GREEN_LIGHT

```
┌─────────┐    ┌───────────────────────┐    ┌───────────┐    ┌───────────┐    ┌────────────┐    ┌─────────────┐
│   DEV   │───►│ WHITEBOX ∥ BLACKBOX   │───►│ JUNIOR_QA │───►│ SENIOR_QA │───►│ ACCEPTANCE │───►│ GREEN_LIGHT │
└─────────┘    └───────────────────────┘    └───────────┘    └───────────┘    └────────────┘    └─────────────┘
     │                    │                       │                │                 │                  │
     ▼                    ▼                       ▼                ▼                 ▼                  ▼
  Compiles           Tests pass              All pass         Edge cases        No regression        Merge
```

### Stage Definitions

| Stage | Owner | Exit Criteria | Artifacts |
|-------|-------|---------------|-----------|
| **DEV** | Developer | Code compiles, basic functionality | Source files |
| **WHITEBOX** | Developer | Internal unit tests pass | `tests/whitebox_*.rs` |
| **BLACKBOX** | QA (cleanroom) | Public API tests pass | `tests/blackbox_*.rs` |
| **JUNIOR_QA** | Automated | All tests pass | Test report |
| **SENIOR_QA** | Senior dev | Edge cases, security review | Review notes |
| **ACCEPTANCE** | Lead | No regressions, fits architecture | Sign-off |
| **GREEN_LIGHT** | CI | Merge to master | Commit |

---

## Branch Strategy

### Pattern: `task/T-{PHASE}-{NUM}`

Each task gets its own branch:

```
master
├── task/T-HARNESS-1.1  (merged)
├── task/T-HARNESS-1.2  (merged)
├── task/T-GRAPH-2.1    (merged)
│   ...
└── task/T-CONT-6.8     (merged)
```

### Commit Message Format

```
T-{PHASE}-{NUM}: {Description} — {VERDICT}

- Change bullet 1
- Change bullet 2
- N new tests (whitebox + blackbox)
- Total tests passing

Co-Authored-By: claude-flow <ruv@ruv.net>
```

### Verdicts

| Verdict | Meaning |
|---------|---------|
| GREEN_LIGHT | Passed all stages, ready to merge |
| YELLOW_LIGHT | Minor issues, conditional merge |
| RED_LIGHT | Failed, requires rework |

---

## Testing Philosophy

### Dual Test Architecture

Every module gets **two** test files:

| File | Purpose | Written By |
|------|---------|------------|
| `whitebox_*.rs` | Test internals, edge cases | Developer |
| `blackbox_*.rs` | Test public API, workflows | QA (cleanroom) |

### Cleanroom Blackbox

Blackbox tests are written **without looking at implementation**. This:
- Catches API design issues
- Ensures documentation is sufficient
- Prevents implementation bias
- Tests real-world usage patterns

### Test Naming Convention

```rust
// Whitebox: test specific function/behavior
#[test]
fn test_parser_handles_empty_input() { ... }

// Blackbox: test workflow/scenario
#[test]
fn test_full_parsing_workflow() { ... }
```

---

## Phase Organization

### Phase Structure

```
Phase N: {Name}
├── T-{PREFIX}-N.1: Task 1
├── T-{PREFIX}-N.2: Task 2
│   ...
└── T-{PREFIX}-N.8: Task 8 (typically final validation)
```

### Phase Types

| Type | Purpose | Example |
|------|---------|---------|
| **Foundation** | Core data structures | Phase 1: Code Model |
| **Construction** | Build on foundation | Phase 2: Graph Construction |
| **Integration** | Wire systems together | Phase 3: Test Mapping |
| **Tracking** | State management | Phase 4: State Tracking |
| **Workflow** | User-facing features | Phase 5: Workflow Integration |
| **Quality** | Correctness guarantees | Phase 6: Contract Annotation |

---

## Documentation Requirements

### Per-Phase Documentation

1. **TODO.md** — Task list with deliverables
2. **INPROGRESS.md** — Live work log (prepend-only)
3. **Progress Report** — Post-completion summary

### Per-Crate Documentation

1. **lib.rs docs** — Module-level documentation
2. **README.md** — Usage guide (if complex)
3. **Examples** — Runnable examples

---

## Quality Metrics

### Test Count Targets

| Module Size | Whitebox | Blackbox | Total |
|-------------|----------|----------|-------|
| Small (<200 LOC) | 10-15 | 5-8 | 15-23 |
| Medium (200-500 LOC) | 15-25 | 8-12 | 23-37 |
| Large (>500 LOC) | 25-35 | 12-20 | 37-55 |

### Coverage Expectations

| Type | Target |
|------|--------|
| Line coverage | 80%+ |
| Branch coverage | 70%+ |
| API coverage | 100% |

---

## Tooling

### Automated

| Tool | Purpose |
|------|---------|
| `cargo test` | Rust test runner |
| `uv run pytest` | Python test runner (3.13) |
| CI workflow | GitHub Actions |

### Manual

| Tool | Purpose |
|------|---------|
| `git log --oneline` | Verify commit history |
| `INPROGRESS.md` | Track live progress |
| Code review | Senior review |

---

## Anti-Patterns

### Avoid These

1. **Skipping BLACKBOX** — API issues slip through
2. **Merging without GREEN_LIGHT** — Quality regression
3. **Branch reuse** — Git history pollution
4. **Batch commits** — Hard to bisect
5. **Silent failures** — Must log all outcomes

### Encouraged Patterns

1. **Fail fast** — `#[should_panic]` for expected failures
2. **Property testing** — Proptest for invariants
3. **Regression tests** — Add test for every bug found
4. **Documentation tests** — `///` examples that compile

---

## Integration with GAPSET Workflow

This methodology integrates with the larger GAPSET system:

```
GAPSET_N
├── Phase analysis (identify gaps)
├── Task breakdown (create T-* tasks)
├── SDLC pipeline (per task)
└── GAPSET completion (all phases GREEN_LIGHT)
```

---

*This methodology evolved through Phases 1-6 of TRINITY infrastructure development.*
