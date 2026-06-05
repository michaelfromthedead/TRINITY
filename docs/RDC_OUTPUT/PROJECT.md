# PROJECT — V2 Testing Harness

**Version:** 1.0
**Status:** SDLC-READY
**Generated:** RDC_WORKFLOW 2026-06-05

---

## Scope

Build a SOTA testing harness where **Harel statecharts are the fundamental unit**. Track code state (tested, stale, changed) across Rust, Python, and WGSL. Generate tests from contracts, not as separate artifacts.

## Goals

1. **Unified Code Model** — Every code unit (function, module, file) is a node in a graph with a statechart tracking its state
2. **Cross-Language Support** — Rust, Python, WGSL in one unified graph with cross-language edge detection (PyO3, MirrorsLayout)
3. **Contract-Driven Testing** — #[contract] attributes → synth schemas → generated test inputs
4. **Smart Staleness** — Run only stale tests, not all tests (body changes don't propagate)
5. **Verification Levels** — Runtime, Property, Static, Formal (all 4 levels from same contract)

## Constraints

1. **Big-bang migration** — V1 and V2 cannot coexist; dependency graph must be complete
2. **Lightweight contracts** — Rust attributes, not separate files
3. **Rust-first** — Generalize to Python/WGSL later
4. **synth as primary consumer** — Contract language designed for synth integration
5. **Single `brain.db`** — All state in one SuperSQLite file

## Non-Goals

- Text CLI for harness interaction (use graph/DAG visualization instead)
- Git integration (moving to Superfossil)
- Incremental adoption (big-bang only)

## Key Components

| Component | Description |
|-----------|-------------|
| **superstate** | Custom Harel statechart engine (212 tests passing) |
| **synth** | Declarative test data generator with G-F-S loop |
| **SuperSQLite** | Embedded SQLite with 15+ extensions (graph, vector, bitemporal) |
| **trinity_contracts** | Proc macro crate for contract attributes |

## Dependencies

- superstate (local)
- synth (local)
- SuperSQLite (local)
- syn (Rust AST parsing)
- rustpython_parser (Python AST)
- naga (WGSL parsing with struct layouts)
- tree-sitter (fast incremental parsing)
- notify (file watching)

## Success Metrics

| Metric | Target |
|--------|--------|
| Migration time | ~1 week for phases 1-5 |
| Test count migrated | 12,743 existing tests |
| Stale test reduction | Run 80% fewer tests on typical change |
| Alignment bugs | Zero (caught by layout contracts) |
