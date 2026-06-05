# CLARIFICATION — V2 Testing Harness

**Purpose:** Philosophical and pedagogical framing of the V2 design.

---

## The Central Insight

**Code and tests are not two artifacts to keep in sync. They are two facets of the same stateful entity.**

A coin doesn't sync its heads and tails — they're aspects of the same object. Code is the same way:
- **Implementation** is what it does
- **Contract** is what it promises
- **State** is where it stands in its lifecycle
- **Verification** is derived from the contract, not written separately

---

## Why Harel Statecharts?

David Harel's 1987 statecharts introduced:
- **Hierarchical states** (And-states, Or-states)
- **History states** (remember where you were)
- **Orthogonal regions** (parallel composition)

This maps perfectly to code:
- A **crate** is an And-state of **modules**
- A **module** is an And-state of **files**
- A **file** is an And-state of **functions**
- Each function has its own lifecycle state

The entire codebase is a hierarchical state machine.

---

## Why Not Just Coverage?

Coverage counts lines executed. It doesn't capture:
- **Staleness** — Code changed since last test
- **Dependency impact** — A utility function changed, 47 callers are now stale
- **Cross-language effects** — Rust struct layout changed, WGSL shader is broken

The V2 harness tracks **state**, not just **coverage**.

---

## Why Contracts Over Tests?

Tests are:
- Written separately (can drift)
- Manually maintained (can rot)
- Often redundant (multiple tests check similar things)
- Sometimes wrong (test passes but doesn't verify what it claims)

Contracts are:
- Part of the code (can't drift)
- Machine-verifiable (derive tests automatically)
- Declarative (state what must hold, not how to check it)
- Multi-level (runtime, property, static, formal from same spec)

---

## Why synth as Primary?

synth is our declarative test data generator. When contracts specify:
```rust
#![requires(x >= 0.0 && x <= 100.0)]
```

synth generates inputs satisfying those constraints. No manual test case creation.

The contract language is designed for synth:
- Constraints map to schemas
- Properties map to generators
- Ranges map to distributions

---

## Why Big-Bang Migration?

V1 and V2 can't coexist because:
- V2 requires a **complete** dependency graph
- Partial graphs give wrong staleness propagation
- Tests must map to nodes, not exist in isolation

It's like switching from a file-based VCS to a database-backed one. You migrate once, fully.

---

## Why SuperSQLite?

All state in one place:
- **Graph** for dependencies
- **Events** for history
- **Bitemporal** for "what was the state at time X?"
- **Vectors** for semantic code search
- **Memory** for fast queries

No network, no separate servers, just `brain.db`.

---

## Why Multi-Language?

TRINITY spans:
- **Rust** — GPU backend
- **Python** — Engine frontend
- **WGSL** — Shader code

A struct in Rust must match its WGSL counterpart. A function in Python calls Rust via PyO3. The graph must be unified.

---

## The Alignment Bug Story

We found a bug: Rust struct had `lod_distances` at offset 104, WGSL expected 112.

The test passed. The code ran. The GPU read wrong data.

This is why:
- Layout contracts exist (`#[layout(size = N, align = M)]`)
- MirrorsLayout edges connect Rust to WGSL
- The harness validates layouts automatically

---

## Principles

1. **Code is state** — Track it, don't pretend it's static
2. **Verification is derived** — From contracts, not handwritten
3. **Cross-language is first-class** — Not an afterthought
4. **Smart staleness** — Body changes don't propagate (behavioral tests)
5. **One source of truth** — `brain.db` knows all

---

## What This Enables

| Before | After |
|--------|-------|
| Run 12,743 tests on every change | Run only stale tests |
| Hope tests catch alignment bugs | Layout contracts enforce |
| Manual test coverage analysis | Automatic from graph |
| Tests drift from code | Contracts are part of code |
| No dependency awareness | Full DAG with propagation |

---

*Build the harness. Trust the harness. Let the harness verify.*
