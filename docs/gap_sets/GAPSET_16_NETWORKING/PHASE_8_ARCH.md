# Phase 8 Architecture -- Deterministic Lockstep

> **Cluster**: GAPSET_16_NETWORKING
> **Module**: `engine/networking/determinism/network/`

---

## Overview

**Status: NOT IMPLEMENTED (0 of 7 tasks complete).**

Deterministic lockstep is the networking model used by real-time strategy (RTS) games and some fighting games. Instead of replicating entity state, it synchronizes player inputs on a common timeline, with each client simulating the game world identically from the same inputs.

---

## Current State

The entire phase consists of a single empty file:

```
determinism/network/__init__.py  -- 1 line (empty __init__)
```

No implementation exists for any of the 7 TODO tasks.

---

## TODO Task Inventory

| Task ID | Description | Status |
|---------|-------------|--------|
| T-NET-8.1 | Fixed-point math library | **[-]** |
| T-NET-8.2 | Command-based input system | **[-]** |
| T-NET-8.3 | 13-phase tick architecture | **[-]** |
| T-NET-8.4 | Deterministic random number generator | **[-]** |
| T-NET-8.5 | Hierarchical checksums | **[-]** |
| T-NET-8.6 | Lockstep synchronization protocol | **[-]** |
| T-NET-8.7 | Input queue with prediction | **[-]** |

---

## Architecture (Design)

When implemented, the lockstep architecture is expected to follow:

### Fixed-Point Math
- Deterministic across platforms (no floating-point drift)
- Configurable precision (Q16.16 or similar)
- Trigonometric and geometric operations in fixed-point
- **Blocks on**: Grail 1 (S15 Core Systems) provides this foundation

### Command-Based Input System
- Player inputs encoded as deterministic commands
- Commands are ordered and sequenced
- No variable-timestep dependency
- **Blocks on**: Grail 1 (command-based mutation pattern)

### 13-Phase Tick
- Strictly ordered per-tick phases for deterministic execution
- Each phase produces identical results given identical inputs
- **Blocks on**: Grail 1 (S15 tick architecture)

### Deterministic RNG
- Seeded PRNG, reproducible from initial seed
- Separate RNG streams per entity/system
- **Blocks on**: Grail 1 (deterministic RNG)

### Hierarchical Checksums
- Entity-level -> System-level -> World-level checksums
- Detects desync and identifies divergent entity
- Requires fixed-point math and deterministic execution

### Lockstep Protocol
- Clients send input commands (not entity state)
- Server collects N frames of input, broadcasts to all
- Clients simulate N frames deterministically
- Used in RTS games (StarCraft, Age of Empires)

### Input Queue with Prediction / Rollback
- For fighting games: GGPO-style rollback
- Execute input immediately, predict result, correct on disagreement
- Requires deterministic simulation for rollback

---

## Dependencies

| Dependency | Status | Provides |
|-----------|--------|----------|
| Grail 1 (S15 Core Systems) | Not started | Fixed-point math, command mutations, 13-phase tick, deterministic RNG, hierarchical checksums |
| Replication system | Complete | Entity state replication (complementary, not prerequisite) |

---

## Recommendation

Do not begin Phase 8 until Grail 1 completes. The lockstep phase is unique among all 10 phases in that it is 100% blocked on another gap set. Premature implementation would require rewriting once the foundational systems are available.

---

## Reality Status

- All 7 tasks: **[-]** Not implemented
- Empty `__init__.py`: **[-]** Placeholder only
- Grail 1 dependency: **[-]** Not started

---

*End of PHASE_8_ARCH.md*
