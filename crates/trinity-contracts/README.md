# Trinity Contracts

Design-by-contract annotations for Rust with automatic test generation.

## Quick Start

```rust
use trinity_contracts::contract;

#[contract]
#[requires(x > 0)]
#[ensures(*result >= x)]
fn double(x: i32) -> i32 {
    x * 2
}
```

## Features

### Runtime Checks

Preconditions and postconditions become `debug_assert!`:

```rust
#[contract]
fn sqrt(x: f64) -> f64 {
    #![requires(x >= 0.0)]
    #![ensures(*result >= 0.0)]
    x.sqrt()
}
```

### Layout Contracts

Verify struct sizes match GPU expectations:

```rust
use trinity_contracts::assert_layout;

#[repr(C)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

assert_layout!(Vertex, size = 24, align = 4);
```

### Algebraic Properties

Generate property tests for mathematical invariants:

```rust
use trinity_contracts::{verify_commutative, verify_associative};

// Verifies: add(a, b) == add(b, a)
assert!(verify_commutative(|a, b| a + b, 3, 5));

// Verifies: add(add(a, b), c) == add(a, add(b, c))
assert!(verify_associative(|a, b| a + b, 1, 2, 3));
```

## Modules

| Module | Purpose |
|--------|---------|
| `runtime` | `check_requires`, `check_ensures`, `ContractChecker` |
| `layout` | `LayoutSpec`, `WgslMirror`, `MirrorRegistry` |
| `algebra` | `Property`, `PropertyTestGenerator`, verification functions |
| `proptest` | `ParsedConstraint`, `PropertyTest`, strategy generation |
| `schema` | `ConstraintSchema`, JSON export for synth integration |
| `rollout` | `RolloutTracker`, adoption metrics |

## Constraint Types

```rust
#[requires(x > 0)]           // Precondition
#[ensures(*result < 100)]    // Postcondition
#[invariant(self.len > 0)]   // Class invariant
```

Inner attribute syntax also supported:

```rust
#[contract]
fn foo(x: i32) -> i32 {
    #![requires(x != 0)]
    #![ensures(*result > 0)]
    x.abs()
}
```

## Integration with Synth

Contracts export JSON schemas for test data generation:

```rust
use trinity_contracts::schema::{parse_constraint, infer_type};

let constraints = parse_constraint("x > 0 && x < 100");
// → [Constraint::min(1), Constraint::max(99)]
```

## Stats

- **8 source files** (2,377 lines)
- Proc-macro crate: `trinity-contracts-macros`
