# PHASE 6: Contract Annotation — Architecture

**Duration:** Ongoing
**Depends On:** Phase 5 (Workflow Active)

---

## Overview

Incrementally add contract annotations to code. Contracts enable test generation via synth.

## Components

### 6.1 trinity_contracts Crate

```rust
// crates/trinity-contracts/src/lib.rs
pub use trinity_contracts_macros::*;

// Re-export for convenience
pub use synth_core as synth;
```

### 6.2 Proc Macro Expansion

```rust
#[contract]
pub fn divide(a: i32, b: i32) -> i32 {
    #![requires(b != 0)]
    #![ensures(result * b == a)]
    a / b
}
```

Expands to:

```rust
pub fn divide(a: i32, b: i32) -> i32 {
    // Level 1: Runtime check (debug only)
    debug_assert!(b != 0, "Precondition failed: b != 0");
    
    let result = { a / b };
    
    debug_assert!(result * b == a, "Postcondition failed: result * b == a");
    result
}

// Level 2: Property test generated
#[cfg(test)]
mod __contract_tests_divide {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_divide_contract(a in any::<i32>(), b in 1..=i32::MAX) {
            let result = divide(a, b);
            prop_assert_eq!(result * b, a);
        }
    }
}
```

### 6.3 synth Schema Generation

From contract:
```rust
#![requires(x >= 0.0 && x <= 100.0)]
```

Generate:
```json
{
  "x": {
    "type": "number",
    "subtype": "f64",
    "range": { "low": 0.0, "high": 100.0 }
  }
}
```

## Acceptance Criteria

- [ ] trinity_contracts crate created
- [ ] Proc macro parses #[contract], #![requires], #![ensures]
- [ ] Runtime checks generated (debug mode)
- [ ] Property tests generated
- [ ] synth schemas extracted
- [ ] Integration with harness (contracts stored in DB)
