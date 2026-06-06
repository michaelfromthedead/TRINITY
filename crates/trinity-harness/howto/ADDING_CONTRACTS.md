# How To: Adding Contracts to Your Code

> **Note:** Contracts work the same in v1 and v2. The persistence layer changes, not the macro behavior.

Step-by-step guide for annotating functions with contracts.

## Basic Pattern

```rust
use trinity_contracts::contract;

#[contract]
#[requires(/* precondition */)]
#[ensures(/* postcondition */)]
fn your_function(args...) -> ReturnType {
    // body
}
```

## Preconditions (`#[requires]`)

Conditions that must be true **before** the function runs.

```rust
// Single condition
#[requires(x > 0)]

// Multiple conditions
#[requires(x > 0)]
#[requires(y != 0)]

// Complex expression
#[requires(buffer.len() >= offset + size)]
```

## Postconditions (`#[ensures]`)

Conditions that must be true **after** the function runs. Use `*result` to refer to the return value.

```rust
// Result constraint
#[ensures(*result >= 0)]

// Result equals expression
#[ensures(*result == a + b)]

// Result property
#[ensures(result.is_some())]
```

## Inner Attributes

For longer functions, use inner attributes to keep contracts near the body:

```rust
#[contract]
fn complex_operation(data: &[u8], key: &str) -> Result<Vec<u8>, Error> {
    #![requires(!data.is_empty())]
    #![requires(!key.is_empty())]
    #![ensures(result.is_ok())]
    
    // ... long implementation
}
```

## Common Patterns

### Division Safety

```rust
#[contract]
#[requires(divisor != 0)]
fn safe_div(dividend: i32, divisor: i32) -> i32 {
    dividend / divisor
}
```

### Array Bounds

```rust
#[contract]
#[requires(index < slice.len())]
fn get_element<T: Copy>(slice: &[T], index: usize) -> T {
    slice[index]
}
```

### Non-Empty Collections

```rust
#[contract]
#[requires(!items.is_empty())]
#[ensures(*result <= items.len())]
fn find_max_index(items: &[i32]) -> usize {
    items.iter().enumerate().max_by_key(|(_, v)| *v).unwrap().0
}
```

### Option Unwrapping

```rust
#[contract]
#[requires(opt.is_some())]
fn unwrap_safe<T: Clone>(opt: &Option<T>) -> T {
    opt.clone().unwrap()
}
```

### Range Constraints

```rust
#[contract]
#[requires(value >= min && value <= max)]
fn clamp(value: i32, min: i32, max: i32) -> i32 {
    value.clamp(min, max)
}
```

## Algebraic Properties

For mathematical functions, declare algebraic properties:

```rust
use trinity_contracts::{verify_commutative, verify_associative};

// Addition is commutative: a + b == b + a
#[test]
fn test_add_commutative() {
    assert!(verify_commutative(|a, b| a + b, 3, 5));
}

// Multiplication is associative: (a * b) * c == a * (b * c)
#[test]
fn test_mul_associative() {
    assert!(verify_associative(|a, b| a * b, 2, 3, 4));
}
```

## Layout Contracts

For GPU-shared structs:

```rust
use trinity_contracts::assert_layout;

#[repr(C)]
struct GpuVertex {
    position: [f32; 4],  // 16 bytes
    color: [f32; 4],     // 16 bytes
}

// Compile-time verification
assert_layout!(GpuVertex, size = 32, align = 4);
```

## What Contracts Generate

The `#[contract]` macro expands to:

```rust
fn your_function(x: i32) -> i32 {
    debug_assert!(x > 0, "Precondition violated: x > 0");
    
    let __contract_result = (|| {
        // original body
    })();
    
    debug_assert!({
        let result = &__contract_result;
        *result >= 0
    }, "Postcondition violated: *result >= 0");
    
    __contract_result
}
```

## Best Practices

1. **Start with high-value functions** — Division, indexing, unwrapping
2. **Keep contracts simple** — Complex logic belongs in the function body
3. **Use inner attributes for long functions** — Keeps the signature clean
4. **Test contract violations** — Ensure they catch bad inputs in debug builds
5. **Document non-obvious constraints** — Why is this precondition needed?

## Rollout Strategy

```rust
use trinity_contracts::rollout::{RolloutTracker, TrackedFunction, AdoptionStatus};

let mut tracker = RolloutTracker::new();
tracker.set_phase(1);

// Track adoption
tracker.track(TrackedFunction::new("safe_div", "math").priority(10));
tracker.update_status("math::safe_div", AdoptionStatus::Validated);

// Check progress
let stats = tracker.stats();
println!("Adoption: {:.1}%", stats.adoption_rate());
```
