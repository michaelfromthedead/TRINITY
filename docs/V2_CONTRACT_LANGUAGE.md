# V2 Contract Language

**Started:** 2026-06-05
**Status:** DESIGN PHASE
**Depends On:** V2_SUPERSTATE_VISION.md, synth

---

## Design Principles

| Principle | Decision |
|-----------|----------|
| **Ceremony** | Lightweight — Rust attributes, not separate files |
| **Language** | Rust-first — generalize to Python/WGSL later |
| **Primary Use** | synth integration — generate test inputs |
| **Secondary Use** | Documentation — contracts become docs |
| **Verification** | All 4 levels — runtime, property, static, formal |

---

## Part 1: The Attribute System

### 1.1 Basic Syntax

```rust
use trinity_contracts::*;

#[contract]
pub fn divide(a: i32, b: i32) -> i32 {
    #![requires(b != 0)]
    #![ensures(result * b == a)]
    
    a / b
}
```

The `#[contract]` attribute enables contract checking on this function.
Inner attributes `#![requires(...)]` and `#![ensures(...)]` specify conditions.

### 1.2 Full Attribute Set

```rust
// Function contracts
#![requires(condition)]      // Precondition (caller's responsibility)
#![ensures(condition)]       // Postcondition (function's promise)
#![invariant(condition)]     // Must hold before AND after

// Struct contracts
#[invariant(condition)]      // Type invariant (always holds)
#[layout(size = N, align = M)] // Memory layout constraint

// Algebraic properties
#![property(commutative)]
#![property(associative)]
#![property(identity = VALUE)]
#![property(inverse)]
#![property(idempotent)]
#![property(monotonic)]

// synth integration
#![synth(field: constraint)]  // Override default synth schema
```

### 1.3 Examples

```rust
use trinity_contracts::*;

// =============================================================================
// BASIC PRECONDITION/POSTCONDITION
// =============================================================================

#[contract]
pub fn sqrt(x: f64) -> f64 {
    #![requires(x >= 0.0)]
    #![ensures(result >= 0.0)]
    #![ensures((result * result - x).abs() < 0.0001)]
    
    x.sqrt()
}

// =============================================================================
// STRUCT INVARIANT
// =============================================================================

#[contract]
#[invariant(self.len <= self.capacity)]
#[invariant(self.capacity <= MAX_CAPACITY)]
pub struct Buffer {
    data: *mut u8,
    len: usize,
    capacity: usize,
}

// =============================================================================
// MEMORY LAYOUT (for Rust/WGSL matching)
// =============================================================================

#[contract]
#[layout(size = 144, align = 16)]
#[repr(C)]
pub struct ObjectData {
    pub transform: Mat4,      // offset 0, size 64
    pub aabb: AABB,           // offset 64, size 24
    pub mesh_index: u32,      // offset 88, size 4
    pub material_index: u32,  // offset 92, size 4
    pub flags: u32,           // offset 96, size 4
    pub _padding: [u32; 11],  // offset 100, size 44
}

// =============================================================================
// ALGEBRAIC PROPERTIES
// =============================================================================

#[contract]
pub fn add(a: i32, b: i32) -> i32 {
    #![property(commutative)]
    #![property(associative)]
    #![property(identity = 0)]
    
    a + b
}

#[contract]
pub fn max(a: i32, b: i32) -> i32 {
    #![property(commutative)]
    #![property(associative)]
    #![property(idempotent)]  // max(a, a) == a
    
    if a > b { a } else { b }
}

// =============================================================================
// COMPLEX CONDITIONS
// =============================================================================

#[contract]
pub fn binary_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    #![requires(slice.is_sorted())]
    #![ensures(match result {
        Some(i) => slice[i] == *target,
        None => !slice.contains(target)
    })]
    
    slice.binary_search(target).ok()
}

// =============================================================================
// SYNTH OVERRIDES
// =============================================================================

#[contract]
pub fn process_batch(items: Vec<Item>) -> Summary {
    #![requires(!items.is_empty())]
    #![requires(items.len() <= 1000)]
    
    // Override default synth schema for more realistic inputs
    #![synth(items: {
        len: 1..=100,
        element: Item {
            id: unique(),
            weight: 0.1..100.0,
            category: enum_of(Category)
        }
    })]
    
    // ...
}
```

---

## Part 2: Contract Expansion

The `#[contract]` proc macro expands to multiple verification levels.

### 2.1 Expansion Targets

```
Source Code + Contracts
         │
         │  #[contract] proc macro
         ▼
    ┌────────────────────────────────────────────────────────┐
    │                                                        │
    ▼                    ▼                    ▼              ▼
┌────────┐         ┌──────────┐        ┌──────────┐   ┌───────────┐
│RUNTIME │         │ PROPERTY │        │  STATIC  │   │  FORMAL   │
│CHECKS  │         │  TESTS   │        │ ANALYSIS │   │  PROOFS   │
│        │         │          │        │          │   │           │
│debug   │         │ proptest │        │ kani/    │   │ creusot/  │
│asserts │         │ synth    │        │ miri     │   │ prusti    │
└────────┘         └──────────┘        └──────────┘   └───────────┘
```

### 2.2 Level 1: Runtime Checks

Debug-mode assertions inserted at function entry/exit.

```rust
// SOURCE
#[contract]
pub fn divide(a: i32, b: i32) -> i32 {
    #![requires(b != 0)]
    #![ensures(result * b == a)]
    a / b
}

// EXPANDS TO (debug mode)
pub fn divide(a: i32, b: i32) -> i32 {
    debug_assert!(b != 0, "contract violation: requires(b != 0)");
    
    let result = {
        a / b
    };
    
    debug_assert!(result * b == a, "contract violation: ensures(result * b == a)");
    
    result
}
```

**Controlled by:** `#[cfg(debug_assertions)]` — zero cost in release.

### 2.3 Level 2: Property Tests

Generated proptest/quickcheck tests with synth integration.

```rust
// SOURCE
#[contract]
pub fn add(a: i32, b: i32) -> i32 {
    #![requires(a > 0)]
    #![requires(b > 0)]
    #![ensures(result > a)]
    #![ensures(result > b)]
    #![property(commutative)]
    a + b
}

// GENERATES (in tests module)
#[cfg(test)]
mod contract_tests_add {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        // Test precondition doesn't panic
        #[test]
        fn precondition_valid(
            a in 1i32..=i32::MAX/2,  // Derived from requires(a > 0)
            b in 1i32..=i32::MAX/2,  // Derived from requires(b > 0)
        ) {
            let _ = add(a, b);
        }
        
        // Test postcondition holds
        #[test]
        fn postcondition_holds(
            a in 1i32..=i32::MAX/2,
            b in 1i32..=i32::MAX/2,
        ) {
            let result = add(a, b);
            prop_assert!(result > a);
            prop_assert!(result > b);
        }
        
        // Test commutativity
        #[test]
        fn property_commutative(
            a in 1i32..=i32::MAX/2,
            b in 1i32..=i32::MAX/2,
        ) {
            prop_assert_eq!(add(a, b), add(b, a));
        }
    }
}
```

### 2.4 Level 3: Static Analysis

Integration with Rust verification tools.

```rust
// SOURCE
#[contract]
pub fn divide(a: i32, b: i32) -> i32 {
    #![requires(b != 0)]
    #![ensures(result * b == a)]
    a / b
}

// GENERATES (for Kani verification)
#[cfg(kani)]
mod kani_proofs_divide {
    use super::*;
    
    #[kani::proof]
    fn verify_divide() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        
        // Assume precondition
        kani::assume(b != 0);
        
        let result = divide(a, b);
        
        // Assert postcondition
        kani::assert(result * b == a, "postcondition violated");
    }
}

// GENERATES (for Miri analysis)
// Run with: MIRIFLAGS="-Zmiri-symbolic-alignment-check" cargo +nightly miri test
```

### 2.5 Level 4: Formal Verification

For critical code, generate Creusot/Prusti annotations.

```rust
// SOURCE
#[contract]
pub fn binary_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    #![requires(slice.is_sorted())]
    #![ensures(match result {
        Some(i) => i < slice.len() && slice[i] == *target,
        None => !slice.contains(target)
    })]
    // ...
}

// GENERATES (Creusot annotations)
#[requires(slice.is_sorted())]
#[ensures(match result {
    Some(i) => i@ < slice@.len() && slice@[i@] == *target,
    None => !slice@.contains(target)
})]
pub fn binary_search<T: Ord>(slice: &[T], target: &T) -> Option<usize> {
    // ...
}
```

**Note:** Level 4 requires manual verification setup. The harness generates the annotations; you run the prover.

---

## Part 3: synth Integration

The primary purpose of contracts: generate test inputs.

### 3.1 Constraint → synth Schema

```rust
// CONTRACT
#[contract]
pub fn process(x: i32, y: f64, name: String) {
    #![requires(x > 0)]
    #![requires(x < 1000)]
    #![requires(y >= 0.0)]
    #![requires(y <= 1.0)]
    #![requires(!name.is_empty())]
    #![requires(name.len() <= 100)]
}

// GENERATES synth SCHEMA
{
    "x": {
        "type": "i32",
        "range": { "low": 1, "high": 999 }
    },
    "y": {
        "type": "f64",
        "range": { "low": 0.0, "high": 1.0 }
    },
    "name": {
        "type": "String",
        "length": { "min": 1, "max": 100 }
    }
}
```

### 3.2 Property → synth Relationship

```rust
// CONTRACT
#[contract]
pub fn sort(items: &mut [i32]) {
    #![ensures(items.is_sorted())]
    #![ensures(items.len() == old(items.len()))]
    #![ensures(items.is_permutation_of(old(items)))]
}

// GENERATES synth RELATIONSHIP
{
    "function": "sort",
    "input": "items",
    "output_properties": [
        { "check": "is_sorted", "args": ["result"] },
        { "check": "len_eq", "args": ["result", "input"] },
        { "check": "is_permutation", "args": ["result", "input"] }
    ]
}
```

### 3.3 synth Override Syntax

When default inference isn't enough:

```rust
#[contract]
pub fn matrix_multiply(a: &Matrix, b: &Matrix) -> Matrix {
    #![requires(a.cols() == b.rows())]
    
    // Default synth would generate arbitrary matrices
    // Override to generate compatible dimensions
    #![synth(a: Matrix {
        rows: 1..=100,
        cols: 1..=100,
        elements: -1000.0..1000.0
    })]
    #![synth(b: Matrix {
        rows: a.cols,  // Reference to a's cols
        cols: 1..=100,
        elements: -1000.0..1000.0
    })]
}
```

### 3.4 synth API

```rust
use trinity_contracts::synth;

// Generate inputs from contracts
let inputs: Vec<(i32, f64, String)> = synth::generate::<process>(100);

// Generate with custom config
let inputs = synth::generate_with::<process>(synth::Config {
    count: 1000,
    seed: Some(42),
    strategy: synth::Strategy::Boundary,  // Focus on edge cases
});

// Shrink failing input
let minimal = synth::shrink(failing_input, |input| {
    // Returns true if input still triggers the bug
    process(input.0, input.1, input.2).is_err()
});
```

---

## Part 4: Layout Contracts

Special handling for cross-language struct alignment.

### 4.1 Rust Layout Contract

```rust
#[contract]
#[layout(size = 144, align = 16)]
#[repr(C)]
pub struct ObjectData {
    #[layout(offset = 0)]
    pub transform: Mat4,
    
    #[layout(offset = 64)]
    pub aabb: AABB,
    
    #[layout(offset = 88)]
    pub mesh_index: u32,
    
    #[layout(offset = 92)]
    pub material_index: u32,
    
    #[layout(offset = 96)]
    pub flags: u32,
    
    #[layout(offset = 100)]
    pub _padding: [u32; 11],
}
```

### 4.2 Compile-Time Verification

```rust
// The proc macro generates:
const _: () = {
    assert!(std::mem::size_of::<ObjectData>() == 144);
    assert!(std::mem::align_of::<ObjectData>() == 16);
    assert!(std::mem::offset_of!(ObjectData, transform) == 0);
    assert!(std::mem::offset_of!(ObjectData, aabb) == 64);
    assert!(std::mem::offset_of!(ObjectData, mesh_index) == 88);
    // ...
};
```

### 4.3 Cross-Language Matching

The harness extracts layout contracts and compares across languages:

```
Rust: ObjectData { size: 144, align: 16, fields: [...] }
WGSL: ObjectData { size: 144, align: 16, fields: [...] }
                                                     
                       MATCH ✓
```

If mismatched:

```
ERROR: Layout mismatch detected

Struct: ObjectData

Rust (src/gpu_driven/mod.rs:45):
  size: 144, align: 16
  fields:
    transform: offset 0, size 64
    aabb: offset 64, size 24
    mesh_index: offset 88, size 4
    ...

WGSL (shaders/build_indirect.wgsl:12):
  size: 160, align: 16       ← SIZE MISMATCH
  fields:
    transform: offset 0, size 64
    aabb: offset 64, size 32  ← FIELD SIZE MISMATCH
    mesh_index: offset 96, size 4
    ...
```

---

## Part 5: Contract Extraction

The harness extracts contracts into the database.

### 5.1 Database Schema

```sql
-- See V2_SUPERSQLITE_PERSISTENCE.md
CREATE TABLE code_contracts (
    node_id TEXT PRIMARY KEY REFERENCES code_nodes(node_id),
    
    requires TEXT,    -- JSON array of predicates
    ensures TEXT,     -- JSON array of predicates  
    invariants TEXT,  -- JSON array of predicates
    properties TEXT,  -- JSON array of properties
    
    synth_schema TEXT, -- Generated synth schema
    
    last_verified_at TEXT,
    verification_result TEXT,
    
    updated_at TEXT DEFAULT (datetime('now'))
);
```

### 5.2 Predicate Representation

```json
{
    "node_id": "fn_divide",
    "requires": [
        {
            "expr": "b != 0",
            "params": ["b"],
            "kind": "non_zero"
        }
    ],
    "ensures": [
        {
            "expr": "result * b == a",
            "params": ["result", "a", "b"],
            "kind": "custom"
        }
    ],
    "properties": [
        {
            "kind": "total",
            "note": "defined for all inputs satisfying requires"
        }
    ],
    "synth_schema": {
        "a": { "type": "i32" },
        "b": { "type": "i32", "constraint": "!= 0" }
    }
}
```

### 5.3 Extraction Process

```rust
impl ContractExtractor {
    pub fn extract_from_file(&self, path: &Path) -> Result<Vec<ContractInfo>> {
        let source = fs::read_to_string(path)?;
        let file = syn::parse_file(&source)?;
        
        let mut contracts = Vec::new();
        
        for item in &file.items {
            if let Some(contract) = self.extract_item_contract(item)? {
                contracts.push(contract);
            }
        }
        
        Ok(contracts)
    }
    
    fn extract_item_contract(&self, item: &syn::Item) -> Result<Option<ContractInfo>> {
        match item {
            syn::Item::Fn(func) => {
                if !has_contract_attr(&func.attrs) {
                    return Ok(None);
                }
                
                let requires = extract_requires(&func.block);
                let ensures = extract_ensures(&func.block);
                let properties = extract_properties(&func.block);
                let synth_schema = derive_synth_schema(&requires, &func.sig);
                
                Ok(Some(ContractInfo {
                    node_id: func_node_id(&func),
                    requires,
                    ensures,
                    properties,
                    synth_schema,
                    ..Default::default()
                }))
            }
            syn::Item::Struct(s) => {
                // Extract invariants and layout contracts
                // ...
            }
            _ => Ok(None),
        }
    }
}
```

---

## Part 6: Verification Pipeline

### 6.1 Verification Levels

```rust
#[derive(Debug, Clone, Copy)]
pub enum VerificationLevel {
    /// Runtime assertions (debug mode)
    Runtime,
    
    /// Property-based testing (proptest + synth)
    Property,
    
    /// Static analysis (Kani, Miri)
    Static,
    
    /// Formal verification (Creusot, Prusti)
    Formal,
}
```

### 6.2 Running Verification

```rust
impl ContractVerifier {
    pub fn verify(&self, node_id: &str, level: VerificationLevel) -> Result<VerificationResult> {
        let contract = self.db.get_contract(node_id)?;
        
        match level {
            VerificationLevel::Runtime => {
                // Just compile with debug_assertions
                // Verification happens at runtime
                Ok(VerificationResult::Deferred)
            }
            
            VerificationLevel::Property => {
                // Generate and run property tests
                let tests = self.generate_property_tests(&contract)?;
                self.run_tests(&tests)
            }
            
            VerificationLevel::Static => {
                // Run Kani/Miri
                let proofs = self.generate_kani_proofs(&contract)?;
                self.run_kani(&proofs)
            }
            
            VerificationLevel::Formal => {
                // Generate Creusot annotations
                // User must run the prover manually
                let annotations = self.generate_creusot(&contract)?;
                Ok(VerificationResult::Generated(annotations))
            }
        }
    }
}
```

### 6.3 CI Integration

```yaml
# CI pipeline stages

verify-contracts:
  stages:
    - level1-runtime:
        script:
          - cargo build  # Compiles with contract assertions
          
    - level2-property:
        script:
          - cargo test contract_tests  # Generated property tests
          
    - level3-static:
        script:
          - cargo kani --tests  # Kani verification
          
    - level4-formal:
        when: manual  # Formal proofs are expensive
        script:
          - creusot verify
```

---

## Part 7: Proc Macro Implementation

### 7.1 Crate Structure

```
trinity-contracts/
├── Cargo.toml
├── src/
│   ├── lib.rs           # Re-exports
│   ├── attributes.rs    # #[contract], #[requires], etc.
│   ├── expansion/
│   │   ├── mod.rs
│   │   ├── runtime.rs   # Debug assertion expansion
│   │   ├── proptest.rs  # Property test generation
│   │   ├── kani.rs      # Kani proof generation
│   │   └── creusot.rs   # Creusot annotation generation
│   ├── extraction/
│   │   ├── mod.rs
│   │   ├── predicates.rs
│   │   └── synth.rs     # synth schema derivation
│   └── layout.rs        # Layout verification
│
├── trinity-contracts-macros/  # Proc macro crate
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs
│
└── tests/
    ├── runtime.rs
    ├── property.rs
    └── layout.rs
```

### 7.2 Core Proc Macro

```rust
// trinity-contracts-macros/src/lib.rs

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, ItemStruct};

#[proc_macro_attribute]
pub fn contract(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::Item);
    
    match input {
        syn::Item::Fn(func) => expand_function_contract(func),
        syn::Item::Struct(s) => expand_struct_contract(s),
        _ => panic!("#[contract] only supports functions and structs"),
    }
}

fn expand_function_contract(func: ItemFn) -> TokenStream {
    let requires = extract_requires_attrs(&func);
    let ensures = extract_ensures_attrs(&func);
    let properties = extract_property_attrs(&func);
    
    let name = &func.sig.ident;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    
    // Generate runtime checks
    let require_checks = requires.iter().map(|r| {
        let expr = &r.expr;
        let msg = format!("contract violation: requires({})", r.source);
        quote! {
            debug_assert!(#expr, #msg);
        }
    });
    
    let ensure_checks = ensures.iter().map(|e| {
        let expr = &e.expr;
        let msg = format!("contract violation: ensures({})", e.source);
        quote! {
            debug_assert!(#expr, #msg);
        }
    });
    
    // Generate property tests
    let property_tests = generate_property_tests(name, &sig, &requires, &ensures, &properties);
    
    let expanded = quote! {
        #vis #sig {
            #(#require_checks)*
            
            let result = { #block };
            
            #(#ensure_checks)*
            
            result
        }
        
        #[cfg(test)]
        mod #name_contract_tests {
            use super::*;
            #property_tests
        }
    };
    
    expanded.into()
}
```

### 7.3 synth Schema Derivation

```rust
// src/extraction/synth.rs

use crate::predicates::Predicate;

pub fn derive_synth_schema(
    requires: &[Predicate],
    sig: &syn::Signature,
) -> SynthSchema {
    let mut schema = SynthSchema::new();
    
    for param in &sig.inputs {
        if let syn::FnArg::Typed(pat_type) = param {
            let name = extract_param_name(&pat_type.pat);
            let ty = &pat_type.ty;
            
            // Start with type-based defaults
            let mut field = SynthField::from_type(ty);
            
            // Apply constraints from requires
            for req in requires {
                if req.params.contains(&name) {
                    field.apply_constraint(&req);
                }
            }
            
            schema.add_field(name, field);
        }
    }
    
    schema
}

impl SynthField {
    fn apply_constraint(&mut self, predicate: &Predicate) {
        match &predicate.kind {
            PredicateKind::Comparison { op, lhs, rhs } => {
                // x > 0 → range.low = 1
                // x < 100 → range.high = 99
                // x != 0 → exclude 0
                match op {
                    Op::Gt => self.range.low = Some(rhs.as_int()? + 1),
                    Op::Gte => self.range.low = Some(rhs.as_int()?),
                    Op::Lt => self.range.high = Some(rhs.as_int()? - 1),
                    Op::Lte => self.range.high = Some(rhs.as_int()?),
                    Op::Ne => self.exclude.push(rhs.clone()),
                    _ => {}
                }
            }
            PredicateKind::MethodCall { method, .. } => {
                // !s.is_empty() → length.min = 1
                // slice.is_sorted() → generator = sorted
                match method.as_str() {
                    "is_empty" if predicate.negated => {
                        self.length.min = Some(1);
                    }
                    "is_sorted" => {
                        self.generator = Some(Generator::Sorted);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}
```

---

## Part 8: Python & WGSL (Future)

### 8.1 Python Contracts

Using docstrings (to be parsed by harness):

```python
def divide(a: int, b: int) -> int:
    """
    Divides a by b.
    
    Contract:
        @requires b != 0
        @ensures result * b == a
    """
    return a // b
```

Or type annotations:

```python
from trinity_contracts import requires, ensures

@requires(lambda a, b: b != 0)
@ensures(lambda result, a, b: result * b == a)
def divide(a: int, b: int) -> int:
    return a // b
```

### 8.2 WGSL Contracts

Using comments (parsed by harness):

```wgsl
// @contract
// @requires index < arrayLength(&objects)
// @ensures result.visible == true || result.visible == false
fn get_object(index: u32) -> ObjectData {
    return objects[index];
}

// @contract
// @layout(size = 144, align = 16)
struct ObjectData {
    transform: mat4x4<f32>,
    aabb: AABB,
    mesh_index: u32,
    material_index: u32,
    flags: u32,
    _padding: array<u32, 11>,
}
```

---

## Part 9: Harness Integration

### 9.1 Contract Events

```rust
pub enum ContractEvent {
    ContractDefined {
        node_id: String,
        requires: Vec<Predicate>,
        ensures: Vec<Predicate>,
        properties: Vec<Property>,
    },
    ContractChanged {
        node_id: String,
        old_hash: Hash,
        new_hash: Hash,
    },
    ContractVerified {
        node_id: String,
        level: VerificationLevel,
        result: VerificationResult,
    },
    ContractViolated {
        node_id: String,
        violation: Violation,
        input: Value,
    },
}
```

### 9.2 State Transitions

```
                          Contract defined
                                │
                                ▼
                    ┌─────────────────────┐
                    │   CONTRACT_PENDING  │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
    Level 1 OK         Level 2 OK         Level 3 OK
            │                  │                  │
            ▼                  ▼                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   RUNTIME   │    │  PROPERTY   │    │   STATIC    │
    │  VERIFIED   │    │  VERIFIED   │    │  VERIFIED   │
    └─────────────┘    └─────────────┘    └─────────────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                        All levels OK
                               │
                               ▼
                    ┌─────────────────────┐
                    │  FULLY VERIFIED     │
                    └─────────────────────┘
```

### 9.3 synth → Test → State Loop

```
┌────────────────────────────────────────────────────────────────────┐
│                     CONTRACT-DRIVEN TESTING                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐    │
│   │  CONTRACT   │──────▶│    SYNTH    │──────▶│    TEST     │    │
│   │             │       │             │       │             │    │
│   │ requires:   │       │ Generate    │       │ Run with    │    │
│   │   x > 0     │       │ inputs in   │       │ generated   │    │
│   │   x < 1000  │       │ valid range │       │ inputs      │    │
│   └─────────────┘       └─────────────┘       └──────┬──────┘    │
│                                                       │           │
│                                              ┌────────┴────────┐ │
│                                              │                 │ │
│                                         PASS │                 │ FAIL
│                                              │                 │ │
│                                              ▼                 ▼ │
│                                    ┌─────────────┐    ┌─────────────┐
│                                    │   GREEN     │    │  SHRINK +   │
│                                    │   STATE     │    │   REPORT    │
│                                    └─────────────┘    └─────────────┘
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Part 10: Quick Reference

### Attribute Cheatsheet

```rust
// === FUNCTION CONTRACTS ===

#[contract]                        // Enable contract checking
#![requires(condition)]            // Precondition
#![ensures(condition)]             // Postcondition
#![invariant(condition)]           // Pre AND post condition

// === PROPERTIES ===

#![property(commutative)]          // f(a,b) == f(b,a)
#![property(associative)]          // f(f(a,b),c) == f(a,f(b,c))
#![property(identity = VALUE)]     // f(a, VALUE) == a
#![property(inverse)]              // f(a, inv(a)) == identity
#![property(idempotent)]           // f(a,a) == a
#![property(monotonic)]            // a < b → f(a) <= f(b)

// === STRUCT CONTRACTS ===

#[contract]                        // Enable on struct
#[invariant(self.x > 0)]           // Type invariant
#[layout(size = N, align = M)]     // Memory layout

// === SYNTH OVERRIDE ===

#![synth(field: { range: 0..100 })]  // Custom synth constraints
```

### Common Patterns

```rust
// Non-null pointer
#![requires(!ptr.is_null())]

// Valid index
#![requires(index < slice.len())]

// Non-empty collection
#![requires(!items.is_empty())]

// Sorted input
#![requires(slice.is_sorted())]

// Result in range
#![ensures(result >= min && result <= max)]

// No side effects (pure)
#![ensures(result == old(f(args)))]  // Calling again gives same result

// Permutation
#![ensures(output.is_permutation_of(input))]
```

---

## Appendix: Predicate Grammar

```ebnf
predicate   = comparison | method_call | logical | literal
comparison  = expr op expr
op          = "==" | "!=" | ">" | ">=" | "<" | "<="
method_call = expr "." ident "(" [args] ")"
logical     = predicate ("&&" | "||") predicate | "!" predicate
expr        = ident | literal | expr "." ident | expr "[" expr "]"
literal     = number | string | "true" | "false"
ident       = [a-zA-Z_][a-zA-Z0-9_]*
```

Supported special identifiers:
- `result` — the return value
- `old(expr)` — value of expr before function call
- `self` — the struct instance (for invariants)

---

**Lightweight attributes. Central synth integration. Four verification levels. Start with Rust, generalize later.**
