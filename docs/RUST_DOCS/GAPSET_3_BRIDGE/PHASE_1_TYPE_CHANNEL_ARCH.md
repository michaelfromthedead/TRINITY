# PHASE 1: Type Channel Protocol

**Scope:** Register Python component type schemas with the Rust runtime -- field names, types, and byte offsets so that Rust can interpret component data without Python introspection.
**Depends on:** Phase 0 (for the type_registry.rs struct definitions)
**Produces:** ComponentMeta._build_rust_layout(), TypeRegistry in Rust, PyO3 type_register() function
**Status:** PARTIALLY IMPLEMENTED -- The Python side has a working layout builder and the _omega import wiring, but the Rust side has no PyO3 bindings, so type_register() always fails silently.

## 1. Overview

The Type Channel is the narrowest of the three bridge channels: it carries type metadata only, not live entity data. When `ComponentMeta.__new__()` creates a new component class, it computes a Rust-compatible memory layout (field names, Rust type codes, byte offsets), then calls `_omega.type_register()` to push that schema into the Rust `TypeRegistry`. Once registered, the Rust runtime knows the byte-level structure of every component and can read/write fields without Python's type system.

## 2. Architectural decisions

- **TYPE_MAP translates Python types to Rust equivalents**: `int->("i32",4)`, `float->("f32",4)`, `bool->("u8",1)`, `str->("string",8)`. Variable-width types (string) get a fixed-width placeholder offset and require special handling at the data channel layer.
- **ImportError as gate**: Python code calls `from _omega import type_register` inside a try/except ImportError block. When the PyO3 module is absent (current state), registration is silently skipped -- no crash, no warning.
- **Rust TypeRegistry uses HashMap<u32, ComponentTypeInfo>**: Component IDs (u32) map to type info structs containing name, size, and field layout vector. Does not yet use RwLock -- the TypeRegistry is created as a simple struct with no thread-safety wrapper.
- **Field offsets computed in Python, consumed in Rust**: `_build_rust_layout()` at `component_meta.py:179-186` accumulates byte offsets sequentially. The Rust side trusts these offsets to match its own struct layout.

## 3. Constraints specific to this phase

- The type registration call happens once per component class at definition time (in `ComponentMeta.__new__`), not at entity spawn time.
- Field offsets must match Rust's struct layout exactly -- padding, alignment, and ordering are critical for the data channel.
- String fields are a known weak point: variable width cannot be captured by a simple byte offset. The current solution uses string length prefix + heap storage.

## 4. Component breakdown

| File/Function | Role | Status |
|---------------|------|--------|
| `component_meta.py:64-69` | TYPE_MAP dict | EXISTS -- translates int/float/bool/str to Rust type codes |
| `component_meta.py:179-186` | `_build_rust_layout()` | EXISTS -- iterates _field_types, computes offsets |
| `component_meta.py:121-127` | `_omega.type_register()` call in Step 6b | EXISTS -- wired but always raises ImportError |
| `component_meta.py:111-112` | `Op.REGISTER` step for component registry | EXISTS -- Python-side registration works |
| `type_registry.rs:23-41` | TypeRegistry struct + register/get methods | EXISTS (no RwLock, no PyO3 bindings) |
| `type_registry.rs:1-5` | FieldLayout struct | EXISTS |
| `type_registry.rs:7-14` | FieldType enum (F32, I32, U8, String, Fixed16, Fixed32) | EXISTS |
| `bridge.rs` | PyO3 type_register() function | STUB only -- no implementation |
| `crates/renderer-backend/Cargo.toml` | PyO3 dependency | NOT ADDED -- no `pyo3` in deps |

## 5. Testing strategy

- Python side: test that `_build_rust_layout()` computes correct offsets for known field layouts.
- Python side: test that ImportError is caught silently when _omega is unavailable.
- Rust side: test TypeRegistry.register() + get() round-trip for ComponentTypeInfo.
- Integration: add PyO3 stub to omega, verify `from _omega import type_register` succeeds from Python.

## 6. Open questions

- Should the PyO3 shim live in omega (adding pyo3 dep to the math crate) or in a dedicated `trinity-bridge` crate? The existing `bridge.rs` is in renderer-backend, but Python imports it as `_omega` -- the namespace mismatch needs resolution.
- Does `RwLock` need to be added to TypeRegistry before PyO3 functions are exposed? Multiple Python threads could theoretically call type_register concurrently, though in practice ComponentMeta uses a class-level `_lock` (threading.Lock).

## 7. References

- Phase 2 (Component Store) consumes the TypeRegistry to route field reads/writes.
- Phase 0 provides the type_registry.rs struct definitions.
- GAP_3_SUMMARY.md section "Phase 1: Type Channel Protocol" (corrected status).
