# PHASE 0: Crate Scaffolding

**Scope:** Create the Rust workspace skeleton that hosts the bridge's three components (type registry, data channel, renderer backend) plus the standalone math crate.
**Depends on:** none
**Produces:** Workspace Cargo.toml, renderer-backend crate (with module declarations and stub files), omega math crate (already independent)
**Status:** PARTIALLY IMPLEMENTED -- Workspace and crate skeleton exist with correct dependencies, but type_registry.rs and bridge.rs are stubs awaiting PyO3 integration.

## 1. Overview

Phase 0 establishes the Rust compilation environment. The workspace (`/home/user/dev/USER/PROJECTS_VOID/TRINITY/Cargo.toml`) links two crates: `omega` (the deterministic math library, independently usable without GPU) and `renderer-backend` (the wgpu-based rendering layer). The renderer-backend crate declares four public modules -- `frame_graph`, `type_registry`, `bridge`, and `gpu_driven` -- of which `frame_graph` and `gpu_driven` have full implementations, while `type_registry` and `bridge` exist as structural stubs.

## 2. Architectural decisions

- **Workspace resolver = "2"**: Uses Rust edition 2021 resolver for unambiguous dependency resolution across both crates.
- **omega as path dependency**: `renderer-backend/Cargo.toml` references `omega = { path = "../../omega" }` -- the math crate is consumed as a local path dep, not published.
- **renderer-backend dep set includes everything needed for GPU work**: wgpu 24, bytemuck 1 (with derive), crossbeam 0.8, parking_lot 0.12, slotmap 1. The dev-dependencies (naga 24, regex 1) support WGSL validation in tests.
- **omega has minimal deps**: only bytemuck (required) and serde (optional). No GPU deps -- can be compiled and tested in CI without a GPU.
- **type_registry.rs as a standalone struct file**: Contains `TypeRegistry` with `HashMap<u32, ComponentTypeInfo>` and `FieldLayout`/`FieldType`/`ComponentTypeInfo` struct definitions, but no `RwLock` wrapper and no PyO3 bindings.
- **bridge.rs as a comment stub**: Contains a doc-comment describing the five PyO3 functions needed (`type_register`, `component_read`, `component_write`, `component_delete`, plus world operations). No executable code.

## 3. Constraints specific to this phase

- All types destined for GPU upload must implement `bytemuck::Pod` and `bytemuck::Zeroable`.
- The bridge module must not pull wgpu into omega's dependency chain.
- Python-facing functions require PyO3, which introduces platform-specific compilation requirements (maturin or setuptools-rust).

## 4. Component breakdown

| File | Role | Status |
|------|------|--------|
| `Cargo.toml` (root) | Workspace manifest linking omega + renderer-backend | EXISTS |
| `crates/renderer-backend/Cargo.toml` | Crate manifest with wgpu/bytemuck/crossbeam/parking_lot/slotmap deps | EXISTS (corrected from original) |
| `crates/renderer-backend/src/lib.rs` | Crate root: pub mod declarations for all four modules | EXISTS |
| `crates/renderer-backend/src/type_registry.rs` | TypeRegistry, FieldLayout, ComponentTypeInfo structs | STUB (structs exist, no PyO3) |
| `crates/renderer-backend/src/bridge.rs` | PyO3 bridge function declarations | STUB (doc-comment only) |
| `crates/renderer-backend/src/frame_graph/mod.rs` | Frame graph IR type system | FULL (1,681 lines) |
| `crates/renderer-backend/src/gpu_driven/` | Buffer staging, mesh/material/texture tables | FULL |
| `omega/Cargo.toml` | Math library manifest (bytemuck, optional serde) | EXISTS |
| `omega/src/lib.rs` | Math crate root: vec, mat, quat, fixed, trig, rng modules | FULL |

## 5. Testing strategy

- `cargo build --workspace` validates all crates compile together.
- `cargo test --workspace` runs 25+ frame graph IR unit tests plus omega math tests.
- Integration tests in `crates/renderer-backend/tests/` cover buffer registry, frame graph IR, mesh/material tables, and WGSL shader validation via `include_str!`.
- Omega tests in `omega/tests/math_tests.rs` cover vector/matrix/quaternion arithmetic.

## 6. Open questions

- Should PyO3 be added to `omega` (making it a combined math+bridge crate) or should a separate `trinity-bridge` crate exist? Adding to omega is simpler but couples math to Python bindings.
- The `bridge.rs` stub needs crossbeam channels -- should these be in renderer-backend or a separate bridge crate?
- Should the workspace include a third member for the PyO3 shim crate?

## 7. References

- Phase 1 (Type Channel) depends on Phase 0 for the type_registry.rs holding structs.
- Phase 4 (wgpu Renderer) depends on Phase 0 for the renderer-backend crate skeleton.
- Phase 7 (Frame Graph) already lives in Phase 0's crate as frame_graph/mod.rs.
