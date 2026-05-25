# PHASE 0: Deterministic Math Library

**Scope:** Implement all deterministic math types (fixed-point, vector, matrix, quaternion) plus standard f32 variants, deterministic PRNG, and trigonometric lookup table, all within the `omega` crate.
**Depends on:** none
**Produces:** `omega` crate with 8 modules, 316 passing tests, bytemuck Pod/Zeroable for GPU upload
**Status:** COMPLETE

## 1. Overview

Phase 0 delivers the `omega` crate, a standalone deterministic math library that forms the foundation of the TRINITY engine's simulation numerics. It provides two parallel type families:

- **Fixed-point** (`Fixed16`, `Fixed32`, `FVec2/3/4`, `M64`, `FQuat`) -- bit-exact deterministic arithmetic for networked simulation, replay, and lockstep multiplayer. Cross-platform identical results guaranteed.
- **f32 standard** (`Vec2/3/4`, `Mat3/4`, `Quat`) -- same API surface, GPU-optimized via bytemuck, for the rendering path where slight numeric variations don't matter.

Additionally: `SimRng` (splitmix64 deterministic PRNG), `TrigLUT` (4096-entry precomputed sin/cos/tan with linear interpolation), and spatial types (`AABB`, `Ray`, `Frustum`).

All GPU-uploadable types derive `bytemuck::Pod` and `bytemuck::Zeroable`, allowing direct casting to `&[u8]` for wgpu buffer upload. The crate has zero GPU dependencies; it compiles and tests on any machine.

## 2. Architectural decisions

- **Omega is GPU-free by design.** No wgpu dependency. GPU data upload is the consumer's responsibility via bytemuck casts. This allows CI without a GPU and keeps the crate reusable across physics, networking, and rendering.
- **Dual type families (fixed + f32).** Fixed-point types are the default for deterministic simulation. f32 variants exist for rendering where performance matters more than determinism. Both share the same method naming conventions.
- **Fixed32 uses Q16.16 signed.** i32 backing with scale 65536 provides sufficient precision for ECS transforms while saturating on overflow rather than wrapping or panicking. Fixed16 uses Q8.8 (i16, scale 256) for smaller state.
- **Column-major layout matches WGSL.** Matrices are stored column-major (`mat4x4<f32>` layout), allowing direct upload as uniform buffers without reordering.
- **TrigLUT table at 4096 intervals.** Precomputed at compile-time via `OnceLock`, providing identical sin/cos/tan across all platforms. Linear interpolation keeps error below 0.0001 vs. std.
- **SimRng uses splitmix64.** Simple, fast, cross-platform identical output from a given seed. Suitable for simulation and procedural generation, not cryptography.
- **Spatial types live in omega.** AABB, Ray, Frustum use f32 Vec3, making them rendering-path types. Fixed-point spatial types could be added later.

## 3. Constraints specific to this phase

- All GPU-uploadable types must implement `bytemuck::Pod` and `Zeroable` with `#[repr(C)]` layout.
- Fixed-point operations must saturate on overflow to keep the engine in a well-defined state.
- No floating-point non-determinism permitted in fixed-point types (no `std::sync::OnceLock` in hot paths, no platform-dependent float intrinsics).
- The crate must compile and pass all tests on x86 and ARM without a GPU.

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| `omega/Cargo.toml` | Crate manifest with bytemuck, serde, parking_lot, crossbeam, slotmap, pyo3 deps | DONE |
| `omega/src/lib.rs` | Crate root: re-exports, PyO3 `_omega` module definition | DONE |
| `omega/src/fixed.rs` | `Fixed16` (Q8.8/i16), `Fixed32` (Q16.16/i32) with arithmetic, comparison, conversion, bytemuck Pod/Zeroable. 240 inline+external tests. | DONE |
| `omega/src/vec.rs` | `FVec2/3/4` (Fixed32), `Vec2/3/4` (f32). Dot, cross, length, normalize, lerp. Pod/Zeroable. 28 tests. | DONE |
| `omega/src/mat.rs` | `M64` (4x4 Fixed32), `Mat3`/`Mat4` (f32). Identity, mul, inverse, transpose, look_at, perspective, translate/rotate/scale. Pod/Zeroable. | DONE |
| `omega/src/quat.rs` | `FQuat` (Fixed32), `Quat` (f32). Identity, mul, conjugate, inverse, rotate_vector, slerp (threshold 0.9995), from_axis_angle, from_euler. Pod/Zeroable. | DONE |
| `omega/src/trig.rs` | `TrigLUT` with 4096-entry precomputed sin/cos/tan, linear interpolation, OnceLock lazy init. | DONE |
| `omega/src/rng.rs` | `SimRng` splitmix64 deterministic PRNG. u64, u32, f32, f64 output, new_skip for sub-steam positioning. | DONE |
| `omega/src/spatial.rs` | `AABB` (24 bytes), `Ray` (24 bytes), `Frustum` (96 bytes, 6 planes). Frustum from Mat4 via extract_planes. Pod/Zeroable. | DONE |
| `omega/tests/math_tests.rs` | 240 comprehensive tests covering fixed arithmetic, vector ops, quaternion ops, matrix ops, TrigLUT accuracy, SimRng determinism. | DONE |

## 5. Testing strategy

- **316 total tests** (240 in `omega/tests/math_tests.rs`, ~76 inline across modules).
- Fixed-point: arithmetic edge cases (overflow, saturation, zero, negative), round-trip f32 conversion within 1 ULP, bytemuck cast_slice.
- Vector/Matrix/Quaternion: identity operations, inverse checks, look_at/perspective correctness, slerp edge cases (identical, opposite, threshold).
- TrigLUT: error vs `std::f32::sin/cos/tan` < 0.0001 across full range.
- SimRng: deterministic sequence from same seed, distribution chi-squared.
- All tests pass on both x86 and ARM; no GPU required.

## 6. Open questions

- Should omega publish to crates.io? Currently consumed as path dependency; publication would allow external consumers but requires semver commitment.
- Fixed-point spatial types (Frustum/AABB with Fixed32 fields) would enable deterministic culling but aren't implemented. Needed for fully deterministic rendering pipeline.

## 7. References

- `omega/src/fixed.rs` -- Fixed16/Fixed32 implementation
- `omega/src/vec.rs` -- Vector types (fixed + f32)
- `omega/src/mat.rs` -- Matrix types (M64, Mat3, Mat4)
- `omega/src/quat.rs` -- Quaternion types (FQuat, Quat)
- `omega/src/trig.rs` -- Trigonometric lookup table
- `omega/src/rng.rs` -- Deterministic PRNG
- `omega/src/spatial.rs` -- AABB, Ray, Frustum
- GAP_1_SUMMARY.md -- Full source-code investigation (items T-CORE-0.1 through T-CORE-0.6)
- CLARIFICATION.md -- Design rationale for omega's GPU-free architecture
