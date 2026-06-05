# PHASE 3: Data Channel -- GPU Math Library

**Scope:** Provide deterministic math types (Fixed32 and f32 vectors, matrices, quaternions) with bytemuck Pod/Zeroable guarantees for direct GPU upload, plus a Python-side math library that mirrors the Rust API.
**Depends on:** Phase 0 (for the omega crate in the workspace)
**Produces:** omega crate (vec, mat, quat, fixed, trig, rng modules), Python math library (engine/core/math/)
**Status:** MOSTLY IMPLEMENTED -- The omega crate is real, compiles, has tests, and covers Vec2/3/4, FVec2/3/4, Mat3/4, M64, Quat, Fixed16/32, Trig, RNG. Missing: AABB/Frustum/Ray in Rust, some vec operations (reflect, refract, min, max, clamp, distance, project, reject, homogenize).

## 1. Overview

The GPU Math Library is the best-implemented component of GAPSET_3_BRIDGE. The `omega` crate at `/home/user/dev/USER/PROJECTS_VOID/TRINITY/omega/` contains six public modules and a test suite. Every vector, matrix, and quaternion type implements `bytemuck::Pod` and `bytemuck::Zeroable`, meaning they can be cast directly to byte slices for wgpu buffer upload. Two parallel type families exist: `Vec2/3/4` and `Mat3/4` use f32 for the rendering path; `FVec2/3/4` and `M64` use `Fixed32` (Q16.16) for deterministic ECS math.

## 2. Architectural decisions

- **Fixed32 for ECS determinism, f32 for rendering**: Fixed-point arithmetic is deterministic across platforms (no NaN, no rounding-mode differences), making it suitable for networked ECS state. The rendering path uses f32 for GPU compatibility (WGSL operates on f32 natively).
- **Column-major matrix storage for WGSL compatibility**: Rust's `Mat4` stores `[[f32; 4]; 4]` in column-major order, matching WGSL's matrix layout. No transpose needed on upload.
- **Two type families**: `Vec2` (f32, f32) and `FVec2` (Fixed32, Fixed32) coexist. The Fixed32 variants use the same memory layout (64-bit structs) for ECS storage; the f32 variants are the rendering path.
- **bytemuck Pod + Zeroable on all GPU types**: This is enforced by derive macros. Any type that lacks Pod cannot be uploaded to the GPU -- this is a compile-time guarantee.
- **Python math library mirrors the Rust API**: `engine/core/math/` has vec.py, mat.py, quat.py, transform.py, geometry.py. These are the Python-side implementations for CPU-side math.

## 3. Constraints specific to this phase

- No GPU dependencies in omega -- can compile and test in CI without a GPU.
- Fixed16 (Q8.8) and Fixed32 (Q16.16) use Rust's i16 and i32 as backing storage with manual shift arithmetic.
- `serde` support is optional (behind a feature flag) -- not needed for GPU upload, useful for serialization.
- The Rust Transform struct does not exist -- Transform is represented as a Mat4 + Quat composition.

## 4. Component breakdown

| File | Types | Status |
|------|-------|--------|
| `omega/src/vec.rs:343-399` | Vec2 (f32, f32) | EXISTS -- bytemuck Pod+Zeroable, length/normalize/dot/lerp |
| `omega/src/vec.rs:405-478` | Vec3 (f32, f32, f32) | EXISTS -- cross product, all vec2 ops + more |
| `omega/src/vec.rs:484-543` | Vec4 (f32, f32, f32, f32) | EXISTS -- homogenize support |
| `omega/src/vec.rs` | FVec2/FVec3/FVec4 (Fixed32) | EXISTS -- additional, not in original spec |
| `omega/src/mat.rs` | Mat4 (column-major), Mat3, M64 (Fixed32) | EXISTS -- identity/transpose/det/inverse/multiply/translate/rotate/scale/perspective/look_at |
| `omega/src/quat.rs` | Quat (f32) | EXISTS -- slerp, bytemuck Pod+Zeroable |
| `omega/src/fixed.rs` | Fixed16 (Q8.8), Fixed32 (Q16.16) | EXISTS |
| `omega/src/trig.rs` | sin, cos, tan, atan2, etc. | EXISTS |
| `omega/src/rng.rs` | Deterministic PRNG | EXISTS |
| `omega/tests/math_tests.rs` | All math tests | EXISTS |
| `engine/core/math/vec.py` | Python Vec2/3/4 | EXISTS |
| `engine/core/math/mat.py` | Python Mat4 | EXISTS |
| `engine/core/math/quat.py` | Python Quat | EXISTS |
| `engine/core/math/transform.py` | Python Transform | EXISTS |
| `engine/core/math/geometry.py` | Python AABB/Frustum/Ray | EXISTS |

### Missing operations on existing Vec types:
- reflect, refract (shader math, used in PBR)
- min, max, clamp (component-wise)
- distance (between two points)
- project, reject (vector projection/decomposition)
- homogenize (w-divide for Vec4)

### Missing Rust port of spatial types:
- AABB (exists in Python geometry.py)
- Frustum (exists in Python light_culling.py)
- Ray (exists in Python collision system)

## 5. Testing strategy

- `cargo test -p omega` runs all math unit tests.
- Tests cover: vector arithmetic, matrix multiplication, quaternion slerp, fixed-point round-trip, trig function accuracy, RNG determinism.
- Missing test coverage: edge cases for transform composition (no Rust Transform struct), spatial queries against AABB/Frustum/Ray (not ported to Rust).
- Integration: verify that bytemuck Pod cast of Vec4 to `[u8; 16]` produces correct byte layout for wgpu.

## 6. Open questions

- Should AABB, Frustum, and Ray be added to omega or to a separate spatial crate? They have no GPU dependency, so omega is reasonable, but they are not "math" in the same sense as Vec/Mat.
- Should the existing Python spatial types be ported to Rust or should the Rust types be a fresh implementation? The Python versions are proven; a fresh Rust implementation risks divergence.
- Add reflect/refract to Vec3 (needed for PBR) or implement in WGSL? Doing it in Rust allows Python-side unit tests.

## 7. References

- Phase 6 (PBR) needs Vec3 reflect/refract and Mat4 operations.
- Phase 5 (Scene Rendering) uploads Vec3/Mat4 to GPU buffers via bytemuck Pod.
- GAP_3_SUMMARY.md section "Key Discovery: The omega Crate" (detailed module structure).
- PHASE_N_TODO.md T-BRG-3.1 through T-BRG-3.4 (corrected status: 15 real, 4 partial, 5 absent).
