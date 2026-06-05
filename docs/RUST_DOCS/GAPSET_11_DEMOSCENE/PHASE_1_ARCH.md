# PHASE 1 ARCH: SDF Primitive Library and Combinators

## Overview

Phase 1 implements the foundational WGSL library for signed distance functions, domain deformation operations, and noise functions -- the building blocks for demoscene SDF ray marching.

## Structure

```
crates/renderer-backend/src/demoscene/          [Crate WGSL -- compilable via naga]
    noise_hash.wgsl                              T-DEMO-1.28
    noise_value.wgsl                             T-DEMO-1.29
    sdf_domain.wgsl                              T-DEMO-1.22-1.27

engine/rendering/demoscene/wgsl/                 [Engine WGSL -- individual shader files]
    sdf_sphere.wgsl                              T-DEMO-1.1
    sdf_box.wgsl                                 T-DEMO-1.2
    sdf_torus.wgsl                               T-DEMO-1.3
    sdf_cylinder.wgsl                            T-DEMO-1.4
    sdf_cone.wgsl                                T-DEMO-1.5
    sdf_plane.wgsl                               T-DEMO-1.6
    sdf_capsule.wgsl                             T-DEMO-1.7
    sdf_ellipsoid.wgsl                           T-DEMO-1.8
    sdf_box_frame.wgsl                           T-DEMO-1.9
    sdf_rounded_box.wgsl                         T-DEMO-1.10

.claude/worktrees/t-au-2.14-fix/                 [Worktree -- NOT in main source]
    crates/renderer-backend/src/demoscene/
        noise_perlin.wgsl                        T-DEMO-1.30
        noise_fbm.wgsl                           T-DEMO-1.31
```

## SDF Primitives (T-DEMO-1.1 through 1.12)

All primitives follow the Inigo Quilez convention: pure functions taking `vec3<f32>` position and shape parameters, returning `f32` signed distance (negative = inside, zero = surface, positive = outside).

### Implemented in main source (10/12):

| Primitive | Signature | Formula |
|-----------|-----------|---------|
| Sphere | `sdSphere(p, r)` | `length(p) - abs(r)` |
| Box | `sdBox(p, b)` | `length(max(q,0)) + min(maxComponent(q),0)` where `q = abs(p)-b` |
| Torus | `sdTorus(p, t)` | `length(vec2(length(p.xz)-t.x, p.y)) - t.y` |
| Cylinder | `sdCylinder(p, h, r)` | `min(max(d.x,d.y),0) + length(max(d,0))` where `d = abs(vec2(len(p.xz),p.y)) - (r,h)` |
| Cone | `sdCone(p, h, r1, r2)` | Frustum distance via 2D trapezoid projection (IQ reference sdCappedCone) |
| Plane | `sdPlane(p, n, d)` | `dot(p, normalize(n)) + d` (zero-normal guard) |
| Capsule | `sdCapsule(p, a, b, r)` | `length(pa - ba*clamp(dot(pa,ba)/max(dot(ba,ba),1e-10))) - abs(r)` |
| Ellipsoid | `sdEllipsoid(p, r)` | `(length(p/safe_r) - 1.0) * min(safe_r)` |
| Box Frame | `sdBoxFrame(p, b, e)` | `length(max(q,0)) + min(maxComponent(q),0) - e` where `q = abs(p)-b` |
| Rounded Box | `sdRoundedBox(p, b, r)` | Box SDF with shrunken q by r, minus r |

### NOT implemented anywhere:

| Primitive | Task | Status |
|-----------|------|--------|
| Octahedron | T-DEMO-1.11 | No file, no AST node, no codegen template |
| Pyramid | T-DEMO-1.12 | No file, no AST node, no codegen template |

## Domain Operations (T-DEMO-1.22 through 1.27)

All in `sdf_domain.wgsl`. These transform the coordinate space BEFORE SDF evaluation.

| Op | Function | Isometric | Compensation |
|----|----------|-----------|-------------|
| Repeat | `domain_repeat(p, c)` | Yes | None (1.0) |
| Mirror | `domain_mirror_{x,y,z}(p)` | Yes | None (1.0) |
| KIFS | `domain_kifs(p, folds)` | **No** | `domain_kifs_compensation(folds)` |
| Twist | `domain_twist(p, k)` | Yes | None (1.0) |
| Bend | `domain_bend(p, r)` | Yes | None (1.0) |
| Stretch | `domain_stretch_{x,y,z}(p, s)` | **No** | `domain_stretch_compensation(s)` |

Key detail: KIFS and Stretch are non-isometric. They compress the distance metric, so the SDF distance must be divided by the compensation factor before sphere tracing. The compensation functions are:
- KIFS: `cos(PI/N)^N` where N = number of folds
- Stretch: `min(|s|, 1/|s|)`

## Noise Functions (T-DEMO-1.28 through 1.33)

### Hash functions (T-DEMO-1.28 -- fully implemented)
`noise_hash.wgsl` implements the `fract(sin(dot(...)))` pattern with 8 hash functions:
- Scalar: hash11 (1D->f32), hash21 (2D->f32), hash31 (3D->f32), hash41 (4D->f32)
- Vector: hash22 (2D->vec2), hash32 (3D->vec2), hash33 (3D->vec3)
All use large irrational constants (0.1031, 0.1030, 0.0973) and return values in [0, 1).

### Value noise (T-DEMO-1.29 -- fully implemented)
`noise_value.wgsl` implements 1D, 2D, 3D value noise with:
- Hash-driven scalar values at integer grid points
- 6t^5 - 15t^4 + 10t^3 smoothstep fade curve
- Linear/bilinear/trilinear interpolation
- Output remapped to [-1, 1] range

### Perlin noise (T-DEMO-1.30 -- worktree only)
`noise_perlin.wgsl` implements gradient-based 3D Perlin noise with:
- 12 edge-centered gradient vectors selected via hash index
- 8-corner gradient dot product interpolation
- Same smoothstep fade curve
- Zero mean property (unique to Perlin vs value noise)
- Output in approximately [-sqrt(3), sqrt(3)]

### FBM noise (T-DEMO-1.31 -- worktree only)
`noise_fbm.wgsl` implements fractal Brownian motion with:
- 1D/2D/3D value-noise-based variants + 3D Perlin-based variant
- Configurable octaves, lacunarity (typ. 2.0), gain (typ. 0.5)
- Normalization by max_amplitude sum for consistent [-1, 1] output

### Not implemented:
- T-DEMO-1.32: Ridged noise (`1.0 - abs(FBM)`)
- T-DEMO-1.33: Domain warping (FBM-warped FBM)

## Combinators (T-DEMO-1.13 through 1.21)

**CRITICAL: NONE are implemented as standalone WGSL functions.** A `sdf_combinators.wgsl` file does not exist. The codegen in Phase 2 uses WGSL's built-in `select()` function for pairwise min/max decisions between vec2<f32>(distance, material_id) values, but:
- No `fn min2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32>` exists
- No `fn max2(...)` exists
- No `fn smin(...)` / `fn smax(...)` smooth blending exists
- No `fn sdf_displaced(...)` exists

These are critical for Phase 3 (ray marching) since the ray marching loop needs to evaluate the combined scene SDF.
