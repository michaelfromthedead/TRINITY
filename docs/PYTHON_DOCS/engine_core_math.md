# Investigation: engine/core/math

## Summary

The `engine/core/math/` directory contains a **fully implemented pure Python math library** providing vectors (Vec2, Vec3, Vec4), matrices (Mat3, Mat4), quaternions (Quat), transforms, geometric primitives, and interpolation utilities. This is GRANDPHASE1 code (committed 2026-03-22) that is actively used throughout the engine's rendering and animation systems. There are **no connections to the omega Rust crate** -- the Python math library and the Rust omega crate are completely independent implementations.

## Files

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 23 | REAL | Re-exports all public types |
| `vec.py` | 303 | REAL | Vec2, Vec3, Vec4 with full operators |
| `mat.py` | 277 | REAL | Mat3, Mat4 with inverse, determinant, projections |
| `quat.py` | 171 | REAL | Unit quaternion with slerp, euler, axis-angle |
| `transform.py` | 138 | REAL | Transform and RigidTransform (TRS decomposition) |
| `geometry.py` | 109 | REAL | Ray, AABB, Sphere, Plane, Frustum |
| `interpolation.py` | 94 | REAL | lerp, easing curves, SpringDamper |
| **Total** | **1115** | | |

## Key Classes/Functions

### Vectors (`vec.py`)
- `Vec2`: 2D vector with dot, normalize, lerp, distance, min/max/clamp
- `Vec3`: 3D vector with cross product, reflect, up/forward/right constants
- `Vec4`: 4D vector with xyz property, perspective_divide

### Matrices (`mat.py`)
- `Mat3`: 3x3 column-major matrix with inverse, determinant, transpose
- `Mat4`: 4x4 column-major matrix with:
  - Factory methods: `identity()`, `translation()`, `rotation_x/y/z()`, `scale()`
  - Camera: `look_at()`, `perspective()`, `orthographic()`
  - Operations: `inverse()`, `determinant()`, `transposed()`
  - Transforms: `transform_point()`, `transform_direction()`

### Quaternion (`quat.py`)
- `Quat`: Unit quaternion (x, y, z, w) with:
  - Factory: `identity()`, `from_axis_angle()`, `from_euler()`
  - Operations: `conjugate()`, `inverse()`, `normalized()`, `slerp()`, `nlerp()`
  - Conversions: `to_mat3()`, `to_mat4()`, `to_euler()`
  - Direction helpers: `forward()`, `up()`, `right()`

### Transforms (`transform.py`)
- `Transform`: Translation + Rotation + Scale (TRS decomposition)
- `RigidTransform`: Translation + Rotation only (faster inverse)
- Both include `to_matrix()`, `from_matrix()`, `transform_point()`, `inverse()`, `lerp()`

### Geometry (`geometry.py`)
- `Ray`: origin + direction with `point_at(t)`
- `AABB`: axis-aligned bounding box with `contains()`, `intersects()`
- `Sphere`: center + radius with `contains()`, `intersects()`
- `Plane`: normal + distance with `signed_distance()`, `closest_point()`
- `Frustum`: plane array with `contains_point()`, `intersects_aabb()`

### Interpolation (`interpolation.py`)
- Functions: `lerp`, `inverse_lerp`, `remap`, `clamp`, `smoothstep`, `smootherstep`
- Easing: `in_quad`, `out_quad`, `in_out_quad`, `in_cubic`, `out_cubic`, `in_out_cubic`
- `SpringDamper`: Critically-damped spring for smooth animation

## Math Implementation

- Uses numpy? **NO**
- Uses SIMD? **NO**
- Pure Python? **YES** (uses only `math` module from stdlib)
- Fixed-point types? **NO** (all floating-point)

## Connections

### To omega/
- **NO imports from _omega** in any math file
- The omega Rust crate has its OWN math types (`vec.rs`, `mat.rs`, `quat.rs`, `fixed.rs`)
- The `_omega` Python module (via PyO3) does NOT export math types -- only bridge functions for type registry, component store, and renderer operations
- **These are parallel implementations**: Python math for GRANDPHASE1, Rust omega for GRANDPHASE2 deterministic/fixed-point math

### To rendering
- **Heavily used** by rendering system:
  - `engine/rendering/lighting/` (shadows.py, light_types.py, light_culling.py, gi_probes.py, gi_ddgi.py)
  - `engine/rendering/materials/` (pbr_model.py, material_system.py, advanced_models.py, material_graph.py)

### To animation
- **Heavily used** by animation system:
  - `engine/animation/ik/` (fabrik.py, ccd.py, ik_goal.py)
  - `engine/animation/skeletal/clip.py`
  - `engine/animation/systems/` (ik_system.py, motion_matching_system.py)
  - `engine/animation/crowds/crowd_behavior.py`

## Verdict

**REAL IMPLEMENTATION**

This is a complete, production-quality pure Python math library. All classes have full implementations with proper edge-case handling (epsilon comparisons, singular matrix warnings, division-by-zero protection). The code quality is high with docstrings, `__slots__` for memory efficiency, and comprehensive operator overloading.

## Evidence

### Vec3 cross product (line 135-140 of vec.py)
```python
def cross(self, other: Vec3) -> Vec3:
    return Vec3(
        self.y * other.z - self.z * other.y,
        self.z * other.x - self.x * other.z,
        self.x * other.y - self.y * other.x,
    )
```

### Mat4 inverse (line 177-206 of mat.py)
Full Cramer's rule implementation with determinant check:
```python
def inverse(self) -> Mat4:
    m = self.m
    inv = [0.0] * 16
    inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + ...
    # ... full cofactor expansion
    det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12]
    if abs(det) < MATH_EPSILON_TIGHT:
        logger.warning("Mat4.inverse(): singular matrix (det=%.2e), returning identity", det)
        return Mat4()  # fallback to identity
    inv_det = 1.0 / det
    return Mat4([x * inv_det for x in inv])
```

### Quat slerp (line 85-102 of quat.py)
Proper spherical linear interpolation with shortest-path handling:
```python
def slerp(self, other: Quat, t: float) -> Quat:
    d = self.dot(other)
    o = other
    if d < 0.0:  # Shortest path
        o = Quat(-o.x, -o.y, -o.z, -o.w)
        d = -d
    if d > SLERP_THRESHOLD:  # Fallback to nlerp for near-parallel
        return self.nlerp(o, t)
    theta = math.acos(min(d, 1.0))
    sin_theta = math.sin(theta)
    s0 = math.sin((1.0 - t) * theta) / sin_theta
    s1 = math.sin(t * theta) / sin_theta
    return Quat(s0*self.x + s1*o.x, s0*self.y + s1*o.y, ...)
```

### SpringDamper (line 73-94 of interpolation.py)
Critically-damped spring implementation with closed-form solution:
```python
def update(self, dt: float) -> float:
    if dt < 0.0:
        raise ValueError(f"SpringDamper.update: dt must be non-negative, got {dt}")
    delta = self.position - self.target
    exp_term = math.exp(-self.omega * dt)
    self.position = self.target + (delta + (self.velocity + self.omega * delta) * dt) * exp_term
    self.velocity = (self.velocity - self.omega * (self.velocity + self.omega * delta) * dt) * exp_term
    return self.position
```
