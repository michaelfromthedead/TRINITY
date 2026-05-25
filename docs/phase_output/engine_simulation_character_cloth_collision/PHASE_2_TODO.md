# PHASE 2 TODO: Math Primitive Unification

## Objective

Extract duplicated `Vec3`, `Quaternion`, `Transform`, `AABB`, and `Ray` into a shared `engine/math/` module.

---

## Task 1: Create Math Module Structure

**Files**: New `engine/math/` directory

**Description**: Create the directory structure and `__init__.py` for the shared math module.

**Acceptance Criteria**:
- [ ] `engine/math/` directory exists
- [ ] `engine/math/__init__.py` re-exports all primitives
- [ ] Module is importable: `from engine.math import Vec3`

---

## Task 2: Implement Vec3

**File**: `engine/math/vector.py`

**Description**: Canonical Vec3 implementation with full operator overloading.

**Methods Required**:
- [ ] `__init__(x, y, z)`
- [ ] `__add__`, `__sub__`, `__mul__`, `__truediv__`
- [ ] `__neg__` (unary negation)
- [ ] `__iadd__`, `__isub__`, `__imul__`, `__itruediv__`
- [ ] `dot(other)` -> float
- [ ] `cross(other)` -> Vec3
- [ ] `length()` -> float
- [ ] `length_squared()` -> float
- [ ] `normalized()` -> Vec3
- [ ] `distance(other)` -> float
- [ ] `lerp(other, t)` -> Vec3
- [ ] Class methods: `zero()`, `one()`, `up()`, `right()`, `forward()`

**Acceptance Criteria**:
- [ ] All methods from `broadphase.py` Vec3 implemented
- [ ] All methods from `character_controller.py` Vec3 implemented
- [ ] Unit tests for each method
- [ ] `@dataclass(slots=True)` for memory efficiency

---

## Task 3: Implement Quaternion

**File**: `engine/math/quaternion.py`

**Description**: Canonical Quaternion implementation for rotations.

**Methods Required**:
- [ ] `__init__(x, y, z, w)`
- [ ] `__mul__` (quaternion composition)
- [ ] `conjugate()` -> Quaternion
- [ ] `inverse()` -> Quaternion
- [ ] `normalized()` -> Quaternion
- [ ] `rotate_vector(vec)` -> Vec3
- [ ] `slerp(other, t)` -> Quaternion
- [ ] `to_euler()` -> tuple[float, float, float]
- [ ] `to_axis_angle()` -> tuple[Vec3, float]
- [ ] Class methods: `identity()`, `from_euler(pitch, yaw, roll)`, `from_axis_angle(axis, angle)`

**Acceptance Criteria**:
- [ ] All methods from `character_controller.py` Quaternion implemented
- [ ] Quaternion error calculation for PD control supported
- [ ] Unit tests for composition, rotation, SLERP
- [ ] Handles edge cases (near-identity, opposite quaternions)

---

## Task 4: Implement Transform

**File**: `engine/math/transform.py`

**Description**: Canonical Transform implementation (position, rotation, scale).

**Methods Required**:
- [ ] `__init__(position, rotation, scale)`
- [ ] `transform_point(point)` -> Vec3
- [ ] `transform_vector(vector)` -> Vec3 (no translation)
- [ ] `inverse()` -> Transform
- [ ] `compose(other)` -> Transform
- [ ] `lerp(other, t)` -> Transform (position lerp, rotation slerp, scale lerp)
- [ ] Class method: `identity()`

**Acceptance Criteria**:
- [ ] All methods from `character_controller.py` Transform implemented
- [ ] Supports hierarchical transforms
- [ ] Unit tests for transform composition and inversion

---

## Task 5: Implement Geometry Primitives

**File**: `engine/math/geometry.py`

**Description**: AABB, Ray, Plane, Sphere primitives.

**AABB Methods**:
- [ ] `__init__(min, max)`
- [ ] `contains(point)` -> bool
- [ ] `intersects(other)` -> bool
- [ ] `expand(point)` -> AABB
- [ ] `merge(other)` -> AABB
- [ ] `center()` -> Vec3
- [ ] `extents()` -> Vec3
- [ ] `surface_area()` -> float

**Ray Methods**:
- [ ] `__init__(origin, direction)`
- [ ] `point_at(t)` -> Vec3
- [ ] `intersects_aabb(aabb)` -> tuple[bool, float] (hit, t)
- [ ] `intersects_sphere(center, radius)` -> tuple[bool, float]

**Acceptance Criteria**:
- [ ] All methods from `broadphase.py` AABB implemented
- [ ] All methods from `broadphase.py` Ray implemented
- [ ] Intersection tests match existing narrowphase behavior
- [ ] Unit tests for all intersection methods

---

## Task 6: Migrate broadphase.py

**File**: `engine/simulation/collision/broadphase.py`

**Description**: Update imports to use shared math module.

**Changes**:
- [ ] Replace local `Vec3` definition with `from engine.math import Vec3`
- [ ] Replace local `AABB` definition with `from engine.math import AABB`
- [ ] Replace local `Ray` definition with `from engine.math import Ray`
- [ ] Remove ~150 lines of duplicated math code

**Acceptance Criteria**:
- [ ] All broadphase tests pass
- [ ] No functional changes to algorithms
- [ ] File line count reduced by ~150

---

## Task 7: Migrate character_controller.py

**File**: `engine/simulation/character/character_controller.py`

**Description**: Update imports to use shared math module.

**Changes**:
- [ ] Replace local `Vec3` definition with `from engine.math import Vec3`
- [ ] Replace local `Quaternion` definition with `from engine.math import Quaternion`
- [ ] Replace local `Transform` definition with `from engine.math import Transform`
- [ ] Remove ~200 lines of duplicated math code

**Acceptance Criteria**:
- [ ] All character controller tests pass
- [ ] No functional changes to move-and-slide
- [ ] File line count reduced by ~200

---

## Task 8: Scan for Additional Consumers

**Description**: Check all simulation files for additional math primitive usage.

**Files to Check**:
- [ ] `narrowphase.py` - likely uses Vec3
- [ ] `ccd.py` - likely uses Vec3, Transform
- [ ] `cloth_simulation.py` - check for Vec3 usage
- [ ] `active_ragdoll.py` - uses Quaternion for PD control

**Acceptance Criteria**:
- [ ] All consumers identified
- [ ] All consumers migrated to shared module
- [ ] No remaining duplicate Vec3/Quaternion/Transform definitions

---

## Task 9: Add Comprehensive Tests

**Directory**: `tests/engine/math/`

**Test Files**:
- [ ] `test_vector.py`
- [ ] `test_quaternion.py`
- [ ] `test_transform.py`
- [ ] `test_geometry.py`

**Test Coverage**:
- [ ] Normal operations
- [ ] Edge cases (zero vectors, identity quaternions)
- [ ] Numerical precision (near-zero, very large values)
- [ ] Property-based tests for mathematical invariants

**Acceptance Criteria**:
- [ ] 100% method coverage for math module
- [ ] Property tests verify: `(a + b) + c == a + (b + c)`, `q * q.inverse() == identity`, etc.
- [ ] All tests pass

---

## Dependencies

- No external dependencies
- Must complete before any further physics development

## Estimated Effort

| Task | Complexity | Estimate |
|------|------------|----------|
| Task 1: Module Structure | Low | 0.5 hours |
| Task 2: Vec3 | Medium | 2 hours |
| Task 3: Quaternion | Medium | 3 hours |
| Task 4: Transform | Medium | 2 hours |
| Task 5: Geometry | Medium | 2 hours |
| Task 6: Migrate broadphase | Low | 1 hour |
| Task 7: Migrate character | Low | 1 hour |
| Task 8: Scan Consumers | Low | 1 hour |
| Task 9: Tests | Medium | 3 hours |
| **Total** | | **15.5 hours** |
