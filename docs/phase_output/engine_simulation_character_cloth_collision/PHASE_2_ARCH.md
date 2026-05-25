# PHASE 2 ARCHITECTURE: Math Primitive Unification

## Objective

Extract duplicated `Vec3`, `Quaternion`, and `Transform` definitions into a shared math module, eliminating code duplication across physics modules.

## Current State

### Duplication Inventory

| File | Duplicated Types | Lines |
|------|------------------|-------|
| `collision/broadphase.py` | `Vec3`, `AABB`, `Ray` | ~150 |
| `character/character_controller.py` | `Vec3`, `Quaternion`, `Transform` | ~200 |

Total estimated duplication: ~350 lines of math primitive code.

### Issues with Duplication

1. **Maintenance burden**: Bug fixes must be applied in multiple places
2. **Inconsistency risk**: Implementations may diverge over time
3. **Import confusion**: Which `Vec3` to use?
4. **Testing overhead**: Same tests duplicated across modules

## Architecture Decisions

### ADR-MATH-001: Shared Math Module Location

**Decision**: Create `engine/math/` module with primitives.

**Structure**:
```
engine/
  math/
    __init__.py       # Re-exports all primitives
    vector.py         # Vec3, Vec2, Vec4
    quaternion.py     # Quaternion
    transform.py      # Transform
    geometry.py       # AABB, Ray, Plane, Sphere
    matrix.py         # Mat3, Mat4 (future)
```

**Rationale**:
- Central location for all math types
- Clear module boundaries
- Enables future matrix operations

### ADR-MATH-002: Dataclass-Based Implementation

**Decision**: Use `@dataclass` with `__slots__` for all primitives.

**Rationale**:
- Memory efficient (slots)
- Automatic `__init__`, `__repr__`, `__eq__`
- Type hints built-in
- Compatible with existing code style

**Example**:
```python
@dataclass(slots=True)
class Vec3:
    x: float
    y: float
    z: float
```

### ADR-MATH-003: Operator Overloading Preservation

**Decision**: Maintain full operator overloading for all primitives.

**Operators Required**:
- Arithmetic: `+`, `-`, `*`, `/`, `@` (matmul for quaternion)
- Comparison: `==`, `!=`
- Unary: `-` (negation)
- In-place: `+=`, `-=`, `*=`, `/=`

**Rationale**: Physics code relies heavily on operator syntax for readability.

### ADR-MATH-004: No External Dependencies

**Decision**: Pure Python implementation, no NumPy dependency.

**Rationale**:
- Consistent with existing codebase
- No native library complications
- Explicit control over all operations

**Future Consideration**: NumPy-backed fast path could be optional.

### ADR-MATH-005: Type Protocol for Interoperability

**Decision**: Define protocols for vector-like and transform-like types.

```python
from typing import Protocol

class VectorLike(Protocol):
    x: float
    y: float
    z: float

class TransformLike(Protocol):
    position: VectorLike
    rotation: QuaternionLike
    scale: VectorLike
```

**Rationale**: Enables duck-typing with external libraries if needed.

## Migration Strategy

### Phase 2a: Create Shared Module

1. Create `engine/math/` directory structure
2. Implement canonical versions of all primitives
3. Add comprehensive unit tests
4. Document all methods

### Phase 2b: Migrate Consumers

1. Update import statements in `broadphase.py`
2. Update import statements in `character_controller.py`
3. Update any other consumers (cloth, narrowphase, etc.)
4. Remove duplicated definitions

### Phase 2c: Verify Compatibility

1. Run all existing tests
2. Verify no behavioral changes
3. Check for import cycle issues

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Import cycles | High | Careful module ordering, lazy imports if needed |
| Subtle behavior differences | Medium | Extensive unit tests, diff existing implementations |
| Performance regression | Low | Profile critical paths, optimize hot methods |
| Missing methods | Medium | Inventory all usages before migration |

## File Changes Summary

| Action | File |
|--------|------|
| Create | `engine/math/__init__.py` |
| Create | `engine/math/vector.py` |
| Create | `engine/math/quaternion.py` |
| Create | `engine/math/transform.py` |
| Create | `engine/math/geometry.py` |
| Modify | `engine/simulation/collision/broadphase.py` |
| Modify | `engine/simulation/character/character_controller.py` |
| Create | `tests/engine/math/` test directory |
