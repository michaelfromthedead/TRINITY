# XR Utils Module Investigation

**Date:** 2026-05-22  
**Path:** `engine/xr/utils/`  
**Total Lines:** 391

## Summary

The `engine/xr/utils/` package provides shared utility classes and functions for the XR subsystem. All four files contain **REAL implementations** with complete, working code -- no stubs or placeholder logic detected.

## Module Classification

| File | Lines | Classification | Description |
|------|-------|----------------|-------------|
| `math_utils.py` | 152 | REAL | Quaternion and rotation utilities using Shepperd method |
| `shading.py` | 97 | REAL | VRS shading rate conversion and pixel coverage utilities |
| `markers.py` | 93 | REAL | Type annotation markers for descriptor metadata |
| `__init__.py` | 49 | REAL | Package exports aggregating all utilities |

---

## Math Utilities (`math_utils.py`)

### Purpose
Provides rotation and quaternion conversion operations for XR components.

### Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `rotation_from_direction` | `forward: Vec3`, `up: Vec3 = None` | `Quat` | Creates quaternion rotation from forward direction; handles parallel vector edge case |
| `rotation_from_axes` | `forward: Vec3`, `up: Vec3`, `right: Vec3` | `Quat` | Builds quaternion from orthonormal basis using Shepperd method for numerical stability |
| `multiply_quaternions` | `q1: Tuple`, `q2: Tuple` | `Tuple` | Tuple-based quaternion multiplication for rendering code |
| `quaternion_to_tuple` | `q: Quat` | `Tuple[float, float, float, float]` | Converts Quat object to (x, y, z, w) tuple |
| `tuple_to_quaternion` | `t: Tuple` | `Quat` | Converts (x, y, z, w) tuple to Quat object |

### Implementation Notes
- Uses Shepperd method for matrix-to-quaternion conversion (numerically stable)
- Handles edge case where forward vector is parallel to up vector
- Forward direction is negated for OpenGL convention (-Z forward)
- Imports `Vec3` and `Quat` from `engine.core.math`

---

## Shading Utilities (`shading.py`)

### Purpose
Variable Rate Shading (VRS) utilities for foveated rendering implementations.

### Constants

**Shading Rate Values** (VRS tile size encoding):
| Rate | Integer Value | Pixel Multiplier |
|------|---------------|------------------|
| FULL | 0 | 1.0 |
| HALF_X | 1 | 0.5 |
| HALF_Y | 2 | 0.5 |
| HALF | 3 | 0.25 |
| QUARTER_X | 4 | 0.25 |
| QUARTER_Y | 5 | 0.25 |
| QUARTER | 6 | 0.0625 |

### Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `shading_rate_to_int` | `rate: ShadingRate` | `int` | Converts enum to VRS integer value (0-6) |
| `get_rate_multiplier` | `rate: ShadingRate` | `float` | Returns pixel coverage fraction for rate |

### Classes

**`ShadingRateUtils`**
- Static utility class for VRS operations
- Can be used as mixin or via static method calls
- Wraps module-level functions for class-based access

---

## Type Markers (`markers.py`)

### Purpose
Marker classes for Python `Annotated` type hints providing descriptor field metadata.

### Marker Classes

| Class | Arguments | Description |
|-------|-----------|-------------|
| `Tracked` | None | Enables change detection; notifies observers on value change |
| `Range` | `min_val: float`, `max_val: float` | Specifies value bounds; provides `clamp()` and `contains()` methods |
| `Observable` | None | Triggers UI updates when modified |
| `Transient` | None | Excludes field from serialization (runtime-only state) |
| `Immutable` | None | Marks field as read-only configuration (not enforced at runtime) |

### Range Class Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `clamp` | `value: float` | `float` | Clamps value to [min, max] |
| `contains` | `value: float` | `bool` | Returns True if value within range |

### Usage Example
```python
from typing import Annotated
from engine.xr.utils.markers import Tracked, Range

class MyComponent:
    speed: Annotated[float, Tracked(), Range(0.0, 10.0)] = 5.0
```

---

## Package Exports (`__init__.py`)

### Exported Symbols

**Math Utilities:**
- `rotation_from_direction`
- `rotation_from_axes`
- `multiply_quaternions`
- `quaternion_to_tuple`
- `tuple_to_quaternion`

**Type Markers:**
- `Tracked`
- `Range`
- `Observable`
- `Transient`
- `Immutable`

**Shading Utilities:**
- `ShadingRateUtils`
- `shading_rate_to_int`
- `get_rate_multiplier`

---

## Dependencies

| Module | External Dependencies |
|--------|----------------------|
| `math_utils.py` | `engine.core.math.vec.Vec3`, `engine.core.math.quat.Quat` |
| `shading.py` | `engine.xr.rendering.foveated.ShadingRate` |
| `markers.py` | None (pure Python) |

---

## Architecture Observations

1. **Clean Separation**: Each module has a distinct responsibility (math, shading, metadata)
2. **TYPE_CHECKING Guards**: All cross-module imports use conditional imports to avoid circular dependencies
3. **Dual API Pattern**: Shading provides both module functions and a utility class for flexibility
4. **Annotation System**: Markers enable a descriptor-based reactive system for XR components
5. **OpenGL Convention**: Math utilities follow -Z forward convention standard in OpenGL/Vulkan

## Quality Assessment

- **Documentation**: Comprehensive docstrings with Args/Returns sections
- **Type Hints**: Full type annotations on all functions
- **Edge Cases**: Math utilities handle degenerate cases (parallel vectors)
- **Numerical Stability**: Uses Shepperd method for robust quaternion conversion
