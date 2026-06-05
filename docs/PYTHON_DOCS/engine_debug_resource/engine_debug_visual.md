# Engine Debug Visual Investigation

**Date**: 2026-05-22
**Module**: `engine/debug/visual/`
**Classification**: REAL (Production-Quality Implementation)
**Total Lines**: 3,762

## Executive Summary

The visual debug subsystem is a **complete, production-ready implementation** providing debug drawing primitives, interactive gizmos, render view modes, and overlay management. All components are fully implemented with proper abstractions, configuration, callbacks, and extensibility hooks. This is NOT a stub.

## Files Analyzed

| File | Lines | Classification | Purpose |
|------|-------|---------------|---------|
| `gizmos.py` | 1,282 | REAL | Interactive transform/bounds/light/camera gizmos |
| `draw.py` | 1,083 | REAL | Debug draw primitives API |
| `render_views.py` | 596 | REAL | Render buffer visualization modes |
| `overlays.py` | 573 | REAL | Subsystem debug overlay management |
| `__init__.py` | 131 | REAL | Module exports and documentation |
| `config.py` | 97 | REAL | Centralized configuration constants |

---

## Component Details

### 1. Debug Draw System (`draw.py`)

**Classification**: REAL - Full implementation

**Core Classes**:
- `Color`: Immutable RGBA color with 14 predefined constants, hex conversion, alpha modification
- `DrawOptions`: Per-primitive options (color, duration, thickness, depth_test, wireframe)
- `DrawPrimitive`: Container for primitive type, options, expiration, and shape data
- `DebugDrawBatch`: Internal batch storage with frame/persistent primitive separation
- `DebugDraw`: Static API class for all drawing operations

**Supported Primitives** (14 types):
| Primitive | Parameters | Use Case |
|-----------|------------|----------|
| `line` | start, end, color, thickness | Ray visualization, connections |
| `arrow` | origin, direction, length, head_size | Vectors, forces, directions |
| `point` | position, size | Markers, targets |
| `sphere` | center, radius, segments | Bounds, influence areas |
| `box` | center, extent, rotation | AABBs, OBBs |
| `capsule` | start, end, radius | Collision shapes |
| `cylinder` | start, end, radius | Collision shapes |
| `cone` | apex, direction, height, angle | Spotlights, vision cones |
| `circle` | center, normal, radius | Rotation arcs |
| `arc` | center, normal, start_dir, radius, angle | Partial circles |
| `triangle` | v0, v1, v2 | Surface patches |
| `plane` | center, normal, size | Ground planes |
| `screen_text` | text, x, y, scale | HUD debug info |
| `world_text` | text, position, scale, face_camera | 3D labels |

**Convenience Methods**:
- `coordinate_axes(origin, size)`: RGB XYZ axes with arrows
- `frustum(origin, direction, up, fov, aspect, near, far)`: Camera frustum visualization

**Key Features**:
- Duration-based persistence (0 = single frame)
- Configurable max primitives with auto-culling
- Warning threshold for performance monitoring
- Custom time provider injection for testing
- Vector math utilities: add, sub, scale, normalize, cross, length

---

### 2. Gizmo System (`gizmos.py`)

**Classification**: REAL - Full implementation with interaction

**Enums**:
- `GizmoType`: TRANSLATE, ROTATE, SCALE, UNIVERSAL
- `GizmoSpace`: WORLD, LOCAL, VIEW
- `GizmoAxis`: NONE, X, Y, Z, XY, XZ, YZ, XYZ, VIEW

**Base Infrastructure**:
- `GizmoStyle`: Customizable colors (x/y/z/highlight), thickness, opacity, handle sizes
- `GizmoState`: Runtime state tracking (active, hovered_axis, dragging, positions, delta)
- `BaseGizmo`: Abstract base with enable/disable, visibility, callbacks, axis color logic

**Concrete Gizmos**:

#### TransformGizmo
- **Modes**: Translate (arrows + plane handles), Rotate (circles), Scale (boxes)
- **Snapping**: Configurable translation/rotation/scale snap increments
- **Interaction**: Full drag begin/update/end pipeline
- **Plane handles**: XY, XZ, YZ for 2D constraint translation
- **Universal mode**: Combined translate + rotate

#### BoundsGizmo
- AABB/OBB visualization via center + extent + optional rotation
- Optional local axes display
- Optional size label display
- `set_from_min_max()` convenience method

#### LightGizmo
- **Light types**: POINT, SPOT, DIRECTIONAL, AREA
- Point: Central sphere + attenuation radius sphere + cross lines
- Spot: Direction arrow + inner/outer cones
- Directional: 3x3 grid of parallel arrows

#### CameraGizmo
- Full frustum visualization (near/far planes, connecting edges)
- FOV, aspect, near/far configuration
- Camera position sphere indicator

---

### 3. Render View Modes (`render_views.py`)

**Classification**: REAL - Complete mode management

**RenderViewMode Enum** (26 modes):
| Category | Modes |
|----------|-------|
| Standard | NORMAL |
| Geometry | WIREFRAME, UV_CHECKER, VERTEX_COLORS, TANGENT_SPACE, BITANGENT_SPACE |
| Material | BASE_COLOR, NORMALS, ROUGHNESS, METALLIC, AO, EMISSIVE |
| Lighting | UNLIT, SPECULAR, DIFFUSE, LIGHT_ONLY, REFLECTIONS, SHADOWS, LIGHTMAP |
| Buffers | DEPTH, STENCIL, MOTION_VECTORS |
| Performance | OVERDRAW, SHADER_COMPLEXITY, LOD_COLORING, MIPMAP_LEVEL |

**RenderViewConfig** (per-mode metadata):
- Display name and description
- Category for UI grouping
- GBuffer requirement flag
- Overlay strength
- Optional custom shader name

**RenderViewManager** (static controller):
- Mode get/set/toggle/cycle
- Previous mode tracking
- Enable/disable mode changes
- Overlay strength control
- Mode change callbacks
- Category-based mode retrieval
- GBuffer requirement queries

**Module-level convenience functions**:
- `set_view_mode()`, `get_view_mode()`, `toggle_view_mode()`, `cycle_view_mode()`

---

### 4. Debug Overlays (`overlays.py`)

**Classification**: REAL - Full overlay management system

**OverlayType Flag Enum** (14 individual + 3 combinations):
| Type | Category | Description |
|------|----------|-------------|
| PHYSICS | Simulation | Collision shapes, contacts, forces |
| NAVIGATION | Simulation | NavMesh, paths, agents |
| AI | Simulation | Perception, behavior trees, blackboard |
| RENDERING | Graphics | Bounds, wireframes, buffers |
| PARTICLES | Graphics | Emitters and bounds |
| CULLING | Graphics | Frustum/occlusion culling |
| LOD | Graphics | Level of detail transitions |
| AUDIO | Audio | Sound sources, attenuation |
| NETWORK | Network | Replication, ownership, bandwidth |
| ANIMATION | Animation | Skeleton, bones, state |
| STREAMING | Resources | Asset streaming state |
| MEMORY | System | Memory usage |
| PERFORMANCE | System | Performance metrics |
| CUSTOM | Custom | User-defined |

**Convenience combinations**: ALL, GAMEPLAY (Physics+AI+Animation), GRAPHICS (Rendering+Particles+Culling+LOD)

**Per-Overlay Settings Classes**:
- `PhysicsOverlaySettings`: collision shapes, contacts, joints, raycasts, velocities, CoM, inertia, sleep state
- `NavigationOverlaySettings`: navmesh, paths, off-mesh links, agents, obstacles, regions
- `AIOverlaySettings`: perception cones, behavior tree, EQS, blackboard, goals, debug strings
- `RenderingOverlaySettings`: bounds, wireframe, normals, tangents, UV seams, vertex colors
- `AudioOverlaySettings`: sound positions, attenuation spheres, voices, listener, reverb zones
- `NetworkOverlaySettings`: replication status, ownership, relevancy, bandwidth, packet flow, prediction errors

**DebugOverlay Manager**:
- Global enable/disable and opacity
- Per-overlay enable/toggle/disable
- Opacity and priority per overlay
- Render callback registration/unregistration
- Priority-sorted render execution with error handling
- Complete reset capability

---

### 5. Configuration (`config.py`)

**Classification**: REAL - Centralized constants

**DebugDrawConfig**:
- `vector_normalize_epsilon`: 1e-10
- `max_primitives`: 10000 (0 = unlimited)
- `primitive_warning_threshold`: 5000
- Default head/thickness/segments values

**GizmoConfig**:
- Translate plane ratios (size: 0.3, offset: 0.4, thickness: 0.01)
- Scale handle ratios (box: 0.08, center multiplier: 1.5)
- Hit test radius ratio: 0.15
- Bounds/Light/Camera gizmo-specific values
- Directional light arrow spacing and sizes

**OverlayConfig**:
- Default opacity: 1.0
- Default priority: 0

---

### 6. Module Exports (`__init__.py`)

**Classification**: REAL - Clean public API

Exports 43 symbols across:
- Config classes and instances (6)
- Draw types and classes (8)
- Overlay types, callbacks, settings (10)
- Render view types and functions (8)
- Gizmo types and classes (11)

---

## Architecture Quality

### Strengths
1. **Separation of Concerns**: Each file handles one aspect (draw, gizmo, overlay, view modes)
2. **Configuration Externalization**: All magic numbers centralized in `config.py`
3. **Static API Pattern**: `DebugDraw`, `DebugOverlay`, `RenderViewManager` are singleton-like
4. **Callback Support**: Both gizmo value changes and mode/overlay state changes
5. **Flag Enum for Overlays**: Allows bitwise combination (Physics | AI)
6. **Duration-based Primitives**: Clean single-frame vs persistent primitive handling
7. **Type Safety**: Type hints throughout, frozen dataclasses where appropriate
8. **Error Handling**: Try/except in callback invocation with graceful degradation

### Integration Points
- `DebugDraw.get_batch()` returns primitives for renderer submission
- `DebugOverlay.render(world, camera, dt)` invokes registered callbacks
- `RenderViewManager.is_gbuffer_required()` queries renderer capabilities
- Gizmos use `DebugDraw` internally for visualization

### Missing/Future Work
1. No actual GPU rendering code (primitives are batched, not rasterized)
2. Hit testing is simplified (production would use precise ray-geometry intersection)
3. No persistent serialization of overlay/gizmo state
4. No undo/redo integration for gizmo manipulation

---

## Dependencies

| Module | Depends On |
|--------|------------|
| `gizmos.py` | `config.py`, `draw.py` (Color, DebugDraw, Vec3, Quat) |
| `draw.py` | `config.py` |
| `render_views.py` | (standalone) |
| `overlays.py` | (standalone) |
| `config.py` | (standalone) |

Standard library only: `math`, `time`, `warnings`, `dataclasses`, `enum`, `typing`, `abc`

---

## Classification Justification

This module is classified as **REAL** because:
1. All methods have full implementations, not `pass` or `raise NotImplementedError`
2. Complex algorithms are present (frustum calculation, vector math, drag handling)
3. Extensive per-type configurations with sensible defaults
4. Proper state machines for gizmo interaction (begin_drag, update_drag, end_drag)
5. Resource management (primitive expiration, batch limits, warnings)
6. Production patterns (callbacks, type hints, docstrings, error handling)

---

## Usage Example

```python
from engine.debug.visual import (
    DebugDraw, Color, DebugOverlay, OverlayType,
    RenderViewMode, set_view_mode, TransformGizmo, GizmoType
)

# Debug drawing
DebugDraw.line((0,0,0), (10,0,0), Color.RED, duration=2.0)
DebugDraw.sphere((5,5,5), radius=1.0, color=Color.GREEN)
DebugDraw.coordinate_axes((0,0,0), size=2.0)

# Overlays
DebugOverlay.enable(OverlayType.PHYSICS | OverlayType.AI)
DebugOverlay.register_callback(OverlayType.PHYSICS, render_physics_debug)
DebugOverlay.render(world, camera, dt)

# Render views
set_view_mode(RenderViewMode.WIREFRAME)

# Gizmos
gizmo = TransformGizmo(mode=GizmoType.TRANSLATE)
gizmo.set_target(position=(0,0,0))
gizmo.set_snap(translation=0.5)
gizmo.render(camera)
```
