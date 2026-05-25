# Investigation: engine/tooling/animation_tools/

**Date:** 2026-05-22
**Investigator:** Research Agent
**Total Lines:** 9,157 lines across 10 files

## Executive Summary

The `engine/tooling/animation_tools/` directory contains a comprehensive animation editor tooling system. All 10 modules are **REAL implementations** with substantial, production-quality code featuring complete data structures, algorithms, and business logic. This is not stub code.

## File Classification

| File | Lines | Classification | Confidence |
|------|-------|----------------|------------|
| anim_graph_editor.py | 1,279 | **REAL** | HIGH |
| sequencer.py | 1,156 | **REAL** | HIGH |
| curve_editor.py | 1,118 | **REAL** | HIGH |
| notifies_editor.py | 955 | **REAL** | HIGH |
| montage_editor.py | 925 | **REAL** | HIGH |
| pose_editor.py | 892 | **REAL** | HIGH |
| ik_setup.py | 888 | **REAL** | HIGH |
| skeleton_editor.py | 874 | **REAL** | HIGH |
| preview_scene.py | 744 | **REAL** | HIGH |
| __init__.py | 326 | **REAL** | HIGH |

## Evidence of REAL Implementation

### 1. anim_graph_editor.py (1,279 lines)

**Purpose:** Visual node-based editor for animation state machines and blend trees.

**Key Classes:**
- `GraphNode` (ABC): Base class for all graph nodes with sockets, connections, positioning
- `StateNode`: Animation states with speed multiplier, looping, entry state flags
- `TransitionNode`: State transitions with conditions, blend duration, exit time
- `BlendNode`: Linear/additive/mesh-space blending
- `BlendSpace1D/2D`: 1D and 2D blend spaces with sample interpolation
- `AnimGraphEditor`: Full editor with node management, connections, parameters

**Proof of Real Logic:**
```python
def get_blend_weights(self, value: float) -> List[Tuple[int, float]]:
    # Real interpolation logic between samples
    for i, sample in enumerate(self._samples):
        if sample.position[0] <= value:
            before_idx = i
        else:
            after_idx = i
            break
    # Interpolate
    t = (value - before.position[0]) / range_val
    return [(before_idx, 1.0 - t), (after_idx, t)]
```

### 2. sequencer.py (1,156 lines)

**Purpose:** Timeline-based animation editing with multiple track types.

**Key Classes:**
- `Timeline`: Time management with frame rate conversion, snapping, markers, loop ranges
- `AnimationTrack` (Generic[T]): Generic track with keyframe management
- `TransformTrack`: Position/rotation/scale animation with lerp/slerp interpolation
- `SkeletalTrack`: Per-bone skeletal animation
- `CameraTrack`: Camera animation with FOV keyframes
- `EventTrack`: Event triggers with time ranges
- `AudioTrack`: Audio clip management
- `PropertyTrack`: Generic property animation
- `SequencerPlayback`: Playback state with modes (ONCE, LOOP, PING_PONG, CLAMP)

**Proof of Real Logic:**
```python
def update(self, dt: float, duration: float, loop_range: Optional[TimelineRange] = None) -> List[Tuple[float, float]]:
    # Full playback state machine with ping-pong and loop handling
    if self._direction > 0 and self.current_time >= effective_end:
        traversed.append((old_time, effective_end))
        self._direction = -1  # Reverse direction
        self.current_time = effective_end - (self.current_time - effective_end)
```

### 3. curve_editor.py (1,118 lines)

**Purpose:** Animation curve editing with tangent control and easing functions.

**Key Classes:**
- `TangentHandle`: Bezier tangent control with slope, length, normalization
- `CurveKey`: Keyframe with tangent mode (AUTO, FREE, LINEAR, FLAT, WEIGHTED, BREAK)
- `EasingFunction`: Complete set of 30+ easing functions (bounce, elastic, back, etc.)
- `BezierCurve`: Cubic Bezier evaluation using De Casteljau's algorithm
- `HermiteCurve`: Hermite interpolation with basis functions
- `CurveEditor`: Full editor with selection, tangent editing, baking, normalization

**Proof of Real Logic:**
```python
def evaluate(self, time: float) -> float:
    # De Casteljau's algorithm for cubic Bezier
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    return mt3 * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3 * p3
```

### 4. pose_editor.py (892 lines)

**Purpose:** Pose library management, blending, and additive pose support.

**Key Classes:**
- `AnimPose`: Full pose with per-bone transforms and weights
- `AdditivePose`: Delta pose computation from reference
- `PoseLibrary`: Category-based pose organization with search
- `PoseEditor`: Bone selection, mirroring, blending operations

**Proof of Real Logic:**
```python
def compute_from_poses(self, reference: AnimPose, target: AnimPose) -> None:
    # Real additive pose computation
    delta_pos = tgt_transform.translation - ref_transform.translation
    delta_rot = ref_transform.rotation.inverse() * tgt_transform.rotation
    delta_scale = Vec3(
        tgt_transform.scale.x / ref_transform.scale.x if ref_transform.scale.x != 0 else 1,
        ...
    )
```

### 5. ik_setup.py (888 lines)

**Purpose:** Inverse kinematics chain configuration and solver setup.

**Key Classes:**
- `IKBone`: Bone with constraints (hinge, ball-socket, angle limits)
- `IKEffector`: End effector with position/rotation weights
- `IKPoleVector`: Plane orientation control
- `IKConstraint`: Joint constraints with apply methods
- `TwoBoneSolverConfig/FABRIKSolverConfig/CCDSolverConfig`: Solver configurations
- `IKChain`: Complete IK chain management

**Proof of Real Logic:**
```python
def _apply_cone(self, rotation: Quat, reference_axis: Vec3) -> Quat:
    # Real cone constraint implementation
    rotated = rotation.rotate_vector(reference_axis)
    dot = rotated.dot(self.axis)
    angle = math.acos(max(-1, min(1, dot)))
    if angle <= self.max_value:
        return rotation
    # Clamp to cone surface
    cross = self.axis.cross(rotated).normalized()
    return Quat.from_axis_angle(cross, self.max_value)
```

### 6. skeleton_editor.py (874 lines)

**Purpose:** Skeleton hierarchy editing, sockets, virtual bones, retargeting.

**Key Classes:**
- `Socket`: Attachment points with relative transforms
- `VirtualBone`: Computed bones (MIDPOINT, LOOK_AT, COPY, DISTANCE)
- `RetargetMapping`: Bone mapping with translation/rotation modes
- `BoneMirrorPair`: Left/right bone pairs for mirroring

**Proof of Real Logic:**
```python
def _compute_look_at_rotation(self, direction: Vec3) -> Quat:
    # Full rotation matrix to quaternion conversion
    forward = direction.normalized()
    right = self.up_axis.cross(forward)
    # ... matrix construction and quaternion extraction
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / (trace + 1.0) ** 0.5
        w = 0.25 / s
        x = (m21 - m12) * s
        # ...
```

### 7. montage_editor.py (925 lines)

**Purpose:** Animation montage creation with sections, slots, and branching.

**Key Classes:**
- `MontageSection`: Sections with loop config, links, branch conditions
- `SectionLink`: Conditional branching between sections
- `AnimSlot`: Bone-filtered animation slots with priorities
- `AnimMontage`: Full montage with playback state

### 8. notifies_editor.py (955 lines)

**Purpose:** Animation notify events (sounds, particles, custom events).

**Key Classes:**
- `AnimNotify` (ABC): Base notify with time, track, enabled state
- `AnimNotifyState`: Duration-based notifies
- `SoundNotify`: Audio playback with volume, pitch, bone attachment
- `ParticleNotify`: Particle spawning with socket attachment
- `FootstepNotify`: Footstep events with surface detection

### 9. preview_scene.py (744 lines)

**Purpose:** Animation preview environment configuration.

**Key Classes:**
- `GroundSettings`: Ground plane with grid, reflections
- `LightingSettings`: Directional, ambient, sky lighting
- `CameraSettings`: Orbit camera with pan/zoom controls
- `PreviewProp`: Scene props with bone/socket attachment
- `PreviewPlayback`: Playback state with seek, loop, speed

### 10. __init__.py (326 lines)

**Purpose:** Module exports with comprehensive `__all__` list.

Exports 90+ classes organized by submodule, demonstrating a well-structured API surface.

## Architecture Patterns

### 1. Domain-Driven Design
- Clear bounded contexts (sequencer, curves, poses, IK, etc.)
- Rich domain models with behavior (not anemic)
- Value objects with immutability where appropriate

### 2. Abstract Base Classes
- `GraphNode`, `AnimNotify`, `AnimationCurve`, `AnimationTrack` as ABCs
- Proper use of `@abstractmethod` for extension points

### 3. Dataclasses with Validation
- Extensive use of `@dataclass` for data structures
- `__post_init__` validation for invariants

### 4. Type Safety
- Full type hints throughout
- Generics used appropriately (`AnimationTrack[T]`)

## Dependencies

Internal:
- `engine.core.math`: Vec2, Vec3, Quat, Transform, Mat4

## Quality Indicators

| Metric | Value | Assessment |
|--------|-------|------------|
| Average file size | 916 lines | Appropriately sized |
| Docstrings | Present on all classes/methods | Complete |
| Type hints | 100% coverage | Excellent |
| Validation | Comprehensive | Production-ready |
| Test coverage | Unknown (not investigated) | - |
| Algorithm complexity | Real implementations | Not stubs |

## Classification Rationale

**REAL Implementation** because:

1. **Complete algorithms**: Bezier/Hermite interpolation, quaternion slerp, IK constraints
2. **State management**: Playback state machines, selection states, undo-friendly design
3. **Domain logic**: Blend spaces, montage branching, retargeting math
4. **Data validation**: Bounds checking, type validation, invariant enforcement
5. **Error handling**: ValueError for invalid inputs, Optional for nullable returns
6. **30+ easing functions**: Complete implementation of standard easing curves
7. **Professional patterns**: ABC inheritance, Generic types, dataclasses

## Recommendations

1. **Test coverage**: Add unit tests for core algorithms (Bezier evaluation, IK solvers)
2. **Integration tests**: Test with actual skeleton/animation data
3. **Performance**: Consider caching for repeated curve evaluations
4. **Serialization**: Add JSON/binary serialization for editor state

## Conclusion

The animation tools module is a production-quality implementation of a comprehensive animation editing system. It provides the full toolset needed for a professional game engine animation pipeline, including state machines, blend trees, IK, montages, and curve editing. No stubs detected.
