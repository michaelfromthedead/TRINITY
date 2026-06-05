# PHASE 3 ARCHITECTURE: Camera Subsystem

**Phase**: 3 of 3
**Subsystem**: engine/gameplay/camera
**Lines**: 7,060
**Status**: REAL IMPLEMENTATION

---

## 1. Overview

The camera subsystem provides a complete cinematic camera toolkit: 8 camera controllers, collision handling, visual effects, blending, split-screen, and spline-based rails. This is AAA-quality camera infrastructure.

---

## 2. Module Structure

```
engine/gameplay/camera/
    __init__.py          # 330 lines - Exports and docs
    constants.py         # 607 lines - 150+ parameters
    controller.py        # 1,660 lines - 8 camera modes
    collision.py         # 709 lines - Sphere-cast collision
    effects.py           # 1,317 lines - Shake, DOF, effects
    blending.py          # 1,091 lines - Transitions, split-screen
    rails.py             # 1,346 lines - Spline rails, cinematic
```

---

## 3. Component Architecture

### 3.1 Camera Controllers (controller.py)

```
CameraController (abstract)
    |-- _position: Vec3
    |-- _rotation: Quat
    |-- _fov: float
    |-- update(delta_time)
    |-- get_view_matrix() -> Mat4

FirstPersonController
    |-- _head_bob_enabled: bool
    |-- _head_bob_amplitude / frequency

ThirdPersonController
    |-- _target: Entity
    |-- _boom_arm_length: float
    |-- _target_boom_length: float
    |-- _lagged_position: Vec3
    |-- _lag_speed: float
    |-- _pitch_limits: Tuple[float, float]

OrbitController
    |-- _focus_point: Vec3
    |-- _orbit_distance: float
    |-- _auto_rotate: bool

FollowController
    |-- _target: Entity
    |-- _offset: Vec3
    |-- _lead_prediction: float

FreeController
    |-- _movement_speed: float
    |-- _look_sensitivity: float

CinematicController
    |-- _keyframes: List[CameraKeyframe]
    |-- _timeline_position: float
    |-- _easing: EasingFunction

TopDownController
    |-- _bounds: AABB
    |-- _zoom_limits: Tuple[float, float]

IsometricController
    |-- _snap_angle: float (45 degrees)
    |-- _current_rotation_index: int
```

**Lag Formula**:
```python
lag_factor = 1.0 - math.exp(-lag_speed * delta_time)
lagged_position = lagged_position.lerp(desired_pos, lag_factor)
```

### 3.2 Collision (collision.py)

```
CollisionResponse (enum)
    |-- PULL_IN: Move camera closer to target
    |-- PUSH_OUT: Move camera away from obstacle
    |-- FADE: Fade occluding object
    |-- CLIP: Adjust near clip plane
    |-- BLEND: Interpolate position

CollisionHit
    |-- hit: bool
    |-- position: Vec3
    |-- normal: Vec3
    |-- distance: float
    |-- object: Optional[Entity]

CameraCollision
    |-- _response_mode: CollisionResponse
    |-- _sphere_radius: float
    |-- sphere_cast_check(start, end) -> CollisionHit

OcclusionDetector
    |-- _fade_state: FadeState (NONE, FADING_IN, FADING_OUT)
    |-- _fade_progress: float
    |-- _hysteresis: float

TransparencyManager
    |-- _transparent_objects: Set[Entity]
    |-- mark_transparent() / restore()
```

**Sphere Cast Algorithm**:
```python
# 9 rays: center + 8 offsets
offsets = [
    right * radius, -right * radius,
    up * radius, -up * radius,
    (right + up).normalized() * radius,
    (right - up).normalized() * radius,
    (-right + up).normalized() * radius,
    (-right - up).normalized() * radius,
]
# Find closest hit, compute safe position
```

### 3.3 Camera Effects (effects.py)

```
CameraEffect (abstract)
    |-- _intensity: float
    |-- _duration: float
    |-- update(delta_time) -> Tuple[Vec3, Vec3]  # (position_offset, rotation_offset)

CameraShake
    |-- _shake_type: ShakeType
    |-- _trauma: float (0.0 - 1.0)
    +-- PerlinShake (octave noise)
    +-- SineShake (sinusoidal)
    +-- RandomShake (random displacement)
    +-- DirectionalShake (along axis)
    +-- ExplosionShake (radial falloff)
    +-- ImpactShake (impulse decay)
    +-- ContinuousShake (persistent)

FOVEffect
    |-- _modifier_stack: List[FOVModifier]
    |-- punch() / zoom()

TiltEffect (Dutch angle)
    |-- _tilt_angle: float
    |-- _tilt_speed: float

DOFEffect
    |-- _focus_distance: float
    |-- _aperture: float
    |-- _auto_focus: bool

MotionBlurEffect
    |-- _velocity_buffer: List[Vec3]
    |-- _blur_strength: float

VignetteEffect
    |-- _inner_radius: float
    |-- _outer_radius: float
    |-- _intensity: float
```

**Perlin Shake Algorithm**:
```python
amp = 1.0
freq = 1.0
for _ in range(OCTAVES):
    pos_x += sin(t * freq + offset) * amp
    pos_y += sin(t * freq * 1.1 + offset * 2) * amp
    amp *= PERSISTENCE  # e.g., 0.5
    freq *= 2.0
```

### 3.4 Blending (blending.py)

```
BlendType (enum)
    |-- LINEAR, EASE_IN, EASE_OUT, EASE_IN_OUT
    |-- CUBIC, EXPONENTIAL, ELASTIC, BOUNCE
    |-- CUSTOM

CameraBlend
    |-- _source: CameraState
    |-- _target: CameraState
    |-- _blend_type: BlendType
    |-- _progress: float
    |-- _duration: float

BlendStack
    |-- _active_blends: List[CameraBlend]
    |-- push() / pop() / update()

ViewportSplit (7 layouts)
    |-- SINGLE
    |-- HORIZONTAL_2
    |-- VERTICAL_2
    |-- QUAD
    |-- TRIPLE_LEFT, TRIPLE_RIGHT
    |-- PIP (picture-in-picture)

CameraPriority
    |-- _priority_list: List[Tuple[CameraController, int]]
    |-- get_active_camera() -> CameraController

CameraDirector
    |-- _active_camera: CameraController
    |-- _blend_stack: BlendStack
    |-- cut_to() / blend_to()
```

**Easing Formulas**:
```python
# Elastic (overshoot)
pow(2, 10 * (t - 1)) * sin((t - s) * 2 * pi / p)

# Bounce (piecewise quadratic)
if t < 1/d1: return n1 * t * t
elif t < 2/d1: return n1 * (t - 1.5/d1) * t + 0.75
# ... more segments
```

### 3.5 Rails (rails.py)

```
SplineType (enum)
    |-- LINEAR
    |-- CATMULL_ROM
    |-- BEZIER
    |-- HERMITE

CameraRail
    |-- _control_points: List[Vec3]
    |-- _spline_type: SplineType
    |-- _arc_length_table: List[float]
    |-- evaluate(t) -> Vec3
    |-- evaluate_arc_length(s) -> Vec3

RailFollower
    |-- _rail: CameraRail
    |-- _position: float (0.0 - 1.0)
    |-- _speed: float
    |-- _loop_mode: LoopMode (ONCE, LOOP, PING_PONG)

TriggerVolume
    |-- _bounds: AABB
    |-- _on_enter / _on_exit / _on_stay: Callable

BlendRegion
    |-- _entry_t / _exit_t: float
    |-- _blend_duration: float

Dolly
    |-- _rail: CameraRail
    |-- _look_at: Optional[Entity]

Crane
    |-- _arm_length: float
    |-- _arm_angle: float
    |-- _base_position: Vec3
```

**Spline Algorithms**:

```python
# Catmull-Rom (tension-adjusted)
s = (1.0 - tension) / 2.0
b0 = -s*t3 + 2*s*t2 - s*t
b1 = (2-s)*t3 + (s-3)*t2 + 1
b2 = (s-2)*t3 + (3-2*s)*t2 + s*t
b3 = s*t3 - s*t2
result = p0*b0 + p1*b1 + p2*b2 + p3*b3

# Hermite
h00 = 2*t3 - 3*t2 + 1
h10 = t3 - 2*t2 + t
h01 = -2*t3 + 3*t2
h11 = t3 - t2
result = p0*h00 + m0*h10 + p1*h01 + m1*h11

# Arc-length parameterization
# Binary search in arc_length_table for uniform t
```

---

## 4. Integration Points

| Component | Trinity Integration |
|-----------|---------------------|
| CameraController | ComponentMeta, TrackedDescriptor on position/rotation |
| CameraEffect | ComponentMeta for effect stacking |
| BlendStack | EventMeta for BlendStartEvent, BlendEndEvent |
| CameraRail | AssetMeta for serialized rails |

---

## 5. Dependencies

```python
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.mat import Mat4
from engine.simulation.physics import PhysicsWorld  # TYPE_CHECKING
from engine.gameplay.camera.constants import *
from engine.gameplay.components.transform import TransformComponent
```

---

## 6. Design Decisions

### 6.1 Why 8 Controllers?
Covers all common camera paradigms. Each is distinct enough to warrant separation. Shared base enables polymorphic handling.

### 6.2 Why Sphere Cast with 9 Rays?
Single raycast misses thin occluders. 8 offset rays around center ensure coverage of camera frustum corners.

### 6.3 Why Arc-Length Parameterization?
Raw spline t-parameter gives non-uniform speed. Arc-length reparameterization ensures constant velocity along rail.

### 6.4 Why Elastic/Bounce Easing?
Adds "juice" to camera transitions. Elastic overshoot conveys energy. Bounce conveys impact.
