# Investigation: engine/animation/procedural

## Summary
The procedural animation subsystem is a comprehensive, production-quality implementation spanning 4,872 lines across 9 Python modules. It provides real physics-based spring bones with Verlet integration, complete ragdoll physics with joint limits and motors, procedural locomotion for biped/quadruped gaits, realistic look-at with saccades, breathing animation, twist bone distribution, and composable secondary motion effects. All implementations use proper physics formulas with numerical stability considerations.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 129 | REAL | Comprehensive exports, well-documented |
| `spring_bone.py` | 653 | REAL | Verlet integration, collision detection, wind forces |
| `ragdoll.py` | 809 | REAL | Full/partial ragdoll, active ragdoll motors, physics world protocol |
| `secondary_motion.py` | 720 | REAL | 5 motion types + composer, Perlin noise impl |
| `locomotion.py` | 676 | REAL | Walk/run/trot/gallop gaits, biped+quadruped support |
| `lookat.py` | 647 | REAL | Head/neck/eye tracking, saccades, interest points |
| `breathing.py` | 477 | REAL | 5 exertion levels, multi-bone effects |
| `twist.py` | 497 | REAL | Quaternion twist extraction, distribution modes |
| `config.py` | 273 | REAL | Centralized constants, physics presets |

## Procedural Components
- **SpringBone/SpringChain**: Verlet integration spring physics for hair/cloth/accessories
- **Ragdoll**: Full/partial/kinematic/active modes with physics world protocol
- **LookAtController**: Head/neck/eye rotation with angle limits and saccades
- **ProceduralLocomotion**: Walk/run cycle generation with gait configs
- **BreathingController**: Multi-bone breathing with exertion levels
- **TwistBone**: Twist distribution for joint deformation helpers
- **SecondaryMotion**: Delayed/oscillating/noise/impulse motion effects
- **Collision**: Sphere and capsule collision primitives for spring bones

## Implementation
- Real ragdoll? **YES** - Complete implementation with RagdollBody, RagdollJoint, JointLimits, JointMotor, collision groups. Supports dynamic/kinematic/blending states. Physics world protocol allows external physics engine integration.
- Real physics blending? **YES** - `blend_weight` with slerp/lerp interpolation between animation and physics poses. Smooth transitions via `blend_duration`. Partial ragdoll support with `active_bodies` set.
- Real secondary motion? **YES** - 5 distinct motion types: DelayedMotion (time-buffered), OscillatingMotion (sine), NoiseMotion (Perlin FBM), ImpulseResponse (damped spring), MotionComposer (stacking).

## Verdict
**REAL IMPLEMENTATION** - Production-quality procedural animation system with correct physics formulas, numerical stability handling, and comprehensive feature coverage.

## Evidence

### Spring Physics (Verlet Integration)
```python
# From spring_bone.py:364-369
# Verlet integration: x_new = 2*x - x_old + a*dt^2
dt_sq = dt * dt
new_position = vec3_add(
    vec3_sub(vec3_scale(self._position, 2.0), self._previous_position),
    vec3_scale(acceleration, dt_sq)
)
```

### Ragdoll Joint Limits
```python
# From ragdoll.py:181-199
@dataclass
class JointLimits:
    twist_lower: float = math.radians(-45.0)
    twist_upper: float = math.radians(45.0)
    swing1_limit: float = math.radians(45.0)
    swing2_limit: float = math.radians(45.0)
    contact_distance: float = math.radians(5.0)
```

### Physics Blending
```python
# From ragdoll.py:649-658
if self.state == RagdollState.BLENDING:
    anim_pos = self._animation_pose.get_bone_position(body.bone_index)
    anim_rot = self._animation_pose.get_bone_rotation(body.bone_index)
    blended_pos = self._lerp_vec3(anim_pos, transform.position, self.blend_weight)
    blended_rot = self._slerp_quat(anim_rot, transform.rotation, self.blend_weight)
```

### Saccade Generator
```python
# From lookat.py:277-290
@dataclass
class SaccadeGenerator:
    min_interval: float = 0.1  # Minimum time between saccades
    max_interval: float = 3.0  # Maximum time between saccades
    max_offset: float = 0.05  # Maximum saccade offset in radians (~3 degrees)
    speed: float = 500.0  # Saccade speed in degrees/second
```

### Perlin Noise for Secondary Motion
```python
# From secondary_motion.py:140-183
class PerlinNoise:
    def _fade(self, t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    def fbm(self, x: float, octaves: int = 4, persistence: float = 0.5) -> float:
        # Fractal Brownian Motion implementation
```

### Twist Extraction
```python
# From twist.py:186-206
def extract_twist_rotation(rotation: Quaternion, twist_axis: Vec3) -> Quaternion:
    axis, angle = quat_to_axis_angle(rotation)
    dot = axis[0] * twist_axis[0] + axis[1] * twist_axis[1] + axis[2] * twist_axis[2]
    twist_angle = angle * dot
    return quat_from_axis_angle(twist_axis, twist_angle)
```
